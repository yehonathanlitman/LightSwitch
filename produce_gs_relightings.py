import numpy as np
import torchvision.transforms as T
import torch
from tqdm.auto import tqdm
import argparse
import os, shutil
import atexit
import imageio.v3 as imageio

from diffusers.utils.import_utils import is_xformers_available
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
)
from accelerate.utils import gather_object
from accelerate import Accelerator

from pipeline_relighting_multi_vae import RelightingPipelineMVVAE
from stablematerial.pipeline_stablematerial_mv import StableMaterialPipelineMV
from dataset_colmap import DatasetCOLMAP
from models.unet_2d_condition import UNet2DConditionModel

def reverse_order(images, perm_history):
    for permuted_indices in reversed(perm_history):
        images = images[np.argsort(permuted_indices)]
    return images

def get_materials_parallel_denoise(accelerate, stable_material, save_path, input_image, pose, mask, args, num_inference_loops=1, num_inference_steps=50, weight_dtype=torch.float16, modifier=""):
    pred_albedo, pred_orm = [], []
    albedo_path, orm_path = os.path.join(save_path, "albedo" + modifier), os.path.join(save_path, "orm" + modifier)
    if os.path.exists(albedo_path):
        print("Loading albedo and orm from file...")
        for f_albedo in sorted(os.listdir(albedo_path)):
            pred_albedo.append(imageio.imread(os.path.join(albedo_path, f_albedo))/255.)
        for f_orm in sorted(os.listdir(orm_path)):
            pred_orm.append(imageio.imread(os.path.join(orm_path, f_orm))/255.)
        pred_albedo = np.stack(pred_albedo)
        pred_orm = np.stack(pred_orm)
    else:
        h, w = input_image.shape[2:]

        # 1. Check input image
        stable_material.check_inputs(input_image, h, w, 1)

        # 2. Define call parameters
        if isinstance(input_image, list):
            batch_size = len(input_image)
        else:
            batch_size = input_image.shape[0]

        # 3. Encode input image as prompt
        with torch.no_grad():
            prompt_embeds = stable_material._encode_image_with_pose(input_image, pose, accelerate.device, 1, args.sm_guidance_scale > 1.0)

        # 4. Prepare timesteps
        stable_material.scheduler.set_timesteps(num_inference_steps, device=accelerate.device)
        timesteps = stable_material.scheduler.timesteps

        # 5. Prepare latent variables
        with torch.no_grad():
            latents = stable_material.prepare_latents(
                batch_size,
                8,
                h,
                w,
                prompt_embeds.dtype,
                accelerate.device,
                None
            )

            # 6. Prepare image latents
            img_latents = stable_material.prepare_img_latents(
                input_image,
                batch_size,
                prompt_embeds.dtype,
                accelerate.device,
                None,
                args.sm_guidance_scale > 1.0,
            )
            if args.sm_guidance_scale > 1.0:
                img_latents_zero, img_latents = img_latents.chunk(2)
                prompt_embeds_zero, prompt_embeds = prompt_embeds.chunk(2)
            for _ in range(num_inference_loops):
                for i, t in enumerate(tqdm(timesteps)):
                    latents_B = latents.shape[0]
                    perm_indices = torch.randperm(latents_B)
                    latents = latents[perm_indices]
                    prompt_embeds = prompt_embeds[perm_indices]
                    img_latents = img_latents[perm_indices]
                    latents_chunks = torch.chunk(latents, accelerate.state.num_processes, dim=0)
                    prompt_embeds_chunks = torch.chunk(prompt_embeds, accelerate.state.num_processes, dim=0)
                    img_latents_chunks = torch.chunk(img_latents, accelerate.state.num_processes, dim=0)
                    prompts_all = {"latents": latents_chunks, "img_latents": img_latents_chunks, "prompt_embeds": prompt_embeds_chunks}
                    with accelerate.split_between_processes(prompts_all) as prompts:
                        latents_combined = []
                        for latents_t, img_latents_t, prompt_embeds_t in zip(prompts["latents"], prompts["img_latents"], prompts["prompt_embeds"]):
                            img_latents_t = torch.cat([torch.zeros_like(img_latents_t), img_latents_t])
                            prompt_embeds_t = torch.cat([torch.zeros_like(prompt_embeds_t), prompt_embeds_t])
                            with torch.no_grad():
                                pred_images = stable_material.call_1_denoise_permute(latents=latents_t, img_latents=img_latents_t, prompt_embeds=prompt_embeds_t, guidance_scale=args.sm_guidance_scale, t=t).images
                            latents_gpu = pred_images[0]
                            latents_combined.append(latents_gpu.cpu())
                    latents_combined = gather_object(latents_combined)
                    latents = torch.cat(latents_combined).to(accelerate.device)
                    latents = reverse_order(latents, [perm_indices])
                    prompt_embeds = reverse_order(prompt_embeds, [perm_indices])
                    img_latents = reverse_order(img_latents, [perm_indices])
            albedo, orm = stable_material.decode_latents(latents, permute=True)

            pred_albedo.append(albedo)
            pred_orm.append(orm)


        os.makedirs(albedo_path, exist_ok=True)
        os.makedirs(orm_path, exist_ok=True)
        pred_albedo = np.mean(np.stack(pred_albedo), axis=0)
        pred_orm = np.mean(np.stack(pred_orm), axis=0)
        if accelerate.is_main_process:
            for i in range(len(pred_albedo)):
                pred_albedo_temp_i = np.clip(np.rint(pred_albedo[i] * mask[i].permute(1,2,0).float().numpy() * 255.0), 0, 255).astype(np.uint8)
                pred_orm_temp_i = np.clip(np.rint(pred_orm[i] * mask[i].permute(1,2,0).float().numpy() * 255.0), 0, 255).astype(np.uint8)
                imageio.imwrite(os.path.join(albedo_path, f"{i:05d}.png"), pred_albedo_temp_i)
                imageio.imwrite(os.path.join(orm_path, f"{i:05d}.png"), pred_orm_temp_i)
        accelerate.wait_for_everyone()

    pred_albedo = torch.from_numpy(pred_albedo).permute(0, 3, 1, 2).to(accelerate.device)
    pred_orm = torch.from_numpy(pred_orm).permute(0, 3, 1, 2).to(accelerate.device)

    pred_albedo = pred_albedo.to(dtype=weight_dtype)
    pred_orm = pred_orm.to(dtype=weight_dtype)
    pred_albedo = pred_albedo.to(dtype=weight_dtype) * mask.to(accelerate.device)
    pred_orm = pred_orm.to(dtype=weight_dtype) * mask.to(accelerate.device)

    pred_albedo = T.Normalize([0.5], [0.5])(pred_albedo)
    pred_orm = T.Normalize([0.5], [0.5])(pred_orm)

    return pred_albedo, pred_orm

def get_relightings_parallel_denoise(accelerate, pipeline, input_image, pred_albedo, pred_orm, envs_darker_target, envs_brighter_target, dir_embeds, pluckers, args, generator=None, num_inference_steps=50, num_inference_loops=1):
    temp_relit_images_ours = []
    h, w = input_image.shape[2:]

    pipeline.check_inputs(input_image, h, w, 1)

    # 2. Define call parameters
    if isinstance(input_image, list):
        batch_size = len(input_image)
    else:
        batch_size = input_image.shape[0]

    # 3. Encode input image as prompt
    do_classifier_free_guidance = args.guidance_scale > 1.0
    scene_features = pipeline.encode_env(envs_darker_target, envs_brighter_target, dir_embeds, do_classifier_free_guidance, accelerate.device, input_image.dtype)

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=accelerate.device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    latents = pipeline.prepare_latents(
        batch_size,
        4,
        h,
        w,
        scene_features.dtype,
        accelerate.device,
        None
    )

    # 6. Prepare image latents
    with torch.no_grad():
        condition_latents = pipeline.prepare_condition_latents(
            input_image,
            pred_albedo,
            pred_orm,
            pluckers,
            batch_size,
            scene_features.dtype,
            accelerate.device,
            None,
            do_classifier_free_guidance,
        )

        if args.guidance_scale > 1.0:
            scene_features_zero, scene_features = scene_features.chunk(2)
        with torch.no_grad():
            for _ in range(num_inference_loops):
                for i, t in enumerate(tqdm(timesteps)):
                    latents_B = latents.shape[0]
                    perm_indices = torch.randperm(latents_B)
                    latents = latents[perm_indices]
                    scene_features = scene_features[perm_indices]
                    condition_latents = condition_latents[perm_indices]
                    latents_chunks = torch.chunk(latents, accelerate.state.num_processes, dim=0)
                    scene_features_chunks = torch.chunk(scene_features, accelerate.state.num_processes, dim=0)
                    condition_latents_chunks = torch.chunk(condition_latents, accelerate.state.num_processes, dim=0)
                    prompts_all = {"latents": latents_chunks, "condition_latents": condition_latents_chunks, "scene_features": scene_features_chunks}
                    with accelerate.split_between_processes(prompts_all) as prompts:
                        latents_combined = []
                        for latents_t, condition_latents_t, scene_features_t in zip(prompts["latents"], prompts["condition_latents"], prompts["scene_features"]):
                            scene_features_t = torch.cat([torch.zeros_like(scene_features_t), scene_features_t])
                            pred_images = pipeline.call_1_denoise_permute(latents=latents_t, condition_latents=condition_latents_t, scene_features=scene_features_t, guidance_scale=args.guidance_scale, t=t).images
                            latents_gpu = pred_images[0]
                            latents_combined.append(latents_gpu.cpu())
                    latents_combined = gather_object(latents_combined)
                    latents = torch.cat(latents_combined).to(accelerate.device)
                    latents = reverse_order(latents, [perm_indices])
                    scene_features = reverse_order(scene_features, [perm_indices])
                    condition_latents = reverse_order(condition_latents, [perm_indices])
                relit_images = pipeline.decode_latents(latents, permute=True)

                temp_relit_images_ours.append(relit_images)

    return temp_relit_images_ours

def produce_colmap(accelerate, args, data_path, pipeline, stable_material, weight_dtype=torch.float16, generator=None):
    image_transforms = T.Compose(
        [
            T.Normalize([0.5], [0.5])
        ]
    )
    env_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((256, 512), antialias=True),
            T.Normalize([0.5], [0.5]),
        ]
    )

    parent_dir, dir_name = os.path.split(data_path)
    target_envmap_name = os.path.split(args.envmap_path)[-1].split(".")[0]
    results_dir = os.path.join("relighting_outputs", f"rm_{args.guidance_scale}_{args.sm_guidance_scale}", dir_name, target_envmap_name)
    os.makedirs(results_dir, exist_ok=True)

    out_dir = os.path.join(results_dir, args.image_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    train_dataset = DatasetCOLMAP(os.path.join(data_path), args, image_transforms, env_transform=env_transform, envmap_path=args.envmap_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate,
        shuffle=False,
        batch_size=1,
        num_workers=0,
    )
    shutil.copytree(os.path.join(data_path, "masks"), os.path.join(results_dir, "masks"), dirs_exist_ok = True)
    shutil.copytree(os.path.join(data_path, "sparse"), os.path.join(results_dir, "sparse"), dirs_exist_ok = True)

    batch = next(iter(train_dataloader))

    mask = batch["mask"]#.to(dtype=weight_dtype)
    input_image = batch["image"].to(dtype=weight_dtype)# * mask
    dir_embeds = batch["dir_embeds"].to(dtype=weight_dtype)
    pluckers = batch["pluckers"].to(dtype=weight_dtype)
    pose = batch["T"].to(dtype=weight_dtype)

    envs_darker_target = batch["envs_darker"].to(dtype=weight_dtype).to(pipeline.device)
    envs_brighter_target = batch["envs_brighter"].to(dtype=weight_dtype).to(pipeline.device)
    B = input_image.shape[0]
    remainder = B % 16
    if remainder != 0:
        pad_size = 16 - remainder

        input_image = torch.cat((input_image, input_image[:pad_size]), dim=0)
        mask = torch.cat((mask, mask[:pad_size]), dim=0)
        dir_embeds = torch.cat((dir_embeds, dir_embeds[:pad_size]), dim=0)
        pluckers = torch.cat((pluckers, pluckers[:pad_size]), dim=0)
        pose = torch.cat((pose, pose[:pad_size]), dim=0)
        envs_darker_target = torch.cat((envs_darker_target, envs_darker_target[:pad_size]), dim=0)
        envs_brighter_target = torch.cat((envs_brighter_target, envs_brighter_target[:pad_size]), dim=0)

    # h, w = input_image.shape[2:]
    temp_relit_images_ours = []

    pred_albedo, pred_orm = get_materials_parallel_denoise(accelerate, stable_material, os.path.split(results_dir)[0], input_image, pose, mask, args, num_inference_steps=35, num_inference_loops=1)
    temp_relit_images_ours = get_relightings_parallel_denoise(accelerate, pipeline, input_image, pred_albedo, pred_orm, envs_darker_target, envs_brighter_target, dir_embeds, pluckers, args, generator=generator, num_inference_steps=35, num_inference_loops=1)

    relit_image_mean = np.mean(np.stack(temp_relit_images_ours), axis=0)

    relit_image_mean = torch.from_numpy(relit_image_mean).permute(0, 3, 1, 2)
    relit_images_rescaled = relit_image_mean

    relit_images_mean_save = relit_images_rescaled * mask
    relit_images_mean_save = relit_images_mean_save.permute(0, 2, 3, 1).cpu().numpy()
    mask = mask.permute(0, 2, 3, 1).cpu().numpy()[..., :1]
    relit_images_mean_save = np.concatenate((relit_images_mean_save, mask), axis=-1)
    relit_images_mean_save = np.clip(np.rint(relit_images_mean_save * 255.0), 0, 255).astype(np.uint8)
    if accelerate.is_main_process:
        print(f"Saving {train_dataset.n_images} images...")
        for it, img_idx in enumerate(np.arange(train_dataset.n_images)):
            imageio.imwrite(os.path.join(out_dir, f"{os.path.split(train_dataset.all_img[img_idx])[1]}".replace(".JPG", ".jpg")), relit_images_mean_save[it][..., :3])
    accelerate.wait_for_everyone()

    torch.cuda.empty_cache()

def produce_gs_relightings(args, weight_dtype=torch.float16):
    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", prediction_type="v_prediction")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        use_safetensors=True,
    )
    pipeline = RelightingPipelineMVVAE(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None
    ).to(dtype=weight_dtype)    
    stable_material = StableMaterialPipelineMV.from_pretrained(args.pretrained_model_sm, torch_dtype=weight_dtype, trust_remote_code=True)
    if args.torch_compile:
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
        stable_material.unet = torch.compile(stable_material.unet, mode="reduce-overhead", fullgraph=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            pipeline.enable_xformers_memory_efficient_attention()
            stable_material.enable_xformers_memory_efficient_attention()
            pipeline.vae.enable_tiling()
            stable_material.vae.enable_tiling()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    accelerate = Accelerator()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerate.device).manual_seed(args.seed)

    stable_material.to(accelerate.device)
    pipeline.to(accelerate.device)
    produce_colmap(accelerate, args, args.scene_dir, pipeline, stable_material, weight_dtype=weight_dtype, generator=generator)
    def cleanup_distributed():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    if accelerate.num_processes > 1:
        atexit.register(cleanup_distributed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", 
                        type=str, 
                        default="thebluser/lightswitch", 
                        nargs='?', 
                        choices=["thebluser/lightswitch", "thebluser/lightswitch-multi-fov"],)
    parser.add_argument("--pretrained_model_sm", type=str, default="thebluser/stable-material-mv")
    parser.add_argument("--scene_dir", type=str, default="data/sedan")
    parser.add_argument("--image_dir_name", type=str, default="images_4")
    parser.add_argument("--envmap_path", type=str, default="data/light_probes/aerodynamics_workshop.hdr")
    parser.add_argument("--guidance_scale", type=float, default=3)
    parser.add_argument("--sm_guidance_scale", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=False, action="store_true")
    parser.add_argument("--torch_compile", default=False, action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    produce_gs_relightings(args, dtype)