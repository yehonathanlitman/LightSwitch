import os
import argparse
import numpy as np
from PIL import Image
import torch
import imageio.v3 as iio

# You must install SAM 2 and its dependencies.
# The `sam2` library has a different structure and requires a specific installation process.
# This import will fail if the library is not correctly installed.
from sam2.build_sam import build_sam2_video_predictor

def create_mask_directory(image_dir):
    """
    Creates a 'masks' directory in the parent directory of the image directory.
    """
    parent_dir = os.path.dirname(os.path.abspath(image_dir))
    mask_dir = os.path.join(parent_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    return mask_dir

def load_sam2_predictor(sam_checkpoint, model_type, device):
    """
    Loads and returns the SAM 2 video predictor.
    """
    # SAM 2 uses a config file along with the checkpoint.
    # The model_type argument from the original script is used to infer the config.
    config_map = {
        "vit_l": "sam2.1_hiera_l.yaml",
        "vit_b": "sam2.1_hiera_b+.yaml",
    }
    config_path = os.path.join("configs/sam2.1", config_map[model_type])

    if not os.path.exists(config_path):
        print(f"Error: SAM 2 config file not found at {config_path}.")
        print("Please ensure the corresponding YAML config file is in the same directory as the checkpoint.")
        exit(1)
        
    predictor = build_sam2_video_predictor(config_path, sam_checkpoint, device)
    return predictor

def main():
    parser = argparse.ArgumentParser(description="Propagate a mask across a video sequence using SAM 2's video propagation capabilities.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the directory containing the video frames.")
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam2.1_hiera_large.pt",
                        help="Path to the SAM 2 model checkpoint (.pt file).")
    parser.add_argument("--model_type", type=str, default="vit_l",
                        choices=["vit_l", "vit_b"],
                        help="The type of SAM 2 model to use (vit_h, vit_l, or vit_b).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="The device to run the model on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--initial_prompt", type=int, nargs='+',
                        help="Optional point prompts for the first image to define the initial object. Provide as a space-separated list of coordinates (e.g., X1 Y1 X2 Y2). If not provided, the image center is used.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' not found.")
        return

    mask_dir = create_mask_directory(args.image_dir)
    print(f"Masks will be saved in: {mask_dir}")

    print(f"Loading SAM 2 model of type '{args.model_type}' on '{args.device}'...")
    try:
        predictor = load_sam2_predictor(args.sam_checkpoint, args.model_type, args.device)
    except Exception as e:
        print(f"An error occurred while loading the SAM 2 model: {e}")
        return
        
    image_names = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
    if not image_names:
        print("Error: Image directory is empty.")
        return
        
    # Load all images at once into a list for SAM 2
    video_frames = []
    for name in image_names:
        path = os.path.join(args.image_dir, name)
        try:
            image = iio.imread(path)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.shape[2] == 4:
                image = image[:, :, :3]
            video_frames.append(image)
        except Exception as e:
            print(f"Could not read image {path}: {e}")
            return
    
    # SAM 2 requires a single inference state for the entire video
    inference_state = predictor.init_state(args.image_dir)
    
    # Determine the initial prompt
    initial_frame_idx = 0
    first_image = video_frames[initial_frame_idx]
    
    if args.initial_prompt and len(args.initial_prompt) % 2 == 0:
        num_points = len(args.initial_prompt) // 2
        prompt_coords = np.array(args.initial_prompt).reshape(num_points, 2)
        prompt_labels = np.ones(num_points, dtype=np.int32)
        print(f"Using {num_points} user-provided points for frame 0: {prompt_coords.tolist()}")
    else:
        if args.initial_prompt:
            print("Warning: Invalid prompt. The number of coordinates for initial_prompt must be even. Falling back to image center.")
        h, w, _ = first_image.shape
        prompt_coords = np.array([[w // 2, h // 2]])
        prompt_labels = np.array([1], dtype=np.int32)
        print(f"No valid prompt provided. Using image center for frame 0: ({w // 2}, {h // 2})")

    # Add the initial prompt to the state for the first object (ID 1)
    obj_id = 1
    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=initial_frame_idx,
        obj_id=obj_id,
        points=prompt_coords,
        labels=prompt_labels,
    )
    
    # Use SAM 2's dedicated propagation method
    print("Starting video mask propagation...")
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[frame_idx] = {
            out_obj_id: (mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(object_ids)
        }

    for out_frame_idx in range(len(image_names)):
        composed_mask = None

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            print(f"frame {out_frame_idx}, object {out_obj_id}")
            mask_np = out_mask.squeeze(0).astype(np.uint8) * 255

            if composed_mask is None:
                composed_mask = mask_np
            else:
                # Combine masks, making sure values don’t exceed 255
                composed_mask = np.maximum(composed_mask, mask_np)
        mask_filename = image_names[out_frame_idx]
        mask_path = os.path.join(mask_dir, mask_filename)

        Image.fromarray(composed_mask).save(mask_path)

    print("Video mask propagation complete.")

if __name__ == "__main__":
    main()