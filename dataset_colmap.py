import os
import glob
import imageio.v3 as imageio

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from colmap_loader import *

def generate_directional_embeddings(shape=(32, 64)):
    height, width = shape
    u = np.linspace(0, 1, width, endpoint=False)
    v = np.linspace(0, 1, height, endpoint=False)
    U, V = np.meshgrid(u, v)

    theta = np.pi * V
    phi = 2 * np.pi * U

    # Blender frame
    x = np.sin(theta) * np.cos(phi)
    y = -np.sin(theta) * np.sin(phi)
    z = -np.cos(theta)

    embeddings = np.stack((x, y, z), axis=-1)

    return embeddings

def generate_plucker_rays(T, shape, fov, sensor_size=(1.0, 1.0)):
    R, t = T[:3, :3], T[:3, 3]
    H, W = shape
    H //= 8
    W //= 8
    fov_x, fov_y = fov

    i = np.linspace(-sensor_size[1], sensor_size[1], H) 
    j = np.linspace(-sensor_size[0], sensor_size[0], W) 

    x_ndc, y_ndc = np.meshgrid(j, i)
    x_cam = x_ndc * np.tan(fov_x / 2.0)
    y_cam = y_ndc * np.tan(fov_y / 2.0)
    z_cam = np.ones_like(x_cam) # Z=1 plane

    rays_d_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rays_d_cam /= np.linalg.norm(rays_d_cam, axis=-1, keepdims=True)
    rays_d_world = -np.matmul(R, rays_d_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
    rays_m_world = np.cross(t, rays_d_world, axisa=0, axisb=2)

    plucker_rays = np.stack([
        rays_d_world[..., 0], rays_d_world[..., 1], rays_d_world[..., 2],
        rays_m_world[..., 0], rays_m_world[..., 1], rays_m_world[..., 2]
    ], axis=0)
    return plucker_rays

def hlg_oetf(x):
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return np.where(x <= 1/12, np.sqrt(3 * x), a * np.log((12*x - b).clip(1e-5, np.inf)) + c)

def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def _load_img(path):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = imageio.imread(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _load_mask(fn):
    img = _load_img(fn)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img

def pad_to_multiple(tensor, multiple=8, mode='constant', value=0):
    """
    Pads a (C, H, W) tensor so that H and W are divisible by `multiple`.
    """
    _, h, w = tensor.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode=mode, value=value)
    return padded_tensor

from typing import NamedTuple
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    mask_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

def focal2fov(focal, pixels):
    return 2*np.arctan(pixels/(2*focal))

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, masks_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        mask_path = os.path.join(masks_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, mask_path=mask_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    return cam_infos

class DatasetCOLMAP(Dataset):
    def __init__(self, cfg_path, args, transform, env_transform=None, envmap_path=None):
        self.args = args
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        try:
            cameras_extrinsic_file = os.path.join(cfg_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(cfg_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(cfg_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(cfg_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = args.image_dir_name
        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=None,
            images_folder=os.path.join(cfg_path, reading_dir), 
            depths_folder="", masks_folder=os.path.join(cfg_path, "masks"), test_cam_names_list=[])
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        self.train_cam_infos = [c for c in cam_infos if not c.is_test]
        self.all_img = [os.path.split(c.image_path)[1] for c in self.train_cam_infos]

        self.n_images = len(self.train_cam_infos)
        self.downsample = args.downsample
        self.envmap = imageio.imread(envmap_path)[..., :3]

        if env_transform is None:
            self.env_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 512), antialias=True),
                transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.env_transform = env_transform

        self.env_darker = (np.log10(self.envmap + 1) / np.log10(self.envmap.max())).clip(0, 1)
        self.env_brighter = hlg_oetf(self.env_darker).clip(0, 1)
        self.env_darker = self.env_transform(self.env_darker)
        self.env_brighter = self.env_transform(self.env_brighter)

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.splitext(self.train_cam_infos[0].image_path)[0]).shape[:2]
        # self.resolution = [self.train_cam_infos[0].height, self.train_cam_infos[0].width]
        self.aspect = self.resolution[1] / self.resolution[0]
        self.h = self.resolution[0]//self.downsample
        self.w = self.resolution[1]//self.downsample

        self.transform = transform
        self.resize = transforms.Resize((self.h, self.w), antialias=True)
        self.dir_embeds = torch.tensor(generate_directional_embeddings(), dtype=torch.float32).permute(2, 0, 1)

        print(f"DatasetCOLMAP: {self.n_images} images with shape [{self.resolution[0]}, {self.resolution[1]}], downsample {self.downsample}")

    def _parse_frame(self, cam_near_far=[0.1, 1000.0]):
        imgs, masks, mvps, envs_darker, envs_brighter, dir_embeds, pluckers = [], [], [], [], [], [], []

        for i in np.arange(self.n_images):
            img = _load_img(os.path.splitext(self.train_cam_infos[i].image_path)[0])
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = self.train_cam_infos[i].R#.transpose()
            Rt[:3, 3] = self.train_cam_infos[i].T
            Rt[:3, 1:3] *= -1
            Rt[3, 3] = 1.0

            frame_transform = torch.tensor(Rt, dtype=torch.float32)
            mv = torch.linalg.inv(frame_transform)

            mask = _load_mask(os.path.splitext(self.train_cam_infos[i].mask_path)[0])
            img = self.resize(img.permute(2, 0, 1))
            mask = self.resize(mask.permute(2, 0, 1))
            img = pad_to_multiple(img, multiple=8)
            mask = pad_to_multiple(mask, multiple=8)
            img = img * mask
            img = 2*(img - 0.5)

            frame_pluckers = torch.tensor(generate_plucker_rays(frame_transform, img.shape[1:3], [self.train_cam_infos[i].FovX, self.train_cam_infos[i].FovY]))

            imgs.append(img)
            masks.append(mask)
            dir_embeds.append(self.dir_embeds)
            pluckers.append(frame_pluckers)

            env_brighter = self.env_brighter
            env_darker = self.env_darker
            envs_darker.append(env_darker)
            envs_brighter.append(env_brighter)

            proj = perspective(self.train_cam_infos[i].FovY, self.aspect, cam_near_far[0], cam_near_far[1])
            mv = mv @ rotate_x(-np.pi / 2)
            mvp = proj @ mv
            t = mvp[:3, 3]
            r = torch.linalg.norm(t)
            theta = torch.arccos(t[2] / r)
            phi = torch.arctan2(t[1], t[0])
            mvp = torch.tensor([theta, torch.sin(phi), torch.cos(phi), r])
            mvps.append(mvp)

        return torch.stack(imgs), torch.stack(masks), torch.stack(mvps), torch.stack(envs_darker), torch.stack(envs_brighter), torch.stack(dir_embeds), torch.stack(pluckers)

    def __len__(self):
        return 1

    def __getitem__(self, itr):
        img, mask, mvp, envs_darker, envs_brighter, dir_embeds, pluckers = self._parse_frame()

        return {
            'T' : mvp,
            'img' : img,
            'mask' : mask,
            'dir_embeds' : dir_embeds,
            'pluckers' : pluckers,
            'envs_darker' : envs_darker,
            'envs_brighter' : envs_brighter,
        }
    
    def collate(self, batch):
        out_batch = {
            'image': torch.cat([b['img'] for b in batch]),
            'mask': torch.cat([b['mask'] for b in batch]),
            'T': torch.cat([b['T'] for b in batch]),
            'envs_darker': torch.cat([b['envs_darker'] for b in batch]),
            'envs_brighter': torch.cat([b['envs_brighter'] for b in batch]),
            'dir_embeds': torch.cat([b['dir_embeds'] for b in batch]),
            'pluckers': torch.cat([b['pluckers'] for b in batch]),
        }
        return out_batch