# <p align="center"> 💡<font color=#F0F000>LightSwitch</font>💡: Multi-view Relighting with Material-guided Diffusion </p>

#####  <p align="center"> [Yehonathan Litman](https://yehonathanlitman.github.io/), [Fernando De la Torre](https://www.cs.cmu.edu/~ftorre/), [Shubham Tulsiani](https://shubhtuls.github.io/)</p>
##### <p align="center"> ICCV 2025

#### <p align="center">[📑 Paper](http://arxiv.org/abs/2508.06494) | [🖥️ Webpage](https://yehonathanlitman.github.io/light_switch/) <br><br>
    
![Teaser image](assets/lightswitch_v2.svg)

# Installation

#### Tested on one A100

```
git clone https://github.com/yehonathanlitman/LightSwitch --recursive && cd LightSwitch

conda create -n lightswitch python=3.10 -y
conda activate lightswitch

pip install torch==2.7.1 torchvision
pip install -r requirements.txt
imageio_download_bin freeimage
```

# Running
In addition to images, our model needs masks, extrinsics, and intrinsic information in COLMAP format. If you already have these you can skip steps 1-2. Your data should look like this
```
data/
└── light_probes/ #Target lightings
    ├── your_envmap.hdr
    └── your_envmap2.exr
└── your_data/
    ├── images/
    ├── masks/
    └── sparse/ #COLMAP information
```
### 0. Set Environment Variables
Set your variables to the name of your data and image directories. For our example, we will set:
```
export OBJ=sedan
export IMAGES=images_4
export ENVMAP=aerodynamics_workshop
```
### 1. Compute Masks

Alternatively, use `scripts/generate_masks.py` to generate masks with SAM2. We'll use data from [Ref-NeRF](https://dorverbin.github.io/refnerf) as an example:
```
bash scripts/download_preprocess.sh
python scripts/generate_masks.py \
      --image_dir data/${OBJ}/${IMAGES} \
      --initial_prompt 400 400 800 400 1000 300
```
For your image data, change the initial pixels with `--initial_prompt` and SAM2 will propagate the mask throughout all the images.

### 2. Compute Poses
`sedan` already has poses so this step can be skipped, but for your own data use [VGGT](https://github.com/facebookresearch/vggt) or [COLMAP](https://github.com/colmap/colmap) to compute poses. For VGGT:
```
pip install git+https://github.com/facebookresearch/vggt
python scripts/vggt_colmap.py \
      --scene_dir data/${OBJ} \
      --image_dir_name ${IMAGES}
```
- [ ] Refactor: Poses recovered for `sedan` break relighting due to mask
### 3. Multi-view Relighting
`produce_gs_relightings.py` will produce the relit views that will then be used for gaussian splatting. The script will denoise and shuffle the input latents by automatically distributing across your GPUs. More GPUs + VRAM = faster relighting! 
```
accelerate launch produce_gs_relightings.py \
      --scene_dir data/${OBJ} \
      --image_dir_name ${IMAGES} \
      --envmap_path data/light_probes/${ENVMAP}.hdr
      --downsample 2
```

The inferred material images will be stored in `relighting_outputs/rm_{guidance_scale}_{sm_guidance_scale}`. The relit images will be under the corresponding environment lighting map (e.g. `aerodynamics_workshop`).

<details>
  <summary>(Optional) High Resolution Multi-FOV Model</summary>

  We also offer a model trained on 768x768 images with multiple FOVs that can be enabled with `--pretrained_model thebluser/lightswitch-multi-fov`. This model is more suitable for high resolution images.
</details> 

### 4. Relighting 3DGS Assets 
First, optimize a gaussian splat on your input data and verify it looks good:
```
python gaussian-splatting/train.py -s data/${OBJ} \
      -m gs_outputs/${OBJ} \
      --images ${IMAGES} \
      --resolution 2 \
      --checkpoint_iterations 30000
python gaussian-splatting/render.py -m gs_outputs/${OBJ}
```
Now, fix the splat positions and continue optimizing only the appearance using the relit images:
```
python gaussian-splatting/train.py -s relighting_outputs/rm_3_3/${OBJ}/${ENVMAP} \
      --start_checkpoint gs_outputs/${OBJ}/chkpnt30000.pth \
      --iterations 40000 \
      -m gs_outputs/relit_gs/${OBJ}/${ENVMAP} \
      --images ${IMAGES} \
      --position_lr_init 0.0 \
      --position_lr_final 0.0 \
      --opacity_lr 0.0 \
      --scaling_lr 0.0 \
      --rotation_lr 0.0
python gaussian-splatting/render.py -m gs_outputs/relit_gs/${OBJ}/${ENVMAP}
```
The relit 3DGS output will be in `gs_outputs/relit_gs/sedan/aerodynamics_workshop`.
# Citation

If you use any parts of our work, please cite the following:

```
@inproceedings{litman2025lightswitch,
  author    = {Yehonathan Litman and Fernando De la Torre and Shubham Tulsiani},
  title     = {LightSwitch: Multi-view Relighting with Material-guided Diffusion},
  booktitle = {ICCV},
  year      = {2025}
}
```
