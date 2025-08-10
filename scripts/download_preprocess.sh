#!/bin/bash

# Download Ref-NeRF data and SAM2 weights
cd data && wget https://storage.googleapis.com/gresearch/refraw360/ref_real.zip && unzip ref_real.zip && cd ..
wget -P weights https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cp -r SAM2/sam2/configs .
pip install -e SAM2

# Implement fixes to SAM and GS repos
sed -i '248c\    frame_names.sort(key=lambda p: os.path.splitext(p)[0])' SAM2/sam2/utils/misc.py
sed -i '21c\    image = Image.open(cam_info.image_path.replace("JPG", "jpg"))' gaussian-splatting/utils/camera_utils.py
sed -i '54c\        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)' gaussian-splatting/train.py
