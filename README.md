
## Installation with conda (recommended)

```
conda create -n colorsurge python=3.9
conda activate colorsurge
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

python3 setup.py develop  # install basicsr

pip3 install openmim
mim install mmcv>=2.0.0
mim install mmengine
```

## Quick Start
1. **Download the pretrained model files**

   Download the pretrained model files and place them into the `pretrain_models` directory.

   - Download [convnextv2_large_22k_384_ema.pt](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt). Place it as: `pretrain_models/convnextv2_large_22k_384_ema.pt`

   - Download [Tiny model](https://drive.google.com/file/d/11OonnMGKSEewILHlAYnx5ALGfsPvB_wB/view?usp=drive_link). Place it as: `pretrain_models/colorsurge_tiny.pth`
   
   - Download [Large model](https://drive.google.com/file/d/1EOeNgMrizWrwzfEp_jDAi5URULDLJLaQ/view?usp=drive_link). Place it as: `pretrain_models/colorsurge_large.pth`

2. **Run the pipeline**

   ```bash
   bash colorsurge_video.sh
