# Data Preparation

This directory contains the code for generating the S4V features on the CMD-AD dataset.

## Setup

After creating the conda environment as described in the main README, install the additional dependencies required for the data preparation scripts:

```bash
pip install dotmap ftfy torchnet
```

Download the model weights:
```bash
mkdir data_preparation/ckpt
cd data_preparation/ckpt
```
**Download: [vitb-16-f8](https://1drv.ms/u/c/fd682d23eb414404/IQCmi5jaupr8TLrHObCtSycYAdbdBLY1OHEPLCijgD2JIPI)**


## Usage

Update the ```val_list``` path in ```configs/k400_train_rgb_vitb-16-f8-side4video.yaml``` to point to the text file containing the video paths. An example is provided in ```lists/cmd_ad/video_paths.txt```.

To run the S4V data preparation script:

```bash
python data_preparation/run_s4v.py \
    --nproc_per_node 1 \
    --config configs/k400_train_rgb_vitb-16-f8-side4video.yaml \
    --weights ckpt/k400_vitb16_f8_82.5.pt \
    --output_dir data_preparation/output/
```

## Acknowledgement

The original code can be found here: [Side4Video](https://github.com/HJYao00/Side4Video): Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning 