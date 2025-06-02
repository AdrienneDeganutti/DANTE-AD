# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description

<div>

This is the code implementation for the paper titled: "DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description" (Accepted to CVPR Workshop AI4CC 2025) [[arXiv]](https://arxiv.org/abs/2503.24096).

<div>

## Introduction

This work proposes an enhanced video description model that improves context-awareness across scenes. DANTE-AD sequentially fuses both frame- and scene-level embeddings within a dual-vision Transformer-based architecture to improve contextual understanding.

<p align="center" width="100%">
<a target="_blank"><img src="figures/DANTE-AD-model-overview.jpg" alt="DANTE-AD" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Setup

First, clone the repo and install required packages:
```bash
git clone https://github.com/AdrienneDeganutti/DANTE-AD.git
cd Dante-AD/

conda env create -f environment.yml
conda activate dante_env

cd eval_metrics
git clone https://github.com/sks3i/pycocoevalcap.git
git clone https://github.com/Tiiiger/bert_score.git

```

<div>

## Training

<div>

## Evaluation

<div>

## Inference

<div>

## Dataset

The dataset used in this paper is a reduced version of the [CMD-AD dataset](https://www.robots.ox.ac.uk/~vgg/research/autoad/#datasets). Due to various encoding issues with the raw videos, our version of the CMD-AD dataset used in this paper is reduced from approximately 101k down to 96k AD segments as shown in the table below.

|                    | CMD-AD        | DANTE-AD      |
| -------------------| ------------- | ------------- |
| Total AD segments  | 101,268       | 96,873        |
| Train AD segments  | 93,952        | 89,798        |
| Eval AD segments   | 7,316         | 7,075         |

To enhance computational efficency, we pre-compute the frame-level (CLIP) and scene-level (S4V) visual embeddings offline. We provide these pre-processed visual embeddings and ground-truth annotations here: [Preprocessed CMD-AD](https://1drv.ms/f/c/fd682d23eb414404/EuJTjSzt5qBOsRpH2CaX7MQBeJzIlBov2HXDkZwzYMP9iQ?e=wNlbgw).

# Frame-Level Embeddings

For the frame-level CLIP features, we process the following modules offline: EVA-CLIP feature extraction, Q-Former, positional embedding and Video Q-Former. Therefore, the features provided to replicate our work are the output of the Video Q-Former with shape ([1, 32, 768]). For reproducibility, the code used for these steps is taken from [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA.git).

# Scene-Level Embeddings

The scene-level S4V features provided are processed from the action recognition module of [Side4Video](https://github.com/HJYao00/Side4Video) pre-trained on Kinetics-400. The S4V features we provide are the output of the Side4Video module after Global Average Pooling over each frame within the video sequence. The output features are of shape ([1, 1, 320]).

<div>

## Acknowledgment

Our implementation builds upon the following codebases:
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA.git): An Instruction-tuned Audio-Visual Language Model for Video Understanding
- [AutoAD](https://github.com/TengdaHan/AutoAD): Movie Description in Context
- [GRIT](https://github.com/davidnvq/grit): Faster and Better Image-Captioning Transformer

<div>

## Citation

If you find our project useful, please kindly cite the paper with the following bibtex:
```bibtex
@article{deganutti2025dante,
  title={DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description},
  author={Deganutti, Adrienne and Hadfield, Simon and Gilbert, Andrew},
  journal={arXiv preprint arXiv:2503.24096},
  year={2025}
}
```
