# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description

<div>

This is the code implementation for the paper titled: "DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description" (Accepted to CVPR Workshop AI4CC)[arXiv](https://arxiv.org/abs/2503.24096).

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
