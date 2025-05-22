# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description

<div>

<p align="center" width="100%">
<a target="_blank"><img src="figures/DANTE-AD-model-overview.jpg" alt="DANTE-AD" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Setup

First, clone the repo and install required packages:
```bash
git clone https://github.com/AdrienneDeganutti/DANTE-AD.git
cd DanteAD/

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

Our implementation is mainly based on the following codebase:
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA.git): An Instruction-tuned Audio-Visual Language Model for Video Understanding

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
