# Copyright (c) 2025 Adrienne Deganutti
# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description
# Licensed under the MIT License

""" Main file for DANTE-AD """

import argparse

from src.configs.utils.load_args import load_dante_args

from src.modeling.video_llama.common.registry import registry

from torch.utils.data import DataLoader
from src.dataset.build_dataset import VideoDataset
from src.modeling.video_llama.common.config import Config
from src.tasks.train import training_loop
from src.tasks.eval import eval_loop

def load_config(config_path):
    config = load_dante_args(config_path)
    return config  

def get_dataset(cfg):
    train_dataset = VideoDataset(cfg.datasets_cfg['cmd_AD'], split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.args.batch_size, shuffle=False, num_workers=cfg.args.num_workers)

    val_dataset = VideoDataset(cfg.datasets_cfg['cmd_AD'], split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.args.batch_size, shuffle=False, num_workers=cfg.args.num_workers)

    return train_dataloader, val_dataloader


def main(config):
    cfg = Config(config)
    cfg.model_cfg.device_8bit = 0
    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg, cfg.args).to('cuda:{}'.format(cfg.args.gpu_id))
    
    train_dataset, val_dataset = get_dataset(cfg)

    if cfg.args.do_train:
        training_loop(
            model=model,
            args=cfg.args,
            data=train_dataset
        )
    
    if cfg.args.do_eval:
        eval_loop(
            model=model,
            args=cfg.args,
            data=val_dataset
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)