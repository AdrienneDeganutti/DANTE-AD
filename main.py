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

def get_dataset(cfg, vis_processor):
    train_dataloader = None
    val_dataloader = None

    if cfg.args.do_train:
        train_dataset = VideoDataset(cfg.args, cfg.datasets_cfg['cmd_AD'], vis_processor, split='train')
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.args.batch_size,
                                    shuffle=False, num_workers=cfg.args.num_workers)

    if cfg.args.do_eval:
        val_dataset = VideoDataset(cfg.args, cfg.datasets_cfg['cmd_AD'], vis_processor, split='val')
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.args.batch_size,
                                    shuffle=False, num_workers=cfg.args.num_workers)

    return train_dataloader, val_dataloader


def main(config):
    cfg = Config(config)
    cfg.model_cfg.device_8bit = 0
    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg, cfg.args).to('cuda:{}'.format(cfg.args.gpu_id))
    
    vis_processor_cfg = cfg.datasets_cfg.cmd_AD.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    train_dataset, val_dataset = get_dataset(cfg, vis_processor)

    if cfg.args.do_train:
        training_loop(
            model=model,
            args=cfg.args,
            train_data=train_dataset,
            val_data=val_dataset
        )
    
    elif cfg.args.do_eval and not cfg.args.do_train:
        eval_loop(
            model=model,
            args=cfg.args,
            data=val_dataset,
            epoch=None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)