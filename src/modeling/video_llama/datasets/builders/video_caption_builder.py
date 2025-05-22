import os
import logging
import warnings

from src.modeling.video_llama.common.registry import registry
from src.modeling.video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from src.dataset.cmd_dataset import CMDDataset

@registry.register_builder("cmd_AD")
class CMDBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMDDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/cmd_ad.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets