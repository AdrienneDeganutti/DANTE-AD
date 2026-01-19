import os
import torch
import pandas as pd

from typing import Dict, Any
from pathlib import Path

class VideoDataset:
    def __init__(self, args, config: Dict[str, Any], vis_processor, split):
        
        self.args = args
        build_info = config['build_info']
        
        if split == 'train':
            self.anno_path = build_info['train_anno_dir']
        else:
            self.anno_path = build_info['val_anno_dir']
        
        self.s4v_dir = build_info['s4v_features']

        if self.args.load_frame_features:    # Load pre-computed frame-level features
            self.video_qformer_ft = build_info['video_qformer_ft']
        else:
            self.videos_dir = build_info['videos_dir']

        self.vis_processor_config = config['vis_processor']['train']
        self.text_processor_config = config['text_processor']['train']
        
        self.vis_processor = vis_processor

        self.annotations = self._load_annotations()
    

    def _load_annotations(self):
        """Load the annotations from a TSV file."""
        
        assert Path(self.anno_path).exists()
        self.df = pd.read_csv(self.anno_path, sep='\t', usecols=["text", "cmd_ad_filename"])

        # Convert each text entry into a list of dictionaries
        self.df['captions'] = self.df['text'].apply(lambda x: [{"caption": x}])

        # Convert the DataFrame to a list of dictionaries
        self.data = self.df.to_dict('records')
        self.dict_data = {str(row['cmd_ad_filename']): row for row in self.data}
        
        return self.data
    
    
    def load_features(self, features_path):
        features = torch.load(features_path, map_location='cpu', weights_only=False)
        return features


    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, idx):
        row = self.annotations[idx]

        caption = row['text']

        s4v_path = os.path.join(self.s4v_dir, row['cmd_ad_filename'] + '.pt')
        s4v_features = self.load_features(s4v_path)

        if self.args.load_frame_features:
            qformer_ft = os.path.join(self.video_qformer_ft, row['cmd_ad_filename'] + '.pt')
            vid_qformer_ft = self.load_features(qformer_ft)

            return {'frame_fts': vid_qformer_ft, 'caption': caption, 'filename': row['cmd_ad_filename'], 's4v_features': s4v_features}
        
        else:
            video_path = os.path.join(self.videos_dir, row["cmd_ad_filename"] + '.mkv')
            vid_frames = self.vis_processor(video_path)
        
            return {'frame_fts': vid_frames, 'caption': caption, 'filename': row['cmd_ad_filename'], 's4v_features': s4v_features}