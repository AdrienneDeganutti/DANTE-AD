import torch
import os
from os.path import join

def save_model(model, epoch):

    ckpt_path = model.videollama_config.output_ckpt_path
    output_dir = join(ckpt_path, f'epoch_{epoch}')

    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), join(output_dir, "model.pth"))

    print(f"Model saved to {output_dir}/model.pth")