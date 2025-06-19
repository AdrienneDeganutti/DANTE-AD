import torch
import csv
import os
from os.path import join

def save_model(model, epoch):

    ckpt_path = model.videollama_config.output_ckpt_path
    output_dir = join(ckpt_path, f'epoch_{epoch}')

    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), join(output_dir, "model.pth"))

    print(f"Model saved to {output_dir}/model.pth")


def log_predictions(args, decoded_text, filename, ground_truth, epoch):

    output_dir = join(args.output_results_path, f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = join(output_dir, 'eval_predictions.tsv')

    if not os.path.isfile(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['filename', 'prediction', 'ground truth'])
            sanitized_pred = decoded_text[0].replace('\n', ' ')
            writer.writerow([filename[0], f'{sanitized_pred}', ground_truth[0]])
    else:
        with open(output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            sanitized_pred = decoded_text[0].replace('\n', ' ')
            writer.writerow([filename[0], f'{sanitized_pred}', ground_truth[0]])

