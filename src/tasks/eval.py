# Copyright (c) 2025 Adrienne Deganutti
# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description
# Licensed under the MIT License

import torch

from src.utils.save_outputs import log_predictions

def compute_acc(pred, target):
    batch_score = pred == target
    accuracy = torch.mean(batch_score.float())
    return accuracy


def eval_loop(model, args, data, epoch):

    model.eval()
    total_acc = 0

    with torch.no_grad():
        for batch in data:
            
            frame_fts, annotations, filename, s4v_features = batch["frame_fts"], batch["caption"], batch["filename"], batch["s4v_features"]

            if args.load_frame_features:
                assert frame_fts.shape == (1, 1, 32, 768)
            else:
                assert frame_fts.shape == (1, 3, 8, 224, 224)

            frame_fts = frame_fts.to('cuda:{}'.format(args.gpu_id))
            s4v_features = s4v_features.to('cuda:{}'.format(args.gpu_id))

            generated_texts, generated_ids, targets = model(frame_fts, annotations, s4v_features)

            accuracy = compute_acc(generated_ids, targets)
            total_acc += accuracy.item()

            log_predictions(args, generated_texts, filename, annotations, epoch)
        
        avg_acc = total_acc / len(data)
        print(f"Validation accuracy = {avg_acc:.4f}")