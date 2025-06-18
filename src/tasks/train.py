# Copyright (c) 2025 Adrienne Deganutti
# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description
# Licensed under the MIT License

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.save_model_ckpt import save_model

def compute_acc(logits, targets):
    logits = torch.max(logits, -1)[1].data
    batch_score = logits == targets
    accuracy = torch.mean(batch_score.float())
    return accuracy


def mixed_precision(cfg, model):
    learning_rate = cfg.learning_rate
    optimizer = optim.AdamW([
        {"params": model.llama_proj.parameters(), "lr": learning_rate},
        {"params": model.s4v_proj.parameters(), "lr": learning_rate},
        {"params": model.crossattention.parameters(), "lr": learning_rate},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-5)

    return learning_rate, optimizer, scheduler


def training_loop(model, args, data):

    learning_rate, optimizer, scheduler = mixed_precision(args, model)

    model.train()

    for epoch in range(args.num_epochs):
        total_loss = 0
        total_acc = 0

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        model.train()
    
        for batch in data:
            vid_qformer_ft, annotations, filename, s4v_features = batch["vid_qformer_ft"], batch["caption"], batch["filename"], batch["s4v_features"]
        
            vid_qformer_ft = vid_qformer_ft.to('cuda:{}'.format(args.gpu_id))
            s4v_features = s4v_features.to('cuda:{}'.format(args.gpu_id))

            optimizer.zero_grad()
            loss = model(vid_qformer_ft, annotations, s4v_features, filename, epoch)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step()

        avg_loss = total_loss / len(data)
        avg_acc = total_acc / len(data)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.8f}")

        save_model(model, epoch)