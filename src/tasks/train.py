# Copyright (c) 2025 Adrienne Deganutti
# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description
# Licensed under the MIT License

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.save_outputs import save_model
from src.tasks.eval import eval_loop


def mixed_precision(cfg, model):
    learning_rate = cfg.learning_rate
    optimizer = optim.AdamW([
        {"params": model.llama_proj.parameters(), "lr": learning_rate},
        {"params": model.s4v_proj.parameters(), "lr": learning_rate},
        {"params": model.crossattention.parameters(), "lr": learning_rate},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-5)

    return learning_rate, optimizer, scheduler


def training_loop(model, args, train_data, val_data):

    learning_rate, optimizer, scheduler = mixed_precision(args, model)

    model.train()

    for epoch in range(args.num_epochs):
        total_loss = 0

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        model.train()
    
        for batch in train_data:
            frame_fts, annotations, filename, s4v_features = batch["frame_fts"], batch["caption"], batch["filename"], batch["s4v_features"]

            if args.load_frame_features:
                assert frame_fts.shape == (1, 1, 32, 768)
            else:
                assert frame_fts.shape == (1, 3, 8, 224, 224)

            frame_fts = frame_fts.to('cuda:{}'.format(args.gpu_id))
            s4v_features = s4v_features.to('cuda:{}'.format(args.gpu_id))

            optimizer.zero_grad()
            loss = model(frame_fts, annotations, s4v_features)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step()

        avg_loss = total_loss / len(train_data)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.8f}")

        save_model(model, epoch)

        if args.do_eval:
            eval_loop(
                model=model,
                args=args,
                data=val_data,
                epoch=epoch
            )