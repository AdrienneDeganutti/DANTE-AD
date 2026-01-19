# Modified based on the original code from:
# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from src.modeling.fusion_module.attention import MultiHeadAttention
from src.modeling.fusion_module.pos_embed import FeedForward, sinusoid_encoding_table


class GeneratorLayer(nn.Module):

    def __init__(self, d_model=4096, n_heads=8, dropout=.1, n_memories=0):
        super().__init__()

        self.self_att = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories, can_be_stateful=True)
        self.pwff = FeedForward(d_model, d_model, dropout)


class ConcatAttentionLayer(GeneratorLayer):
    def __init__(self, d_model=4096, n_heads=8, dropout=0.1, n_memories=0):
        super().__init__()

        self.vis_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

    def forward(self, x, y, mask_x, mask_y, mask_pad):

        self_att_out, self_attn_weights = self.self_att(x, x, x, mask_x) 
        if self.training:
            self_att_out = self_att_out * mask_pad
        
        cross_att_out, _ = self.vis_att(self_att_out, y, y, mask_y) 
        if self.training:
            cross_att_out = cross_att_out * mask_pad

        out = self.pwff(cross_att_out) 
        if self.training:
            out = out * mask_pad

        return out, self_attn_weights


class SequentialAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, dropout=.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, dropout=dropout, n_memories=0)

        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_model, dropout)

    def forward(self, x, y1, y2, mask_x, mask_y1, mask_y2, mask_pad):
        out, self_attn_weights = self.self_att(x, x, x, mask_x)
        if self.training:
            out = out * mask_pad
        out, _ = self.vis_att1(out, y1, y1, mask_y1)
        if self.training:
            out = out * mask_pad
        out, _ = self.vis_att2(out, y2, y2, mask_y2)
        if self.training:
            out = out * mask_pad
        ff = self.pwff(out)
        if self.training:
            ff = ff * mask_pad
        return ff, self_attn_weights



class CaptionGenerator(nn.Module):
    GENERATOR_LAYER = {
        'concat': ConcatAttentionLayer,
        'sequential': SequentialAttentionLayer,
    }

    def __init__(self, cfg, d_model=4096, n_heads=8, dropout=0.1, n_memories=0):
        super().__init__()

        self.device = 'cuda:{}'.format(cfg.gpu_id)
        self.n_layers = 3
        self.max_len = 54

        self.decoder_name = cfg.crossattention_decoder
        generator_layer = self.GENERATOR_LAYER[self.decoder_name]

        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(self.max_len + 1, d_model, 0), freeze=True)

        self.layers = nn.ModuleList([
            generator_layer(d_model, n_heads, dropout, n_memories)
            for _ in range(self.n_layers)
        ])


    def forward(self, img_embeds, s4v_features, text_embeds, text_tokens):

        seq_len = text_tokens.shape[1]
        b_s = img_embeds.shape[0]

        mask_pad = (text_tokens != -100).unsqueeze(-1).float()      # (B, seq_len, 1)

        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device), diagonal=1)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0).repeat(b_s, 1, 1, 1).clone()  # (B, 1, seq_len, seq_len)
        mask_x |= (text_tokens == -100).unsqueeze(1).unsqueeze(1)
        mask_x = mask_x.gt(0)   # (B, 1, seq_len, seq_len)

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(self.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)

        x = text_embeds + self.pos_emb(seq)
        

        if self.decoder_name == 'concat':
            # Concatenate visual features (used in ablation)
            y = torch.cat([s4v_features, img_embeds], dim=1)
            V = y.shape[1]
            mask_y = torch.zeros((b_s, 1, 1, V), dtype=torch.bool, device=self.device)

            for layer in self.layers:
                x, self_attn_weights = layer(x, y, mask_x, mask_y, mask_pad)
        

        elif self.decoder_name == 'sequential':
            # Sequential fusion of visual features
            y1 = s4v_features
            y2 = img_embeds
            mask_y1 = torch.zeros((b_s, 1, 1, y1.shape[1]), dtype=torch.bool, device=self.device)
            mask_y2 = torch.zeros((b_s, 1, 1, y2.shape[1]), dtype=torch.bool, device=self.device)

            for layer in self.layers:
                x, self_attn_weights = layer(x, y1, y2, mask_x, mask_y1, mask_y2, mask_pad)
        
        return x
    

    def eval(self, img_embeds, s4v_features, bos_embed, max_length=20):

        b_s = img_embeds.shape[0]
        generated_embeds = bos_embed

        if self.decoder_name == 'concat':
            fused_features = torch.cat([s4v_features, img_embeds], dim=1)

        for t in range(max_length):
        
            seq_len = generated_embeds.shape[1]
        
            mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device), diagonal=1)
            mask_x = mask_x.unsqueeze(0).unsqueeze(0).repeat(b_s, 1, 1, 1).clone()

            seq = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(b_s, -1)

            x = generated_embeds + self.pos_emb(seq)

            if self.decoder_name == 'concat':
                y = fused_features
                V = y.shape[1]
                mask_y = torch.zeros((b_s, 1, 1, V), dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    for layer in self.layers:
                        x, self_attn_weights = layer(x, y, mask_x, mask_y, mask_pad=None)
            
            elif self.decoder_name == 'sequential':
                y1 = s4v_features
                y2 = img_embeds
                mask_y1 = torch.zeros((b_s, 1, 1, y1.shape[1]), dtype=torch.bool, device=self.device)
                mask_y2 = torch.zeros((b_s, 1, 1, y2.shape[1]), dtype=torch.bool, device=self.device)

                for layer in self.layers:
                    x, self_attn_weights = layer(x, y1, y2, mask_x, mask_y1, mask_y2, mask_pad=None)

            logits = x[:, -1, :].unsqueeze(1)

            generated_embeds = torch.cat([generated_embeds, logits], dim=1)

        return generated_embeds
