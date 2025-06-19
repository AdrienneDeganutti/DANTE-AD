# Copyright (c) 2025 Adrienne Deganutti
# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description
# Licensed under the MIT License

# Modified from DAMO-NLP-SG/Video-LLaMA/video_llama/models/video_llama.py

import torch
import torch.nn as nn

from rich.console import Console
from rich.table import Table

from src.modeling.video_llama.common.registry import registry
from src.modeling.video_llama.models.blip2 import Blip2Base
from src.modeling.fusion_module.cross_attention import CaptionGenerator
from src.modeling.utils.freeze import(freeze_proj, unfreeze_proj, freeze_llama,
                                        freeze_s4v_proj, unfreeze_s4v_proj, 
                                        freeze_multihead_attn, unfreeze_multihead_attn)
from src.modeling.video_llama.models.modeling_llama import LlamaForCausalLM

from transformers import(
    LlamaTokenizer,
    AutoTokenizer)

console = Console()

@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama_v2": "configs/video_llama/video_llama.yaml",
    }

    def __init__(self, videollama_config):
        super().__init__()
        
        self.videollama_config = videollama_config
        self.end_sym = self.videollama_config.end_sym
        self.max_txt_len = self.videollama_config.max_txt_len
        
        console.rule("[bold yellow]Initializing VideoLLAMA Modules")

        console.print("[bold cyan]Loading LLaMA Tokenizer...[/bold cyan]")
        if self.videollama_config.llama_model == 'meta-llama/Llama-2-7b-hf':
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.videollama_config.llama_model, use_fast=False)
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        else:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(self.videollama_config.llama_model, use_fast=False)
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 


        console.print("[bold cyan]Loading LLaMA Model...[/bold cyan]")
        if self.videollama_config.llama_model == 'meta-llama/Llama-2-7b-hf':
            if self.videollama_config.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    self.videollama_config.llama_model,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map={'': self.videollama_config.device_8bit})
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    self.videollama_config.llama_model,
                    torch_dtype=torch.bfloat16,)
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                self.videollama_config.llama_model,
                torch_dtype=torch.bfloat16)
        freeze_llama(self.llama_model)


        console.print("[bold cyan]Loading LLaMA Projection Layer...[/bold cyan]")
        self.llama_proj = nn.Linear(
            768, self.llama_model.config.hidden_size
        )
        if self.videollama_config.llama_proj_model:
            print("load llama proj weight: {}".format(self.videollama_config.llama_proj_model))
            llama_proj_weight = torch.load(self.videollama_config.llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)
        if self.videollama_config.frozen_llama_proj:
            freeze_proj(self.llama_proj)
        else:
            unfreeze_proj(self.llama_proj)
        

        console.print("[bold cyan]Loading S4V Projection Layer...[/bold cyan]")
        self.s4v_proj = nn.Linear(320, self.llama_model.config.hidden_size)
        if self.videollama_config.freeze_s4v_proj:
            freeze_s4v_proj(self.s4v_proj)
        else:
            unfreeze_s4v_proj(self.s4v_proj)
        

        console.print("[bold cyan]Loading Fusion Module...[/bold cyan]")
        self.crossattention = CaptionGenerator(self.videollama_config)
        if self.videollama_config.freeze_crossattention:
            freeze_multihead_attn(self.crossattention)
        else:
            unfreeze_multihead_attn(self.crossattention)
        
        # Summary Table
        table = Table(title="VideoLLAMA Module Summary", show_lines=True)
        table.add_column("Module", style="bold green")
        table.add_column("Frozen?", justify="center")
        table.add_column("Parameters", justify="right")

        def param_count(model):
            return f"{sum(p.numel() for p in model.parameters()):,}"

        def is_frozen(model):
            return all(not p.requires_grad for p in model.parameters())

        table.add_row("LLaMA Model", str(is_frozen(self.llama_model)), param_count(self.llama_model))
        table.add_row("LLaMA Projection", str(is_frozen(self.llama_proj)), param_count(self.llama_proj))
        table.add_row("S4V Projection", str(is_frozen(self.s4v_proj)), param_count(self.s4v_proj))
        table.add_row("Fusion Module", str(is_frozen(self.crossattention)), param_count(self.crossattention))

        console.print(table)
    

    def forward(self, vid_qformer_ft, annotation, s4v_ft):
        
        # EVA-CLIP features
        vid_qformer_ft = vid_qformer_ft.squeeze(dim=1)
        img_embeds = self.llama_proj(vid_qformer_ft)

        # S4V features
        s4v_features = self.s4v_proj(s4v_ft)

        # Ground truth text processing
        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in annotation]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        batch_size = vid_qformer_ft.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=self.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        # CROSS-ATTENTION
        if self.training:
            x = self.crossattention(img_embeds=img_embeds, s4v_features=s4v_features, 
                                    text_embeds=to_regress_embeds, text_tokens=to_regress_tokens.input_ids)
        else:
            x = self.crossattention.eval(img_embeds=img_embeds, s4v_features=s4v_features, bos_embed=bos_embeds)
        x_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(self.device)


        if self.training:
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones([x_atts.shape[0], x_atts.shape[1]+1],
                    dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            eos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=self.device) * self.llama_tokenizer.eos_token_id
            eos_embeds = self.llama_model.model.embed_tokens(eos)
            eos_target = torch.full((targets.shape[0], 1), self.llama_tokenizer.eos_token_id, dtype=torch.long, device=self.device)
        
            atts_bos = x_atts[:, :1]
            atts_eos = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)

            targets = torch.cat([empty_targets, targets, eos_target], dim=1)

            inputs_embeds = torch.cat([bos_embeds, x, to_regress_embeds, eos_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, x_atts, to_regress_tokens.attention_mask, atts_eos], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return loss
        
        else:
            
            inputs_embeds = torch.cat([bos_embeds, x], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                generated_ids = self.llama_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_txt_len,
                    num_beams=3 
                )
            
            generated_texts = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Padding for accuracy computation
            if generated_ids.shape[1] > to_regress_tokens.input_ids.shape[1]:
                empty_target_len = generated_ids.shape[1] - to_regress_tokens.input_ids.shape[1]
                empty_targets = (
                    torch.ones([generated_ids.shape[0], empty_target_len-1],   # minus 1 for eos
                    dtype=torch.long).to(self.device).fill_(-100)
                )
                eos_target = torch.full((generated_ids.shape[0], 1), self.llama_tokenizer.eos_token_id, dtype=torch.long, device=self.device)
                targets = torch.cat([to_regress_tokens.input_ids, eos_target, empty_targets], dim=1)
            
            elif generated_ids.shape[1] < to_regress_tokens.input_ids.shape[1]:
                empty_target_len = to_regress_tokens.input_ids.shape[1] - generated_ids.shape[1]
                empty_targets = (
                    torch.ones([generated_ids.shape[0], empty_target_len+1],   # plus 1 for eos
                    dtype=torch.long).to(self.device).fill_(-100)
                )
                generated_ids = torch.cat([generated_ids, empty_targets], dim=1)
                eos_target = torch.full((generated_ids.shape[0], 1), self.llama_tokenizer.eos_token_id, dtype=torch.long, device=self.device)
                targets = torch.cat([to_regress_tokens.input_ids, eos_target], dim=1)
            else:
                empty_targets = (
                    torch.ones([generated_ids.shape[0], 1],   # plus 1 for eos
                    dtype=torch.long).to(self.device).fill_(-100)
                )
                generated_ids = torch.cat([empty_targets, generated_ids], dim=1)
                eos_target = torch.full((generated_ids.shape[0], 1), self.llama_tokenizer.eos_token_id, dtype=torch.long, device=self.device)
                targets = torch.cat([to_regress_tokens.input_ids, eos_target], dim=1)
            
            return generated_texts, generated_ids, targets
    

    @classmethod
    def from_config(cls, cfg, videollama_config):
        model = cls(videollama_config)

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Loading first checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['state_dict'], strict=False)
        
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Loading second checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt, strict=False)

        return model