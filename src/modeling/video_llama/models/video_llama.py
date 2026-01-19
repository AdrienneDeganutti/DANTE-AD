# Copyright (c) 2025 Adrienne Deganutti
# DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description
# Licensed under the MIT License

# Modified from DAMO-NLP-SG/Video-LLaMA/video_llama/models/video_llama.py

import torch
import torch.nn as nn

import einops
from rich.console import Console
from rich.table import Table

from src.modeling.video_llama.common.registry import registry
from src.modeling.video_llama.models.blip2 import Blip2Base
from src.modeling.fusion_module.cross_attention import CaptionGenerator
from src.modeling.utils.freeze import(freeze_proj, unfreeze_proj, freeze_llama,
                                        freeze_s4v_proj, unfreeze_s4v_proj, 
                                        freeze_multihead_attn, unfreeze_multihead_attn,
                                        freeze_vit, freeze_qformer, unfreeze_qformer,
                                        freeze_video_qformer, unfreeze_video_qformer)
from src.modeling.video_llama.models.modeling_llama import LlamaForCausalLM
from src.modeling.video_llama.models.Qformer import BertConfig, BertLMHeadModel

from transformers import(
    LlamaTokenizer,
    AutoTokenizer)

console = Console()

@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama_v2": "configs/video_llama/video_llama.yaml",
    }


    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention at every layer 
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


    def __init__(self, videollama_config):
        super().__init__()
        
        self.videollama_config = videollama_config
        self.end_sym = self.videollama_config.end_sym
        self.max_txt_len = self.videollama_config.max_txt_len
        
        console.rule("[bold yellow]Initializing VideoLLAMA Modules")

        if not self.videollama_config.load_frame_features:
            console.print("[bold cyan]Loading Vision Transformer...[/bold cyan]")
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                self.videollama_config.vit_model, self.videollama_config.img_size, self.videollama_config.drop_path_rate,
                self.videollama_config.use_grad_checkpoint, self.videollama_config.vit_precision)
            if self.videollama_config.freeze_vit:
                freeze_vit(self.visual_encoder, self.ln_vision)

            console.print("[bold cyan]Loading Q-Former...[/bold cyan]")
            self.Qformer, self.query_tokens = self.init_Qformer(
                self.videollama_config.num_query_token, self.visual_encoder.num_features)
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(url_or_filename=self.videollama_config.q_former_model)
            if self.videollama_config.freeze_qformer:
                freeze_qformer(self.Qformer, self.query_tokens)
            else:
                unfreeze_qformer(self.Qformer, self.query_tokens)
            
            self.video_frame_position_embedding = nn.Embedding(self.videollama_config.max_frame_pos, self.Qformer.config.hidden_size)
            self.video_frames_position_embedding = nn.Embedding(self.videollama_config.num_subsampled_frames, self.Qformer.config.hidden_size)

            console.print("[bold cyan]Loading Video Q-Former...[/bold cyan]")
            self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = videollama_config.num_video_query_token,
                vision_width=self.Qformer.config.hidden_size, num_hidden_layers=2)
            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            if videollama_config.frozen_video_Qformer:
                freeze_video_qformer(self.video_Qformer, self.video_frames_position_embedding, self.video_query_tokens)
            else:
                unfreeze_video_qformer(self.video_Qformer, self.video_frames_position_embedding, self.video_query_tokens)

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

        if not self.videollama_config.load_frame_features:
            table.add_row("Vision Transformer", str(is_frozen(self.visual_encoder)), param_count(self.visual_encoder))
            table.add_row("Q-Former", str(is_frozen(self.Qformer)), param_count(self.Qformer))
            table.add_row("Video Q-Former", str(is_frozen(self.video_Qformer)), param_count(self.video_Qformer))
        table.add_row("LLaMA Model", str(is_frozen(self.llama_model)), param_count(self.llama_model))
        table.add_row("LLaMA Projection", str(is_frozen(self.llama_proj)), param_count(self.llama_proj))
        table.add_row("S4V Projection", str(is_frozen(self.s4v_proj)), param_count(self.s4v_proj))
        table.add_row("Fusion Module", str(is_frozen(self.crossattention)), param_count(self.crossattention))

        console.print(table)
    

    def forward(self, frame_fts, annotation, s4v_ft):
        
        if self.videollama_config.load_frame_features:
            # EVA-CLIP features
            frame_fts = frame_fts.squeeze(dim=1)
            img_embeds = self.llama_proj(frame_fts)
        else:
            img_embeds = self.encode_Qformer_visual(frame_fts)

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

        batch_size = frame_fts.shape[0]
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
    

    def encode_Qformer_visual(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            inputs_llama = inputs_llama.float()

        return inputs_llama


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