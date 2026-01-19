from src.modeling.video_llama.models.blip2 import disabled_train

def freeze_vit(visual_encoder, ln_vision):
    for name, param in visual_encoder.named_parameters():
        param.requires_grad = False
    visual_encoder = visual_encoder.eval()
    visual_encoder.train = disabled_train
    for name, param in ln_vision.named_parameters():
        param.requires_grad = False
    ln_vision = ln_vision.eval()
    ln_vision.train = disabled_train

def freeze_qformer(Qformer, query_tokens):
    for name, param in Qformer.named_parameters():
        param.requires_grad = False
    Qformer = Qformer.eval()
    Qformer.train = disabled_train
    query_tokens.requires_grad = False

def unfreeze_qformer(Qformer, query_tokens):
    for name, param in Qformer.named_parameters():
        param.requires_grad = True
    query_tokens.requires_grad = True

def freeze_video_qformer(video_Qformer, video_frame_position_embedding, video_query_tokens):
    for name, param in video_Qformer.named_parameters():
        param.requires_grad = False
    for name, param in video_frame_position_embedding.named_parameters():
        param.requires_grad = False
    video_query_tokens.requires_grad = False       

def unfreeze_video_qformer(video_Qformer, video_frame_position_embedding, video_query_tokens):
    for name, param in video_Qformer.named_parameters():
        param.requires_grad = True
    for name, param in video_frame_position_embedding.named_parameters():
        param.requires_grad = True
    video_query_tokens.requires_grad = True

def freeze_proj(llama_proj):
    for name, param in llama_proj.named_parameters():
        param.requires_grad = False

def unfreeze_proj(llama_proj):
    for name, param in llama_proj.named_parameters():
        param.requires_grad = True

def freeze_llama(llama_model):
    for name, param in llama_model.named_parameters():
        param.requires_grad = False

def freeze_s4v_proj(s4v_proj):
    for name, param in s4v_proj.named_parameters():
        param.requires_grad = False

def unfreeze_s4v_proj(s4v_proj):
    for name, param in s4v_proj.named_parameters():
        param.requires_grad = True

def freeze_multihead_attn(multihead_attn):
    for name, param in multihead_attn.named_parameters():
        param.requires_grad = False

def unfreeze_multihead_attn(multihead_attn):
    for name, param in multihead_attn.named_parameters():
        param.requires_grad = True