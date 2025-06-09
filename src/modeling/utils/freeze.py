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