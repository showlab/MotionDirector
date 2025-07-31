from typing import List
import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from diffusers.utils.import_utils import is_xformers_available

def is_attn(name: str) -> bool:
    return 'attn1' in name or 'attn2' in name

def set_processors(attentions):
    for attn in attentions:
        attn.set_processor(AttnProcessor2_0())

def set_torch_2_attn(unet):
    optim_count = 0
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention: bool, enable_torch_2_attn: bool, unet):
    try:
        is_torch_2 = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        if enable_torch_2:
            set_torch_2_attn(unet)
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def cast_to_gpu_and_type(model_list: List, accelerator, weight_dtype: torch.dtype):
    for model in model_list:
        if model is not None:
            model.to(accelerator.device, dtype=weight_dtype)