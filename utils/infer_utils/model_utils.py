import torch
import warnings
from diffusers import DDIMScheduler, TextToVideoSDPipeline
import os
import sys
# Add the directory of the current file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.lora_handler import LoraHandler
from utils.video_pipeline import (
    load_primary_models, 
    freeze_models, 
    handle_memory_attention, 
    unet_and_text_g_c
)

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    spatial_lora_path: str = "",
    temporal_lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
    spatial_lora_scale: float = 1.0,
    temporal_lora_scale: float = 1.0,
    is_multi: bool = False
) -> TextToVideoSDPipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)
    
    freeze_models([vae, text_encoder, unet])
    handle_memory_attention(xformers, sdp, unet)
    
    if is_multi:
        lora_manager_spatial = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=True,
            use_text_lora=False,
            save_for_webui=False,
            only_for_webui=False,
            unet_replace_modules=["Transformer2DModel"],
            text_encoder_replace_modules=None,
            lora_bias=None
        )
        lora_manager_temporal = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=True,
            use_text_lora=False,
            save_for_webui=False,
            only_for_webui=False,
            unet_replace_modules=["TransformerTemporalModel"],
            text_encoder_replace_modules=None,
            lora_bias=None
        )
        lora_manager_spatial.add_lora_to_model(
            True, unet, lora_manager_spatial.unet_replace_modules, 0, spatial_lora_path, r=lora_rank, scale=spatial_lora_scale
        )
        lora_manager_temporal.add_lora_to_model(
            True, unet, lora_manager_temporal.unet_replace_modules, 0, temporal_lora_path, r=lora_rank, scale=temporal_lora_scale
        )
    else:
        lora_manager_temporal = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=True,
            use_text_lora=False,
            save_for_webui=False,
            only_for_webui=False,
            unet_replace_modules=["TransformerTemporalModel"],
            text_encoder_replace_modules=None,
            lora_bias=None
        )
        lora_manager_temporal.add_lora_to_model(
            True, unet, lora_manager_temporal.unet_replace_modules, 0, lora_path, r=lora_rank, scale=lora_scale
        )
    
    unet.eval()
    text_encoder.eval()
    unet_and_text_g_c(unet, text_encoder, False, False)
    
    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.float16),
        vae=vae.to(device=device, dtype=torch.float16),
        unet=unet.to(device=device, dtype=torch.float16),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    return pipe