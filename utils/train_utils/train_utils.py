import torch
import torch.nn.functional as F
import random
import logging
import math
import gc
import copy
import os
from torchvision import transforms
from diffusers import DDIMScheduler, TextToVideoSDPipeline
import numpy as np
import imageio
from typing import Dict, Tuple, List

from utils.lora import extract_lora_child_module

already_printed_trainables = False

def sample_noise(latents: torch.Tensor, noise_strength: float, use_offset_noise: bool = False) -> torch.Tensor:
    b, c, f, *_ = latents.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_latents = torch.randn_like(latents, device=device)
    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=device)
        noise_latents = noise_latents + noise_strength * offset_noise
    return noise_latents

def enforce_zero_terminal_snr(betas):
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

def handle_trainable_modules(model, trainable_modules: Tuple[str] = None, is_enabled: bool = True, negation: List[str] = None):
    global already_printed_trainables
    unfrozen_params = 0
    if trainable_modules:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break
                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled:
                            unfrozen_params += 1
    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")

def export_to_video(video_frames, output_path: str, fps: int):
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    elif isinstance(video_frames, list):
        video_frames = np.array(video_frames)
    if len(video_frames.shape) == 5:
        video_frames = video_frames[0]
    if video_frames.shape[1] == 3 or video_frames.shape[1] == 4:
        video_frames = video_frames.transpose(0, 2, 3, 1)
    if video_frames.max() <= 1.0:
        video_frames = (video_frames * 255).astype(np.uint8)
    else:
        video_frames = video_frames.astype(np.uint8)
    if video_frames.shape[-1] == 4:
        video_frames = video_frames[..., :3]
    elif video_frames.shape[-1] == 1:
        video_frames = np.repeat(video_frames, 3, axis=-1)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()

def should_sample(global_step: int, validation_steps: int, validation_data: Dict) -> bool:
    return global_step % validation_steps == 0 and validation_data.get('sample_preview', False)

def save_pipe(
    path: str,
    global_step: int,
    accelerator,
    unet,
    text_encoder,
    vae,
    output_dir: str,
    lora_manager_spatial,
    lora_manager_temporal,
    unet_target_replace_module: List[str] = None,
    text_target_replace_module: List[str] = None,
    is_checkpoint: bool = False,
    save_pretrained_model: bool = True
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet.cpu(), keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder.cpu(), keep_fp32_wrapper=False))
    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch.float32)
    lora_manager_spatial.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/spatial', step=global_step)
    if lora_manager_temporal:
        lora_manager_temporal.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/temporal', step=global_step)
    if save_pretrained_model:
        pipeline.save_pretrained(save_path)
    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]
    logging.info(f"Saved model at {save_path} on step {global_step}")
    del pipeline, unet_out, text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()

def unet_and_text_g_c(unet, text_encoder, unet_enable: bool, text_enable: bool):
    if hasattr(unet, '_set_gradient_checkpointing'):
        unet._set_gradient_checkpointing(unet_enable)
    if hasattr(text_encoder, '_set_gradient_checkpointing'):
        text_encoder._set_gradient_checkpointing(text_enable)
