# import argparse
# import datetime
# import logging
# import inspect
# import math
# import os
# import random
# import gc
# import copy

# from typing import Dict, Optional, Tuple
# from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
# import torch.utils.checkpoint
import diffusers
import transformers

# from torchvision import transforms
# from tqdm.auto import tqdm

# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, TextToVideoSDPipeline
# from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
# from diffusers.models.attention_processor import AttnProcessor2_0, Attention
# from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
# from transformers.models.clip.modeling_clip import CLIPEncoder
# from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
#     ImageDataset, VideoFolderDataset, CachedDataset
# from einops import rearrange, repeat
# from utils.lora_handler import LoraHandler
# from utils.lora import extract_lora_child_module
# from utils.ddim_utils import ddim_inversion
import imageio
import numpy as np

def export_to_video(video_frames, output_path, fps):
    # Ensure video_frames is a list or tensor of shape (batch_size, num_frames, channels, height, width)
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()  # Convert tensor to NumPy
    elif isinstance(video_frames, list):
        video_frames = np.array(video_frames)

    # Check shape and adjust if necessary
    if len(video_frames.shape) == 5:  # (batch_size, num_frames, channels, height, width)
        video_frames = video_frames[0]  # Take first batch if batch_size > 1
    if video_frames.shape[1] == 3 or video_frames.shape[1] == 4:  # Channels in second dim (num_frames, channels, height, width)
        video_frames = video_frames.transpose(0, 2, 3, 1)  # Reshape to (num_frames, height, width, channels)

    # Ensure pixel values are in [0, 255] and uint8
    if video_frames.max() <= 1.0:
        video_frames = (video_frames * 255).astype(np.uint8)
    else:
        video_frames = video_frames.astype(np.uint8)

    # Ensure exactly 3 channels (RGB)
    if video_frames.shape[-1] == 4:  # If RGBA, drop alpha channel
        video_frames = video_frames[..., :3]
    elif video_frames.shape[-1] == 1:  # If grayscale, convert to RGB
        video_frames = np.repeat(video_frames, 3, axis=-1)

    # Write video
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()

def handle_memory_attention(enable_xformers_memory_efficient_attention,
                            enable_torch_2_attn, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
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

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    if hasattr(unet, '_set_gradient_checkpointing'):
        print("unet._set_gradient_checkpointing(unet_enable)")
        unet._set_gradient_checkpointing(unet_enable)
    else:
        print("NO unet._set_gradient_checkpointing(unet_enable)")
        
    if hasattr(text_encoder, '_set_gradient_checkpointing'):
        text_encoder._set_gradient_checkpointing(text_enable)
    else:
        print("NO text_encoder._set_gradient_checkpointing(text_enable)")

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False)