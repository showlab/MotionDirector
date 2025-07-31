import torch
import random
import os
import imageio
from typing import List, Dict, Optional, Tuple
import sys
import uuid

# Add the directory of the current file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video_pipeline import export_to_video

def inference(
    model: str,
    prompt: List[str],
    negative_prompt: Optional[List[str]],
    width: int,
    height: int,
    num_frames: int,
    num_steps: int,
    guidance_scale: float,
    device: str,
    xformers: bool,
    sdp: bool,
    lora_path: str = "",
    spatial_lora_path: str = "",
    temporal_lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
    spatial_lora_scale: float = 1.0,
    temporal_lora_scale: float = 1.0,
    seed: Optional[int] = None,
    latents_path: str = "",
    noise_prior: float = 0.,
    repeat_num: int = 1,
    fps: int = 8,
    out_name: str = "./outputs/inference",
    is_multi: bool = False,
    no_prompt_name: bool = False
) -> List:
    from model_utils import initialize_pipeline
    from latent_utils import prepare_input_latents
    
    video_frames_list = []
    with torch.autocast(device, dtype=torch.float16):
        pipe = initialize_pipeline(
            model, device, xformers, sdp, lora_path, spatial_lora_path, temporal_lora_path,
            lora_rank, lora_scale, spatial_lora_scale, temporal_lora_scale, is_multi
        )
        
        for i in range(repeat_num):
            random_seed = seed if seed is not None else random.randint(100, 10000000)
            torch.manual_seed(random_seed)
            
            init_latents = prepare_input_latents(
                pipe=pipe,
                batch_size=len(prompt),
                num_frames=num_frames,
                height=height,
                width=width,
                latents_path=latents_path,
                noise_prior=noise_prior,
                device=device
            )
            
            with torch.no_grad():
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    latents=init_latents
                ).frames
            
            out_dir = os.path.dirname(out_name)
            os.makedirs(out_dir, exist_ok=True)
            if no_prompt_name:
                out_name = os.path.join(out_dir, f"{uuid.uuid4()}_{random_seed}")

            export_to_video(video_frames, f"{out_name}_{random_seed}.mp4", fps)
            # imageio.mimsave(f"{out_name}_{random_seed}.gif", video_frames, 'GIF', duration=1000 * 1 / fps, loop=0)
            video_frames_list.append(video_frames)
    
    return video_frames_list
