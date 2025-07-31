import os
import sys
import torch
from typing import Optional
from einops import rearrange
from diffusers import TextToVideoSDPipeline, DDIMScheduler
from tqdm.auto import tqdm
import copy
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.ddim_utils import ddim_inversion
from utils.dataset import CachedDataset

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch_dtype = torch.float16
else:
    DEVICE = "cpu"
    torch_dtype = torch.float32

def tensor_to_vae_latent(t: torch.Tensor, vae) -> torch.Tensor:
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    return latents

def handle_cache_latents(
    should_cache: bool,
    output_dir: str,
    train_dataloader,
    train_batch_size: int,
    vae,
    unet,
    pretrained_model_path: str,
    noise_prior: float,
    cached_latent_dir: Optional[str] = None,
):
    if not should_cache:
        return None
    vae.to(DEVICE, dtype=torch_dtype)
    vae.enable_slicing()
    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        vae=vae,
        unet=copy.deepcopy(unet).to(DEVICE, dtype=torch_dtype)
    )
    pipe.text_encoder.to(DEVICE, dtype=torch_dtype)
    cached_latent_dir = os.path.abspath(cached_latent_dir) if cached_latent_dir else None
    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)
        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):
            save_name = f"cached_{i}"
            full_out_path = f"{cache_save_dir}/{save_name}.pt"
            pixel_values = batch['pixel_values'].to(DEVICE, dtype=torch_dtype)
            batch['latents'] = tensor_to_vae_latent(pixel_values, vae)
            if noise_prior > 0.:
                batch['inversion_noise'] = ddim_inversion(pipe, DDIMScheduler.from_config(pipe.scheduler.config), video_latent=batch['latents'].to(DEVICE), num_inv_steps=50, prompt="")[-1]
            for k, v in batch.items():
                batch[k] = v[0]
            torch.save(batch, full_out_path)
            del pixel_values
            del batch
            torch.cuda.empty_cache()
            gc.collect()
    else:
        cache_save_dir = cached_latent_dir
    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir),
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0
    )
