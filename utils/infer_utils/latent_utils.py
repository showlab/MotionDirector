import torch
from torch import Tensor
from torch.nn.functional import interpolate
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from einops import rearrange
from utils.ddim_utils import ddim_inversion
from tqdm import trange

def inverse_video(pipe: TextToVideoSDPipeline, latents: Tensor, num_steps: int) -> Tensor:
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)
    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt=""
    )[-1]
    return ddim_inv_latent

def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    latents_path: str,
    noise_prior: float,
    device: str = "cuda"
) -> Tensor:
    scale = pipe.vae_scale_factor
    shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
    
    if noise_prior > 0.:
        cached_latents = torch.load(latents_path, map_location=torch.device(device))
        for key in cached_latents:
            try:
                print(f"cached_latents Key: {key}, Value:\n{cached_latents[key].shape}\n")
            except:
                print(f"cached_latents Key: {key}, Value:\n{cached_latents[key]}\n")
        
        if 'inversion_noise' not in cached_latents:
            latents = inverse_video(pipe, cached_latents['latents'].unsqueeze(0), 50).squeeze(0)
        else:
            latents = cached_latents['inversion_noise'].unsqueeze(0)
        
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        
        if latents.shape[2] != num_frames:
            latents = rearrange(latents, "b c f h w -> b c h w f")
            latents = interpolate(latents, size=(latents.shape[2], latents.shape[3], num_frames), mode='trilinear', align_corners=False)
            latents = rearrange(latents, "b c h w f -> b c f h w")
        
        if latents.shape[3:] != shape[3:]:
            latents = interpolate(rearrange(latents, "b c f h w -> (b f) c h w", b=batch_size), 
                                size=(height // scale, width // scale), mode='bilinear')
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
        
        noise = torch.randn_like(latents, dtype=torch.float16)
        latents = (noise_prior ** 0.5) * latents + ((1 - noise_prior) ** 0.5) * noise
    else:
        latents = torch.randn(shape, dtype=torch.float16)
    
    return latents

def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8) -> Tensor:
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")
    latents = []
    for idx in trange(0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.float16)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)
    return latents
