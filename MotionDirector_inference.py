import argparse
import os
import platform
import re
import warnings
from typing import Optional

import torch
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
import random

from MotionDirector_train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
import imageio


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(xformers, sdp, unet)

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

    unet_lora_params, unet_negation = lora_manager_temporal.add_lora_to_model(
        True, unet, lora_manager_temporal.unet_replace_modules, 0, lora_path, r=lora_rank, scale=lora_scale)

    unet.eval()
    text_encoder.eval()
    unet_and_text_g_c(unet, text_encoder, False, False)

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe


def inverse_video(pipe, latents, num_steps):
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)

    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent


def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    latents_path:str,
    noise_prior: float
):
    # initialize with random gaussian noise
    scale = pipe.vae_scale_factor
    shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
    if noise_prior > 0.:
        cached_latents = torch.load(latents_path)
        if 'inversion_noise' not in cached_latents:
            latents = inverse_video(pipe, cached_latents['latents'].unsqueeze(0), 50).squeeze(0)
        else:
            latents = torch.load(latents_path)['inversion_noise'].unsqueeze(0)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        if latents.shape != shape:
            latents = interpolate(rearrange(latents, "b c f h w -> (b f) c h w", b=batch_size), (height // scale, width // scale), mode='bilinear')
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
        noise = torch.randn_like(latents, dtype=torch.half)
        latents = (noise_prior) ** 0.5 * latents + (1 - noise_prior) ** 0.5 * noise
    else:
        latents = torch.randn(shape, dtype=torch.half)

    return latents


def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents




@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    num_steps: int = 50,
    guidance_scale: float = 15,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
    seed: Optional[int] = None,
    latents_path: str="",
    noise_prior: float = 0.
):
    if seed is not None:
        torch.manual_seed(seed)

    with torch.autocast(device, dtype=torch.half):
        # prepare models
        pipe = initialize_pipeline(model, device, xformers, sdp, lora_path, lora_rank, lora_scale)

        # prepare input latents
        init_latents = prepare_input_latents(
            pipe=pipe,
            batch_size=len(prompt),
            num_frames=num_frames,
            height=height,
            width=width,
            latents_path=latents_path,
            noise_prior=noise_prior
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

    return video_frames


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt to condition on")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-o", "--output_dir", type=str, default="./outputs/inference", help="Directory to save output video to")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=384, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=384, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-s", "--num-steps", type=int, default=30, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=12, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-f", "--fps", type=int, default=8, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-cf", "--checkpoint_folder", type=str, required=True, help="Path to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-lr", "--lora_rank", type=int, default=32, help="Size of the LoRA checkpoint's projection matrix (defaults to 32).")
    parser.add_argument("-ls", "--lora_scale", type=float, default=1.0, help="Scale of LoRAs.")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("-np", "--noise_prior", type=float, default=0., help="Random seed to make generations reproducible.")
    parser.add_argument("-ci", "--checkpoint_index", type=int, required=True,
                        help="Random seed to make generations reproducible.")

    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    out_name = f"{args.output_dir}/"
    prompt = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", args.prompt) if platform.system() == "Windows" else args.prompt
    out_name += f"{prompt}".replace(' ','_').replace(',', '').replace('.', '')

    args.prompt = [prompt] * args.batch_size
    if args.negative_prompt is not None:
        args.negative_prompt = [args.negative_prompt] * args.batch_size

    # =========================================
    # ============= sample videos =============
    # =========================================

    lora_path = f"{args.checkpoint_folder}/checkpoint-{args.checkpoint_index}/temporal/lora"
    latents_folder = f"{args.checkpoint_folder}/cached_latents"
    latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
    if args.seed is None:
        args.seed = random.randint(100, 10000000)
    assert os.path.exists(lora_path)
    video_frames = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        lora_path=lora_path,
        lora_rank=args.lora_rank,
        lora_scale = args.lora_scale,
        seed=args.seed,
        latents_path=latents_path,
        noise_prior=args.noise_prior
    )
    # =========================================
    # ========= write outputs to file =========
    # =========================================
    os.makedirs(args.output_dir, exist_ok=True)

    # save to mp4
    export_to_video(video_frames, f"{out_name}_{args.seed}.mp4", args.fps)

    # # save to gif
    # file_name = f"{out_name}_{args.seed}.gif"
    # imageio.mimsave(file_name, video_frames, 'GIF', duration=1000 * 1 / args.fps, loop=0)

    