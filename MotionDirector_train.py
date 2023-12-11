import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset
from einops import rearrange, repeat
from utils.lora_handler import LoraHandler
from utils.lora import extract_lora_child_module
from utils.ddim_utils import ddim_inversion
import imageio
import numpy as np


already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def get_train_dataset(dataset_types, train_data, tokenizer):
    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))

    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")


def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)


def export_to_video(video_frames, output_video_path, fps):
    video_writer = imageio.get_writer(output_video_path, fps=fps)
    for img in video_frames:
        video_writer.append_data(np.array(img))
    video_writer.close()


def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    # OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir


def load_primary_models(pretrained_model_path):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)


def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False)


def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])


def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())


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


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
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


def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }


def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params


def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params


def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW


def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype


def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)


def inverse_video(pipe, latents, num_steps):
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)

    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent


def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        pretrained_model_path,
        noise_prior,
        cached_latent_dir=None,
):
    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        vae=vae,
        unet=copy.deepcopy(unet).to('cuda', dtype=torch.float16)
    )
    pipe.text_encoder.to('cuda', dtype=torch.float16)

    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None
    )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path = f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float16)
            batch['latents'] = tensor_to_vae_latent(pixel_values, vae)
            if noise_prior > 0.:
                batch['inversion_noise'] = inverse_video(pipe, batch['latents'], 50)
            for k, v in batch.items(): batch[k] = v[0]

            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )


def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break

                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params += 1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents


def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents


def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
            alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def should_sample(global_step, validation_steps, validation_data):
    return global_step % validation_steps == 0 and validation_data.sample_preview


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        output_dir,
        lora_manager_spatial: LoraHandler,
        lora_manager_temporal: LoraHandler,
        unet_target_replace_module=None,
        text_target_replace_module=None,
        is_checkpoint=False,
        save_pretrained_model=True
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet.cpu(), keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder.cpu(), keep_fp32_wrapper=False))

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)

    lora_manager_spatial.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/spatial', step=global_step)
    lora_manager_temporal.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/temporal', step=global_step)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        extra_train_data: list = [],
        dataset_types: Tuple[str] = ('json'),
        validation_steps: int = 100,
        trainable_modules: Tuple[str] = None,  # Eg: ("attn1", "attn2")
        extra_unet_params=None,
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 5e-5,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        text_encoder_gradient_checkpointing: bool = False,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        resume_step: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        enable_torch_2_attn: bool = False,
        seed: Optional[int] = None,
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        offset_noise_strength: float = 0.1,
        extend_dataset: bool = False,
        cache_latents: bool = False,
        cached_latent_dir=None,
        use_unet_lora: bool = False,
        unet_lora_modules: Tuple[str] = [],
        text_encoder_lora_modules: Tuple[str] = [],
        save_pretrained_model: bool = True,
        lora_rank: int = 16,
        lora_path: str = '',
        lora_unet_dropout: float = 0.1,
        logger_type: str = 'tensorboard',
        **kwargs
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # Handle the output folder creation
    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Get the training dataset based on types (json, single_video, image)
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    # If you have extra train data, you can add a list of however many you would like.
    # Eg: extra_train_data: [{: {dataset_types, train_data: {etc...}}}]
    try:
        if extra_train_data is not None and len(extra_train_data) > 0:
            for dataset in extra_train_data:
                d_t, t_d = dataset['dataset_types'], dataset['train_data']
                train_datasets += get_train_dataset(d_t, t_d, tokenizer)

    except Exception as e:
        print(f"Could not process extra train datasets due to an error : {e}")

    # Extend datasets that are less than the greatest one. This allows for more balanced training.
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    # Process one dataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]

    # Process many datasets
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    # Use LoRA if enabled.
    # one temporal lora
    lora_manager_temporal = LoraHandler(use_unet_lora=use_unet_lora, unet_replace_modules=["TransformerTemporalModel"])

    unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
        use_unet_lora, unet, lora_manager_temporal.unet_replace_modules, lora_unet_dropout,
        lora_path + '/temporal/lora/', r=lora_rank)

    optimizer_temporal = optimizer_cls(
        create_optimizer_params([param_optim(unet_lora_params_temporal, use_unet_lora, is_lora=True,
                                             extra_params={**{"lr": learning_rate}, **extra_text_encoder_params}
                                             )], learning_rate),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    lr_scheduler_temporal = get_scheduler(
        lr_scheduler,
        optimizer=optimizer_temporal,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # one spatial lora for each video
    spatial_lora_num = train_dataset.__len__()
    # if 'folder' in dataset_types:
    #     spatial_lora_num = train_dataset.__len__()
    # else:
    #     spatial_lora_num = 1

    lora_manager_spatials = []
    unet_lora_params_spatial_list = []
    optimizer_spatial_list = []
    lr_scheduler_spatial_list = []
    for i in range(spatial_lora_num):
        lora_manager_spatial = LoraHandler(use_unet_lora=use_unet_lora, unet_replace_modules=["Transformer2DModel"])
        lora_manager_spatials.append(lora_manager_spatial)
        unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
            use_unet_lora, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
            lora_path + '/spatial/lora/', r=lora_rank)

        unet_lora_params_spatial_list.append(unet_lora_params_spatial)

        optimizer_spatial = optimizer_cls(
            create_optimizer_params([param_optim(unet_lora_params_spatial, use_unet_lora, is_lora=True,
                                                 extra_params={**{"lr": learning_rate}, **extra_text_encoder_params}
                                                 )], learning_rate),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        optimizer_spatial_list.append(optimizer_spatial)

        # Scheduler
        lr_scheduler_spatial = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_spatial,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        lr_scheduler_spatial_list.append(lr_scheduler_spatial)

        unet_negation_all = unet_negation_spatial + unet_negation_temporal

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    # Latents caching
    cached_data_loader = handle_cache_latents(
        cache_latents,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        pretrained_model_path,
        validation_data.noise_prior,
        cached_latent_dir,
    )

    if cached_data_loader is not None:
        train_dataloader = cached_data_loader

    # Prepare everything with our `accelerator`.
    unet, optimizer_spatial_list, optimizer_temporal, train_dataloader, lr_scheduler_spatial_list, lr_scheduler_temporal, text_encoder = accelerator.prepare(
        unet,
        optimizer_spatial_list, optimizer_temporal,
        train_dataloader,
        lr_scheduler_spatial_list, lr_scheduler_temporal,
        text_encoder
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet,
        text_encoder,
        gradient_checkpointing,
        text_encoder_gradient_checkpointing
    )

    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # Fix noise schedules to predcit light and dark areas if available.
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def finetune_unet(batch, step, mask_spatial_lora=False, mask_temporal_lora=False):
        nonlocal use_offset_noise
        nonlocal rescale_schedule

        # Unfreeze UNET Layers
        if global_step == 0:
            already_printed_trainables = False
            unet.train()
            handle_trainable_modules(
                unet,
                trainable_modules,
                is_enabled=True,
                negation=unet_negation_all
            )

        # Convert videos to latent space
        if not cache_latents:
            latents = tensor_to_vae_latent(batch["pixel_values"], vae)
        else:
            latents = batch["latents"]

        # Sample noise that we'll add to the latents
        use_offset_noise = use_offset_noise and not rescale_schedule
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        # Sample a random timestep for each video
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # *Potentially* Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if kwargs.get('eval_train', False):
            unet.eval()
            text_encoder.eval()

        # Encode text embeddings
        token_ids = batch['prompt_ids']
        encoder_hidden_states = text_encoder(token_ids)[0]
        detached_encoder_state = encoder_hidden_states.clone().detach()

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise

        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        encoder_hidden_states = detached_encoder_state

        if mask_spatial_lora:
            loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
            for lora_i in loras:
                lora_i.scale = 0.
            loss_spatial = None
        else:
            loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
            for lora_i in loras:
                lora_i.scale = 0.

            for lora_idx in range(0, len(loras), spatial_lora_num):
                loras[lora_idx + step].scale = 1.

            loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
            for lora_i in loras:
                lora_i.scale = 0.

            ran_idx = torch.randint(0, noisy_latents.shape[2], (1,)).item()

            if random.uniform(0, 1) < -0.5:
                pixel_values_spatial = transforms.functional.hflip(
                    batch["pixel_values"][:, ran_idx, :, :, :]).unsqueeze(1)
                latents_spatial = tensor_to_vae_latent(pixel_values_spatial, vae)
                noise_spatial = sample_noise(latents_spatial, offset_noise_strength, use_offset_noise)
                noisy_latents_input = noise_scheduler.add_noise(latents_spatial, noise_spatial, timesteps)
                target_spatial = noise_spatial
                model_pred_spatial = unet(noisy_latents_input, timesteps,
                                          encoder_hidden_states=encoder_hidden_states).sample
                loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                          target_spatial[:, :, 0, :, :].float(), reduction="mean")
            else:
                noisy_latents_input = noisy_latents[:, :, ran_idx, :, :]
                target_spatial = target[:, :, ran_idx, :, :]
                model_pred_spatial = unet(noisy_latents_input.unsqueeze(2), timesteps,
                                          encoder_hidden_states=encoder_hidden_states).sample
                loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                          target_spatial.float(), reduction="mean")

        if mask_temporal_lora:
            loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
            for lora_i in loras:
                lora_i.scale = 0.
            loss_temporal = None
        else:
            loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
            for lora_i in loras:
                lora_i.scale = 1.
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            beta = 1
            alpha = (beta ** 2 + 1) ** 0.5
            ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()
            model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
            target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2)
            loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
            loss_temporal = loss_temporal + loss_ad_temporal

        return loss_spatial, loss_temporal, latents, noise

    for epoch in range(first_epoch, num_train_epochs):
        train_loss_spatial = 0.0
        train_loss_temporal = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):

                text_prompt = batch['text_prompt'][0]

                for optimizer_spatial in optimizer_spatial_list:
                    optimizer_spatial.zero_grad(set_to_none=True)

                optimizer_temporal.zero_grad(set_to_none=True)

                mask_temporal_lora = False
                # mask_spatial_lora = False
                mask_spatial_lora = random.uniform(0, 1) < 0.2 and not mask_temporal_lora

                with accelerator.autocast():
                    loss_spatial, loss_temporal, latents, init_noise = finetune_unet(batch, step, mask_spatial_lora=mask_spatial_lora, mask_temporal_lora=mask_temporal_lora)

                # Gather the losses across all processes for logging (if we use distributed training).
                if not mask_spatial_lora:
                    avg_loss_spatial = accelerator.gather(loss_spatial.repeat(train_batch_size)).mean()
                    train_loss_spatial += avg_loss_spatial.item() / gradient_accumulation_steps

                if not mask_temporal_lora:
                    avg_loss_temporal = accelerator.gather(loss_temporal.repeat(train_batch_size)).mean()
                    train_loss_temporal += avg_loss_temporal.item() / gradient_accumulation_steps

                # Backpropagate
                if not mask_spatial_lora:
                    accelerator.backward(loss_spatial, retain_graph = True)
                    optimizer_spatial_list[step].step()

                if not mask_temporal_lora:
                    accelerator.backward(loss_temporal)
                    optimizer_temporal.step()

                lr_scheduler_spatial_list[step].step()
                lr_scheduler_temporal.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss_temporal}, step=global_step)
                train_loss_temporal = 0.0
                if global_step % checkpointing_steps == 0 and global_step > 0:
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        accelerator,
                        unet,
                        text_encoder,
                        vae,
                        output_dir,
                        lora_manager_spatial,
                        lora_manager_temporal,
                        unet_lora_modules,
                        text_encoder_lora_modules,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            unet.eval()
                            text_encoder.eval()
                            unet_and_text_g_c(unet, text_encoder, False, False)
                            loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
                            for lora_i in loras:
                                lora_i.scale = validation_data.spatial_scale

                            if validation_data.noise_prior > 0:
                                preset_noise = (validation_data.noise_prior) ** 0.5 * batch['inversion_noise'] + (
                                    1-validation_data.noise_prior) ** 0.5 * torch.randn_like(batch['inversion_noise'])
                            else:
                                preset_noise = None

                            pipeline = TextToVideoSDPipeline.from_pretrained(
                                pretrained_model_path,
                                text_encoder=text_encoder,
                                vae=vae,
                                unet=unet
                            )

                            diffusion_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            prompt_list = text_prompt if len(validation_data.prompt) <= 0 else validation_data.prompt
                            for prompt in prompt_list:
                                save_filename = f"{global_step}_{prompt.replace('.', '')}"

                                out_file = f"{output_dir}/samples/{save_filename}.mp4"

                                with torch.no_grad():
                                    video_frames = pipeline(
                                        prompt,
                                        width=validation_data.width,
                                        height=validation_data.height,
                                        num_frames=validation_data.num_frames,
                                        num_inference_steps=validation_data.num_inference_steps,
                                        guidance_scale=validation_data.guidance_scale,
                                        latents=preset_noise
                                    ).frames
                                export_to_video(video_frames, out_file, train_data.get('fps', 8))
                                logger.info(f"Saved a new sample to {out_file}")
                            del pipeline
                            torch.cuda.empty_cache()

                    unet_and_text_g_c(
                        unet,
                        text_encoder,
                        gradient_checkpointing,
                        text_encoder_gradient_checkpointing
                    )

            accelerator.log({"loss_temporal": loss_temporal.detach().item()}, step=step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
            pretrained_model_path,
            global_step,
            accelerator,
            unet,
            text_encoder,
            vae,
            output_dir,
            lora_manager_spatial,
            lora_manager_temporal,
            unet_lora_modules,
            text_encoder_lora_modules,
            is_checkpoint=False,
            save_pretrained_model=save_pretrained_model
        )
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config_multi_videos.yaml')
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))

