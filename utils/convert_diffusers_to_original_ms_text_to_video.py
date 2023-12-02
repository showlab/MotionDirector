# Script for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
# *Only* converts the UNet, and Text Encoder.
# Does not convert optimizer state or any other thing.

import argparse
import os.path as osp
import re

import torch
from safetensors.torch import load_file, save_file

# =================#
# UNet Conversion #
# =================#

print ('Initializing the conversion map')

unet_conversion_map = [
    # (ModelScope, HF Diffusers)

    # from Vanilla ModelScope/StableDiffusion
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),


    # from Vanilla ModelScope/StableDiffusion
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),


    # from Vanilla ModelScope/StableDiffusion
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (ModelScope, HF Diffusers)

    # SD
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),

    # MS
    #("temopral_conv", "temp_convs"), # ROFL, they have a typo here --kabachuha
]

unet_conversion_map_layer = []

# Convert input TemporalTransformer
unet_conversion_map_layer.append(('input_blocks.0.1', 'transformer_in'))

# Reference for the default settings

# "model_cfg": {
#     "unet_in_dim": 4,
#     "unet_dim": 320,
#     "unet_y_dim": 768,
#     "unet_context_dim": 1024,
#     "unet_out_dim": 4,
#     "unet_dim_mult": [1, 2, 4, 4],
#     "unet_num_heads": 8,
#     "unet_head_dim": 64,
#     "unet_res_blocks": 2,
#     "unet_attn_scales": [1, 0.5, 0.25],
#     "unet_dropout": 0.1,
#     "temporal_attention": "True",
#     "num_timesteps": 1000,
#     "mean_type": "eps",
#     "var_type": "fixed_small",
#     "loss_type": "mse"
# }

# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks

        # Spacial SD stuff
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))
        
        # Temporal MS stuff
        hf_down_res_prefix = f"down_blocks.{i}.temp_convs.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0.temopral_conv."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.temp_attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.2."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks

        # Spacial SD stuff
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))
        
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.temp_convs.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0.temopral_conv."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.temp_attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.2."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    # Up/Downsamplers are 2D, so don't need to touch them
    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 3}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))


# Handle the middle block

# Spacial
hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{3*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

# Temporal
hf_mid_atn_prefix = "mid_block.temp_attentions.0."
sd_mid_atn_prefix = "middle_block.2."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.temp_convs.{j}."
    sd_mid_res_prefix = f"middle_block.{3*j}.temopral_conv."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

# The pipeline
def convert_unet_state_dict(unet_state_dict, strict_mapping=False):
    print ('Converting the UNET')
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}

    for sd_name, hf_name in unet_conversion_map:
        if strict_mapping:
            if hf_name in mapping:
                mapping[hf_name] = sd_name
        else:
            mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
        # elif "temp_convs" in k:
        #     for sd_part, hf_part in unet_conversion_map_resnet:
        #         v = v.replace(hf_part, sd_part)
        #     mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    

    # there must be a pattern, but I don't want to bother atm
    do_not_unsqueeze = [f'output_blocks.{i}.1.proj_out.weight' for i in range(3, 12)] + [f'output_blocks.{i}.1.proj_in.weight' for i in range(3, 12)] + ['middle_block.1.proj_in.weight', 'middle_block.1.proj_out.weight'] + [f'input_blocks.{i}.1.proj_out.weight' for i in [1, 2, 4, 5, 7, 8]] + [f'input_blocks.{i}.1.proj_in.weight' for i in [1, 2, 4, 5, 7, 8]]
    print (do_not_unsqueeze)

    new_state_dict = {v: (unet_state_dict[k].unsqueeze(-1) if ('proj_' in k and ('bias' not in k) and (k not in do_not_unsqueeze)) else unet_state_dict[k]) for k, v in mapping.items()}
    # HACK: idk why the hell it does not work with list comprehension
    for k, v in new_state_dict.items():
        has_k = False
        for n in do_not_unsqueeze:
            if k == n:
                has_k = True

        if has_k:
            v = v.squeeze(-1)
        new_state_dict[k] = v

    return new_state_dict

# TODO: VAE conversion. We doesn't train it in the most cases, but may be handy for the future --kabachuha

# =========================#
# Text Encoder Conversion #
# =========================#

# IT IS THE SAME CLIP ENCODER, SO JUST COPYPASTING IT --kabachuha

# =========================#
# Text Encoder Conversion #
# =========================#


textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}


def convert_text_enc_state_dict_v20(text_enc_dict):
    #print ('Converting the text encoder')
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if (
            k.endswith(".self_attn.q_proj.weight")
            or k.endswith(".self_attn.k_proj.weight")
            or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
            k.endswith(".self_attn.q_proj.bias")
            or k.endswith(".self_attn.k_proj.bias")
            or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict

textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}


def convert_text_enc_state_dict_v20(text_enc_dict):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if (
            k.endswith(".self_attn.q_proj.weight")
            or k.endswith(".self_attn.k_proj.weight")
            or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
            k.endswith(".self_attn.q_proj.bias")
            or k.endswith(".self_attn.k_proj.bias")
            or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--clip_checkpoint_path", default=None, type=str, help="Path to the output CLIP model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument(
        "--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt."
    )

    args = parser.parse_args()

    assert args.model_path is not None, "Must provide a model path!"

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"

    assert args.clip_checkpoint_path is not None, "Must provide a CLIP checkpoint path!"

    # Path for safetensors
    unet_path = osp.join(args.model_path, "unet", "diffusion_pytorch_model.safetensors")
    #vae_path = osp.join(args.model_path, "vae", "diffusion_pytorch_model.safetensors")
    text_enc_path = osp.join(args.model_path, "text_encoder", "model.safetensors")

    # Load models from safetensors if it exists, if it doesn't pytorch
    if osp.exists(unet_path):
        unet_state_dict = load_file(unet_path, device="cpu")
    else:
        unet_path = osp.join(args.model_path, "unet", "diffusion_pytorch_model.bin")
        unet_state_dict = torch.load(unet_path, map_location="cpu")

    # if osp.exists(vae_path):
    #     vae_state_dict = load_file(vae_path, device="cpu")
    # else:
    #     vae_path = osp.join(args.model_path, "vae", "diffusion_pytorch_model.bin")
    #     vae_state_dict = torch.load(vae_path, map_location="cpu")

    if osp.exists(text_enc_path):
        text_enc_dict = load_file(text_enc_path, device="cpu")
    else:
        text_enc_path = osp.join(args.model_path, "text_encoder", "pytorch_model.bin")
        text_enc_dict = torch.load(text_enc_path, map_location="cpu")

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    #unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Convert the VAE model
    # vae_state_dict = convert_vae_state_dict(vae_state_dict)
    # vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Easiest way to identify v2.0 model seems to be that the text encoder (OpenCLIP) is deeper
    is_v20_model = "text_model.encoder.layers.22.layer_norm2.bias" in text_enc_dict

    if is_v20_model:

        # MODELSCOPE always uses the 2.X encoder, btw --kabachuha

        # Need to add the tag 'transformer' in advance so we can knock it out from the final layer-norm
        text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()}
        text_enc_dict = convert_text_enc_state_dict_v20(text_enc_dict)
        #text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
    else:
        text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
        #text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}

    # DON'T PUT TOGETHER FOR THE NEW CHECKPOINT AS MODELSCOPE USES THEM IN THE SPLITTED FORM --kabachuha
    # Save CLIP and the Diffusion model to their own files

    #state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
    print ('Saving UNET')
    state_dict = {**unet_state_dict}

    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    if args.use_safetensors:
        save_file(state_dict, args.checkpoint_path)
    else:
        #state_dict = {"state_dict": state_dict}
        torch.save(state_dict, args.checkpoint_path)

    # TODO: CLIP conversion doesn't work atm
    # print ('Saving CLIP')
    # state_dict = {**text_enc_dict}

    # if args.half:
    #     state_dict = {k: v.half() for k, v in state_dict.items()}

    # if args.use_safetensors:
    #     save_file(state_dict, args.checkpoint_path)
    # else:
    #     #state_dict = {"state_dict": state_dict}
    #     torch.save(state_dict, args.clip_checkpoint_path)
    
    print('Operation successfull')
