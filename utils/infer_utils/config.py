import argparse
import random
import os
import platform
import re
from typing import Optional, List, Tuple

def parse_args(is_multi: bool = False) -> argparse.Namespace:
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
    parser.add_argument("-g", "--guidance-scale", type=float, default=12, help="Scale for guidance loss")
    parser.add_argument("-f", "--fps", type=int, default=8, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda)")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attention")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention")
    parser.add_argument("-lr", "--lora_rank", type=int, default=32, help="Size of the LoRA checkpoint's projection matrix")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed for reproducible generations")
    parser.add_argument("-np", "--noise_prior", type=float, default=0., help="Scale of the influence of inversion noise")
    parser.add_argument("-ci", "--checkpoint_index", type=int, default=None, help="The index of checkpoint, e.g., 300")
    parser.add_argument("-rn", "--repeat_num", type=int, default=1, help="Number of results to generate with the same prompt")
    parser.add_argument("-npn", "--no-prompt-name", action="store_true", default=True, help="Use a generic filename instead of a prompt-based filename")
    
    if is_multi:
        parser.add_argument("-slp", "--spatial_path_folder", type=str, default=None, help="Path to spatial LoRA checkpoint")
        parser.add_argument("-tlp", "--temporal_path_folder", type=str, default=None, help="Path to temporal LoRA checkpoint")
        parser.add_argument("-sps", "--spatial_path_scale", type=float, default=1.0, help="Scale of spatial LoRAs")
        parser.add_argument("-tps", "--temporal_path_scale", type=float, default=1.0, help="Scale of temporal LoRAs")
    else:
        parser.add_argument("-cf", "--checkpoint_folder", type=str, required=True, help="Path to temporal LoRA checkpoint")
        parser.add_argument("-ls", "--lora_scale", type=float, default=1.0, help="Scale of LoRAs")
    
    return parser.parse_args()

def prepare_inputs(args: argparse.Namespace, is_multi: bool = False) -> Tuple[str, str, List[str], Optional[List[str]]]:
    out_name = f"{args.output_dir}/"
    prompt = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", args.prompt) if platform.system() == "Windows" else args.prompt
    out_name += f"{prompt}".replace(' ', '_').replace(',', '').replace('.', '')
    prompt_list = [prompt] * args.batch_size
    negative_prompt_list = [args.negative_prompt] * args.batch_size if args.negative_prompt else None
    
    if is_multi:
        assert os.path.exists(args.spatial_path_folder), f"Spatial LoRA path {args.spatial_path_folder} does not exist"
        assert os.path.exists(args.temporal_path_folder), f"Temporal LoRA path {args.temporal_path_folder} does not exist"
        lora_paths = {
            "spatial_lora_path": args.spatial_path_folder,
            "temporal_lora_path": args.temporal_path_folder
        }
    else:
        lora_path = f"{args.checkpoint_folder}/checkpoint-{args.checkpoint_index}/temporal/lora" if args.checkpoint_index else f"{args.checkpoint_folder}/temporal/lora"
        assert os.path.exists(lora_path), f"LoRA path {lora_path} does not exist"
        lora_paths = {"lora_path": lora_path}
    
    latents_path = None
    if args.noise_prior > 0:
        latents_folder = f"{args.checkpoint_folder if not is_multi else os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.temporal_path_folder))))}/cached_latents"
        latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        assert os.path.exists(latents_path), f"Latents path {latents_path} does not exist"
    
    return out_name, latents_path, prompt_list, negative_prompt_list, lora_paths
