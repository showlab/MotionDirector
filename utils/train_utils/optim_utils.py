from typing import List, Optional
import torch
import itertools

def get_optimizer(use_8bit_adam: bool):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def create_optim_params(name: str = 'param', params=None, lr: float = 5e-6, extra_params: Optional[dict] = None):
    params_dict = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params_dict[k] = v
    return params_dict

def param_optim(model, condition: bool, extra_params: Optional[dict] = None, is_lora: bool = False, negation: Optional[List] = None):
    extra_params = extra_params if extra_params and len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def create_optimizer_params(model_list: List[dict], lr: float):
    optimizer_params = []
    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(params=itertools.chain(*model), extra_params=extra_params)
            optimizer_params.append(params)
            continue
        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate:
                    continue
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)
    return optimizer_params