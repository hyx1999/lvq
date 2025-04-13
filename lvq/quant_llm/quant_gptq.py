import math
import logging
import gc
import functools
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from collections import defaultdict
from typing import Dict, List, Tuple
from .utils import model_utils
from lvq.quant_llm.quantize import (
    quant_gptq,
)
from lvq.modules import QLinear
from torch.optim.adamw import AdamW


@torch.no_grad()
def gptq(
    args,
    model: PreTrainedModel,
    dataloader: List[Tuple[torch.Tensor, ...]],
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    logging.info("Starting ...")
    dev = args.device
    
    model_utils.check_model(model)
    layers = model_utils.get_layers(model)

    model_utils.move_embed(model, dev)
    layers[0] = layers[0].to(dev)

    inps = []
    layer_kwargs = {}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(**kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])

    samples = torch.cat([sample[0] for sample in dataloader], dim=0)
    try:
        model(samples.to(dev))
    except ValueError:
        pass
    del samples
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    inps = inps[0]

    model_utils.move_embed(model, "cpu")
    layers[0] = layers[0].cpu()
    
    lvq_results = {}
        
    for i in range(len(layers)):
        logging.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        
        sequential: List[Tuple[List[str], List[nn.Linear]]] = [
            model_utils.get_qkv_linears(model, layer),
            model_utils.get_o_linears(model, layer),
            model_utils.get_gate_up_linears(model, layer),
            model_utils.get_down_linears(model, layer),
        ]
        def cache_input_hook(m, x, y, name, hessian_dict, count_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, x.shape[-1]).float()
            num_new = x.shape[0]
            num_old = count_dict[name]
            x = (x * math.sqrt(2 / (num_old + num_new)))
            count_dict[name] = num_old + num_new
            if hessian_dict[name] is None:
                hessian_dict[name] = x.T @ x
            else:
                H = hessian_dict[name]
                hessian_dict[name] = H * (num_old / (num_old + num_new)) + x.T @ x

        input_hessian = defaultdict(lambda: None)
        input_count = defaultdict(lambda: 0)
        handles = []
        for names, modules in sequential:
            handles.append(
                modules[0].register_forward_hook(
                    functools.partial(cache_input_hook, name=names[0], hessian_dict=input_hessian, count_dict=input_count)
                )
            )
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        
        for names, modules in sequential:
            logging.info("Quant Linear[{}]".format(", ".join(names)))
            hessian = input_hessian[names[0]].type_as(modules[0].weight)
            for name, linear in zip(names, modules):
                quant_results, new_weight = \
                    quant_gptq.quant_weight_gptq(
                        args,
                        linear.weight,
                        hessian,
                        args.w_bits,
                        args.group_size,
                        return_weight=True
                    )
                if linear.bias is not None:
                    quant_results.update({"bias": linear.bias})
                qlinear = QLinear(
                    linear.in_features,
                    linear.out_features,
                    num_bits=args.w_bits,
                    group_size=args.group_size,
                    bias=True if linear.bias is not None else False,
                    dtype=linear.weight.dtype,
                    device=linear.weight.device,
                )
                qlinear.load_state_dict(quant_results)
                model_utils.replace_module(layer, name, qlinear)
        
        

        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return lvq_results
