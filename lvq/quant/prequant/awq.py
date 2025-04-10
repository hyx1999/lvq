import math
import logging
import gc
import functools
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from collections import defaultdict
from typing import Dict, List, Tuple
from lvq.quant.utils import model_utils
from lvq.quant.quantize import (
    auto_scale,
    auto_clip,
)

@torch.no_grad()
def prequant_awq(
    args,
    model: PreTrainedModel,
    dataloader: List[Tuple[torch.Tensor, ...]],
    max_seqlen: int = 512,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    logging.info("AWQ Starting ...")
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
    assert len(samples.shape) == 2
    if samples.shape[1] > max_seqlen:
        samples = samples[:, :max_seqlen]
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
        logging.info(f"=== Start prequant layer {i} ===")
        layer = layers[i].to(dev)
        
        sequential: List[Tuple[List[str], List[nn.Linear]]] = [
            model_utils.get_qkv_linears(model, layer),
            model_utils.get_o_linears(model, layer),
            model_utils.get_gate_up_linears(model, layer),
            model_utils.get_down_linears(model, layer),
        ]
        
        def cache_input_hook_awq(m, x, y, name, feat_dict):
            x: torch.Tensor = x[0]
            feat_dict[name].append(x.cpu())

        input_feat = defaultdict(list)
        handles = []
        for names, modules in sequential:
            handles.append(
                modules[0].register_forward_hook(
                    functools.partial(cache_input_hook_awq, name=names[0], feat_dict=input_feat)
                )
            )

        layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        for names, _ in sequential:
            for name in names[1:]:
                input_feat[name] = input_feat[names[0]]
        
        q_config = {
            "q_group_size": args.group_size,
            "zero_point": True,
        }
        
        logging.info("Auto scale...")
        scales_list = auto_scale.auto_scale_block(
            layer,
            layer_kwargs,
            w_bit=4,
            q_config=q_config,
            input_feat=input_feat,
        )
        auto_scale.apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

        logging.info("Auto clip...")        
        clip_list = auto_clip.auto_clip_block(
            layer,
            w_bit=4,
            q_config=q_config,
            input_feat=input_feat,
        )
        auto_clip.apply_clip(layer, clip_list)        

        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return lvq_results
