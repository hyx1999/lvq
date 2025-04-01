import logging
import gc
import functools
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from collections import defaultdict
from typing import Dict, List, Tuple
from .utils import (
    model_utils,
    quant_utils,
)

@torch.no_grad()
def lvq_quant(
    args,
    model: PreTrainedModel,
    dataloader: List[Tuple[torch.Tensor, ...]],
):
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
        
        named_linears: Dict[str, nn.Linear] = model_utils.get_named_linears(layer)
        
        def cache_input_hook(m, x, y, name, feat_dict):
            x: torch.Tensor = x[0]
            x = x.view(-1, x.shape[-1])
            hessian = (x.T @ x).cpu().to(torch.float64) / x.shape[0]
            feat_dict[name] = hessian
        
        input_feat = defaultdict(lambda: None)
        handles = []
        for name in named_linears.keys():
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(
                        cache_input_hook, 
                        name=name, 
                        feat_dict=input_feat, 
                    )
                )
            )
        
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        
        for name in named_linears.keys():
            logging.info("Quant Linear[{}]".format(name))
            hessian = input_feat[name].type_as(named_linears[name].weight)
            quant_results, quant_weight = \
                quant_utils.reconstruct_weight(
                    args,
                    named_linears[name].weight,
                    hessian,
                    args.num_lut,
                    args.lut_size,
                    args.vec_size,
                    args.group_size,
                    args.train_iters,
                    return_weight=True
                )
            named_linears[name].weight.data.copy_(quant_weight)
            # prefix = "{}.{}".format(model_utils.get_layers_prefix(model), i)
            # for key, value in quant_results.items():
            #     lvq_results[f"{prefix}.{key}"] = value
        print(end="\n")

        del input_feat
        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        # !!!!!!!!!!!!!!!!
        break

    return lvq_results
