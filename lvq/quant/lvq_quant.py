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
from lvq.modules import LvqLinear

def RTN(weight: torch.Tensor, group_size: int):
    max_int = 3
    min_int = -4
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    weight = weight.reshape(out_features, in_features // group_size, group_size)
    scales = weight.abs().max(dim=-1, keepdim=True).values / max_int

    qweight = torch.clamp(torch.round(weight / scales), min_int, max_int)
    new_weight = qweight * scales
    return new_weight.reshape(out_features, in_features)

@torch.no_grad()
def lvq_quant(
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
        
        for name, linear in named_linears.items():
            logging.info("Quant Linear[{}]".format(name))
            train_iters = args.train_iters
            if model_utils.is_ffn_linear(model, name):
                train_iters = int(train_iters / 4)
            hessian = input_feat[name].type_as(linear.weight)
            quant_results, new_weight = \
                quant_utils.reconstruct_weight(
                    args,
                    linear.weight,
                    hessian,
                    args.num_lut,
                    args.lut_size,
                    args.vec_size,
                    args.group_size,
                    train_iters,
                    return_weight=False
                )
            # rtn_dw = linear.weight - RTN(linear.weight, args.group_size)
            # lvq_dw = linear.weight - new_weight
            # print("rtn mean: {}, max: {}".format(rtn_dw.abs().mean(), rtn_dw.abs().max()))
            # print("lvq mean: {}, max: {}".format(lvq_dw.abs().mean(), lvq_dw.abs().max()))
            quant_results.update({
                "weight": linear.weight,
            })
            if linear.bias is not None:
                quant_results.update({"bias": linear.bias})
            lvq_linear = LvqLinear(
                linear.in_features,
                linear.out_features,
                args.num_lut,
                args.lut_size,
                args.vec_size,
                args.group_size,
                bias=True if linear.bias is not None else False,
                dtype=linear.weight.dtype,
                device=linear.weight.device,
            )
            lvq_linear.load_state_dict(quant_results)
            model_utils.replace_module(layer, name, lvq_linear)

            del hessian
            linear.to("cpu")
            lvq_linear.to("cpu")
            torch.cuda.empty_cache()

        del input_feat
        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return lvq_results
