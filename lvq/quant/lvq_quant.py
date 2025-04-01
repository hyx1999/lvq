import logging
import gc
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .utils import model_utils

@torch.no_grad()
def lvq_quant(
    args,
    lm: PreTrainedModel,
    dataloader: torch.Tensor,
    dev: str,
):
    logger = logging.getLogger(__name__)
    
    model_utils.check_model(model)

    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    layers = model_utils.get_layers(model)
    
    layers[0] = layers[0].to(dev)
    dtype = next(layers[0].parameters()).dtype

    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        
        outps = torch.zeros_like(inps)
        for j in range(args.nsamples):
            outps[j] = layer(inps[j].unsqueeze(0), attention_mask=cache["attention_mask"])[0]
        
                
        inps.copy_(outps)
        del layer
        del outps
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
