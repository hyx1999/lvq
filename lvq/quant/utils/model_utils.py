import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaForCausalLM

def check_model(model: PreTrainedModel):
    if isinstance(model, LlamaForCausalLM):
        return
    raise ValueError


def move_embed(model: PreTrainedModel, dev: str):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(dev)
        return
    raise ValueError


def get_layers(model: PreTrainedModel):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    raise ValueError


def get_layers_prefix(model: PreTrainedModel):
    if isinstance(model, LlamaForCausalLM):
        return "model.layers"
    raise ValueError


def get_named_linears(layer: nn.Module):
    named_linears = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            named_linears[name] = module
    return named_linears

def replace_module(layer: nn.Module, name: str, new_module: nn.Module):
    module_name = name.split(".")[-1]
    parent_name = ".".join(name.split(".")[:-1])
    parent = layer.get_submodule(parent_name)
    setattr(parent, module_name, new_module)

def is_ffn_linear(model: PreTrainedModel, name: str):
    if isinstance(model, LlamaForCausalLM):
        return any(n in name for n in ['gate_proj', 'up_proj', 'down_proj'])
    raise ValueError
