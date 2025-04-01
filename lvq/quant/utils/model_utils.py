import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaForCausalLM

def check_model(model: PreTrainedModel):
    if isinstance(model, LlamaForCausalLM):
        return
    raise ValueError

def get_layers(model: PreTrainedModel):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    raise ValueError

def replace_module():
    ...
