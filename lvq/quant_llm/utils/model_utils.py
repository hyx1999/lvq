import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaForCausalLM, Qwen2ForCausalLM

def check_model(model: PreTrainedModel):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return
    raise ValueError


def move_embed(model: PreTrainedModel, dev: str):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        model.model.embed_tokens.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(dev)
        return
    raise ValueError


def get_layers(model: PreTrainedModel):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return model.model.layers
    raise ValueError


def get_layers_prefix(model: PreTrainedModel):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return "model.layers"
    raise ValueError


def get_named_linears(layer: nn.Module):
    named_linears = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            named_linears[name] = module
    return named_linears


def get_qkv_linears(model: PreTrainedModel, layer: nn.Module):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        modules = [layer.get_submodule(n) for n in names]
        return names, modules
    raise ValueError


def get_o_linears(model: PreTrainedModel, layer: nn.Module):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        names = ["self_attn.o_proj"]
        modules = [layer.get_submodule(n) for n in names]
        return names, modules
    raise ValueError


def get_gate_up_linears(model: PreTrainedModel, layer: nn.Module):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        names = ["mlp.gate_proj", "mlp.up_proj"]
        modules = [layer.get_submodule(n) for n in names]
        return names, modules
    raise ValueError


def get_down_linears(model: PreTrainedModel, layer: nn.Module):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        names = ["mlp.down_proj"]
        modules = [layer.get_submodule(n) for n in names]
        return names, modules
    raise ValueError


def get_qk_scaler(model: PreTrainedModel, layer: nn.Module):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return layer.self_attn.qk_scaler
    raise ValueError


def get_self_attn(model: PreTrainedModel, layer: nn.Module):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return layer.self_attn
    raise ValueError


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def replace_module(layer: nn.Module, name: str, new_module: nn.Module):
    module_name = name.split(".")[-1]
    parent_name = ".".join(name.split(".")[:-1])
    parent = layer.get_submodule(parent_name)
    setattr(parent, module_name, new_module)


def is_ffn_linear(model: PreTrainedModel, name: str):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return any(n in name for n in ['gate_proj', 'up_proj', 'down_proj'])
    raise ValueError


def get_lm_head(model):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type')


def get_embeddings(model) -> list[torch.nn.Module]:
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type')


def get_pre_head_layernorm(model):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        pre_head_layernorm = model.model.norm
    else:
        raise ValueError(f'Unknown model type')
    return pre_head_layernorm


def untie_word_embedding(model):
    embeddings = get_embeddings(model)
    if model.config.tie_word_embeddings:    
        for emb in embeddings:
            emb: torch.nn.Embedding = emb
            emb.weight = torch.nn.Parameter(
                emb.weight.data.clone(),
                requires_grad=emb.weight.requires_grad,
            )
