from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from .llama.attention import (
    register_attn_modules as llama_register_attn_modules
)
from .qwen2.attention import (
    register_attn_modules as qwen2_register_attn_modules
)

# from .qwen2.attention import 

__all__ = ['register_attn_scale']

def register_attn_modules(args, model):
    if isinstance(model, LlamaForCausalLM):
        llama_register_attn_modules(args, model)
    elif isinstance(model, Qwen2ForCausalLM):
        qwen2_register_attn_modules(args, model)
    else:
        raise ValueError
