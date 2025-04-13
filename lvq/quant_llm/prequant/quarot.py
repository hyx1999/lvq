import torch
import typing
import transformers
import tqdm, math
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2ForCausalLM
from lvq.quant_llm.utils import model_utils
from lvq.ops.hadmard import random_hadamard_matrix, is_pow2


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
    layernorm.weight.data.zero_().add_(1.0)
    if hasattr(layernorm, 'bias'):
        layernorm.bias.data.zero_()


def fuse_layer_norms(model):
    model_utils.untie_word_embedding(model)

    # Embedding fusion
    for W in model_utils.get_embeddings(model):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_layers(model)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        else:
            raise ValueError(f'Unknown model type')

    fuse_ln_linear(model_utils.get_pre_head_layernorm(model), [model_utils.get_lm_head(model)])
    

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=None):
    if mode == 'random':
        Q = random_orthogonal_matrix(size, device)
        return Q
    elif mode == 'hadamard':
        Q = random_hadamard_matrix(size, device)
        return Q
    else:
        raise ValueError(f'Unknown mode {mode}')


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_utils.get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=None, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    

def rotate_attention_inputs(model, layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=None, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(model, layer, Q) -> None:
    # Rotate output matrix of the self-attention layer.
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=None, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=None, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(model, layer, Q):
    # Rotate the MLP input weights.
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=None, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    
def rotate_mlp_output(model, layer, Q):
    # Rotate the MLP output weights and bias.
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=None, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=None, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=None, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


@torch.no_grad()
def prequant_quarot(args, model):
    fuse_layer_norms(model)

    Q = get_orthogonal_matrix(model.config.hidden_size, "random")
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    layers = model_utils.get_layers(model)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(model, layers[idx], Q)
        rotate_attention_output(model, layers[idx], Q)
        rotate_mlp_input(model, layers[idx], Q)
        rotate_mlp_output(model, layers[idx], Q)
