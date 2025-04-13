import torch
import torch.nn as nn


def pseudo_quantize_tensor(
    x: torch.Tensor, n_bit=8, zero_point=False, inplace=False, get_scale_zp=False
):
    org_w_shape = x.shape
    assert x.dim() == 4
    if zero_point:
        max_val = x.amax(dim=-1, keepdim=True)
        min_val = x.amin(dim=-1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = x.abs().amax(dim=-1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(x).sum() == 0

    if inplace:
        (
            (x.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        x = (
            torch.clamp(torch.round(x / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(x).sum() == 0

    x = x.reshape(org_w_shape)

    if get_scale_zp:
        return x, scales.view(x.shape[0], -1), zeros.view(x.shape[0], -1)
    else:
        return x


def get_mean_key_states(key_states: torch.Tensor, group_size: int):
    B, H, L, D = key_states.shape
    assert L % group_size == 0
    mean_key_states = key_states.view(B, H, L // group_size, group_size, D)\
        .mean(dim=-2)\
        .repeat(1, 1, 1, group_size, 1)\
        .reshape(B, H, L, D)
    return mean_key_states


class PseudoKVQuantizer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.key_group_size = 32
    
    def forward(self, 
        key_states: torch.Tensor,    # [B, H, L, D]
        value_states: torch.Tensor,  # [B, H, L, D]
    ):
        if self.config.enable_kv_quant:
            if self.k_bits < 16:
                mean_key_states = get_mean_key_states(key_states, self.key_group_size)
                key_states = pseudo_quantize_tensor(
                    key_states - mean_key_states, n_bit=self.k_bits, zero_point=True
                ) + mean_key_states
            if self.v_bits < 16:
                value_states = pseudo_quantize_tensor(value_states, n_bit=self.v_bits, zero_point=True)
        return key_states, value_states
