import torch
import torch.nn as nn


def repeat_scales(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QKScaler(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)        
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.scales = nn.Parameter(torch.ones(self.num_key_value_heads, self.head_dim))
    
    def forward(self, 
        query_stats: torch.Tensor,   # [B, H, L, D]
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        scales = self.scales.view(1, self.num_key_value_heads, 1, self.head_dim)
        q_scales = repeat_scales(scales, self.num_key_value_groups)
        k_scales = repeat_scales(scales, 1)
        query_stats = query_stats / q_scales
        key_states = key_states * k_scales
        return query_stats, key_states
