import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LutLinear(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        group_size: int,
        lora_rank: int,
        bias: bool = True,
        device: torch.device | str = None,
        dtype: torch.dtype | str = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        group_size = in_features if group_size == -1 else group_size

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_bits = num_bits
        self.min_int = 0
        self.max_int = ((2 ** num_bits) - 1)

        assert in_features % group_size == 0
        
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self.row_scale = None
        self.factor = Parameter(torch.empty(out_features, in_features // group_size, **factory_kwargs))
        self.col_scale = Parameter(torch.empty(in_features, **factory_kwargs))
        self.col_bias   = Parameter(torch.empty(in_features, **factory_kwargs))
        self.A = torch.empty(out_features, lora_rank, **factory_kwargs)
        self.B = torch.empty(lora_rank, in_features, **factory_kwargs)
    
    def update_row_params(self, idx: int, row_scale: torch.Tensor):
        self.row_scale[:, idx].copy_(row_scale)
    
    def update_col_params(self, idx: int, col_scale: torch.Tensor, col_bias: torch.Tensor):
        self.col_scale[idx].copy_(col_scale)
        self.col_bias[idx].copy_(col_bias)

    def quantize_weight(self) -> torch.Tensor:
        weight = (weight - self.col_bias[None]) / self.col_scale[None].clamp(min=1e-5)
        weight = self.weight.view(self.out_features, self.in_features // self.group_size, -1)
        if self.row_scale is None:
            row_scale = weight.amax(dim=-1) * torch.sigmoid(self.factor)
        else:
            row_scale: torch.Tensor = self.row_scale
        weight = weight / row_scale[..., None].clamp(min=1e-5)
        weight = weight.view(self.out_features, self.in_features)
        weight = torch.clamp(
            torch.round(weight * self.max_int),
            self.min_int, self.max_int
        ) / self.max_int
        weight = weight.view(self.out_features, self.in_features // self.group_size, -1)
        weight = weight * row_scale[..., None].clamp(min=1e-5)
        weight = weight.view(self.out_features, self.in_features)
        weight = weight * self.col_scale[None].clamp(min=1e-5) + self.col_bias[None]
        weight = weight + self.A @ self.B
        return weight

    def forward(self, x):
        weight = self.quantize_weight()
        return F.linear(x, weight, self.bias)
