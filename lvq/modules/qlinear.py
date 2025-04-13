import torch
import torch.nn as nn
import torch.nn.functional as F

class QLinear(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        group_size: int,
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
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self.scales = nn.Parameter(torch.empty(out_features, in_features // group_size, **factory_kwargs))
        self.zero_points = nn.Parameter(torch.empty(out_features, in_features // group_size, **factory_kwargs))
    
    def set_params(self, idx: int, scale: torch.Tensor, zero_point: torch.Tensor):
        self.scales[:, idx].copy_(scale)
        self.zero_points[:, idx].copy_(zero_point)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
