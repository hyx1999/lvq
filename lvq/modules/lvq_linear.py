import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .vec_quantizer import ResidualVectorQuantizer


class LvqLinear(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_lut: int,
        lut_size: int,
        vec_size: int,
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
        self.num_lut = num_lut
        self.lut_size = lut_size
        self.vec_size = vec_size
        self.group_size = group_size

        assert in_features % vec_size == 0
        assert in_features % group_size == 0
        
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.scales = Parameter(torch.empty(out_features, in_features // group_size, **factory_kwargs))
        self.quantizer = ResidualVectorQuantizer(
            num_lut,
            shape=(in_features // vec_size, lut_size, vec_size),
            dtype=dtype,
            device=device,
        )

    def quantize_weight(self) -> torch.Tensor:
        weight = self.weight.reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        weight = weight / self.scales.unsqueeze(-1)
        weight = weight.reshape(self.out_features, self.in_features // self.vec_size, self.vec_size).transpose(0, 1)
        q_weight = self.quantizer(weight)
        q_weight = q_weight.transpose(0, 1).reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        new_weight = q_weight * self.scales.unsqueeze(-1)
        return new_weight.reshape(self.out_features, self.in_features)

    def forward(self, x):
        weight = self.quantize_weight()
        return F.linear(x, weight, self.bias)
