import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FakeLvqLinear(nn.Module):
    
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
        
        self.index = Parameter(
            torch.empty(num_lut, in_features // vec_size, out_features),
            dtype=torch.int8,
            device=device,
            requires_grad=False,
        )
        self.luts = Parameter(torch.empty(num_lut, in_features // vec_size, lut_size, vec_size, **factory_kwargs))
        self.scales = Parameter(torch.empty(out_features, in_features // group_size, **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def reconstruct_weight(self) -> torch.Tensor:
        weight = torch.gather(
            self.luts, dim=-2, 
            index=self.index.unsqueeze(-1).repeat(1, 1, 1, self.vec_size).int()
        ).transpose(1, 2).sum(dim=0).reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        weight = weight * self.scales.unsqueeze(-1)
        weight = weight.view(self.out_features, self.in_features)
        return weight

    def forward(self, x):
        weight = self.reconstruct_weight()
        return F.linear(x, weight, self.bias)
