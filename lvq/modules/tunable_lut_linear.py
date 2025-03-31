import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .diff_index import DifferentiableIndex


class TunableLutLinear(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        args,
        num_lut: int,
        lut_size: int,
        vec_size: int,
        bias: bool = True,
        device: torch.device | str = None,
        dtype: torch.dtype | str = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert in_features % vec_size == 0
        
        self.weight = DifferentiableIndex(
            args,
            torch.Size((num_lut, out_features, in_features // vec_size)),
            N=1,
            M=lut_size,
            hard=False,
            device=device,
            dtype=dtype,
        )
        self.luts = Parameter(torch.empty(num_lut, 1, in_features // vec_size, lut_size, vec_size, **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
            
    
    def sample_weight(self) -> torch.Tensor:
        # [N_LUT, N, M // 4, 1, 16] @ [N_LUT, 1, M // 4, 16, 4]
        #  -> [N_LUT, N, M // 4, 1, 4] 
        #  -> [N, M // 4, 1, 4] 
        #  -> [N, M]
        weighted_index: torch.Tensor = self.weight().unsqueeze(-2) 
        weight = (weighted_index @ self.luts)\
            .sum(dim=0)\
            .reshape(self.out_features, self.in_features)
        return weight

    def forward(self, x):
        weight = self.sample_weight()
        return F.linear(x, weight, self.bias)
