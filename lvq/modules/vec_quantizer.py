import torch
import torch.nn as nn
from typing import Tuple, List


class VectorQuantizer(nn.Module):
    
    def __init__(self, 
        shape: Tuple[int, ...],
        scale: float,
        dtype=None,
        device=None
    ):
        super().__init__()
        self.codebook = nn.Parameter(torch.empty(shape, dtype=dtype, device=device))  # [code_group, num_code, code_dim]
        self.scale = nn.Buffer(torch.tensor(scale, dtype=dtype, device=device))
        self.code_group = shape[:-2]
        self.num_code = shape[-2]
        self.code_dim = shape[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [code_group, num_features, code_dim]
        codebook = self.codebook * self.scale
        dist = (
            x.pow(2).sum(dim=-1).unsqueeze(-1)
            + codebook.pow(2).sum(dim=-1).unsqueeze(-2)
            - 2 * x @ codebook.transpose(-2, -1)
        ) # [code_group, num_features, code_dim] @ [code_group, num_code, code_dim].T => [code_group, num_features, num_code]
        indices = dist.argmin(dim=-1)
        x_q = self.embedding(codebook, indices)
        return x_q
    
    def embedding(self, codebook: torch.Tensor, indices: torch.Tensor):
        # codebook: [code_group, num_code, code_dim]
        # indices:  [code_group, num_features] => [code_group, num_features, code_dim]
        indices = indices.unsqueeze(-1).expand(*self.code_group, -1, self.code_dim)
        return torch.gather(codebook, dim=-2, index=indices)


class ResidualVectorQuantizer(nn.Module):
    
    def __init__(self,
        num_quantizer: int,
        shape: Tuple[int, ...],
        scales=None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_quantizers = num_quantizer
        if scales is None:
            scales = [1.0] * num_quantizer
        self.quantizers: List[VectorQuantizer] = nn.ModuleList(
            [VectorQuantizer(shape, scales[i], dtype=dtype, device=device) for i in range(num_quantizer)]
        )
    
    def forward(self, x: torch.Tensor):
        x_qs = []
        for i in range(self.num_quantizers):
            x_q = self.quantizers[i](x)
            x = x - x_q
            x_qs.append(x_q)
        x_q = torch.stack(x_qs).sum(dim=0)
        return x_q
