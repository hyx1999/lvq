import torch
import torch.nn.functional as F
from lvq.quant.quantize import quant_vq

x = torch.randn(1, 4, 4).cuda()

print(x)

indices, code = quant_vq.find_params(x, n_centroids=4)

print(indices.shape)
print(indices)
print(code.shape)
print(code)

indices = indices.unsqueeze(-1).repeat(1, 1, x.shape[-1])
x_q = torch.gather(code, dim=-2, index=indices)
print(x_q)
print(x - x_q)
