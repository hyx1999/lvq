import math
import torch
import torch.nn.functional as F
import fast_hadamard_transform

# x = torch.randn((4,))
# x_scale = x.pow(2).mean().sqrt()
# x = x / (x_scale + 1e-5)
# print(x)
# print(F.gumbel_softmax(x, tau=4))

# x = torch.empty((4,))
# torch.nn.init.normal_(x, std=1)
# print(x * 0.1)

d = 128
x = torch.randn((d,), dtype=torch.float16, device="cuda")
org_x = x
print(x)
x = fast_hadamard_transform.hadamard_transform(x, 1 / math.sqrt(d))
x = fast_hadamard_transform.hadamard_transform(x, 1 / math.sqrt(d))
print(x)
print((x - org_x).abs().mean())
