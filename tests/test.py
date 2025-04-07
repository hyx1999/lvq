import torch
import torch.nn.functional as F

# x = torch.randn((4,))
# x_scale = x.pow(2).mean().sqrt()
# x = x / (x_scale + 1e-5)
# print(x)
# print(F.gumbel_softmax(x, tau=4))

x = torch.empty((4,))
torch.nn.init.normal_(x, std=1)
print(x * 0.1)
