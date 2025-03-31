import torch
from lvq.modules.tunable_lut_linear import TunableLutLinear

# class Args:
    
#     def __init__(self):
#         self.iteration = 0
#         self.train_iters = 1000

# args = Args()

# fc = TunableLutLinear(
#     128,
#     128,
#     args,
#     num_lut=3,
#     lut_size=16,
#     vec_size=4,
#     device="cuda",
#     dtype=torch.bfloat16
# )

# x = torch.randn((16, 128)).to(torch.bfloat16).cuda()
# y: torch.Tensor = fc(x)
# print("y = \n{}".format(y))
# y.mean().backward()
# print("grad = \n{}".format(fc.weight.gate.grad))

import torch.nn.functional as F
x = torch.zeros((16,))
x[0] = 1
print(x)
print(F.softmax(x))
print(F.gumbel_softmax(F.softmax(x) * 150, tau=4))
