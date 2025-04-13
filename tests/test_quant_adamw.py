import torch
from transformers import LlamaForCausalLM
from lvq.quant.quantize import quant_adamw

# model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained("/data/models/Llama-3.2-1B")
# state_dict = model.state_dict()
# weight = state_dict["model.layers.0.self_attn.q_proj.weight"]
# torch.save(weight, "misc/weight.pt")

def RTN(weight: torch.Tensor, group_size: int, nbit=3):
    max_int = 2**(nbit - 1) - 1
    min_int = -max_int - 1
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    weight = weight.reshape(out_features, in_features // group_size, group_size)
    scales = weight.abs().max(dim=-1, keepdim=True).values / max_int

    qweight = torch.clamp(torch.round(weight / scales), min_int, max_int)
    new_weight = qweight * scales
    return new_weight.reshape(out_features, in_features)


class Args:
    
    def __init__(self):
        self.num_lut = 3
        self.lut_size = 16
        self.vec_size = 4
        self.group_size = 128
        self.train_iters = 512
        self.lr = 1e-4
        self.min_lr = 1e-6

args = Args()
weight: torch.Tensor = torch.load("misc/weight.pt").to("cuda:1")
hessian = torch.eye(weight.shape[1]).type_as(weight).to(weight.device)

rtn_weight = RTN(weight.clone(), args.group_size)
print("dw:\n{}".format(rtn_weight - weight))
print("max(abs(dw)): {}".format((rtn_weight - weight).abs().max()))
print("mean(abs(dw)): {}".format((rtn_weight - weight).abs().mean()))
print("norm(dw): {}".format((rtn_weight - weight).norm().item()))
d_w = rtn_weight - weight
print("trace: {}".format(torch.trace(d_w @ hessian @ d_w.T)))

quant_results, lvq_weight = quant_adamw.reconstruct_weight_adamw(
    args,
    weight.clone(),
    hessian,
    args.num_lut,
    args.lut_size,
    vec_size=args.vec_size,
    group_size=args.group_size,
    train_iters=args.train_iters,    
    return_weight=True,
    init_method="kmeans"
)
print("dw:\n{}".format(lvq_weight - weight))
print("max(abs(dw)): {}".format((lvq_weight - weight).abs().max()))
print("mean(abs(dw)): {}".format((lvq_weight - weight).abs().mean()))
print("norm(dw): {}".format((lvq_weight - weight).norm().item()))
d_w = lvq_weight - weight
print("trace: {}".format(torch.trace(d_w @ hessian @ d_w.T)))
