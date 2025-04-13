import torch
from transformers import LlamaForCausalLM
from lvq.quant_llm.quantize import quant_gptq

# model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained("/data/models/Llama-3.2-1B")
# state_dict = model.state_dict()
# weight = state_dict["model.layers.0.self_attn.q_proj.weight"]
# torch.save(weight, "misc/weight.pt")

def RTN(weight: torch.Tensor, group_size: int, nbits: int):
    max_int = (2 ** (nbits - 1)) - 1
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
        self.num_bits = 4
        self.group_size = 128
        self.gptq_percdamp = 0.01
        self.gptq_blocksize = 128

args = Args()
weight: torch.Tensor = torch.load("misc/weight.pt").to("cuda:1")
hessian = torch.eye(weight.shape[1]).type_as(weight).to(weight.device)

rtn_weight = RTN(weight.clone(), args.group_size, args.num_bits)
print("dw:\n{}".format(rtn_weight - weight))
print("max(abs(dw)): {}".format((rtn_weight - weight).abs().max()))
print("mean(abs(dw)): {}".format((rtn_weight - weight).abs().mean()))
print("norm(dw): {}".format((rtn_weight - weight).norm().item()))
d_w = rtn_weight - weight
print("trace: {}".format(torch.trace(d_w @ hessian @ d_w.T)))

_, lvq_weight = quant_gptq.quant_weight_gptq(
    args,
    weight.clone(),
    hessian,
    args.num_bits,
    group_size=args.group_size,
    return_weight=True,
)
print("dw:\n{}".format(lvq_weight - weight))
print("max(abs(dw)): {}".format((lvq_weight - weight).abs().max()))
print("mean(abs(dw)): {}".format((lvq_weight - weight).abs().mean()))
print("norm(dw): {}".format((lvq_weight - weight).norm().item()))
d_w = lvq_weight - weight
print("trace: {}".format(torch.trace(d_w @ hessian @ d_w.T)))
