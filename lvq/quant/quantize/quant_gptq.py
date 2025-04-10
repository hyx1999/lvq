import logging
import torch
from lvq.modules import LvqLinear
from tqdm import tqdm
from lvq.quant.quantize import quant_vq

DISABLE_TQDM = False

@torch.no_grad()
def normalize(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm_x = x / (scale.unsqueeze(-1) + eps)
    return norm_x

class Quantizer:
    
    def __init__(self, 
        num_lut: int,
        lut_size: int,
        vec_size: int,
    ):
        self.num_lut = num_lut
        self.lut_size = lut_size
        self.vec_size = vec_size
        self.scales = None
        self.codebook = None
    
    @torch.no_grad()
    def normalize(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        norm_x = x / (self.scales.unsqueeze(-1) + eps)
        return norm_x
    
    def find_params(self, weight: torch.Tensor):
        self.scales = weight.abs().max(dim=-1).values
    
    @torch.no_grad()
    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        weight = self.normalize(weight)
        qweight = torch.zeros_like(weight)
        self.codebook = torch.zeros((self.num_lut, self.lut_size, self.vec_size)).type_as(weight)
        for i in range(self.num_lut):
            indices, code = quant_vq.find_params(weight.unsqueeze(0))
            indices = indices.squeeze(0)
            code = code.squeeze(0)
            weight.sub_(code[indices])
            qweight.add_(code[indices])
            self.codebook[i].copy_(code)
        new_weight = qweight * self.scales.unsqueeze(-1)
        return new_weight

@torch.no_grad()
def reconstruct_gptq(
    args,
    reconstructor: LvqLinear,
    weight: torch.Tensor,
    hessian: torch.Tensor,
):
    out_features = reconstructor.out_features
    in_features = reconstructor.in_features
    num_lut = reconstructor.num_lut
    lut_size = reconstructor.lut_size
    vec_size = reconstructor.vec_size
    group_size = reconstructor.group_size
    percdamp = args.gptq_percdamp
    blocksize = args.gptq_blocksize
    dev = weight.device
    quantizer = Quantizer(
        num_lut,
        lut_size,
        vec_size
    )
    
    reconstructor.weight.data.copy_(weight)
    
    H = hessian.clone()
    W = weight.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(in_features, device=dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    bar = tqdm(list(range(in_features // vec_size)), desc="GPTQ...")
    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        count = i2 - i1
        assert count % vec_size == 0

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(0, count, vec_size):
            if (i1 + i) % group_size == 0:
                quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)])
                reconstructor.update_scale((i1 + i) // group_size, quantizer.scales)

            ws = W1[:, i:i+vec_size]
            ds = torch.diag(Hinv1)[i:i+vec_size]
        
            qs = quantizer.quantize(ws)
            for lut_id in range(num_lut):
                reconstructor.update_codebook(lut_id, (i1 + i) // vec_size, quantizer.codebook[lut_id])

            Q1[:, i:i+vec_size] = qs
            for j in range(vec_size):
                w = ws[:, j]
                q = qs[:, j]
                d = ds[j]
                Losses1[:, i+j] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i+j:] -= err1.unsqueeze(1).matmul(Hinv1[i+j, i+j:].unsqueeze(0))
                Err1[:, i+j] = err1
            bar.update(1)

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


@torch.no_grad()
def reconstruct_weight_gptq(
    args,
    weight: torch.Tensor,
    hessian: torch.Tensor,  # [in_features, in_features]
    num_lut: int = 3,
    lut_size: int = 16,
    vec_size: int = 4,
    group_size: int = 128,
    return_weight: bool = False,
):
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype
    
    group_size = in_features if group_size == -1 else group_size
    
    assert in_features % group_size == 0
    assert in_features % vec_size == 0
    
    reconstructor = LvqLinear(
        in_features=in_features,
        out_features=out_features,
        num_lut=num_lut,
        lut_size=lut_size,
        vec_size=vec_size,
        group_size=group_size,
        bias=False,
        device=device,
        dtype=dtype,
    )
    # print("hessian=\n{}".format(hessian))
    reconstruct_gptq(args, reconstructor, weight, hessian)
    
    # compute loss
    reconstructor.eval()
    recons_weight: torch.Tensor = reconstructor.quantize_weight()
    d_w = weight - recons_weight
    loss = torch.trace(d_w @ hessian @ d_w.T).item()
    logging.info("loss = {:4f}".format(loss))
    # print("weight=\n{}".format(weight))
    # print("recons_weight=\n{}".format(recons_weight))
    # print("dw=\n{}".format(d_w))

    new_weight = None
    if return_weight:
        reconstructor.eval()
        new_weight = reconstructor.quantize_weight()
    
    quant_results = {
        k: v for k, v in reconstructor.state_dict().items() 
            if "quantizer" in k or "scales" in k
    }
    
    return quant_results, new_weight
