import logging
import torch
from lvq.modules import LutLinear
from tqdm import tqdm
from lvq.quant.quantize import quant_vq

DISABLE_TQDM = False

@torch.no_grad()
def normalize(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm_x = x / (scale.unsqueeze(-1) + eps)
    return norm_x


class Quantizer:
    
    def __init__(self, 
        num_bits: int,
    ):
        self.num_bits = num_bits
        self.min_int = 0
        self.max_int = (2 ** self.num_bits) - 1
        self.row_scale = None
        self.col_scale = None
        self.col_bias = None
        
    def find_params(self, weight: torch.Tensor):
        self.row_scale = weight.abs().max(dim=-1).values.unsqueeze(-1) + 1e-5
    
    @torch.no_grad()
    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight / self.row_scale
        self.col_bias = weight.min(dim=0).values
        weight = weight - self.col_bias[None]
        self.col_scale = weight.max(dim=0).values + 1e-5
        weight = weight / self.col_scale
        weight = torch.clamp(
            torch.round(weight * self.max_int), 
            self.min_int, self.max_int
        )
        weight = weight / self.max_int
        weight = weight * self.col_scale
        weight = weight + self.col_bias
        weight = weight * self.row_scale
        return weight


@torch.no_grad()
def reconstruct_gptq(
    args,
    reconstructor: LutLinear,
    weight: torch.Tensor,
    hessian: torch.Tensor,
):
    out_features = reconstructor.out_features
    in_features = reconstructor.in_features
    num_bits = reconstructor.num_bits
    group_size = reconstructor.group_size
    percdamp = args.gptq_percdamp
    blocksize = args.gptq_blocksize
    dev = weight.device
    quantizer = Quantizer(num_bits)
        
    H = hessian.clone()
    W = weight.clone()
    
    U, S, Vt = torch.svd_lowrank(W, q=args.lora_rank)
    A = U @ torch.diag(S)
    B = Vt.T
    W -= A @ B
    reconstructor.A.copy_(A)
    reconstructor.B.copy_(B)
    reconstructor.weight.data.copy_(W)

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

    bar = tqdm(list(range(in_features)), desc="GPTQ...")
    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if group_size != -1:
                if (i1 + i) % group_size == 0:
                    quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)])
                    reconstructor.update_row_params(
                        (i1 + i) // group_size, 
                        quantizer.row_scale.squeeze(-1)
                    )

            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            reconstructor.update_col_params(
                (i1 + i), 
                quantizer.col_scale.squeeze(-1),
                quantizer.col_bias.squeeze(-1),
            )
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
            bar.update(1)

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


@torch.no_grad()
def reconstruct_weight_gptq(
    args,
    weight: torch.Tensor,
    hessian: torch.Tensor,  # [in_features, in_features]
    num_bits: int = 3,
    group_size: int = 128,
    return_weight: bool = False,
):
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype
    
    group_size = in_features if group_size == -1 else group_size
    
    assert in_features % group_size == 0
    
    reconstructor = LutLinear(
        in_features=in_features,
        out_features=out_features,
        num_bits=num_bits,
        group_size=group_size,
        lora_rank=args.lora_rank,
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

    new_weight = None
    if return_weight:
        reconstructor.eval()
        new_weight = reconstructor.quantize_weight()
    
    quant_results = {
        k: v for k, v in reconstructor.state_dict().items() 
            if "quantizer" in k or "scales" in k
    }
    
    return quant_results, new_weight
