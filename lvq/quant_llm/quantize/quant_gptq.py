import logging
import torch
from tqdm import tqdm
from lvq.modules import QLinear
from .quantizer import WeightQuantizer


@torch.no_grad()
def quant_weight(
    args,
    reconstructor: QLinear,
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
    quantizer = WeightQuantizer(shape=out_features)
    quantizer.configure(
        bits=num_bits,
        perchannel=True,
    )
    
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

            if (i1 + i) % group_size == 0:
                quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)])
                reconstructor.set_params(
                    (i1 + i) // group_size, 
                    quantizer.scale.squeeze(-1),
                    quantizer.zero.squeeze(-1),
                )

            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
            bar.update(1)

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    reconstructor.weight.data.copy_(Q)


@torch.no_grad()
def quant_weight_gptq(
    args,
    weight: torch.Tensor,
    hessian: torch.Tensor,  # [in_features, in_features]
    num_bits: int = 4,
    group_size: int = 128,
    return_weight: bool = False,
):
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype
    
    group_size = in_features if group_size == -1 else group_size
    
    assert in_features % group_size == 0
    
    reconstructor = QLinear(
        in_features=in_features,
        out_features=out_features,
        num_bits=num_bits,
        group_size=group_size,
        bias=False,
        device=device,
        dtype=dtype,
    )
    # print("hessian=\n{}".format(hessian))
    quant_weight(args, reconstructor, weight, hessian)
    
    new_weight = None
    if return_weight:
        reconstructor.eval()
        new_weight = reconstructor.weight
        d_w = weight - new_weight
        loss = torch.trace(d_w @ hessian @ d_w.T).item()
        logging.info("loss = {:4f}".format(loss))

    quant_results = {
        k: v for k, v in reconstructor.state_dict().items()
    }
    
    return quant_results, new_weight
