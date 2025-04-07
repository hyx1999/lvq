import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from lvq.quant.scheduler import get_cosine_schedule_with_warmup
from lvq.modules import LvqLinear
from tqdm import tqdm
from sklearn.cluster import KMeans
import logging

DISABLE_TQDM = False

@torch.no_grad()
def init_reconstructor_linear(
    reconstructor: LvqLinear,
    weight: torch.Tensor,
):
    out_features = reconstructor.out_features
    in_features = reconstructor.in_features
    num_lut = reconstructor.num_lut
    lut_size = reconstructor.lut_size
    vec_size = reconstructor.vec_size
    group_size = reconstructor.group_size
    
    reconstructor.weight.data.copy_(weight)
      
    weight = weight.reshape((
        out_features,
        in_features // group_size,
        group_size,
    ))
    scales = weight.abs().max(dim=-1, keepdim=True).values / (1 - 1 / (2 ** num_lut))

    for i in range(num_lut):
        for j in range(in_features // vec_size):
            code = torch.tensor(
                [[(x >> y) & 1 for y in range(vec_size - 1, -1, -1)] for x in range(lut_size)],
                dtype=weight.dtype,
                device=weight.device,
            )
            code = 2 * code - 1  # {0, 1} => {-1, 1}
            reconstructor.quantizer.quantizers[i].codebook[j].copy_(code)
        reconstructor.quantizer.quantizers[i].scale.zero_().add_(1 / (2 ** (i + 1)))
    reconstructor.scales.data.copy_(scales.squeeze(-1))


@torch.no_grad()
def normalize(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm_x = x / (scale.unsqueeze(-1) + eps)
    return norm_x

@torch.no_grad()
def init_reconstructor_kmeans(
    reconstructor: LvqLinear,
    weight: torch.Tensor,
):
    out_features = reconstructor.out_features
    in_features = reconstructor.in_features
    num_lut = reconstructor.num_lut
    lut_size = reconstructor.lut_size
    vec_size = reconstructor.vec_size
    group_size = reconstructor.group_size
    
    reconstructor.weight.data.copy_(weight)

    weight = weight.reshape((
        out_features,
        in_features // group_size,
        group_size,
    ))
    scales = weight.abs().max(dim=-1).values
    # scales = weight.pow(2).mean(dim=-1).sqrt()
    norm_weight = normalize(weight, scales)
    norm_weight = norm_weight\
        .reshape((out_features, in_features // vec_size, vec_size))\
        .transpose(0, 1)
    for i in range(num_lut):
        for j in tqdm(range(in_features // vec_size), desc="init codebook[{}]...".format(i)):
            weight_group = norm_weight[j, ...]
            kmeans = KMeans(
                n_clusters=lut_size,
            ).fit(weight_group.cpu().float().numpy())
            indices = torch.tensor(kmeans.labels_, dtype=torch.int32, device=weight.device)
            code = torch.tensor(kmeans.cluster_centers_, dtype=weight.dtype, device=weight.device)
            norm_weight[j, ...].sub_(code[indices])
            reconstructor.quantizer.quantizers[i].codebook[j].copy_(code * (2 ** i))
        reconstructor.quantizer.quantizers[i].scale.zero_().add_(1 / (2 ** i))
    reconstructor.scales.data.copy_(scales)


@torch.no_grad()
def reconstruct_weight_adamw(
    args,
    weight: torch.Tensor,
    hessian: torch.Tensor,  # [in_features, in_features]
    num_lut: int = 3,
    lut_size: int = 16,
    vec_size: int = 4,
    group_size: int = 128,
    train_iters: int = 2048,
    return_weight: bool = False,
    init_method: str = "kmeans",
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
    
    # init params...
    if init_method == "linear":
        init_reconstructor_linear(reconstructor, weight)
    elif init_method == "kmeans":
        init_reconstructor_kmeans(reconstructor, weight)
    else:
        raise ValueError
    
    # optimization...
    optimizer = AdamW(reconstructor.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, num_training_steps=train_iters, 
        max_learning_rate=args.lr, min_learning_rate=args.min_lr
    )

    with torch.enable_grad():
        bar = tqdm(range(train_iters), desc="optimize lvq...", disable=DISABLE_TQDM)
        for iter in range(train_iters):
            optimizer.zero_grad()

            recons_weight: torch.Tensor = reconstructor.quantize_weight()
            d_w = weight - recons_weight
            loss = torch.trace(d_w @ hessian @ d_w.T)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            bar.update(1)
            if iter % 8 == 0:
                bar.set_postfix_str("loss = {:.5f}".format(loss.item()))
    
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
