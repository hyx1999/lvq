import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from lvq.quant.scheduler import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

DISABLE_TQDM = False

class TrainingStates:
    
    def __init__(self, train_iters: int):
        self.iteration = 0
        self.train_iters = train_iters

    def step(self):
        self.iteration += 1
        self.iteration = min(self.iteration, self.train_iters)


class StaticIndex(nn.Module):
    def __init__(
        self, 
        index_shape: torch.Size,
        device=None,
        dtype=None,
    ):
        '''
        Implementation of differantiable mask learner
        args:
            temperature: temperature of the gumbel distribution
            gate_param: the parameter to be masked
            init_prob: initial probability of the mask
            scale_multiplier: multiplies learned gates by this value, it is needed to make the gates more sensitive to small learning rates
            initialization: "none" means we start from 0.95 for all and dont bother with initialization, "initial_mask" means we start from the initial mask
            hard: sampling mask for full gumbel
            
        temperature parameter needs more attention, we should do annealing it during training
        '''
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.M = index_shape[-1]
                
        self.gate = nn.Buffer(torch.empty(
            *index_shape,
            device=device, 
            dtype=dtype
        ))
        self.options = nn.Buffer(self.init_options())    
        
    def init_options(self):
        options = torch.zeros(self.M, self.M,
                        device=self.device, dtype=self.dtype)
        for i in range(self.M):
            options[i, :].data += (torch.arange(0, self.M) == i)\
                .type_as(options)
        return options

    def forward(self):
        index = self.options[self.gate.argmax(dim=-1)]
        return index


class Reconstructor(nn.Module):
    
    def __init__(self, 
        in_features: int,
        out_features: int,
        num_lut: int = 3,
        lut_size: int = 16,
        vec_size: int = 4,
        group_size: int = -1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert group_size != -1
        self.in_features = in_features
        self.out_features = out_features
        self.num_lut = num_lut
        self.lut_size = lut_size
        self.vec_size = vec_size
        self.group_size = group_size
        self.dtype = dtype
        self.device = device

        self.learn_index = StaticIndex(
            torch.Size((num_lut, in_features // vec_size, out_features, lut_size)),
            device=device,
            dtype=dtype
        )
        self.codebook = nn.Parameter(
            torch.empty(num_lut, in_features // vec_size, lut_size, vec_size, device=device, dtype=dtype)
        )
        self.scales = nn.Parameter(
            torch.empty(out_features, in_features // group_size, device=device, dtype=dtype)
        )
        self.zeros = nn.Buffer(
            torch.empty(out_features, in_features // group_size, device=device, dtype=torch.int8)
        )

    def forward(self):
        weighted_index: torch.Tensor = self.learn_index()
        weight = (weighted_index @ self.codebook)\
            .transpose(1, 2)\
            .sum(dim=0)\
            .reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        weight = (weight - self.zeros.type_as(weight).unsqueeze(-1)) * self.scales.unsqueeze(-1)
        weight = weight.reshape(self.out_features, self.in_features)
        return weight


@torch.no_grad()
def init_reconstructor_awq(
    reconstructor: Reconstructor,
    weight: torch.Tensor,
):
    out_features = reconstructor.out_features
    in_features = reconstructor.in_features
    num_lut = reconstructor.num_lut
    lut_size = reconstructor.lut_size
    vec_size = reconstructor.vec_size
    group_size = reconstructor.group_size
    max_int: int = (2 ** num_lut) - 1

    assert lut_size == 16
  
    weight = weight.reshape((
        out_features,
        in_features // group_size,
        group_size,
    ))
    max_w = weight.max(dim=-1, keepdim=True).values
    min_w = weight.min(dim=-1, keepdim=True).values
    scales = (max_w - min_w).clamp(min=1e-5) / max_int
    zeros  = (-torch.round(min_w / scales)).clamp_(0, max_int)
    qweight = torch.clamp(torch.round(weight / scales) + zeros, 0, max_int).to(torch.int8)
    qweight = qweight\
        .reshape((out_features, in_features // vec_size, vec_size))\
        .transpose(0, 1)
    
    for iter in range(num_lut):
        for j in range(in_features // vec_size):
            code = torch.tensor(
                [[(x >> 3) & 1, (x >> 2) & 1, (x >> 1) & 1, (x >> 0) & 1] for x in range(lut_size)],
                dtype=weight.dtype,
                device=weight.device,
            )
            index = (qweight[j] >> iter) & 1
            norm_gate = (index[:, None, :] == code.to(torch.int8)[None, :, :]).sum(dim=-1).to(weight.dtype) / 4.0
            reconstructor.codebook[iter, j].copy_(code * (2 ** iter))
            reconstructor.learn_index.gate[iter, j].copy_(norm_gate)
    reconstructor.scales.data.copy_(scales.squeeze(-1))
    reconstructor.zeros.data.copy_(zeros.squeeze(-1).to(torch.int8))


@torch.no_grad()
def reconstruct_weight(
    args,
    weight: torch.Tensor,
    hessian: torch.Tensor,  # [in_features, in_features]
    num_lut: int = 3,
    lut_size: int = 16,
    vec_size: int = 4,
    group_size: int = -1,
    train_iters: int = 1024,
    return_weight: bool = False
):
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype
    
    group_size = in_features if group_size == -1 else group_size
    
    assert in_features % group_size == 0
    assert in_features % vec_size == 0
    
    reconstructor = Reconstructor(
        in_features=in_features,
        out_features=out_features,
        num_lut=num_lut,
        lut_size=lut_size,
        vec_size=vec_size,
        group_size=group_size,
        device=device,
        dtype=dtype,
    )
    
    # init params...
    init_reconstructor_awq(reconstructor, weight)
    
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

            recons_weight: torch.Tensor = reconstructor()
            d_w = weight - recons_weight
            loss = torch.trace(d_w @ hessian @ d_w.T)
            loss.backward()
            optimizer.step()
            scheduler.step()
            bar.update(1)
            if iter % 64 == 0:
                bar.set_postfix_str("loss = {:.5f}".format(loss.item()))

    new_weight = None
    if return_weight:
        reconstructor.eval()
        new_weight = reconstructor()
    
    quant_results = {
        "scales": reconstructor.scales.data,
        "codebook": reconstructor.codebook.data,
        "index": reconstructor.learn_index.gate.argmax(dim=-1),
    }
    
    return quant_results, new_weight
