import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from lvq.quant.scheduler import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


class TrainingStates:
    
    def __init__(self, train_iters: int):
        self.iteration = 0
        self.train_iters = train_iters

    def step(self):
        self.iteration += 1


class LearnIndex(nn.Module):
    def __init__(
        self, 
        states: TrainingStates,
        index_shape: torch.Size,
        hard=False,
        temperature=[4.0, 0.1], 
        scale_multiplier=[1e2, 5e2],
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
        self.states = states
        self.M = index_shape[-1]
        self.dtype = dtype
        self.device = device
        self.mask_difference = 1.0
        self.initial_index_shape = index_shape
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        
        print(index_shape, index_shape.numel())
        
        self.gate = nn.Parameter(torch.empty(
            *index_shape,
            device=device, 
            dtype=dtype
        ))
        self.options = nn.Buffer(self.init_options())
    
        self.hard = hard

        self.current_scale_multiplier = self.scale_multiplier[0]
        self.current_temperature = self.temperature[0]
        self.current_max_prob = 0.0
        
    def init_options(self):
        options = torch.zeros(self.M, self.M,
                        device=self.device, dtype=self.dtype)
        for i in range(self.M):
            options[i, :].data += (torch.arange(0, self.M) == i)\
                .type_as(options)\
                .unsqueeze(0)\
                .repeat(options.shape[0], 1)

    def forward(self): 
        if self.training:
            start_temp, end_temp = self.temperature 
            self.current_temperature = start_temp + (end_temp - start_temp) * (self.states.iteration / self.states.train_iters)
            start_scale, end_scale = self.scale_multiplier
            self.current_scale_multiplier = start_scale + (end_scale - start_scale) * (self.states.iteration / self.states.train_iters)
            
            sampling_tensor = self.gate * self.current_scale_multiplier
            choices = F.gumbel_softmax(sampling_tensor, tau=self.current_temperature, hard=self.hard, dim=-1)
            weighted_index = (choices.unsqueeze(-2) @ self.options).squeeze(-2)  # [..., 1, M] @ [M, M] => [..., 1, M] => [..., M]
            weighted_index = weighted_index.reshape(self.initial_index_shape)

            # metric
            self.current_max_prob = choices.max(-1)[0].mean().item()
        else:
            weighted_index = self.options[self.gate.argmax(dim=-1)]

        return weighted_index


class Reconstructor(nn.Module):
    
    def __init__(self, 
        states: TrainingStates,
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

        self.learn_index = LearnIndex(
            states,
            torch.Size((num_lut, in_features // vec_size, out_features, lut_size)),
            device=device,
            dtype=dtype
        )
        self.codebook = nn.Parameter(
            torch.empty((num_lut, in_features // vec_size, lut_size, vec_size)),
            device=device,
            dtype=dtype,
        )
        self.scales = nn.Parameter(
            torch.empty(out_features, in_features // group_size),
            device=device,
            dtype=dtype,
        )

    def forward(self):
        weighted_index: torch.Tensor = self.learn_index()  # (num_lut, in_features // vec_size, out_features, lut_size)
        # (num_lut, in_features // vec_size, out_features, lut_size) @ (num_lut, in_features // vec_size, lut_size, vec_size)
        #   => (num_lut, in_features // vec_size, out_features, vec_size)
        #   => (num_lut, out_features, in_features // vec_size, vec_size)
        weight = (weighted_index @ self.codebook)\
            .transpose(1, 2)\
            .sum(dim=0)\
            .reshape(self.out_features, self.in_features)
        return weight


@torch.no_grad()
def init_reconstructor_awq(
    reconstructor: Reconstructor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    raise NotImplementedError


def rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_scale = x.pow(2).mean(-1).sqrt()
    norm_x = x / (x_scale.unsqueeze(-1) + eps)
    return norm_x


@torch.no_grad()
def init_reconstructor_kmeans(
    reconstructor: Reconstructor,
    weight: torch.Tensor,
):
    out_features = reconstructor.out_features
    in_features = reconstructor.in_features
    num_lut = reconstructor.num_lut
    lut_size = reconstructor.lut_size
    vec_size = reconstructor.vec_size
    group_size = reconstructor.group_size

    weight = weight.reshape((
        out_features,
        in_features // group_size,
        group_size,
    ))
    scales = weight.pow(2).mean(dim=-1).sqrt()
    norm_weight = rmsnorm(weight)
    norm_weight = norm_weight.reshape((out_features, in_features))

    for iter in range(num_lut):
        norm_weight = norm_weight\
            .reshape((out_features, in_features // vec_size, vec_size))\
            .transpose(0, 1)
        for j in tqdm(range(in_features // vec_size), desc="init codebook[{}]...".format(iter)):
            weight_group = norm_weight[j, ...]
            kmeans = MiniBatchKMeans(
                n_clusters=lut_size
            ).fit(weight_group.cpu().float().numpy())
            indices = torch.tensor(kmeans.labels_, dtype=torch.int32, device=weight.device)
            code = torch.tensor(kmeans.cluster_centers_, dtype=weight.dtype, device=weight.device)
            gate = weight_group @ code.T  # (R, V) @ (V, M)  =>  (R, M)
            norm_gate = rmsnorm(gate)

            norm_weight[j, ...].sub_(code[indices])
            reconstructor.codebook[iter, j].copy_(code)
            reconstructor.learn_index.gate[iter, j].copy_(norm_gate)
    reconstructor.scales.data.copy_(scales)

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
):
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype
    
    group_size = in_features if group_size == -1 else group_size
    
    assert in_features % group_size == 0
    assert in_features % vec_size == 0
    
    states = TrainingStates(train_iters)
    reconstructor = Reconstructor(
        states,
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
    init_reconstructor_kmeans(reconstructor, weight)
    
    # optimization...
    optimizer = AdamW(reconstructor.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, num_training_steps=train_iters, 
        max_learning_rate=args.lr, min_learning_rate=args.min_lr
    )
        
    with torch.enable_grad():
        for iter in tqdm(range(train_iters), desc="optimize lvq..."):
            optimizer.zero_grad()

            recons_weight: torch.Tensor = reconstructor()
            d_w = weight - recons_weight
            loss = torch.trace(d_w @ hessian @ d_w.T)

            loss.backward()
            states.step()
            optimizer.step()
            scheduler.step()
