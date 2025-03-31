import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import gumbel_softmax
from functools import partial

class DifferentiableIndex(nn.Module):
    def __init__(
        self, 
        args,
        index_shape,
        N=1, 
        M=16, 
        hard=False,
        temperature=[4.0, 0.1], 
        scale_multiplier=[1e2, 5e2],
        device=None,
        dtype=None,
        reset_params=False,
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
        self.args = args
        self.N = N
        self.M = M
        self.mask_difference = 1.0
        self.initial_index_shape = index_shape
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        
        print(index_shape, index_shape.numel())
        
        if self.N==1 and self.M==16:
            self.gate = nn.Parameter(torch.randn(
                *index_shape, self.M,
                device=device, 
                dtype=dtype
            ))
            index_options = torch.zeros(1, self.M, self.M,
                                    device=device, dtype=dtype)
            for i in range(self.M):
                index_options[:, i, :].data += (torch.arange(0, self.M) == i)\
                    .type_as(index_options)\
                    .unsqueeze(0)\
                    .repeat(index_options.shape[0], 1)
        else:
            raise NotImplementedError
        
        self.register_buffer("index_options", index_options)
        self.hard = hard

        self.current_scale_multiplier = self.scale_multiplier[0]
        self.current_temperature = self.temperature[0]
        self.current_max_prob = 0.0
        
        self.reset_params()
    
    def reset_params(self):
        nn.init.normal_(self.gate, mean=0, std=0.01)

    def forward(self): 

        if self.training:
            start_temp, end_temp = self.temperature 
            self.current_temperature = start_temp + (end_temp - start_temp) * (self.args.iteration / self.args.train_iters)
            start_scale, end_scale = self.scale_multiplier
            self.current_scale_multiplier = start_scale + (end_scale - start_scale) * (self.args.iteration / self.args.train_iters)
            
            sampling_tensor = self.gate * self.current_scale_multiplier
            choices = gumbel_softmax(sampling_tensor, tau=self.current_temperature, hard=self.hard, dim=-1)
            print("choices: {}".format(choices.shape))
            print(choices.view(-1, self.M))
            print("sum choices: {}".format(choices.view(-1, self.M).sum(-1)))
            backprop_gate = (choices.unsqueeze(-2) @ self.index_options).squeeze(-2)  # [..., 1, M] @ [1, M, M] => [N, 1, M]
            backprop_gate = backprop_gate.reshape(self.initial_index_shape + (self.M,))
            print("backprop_gate: {}".format(backprop_gate.shape))
            print(backprop_gate.view(-1, self.M))
            print("max backprop_gate:")
            print(backprop_gate.view(-1, self.M).max(dim=-1).values)

            # metric
            self.current_max_prob = choices.max(-1)[0].mean().item()
        else:
            # just based on the maximum logit
            backprop_gate = self.mask_options[torch.arange(self.mask_options.shape[0]), self.gate.argmax(dim=-1)]
            backprop_gate = backprop_gate.reshape(self.initial_index_shape + (self.M,))
        self.sampled_gate = backprop_gate
        return backprop_gate
