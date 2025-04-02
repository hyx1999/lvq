import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
from lvq.quant.logger_utils import init_logging
from lvq.quant.lvq_quant import RTN
from lvq.quant.utils import get_loaders, eval_ppl, model_utils
from tqdm import tqdm

def main(args):
    init_logging()

    lm = AutoModelForCausalLM.from_pretrained(args.model)
    lm.seqlen = args.seqlen
    
    dataloader, _ = get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=lm.seqlen,
        model=args.model,
    )

    _, testloader = get_loaders(
        args.eval_dataset,
        seed=args.seed,
        seqlen=lm.seqlen,
        model=args.model,
    )

    lm.to(args.device)
    layers = model_utils.get_layers(lm)
    linears = []
    for name, module in layers.named_modules():
        if isinstance(module, torch.nn.Linear):
            linears.append(module)
    for module in tqdm(linears):
        module.weight.data.copy_(RTN(module.weight, args.group_size))
    
    ppl = eval_ppl(lm, testloader, args.device)
    print("ppl: {}".format(ppl))
    # Llama-3.2-1B => ppl: 294.1290588378906


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General Arguments
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        help='Dataset for Calibration (default: wikitext2)')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='c4',
                        help='calibration data samples for GPTQ.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save quantized model.')
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--num_lut', type=int, default=3)
    parser.add_argument('--lut_size', type=int, default=16)
    parser.add_argument('--vec_size', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=128,
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--train_iters', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    args = parser.parse_args()
    assert args.lut_size == 16
    assert args.vec_size == 4
    main(args)
