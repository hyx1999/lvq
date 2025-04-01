import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
from lvq.quant.logger_utils import init_logging
from lvq.quant.lvq_quant import lvq_quant
from lvq.quant.utils import get_loaders, eval_ppl
import logging

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

    lvq_quant(args, lm, dataloader)
    
    _, testloader = get_loaders(
        args.eval_dataset,
        seed=args.seed,
        seqlen=lm.seqlen,
        model=args.model,
    )
    
    ppl = eval_ppl(lm, testloader, args.device)
    print("ppl: {}".format(ppl))


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
    parser.add_argument('--train_iters', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    args = parser.parse_args()
    assert args.lut_size == 16
    assert args.vec_size == 4
    main(args)
