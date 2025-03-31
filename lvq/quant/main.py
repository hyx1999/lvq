import argparse
import re
import shutil
import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
from lvq.quant.logger_utils import init_logging
from lvq.quant.lvq_quant import lvq_quant
from lvq.quant.utils import get_loaders

def main(args):
    init_logging()

    lm = AutoModelForCausalLM.from_pretrained(args.model)
    lm.seqlen = args.seqlen
    
    dataloader, _ = get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=lm.seqlen,
    )

    lvq_quant(args, lm, dataloader)
    

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
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='c4',
                        help='calibration data samples for GPTQ.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save quantized model.')
    args = parser.parse_args()
    main(args)
