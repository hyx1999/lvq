import argparse
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM
from typing import Dict
from lvq.quant_llm.logger_utils import init_logging
from lvq.patches import register_attn_modules
from lvq.quant_llm.prequant import (
    prequant_quarot,
    prequant_awq,
)
from lvq.quant_llm.quant_gptq import gptq
from lvq.quant_llm.utils import get_loaders, eval_ppl


def main(args):
    init_logging()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.seqlen = args.seqlen
    
    assert isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM))
    
    dataloader, _ = get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        model=args.model,
    )

    register_attn_modules(args, model)
    if args.use_quarot:
        prequant_quarot(args, model)
    if args.use_awq:
        prequant_awq(args, model, dataloader)
    gptq(args, model, dataloader)
    if args.use_kv_quant:
        model.config.enable_kv_quant = True
    
    model.seqlen = 2048
    _, testloader = get_loaders(
        args.eval_dataset,
        seed=args.seed,
        seqlen=model.seqlen,
        model=args.model,
    )
    ppl = eval_ppl(model, testloader, args.device)
    print("ppl: {}".format(ppl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General Arguments
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--seqlen', type=int, default=512)
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

    parser.add_argument('--w_bits', type=int, default=4)
    parser.add_argument('--k_bits', type=int, default=8)
    parser.add_argument('--v_bits', type=int, default=8)
    parser.add_argument('--group_size', type=int, default=128,
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--gptq_percdamp', type=float, default=0.05)
    parser.add_argument('--gptq_blocksize', type=int, default=128)
    parser.add_argument('--use_awq', action="store_true")
    parser.add_argument('--use_quarot', action="store_true")
    parser.add_argument('--use_kv_quant', action="store_true")

    args = parser.parse_args()
    main(args)
