import torch
import datasets
import random
from transformers import AutoTokenizer

def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    traindata = datasets.load_dataset(
        "parquet",
        data_files={'train': "/data/wikitext/wikitext-2-v1/train-*.parquet"},
        split='train'
    )
    testdata = datasets.load_dataset(
        "parquet",
        data_files={'test': "/data/wikitext/wikitext-2-raw-v1/test-*.parquet"},
        split='test'
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    traindata = datasets.load_dataset(
        'json', 
        data_files={'train': '/data/allenai--c4/en/c4-train.00000-of-01024.json.gz'}, 
        split='train'
    )
    valdata = datasets.load_dataset(
        'json', 
        data_files={'validation': '/data/allenai--c4/en/c4-validation.00000-of-00008.json.gz'}, 
        split='validation'
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc
    

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model)
    raise ValueError("dataset [{}] not supported.".format(name))
