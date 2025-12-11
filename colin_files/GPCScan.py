from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import string
import glob as glob
import datasets
import compute_combined as compute

import gzip
import json
import pickle as pkl
import argparse

import collections as collect
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

parser= argparse.ArgumentParser(description='analyze entropy')
parser.add_argument('--savedir',type=str)
parser.add_argument('--model',type=str)
parser.add_argument('--revision',type=str,default='skip')
parser.add_argument('--buffer',type=int)
parser.add_argument('--splitlen',type=int)
parser.add_argument('--nsamples',type=int)
parser.add_argument('--seed', type=int,default=223291)
parser.add_argument('--batchsize', type=int,default=2)
parser.add_argument('--shapecut',type=int)
args=parser.parse_args()



gpc=datasets.load_dataset("biglam/gutenberg-poetry-corpus")

#load the model
if args.revision=='skip':
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda')  
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda',revision=args.revision)

tokenizer = AutoTokenizer.from_pretrained(args.model,device_map='cuda')

#load the text data

tokenizer.pad_token = tokenizer.eos_token

btext=[]
cid=gpc['train'][0]['gutenberg_id']
ctext=[]
for ln,gid in zip(gpc['train']['line'],gpc['train']['gutenberg_id']):
    if gid==cid:
        ctext.append(ln)
    else:
        cid=gid
        btext.append("\n".join(ctext))
        ctext=[ln]

#merve=datasets.load_dataset("merve/poetry")


ftext=compute.filterize(btext)

splitlen=args.splitlen
ftext_elongated=[]
for t in ftext:
    ftext_elongated+=[t[i:i+splitlen] for i in np.arange(0,len(t),splitlen)]

lengths=np.array([len(t) for t in ftext_elongated])
indsort=np.flip(np.argsort(lengths))
nsamples=args.nsamples
buffer=args.buffer
rng=np.random.default_rng(args.seed)
starts=rng.choice(buffer,size=nsamples)
ftext_sorted=[ftext_elongated[i][s:] for i,s in zip(indsort[:nsamples],starts)]


#perform the analysis 
batchsize=args.batchsize
dataloader=DataLoader(ftext_sorted,batch_size=batchsize)

compute.computer(savedir=args.savedir,
                 shapecut=args.shapecut,
                 dataloader=dataloader,
                 batchsize=batchsize, 
                 model=model, 
                 tokenizer=tokenizer,
                 device=device, 
                 printevery=10)




#'/scratch/gpfs/WINGREEN/cs5096/learning/EntropyLLM/experiments/checkpointtest/'
