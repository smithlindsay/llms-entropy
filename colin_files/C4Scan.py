from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np
import glob as glob
import compute_combined as compute

import json
import argparse

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

#load the model
if args.revision=='skip':
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda')  
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda',revision=args.revision)

tokenizer = AutoTokenizer.from_pretrained(args.model,device_map='cuda')

tokenizer.pad_token=tokenizer.eos_token

#load the text data

files=['/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00217-of-01024.json',
       '/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00023-of-01024.json',
       '/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00345-of-01024.json']

data=[]
for file in files:
    with open(file,'r') as f:
        data+=[json.loads(l) for l in f]


btext=[d['text'] for d in data]

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
