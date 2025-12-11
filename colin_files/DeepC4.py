from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np
import glob as glob
import compute

import json
import argparse

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
parser.add_argument('--group_size', type=int)
parser.add_argument('--group_start', type=int)
parser.add_argument('--file_start', type=int)
parser.add_argument('--file_number', type=int)
args=parser.parse_args()

#load the model
if args.revision=='skip':
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda')  
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda',revision=args.revision)

tokenizer = AutoTokenizer.from_pretrained(args.model,device_map='cuda')

tokenizer.pad_token=tokenizer.eos_token


#set up grouping
group_size=args.group_size
group_start=args.group_start
file_start=args.file_start
file_number=args.file_number

fnums=np.arange(file_number)+file_start
ngroups=file_number//group_size
fnums=fnums[:(group_size*ngroups)].reshape((ngroups,group_size))
group_ids=np.arange(ngroups)+group_start

savedir=args.savedir

if not os.path.exists(savedir):
    os.makedirs(savedir)

#group loop: 
for g,f in zip(group_ids,fnums):
    files=[f'/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.{i:05}-of-01024.json' for i in f]

    print(f'group_id {g}')

    groupdir=savedir+f'group_{g:05}/'
    if not os.path.exists(groupdir):
        os.makedirs(groupdir)

    with open(groupdir+'filelist.txt','w') as fl:
        fl.write('\n'.join(files))

    #load the text data
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

    compute.computer(savedir=groupdir,
                    shapecut=args.shapecut,
                    dataloader=dataloader,
                    batchsize=batchsize, 
                    model=model, 
                    tokenizer=tokenizer,
                    device=device, 
                    printevery=10)

