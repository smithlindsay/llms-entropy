# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np
import glob as glob
import datasets
import compute
import argparse
import utils

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
parser.add_argument('--begin_year',type=int,default=2023) #dclm 1B data goes to 2022
args=parser.parse_args()

begin_year = args.begin_year

# #load the model
# if args.revision=='skip':
#     model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda')  
# else:
#     model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda',revision=args.revision)

# tokenizer = AutoTokenizer.from_pretrained(args.model,device_map='cuda')

# tokenizer.pad_token = tokenizer.eos_token

# load my model
model, tokenizer, seqlen = utils.load_dclm_model(device)

#load the text data

ds = []
for year in range(begin_year, 2025):
    for month in range(1, 13):
        month_str = f'{month:02d}'
        ds.append(datasets.load_dataset('RealTimeData/bbc_news_alltime', f'{year}-{month_str}'))


for month in range(1, 7):
    month_str = f'{month:02d}'
    ds.append(datasets.load_dataset('RealTimeData/bbc_news_alltime', f'2025-{month_str}'))


texts = [ds[i]['train']['content'] for i in range(len(ds))]
textflat=[]
for tl in texts: textflat+=tl

ftext=compute.filterize(textflat)

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
                 device=device)




#'/scratch/gpfs/WINGREEN/cs5096/learning/EntropyLLM/experiments/checkpointtest/'
