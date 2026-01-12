import utils
import torch
import torch.nn.functional as F
import numpy as np
import glob as glob
import orjson 
from tqdm import tqdm
import pickle as pkl
import argparse
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

parser= argparse.ArgumentParser(description='analyze entropy')
parser.add_argument('--savedir',type=str)
parser.add_argument('--model',type=str)
parser.add_argument('--checkpoint',type=int,default=47000)
parser.add_argument('--buffer',type=int)
parser.add_argument('--maxchars',type=int) #note change
parser.add_argument('--seed', type=int,default=223291)
parser.add_argument('--saveevery', type=int,default=10) #note change
parser.add_argument('--printevery', type=int,default=10) #note change
parser.add_argument('--shapecut',type=int)
parser.add_argument('--lcut',type=int)
parser.add_argument('--group_size', type=int)
parser.add_argument('--group_start', type=int)
parser.add_argument('--file_start', type=int)
parser.add_argument('--file_number', type=int)
args=parser.parse_args()

checkpoint = args.checkpoint
# load my model
model, tokenizer, seqlen = utils.load_dclm_model(device, checkpoint=checkpoint)

#load the safeset 
safeset_path = '/scratch/gpfs/WBIALEK/ls1546/llms-entropy/colin_files/safeset2.txt'
with open(safeset_path,'rb') as f:
    safeset=pkl.load(f)

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
shapecut = seqlen
indx1=np.repeat([np.arange(1)],shapecut-1,axis=0).T
indx2=np.repeat([np.arange(shapecut-1)],1,axis=0)

rng=np.random.default_rng(args.seed)

alist=[]
elist=[]
tlist=[]
clist=[]

# pnum=0
snum=0
saveevery=args.saveevery
printevery=args.printevery

for g,f in zip(group_ids,fnums):
    files=[f'/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.{i:05}-of-01024.json' for i in f]

    print(f'group_id {g}', flush=True)

    groupdir=savedir+f'group_{g:05}/'
    if not os.path.exists(groupdir):
        os.makedirs(groupdir)

    with open(groupdir+'filelist.txt','w') as fl:
        fl.write('\n'.join(files))

    for subdir in ['attn','tok','entropy_t','codelength_t']:
        directory=groupdir+subdir
        if not os.path.exists(directory):
            os.makedirs(directory)


    for file in tqdm(files):
        with open(file,'r') as input:
            for line in input:
                start_time = time.time()
                text=orjson.loads(line).get('text')
                if text and len(text) > args.lcut and set(text)<= safeset: #put a minimum length just to save time.... 
                    start=rng.choice(args.buffer)                      #stagger the start with a random number
                    text=text[start:start+args.maxchars]               #clip items above a maximum length ~10**5
                    with torch.no_grad(): 
                        inputs = tokenizer(text, return_tensors='pt', return_token_type_ids=False, max_length=shapecut,truncation=True).to(device)
                        input_ids = inputs['input_ids']
                        targets = input_ids[:, 1:].contiguous()
                        model_in = input_ids[:, :-1].contiguous()
                        logits, _ = model(model_in, targets)
                        batch_cur, seq_len = logits.shape[:2]

                        attn = inputs['attention_mask'][:, 1:].cpu().numpy()
                        tok = targets.cpu().numpy()
                        print(logits.shape, tok.shape)
                        if tok.shape[1] == shapecut - 1:
                            parr=F.softmax(logits,dim=-1)

                            # pselect=parr.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                            
                            ent_t=-torch.sum(parr*torch.log2(parr+1e-10),axis=-1).cpu().numpy()
                            pselect=parr.cpu()[indx1,indx2,tok] 
                            codelength_t=-np.log2(pselect).numpy()
                            # drop the final step to align with downstream averaging masks
                            codelength_t = codelength_t[:, :-1]

                            alist.append(attn)
                            elist.append(ent_t)
                            tlist.append(tok)
                            clist.append(codelength_t)
                        
                            snum+=1
                            print("one iter: ", time.time() - start_time)
                            if (snum % saveevery)==0:
                                number=snum//saveevery
                                aname=f'{groupdir}attn/{number:07}.npy'
                                tname=f'{groupdir}tok/{number:07}.npy'
                                ename_t=f'{groupdir}entropy_t/{number:07}.npy'
                                cname_t=f'{groupdir}codelength_t/{number:07}.npy'

                                for fname,lst in zip([aname,ename_t,tname,cname_t],[alist,elist,tlist,clist]):  
                                    tensor=np.vstack(lst)
                                    np.save(fname, tensor) 
                                    lst.clear()

                            if (snum % printevery) ==0:
                                print(snum,flush=True)
