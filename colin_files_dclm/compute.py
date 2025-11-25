import numpy as np
import torch
import os
import pickle as pkl
import torch.nn.functional as F
from tqdm import tqdm

safeset_path = '/scratch/gpfs/WBIALEK/ls1546/llms-entropy/colin_files/safeset2.txt'
with open(safeset_path,'rb') as f:
    safeset=pkl.load(f)

def filterize(textlist , safeset=safeset):
    filtered=[]
    for text in tqdm(textlist):
        if text:
            if len(set(text).difference(safeset)) == 0:
                filtered.append(text)

    return filtered

def computer(savedir,shapecut,dataloader,batchsize,model, tokenizer, device):

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for subdir in ['attn','tok','entropy','codelength','tlens','pleak']:
        directory=savedir+subdir
        if not os.path.exists(directory):
            os.makedirs(directory)

    safelist=sorted(safeset)  

    nrows=len(safelist)
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        ncols=model.config.vocab_size
    elif hasattr(model, "lm_head"):
        ncols=model.lm_head.out_features
    else:
        ncols=len(tokenizer)

    rinds=[]
    cinds=[]
    lens=[]
    for i in np.arange(len(tokenizer)):
        string=tokenizer.decode(i, clean_up_tokenization_spaces=False)
        if set(string).issubset(safeset):
            lens.append(len(string))
            cinds.append(i)
            rinds.append(safelist.index(string[0]))
    
    lengths=np.zeros(len(tokenizer),dtype=np.int64)
    lengths[cinds]=lens
    lookup=np.zeros(len(tokenizer),dtype=np.int64)
    lookup[cinds]=rinds

    smat=torch.sparse_coo_tensor(indices=(rinds,cinds),values=np.ones(len(rinds)),size=(nrows,ncols),dtype=torch.float).to(device)

    print("len dataloader:", len(dataloader))
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader)): 
            inputs = tokenizer(d, return_tensors='pt', return_token_type_ids=False, padding='max_length', max_length=shapecut,truncation=True).to(device)  
            input_ids = inputs['input_ids']
            targets = input_ids[:, 1:].contiguous()
            model_in = input_ids[:, :-1].contiguous()
            logits, _ = model(model_in, targets)
            batch_cur, seq_len = logits.shape[:2]
            
            attn=inputs['attention_mask'][:, 1:].cpu().numpy()
            tok=targets.cpu().numpy()

            # logits=outputs.logits
            parr=F.softmax(logits,dim=-1)

            preprobs=smat.mm(parr.permute(2,1,0).reshape(ncols,-1)).view(nrows,seq_len,batch_cur).permute(2,1,0) #break apart, use view vs reshape
            probs=(preprobs/preprobs.sum(axis=-1,keepdim=True)).cpu()

            pleak=preprobs.sum(axis=-1).cpu().numpy()
            ent=-torch.sum(probs*torch.log2(probs+1e-10),axis=-1).numpy()
            
            charnext=lookup[tok]
            probs_np=probs.numpy()
            batch_index=np.arange(batch_cur)[:,None]
            time_index=np.arange(seq_len)[None,:]
            pselect=probs_np[batch_index,time_index,charnext]
            codelength=-np.log2(pselect) 
            
            tlens=lengths[tok]

            #do next documents... 
            aname=f'{savedir}attn/{i:07}.npy'
            tname=f'{savedir}tok/{i:07}.npy'
            ename=f'{savedir}entropy/{i:07}.npy'
            cname=f'{savedir}codelength/{i:07}.npy'
            lname=f'{savedir}tlens/{i:07}.npy'
            pname=f'{savedir}pleak/{i:07}.npy'

            #save all data
            for fname,tensor in zip([aname,ename,tname,cname, lname, pname],[attn,ent,tok,codelength, tlens, pleak]):
                np.save(fname, tensor)
            



