import numpy as np
import torch
import os
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F


safeset_path = '/scratch/gpfs/WBIALEK/ls1546/llms-entropy/colin_files/safeset2.txt'
with open(safeset_path,'rb') as f:
    safeset=pkl.load(f)

def filterize(textlist , safeset=safeset):
    filtered=[]
    for text in textlist:
        if text:
            if len(set(text).difference(safeset)) == 0:
                filtered.append(text)

    return filtered

def computer(savedir,shapecut,dataloader,batchsize,model, tokenizer, device, printevery=10):

    if not os.path.exists(savedir): #potentially redundant
        os.makedirs(savedir)

    for subdir in ['attn','tok','entropy_t','entropy_c','codelength_t','codelength_c','tlens','pleak']:
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

    with torch.no_grad():
        print("length dataloader: ", len(dataloader))
        for i, d in tqdm(enumerate(dataloader)): 
            inputs = tokenizer(d, return_tensors='pt', return_token_type_ids=False, padding='max_length', max_length=shapecut,truncation=True).to(device)  
            input_ids = inputs['input_ids']
            targets = input_ids[:, 1:].contiguous()
            model_in = input_ids[:, :-1].contiguous()
            logits, _ = model(model_in, targets)
            batch_cur, seq_len = logits.shape[:2]
            
            attn=inputs['attention_mask'][:, 1:].cpu().numpy()
            tok=targets.cpu().numpy()

            parr=F.softmax(logits,dim=-1)
            parr=parr.to(smat.dtype)
            
            #token level
            ent_t=-torch.sum(parr*torch.log2(parr+1e-10),axis=-1)
            pselect=parr.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            codelength_t=-torch.log2(pselect+1e-10)

            #character level 
            preprobs=smat.mm(parr.permute(2,1,0).reshape(ncols,-1)).view(nrows,seq_len,batch_cur).permute(2,1,0) 
            probs=(preprobs/preprobs.sum(axis=-1,keepdim=True))

            pleak=preprobs.sum(axis=-1)
            ent_c=-torch.sum(probs*torch.log2(probs+1e-10),axis=-1)
            
            charnext=lookup[tok]
            charnext_t=torch.as_tensor(charnext, device=probs.device)
            pselect=probs.gather(-1, charnext_t.unsqueeze(-1)).squeeze(-1)
            codelength_c=-torch.log2(pselect+1e-10) 
            # drop the final step to align with downstream averaging masks
            codelength_t = codelength_t[:, :-1]
            codelength_c = codelength_c[:, :-1]
            
            tlens=lengths[tok]

            #do next documents... 
            aname=f'{savedir}attn/{i:07}.npy'
            tname=f'{savedir}tok/{i:07}.npy'
            ename_t=f'{savedir}entropy_t/{i:07}.npy'
            cname_t=f'{savedir}codelength_t/{i:07}.npy'
            ename_c=f'{savedir}entropy_c/{i:07}.npy'
            cname_c=f'{savedir}codelength_c/{i:07}.npy'
            lname=f'{savedir}tlens/{i:07}.npy'
            pname=f'{savedir}pleak/{i:07}.npy'

            #save all data
            for fname,tensor in zip([aname,ename_t,ename_c,tname,cname_t, cname_c, lname, pname],[attn,ent_t.cpu().numpy(), ent_c.cpu().numpy(),tok,codelength_t.cpu().numpy(), codelength_c.cpu().numpy(), tlens, pleak.cpu().numpy()]): #need to add oname 
                np.save(fname, tensor) #modified!
