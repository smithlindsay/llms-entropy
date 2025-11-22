import numpy as np
import torch
import os
import pickle as pkl
import torch.nn.functional as F


with open("/home/cs5096/Learning/EntropyLLM/Tech/safeset2.txt",'rb') as f:
    safeset=pkl.load(f)

def filterize(textlist , safeset=safeset):
    filtered=[]
    for i,text in enumerate(textlist):
        if text:
            if len(set(text).difference(safeset)) ==0:
                filtered.append(text)

    return filtered

def computer(savedir,shapecut,dataloader,batchsize,model, tokenizer, device, printevery=1):

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for subdir in ['attn','tok','entropy','codelength','tlens','pleak']:
        directory=savedir+subdir
        if not os.path.exists(directory):
            os.makedirs(directory)

    smallin=tokenizer(['a'],return_tensors='pt',return_token_type_ids=False).to(device)
    smallout=model(**smallin)

    safelist=sorted(safeset)  

    nrows=len(safelist)
    ncols=smallout['logits'].shape[-1]

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

    indx1=np.repeat([np.arange(batchsize)],shapecut-1,axis=0).T
    indx2=np.repeat([np.arange(shapecut-1)],batchsize,axis=0)

    with torch.no_grad():
        i=0
        for d in dataloader: 
            inputs = tokenizer(d, return_tensors='pt', return_token_type_ids=False, padding='max_length', max_length=shapecut,truncation=True).to(device)  
            outputs=model(**inputs)
            
            attn=inputs['attention_mask'].cpu().numpy()
            tok=inputs['input_ids'].cpu().numpy()

            logits=outputs.logits
            parr=F.softmax(logits,dim=-1)

            preprobs=smat.mm(parr.permute(2,1,0).reshape(ncols,-1)).view(nrows,parr.shape[1],parr.shape[0]).permute(2,1,0) #break apart, use view vs reshape
            probs=(preprobs/preprobs.sum(axis=-1,keepdim=True)).cpu()

            pleak=preprobs.sum(axis=-1).cpu().numpy()
            ent=-torch.sum(probs*torch.log2(probs+1e-10),axis=-1).numpy()
            
            charnext=lookup[tok]
            pselect=probs[indx1,indx2,charnext[:,1:]]
            codelength=-np.log2(pselect).numpy() 
            
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
                with open(fname,'wb') as f:
                    np.save(fname, tensor)
            if (i % printevery) ==0:
                print(i)
            i+=1




