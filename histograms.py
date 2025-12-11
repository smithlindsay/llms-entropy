import numpy as np
import matplotlib.pyplot as plt
import json
import pickle as pkl
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch

with open("safeset2.txt",'rb') as f:
    safeset=pkl.load(f)

def filterize(textlist, safeset=safeset):
    filtered=[]
    for text in textlist:
        # only keep texts with characters in safeset
        if text and len(set(text).difference(safeset)) == 0:
            filtered.append(text)

    return filtered


# run_type = 'c4'
# files=['/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00217-of-01024.json',
#        '/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00023-of-01024.json',
#        '/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00345-of-01024.json']
# data=[]
# for file in files:
#     with open(file,'r') as f:
#         data+=[json.loads(l) for l in f]
# btext=[d['text'] for d in data]

# run_type = 'poetry'
# ptry = datasets.load_dataset("suayptalha/Poetry-Foundation-Poems")
# btext = [x["Poem"] for x in ptry['train']]

run_type = 'bbc'
# use only 2025 data (think knowledge cutoff 2024)
ds = []
for month in range(1, 7):
    month_str = f'{month:02d}'
    ds.append(datasets.load_dataset('RealTimeData/bbc_news_alltime', f'2025-{month_str}'))

texts = [ds[i]['train']['content'] for i in range(len(ds))]
btext = []
for tl in texts: 
    btext += tl

ftext = filterize(btext, safeset=safeset)

splitlen = 15000
ftext_elongated = []
for t in ftext:
    ftext_elongated+=[t[i:i+splitlen] for i in np.arange(0, len(t), splitlen)]

lengths = np.array([len(t) for t in ftext_elongated])
indsort = np.flip(np.argsort(lengths))
nsamples = 2000
buffer = 100
rng = np.random.default_rng(223291)
starts = rng.choice(buffer, size=nsamples)
ftext_sorted = [ftext_elongated[i][s:] for i,s in zip(indsort[:nsamples], starts)]

#perform the analysis 
batchsize = 10
dataloader = DataLoader(ftext_sorted, batch_size=batchsize)

model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-2-0425-1B', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-2-0425-1B', device_map='auto')
tokenizer.pad_token = tokenizer.eos_token

shapecut = 2000
device = model.device

model.eval()
entropies = []
predicted_tokens = []
attention_masks = []
for d in tqdm(dataloader): 
    with torch.no_grad():
        inputs = tokenizer(d, return_tensors='pt', return_token_type_ids=False, padding='max_length', max_length=shapecut, truncation=True).to(device)  
        outputs = model(**inputs)
        # print('calc entropies')
        
        attn = inputs['attention_mask']

        logits = outputs.logits * attn[:, :, None]  # zero out logits where attention mask is 0
        probs = F.softmax(logits, dim=-1)
    
    #token level entropy for each token given the previous context
    ent_t = -torch.sum(probs * torch.log2(probs+1e-10), axis=-1).cpu().numpy()

    # pick the top-1 token as predicted token for each position to then attach to each entropy value at that position
    pred_toks = torch.argmax(probs, dim=-1).cpu().numpy()

    # store entropies, predicted tokens, and attn mask for each batch
    for b in range(ent_t.shape[0]):
        entropies.append(ent_t[b, :])
        predicted_tokens.append(pred_toks[b, :])
        attention_masks.append(attn[b, :].cpu().numpy())
entropies = np.array(entropies)
predicted_tokens = np.array(predicted_tokens)
attention_masks = np.array(attention_masks)

print(entropies.shape, predicted_tokens.shape, attention_masks.shape)

# save entropies, predicted tokens, attention masks
np.save(f'token_entropies_{run_type}.npy', entropies)
np.save(f'predicted_tokens_{run_type}.npy', predicted_tokens)
np.save(f'attention_masks_{run_type}.npy', attention_masks)

# plot histograms of entropies
# subplots where each subplot is a histogram of entropies for a specific token position (1, 3, 40, 100, 1000, 2000)
fig, axs = plt.subplots(6, 1, figsize=(8, 20))
positions = [1, 3, 40, 100, 1000, 1250]
for i, pos in enumerate(positions):
    print('number of valid tokens at position', pos, ':', np.sum(attention_masks[:, pos-1] == 1))
    axs[i].hist(entropies[attention_masks[:, pos-1] == 1, pos-1], density=True, bins=50, color='black')
    axs[i].axvline(np.mean(entropies[attention_masks[:, pos-1] == 1, pos-1]), c='gray', zorder=-1, ls='--')
    axs[i].set_title(f'Entropy Distribution for Token Position {pos}')
    axs[i].set_xlabel('Entropy')
    axs[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'entropy_histograms_{run_type}.png')

# for these context lengths, find the tokens with zero entropy and print their text and counts, find the unique tokens and their counts

token_counts = {}
for i in range(entropies.shape[0]):
    for j in range(entropies.shape[1]):
        if attention_masks[i, j] == 1 and entropies[i, j] < 0.01:
            token_id = predicted_tokens[i, j]
            token_text = tokenizer.decode(token_id, clean_up_tokenization_spaces=False)
            if token_text in token_counts:
                token_counts[token_text] += 1
            else:
                token_counts[token_text] = 1

# for token_text, count in token_counts.items():
#     print(f'Token Text: "{token_text}", Count: {count}')
# sort by count descending
sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
for token_text, count in sorted_token_counts:
    print(f'Token Text: "{token_text}", Count: {count}')

# print tokes with 0 entropy grouped by position
# make bins where each bin is a list of tokens with 0 entropy at that position or earlier
zero_entropy_bins = {}
for i in range(entropies.shape[0]):
    for j in range(entropies.shape[1]):
        if attention_masks[i, j] == 1 and entropies[i, j] < 0.01:
            token_id = predicted_tokens[i, j]
            token_text = tokenizer.decode(token_id, clean_up_tokenization_spaces=False)
            if j not in zero_entropy_bins:
                zero_entropy_bins[j] = set()
            zero_entropy_bins[j].add(token_text)

# print the tokens with zero entropy at positions 1, 3, 40, 100, 1000, 1250
for pos in [1, 3, 40, 100, 1000, 1250]:
    tokens_at_pos = set()
    for i in range(1, pos):
        tokens_at_pos.update(zero_entropy_bins.get(i, set()))
    print(f'Position {pos}:')
    print(f'Token Text: {tokens_at_pos}')

# look at the highest entropy tokens at each of these positions
for pos in [1, 3, 40, 100, 1000, 1250]:
    entropies_at_pos = []
    token_texts_at_pos = []
    for i in range(entropies.shape[0]):
        if attention_masks[i, pos-1] == 1:
            entropies_at_pos.append(entropies[i, pos-1])
            token_id = predicted_tokens[i, pos-1]
            token_text = tokenizer.decode(token_id, clean_up_tokenization_spaces=False)
            token_texts_at_pos.append(token_text)
    # get indices of top 5 highest entropy tokens
    top_indices = np.argsort(entropies_at_pos)[-5:]
    print(f'Position {pos}:')
    for idx in top_indices:
        print(f'Token Text: "{token_texts_at_pos[idx]}", Entropy: {entropies_at_pos[idx]}')