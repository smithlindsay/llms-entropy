import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from pathlib import Path
import sys

project_root = Path.cwd().parent  # adjust if llms-entropy isn’t directly under the project root
sys.path.append(str(project_root / "gpt-circuits"))

from model import GPT, GPTConfig
import utils


# calculate entropy and codelengths over multiple files from C4 dataset
out_dir = '/scratch/gpfs/WBIALEK/ls1546/gpt-circuits/out/shard1_m1337_d1337/'
model_ckpt = f'{out_dir}1756086913.6403742_ckpt_47000.pt'
init_from = 'scratch'

# model
dropout = 0.0
bias = False
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
block_size = 2048
n_layer = 24
n_head = 16
n_embd = 2048
z_loss = 1e-4

# load model checkpoint and see if inference output is reasonable
checkpoint = torch.load(model_ckpt, map_location=device)
checkpoint_model_args = checkpoint['model_args']
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50277, dropout=dropout, z_loss=z_loss) # start with model_args from command line

for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

tok_model = "EleutherAI/gpt-neox-20b"
seqlen=2048
tokenizer = AutoTokenizer.from_pretrained(tok_model, use_fast=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
model.load_state_dict(state_dict)
model.to(device)
model.eval();

# read in a json file of text data
c4path = '/scratch/gpfs/DATASETS/hugging_face/c4/en'
files = ['00000-of-01024']  # add more files as needed
texts = []
for file_id in files:
    dataset = load_dataset('json', data_files=f'{c4path}/c4-train.{file_id}.json', split='train')
    texts.extend(dataset['text'])
    num_entries = len(dataset['text'])
    print(f'loaded file {file_id} with {num_entries}, {len(texts)} entries total')

# calculate entropy
# just do the first 680 batches which takes 1hr
# 680*16=10880 entries
texts = texts[:10880]
total_entropy, total_tokens, pos_sum_ent, pos_count_ent = utils.calc_entropy(
    texts, model, tokenizer, seqlen, device, batch_size=16)
dataset_avg_entropy = total_entropy / total_tokens
print(f'Dataset Average Entropy (bits/token): {dataset_avg_entropy:.4f}')
per_position_avg_ent = pos_sum_ent / pos_count_ent.clamp_min(1)
print(f'Per-position avg entropy shape: {per_position_avg_ent.shape}')

# save results
results_ent = {}
results_ent['total_entropy'] = total_entropy
results_ent['total_tokens'] = total_tokens
results_ent['pos_sum_ent'] = pos_sum_ent.numpy()
results_ent['pos_count_ent'] = pos_count_ent.numpy()
np.save('ent_and_codelen_results/ent_results.npy', results_ent)

