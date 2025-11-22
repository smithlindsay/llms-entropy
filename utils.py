import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# utils functions for calculating entropy and codelength, and also plotting

def calc_entropy(texts, model, tokenizer, seqlen, device, batch_size=8):
    # make batches of data where each batch is tokenized and truncated to seqlen
    total_entropy = 0.0
    total_tokens = 0
    pos_sum = None
    pos_count = None
    bs = batch_size
    # add in a drop last
    texts = texts[:len(texts) - (len(texts) % bs)]
    for i in (pbar := tqdm(range(0, len(texts), bs))):
        text = texts[i:i+bs]
        batch = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=seqlen)
        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            target_ids = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            logits, _ = model(input_ids, target_ids) # BxTxV
            attn_mask = batch['attention_mask'][:, 1:].to(device) # drop BOS/pad slot
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log2(probs.clamp_min(1e-10))

            entropy = -torch.sum(probs * log_probs, dim=-1) # BxT
            entropy = entropy * attn_mask  # mask out padding tokens
            total_entropy += entropy.sum().item()
            total_tokens += attn_mask.sum().item()

            batch_sum = entropy.sum(dim=0)
            batch_count = attn_mask.sum(dim=0)
            if pos_sum is None or pos_sum.shape[0] < batch_sum.shape[0]:
                new_len = batch_sum.shape[0]
                pos_sum = torch.zeros(new_len)
                pos_count = torch.zeros(new_len)
            pos_sum[:batch_sum.shape[0]] += batch_sum.cpu().detach()
            pos_count[:batch_count.shape[0]] += batch_count.cpu().detach()

        dataset_avg_entropy = total_entropy / total_tokens
        # per_position_avg = pos_sum / pos_count.clamp_min(1)
        pbar.set_description(f"avg entropy (bits): {dataset_avg_entropy:.4f}")
    
    return total_entropy, total_tokens, pos_sum, pos_count

# codelength: -log2(p(t_n+1 | t_1,...,t_n))
# get the codelengths at each token for each sequence in a batch of text
# avg the codelengths over all sequences at each token position
def calc_codelength(texts, model, tokenizer, seqlen, device, batch_size=8):
    total_codelengths = 0.0
    total_tokens = 0
    pos_sum = None
    pos_count = None
    bs = batch_size
    texts = texts[:len(texts) - (len(texts) % bs)]
    for i in (pbar := tqdm(range(0, len(texts), bs))):
        text = texts[i:i+bs]
        batch = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=seqlen)
        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            target_ids = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            logits, _ = model(input_ids, target_ids) # BxTxV
            probs = torch.softmax(logits, dim=-1)
            attn_mask = batch['attention_mask'][:, 1:].to(device)  # drop BOS/pad slot

            # Compute codelengths
            target_probs = probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # BxT
            codelengths = -torch.log2(target_probs + 1e-10)  # BxT
            codelengths = codelengths * attn_mask  # mask out padding tokens

            # sum the codelengths over batch to get total codelength per position to then avg over the codelength counts at each position
            total_codelengths += codelengths.sum().item()
            total_tokens += attn_mask.sum().item()

            batch_sum = codelengths.sum(dim=0)
            batch_count = attn_mask.sum(dim=0)
            if pos_sum is None or pos_sum.shape[0] < batch_sum.shape[0]:
                new_len = batch_sum.shape[0]
                pos_sum = torch.zeros(new_len)
                pos_count = torch.zeros(new_len)
            pos_sum[:batch_sum.shape[0]] += batch_sum.cpu().detach()
            pos_count[:batch_count.shape[0]] += batch_count.cpu().detach()
        
        dataset_avg_codelength = total_codelengths / total_tokens
        # per_position_avg = pos_sum / pos_count.clamp_min(1)
        pbar.set_description(f"avg codelength (bits): {dataset_avg_codelength:.4f}")

    return total_codelengths, total_tokens, pos_sum, pos_count

# plot the position-wise average codelength and entropy
def plot_ent_and_codelen(per_position_avg_ent, per_position_avg_codelen, logscale=True, save_path=None):
    # make sure that per_position_avg is already numpy!!
    plt.plot(np.arange(1, len(per_position_avg_codelen)+1), per_position_avg_codelen, marker='.', label='Codelength', color='red')
    plt.plot(np.arange(1, len(per_position_avg_ent)+1), per_position_avg_ent, marker='.', label='Entropy', color='black')
    plt.title('Position-wise Average Codelength and Entropy')
    plt.xlabel('Token Position')
    plt.ylabel('Average (bits)')
    if logscale:
        plt.yscale('log')
        plt.xscale('log')
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
