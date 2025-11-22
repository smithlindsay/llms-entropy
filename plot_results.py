import numpy as np
import torch
import utils
import argparse

# args for ent, codelen files, num entries
parser = argparse.ArgumentParser()
parser.add_argument('--num_entries', type=int, default=10880, help='number of text entries used for calculation')
parser.add_argument('--ent_file', type=str, default='ent_and_codelen_results/ent_results.npy', help='path to entropy results file')
parser.add_argument('--codelen_file', type=str, default='ent_and_codelen_results/codelen_results.npy', help='path to codelength results file')
parser.add_argument('--run_name', type=str, default='', help='name for this run, used in saved plot filenames')
args = parser.parse_args()

entries = args.num_entries
ent_file = args.ent_file
codelen_file = args.codelen_file
run_name = args.run_name

# load data to plot
# results_ent = np.load('ent_and_codelen_results/ent_results.npy', allow_pickle=True).item()
# results_codelen = np.load('ent_and_codelen_results/codelen_results.npy', allow_pickle=True).item()
results_ent = np.load(ent_file, allow_pickle=True).item()
results_codelen = np.load(codelen_file, allow_pickle=True).item()

# compute position-wise average entropy and codelength
per_position_avg_ent = torch.tensor(results_ent['pos_sum_ent']) / torch.tensor(results_ent['pos_count_ent'])
per_position_avg_codelen = torch.tensor(results_codelen['pos_sum_codelen']) / torch.tensor(results_codelen['pos_count_codelen'])
per_position_avg_ent = per_position_avg_ent.numpy()
per_position_avg_codelen = per_position_avg_codelen.numpy()

# entries = 10880

# plot the position-wise average codelength and entropy
utils.plot_ent_and_codelen(per_position_avg_ent, per_position_avg_codelen, logscale=True, save_path=f'ent_and_codelen_plot_{entries}entries_log_{run_name}.pdf')
utils.plot_ent_and_codelen(per_position_avg_ent, per_position_avg_codelen, logscale=False, save_path=f'ent_and_codelen_plot_{entries}entries_{run_name}.pdf')

# plot just the first 200 positions
utils.plot_ent_and_codelen(per_position_avg_ent[:200], per_position_avg_codelen[:200], logscale=True, save_path=f'ent_and_codelen_plot_{entries}entries_first200_log_{run_name}.pdf')
utils.plot_ent_and_codelen(per_position_avg_ent[:200], per_position_avg_codelen[:200], logscale=False, save_path=f'ent_and_codelen_plot_{entries}entries_first200_{run_name}.pdf')