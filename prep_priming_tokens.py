"""
Materialize priming tokens for synthetic-data generation.

Reads the first token of each sample from a DCLM tokshuf shard and writes
them as a single (N, 1) int64 .npy file, which `generate_synth_data.py` then
slices across array tasks.

CPU only. Typical run is fast (seconds to a few minutes for N=10000).

Usage:
    python3 prep_priming_tokens.py \
        --shard 1 --n-seqs 10000 \
        --out synth_data/priming_tokens_shard1_n10000.npy
"""

import argparse
import glob
import itertools
import os

import numpy as np
import torch
import webdataset as wds


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--n-seqs", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--data-base",
        type=str,
        default="/scratch/gpfs/WBIALEK/ls1546/DCLM/rust_processing/tokshuf-rs",
    )
    return p.parse_args()


def first_token(sample):
    tokens = torch.tensor(sample["json.gz"], dtype=torch.long)
    return tokens[:1]


def main():
    args = parse_args()

    shard_dir = os.path.join(args.data_base, f"dclm_tokshuf_{args.shard}")
    urls = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    print(f"[prep] found {len(urls)} tar files in shard {args.shard}", flush=True)
    if not urls:
        raise FileNotFoundError(f"no .tar files in {shard_dir}")

    dataset = (
        wds.WebDataset(
            urls,
            resampled=False,
            shardshuffle=False,
            empty_check=False,
        )
        .decode()
        .map(first_token)
    )

    samples = list(itertools.islice(dataset, args.n_seqs))
    if len(samples) < args.n_seqs:
        raise RuntimeError(
            f"[prep] only got {len(samples)} samples from shard {args.shard}, "
            f"wanted {args.n_seqs}"
        )

    priming = torch.stack(samples).numpy().astype(np.int64)  # (N, 1)
    assert priming.shape == (args.n_seqs, 1), priming.shape

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # `np.save(path, ...)` silently appends ".npy" if the path doesn't already
    # end in ".npy", so use an explicit file handle to keep our .tmp name intact
    # and then atomically rename.
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, priming)
    os.replace(tmp, args.out)
    print(f"[prep] saved {priming.shape} priming tokens to {args.out}", flush=True)


if __name__ == "__main__":
    main()
