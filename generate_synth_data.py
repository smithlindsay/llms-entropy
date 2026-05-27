"""
Generate synthetic data autoregressively from a trained GPT checkpoint.

Designed to be run as a Slurm job array. The work is split into two subcommands:

    1) generate  Per-array-task: load the priming file, take this task's slice,
                 generate at each temperature, save one .npy per (T, task_id)
                 into a parts directory.
    2) merge     After the array job finishes, concatenate the per-task .npy
                 files into final synth_ids_T<T>.npy files.

Priming tokens are materialized separately by `prep_priming_tokens.py` and
passed in via --priming-file.

Typical usage on the cluster:

    # one-time, fast, CPU OK
    python3 prep_priming_tokens.py \
        --shard 1 --n-seqs 10000 \
        --out synth_data/priming_tokens_shard1_n10000.npy

    # array job (see slurmfiles/generate_synth_data.slurm)
    sbatch slurmfiles/generate_synth_data.slurm

    # after the array finishes
    python3 generate_synth_data.py merge \
        --parts-dir synth_data/parts \
        --out-dir   synth_data \
        --total-tasks 100 \
        --temperatures 0.5 0.8 1.0 1.2 1.5 \
        --cleanup
"""

import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- generate ----------------------------------------------------------
    p_gen = sub.add_parser("generate", help="per-task autoregressive generation")
    p_gen.add_argument("--task-id", type=int, required=True)
    p_gen.add_argument("--total-tasks", type=int, required=True)
    p_gen.add_argument("--priming-file", type=str, required=True)
    p_gen.add_argument("--out-dir", type=str, required=True)
    p_gen.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.0, 1.2, 1.5],
    )
    p_gen.add_argument("--batch-size", type=int, default=8)
    p_gen.add_argument(
        "--ckpt-base",
        type=str,
        default="/scratch/gpfs/WBIALEK/ls1546/gpt-circuits/out",
    )
    p_gen.add_argument("--run-name", type=str, default="shard1_m1337_d1337")
    p_gen.add_argument("--ckpt-num", type=int, default=47000)
    p_gen.add_argument("--block-size", type=int, default=2048)
    p_gen.add_argument("--n-layer", type=int, default=24)
    p_gen.add_argument("--n-head", type=int, default=16)
    p_gen.add_argument("--n-embd", type=int, default=2048)
    p_gen.add_argument("--bias", action="store_true")
    p_gen.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="print a progress line every N batches",
    )

    # ---- merge -------------------------------------------------------------
    p_merge = sub.add_parser("merge", help="concatenate per-task parts into final .npy")
    p_merge.add_argument("--parts-dir", type=str, required=True)
    p_merge.add_argument("--out-dir", type=str, required=True)
    p_merge.add_argument("--total-tasks", type=int, required=True)
    p_merge.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.0, 1.2, 1.5],
    )
    p_merge.add_argument(
        "--cleanup",
        action="store_true",
        help="delete the per-task parts after a successful merge",
    )

    return p.parse_args()


# --- generate ---------------------------------------------------------------


def cmd_generate(args):
    from umt.evals.eval_utils import load_model_from_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[gen task={args.task_id}/{args.total_tasks}] device={device} "
        f"dtype={args.dtype} batch_size={args.batch_size}",
        flush=True,
    )

    seed = args.seed + args.task_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---- slice this task's priming tokens
    priming = np.load(args.priming_file)  # (N, 1)
    assert priming.ndim == 2 and priming.shape[1] == 1, priming.shape
    n_total = priming.shape[0]
    chunk = (n_total + args.total_tasks - 1) // args.total_tasks
    start = args.task_id * chunk
    end = min(start + chunk, n_total)
    if start >= n_total:
        print(
            f"[gen task={args.task_id}] no work: start={start} >= n_total={n_total}",
            flush=True,
        )
        return
    my_priming = torch.from_numpy(priming[start:end]).long()  # (M, 1)
    n_my = my_priming.shape[0]
    print(
        f"[gen task={args.task_id}] handling priming[{start}:{end}] "
        f"= {n_my} sequences",
        flush=True,
    )

    # ---- load model
    ckpt_path = os.path.join(
        args.ckpt_base, args.run_name, f"{args.run_name}_ckpt_{args.ckpt_num}.pt"
    )
    print(f"[gen task={args.task_id}] loading checkpoint {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = load_model_from_checkpoint(
        checkpoint,
        args.n_layer,
        args.n_head,
        args.n_embd,
        args.block_size,
        args.bias,
        0.0,
        1e-4,
        device,
    )
    model.eval()

    # ---- autocast ctx for generation
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    autocast_dtype = dtype_map[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if device == "cuda" and args.dtype != "float32"
        else nullcontext()
    )

    max_new_tokens = args.block_size - 1
    os.makedirs(args.out_dir, exist_ok=True)

    for T in args.temperatures:
        out_path = os.path.join(
            args.out_dir, f"synth_ids_T{T}_part{args.task_id:04d}.npy"
        )
        if os.path.exists(out_path):
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (n_my, args.block_size):
                print(
                    f"[gen task={args.task_id}] T={T}: {out_path} "
                    f"already complete {tuple(existing.shape)}, skipping",
                    flush=True,
                )
                continue
            print(
                f"[gen task={args.task_id}] T={T}: stale partial at {out_path} "
                f"({tuple(existing.shape)}), regenerating",
                flush=True,
            )

        print(f"[gen task={args.task_id}] generating T={T} ...", flush=True)

        all_synth = []
        n_batches = (n_my + args.batch_size - 1) // args.batch_size
        for bi, batch_start in enumerate(range(0, n_my, args.batch_size)):
            batch = my_priming[batch_start : batch_start + args.batch_size].to(device)
            with torch.inference_mode(), autocast_ctx:
                generated = model.generate(batch, max_new_tokens, temperature=T)
            all_synth.append(generated.to("cpu", dtype=torch.long).numpy())
            if (bi + 1) % args.log_every == 0 or (bi + 1) == n_batches:
                done = min(batch_start + args.batch_size, n_my)
                print(
                    f"[gen task={args.task_id}]   T={T}: batch {bi+1}/{n_batches} "
                    f"({done}/{n_my} sequences)",
                    flush=True,
                )

        synth = np.concatenate(all_synth, axis=0)
        assert synth.shape == (n_my, args.block_size), synth.shape
        # `np.save(path, ...)` appends ".npy" if missing, so write via a file
        # handle to preserve our .tmp suffix, then atomically rename.
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, synth)
        os.replace(tmp_path, out_path)
        print(
            f"[gen task={args.task_id}] T={T}: saved {synth.shape} -> {out_path}",
            flush=True,
        )


# --- merge ------------------------------------------------------------------


def cmd_merge(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for T in args.temperatures:
        parts = []
        missing = []
        for i in range(args.total_tasks):
            part_path = os.path.join(
                args.parts_dir, f"synth_ids_T{T}_part{i:04d}.npy"
            )
            if not os.path.exists(part_path):
                missing.append(i)
                continue
            parts.append(np.load(part_path))
        if missing:
            print(
                f"[merge] T={T}: MISSING {len(missing)} parts (task ids: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''})",
                flush=True,
            )
            if not parts:
                continue
            print(
                f"[merge] T={T}: continuing with {len(parts)} available parts",
                flush=True,
            )

        full = np.concatenate(parts, axis=0)
        out_path = os.path.join(args.out_dir, f"synth_ids_T{T}.npy")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, full)
        os.replace(tmp_path, out_path)
        print(f"[merge] T={T}: saved {full.shape} -> {out_path}", flush=True)

        if args.cleanup and not missing:
            for i in range(args.total_tasks):
                part_path = os.path.join(
                    args.parts_dir, f"synth_ids_T{T}_part{i:04d}.npy"
                )
                if os.path.exists(part_path):
                    os.remove(part_path)
            print(f"[merge] T={T}: cleaned up part files", flush=True)


# --- main -------------------------------------------------------------------


def main():
    args = parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    else:
        raise ValueError(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
