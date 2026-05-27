"""
Compute per-token code lengths for merged synthetic token-id arrays.

Designed for Slurm: one array task per sampling temperature (default 5 tasks).
Each task loads synth_ids_T<T>.npy, runs a batched forward pass, and writes
synth_codelen_T<T>.npz (or per-task parts + merge for sequence sharding).

Typical usage:

    # one job per temperature (see slurmfiles/calc_synth_codelen.slurm)
    sbatch slurmfiles/calc_synth_codelen.slurm

    # pilot: only T=0.5
    sbatch --array=0 slurmfiles/calc_synth_codelen.slurm

    # optional: shard sequences across tasks, then merge
    python3 calc_synth_codelen.py compute --temperature 0.5 \\
        --task-id 0 --total-tasks 10 ...
    python3 calc_synth_codelen.py merge --temperature 0.5 --total-tasks 10 ...

    # audit
    python3 calc_synth_codelen.py status --data-dir synth_data
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TEMPS = [0.5, 0.8, 1.0, 1.2, 1.5]
LOG2 = float(np.log(2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_comp = sub.add_parser("compute", help="compute codelengths for one temperature")
    p_comp.add_argument("--temperature", type=float, required=True)
    p_comp.add_argument(
        "--data-dir",
        type=str,
        default="/scratch/gpfs/WBIALEK/ls1546/llms-entropy/synth_data",
    )
    p_comp.add_argument("--task-id", type=int, default=0)
    p_comp.add_argument("--total-tasks", type=int, default=1)
    p_comp.add_argument("--batch-size", type=int, default=16)
    p_comp.add_argument(
        "--parts-dir",
        type=str,
        default=None,
        help="if set, write part files here; else write final npz to data-dir",
    )
    p_comp.add_argument(
        "--ckpt-base",
        type=str,
        default="/scratch/gpfs/WBIALEK/ls1546/gpt-circuits/out",
    )
    p_comp.add_argument("--run-name", type=str, default="shard1_m1337_d1337")
    p_comp.add_argument("--ckpt-num", type=int, default=47000)
    p_comp.add_argument("--block-size", type=int, default=2048)
    p_comp.add_argument("--n-layer", type=int, default=24)
    p_comp.add_argument("--n-head", type=int, default=16)
    p_comp.add_argument("--n-embd", type=int, default=2048)
    p_comp.add_argument("--bias", action="store_true")
    p_comp.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    p_comp.add_argument("--log-every", type=int, default=10)
    p_comp.add_argument("--force", action="store_true", help="recompute even if output exists")

    p_merge = sub.add_parser("merge", help="concatenate per-task codelen parts into final npz")
    p_merge.add_argument("--temperature", type=float, required=True)
    p_merge.add_argument("--data-dir", type=str, required=True)
    p_merge.add_argument("--parts-dir", type=str, required=True)
    p_merge.add_argument("--total-tasks", type=int, required=True)
    p_merge.add_argument("--cleanup", action="store_true")

    p_stat = sub.add_parser("status", help="report which temperatures have complete outputs")
    p_stat.add_argument("--data-dir", type=str, required=True)
    p_stat.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPS,
    )
    p_stat.add_argument("--parts-dir", type=str, default=None)
    p_stat.add_argument("--total-tasks", type=int, default=1)

    return p.parse_args()


def ids_path(data_dir: str, temperature: float) -> str:
    return os.path.join(data_dir, f"synth_ids_T{temperature}.npy")


def final_npz_path(data_dir: str, temperature: float) -> str:
    return os.path.join(data_dir, f"synth_codelen_T{temperature}.npz")


def part_npz_path(parts_dir: str, temperature: float, task_id: int) -> str:
    return os.path.join(parts_dir, f"synth_codelen_T{temperature}_part{task_id:04d}.npz")


def expected_codelen_shape(n_seqs: int, block_size: int) -> tuple[int, int]:
    return (n_seqs, block_size - 1)


@torch.no_grad()
def calc_codelength_from_ids(
    model,
    token_ids: np.ndarray | torch.Tensor,
    *,
    device: str,
    batch_size: int,
    dtype: torch.dtype,
    log_every: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (avg_codelen, pos_avg, all_codelen) with all_codelen shape (N, S-1)."""
    if isinstance(token_ids, np.ndarray):
        # mmap'd slices from np.load(..., mmap_mode='r') are read-only; copy once
        if not token_ids.flags.writeable:
            token_ids = np.array(token_ids, copy=True)
        token_ids = torch.from_numpy(token_ids).long()

    n, s = token_ids.shape
    pos_sum = torch.zeros(s - 1, device=device, dtype=torch.float32)
    pos_count = torch.zeros(s - 1, device=device, dtype=torch.float32)
    all_codelen = torch.empty((n, s - 1), dtype=torch.float32)

    use_autocast = device == "cuda" and dtype != torch.float32
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype) if use_autocast else nullcontext()
    )

    n_batches = (n + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, n, batch_size)):
        ids = token_ids[start : start + batch_size].to(device)
        inp = ids[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        with autocast_ctx:
            logits, _ = model(inp, tgt)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="none",
        ).view(tgt.shape)
        codelen = (nll / LOG2).float()

        end = start + codelen.size(0)
        all_codelen[start:end] = codelen
        t_len = codelen.size(1)
        pos_sum[:t_len] += codelen.sum(dim=0)
        pos_count[:t_len] += codelen.size(0)

        if (bi + 1) % log_every == 0 or (bi + 1) == n_batches:
            print(
                f"  batch {bi + 1}/{n_batches} ({end}/{n} sequences)",
                flush=True,
            )

    pos_avg = (pos_sum / pos_count.clamp_min(1)).cpu().numpy()
    all_codelen_np = all_codelen.cpu().numpy()
    return float(all_codelen_np.mean()), pos_avg, all_codelen_np


def load_model(args, device: str):
    from umt.evals.eval_utils import load_model_from_checkpoint

    ckpt_path = os.path.join(
        args.ckpt_base, args.run_name, f"{args.run_name}_ckpt_{args.ckpt_num}.pt"
    )
    print(f"loading checkpoint {ckpt_path}", flush=True)
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
    return model


def save_npz(path: str, pos_avg: np.ndarray, all_codelen: np.ndarray, avg_codelen: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # `np.savez(path, ...)` appends ".npz" unless path already ends in ".npz";
    # a temp name like "foo.npz.tmp" becomes "foo.npz.tmp.npz". Write via a
    # file handle so the temp path is preserved, then atomically rename.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            pos_avg=pos_avg,
            all_codelen=all_codelen,
            avg_codelen=np.float64(avg_codelen),
        )
    os.replace(tmp, path)


def npz_is_complete(path: str, expected_shape: tuple[int, int]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as d:
            if "all_codelen" not in d:
                return False
            return tuple(d["all_codelen"].shape) == expected_shape
    except Exception:
        return False


def cmd_compute(args):
    id_path = ids_path(args.data_dir, args.temperature)
    if not os.path.exists(id_path):
        raise FileNotFoundError(f"missing token ids: {id_path}")

    synth_ids = np.load(id_path, mmap_mode="r")
    n_total, seq_len = synth_ids.shape
    exp_shape = expected_codelen_shape(n_total, seq_len)

    chunk = (n_total + args.total_tasks - 1) // args.total_tasks
    start = args.task_id * chunk
    end = min(start + chunk, n_total)
    if start >= n_total:
        print(f"[compute T={args.temperature} task={args.task_id}] no work", flush=True)
        return

    n_my = end - start
    exp_my_shape = (n_my, seq_len - 1)

    if args.parts_dir:
        out_path = part_npz_path(args.parts_dir, args.temperature, args.task_id)
    else:
        if args.total_tasks != 1:
            raise ValueError("sequence sharding requires --parts-dir")
        out_path = final_npz_path(args.data_dir, args.temperature)

    if not args.force and npz_is_complete(out_path, exp_my_shape):
        print(
            f"[compute T={args.temperature} task={args.task_id}] "
            f"{out_path} already complete {exp_my_shape}, skipping",
            flush=True,
        )
        return

    print(
        f"[compute T={args.temperature} task={args.task_id}/{args.total_tasks}] "
        f"rows [{start}:{end}] = {n_my} sequences from {id_path}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]
    print(f"device={device} dtype={args.dtype} batch_size={args.batch_size}", flush=True)

    model = load_model(args, device)
    my_ids = np.array(synth_ids[start:end], dtype=np.int64, copy=True)

    avg_cl, pos_avg, all_cl = calc_codelength_from_ids(
        model,
        my_ids,
        device=device,
        batch_size=args.batch_size,
        dtype=torch_dtype,
        log_every=args.log_every,
    )
    assert all_cl.shape == exp_my_shape, (all_cl.shape, exp_my_shape)

    save_npz(out_path, pos_avg, all_cl, avg_cl)
    print(
        f"[compute T={args.temperature} task={args.task_id}] "
        f"avg={avg_cl:.4f} bits/token -> {out_path}",
        flush=True,
    )


def cmd_merge(args):
    """Merge per-task codelength parts into a single npz per temperature."""
    id_path = ids_path(args.data_dir, args.temperature)
    synth_ids = np.load(id_path, mmap_mode="r")
    n_total, seq_len = synth_ids.shape
    exp_shape = expected_codelen_shape(n_total, seq_len)
    chunk = (n_total + args.total_tasks - 1) // args.total_tasks

    parts = []
    missing = []
    for i in range(args.total_tasks):
        p = part_npz_path(args.parts_dir, args.temperature, i)
        start = i * chunk
        end = min(start + chunk, n_total)
        n_my = end - start
        if n_my <= 0:
            continue
        exp_my = (n_my, seq_len - 1)
        if not npz_is_complete(p, exp_my):
            missing.append(i)
            continue
        with np.load(p) as d:
            parts.append(d["all_codelen"])

    if missing:
        print(
            f"[merge T={args.temperature}] MISSING {len(missing)} parts "
            f"(ids: {missing[:15]}{'...' if len(missing) > 15 else ''})",
            flush=True,
        )
        if not parts:
            return

    all_cl = np.concatenate(parts, axis=0)
    if all_cl.shape != exp_shape:
        raise RuntimeError(f"merged shape {all_cl.shape} != expected {exp_shape}")

    pos_avg = all_cl.mean(axis=0)
    avg_cl = float(all_cl.mean())
    out_path = final_npz_path(args.data_dir, args.temperature)
    save_npz(out_path, pos_avg, all_cl, avg_cl)
    print(
        f"[merge T={args.temperature}] avg={avg_cl:.4f} -> {out_path} {all_cl.shape}",
        flush=True,
    )

    if args.cleanup and not missing:
        for i in range(args.total_tasks):
            p = part_npz_path(args.parts_dir, args.temperature, i)
            if os.path.exists(p):
                os.remove(p)


def cmd_status(args):
    for t in args.temperatures:
        id_p = ids_path(args.data_dir, t)
        out_p = final_npz_path(args.data_dir, t)
        if not os.path.exists(id_p):
            print(f"T={t}: NO ids ({id_p})", flush=True)
            continue
        n_total, seq_len = np.load(id_p, mmap_mode="r").shape
        exp = expected_codelen_shape(n_total, seq_len)
        if npz_is_complete(out_p, exp):
            with np.load(out_p) as d:
                avg = float(d["avg_codelen"]) if "avg_codelen" in d else float(d["all_codelen"].mean())
            print(f"T={t}: OK  {out_p}  avg={avg:.4f}  shape={exp}", flush=True)
        else:
            print(f"T={t}: INCOMPLETE or missing  {out_p}  (expect {exp})", flush=True)


def main():
    args = parse_args()
    if args.cmd == "compute":
        cmd_compute(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    elif args.cmd == "status":
        cmd_status(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
