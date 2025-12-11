import argparse
import json
import pickle as pkl
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_safeset(path: str) -> set:
    with open(path, "rb") as f:
        return pkl.load(f)


def filterize(texts: Iterable[str], safeset: set) -> List[str]:
    return [t for t in texts if t and len(set(t).difference(safeset)) == 0]


def load_texts(files: Sequence[str]) -> List[str]:
    data = []
    for path in files:
        with open(path, "r") as f:
            data.extend(json.loads(line)["text"] for line in f)
    return data


def chunk_texts(texts: Sequence[str], splitlen: int) -> List[str]:
    windows: List[str] = []
    for t in texts:
        if not t:
            continue
        windows.extend(
            t[i : i + splitlen] for i in np.arange(0, len(t), splitlen) if t[i : i + splitlen]
        )
    return windows


def prepare_samples(
    files: Sequence[str],
    safeset: set,
    splitlen: int,
    nsamples: int,
    buffer: int,
    seed: int,
) -> List[str]:
    btext = load_texts(files)
    ftext = filterize(btext, safeset)
    ftext_chunked = chunk_texts(ftext, splitlen)

    lengths = np.array([len(t) for t in ftext_chunked])
    indsort = np.flip(np.argsort(lengths))

    sample_count = min(nsamples, len(indsort))
    rng = np.random.default_rng(seed)
    starts = rng.choice(buffer, size=sample_count) if buffer > 0 else np.zeros(sample_count, dtype=int)
    ftext_sorted = [ftext_chunked[i][s:] for i, s in zip(indsort[:sample_count], starts)]
    return ftext_sorted


def build_safe_token_ids(tokenizer, safeset: set) -> np.ndarray:
    safe_ids = []
    for tid in range(len(tokenizer)):
        text = tokenizer.decode(tid, clean_up_tokenization_spaces=False)
        if text and set(text).issubset(safeset):
            safe_ids.append(tid)
    return np.array(safe_ids, dtype=np.int64)


def rows_all_safe(tok_batch: np.ndarray, attn_batch: np.ndarray, safe_id_set: set) -> np.ndarray:
    active = attn_batch == 1  # ignore padding
    return np.array([all((tid in safe_id_set) for tid in seq[mask]) for seq, mask in zip(tok_batch, active)])


def compute_entropies(
    dataloader: DataLoader,
    tokenizer,
    model,
    device: torch.device,
    shapecut: int,
    context_lengths: Sequence[int],
    near_zero_thresh: float,
    safe_id_set: set,
    top_k: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[Tuple[int, int]]]]:
    ent_by_ctx: Dict[int, List[np.ndarray]] = {k: [] for k in context_lengths}
    zero_by_ctx: Dict[int, Counter] = {k: Counter() for k in context_lengths}

    model.eval()
    with torch.no_grad():
        for docs in tqdm(dataloader):
            inputs = tokenizer(
                docs,
                return_tensors="pt",
                return_token_type_ids=False,
                padding="max_length",
                max_length=shapecut,
                truncation=True,
            ).to(device)

            logits = model(**inputs).logits  # [B, T, V]
            probs = F.softmax(logits, dim=-1)

            tok = inputs["input_ids"].cpu().numpy()
            attn = inputs["attention_mask"].cpu().numpy()

            keep_rows = rows_all_safe(tok, attn, safe_id_set)
            if not keep_rows.any():
                continue

            tok = tok[keep_rows]
            attn = attn[keep_rows]
            probs = probs[keep_rows]

            token_entropy = -(probs * torch.log2(probs.clamp_min(1e-10))).sum(dim=-1)

            for k in context_lengths:
                if k > token_entropy.shape[1]:
                    continue
                ent_slice = token_entropy[:, k - 1].cpu().numpy()
                attn_slice = attn[:, k - 1]
                ent_valid = ent_slice[attn_slice == 1]
                if not ent_valid.size:
                    continue
                ent_by_ctx[k].append(ent_valid)
                near_zero_mask = ent_valid < near_zero_thresh
                if near_zero_mask.any():
                    zero_tokens = tok[:, k - 1][attn_slice == 1][near_zero_mask]
                    zero_by_ctx[k].update(zero_tokens.tolist())

    ent_by_ctx = {k: (np.concatenate(v) if v else np.array([])) for k, v in ent_by_ctx.items()}

    zero_by_ctx_sorted: Dict[int, List[Tuple[int, int]]] = {}
    for k, counter in zero_by_ctx.items():
        zero_by_ctx_sorted[k] = counter.most_common(top_k)

    return ent_by_ctx, zero_by_ctx_sorted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute next-token entropy distributions with safelist filtering.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "/scratch/gpfs/DATASETS/hugging_face/c4/en/c4-train.00217-of-01024.json",
        ],
        help="JSONL files to read (expects a 'text' field per line).",
    )
    parser.add_argument("--safeset-path", type=str, default="colin_files/safeset2.txt", help="Pickled safeset path.")
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B", help="HF model name.")
    parser.add_argument("--splitlen", type=int, default=15000, help="Non-overlapping window length for chunking text.")
    parser.add_argument("--nsamples", type=int, default=2000, help="How many windows to keep (after sorting by length).")
    parser.add_argument("--buffer", type=int, default=100, help="Random start offset range for each selected window.")
    parser.add_argument("--seed", type=int, default=223291, help="RNG seed for start offsets.")
    parser.add_argument("--batch-size", type=int, default=16, help="Dataloader batch size.")
    parser.add_argument("--shapecut", type=int, default=1024, help="Max tokens per sample (padding/truncation).")
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="1,3,10,40,100,1000",
        help="Comma-separated context lengths to sample entropy from.",
    )
    parser.add_argument("--near-zero-thresh", type=float, default=1e-3, help="Threshold for ~zero-entropy bin.")
    parser.add_argument("--top-k", type=int, default=10000, help="How many zero-entropy tokens to display.")
    parser.add_argument(
        "--save-ent",
        type=str,
        default="",
        help="Optional path to save entropies as npz (keys ctx_<len>).",
    )
    parser.add_argument(
        "--save-zero",
        type=str,
        default="",
        help=(
            "Optional path to save zero-entropy tokens per context "
            "(use .npz for numeric ids/counts, .json for ids/counts/text; TSV fallback otherwise)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",") if x]
    if args.shapecut < max(context_lengths):
        raise ValueError(f"shapecut {args.shapecut} must be >= max context length {max(context_lengths)}")

    safeset = load_safeset(args.safeset_path)

    print("Preparing samples...")
    ftext_sorted = prepare_samples(
        files=args.files,
        safeset=safeset,
        splitlen=args.splitlen,
        nsamples=args.nsamples,
        buffer=args.buffer,
        seed=args.seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.to(device)

    dataloader = DataLoader(ftext_sorted, batch_size=args.batch_size)

    safe_ids = build_safe_token_ids(tokenizer, safeset)
    safe_id_set = set(safe_ids.tolist())

    print("Computing entropies...")
    ent_by_ctx, zero_by_ctx = compute_entropies(
        dataloader=dataloader,
        tokenizer=tokenizer,
        model=model,
        device=device,
        shapecut=args.shapecut,
        context_lengths=context_lengths,
        near_zero_thresh=args.near_zero_thresh,
        safe_id_set=safe_id_set,
        top_k=args.top_k,
    )

    sizes = {k: v.size for k, v in ent_by_ctx.items()}
    print("Entropy counts per context length:", sizes)
    preview = {k: zero_by_ctx[k][:5] for k in zero_by_ctx}
    print("Top zero-entropy tokens per context (id, count):", preview)

    if args.save_ent:
        save_dict = {f"ctx_{k}": v for k, v in ent_by_ctx.items()}
        np.savez_compressed(args.save_ent, **save_dict)
        print(f"Saved entropies to {args.save_ent}")

    if args.save_zero:
        save_path = args.save_zero
        if save_path.endswith(".npz"):
            npz_payload = {}
            for k, items in zero_by_ctx.items():
                ids = np.array([t[0] for t in items], dtype=np.int64)
                counts = np.array([t[1] for t in items], dtype=np.int64)
                npz_payload[f"ctx_{k}_id"] = ids
                npz_payload[f"ctx_{k}_count"] = counts
            np.savez_compressed(save_path, **npz_payload)
        elif save_path.endswith(".json"):
            payload = {}
            for k, items in zero_by_ctx.items():
                payload[str(k)] = [
                    {
                        "token_id": tid,
                        "count": cnt,
                        "text": tokenizer.decode(tid, clean_up_tokenization_spaces=False),
                    }
                    for tid, cnt in items
                ]
            with open(save_path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            # TSV fallback; tabs/newlines are escaped to keep rows aligned
            with open(save_path, "w") as f:
                for k, items in zero_by_ctx.items():
                    for tid, cnt in items:
                        text = tokenizer.decode(tid, clean_up_tokenization_spaces=False)
                        safe_text = text.replace("\t", "\\t").replace("\n", "\\n")
                        f.write(f"{k}\t{tid}\t{cnt}\t{safe_text}\n")
        print(f"Saved zero-entropy tokens to {save_path}")


if __name__ == "__main__":
    main()
