"""Oracle KL block selector (Phase 2.5 — the Phase 3 decision gate).

For each (model, context length L, query example), greedily select
prefix KV blocks to minimize the KL divergence between the pruned-
attention logits and the full-attention teacher logits. Block 0
(sink) and the local window blocks are always visible; the greedy
loop adds blocks from the eligible prefix-block set one at a time,
each time picking the candidate block that most reduces total KL
across all decode-region queries simultaneously.

The trajectory (#selected_blocks, KL) is then turned into:

    B_oracle(L, eps) = min #blocks  s.t.  mean KL <= eps

This is the upper bound any block-retention router could achieve.
Comparing B_oracle vs B_kri vs B_random tells us:

  - oracle ≈ KRI ⇒ router is fine, focus on systems work
  - oracle ≪ KRI ⇒ router is the bottleneck; fix KRI scoring first
  - oracle also bad at low budgets ⇒ raw blocks insufficient;
    Phase 3 learned redundancy is justified

Cost: greedy over ~50 candidate blocks at L=1024 × ~16 additions
per example. We pool examples in batched forward passes; ~1-2
minutes per (model, eps) cell on Blackwell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.eval_pruned_ppl import load_model  # noqa: E402
from src.kri_mask import num_blocks  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--context_lengths", type=str, default="1024",
                   help="Comma-separated. Oracle is expensive; default L=1024 only.")
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--local_window_tokens", type=int, default=128,
                   help="Fixed local window for the oracle's baseline visible set.")
    p.add_argument("--max_blocks", type=int, default=16,
                   help="Greedy adds up to this many global blocks then stops.")
    p.add_argument("--epsilons", type=str, default="0.03,0.1,0.3")
    p.add_argument("--n_examples", type=int, default=4,
                   help="Number of query examples to average over. Each example is one full sequence; the oracle minimizes mean KL across decode-region queries in that example.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--precision", type=str, default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefill_frac", type=float, default=0.25,
                   help="prefill_split = L * prefill_frac")
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_ints(s: str):
    return tuple(int(x) for x in s.split(","))


def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(","))


def _build_mask_from_blocks(visible_blocks: List[int], T: int, block_size: int,
                            local_window_tokens: int, prefill_split: int,
                            B: int, H: int, device: torch.device) -> torch.Tensor:
    """Build an [B, H, T, T] mask given an explicit list of visible
    prefix blocks plus the standard sink + local window + same-block
    structure used elsewhere.
    """
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    mask = torch.zeros(B, H, T, T, dtype=torch.bool, device=device)
    rows_prefill = torch.zeros(T, dtype=torch.bool, device=device)
    rows_prefill[: prefill_split + 1] = True
    mask = mask | (causal & rows_prefill.view(T, 1)).view(1, 1, T, T)

    idx_t = torch.arange(T, device=device)
    idx_k = torch.arange(T, device=device)
    diff = idx_t.view(T, 1) - idx_k.view(1, T)
    local_keep = (diff >= 0) & (diff < local_window_tokens)
    rows_decode = ~rows_prefill
    mask = mask | (local_keep & rows_decode.view(T, 1)).view(1, 1, T, T)

    # Add the explicit visible blocks (sink is block 0, expected to be
    # in `visible_blocks` by the caller)
    block_keep_2d = torch.zeros(T, dtype=torch.bool, device=device)
    for b in visible_blocks:
        lo = b * block_size
        hi = min(T, (b + 1) * block_size)
        block_keep_2d[lo:hi] = True
    block_visible = block_keep_2d.view(1, T) & causal
    mask = mask | (block_visible & rows_decode.view(T, 1)).view(1, 1, T, T)

    # Always visible: current block of t
    t_block_of = idx_t // block_size
    k_block_of = idx_k // block_size
    same_block = (t_block_of.view(T, 1) == k_block_of.view(1, T)) & causal
    mask = mask | (same_block & rows_decode.view(T, 1)).view(1, 1, T, T)

    return mask & causal.view(1, 1, T, T)


@torch.no_grad()
def _kl_to_teacher(model: GPT2KRI, ids: torch.Tensor, full_log_probs: torch.Tensor,
                  mask: torch.Tensor, dtype: torch.dtype, device: torch.device,
                  decode_start: int) -> float:
    """Mean KL(p_pruned || p_full) over decode-region positions."""
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=dtype != torch.float32):
        logits, _ = model(ids, attn_mask=mask)
    pruned_lp = F.log_softmax(logits.float(), dim=-1)
    # KL restricted to decode-region positions
    p = pruned_lp[:, decode_start:, :].exp()
    kl = (p * (pruned_lp[:, decode_start:, :] - full_log_probs[:, decode_start:, :])).sum(-1)
    return float(kl.mean().item())


def greedy_oracle(model: GPT2KRI, ids: torch.Tensor, full_log_probs: torch.Tensor,
                 block_size: int, local_window_tokens: int, prefill_split: int,
                 sink_blocks: int, max_blocks: int, device: torch.device,
                 dtype: torch.dtype) -> List[Tuple[int, float, List[int]]]:
    """For one batch, greedy-select up to max_blocks global blocks
    minimizing mean decode-region KL to the full-attention teacher.

    Returns a list of (step, kl, selected_blocks) tuples, one per
    addition step (including step 0 with only sink + local visible).
    """
    B, T = ids.shape
    H = model.cfg.n_head
    NB = num_blocks(T, block_size)

    # Eligible blocks: any block not in the sink, considered as a
    # global selection that augments the per-query local window. The
    # local window slides per-query, so a block that's inside one
    # query's window may still need to be a global selection for a
    # later query whose window has moved past it. We therefore allow
    # the entire prefix (block 0 excluded as sink) as the candidate
    # pool. Greedy picks the block that most reduces *mean* decode-
    # region KL across all decode queries simultaneously.
    decode_start = prefill_split + 1
    last_block_idx = max(0, (T - 1) // block_size)
    eligible = [b for b in range(NB) if b >= sink_blocks and b <= last_block_idx]

    # Always-visible: sink blocks.
    visible = list(range(sink_blocks))

    # Step 0: only sink + local + same-block visible.
    mask = _build_mask_from_blocks(visible, T, block_size, local_window_tokens,
                                    prefill_split, B, H, device)
    kl0 = _kl_to_teacher(model, ids, full_log_probs, mask, dtype, device, decode_start)
    trajectory = [(0, kl0, list(visible))]

    remaining = list(eligible)
    for step in range(1, max_blocks + 1):
        if not remaining:
            break
        best_kl = float("inf")
        best_block = None
        for cand in remaining:
            trial = visible + [cand]
            mask = _build_mask_from_blocks(trial, T, block_size, local_window_tokens,
                                            prefill_split, B, H, device)
            kl = _kl_to_teacher(model, ids, full_log_probs, mask, dtype, device,
                                 decode_start)
            if kl < best_kl:
                best_kl = kl
                best_block = cand
        if best_block is None:
            break
        visible.append(best_block)
        remaining.remove(best_block)
        trajectory.append((step, best_kl, list(visible)))
    return trajectory


@torch.no_grad()
def _full_log_probs(model: GPT2KRI, ids: torch.Tensor, dtype: torch.dtype,
                   device: torch.device) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=dtype != torch.float32):
        logits, _ = model(ids)
    return F.log_softmax(logits.float(), dim=-1)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_jsonl = out_dir / "oracle_rows.jsonl"
    summary_csv = out_dir / "oracle_summary.csv"

    context_lengths = _csv_ints(args.context_lengths)
    epsilons = _csv_floats(args.epsilons)
    models = args.models.split(",")

    tok = get_tokenizer("openai-community/gpt2")
    rows: List[dict] = []
    summary_rows: List[dict] = []
    fh = rows_jsonl.open("w")

    for m_path in models:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)

        for L in context_lengths:
            print(f"\n  -- L={L}")
            data_cfg = DataConfig(
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                text_column=args.text_column,
                streaming=_truthy(args.streaming),
                train_split=args.train_split,
                val_split=args.val_split,
                seq_len=L,
            )
            _, val_ds = get_train_val_streams(data_cfg, tok)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    collate_fn=collate, num_workers=0)
            batches = []
            for i, b in enumerate(val_loader):
                if i * args.batch_size >= args.n_examples:
                    break
                batches.append(b)

            prefill_split = int(L * args.prefill_frac)
            # Pool examples into batched forwards. One trajectory per
            # batch — within-batch examples share their oracle
            # selection (because the mask is per-batch).
            for bi, batch in enumerate(batches):
                ids = batch["input_ids"][:, :L].to(device)
                if ids.shape[1] < L:
                    continue
                full_lp = _full_log_probs(model, ids, dtype, device)
                traj = greedy_oracle(
                    model, ids, full_lp,
                    block_size=args.block_size,
                    local_window_tokens=args.local_window_tokens,
                    prefill_split=prefill_split,
                    sink_blocks=args.sink_blocks,
                    max_blocks=args.max_blocks,
                    device=device, dtype=dtype,
                )
                for step, kl, selected in traj:
                    row = {
                        "model": tag,
                        "context_length": L,
                        "batch_idx": bi,
                        "step": step,
                        "global_blocks_added": step,
                        "kl_to_full_teacher": kl,
                        "selected_blocks": selected,
                        "block_size": args.block_size,
                        "local_window_tokens": args.local_window_tokens,
                        "sink_blocks": args.sink_blocks,
                        "prefill_split": prefill_split,
                    }
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                print(f"    batch {bi}: trajectory KL = {traj[0][1]:.4f} "
                      f"-> {traj[-1][1]:.4f} after {len(traj)-1} additions")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fh.close()

    # Build B_oracle(L, eps) table by averaging across batches per (model, L)
    # and finding the minimum step where mean_KL <= eps.
    by_key = {}
    for r in rows:
        key = (r["model"], r["context_length"], r["step"])
        by_key.setdefault(key, []).append(r["kl_to_full_teacher"])
    avg_kl = {k: sum(v) / len(v) for k, v in by_key.items()}

    keys_seen = sorted(set((k[0], k[1]) for k in by_key.keys()))
    for (model_tag, L) in keys_seen:
        for eps in epsilons:
            achieved = False
            for step in range(args.max_blocks + 1):
                if (model_tag, L, step) in avg_kl and avg_kl[(model_tag, L, step)] <= eps:
                    summary_rows.append({
                        "model": model_tag, "context_length": L, "epsilon": eps,
                        "B_oracle": step,
                        "mean_kl_at_oracle": avg_kl[(model_tag, L, step)],
                        "achieved": True,
                    })
                    achieved = True
                    break
            if not achieved:
                summary_rows.append({
                    "model": model_tag, "context_length": L, "epsilon": eps,
                    "B_oracle": None,
                    "mean_kl_at_oracle": None,
                    "achieved": False,
                })

    with summary_csv.open("w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()) if summary_rows
                           else ["model", "context_length", "epsilon", "B_oracle",
                                 "mean_kl_at_oracle", "achieved"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\n=== B_oracle summary ===")
    for r in sorted(summary_rows, key=lambda r: (r["model"], r["context_length"], r["epsilon"])):
        b = r["B_oracle"] if r["B_oracle"] is not None else "n/a"
        print(f"  {r['model'][:38]:38s} L={r['context_length']:>4} "
              f"eps={r['epsilon']:.2f}  B_oracle={b}")
    print(f"\nwrote {len(rows)} rows -> {rows_jsonl}")
    print(f"wrote {len(summary_rows)} summary rows -> {summary_csv}")
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    os._exit(rc)
