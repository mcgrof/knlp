"""Router-component ablation (Phase 2.4, eval-side only).

For each (model, retention budget), evaluate KL/PPL under variants
of the KRI score function that selectively zero out individual
components. The cleanest comparison is at iso-actual-retention
(per Phase 2.1), so we keep block_size, local_window, sink_blocks,
and topk_blocks fixed across variants and vary only which terms
of the per-block score contribute.

Variants:

  sink_recent           — no global blocks at all (control baseline)
  random_global         — uniform-random prefix block selection
  cos_only              — w_cos=1, all other terms zero
  cos_value_energy      — w_cos=1 + value-energy term
  cos_recency           — w_cos=1 + recency term
  cos_novelty           — w_cos=1 + novelty term (if available)
  full_kri              — default KRI score (cos + value + recency + novelty)

We compare these at the budgets used in the existing eval
(fixed_blocks {2, 4, 8, 16}; local_window = L/8 = 128 at L=1024,
sink_blocks=1). The headline plot is "KL@budget" per variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.eval_pruned_ppl import load_model  # noqa: E402
from src.kri_mask import KRIConfig, build_kri_mask, fixed_policy_mask  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


SCORE_VARIANTS = {
    "cos_only":        dict(w_cos=1.0, w_value_energy=0.0, w_recency=0.0,
                            w_novelty=0.0, use_novelty=False),
    "cos_value":       dict(w_cos=1.0, w_value_energy=0.2, w_recency=0.0,
                            w_novelty=0.0, use_novelty=False),
    "cos_recency":     dict(w_cos=1.0, w_value_energy=0.0, w_recency=0.15,
                            w_novelty=0.0, use_novelty=False),
    "cos_novelty":     dict(w_cos=1.0, w_value_energy=0.0, w_recency=0.0,
                            w_novelty=0.5, use_novelty=True),
    "full_kri":        dict(w_cos=1.0, w_value_energy=0.2, w_recency=0.15,
                            w_novelty=0.5, use_novelty=True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--local_window_tokens", type=int, default=128)
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--topk_grid", type=str, default="2,4,8,16")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--precision", type=str, default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefill_frac", type=float, default=0.25)
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_ints(s: str):
    return tuple(int(x) for x in s.split(","))


@torch.no_grad()
def _full_log_probs(model: GPT2KRI, ids: torch.Tensor, dtype: torch.dtype,
                   device: torch.device) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=dtype != torch.float32):
        logits, _ = model(ids)
    return F.log_softmax(logits.float(), dim=-1)


@torch.no_grad()
def _eval_kl_ppl_under_mask(model: GPT2KRI, ids: torch.Tensor, labels: torch.Tensor,
                            mask: torch.Tensor, full_lp: torch.Tensor,
                            dtype: torch.dtype, device: torch.device,
                            prefill_split: int):
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=dtype != torch.float32):
        logits, _ = model(ids, attn_mask=mask)
    pruned_lp = F.log_softmax(logits.float(), dim=-1)
    # NLL on shifted labels
    nll_full = -(full_lp[:, :-1, :].gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1))
    nll_sparse = -(pruned_lp[:, :-1, :].gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1))
    # KL(p_pruned || p_full)
    p = pruned_lp.exp()
    kl = (p * (pruned_lp - full_lp)).sum(-1)
    kl_decode = kl[:, prefill_split + 1 :]
    return {
        "nll_full": float(nll_full.mean().item()),
        "nll_sparse": float(nll_sparse.mean().item()),
        "kl_mean": float(kl.mean().item()),
        "kl_decode": float(kl_decode.mean().item()) if kl_decode.numel() > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_out = out.with_suffix(".csv")

    topks = _csv_ints(args.topk_grid)
    models = args.models.split(",")
    tok = get_tokenizer("openai-community/gpt2")

    rows = []
    fh = out.open("w")
    L = args.seq_len
    prefill_split = int(L * args.prefill_frac)

    for m_path in models:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)
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
        cached = []
        for i, b in enumerate(val_loader):
            if i >= args.n_batches:
                break
            cached.append(b)
        H = model.cfg.n_head

        # Anchor: sink_recent baseline (no global blocks)
        for topk in topks:
            # control: random_global at this topk
            for batch in cached[:1]:
                pass  # warm-up not needed; the model is loaded already

            agg_random = {"kl_mean": 0.0, "kl_decode": 0.0,
                          "nll_full": 0.0, "nll_sparse": 0.0}
            for batch in cached:
                ids = batch["input_ids"][:, :L].to(device)
                labels = batch["labels"][:, :L].to(device)
                B = ids.shape[0]
                full_lp = _full_log_probs(model, ids, dtype, device)
                # random_global
                mask, _ = fixed_policy_mask(
                    policy="random_global", seq_len=L, batch_size=B, n_head=H,
                    block_size=args.block_size,
                    local_window_tokens=args.local_window_tokens,
                    sink_blocks=args.sink_blocks,
                    topk_blocks=topk, device=device,
                )
                stats = _eval_kl_ppl_under_mask(model, ids, labels, mask, full_lp,
                                                 dtype, device, prefill_split)
                for k in agg_random:
                    agg_random[k] += stats[k]
            for k in agg_random:
                agg_random[k] /= max(1, len(cached))
            ppl = math.exp(agg_random["nll_sparse"]) if agg_random["nll_sparse"] < 30 else float("inf")
            row = {
                "model": tag, "variant": "random_global", "topk_blocks": topk,
                "local_window_tokens": args.local_window_tokens,
                "sink_blocks": args.sink_blocks, "block_size": args.block_size,
                "seq_len": L, "prefill_split": prefill_split,
                **agg_random, "ppl_sparse": ppl,
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"  random_global topk={topk:3d}  kl={row['kl_mean']:.4f}  "
                  f"kl_decode={row['kl_decode']:.4f}  ppl={ppl:.3f}")

            # score variants via the kri policy with custom KRIConfig weights
            for variant_name, kri_kwargs in SCORE_VARIANTS.items():
                agg = {"kl_mean": 0.0, "kl_decode": 0.0,
                       "nll_full": 0.0, "nll_sparse": 0.0}
                for batch in cached:
                    ids = batch["input_ids"][:, :L].to(device)
                    labels = batch["labels"][:, :L].to(device)
                    B = ids.shape[0]
                    full_lp = _full_log_probs(model, ids, dtype, device)

                    # build KRI mask manually with custom score weights
                    kvs = model.collect_kv(ids)
                    k_per = [kv[0] for kv in kvs]
                    v_per = [kv[1] for kv in kvs]
                    q_per = [kv[0] for kv in kvs]
                    cfg = KRIConfig(
                        block_size=args.block_size,
                        local_window_tokens=args.local_window_tokens,
                        global_topk_blocks=topk,
                        prefill_split=0,  # whole sequence is "decode"
                        protected_blocks=tuple(range(args.sink_blocks)),
                        per_head=False,
                        **kri_kwargs,
                    )
                    kri_mask = build_kri_mask(
                        cfg, L, B, H,
                        k_per_layer=k_per, v_per_layer=v_per, q_per_layer=q_per,
                        device=device,
                    )
                    # add sink + local + same-block + the KRI selection
                    base_mask, _ = fixed_policy_mask(
                        policy="sink_recent", seq_len=L, batch_size=B, n_head=H,
                        block_size=args.block_size,
                        local_window_tokens=args.local_window_tokens,
                        sink_blocks=args.sink_blocks, topk_blocks=0,
                        device=device,
                    )
                    causal = torch.tril(torch.ones(L, L, dtype=torch.bool,
                                                    device=device))
                    mask = (base_mask | kri_mask) & causal.view(1, 1, L, L)

                    stats = _eval_kl_ppl_under_mask(model, ids, labels, mask, full_lp,
                                                     dtype, device, prefill_split)
                    for k in agg:
                        agg[k] += stats[k]
                for k in agg:
                    agg[k] /= max(1, len(cached))
                ppl = math.exp(agg["nll_sparse"]) if agg["nll_sparse"] < 30 else float("inf")
                row = {
                    "model": tag, "variant": variant_name, "topk_blocks": topk,
                    "local_window_tokens": args.local_window_tokens,
                    "sink_blocks": args.sink_blocks, "block_size": args.block_size,
                    "seq_len": L, "prefill_split": prefill_split,
                    **agg, "ppl_sparse": ppl,
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  {variant_name:15s} topk={topk:3d}  "
                      f"kl={row['kl_mean']:.4f}  kl_decode={row['kl_decode']:.4f}  "
                      f"ppl={ppl:.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    if rows:
        with csv_out.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {out} and {csv_out}")
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    os._exit(rc)
