"""Effective retention accounting for KV-routing policies.

For each (model, policy, nominal_budget) combination, build the
attention mask that would be applied at inference and report the
*actual* retained-KV statistics, not just the nominal label. Without
this, x-axis comparisons across policies are unfair: KRI's local
window + sink + topk easily adds up to more KV than `recent` at the
same nominal label.

Output fields (one row per (model, policy, nominal, layer_or_head)):

  model
  policy
  nominal_retention_frac
  local_window, topk_blocks, sink_blocks, block_size, prefill_split
  seq_len
  scope                       "global" | "by_position" | "by_layer" |
                              "by_head_or_kv_group"
  actual_mean_retained_tokens
  actual_mean_retained_frac        normalised to T/2 (the dense
                                   causal-triangle average)
  actual_p50_retained_frac
  actual_p95_retained_frac
  actual_min_retained_frac
  actual_max_retained_frac
  position                          for scope=by_position
  layer                             for scope=by_layer (KRI per-layer)
  head                              for scope=by_head_or_kv_group
                                    (per_head=True KRI)
  n_eval_batches                    number of batches the stats came
                                    from (1 for analytic policies,
                                    more for KRI which depends on K/V)

The "by_layer" scope only applies when `score_layer_index` varies or
when per_head is True. The KRI selection in our implementation uses
one configured score layer with the union across heads (per_head=
False); we therefore emit by_layer rows only for KRI configurations
where `per_head=True`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import statistics
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.eval_pruned_ppl import load_model, retention_to_policy_params  # noqa: E402
from src.kri_mask import KRIConfig, build_kri_mask, fixed_policy_mask  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True,
                   help="Comma-separated list of HF names or checkpoint paths.")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--policies", type=str, default="full,recent,sink_recent,kri")
    p.add_argument("--retention_fracs", type=str, default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--n_batches", type=int, default=8,
                   help="batches used to compute KRI-dependent stats")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--per_head", action="store_true",
                   help="emit per-head retention statistics for KRI")
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--output", type=str, required=True,
                   help="Path to JSONL output. A sibling .csv is also written.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(","))


def _quantile(xs: torch.Tensor, q: float) -> float:
    """Quantile of a 1-D tensor without leaving torch."""
    if xs.numel() == 0:
        return 0.0
    return float(torch.quantile(xs.float(), q).item())


def _mask_stats(mask: torch.Tensor, seq_len: int) -> dict:
    """Aggregate keep-count statistics over [B, H, T, T] bool mask.

    Returns globals + per-position arrays. All retention fractions are
    normalised by T/2 (the dense lower-triangle average).
    """
    B, H, T, _ = mask.shape
    assert T == seq_len
    dense_avg = T / 2.0
    keep_per_query = mask.sum(-1).float()                # [B, H, T]
    keep_flat = keep_per_query.flatten()                 # [B*H*T]
    frac_flat = keep_flat / dense_avg
    by_pos = keep_per_query.mean(dim=(0, 1)) / dense_avg  # [T]
    by_head = keep_per_query.mean(dim=(0, 2)) / dense_avg  # [H]
    return {
        "mean_tokens": float(keep_flat.mean().item()),
        "mean_frac": float(frac_flat.mean().item()),
        "p50_frac": _quantile(frac_flat, 0.50),
        "p95_frac": _quantile(frac_flat, 0.95),
        "min_frac": float(frac_flat.min().item()),
        "max_frac": float(frac_flat.max().item()),
        "by_position": by_pos.detach().cpu().tolist(),
        "by_head": by_head.detach().cpu().tolist(),
    }


def _build_mask_for_policy(model: GPT2KRI, ids: torch.Tensor, policy: str,
                          params: dict, block_size: int, per_head: bool,
                          device: torch.device) -> torch.Tensor:
    """Build a [B, H, T, T] bool mask for the given policy on this batch."""
    B, T = ids.shape
    H = model.cfg.n_head
    if policy == "full":
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        return causal.view(1, 1, T, T).expand(B, H, T, T).clone()
    need_kv = policy == "kri" and params["topk_blocks"] > 0
    kvs = model.collect_kv(ids) if need_kv else None
    if need_kv:
        k_per = [kv[0] for kv in kvs]
        v_per = [kv[1] for kv in kvs]
        q_per = [kv[0] for kv in kvs]
    else:
        k_per = v_per = q_per = None
    mask, _ = fixed_policy_mask(
        policy=policy, seq_len=T, batch_size=B, n_head=H,
        block_size=block_size,
        local_window_tokens=params["local_window_tokens"],
        sink_blocks=params["sink_blocks"],
        topk_blocks=params["topk_blocks"],
        device=device,
        k_per_layer=k_per, v_per_layer=v_per, q_per_layer=q_per,
    )
    return mask


def aggregate_runs(stats_list: List[dict]) -> dict:
    """Mean of per-batch stats, plus elementwise mean of by_position
    and by_head arrays."""
    if not stats_list:
        return {}
    keys_scalar = ("mean_tokens", "mean_frac", "p50_frac", "p95_frac", "min_frac", "max_frac")
    out = {}
    for k in keys_scalar:
        out[k] = sum(s[k] for s in stats_list) / len(stats_list)
    out["by_position"] = [
        sum(s["by_position"][i] for s in stats_list) / len(stats_list)
        for i in range(len(stats_list[0]["by_position"]))
    ]
    out["by_head"] = [
        sum(s["by_head"][i] for s in stats_list) / len(stats_list)
        for i in range(len(stats_list[0]["by_head"]))
    ]
    return out


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)

    out_jsonl = Path(args.output)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_jsonl.with_suffix(".csv")

    data_cfg = DataConfig(
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        text_column=args.text_column, streaming=_truthy(args.streaming),
        train_split=args.train_split, val_split=args.val_split,
        seq_len=args.seq_len,
    )
    tok = get_tokenizer("openai-community/gpt2")
    _, val_ds = get_train_val_streams(data_cfg, tok)

    policies = args.policies.split(",")
    retention_fracs = _csv_floats(args.retention_fracs)
    models = args.models.split(",")

    rows = []
    for m_path in models:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                collate_fn=collate, num_workers=0)
        # cache a few batches so KRI-dependent stats use the same data
        cached = []
        for i, b in enumerate(val_loader):
            if i >= args.n_batches:
                break
            cached.append(b)
        print(f"  cached {len(cached)} batches of size {args.batch_size}")

        for policy in policies:
            if policy == "full":
                params = retention_to_policy_params(1.0, args.seq_len, args.block_size,
                                                    args.sink_blocks, "full")
                stats_list = []
                for batch in cached[:1]:  # full mask is deterministic
                    ids = batch["input_ids"].to(device)
                    mask = _build_mask_for_policy(model, ids, "full", params,
                                                   args.block_size, args.per_head, device)
                    stats_list.append(_mask_stats(mask, args.seq_len))
                agg = aggregate_runs(stats_list)
                row = _emit_row(tag, "full", 1.0, params, args.seq_len, "global", agg, 1)
                rows.append(row)
                _print_row(row)
                continue
            for frac in retention_fracs:
                if frac >= 0.999:
                    continue
                params = retention_to_policy_params(frac, args.seq_len, args.block_size,
                                                    args.sink_blocks, policy)
                # KRI depends on K/V => repeat over multiple batches.
                # Other policies are analytic => one batch is enough.
                n_runs = len(cached) if policy == "kri" else 1
                stats_list = []
                for batch in cached[:n_runs]:
                    ids = batch["input_ids"].to(device)
                    mask = _build_mask_for_policy(model, ids, policy, params,
                                                   args.block_size, args.per_head, device)
                    stats_list.append(_mask_stats(mask, args.seq_len))
                agg = aggregate_runs(stats_list)
                row = _emit_row(tag, policy, frac, params, args.seq_len, "global", agg, n_runs)
                rows.append(row)
                _print_row(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with out_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if rows:
        flat = [
            {k: v for k, v in r.items() if k not in ("by_position", "by_head")}
            for r in rows
        ]
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            for r in flat:
                w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {out_jsonl} and {out_csv}")

    _print_iso_retention_summary(rows)
    return 0


def _emit_row(tag: str, policy: str, frac: float, params: dict, seq_len: int,
              scope: str, agg: dict, n_batches: int) -> dict:
    return {
        "model": tag,
        "policy": policy,
        "nominal_retention_frac": frac,
        "local_window_tokens": params["local_window_tokens"],
        "sink_blocks": params["sink_blocks"],
        "topk_blocks": params["topk_blocks"],
        "block_size": params.get("block_size", 16),
        "seq_len": seq_len,
        "scope": scope,
        "n_eval_batches": n_batches,
        "actual_mean_retained_tokens": agg["mean_tokens"],
        "actual_mean_retained_frac": agg["mean_frac"],
        "actual_p50_retained_frac": agg["p50_frac"],
        "actual_p95_retained_frac": agg["p95_frac"],
        "actual_min_retained_frac": agg["min_frac"],
        "actual_max_retained_frac": agg["max_frac"],
        "by_position": agg["by_position"],
        "by_head": agg["by_head"],
    }


def _print_row(row: dict) -> None:
    print(
        f"  {row['policy']:12s} nom={row['nominal_retention_frac']:.4f} "
        f"actual_mean={row['actual_mean_retained_frac']:.3f} "
        f"p50={row['actual_p50_retained_frac']:.3f} "
        f"p95={row['actual_p95_retained_frac']:.3f} "
        f"tokens={row['actual_mean_retained_tokens']:.1f} "
        f"W={row['local_window_tokens']:4d} topk={row['topk_blocks']:3d} sinkB={row['sink_blocks']}"
    )


def _print_iso_retention_summary(rows: List[dict]) -> None:
    """Sort all rows by actual retention so iso-budget comparisons are
    visible at a glance."""
    print("\n=== iso-actual-retention view (sorted by actual_mean_retained_frac) ===")
    sortable = sorted(rows, key=lambda r: r["actual_mean_retained_frac"])
    for r in sortable:
        print(
            f"  actual={r['actual_mean_retained_frac']:.3f}  "
            f"model={r['model'][:40]:40s} "
            f"policy={r['policy']:12s} "
            f"nom={r['nominal_retention_frac']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
