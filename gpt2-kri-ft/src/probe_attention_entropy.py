"""Phase 2.9d — attention-entropy probe for vanilla models.

Why this exists:
    The Phase 2.9b SmolLM2 result is that vanilla SmolLM2-360M is
    already strikingly sparse-tolerant under KV-cache pruning
    (PPL 14.2 at 6.25% retention, vs vanilla GPT-2 small which is
    brittle at the same setting). A natural hypothesis is that
    SmolLM2 already does implicit soft routing — i.e. its attention
    distributions are flatter / peakier in patterns that match the
    sink+recent+KRI structure. This script measures that directly.

What it measures (per layer, per head, averaged over query
positions and batch):
    - entropy           H = -Σ p log p of the attention distribution
    - normalised entropy H / log T   ∈ [0, 1]  (uniform = 1, dirac = 0)
    - sink mass         avg attention on the first key position
    - top-k cumulative  top-1, top-8, top-32 cumulative mass

Output: per-layer summary row + per-(layer, head) detail row, both
written as JSONL + CSV. Designed to be diffed across models.

Honest scope: this is descriptive, not causal. It can disprove
"SmolLM2 implicit-routes" (if the distributions look the same as
GPT-2's) but can't on its own prove it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.train_kri import pick_device, pick_dtype, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help=(
            "Comma-separated HF names. We forward through "
            "AutoModelForCausalLM with output_attentions=True. "
            "Pass vanilla checkpoints only — this is a baseline probe."
        ),
    )
    p.add_argument(
        "--tokenizer_per_model",
        type=str,
        default="",
        help=(
            "Comma-separated tokenizer names matched 1-1 with --models. "
            "If omitted, each tokenizer defaults to its model name."
        ),
    )
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Held-out stream for the probe. Use same one across models.",
    )
    p.add_argument("--dataset_config", type=str, default="sample-10BT")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="true")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument(
        "--precision", type=str, default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="JSONL output path. A sibling .csv is also written.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


@torch.no_grad()
def _summarise_attentions(attentions, query_start: int = 32) -> dict:
    """Given a tuple of per-layer attention tensors, compute per-layer
    and per-head summaries.

    Each attention tensor is [B, H, Tq, Tk]. We average over batch
    and over query positions q >= query_start (skip the warm-up
    where the distribution is forced to be very peaky by causality).

    Returns:
        per_layer: list of dicts (layer, mean_entropy, mean_norm_entropy,
                                   sink_mass, top1_mass, top8_mass, top32_mass)
        per_layer_head: list of dicts (layer, head, ...)
    """
    per_layer = []
    per_layer_head = []
    for li, A in enumerate(attentions):
        # A: [B, H, Tq, Tk]; causal so Tk == Tq
        B, H, Tq, Tk = A.shape
        # drop early queries where mass is forced onto few keys
        q0 = min(query_start, Tq - 1)
        A_eff = A[:, :, q0:, :]  # [B, H, Tq', Tk]
        # numerical-stable entropy: assume row-stochastic; clamp for log
        eps = 1e-12
        log_p = (A_eff.clamp(min=eps)).log()
        entropy_per_q = -(A_eff * log_p).sum(dim=-1)  # [B, H, Tq']
        H_per_head = entropy_per_q.mean(dim=(0, 2))   # [H]
        log_Tk = math.log(Tk)
        norm_H_per_head = H_per_head / max(log_Tk, eps)
        # sink mass: avg attention on key index 0
        sink_per_head = A_eff[:, :, :, 0].mean(dim=(0, 2))  # [H]
        # top-k cumulative mass per query, averaged
        sorted_A, _ = A_eff.sort(dim=-1, descending=True)  # [B,H,Tq',Tk]
        for k_target in (1, 8, 32):
            kk = min(k_target, Tk)
            cum = sorted_A[:, :, :, :kk].sum(dim=-1).mean(dim=(0, 2))  # [H]
            if k_target == 1:
                top1 = cum
            elif k_target == 8:
                top8 = cum
            elif k_target == 32:
                top32 = cum
        for h in range(H):
            per_layer_head.append({
                "layer": li,
                "head": h,
                "entropy": float(H_per_head[h]),
                "norm_entropy": float(norm_H_per_head[h]),
                "sink_mass": float(sink_per_head[h]),
                "top1_mass": float(top1[h]),
                "top8_mass": float(top8[h]),
                "top32_mass": float(top32[h]),
                "Tk": int(Tk),
            })
        per_layer.append({
            "layer": li,
            "n_heads": int(H),
            "Tk": int(Tk),
            "mean_entropy": float(H_per_head.mean()),
            "mean_norm_entropy": float(norm_H_per_head.mean()),
            "mean_sink_mass": float(sink_per_head.mean()),
            "mean_top1_mass": float(top1.mean()),
            "mean_top8_mass": float(top8.mean()),
            "mean_top32_mass": float(top32.mean()),
        })
    return {"per_layer": per_layer, "per_layer_head": per_layer_head}


def _load_hf_model(name: str, device, dtype):
    """Load an HF causal LM with attention output enabled.

    Forces attn_implementation='eager' because output_attentions=True
    is incompatible with SDPA / Flash paths.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name,
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    model.config.output_attentions = True
    model = model.to(device).eval()
    return model


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    dtype = pick_dtype(args.precision)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_out = out.with_suffix(".csv")
    head_csv_out = out.with_name(out.stem + "_per_head.csv")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.tokenizer_per_model:
        toks = [t.strip() for t in args.tokenizer_per_model.split(",")]
    else:
        toks = list(models)
    assert len(toks) == len(models), (
        "tokenizer_per_model must align with --models"
    )

    fh = out.open("w")
    layer_rows = []
    head_rows = []

    for m_name, t_name in zip(models, toks):
        print(f"\n=== model: {m_name} (tokenizer: {t_name}) ===", flush=True)
        tok = get_tokenizer(t_name)
        model = _load_hf_model(m_name, device, dtype)

        data_cfg = DataConfig(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            text_column=args.text_column,
            streaming=_truthy(args.streaming),
            train_split=args.train_split,
            val_split=args.val_split,
            seq_len=args.seq_len,
        )
        _, val_ds = get_train_val_streams(data_cfg, tok)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                collate_fn=collate, num_workers=0)

        agg_layers = None
        agg_heads = None
        n_used = 0

        for i, batch in enumerate(val_loader):
            if i >= args.n_batches:
                break
            ids = batch["input_ids"][:, :args.seq_len].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=dtype != torch.float32):
                out_hf = model(ids, output_attentions=True, use_cache=False)
            summary = _summarise_attentions(out_hf.attentions)
            # Free GPU memory before next batch
            del out_hf
            if agg_layers is None:
                agg_layers = [
                    {k: (v if isinstance(v, (int,)) else 0.0)
                     for k, v in row.items()}
                    for row in summary["per_layer"]
                ]
                agg_heads = [
                    {k: (v if isinstance(v, (int,)) else 0.0)
                     for k, v in row.items()}
                    for row in summary["per_layer_head"]
                ]
            for j, row in enumerate(summary["per_layer"]):
                for k, v in row.items():
                    if isinstance(v, float):
                        agg_layers[j][k] += v
            for j, row in enumerate(summary["per_layer_head"]):
                for k, v in row.items():
                    if isinstance(v, float):
                        agg_heads[j][k] += v
            n_used += 1
            print(f"  batch {i+1}/{args.n_batches}  L0 H_mean={summary['per_layer'][0]['mean_entropy']:.3f}",
                  flush=True)

        # divide-by-n
        for row in agg_layers:
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] /= max(1, n_used)
        for row in agg_heads:
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] /= max(1, n_used)

        # write per-layer summary lines
        for row in agg_layers:
            line = {"model": m_name, **row}
            fh.write(json.dumps(line) + "\n")
            layer_rows.append(line)
        for row in agg_heads:
            line = {"model": m_name, **row}
            head_rows.append(line)
        fh.flush()

        print(f"  L=0 mean_norm_entropy={agg_layers[0]['mean_norm_entropy']:.3f} "
              f"mean_sink_mass={agg_layers[0]['mean_sink_mass']:.3f} "
              f"mean_top8_mass={agg_layers[0]['mean_top8_mass']:.3f}")
        print(f"  L=mid mean_norm_entropy={agg_layers[len(agg_layers)//2]['mean_norm_entropy']:.3f} "
              f"mean_sink_mass={agg_layers[len(agg_layers)//2]['mean_sink_mass']:.3f} "
              f"mean_top8_mass={agg_layers[len(agg_layers)//2]['mean_top8_mass']:.3f}")
        print(f"  L=last mean_norm_entropy={agg_layers[-1]['mean_norm_entropy']:.3f} "
              f"mean_sink_mass={agg_layers[-1]['mean_sink_mass']:.3f} "
              f"mean_top8_mass={agg_layers[-1]['mean_top8_mass']:.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fh.close()

    if layer_rows:
        with csv_out.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(layer_rows[0].keys()))
            w.writeheader()
            for r in layer_rows:
                w.writerow(r)
    if head_rows:
        with head_csv_out.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(head_rows[0].keys()))
            w.writeheader()
            for r in head_rows:
                w.writerow(r)

    print(f"\nwrote {len(layer_rows)} per-layer rows -> {out}")
    print(f"wrote {len(head_rows)} per-head rows -> {head_csv_out}")
    return 0


if __name__ == "__main__":
    rc = main()
    os._exit(rc)
