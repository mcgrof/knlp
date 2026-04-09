#!/usr/bin/env python3
"""Cartridge-style screening proxy for RA selector evaluation.

Loads a frozen HF GPT-2 124M, monkey-patches the attention forward to apply
the same RA overlay used in fim/reciprocal_attention/gpt2_matched.py at the
heads chosen by a selection JSON, and measures eval loss on val.bin slices
under each arm vs no-overlay baseline.

Goal: produce a Spearman-comparable rank order of selectors *without* running
the full training, so we can validate the cheap-eval hypothesis against the
W7900 3-arm training ground truth.

Usage:
    python screen_ra_selectors.py \
        --model openai-community/gpt2 \
        --val-bin /data/knlp/gpt2/data/finewebedu/val.bin \
        --selections \
            baseline=NONE \
            arm_a=fim/reciprocal_attention/configs/ra_ablation_gpt2_arm_a_fimtrace.json \
            arm_b=fim/reciprocal_attention/configs/ra_ablation_gpt2_arm_b_eigmax.json \
            arm_c=fim/reciprocal_attention/configs/ra_ablation_gpt2_arm_c_jsd.json \
            random=RANDOM:8 \
        --seq-len 1024 --batch-size 8 --num-batches 32 --seed 1337 \
        --alpha-std 0.9375 --alpha-rec 0.0625
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import MethodType
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


def parse_selection(spec: str, num_layers: int, num_heads: int, rng: np.random.Generator) -> Dict[int, List[int]]:
    """Parse a selection spec.
    NONE -> {} (no RA overlay).
    RANDOM:N -> N random heads sampled across all (layer, head) pairs.
    Otherwise: a JSON path with a "layers" dict.
    """
    if spec == "NONE":
        return {}
    if spec.startswith("RANDOM:"):
        n = int(spec.split(":", 1)[1])
        all_pairs = [(l, h) for l in range(num_layers) for h in range(num_heads)]
        idxs = rng.choice(len(all_pairs), size=n, replace=False)
        sel: Dict[int, List[int]] = {}
        for i in idxs:
            l, h = all_pairs[i]
            sel.setdefault(l, []).append(h)
        for v in sel.values():
            v.sort()
        return sel
    raw = json.loads(Path(spec).read_text())
    layers = raw.get("layers", raw)
    return {int(k): [int(x) for x in v] for k, v in layers.items()}


def make_patched_forward(orig_forward, selected_heads: List[int], alpha_std: float, alpha_rec: float):
    """Wrap a GPT2Attention.forward with the RA overlay applied at selected heads."""
    selected_heads = sorted(selected_heads)

    def forward(self, hidden_states, **kwargs):
        # Standard QKV computation
        query_states, key_states, value_states = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        q = query_states.contiguous()
        k = key_states.contiguous()
        v = value_states.contiguous()

        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = q.shape[2] > 1

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale, is_causal=is_causal
        )

        if selected_heads:
            sel = torch.tensor(selected_heads, device=q.device, dtype=torch.long)
            q_sel = q.index_select(1, sel)
            k_sel = k.index_select(1, sel)
            v_sel = v.index_select(1, sel)
            rec_out = F.scaled_dot_product_attention(
                k_sel, q_sel, v_sel,  # Q/K swapped — this is the RA overlay
                dropout_p=0.0,
                scale=scale,
                is_causal=is_causal,
            )
            mixed = (
                attn_output.index_select(1, sel) * alpha_std + rec_out * alpha_rec
            )
            attn_output = attn_output.clone()
            attn_output.index_copy_(1, sel, mixed)

        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(*hidden_states.shape[:-1], -1)
            .contiguous()
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, None

    return forward


def install_overlay(model: GPT2LMHeadModel, selection: Dict[int, List[int]], alpha_std: float, alpha_rec: float):
    """Replace each block's attn.forward with a patched version (selection may be empty for baseline)."""
    for li, block in enumerate(model.transformer.h):
        heads = selection.get(li, [])
        block.attn.forward = MethodType(
            make_patched_forward(None, heads, alpha_std, alpha_rec),
            block.attn,
        )


def eval_loss(model, x_batches, y_batches, device):
    """Returns (harness_loss, correct_loss) per the labels=y vs labels=x distinction.

    harness_loss: matches the harness (labels=y, suffers from HF's internal label
                  shift, so this is effectively double-shifted next-token loss).
                  Use this for Spearman comparison against training ground truth.
    correct_loss: labels=x, the standard 1-token-ahead next-token loss. Reported
                  for sanity-checking the absolute number.
    """
    harness_losses, correct_losses = [], []
    with torch.no_grad():
        for x, y in zip(x_batches, y_batches):
            x_d, y_d = x.to(device), y.to(device)
            out_h = model(input_ids=x_d, labels=y_d)
            harness_losses.append(float(out_h.loss.detach().cpu().item()))
            out_c = model(input_ids=x_d, labels=x_d)
            correct_losses.append(float(out_c.loss.detach().cpu().item()))
    return float(np.mean(harness_losses)), float(np.mean(correct_losses))


def sample_batches(val_bin: Path, n_batches: int, batch_size: int, seq_len: int, seed: int):
    """Pre-sample a fixed set of (x, y) batches so every arm sees the same data."""
    data = np.memmap(str(val_bin), dtype=np.uint16, mode="r")
    n_tokens = len(data)
    rng = np.random.default_rng(seed)
    max_start = n_tokens - seq_len - 1
    starts = rng.integers(0, max_start, size=n_batches * batch_size)
    x_batches, y_batches = [], []
    for b in range(n_batches):
        ss = starts[b * batch_size : (b + 1) * batch_size]
        x = torch.stack([torch.from_numpy(data[s : s + seq_len].astype(np.int64)) for s in ss])
        y = torch.stack([torch.from_numpy(data[s + 1 : s + 1 + seq_len].astype(np.int64)) for s in ss])
        x_batches.append(x)
        y_batches.append(y)
    return x_batches, y_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--val-bin", required=True, type=Path)
    ap.add_argument("--selections", nargs="+", required=True,
                    help="name=spec  where spec is NONE, RANDOM:N, or a path to a selection JSON")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--alpha-std", type=float, default=0.9375)
    ap.add_argument("--alpha-rec", type=float, default=0.0625)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None, help="Optional JSON path to write results to")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[screen] device={device}", flush=True)
    print(f"[screen] loading {args.model}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    cfg = model.config
    num_layers = cfg.n_layer
    num_heads = cfg.n_head
    print(f"[screen] {num_layers} layers x {num_heads} heads", flush=True)

    print(f"[screen] sampling {args.num_batches} batches of {args.batch_size}x{args.seq_len}", flush=True)
    x_batches, y_batches = sample_batches(args.val_bin, args.num_batches, args.batch_size, args.seq_len, args.seed)

    rng = np.random.default_rng(args.seed)
    arms = []
    for spec in args.selections:
        name, raw = spec.split("=", 1)
        sel = parse_selection(raw, num_layers, num_heads, rng)
        n_heads = sum(len(v) for v in sel.values())
        arms.append((name, raw, sel, n_heads))
        print(f"[screen] arm {name}: {n_heads} heads across {len(sel)} layers — {raw}", flush=True)

    results = []
    for name, raw, sel, n_heads in arms:
        install_overlay(model, sel, args.alpha_std, args.alpha_rec)
        t0 = time.time()
        h_loss, c_loss = eval_loss(model, x_batches, y_batches, device)
        dt = time.time() - t0
        print(f"[screen] {name:12s}  harness_loss={h_loss:.6f}  correct_loss={c_loss:.6f}  heads={n_heads}  wall={dt:.1f}s", flush=True)
        results.append({
            "name": name,
            "spec": raw,
            "n_heads": n_heads,
            "selection": {str(k): v for k, v in sel.items()},
            "harness_loss": h_loss,
            "correct_loss": c_loss,
            "harness_ppl": math.exp(h_loss),
            "correct_ppl": math.exp(c_loss),
            "wall_s": dt,
        })

    # Rank twice — once by harness_loss (matches training rank), once by correct_loss
    base = next((r for r in results if r["name"] == "baseline"), None)
    for metric in ("harness_loss", "correct_loss"):
        sorted_results = sorted(results, key=lambda r: r[metric])
        base_loss = base[metric] if base else None
        print(f"\n[screen] === ranked by {metric} (lowest first) ===", flush=True)
        for r in sorted_results:
            delta = (base_loss - r[metric]) if base_loss is not None else None
            delta_str = f"  Δvs_baseline={delta:+.6f}" if delta is not None else ""
            print(f"  {r['name']:12s}  {metric}={r[metric]:.6f}{delta_str}", flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps({
            "model": args.model,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "seed": args.seed,
            "alpha_std": args.alpha_std,
            "alpha_rec": args.alpha_rec,
            "results": results,
            "ranked": [r["name"] for r in sorted_results],
        }, indent=2) + "\n")
        print(f"[screen] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
