#!/usr/bin/env python3
"""Generate an RA selection JSON for GPT-2 124M using the V2 synthetic-Q JSD
per-(layer, head) score (Phase 5-B head pooling weight, applied here as a
head ranker for the reciprocal attention selector).

Produces a file with the same shape consumed by parse_selection_file in
fim/reciprocal_attention/gpt2_matched.py:
    {"layers": {"<layer_idx>": [head_idx, ...], ...}, ...}

The selector mirrors gen_layer_ablation_configs.py:
    1. Rank layers by mean head JSD
    2. Drop the single highest layer (same skip-highest convention)
    3. Keep top-K candidate layers
    4. Within those layers, rank all heads by JSD, take top-N

This is the proposed replacement for Arm B (which used Perron-Frobenius
eigvals on row-stochastic batch-mean attention -- always = 1.0).
"""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_jsd_scores(
    model_id: str,
    n_synth: int,
    t_synth: int,
    calib_text: str,
    device: str,
    seed: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    cfg = model.config
    num_layers = cfg.n_layer
    num_heads = cfg.n_head
    d_model = cfg.n_embd
    d_head = d_model // num_heads
    print(
        f"[arm-c] {model_id}: {num_layers} layers x {num_heads} heads, "
        f"d_model={d_model}, d_head={d_head}"
    )

    # Capture per-layer residual stream (input to each transformer block)
    captured_resid = {}

    def make_resid_hook(li):
        def hook(module, inputs):
            x = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
            captured_resid[li] = x.detach().float().cpu()
        return hook

    handles = []
    for li, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_pre_hook(make_resid_hook(li)))

    # Calibration forward to seed Σ_layer
    enc = tokenizer(
        calib_text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.n_positions,
    )
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()

    # Σ_layer per layer (PSD via eigh, no Cholesky)
    sigmas = {}
    for li, x in captured_resid.items():
        x_flat = x.reshape(-1, x.shape[-1])
        mu = x_flat.mean(dim=0, keepdim=True)
        xc = x_flat - mu
        sigma = (xc.T @ xc) / max(x_flat.shape[0] - 1, 1) + 1e-6 * torch.eye(
            x_flat.shape[-1]
        )
        sigmas[li] = sigma

    # V2 synthetic-Q JSD per (layer, head)
    g = torch.Generator().manual_seed(seed)
    scores = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    for li in range(num_layers):
        sigma = sigmas[li].float()
        sigma_sym = 0.5 * (sigma + sigma.T)
        evals, evecs = torch.linalg.eigh(sigma_sym)
        evals = evals.clamp(min=0.0)
        L = evecs @ torch.diag(evals.sqrt())

        z = torch.randn(n_synth, t_synth, d_model, generator=g)
        x_synth = z @ L.T  # (n_synth, t_synth, d_model)

        block = model.transformer.h[li]
        c_attn_w = block.attn.c_attn.weight.detach().cpu().float()
        # GPT-2 c_attn packs Q,K,V along output dim:
        W_Q = c_attn_w[:, :d_model].T          # (d_model, d_model)
        W_K = c_attn_w[:, d_model : 2 * d_model].T

        Q = (x_synth @ W_Q.T).reshape(n_synth, t_synth, num_heads, d_head).transpose(1, 2)
        K = (x_synth @ W_K.T).reshape(n_synth, t_synth, num_heads, d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(d_head)
        attn_scores = torch.einsum("nhqd,nhkd->nhqk", Q, K) * scale
        attn = torch.softmax(attn_scores, dim=-1)  # (n_synth, num_heads, t, t)

        eps = 1e-12
        m = attn.mean(dim=0, keepdim=True).clamp(min=eps)
        p = attn.clamp(min=eps)
        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_mp = (m * (m.log() - p.log())).sum(dim=-1)
        jsd_per = 0.5 * (kl_pm + kl_mp)         # (n_synth, num_heads, t)
        jsd_head = jsd_per.mean(dim=(0, 2))     # (num_heads,)
        scores[li] = jsd_head.double()

    return scores  # (num_layers, num_heads)


def select(
    scores: torch.Tensor,
    top_layers: int,
    top_heads: int,
    skip_highest: bool,
):
    num_layers, num_heads = scores.shape
    layer_means = scores.mean(dim=1).tolist()
    ranked = sorted(range(num_layers), key=lambda i: layer_means[i], reverse=True)
    if skip_highest and ranked:
        ranked = ranked[1:]
    candidates = ranked[:top_layers]

    scored = []
    for li in candidates:
        for h in range(num_heads):
            scored.append((float(scores[li, h].item()), li, h))
    scored.sort(reverse=True)
    chosen = scored[:top_heads]

    layers = {}
    out_scores = {}
    for s, li, h in chosen:
        layers.setdefault(str(li), []).append(h)
        out_scores.setdefault(str(li), {})[str(h)] = s
    for v in layers.values():
        v.sort()

    return {
        "model": "gpt2",
        "layer_selector": "synthetic_q_jsd",
        "head_selector": "synthetic_q_jsd",
        "selection_method": "ablation-synthetic_q_jsd",
        "candidate_layers": candidates,
        "candidate_layer_scores": {str(i): layer_means[i] for i in candidates},
        "selected_head_count": len(chosen),
        "layers": layers,
        "scores": out_scores,
        "per_layer_mean_jsd": layer_means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="openai-community/gpt2")
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-layers", type=int, default=3)
    ap.add_argument("--top-heads", type=int, default=8)
    ap.add_argument("--skip-highest", action="store_true", default=True)
    ap.add_argument("--n-synth", type=int, default=256)
    ap.add_argument("--t-synth", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--calib-text",
        default=(
            "The quick brown fox jumps over the lazy dog. " * 64
            + "Reciprocal attention is a structural bias that ties query and "
              "key roles in transformer self-attention. "
            * 32
        ),
    )
    args = ap.parse_args()

    scores = compute_jsd_scores(
        model_id=args.model_id,
        n_synth=args.n_synth,
        t_synth=args.t_synth,
        calib_text=args.calib_text,
        device=args.device,
        seed=args.seed,
    )
    sel = select(
        scores=scores,
        top_layers=args.top_layers,
        top_heads=args.top_heads,
        skip_highest=args.skip_highest,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel, indent=2, sort_keys=True) + "\n")
    print(f"[arm-c] wrote {out}")
    print(f"[arm-c] candidate layers: {sel['candidate_layers']}")
    print(f"[arm-c] selected heads: {sel['selected_head_count']}")
    for k in sorted(sel["layers"], key=int):
        print(f"  L{k}: {sel['layers'][k]}")


if __name__ == "__main__":
    main()
