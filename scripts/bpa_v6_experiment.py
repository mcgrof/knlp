#!/usr/bin/env python
"""
BPA v6: Scale + Polish experiment runner.

Extends v5 with:
  - Gate overhead reduction (3 approaches)
  - L sweep up to 2048+
  - Multiple far_budget values
  - Detailed KV accounting per run
  - Scaling law data collection

Overhead reduction approaches:
  A) Sparse RA updates: update RA stats every N batches, reuse last values
  B) Chunk-level accumulation: already in v5 RAValueTracker (no token buffers)
  C) Local-first pipeline: skip gate when position < local_window

Usage:
    python scripts/bpa_v6_experiment.py \
        --checkpoint <path> \
        --gate-dir bpa_v4_gate_results \
        --seq-lens 512,1024,2048 \
        --seeds 1,2,3 \
        --far-budgets 4,8 \
        --n-eval 50 \
        --output-dir bpa_v6_results
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v3_experiment import (
    RunMetrics,
    build_local_mask,
    compute_gate_features,
    compute_ppl,
    get_text_batch,
    load_text_data,
)
from scripts.bpa_v4_gate import V4Gate, find_budget_threshold
from scripts.bpa_v5_experiment import load_v4_gate, select_far_chunks
from utils.kv_accounting import compute_kv_accounting
from utils.ra_value_tracker import RAValueTracker, load_surgical_heads


@dataclass
class V6RunMetrics:
    """Extended metrics for BPA v6 runs."""

    # Identity
    variant: str
    seq_len: int
    seed: int
    far_budget: int
    ra_update_period: int  # 1 = every batch, N = every Nth batch

    # Quality
    ppl_mean: float
    ppl_std: float

    # Gate
    enabled_rate: float

    # KV accounting
    tokens_per_query_mean: float
    effective_kept_tokens: float
    kv_bytes_written_per_token: float
    kv_bytes_read_per_token: float
    kv_bytes_total_per_token: float
    peak_kv_bytes: float

    # FLOPs
    flops_proxy: float
    flops_relative: float

    # Timing
    wall_ms_per_token: float
    gate_ms_per_token: float  # gate overhead isolated
    fwd_ms_per_token: float  # forward pass only
    tokens_per_sec: float

    # Bookkeeping
    tokens_seen: int
    n_eval_batches: int


@torch.no_grad()
def compute_gate_features_fast(
    model: GPT2_BPA,
    idx: torch.Tensor,
    local_window: int,
    n_layers_sample: int = 3,
) -> np.ndarray:
    """Fast gate feature extraction using only a subset of layers.

    Instead of running all 12 layers to get gate features, sample
    only n_layers_sample layers (first, middle, last). This reduces
    gate overhead by ~4x for GPT-2 124M.

    Args:
        model: GPT2_BPA model
        idx: [B, T] input tokens
        local_window: local window size
        n_layers_sample: how many layers to sample (default 3)

    Returns:
        features array [n_valid, 7]
    """
    B, T = idx.shape
    device = idx.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head
    n_valid = max(T - local_window, 0)

    if n_valid == 0:
        return np.zeros((0, 7))

    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )
    local_mask = build_local_mask(T, local_window, device)
    combined_mask = causal_mask | local_mask

    n_layers = len(model.transformer.h)
    # Sample layers: first, middle, last
    if n_layers_sample >= n_layers:
        layer_indices = list(range(n_layers))
    else:
        step = max(1, (n_layers - 1) / (n_layers_sample - 1))
        layer_indices = [round(i * step) for i in range(n_layers_sample)]
        layer_indices = sorted(set(layer_indices))

    all_feats = np.zeros((n_valid, 7))
    n_counted = 0

    for li, block in enumerate(model.transformer.h):
        h_ln = block.ln_1(x)
        attn = block.attn

        if li in layer_indices:
            qkv = attn.c_attn(h_ln)
            q, k, v = qkv.split(n_embd, dim=2)
            q_heads = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k_heads = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v_heads = v.view(B, T, n_head, head_dim).transpose(1, 2)

            scale = 1.0 / (head_dim**0.5)
            scores = (q_heads @ k_heads.transpose(-2, -1)) * scale
            scores = scores.masked_fill(
                combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            local_attn_probs = F.softmax(scores, dim=-1)

            for ti, t in enumerate(range(local_window, T)):
                local_start = max(0, t - local_window + 1)
                local_end = t + 1
                pos_norm = t / max(T - 1, 1)
                boundary_band = min(8, local_window // 4)

                w = local_attn_probs[:, :, t, local_start:local_end]
                w_np = w.detach().cpu().float().numpy()
                w_mean = w_np.mean(axis=(0, 1))

                if len(w_mean) > 0:
                    entropy = float(-(w_mean * np.log(w_mean + 1e-12)).sum())
                    max_attn = float(w_mean.max())
                    edge_mass = float(
                        w_mean[:boundary_band].sum() + w_mean[-boundary_band:].sum()
                    )
                    mid_mass = float(
                        w_mean[boundary_band:-boundary_band].sum()
                        if len(w_mean) > 2 * boundary_band
                        else 0.0
                    )
                    concentration = float(np.sort(w_mean)[-min(4, len(w_mean)) :].sum())
                else:
                    entropy = max_attn = edge_mass = mid_mass = 0.0
                    concentration = 0.0

                all_feats[ti] += np.array(
                    [
                        entropy,
                        max_attn,
                        edge_mass,
                        mid_mass,
                        concentration,
                        pos_norm,
                        float(len(w_mean)) / local_window,
                    ]
                )
                n_counted += 1

            attn_out = (
                (local_attn_probs @ v_heads).transpose(1, 2).contiguous().view(B, T, -1)
            )
            attn_out = attn.resid_dropout(attn.c_proj(attn_out))
        else:
            # Skip feature extraction, use standard RGSA forward
            out = attn(h_ln)
            attn_out = out[0] if isinstance(out, tuple) else out
        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    if n_counted > 0:
        all_feats /= len(layer_indices)

    return all_feats


@torch.no_grad()
def forward_with_chunk_selection_v6(
    model: GPT2_BPA,
    idx: torch.Tensor,
    gate_decisions: np.ndarray,
    local_window: int,
    chunk_size: int,
    strategy: str,
    far_budget: int,
    ra_chunk_values: Optional[torch.Tensor] = None,
    tau: float = 4.0,
    rng: Optional[np.random.RandomState] = None,
    tracker: Optional[RAValueTracker] = None,
) -> Tuple[torch.Tensor, dict]:
    """Forward pass with chunk selection and RA tracking.

    Same as v5 but with timing instrumentation for overhead analysis.
    """
    B, T = idx.shape
    device = idx.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head
    n_chunks = (T + chunk_size - 1) // chunk_size

    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )

    local_restrict = torch.zeros(T, T, dtype=torch.bool, device=device)
    chunks_attended = []

    for ti, t in enumerate(range(local_window, T)):
        far_end = max(0, t - local_window + 1)
        if far_end > 0:
            if gate_decisions[ti]:
                selected = select_far_chunks(
                    strategy=strategy,
                    n_chunks=n_chunks,
                    far_budget=far_budget,
                    ra_chunk_values=ra_chunk_values,
                    query_pos=t,
                    chunk_size=chunk_size,
                    local_window=local_window,
                    tau=tau,
                    rng=rng,
                )
                chunks_attended.append(len(selected))

                local_restrict[t, :far_end] = True
                for c in selected:
                    c_start = c * chunk_size
                    c_end = min((c + 1) * chunk_size, far_end)
                    if c_start < far_end:
                        local_restrict[t, c_start:c_end] = False
            else:
                local_restrict[t, :far_end] = True
                chunks_attended.append(0)

    combined_mask = causal_mask | local_restrict

    if tracker is not None:
        tracker.reset()

    for layer_idx, block in enumerate(model.transformer.h):
        h = block.ln_1(x)
        attn = block.attn

        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(scores, dim=-1)

        if tracker is not None and layer_idx in tracker.heads_by_layer:
            tracker.accumulate(layer_idx, attn_weights)

        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))

        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)

    info = {
        "chunks_attended_mean": (
            float(np.mean(chunks_attended)) if chunks_attended else 0.0
        ),
    }
    if tracker is not None:
        info["ra_stats"] = tracker.get_stats()

    return logits, info


def evaluate_v6_variant(
    model: GPT2_BPA,
    text_data: np.ndarray,
    variant: str,
    strategy: str,
    seq_len: int,
    seed: int,
    n_eval: int,
    batch_size: int,
    local_window: int,
    chunk_size: int,
    far_budget: int,
    gate=None,
    feat_mean=None,
    feat_std=None,
    target_enabled_rate: float = 0.7,
    tau: float = 4.0,
    tracker: Optional[RAValueTracker] = None,
    ra_update_period: int = 1,
    fast_gate: bool = False,
) -> V6RunMetrics:
    """Evaluate a v6 variant with timing instrumentation.

    Args:
        ra_update_period: update RA stats every N batches (1=always)
        fast_gate: use fast gate features (3 layers instead of 12)
    """
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    rng = np.random.RandomState(seed)

    ppls = []
    tokens_per_query_list = []
    enabled_rates = []
    wall_times = []
    gate_times = []
    fwd_times = []
    tokens_seen = 0

    # Warmup for RA values
    warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
    if tracker is not None:
        _, _ = forward_with_chunk_selection_v6(
            model,
            warmup_idx,
            np.ones(max(seq_len - local_window, 0), dtype=bool),
            local_window,
            chunk_size,
            "recency",
            far_budget,
            tracker=tracker,
        )
        ra_values = tracker.get_chunk_values()
    else:
        ra_values = None

    for i in range(n_eval):
        idx = get_text_batch(text_data, batch_size, seq_len, rng)
        B, T = idx.shape
        n_valid = max(T - local_window, 0)

        t_total_start = time.perf_counter()

        # Gate decisions
        t_gate_start = time.perf_counter()
        if variant == "V0_dense":
            gate_decisions = np.ones(n_valid, dtype=bool)
        elif variant == "V1_local_only":
            gate_decisions = np.zeros(n_valid, dtype=bool)
        elif gate is not None and n_valid > 0:
            if fast_gate:
                feats = compute_gate_features_fast(
                    model, idx, local_window, n_layers_sample=3
                )
            else:
                feats = compute_gate_features(model, idx, local_window)
            feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
            gate_logits = (
                gate(torch.tensor(feats_norm, dtype=torch.float32))
                .detach()
                .numpy()
                .flatten()
            )
            gate_probs = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -500, 500)))
            threshold = find_budget_threshold(gate_probs, target_enabled_rate)
            gate_decisions = gate_probs >= threshold
        else:
            gate_decisions = np.ones(n_valid, dtype=bool)
        t_gate_end = time.perf_counter()

        enabled_rate_batch = float(gate_decisions.mean()) if n_valid > 0 else 0.0

        # Forward pass
        t_fwd_start = time.perf_counter()

        # Sparse RA update: only update tracker every ra_update_period batches
        use_tracker = tracker if (i % ra_update_period == 0) else None

        logits, info = forward_with_chunk_selection_v6(
            model,
            idx,
            gate_decisions,
            local_window,
            chunk_size,
            strategy,
            far_budget,
            ra_chunk_values=ra_values,
            tau=tau,
            rng=rng,
            tracker=use_tracker,
        )

        # Update RA values
        if use_tracker is not None and tracker is not None:
            ra_values = tracker.get_chunk_values()

        t_fwd_end = time.perf_counter()
        t_total_end = time.perf_counter()

        # Effective kept tokens
        total_attended = 0
        for t in range(T):
            if t < local_window:
                total_attended += min(t + 1, T)
            else:
                ti = t - local_window
                if gate_decisions[ti]:
                    total_attended += local_window + far_budget * chunk_size
                else:
                    total_attended += local_window
        kept = total_attended / T if T > 0 else 0.0

        ppl = compute_ppl(logits, idx)
        ppls.append(ppl)
        tokens_per_query_list.append(kept)
        enabled_rates.append(enabled_rate_batch)

        total_ms = (t_total_end - t_total_start) * 1000
        gate_ms = (t_gate_end - t_gate_start) * 1000
        fwd_ms = (t_fwd_end - t_fwd_start) * 1000
        wall_times.append(total_ms)
        gate_times.append(gate_ms)
        fwd_times.append(fwd_ms)
        tokens_seen += B * T

        if (i + 1) % 10 == 0 or i == 0:
            total_tokens = B * T
            print(
                f"    [{i+1}/{n_eval}] "
                f"PPL={ppl:.1f} "
                f"kept={kept:.0f} "
                f"rate={enabled_rate_batch:.3f} "
                f"gate={gate_ms:.0f}ms "
                f"fwd={fwd_ms:.0f}ms "
                f"total={total_ms:.0f}ms"
            )

    mean_enabled_rate = float(np.mean(enabled_rates))
    mean_kept = float(np.mean(tokens_per_query_list))
    total_tokens_per_batch = batch_size * seq_len

    kv = compute_kv_accounting(
        seq_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        local_window=local_window,
        enabled_rate=mean_enabled_rate,
        far_budget=far_budget * chunk_size if far_budget < 9999 else None,
        bytes_per_elem=2,
    )

    dense_flops = seq_len * d_model * n_layers
    ms_per_token = float(np.median(wall_times)) / total_tokens_per_batch
    gate_ms_tok = float(np.median(gate_times)) / total_tokens_per_batch
    fwd_ms_tok = float(np.median(fwd_times)) / total_tokens_per_batch

    return V6RunMetrics(
        variant=variant,
        seq_len=seq_len,
        seed=seed,
        far_budget=far_budget,
        ra_update_period=ra_update_period,
        ppl_mean=float(np.mean(ppls)),
        ppl_std=float(np.std(ppls)),
        enabled_rate=mean_enabled_rate,
        tokens_per_query_mean=mean_kept,
        effective_kept_tokens=mean_kept,
        kv_bytes_written_per_token=kv.kv_bytes_written_per_token,
        kv_bytes_read_per_token=kv.kv_bytes_read_per_token,
        kv_bytes_total_per_token=kv.kv_bytes_total_per_token,
        peak_kv_bytes=kv.peak_kv_bytes,
        flops_proxy=kv.flops_proxy,
        flops_relative=(kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0),
        wall_ms_per_token=ms_per_token,
        gate_ms_per_token=gate_ms_tok,
        fwd_ms_per_token=fwd_ms_tok,
        tokens_per_sec=1000.0 / ms_per_token if ms_per_token > 0 else 0.0,
        tokens_seen=tokens_seen,
        n_eval_batches=n_eval,
    )


def run_v6_grid(
    checkpoint: str,
    seq_lens: List[int],
    seeds: List[int],
    far_budgets: List[int],
    n_eval: int,
    batch_size: int,
    local_window: int,
    chunk_size: int,
    top_b: int,
    v4_gate_dir: str,
    text_data_path: str,
    output_dir: str,
    target_enabled_rate: float = 0.7,
    tau: float = 4.0,
    ra_update_period: int = 1,
    fast_gate: bool = False,
    variant_set: str = "full",
) -> Dict:
    """Run the v6 experiment grid.

    Args:
        variant_set: "full" for all variants, "minimal" for dense+ra_value+recency
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "per_run_metrics"), exist_ok=True)

    text_data = load_text_data(text_data_path)
    print(f"Text data: {len(text_data)} tokens from {text_data_path}")

    surgical_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "ra_surgical_gpt2.json",
    )
    surgical_heads = load_surgical_heads(surgical_path)
    print(f"Surgical heads: {len(surgical_heads)} heads from {surgical_path}")

    if variant_set == "minimal":
        strategies = ["recency", "ra_value"]
    else:
        strategies = ["recency", "random", "ra_value", "ra_blend"]

    # Save config
    config = {
        "checkpoint": checkpoint,
        "seq_lens": seq_lens,
        "seeds": seeds,
        "far_budgets": far_budgets,
        "n_eval": n_eval,
        "batch_size": batch_size,
        "local_window": local_window,
        "chunk_size": chunk_size,
        "top_b": top_b,
        "target_enabled_rate": target_enabled_rate,
        "tau": tau,
        "ra_update_period": ra_update_period,
        "fast_gate": fast_gate,
        "variant_set": variant_set,
        "strategies": strategies,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_results = []

    for seq_len in seq_lens:
        # Limit seeds for very long sequences
        run_seeds = seeds
        if seq_len >= 2048 and len(seeds) > 2:
            run_seeds = seeds[:2]
            print(f"\nNote: using seeds {run_seeds} for L={seq_len}")

        for seed in run_seeds:
            print(f"\n{'='*60}")
            print(f"SEQ_LEN={seq_len}, SEED={seed}")
            print(f"{'='*60}")

            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            vocab_size = ckpt["model"]["transformer.wte.weight"].shape[0]

            cfg = BPAConfig(
                n_layer=12,
                n_head=12,
                n_embd=768,
                local_window=local_window,
                chunk_size=chunk_size,
                top_b=top_b,
                vocab_size=vocab_size,
                block_size=max(seq_len, 1024),
            )
            model = GPT2_BPA(cfg)
            model_sd = {
                k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()
            }
            model.load_state_dict(model_sd, strict=False)
            model.eval()
            del ckpt

            device = torch.device("cpu")
            model = model.to(device)

            gate, feat_mean, feat_std = load_v4_gate(v4_gate_dir, seed)

            # V0: Dense baseline (only once per seq_len/seed, not per far_budget)
            print(f"\n  V0_dense (full causal)")
            tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
            metrics = evaluate_v6_variant(
                model,
                text_data,
                "V0_dense",
                "recency",
                seq_len,
                seed,
                n_eval,
                batch_size,
                local_window,
                chunk_size,
                far_budget=9999,
                target_enabled_rate=1.0,
                tracker=tracker,
                ra_update_period=1,
                fast_gate=False,
            )
            all_results.append(metrics)
            _save_run(output_dir, metrics)

            # V1: Local only (only once per seq_len/seed)
            if variant_set == "full":
                print(f"\n  V1_local_only")
                metrics = evaluate_v6_variant(
                    model,
                    text_data,
                    "V1_local_only",
                    "recency",
                    seq_len,
                    seed,
                    n_eval,
                    batch_size,
                    local_window,
                    chunk_size,
                    far_budget=0,
                    target_enabled_rate=0.0,
                    ra_update_period=1,
                    fast_gate=False,
                )
                all_results.append(metrics)
                _save_run(output_dir, metrics)

            # Selection strategies at each far_budget
            for fb in far_budgets:
                for strat in strategies:
                    variant = f"V6_{strat}_fb{fb}"
                    print(
                        f"\n  {variant} (budget={fb}, rate={target_enabled_rate}, "
                        f"ra_period={ra_update_period}, fast_gate={fast_gate})"
                    )
                    tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
                    metrics = evaluate_v6_variant(
                        model,
                        text_data,
                        variant,
                        strat,
                        seq_len,
                        seed,
                        n_eval,
                        batch_size,
                        local_window,
                        chunk_size,
                        far_budget=fb,
                        gate=gate,
                        feat_mean=feat_mean,
                        feat_std=feat_std,
                        target_enabled_rate=target_enabled_rate,
                        tau=tau,
                        tracker=tracker if strat in ("ra_value", "ra_blend") else None,
                        ra_update_period=ra_update_period,
                        fast_gate=fast_gate,
                    )
                    all_results.append(metrics)
                    _save_run(output_dir, metrics)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results
    results_path = os.path.join(output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved {len(all_results)} results to {results_path}")

    return {"results": [asdict(r) for r in all_results]}


def _save_run(output_dir: str, metrics: V6RunMetrics):
    """Save per-run metrics to JSON."""
    fname = f"{metrics.variant}_L{metrics.seq_len}_S{metrics.seed}.json"
    path = os.path.join(output_dir, "per_run_metrics", fname)
    with open(path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)


def run_overhead_benchmark(
    checkpoint: str,
    v4_gate_dir: str,
    text_data_path: str,
    output_dir: str,
    seq_len: int = 1024,
    n_eval: int = 20,
    batch_size: int = 1,
    local_window: int = 256,
    chunk_size: int = 64,
    top_b: int = 8,
    far_budget: int = 4,
    target_enabled_rate: float = 0.7,
) -> Dict:
    """Benchmark gate overhead with different reduction approaches.

    Tests:
      1. Baseline (v5-style): all layers, every batch
      2. Fast gate: 3 layers only
      3. Sparse RA: update every 4 batches
      4. Sparse RA: update every 8 batches
      5. Combined: fast gate + sparse RA(4)
    """
    os.makedirs(output_dir, exist_ok=True)

    text_data = load_text_data(text_data_path)
    surgical_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "ra_surgical_gpt2.json",
    )
    surgical_heads = load_surgical_heads(surgical_path)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab_size = ckpt["model"]["transformer.wte.weight"].shape[0]

    cfg = BPAConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        local_window=local_window,
        chunk_size=chunk_size,
        top_b=top_b,
        vocab_size=vocab_size,
        block_size=max(seq_len, 1024),
    )
    model = GPT2_BPA(cfg)
    model_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    del ckpt

    device = torch.device("cpu")
    model = model.to(device)

    gate, feat_mean, feat_std = load_v4_gate(v4_gate_dir, seed=1)

    configs = [
        ("baseline", False, 1),
        ("fast_gate", True, 1),
        ("sparse_ra_4", False, 4),
        ("sparse_ra_8", False, 8),
        ("fast+sparse4", True, 4),
    ]

    results = []
    for name, fast, ra_period in configs:
        print(f"\n  Overhead benchmark: {name}")
        tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
        metrics = evaluate_v6_variant(
            model,
            text_data,
            f"overhead_{name}",
            "ra_value",
            seq_len,
            seed=1,
            n_eval=n_eval,
            batch_size=batch_size,
            local_window=local_window,
            chunk_size=chunk_size,
            far_budget=far_budget,
            gate=gate,
            feat_mean=feat_mean,
            feat_std=feat_std,
            target_enabled_rate=target_enabled_rate,
            tracker=tracker,
            ra_update_period=ra_period,
            fast_gate=fast,
        )
        entry = {
            "name": name,
            "fast_gate": fast,
            "ra_update_period": ra_period,
            "ppl_mean": metrics.ppl_mean,
            "ppl_std": metrics.ppl_std,
            "wall_ms_per_token": metrics.wall_ms_per_token,
            "gate_ms_per_token": metrics.gate_ms_per_token,
            "fwd_ms_per_token": metrics.fwd_ms_per_token,
        }
        results.append(entry)
        print(
            f"    PPL={metrics.ppl_mean:.1f} "
            f"gate={metrics.gate_ms_per_token:.4f}ms/tok "
            f"fwd={metrics.fwd_ms_per_token:.4f}ms/tok "
            f"total={metrics.wall_ms_per_token:.4f}ms/tok"
        )

    bench_path = os.path.join(output_dir, "overhead_bench.json")
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOverhead benchmark: {bench_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="BPA v6 experiment")
    parser.add_argument(
        "--checkpoint",
        default="test_matrix_results_20260206_184612/"
        "gpt2_adamwspam_none_none/final_model_stepV0.pt",
    )
    parser.add_argument("--gate-dir", default="bpa_v4_gate_results")
    parser.add_argument("--seq-lens", default="512,1024,2048")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--far-budgets", default="4,8")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--top-b", type=int, default=8)
    parser.add_argument("--target-rate", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--ra-update-period", type=int, default=1)
    parser.add_argument("--fast-gate", action="store_true")
    parser.add_argument("--variant-set", default="full", choices=["full", "minimal"])
    parser.add_argument("--output-dir", default="bpa_v6_results")
    parser.add_argument("--text-data", default="data/finewebedu/val.bin")
    parser.add_argument(
        "--overhead-only",
        action="store_true",
        help="Run only the overhead benchmark, not the full grid",
    )

    args = parser.parse_args()

    if args.overhead_only:
        run_overhead_benchmark(
            checkpoint=args.checkpoint,
            v4_gate_dir=args.gate_dir,
            text_data_path=args.text_data,
            output_dir=args.output_dir,
            seq_len=1024,
            n_eval=20,
            batch_size=args.batch_size,
            local_window=args.local_window,
            chunk_size=args.chunk_size,
            top_b=args.top_b,
            far_budget=4,
            target_enabled_rate=args.target_rate,
        )
    else:
        seq_lens = [int(x) for x in args.seq_lens.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        far_budgets = [int(x) for x in args.far_budgets.split(",")]

        run_v6_grid(
            checkpoint=args.checkpoint,
            seq_lens=seq_lens,
            seeds=seeds,
            far_budgets=far_budgets,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            local_window=args.local_window,
            chunk_size=args.chunk_size,
            top_b=args.top_b,
            v4_gate_dir=args.gate_dir,
            text_data_path=args.text_data,
            output_dir=args.output_dir,
            target_enabled_rate=args.target_rate,
            tau=args.tau,
            ra_update_period=args.ra_update_period,
            fast_gate=args.fast_gate,
            variant_set=args.variant_set,
        )


if __name__ == "__main__":
    main()
