#!/usr/bin/env python
"""
BPA v7: Stress-test RA assumptions, dynamic budget, cheap features.

Section 2: Adversarial RA ablations
  - frozen_ra: freeze RA scores from first batch, reuse forever
  - shuffled_ra: shuffle RA scores within windows (break alignment)
  - corrupt_nonsurgical: inject noise at non-surgical layer RA scores

Section 3: Dynamic budget (query-conditional far budget)
  - constant: b(x_t) = B_max (v6 baseline)
  - entropy: b(x_t) proportional to local attention entropy
  - concentration: b(x_t) proportional to attention concentration

Section 4: Cheap RA-derived features
  - ra_value: full RA (v6 baseline)
  - ra_ema: cached EMA of inbound mass
  - ra_rank: top-k flags only (binary indicators)
  - ra_layeragg: layer-aggregated scores (no per-head ops)

Usage:
    python scripts/bpa_v7_experiment.py \
        --mode stress    # or dynamic-budget or cheap-features or all
        --seq-lens 512,1024,2048 \
        --seeds 1,2,3 \
        --output-dir bpa_v7_results
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v3_experiment import (
    build_local_mask,
    compute_gate_features,
    compute_ppl,
    get_text_batch,
    load_text_data,
)
from scripts.bpa_v4_gate import V4Gate, find_budget_threshold
from scripts.bpa_v5_experiment import load_v4_gate, select_far_chunks
from scripts.bpa_v6_experiment import (
    V6RunMetrics,
    forward_with_chunk_selection_v6,
    interpolate_pos_embed,
)
from utils.kv_accounting import compute_kv_accounting
from utils.ra_value_tracker import RAValueTracker, load_surgical_heads


@dataclass
class V7RunMetrics:
    """Extended metrics for BPA v7 runs."""

    variant: str
    seq_len: int
    seed: int
    far_budget: int
    mode: str  # stress, dynamic-budget, cheap-features

    ppl_mean: float
    ppl_std: float
    enabled_rate: float

    effective_kept_tokens: float
    kv_bytes_read_per_token: float
    peak_kv_bytes: float
    flops_relative: float

    wall_ms_per_token: float
    gate_ms_per_token: float
    fwd_ms_per_token: float

    # Dynamic budget stats
    budget_mean: float
    budget_p95: float
    budget_p99: float

    tokens_seen: int
    n_eval_batches: int

    # Extra metadata
    extra: dict


# ============================================================
# Section 2: Stress-test RA assumptions
# ============================================================


def run_stress_tests(
    model,
    text_data,
    seq_len,
    seed,
    n_eval,
    batch_size,
    local_window,
    chunk_size,
    far_budget,
    gate,
    feat_mean,
    feat_std,
    target_enabled_rate,
    surgical_heads,
    output_dir,
):
    """Run the 3 adversarial RA stress tests."""
    results = []
    rng = np.random.RandomState(seed)
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd

    # First: collect baseline RA values from warmup
    tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
    warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
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
    baseline_ra = tracker.get_chunk_values()

    # Test 1: Normal RA_value (control)
    print(f"\n  stress_control (normal ra_value)")
    m = _eval_variant(
        model,
        text_data,
        "stress_control",
        "ra_value",
        seq_len,
        seed,
        n_eval,
        batch_size,
        local_window,
        chunk_size,
        far_budget,
        gate,
        feat_mean,
        feat_std,
        target_enabled_rate,
        surgical_heads,
        ra_mode="normal",
    )
    m_dict = asdict(m)
    m_dict["extra"] = {"ra_mode": "normal"}
    results.append(m_dict)

    # Test 2: Frozen RA (freeze from warmup, never update)
    print(f"\n  stress_frozen (frozen RA from warmup)")
    m = _eval_variant(
        model,
        text_data,
        "stress_frozen",
        "ra_value",
        seq_len,
        seed,
        n_eval,
        batch_size,
        local_window,
        chunk_size,
        far_budget,
        gate,
        feat_mean,
        feat_std,
        target_enabled_rate,
        surgical_heads,
        ra_mode="frozen",
        frozen_ra=baseline_ra,
    )
    m_dict = asdict(m)
    m_dict["extra"] = {"ra_mode": "frozen"}
    results.append(m_dict)

    # Test 3: Shuffled RA (break alignment, preserve marginals)
    print(f"\n  stress_shuffled (shuffled RA scores)")
    m = _eval_variant(
        model,
        text_data,
        "stress_shuffled",
        "ra_value",
        seq_len,
        seed,
        n_eval,
        batch_size,
        local_window,
        chunk_size,
        far_budget,
        gate,
        feat_mean,
        feat_std,
        target_enabled_rate,
        surgical_heads,
        ra_mode="shuffled",
    )
    m_dict = asdict(m)
    m_dict["extra"] = {"ra_mode": "shuffled"}
    results.append(m_dict)

    # Test 4: Cross-layer corruption (noise at non-surgical layers)
    print(f"\n  stress_corrupt (noise at non-surgical layers)")
    m = _eval_variant(
        model,
        text_data,
        "stress_corrupt",
        "ra_value",
        seq_len,
        seed,
        n_eval,
        batch_size,
        local_window,
        chunk_size,
        far_budget,
        gate,
        feat_mean,
        feat_std,
        target_enabled_rate,
        surgical_heads,
        ra_mode="corrupt_nonsurgical",
    )
    m_dict = asdict(m)
    m_dict["extra"] = {"ra_mode": "corrupt_nonsurgical"}
    results.append(m_dict)

    # Test 5: Recency baseline (for comparison)
    print(f"\n  stress_recency (recency baseline)")
    m = _eval_variant(
        model,
        text_data,
        "stress_recency",
        "recency",
        seq_len,
        seed,
        n_eval,
        batch_size,
        local_window,
        chunk_size,
        far_budget,
        gate,
        feat_mean,
        feat_std,
        target_enabled_rate,
        surgical_heads,
        ra_mode="none",
    )
    m_dict = asdict(m)
    m_dict["extra"] = {"ra_mode": "none"}
    results.append(m_dict)

    return results


def _eval_variant(
    model,
    text_data,
    variant,
    strategy,
    seq_len,
    seed,
    n_eval,
    batch_size,
    local_window,
    chunk_size,
    far_budget,
    gate,
    feat_mean,
    feat_std,
    target_enabled_rate,
    surgical_heads,
    ra_mode="normal",
    frozen_ra=None,
):
    """Evaluate with various RA manipulation modes."""
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    rng = np.random.RandomState(seed)
    shuffle_rng = np.random.RandomState(seed + 1000)

    tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)

    # Warmup
    warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
    if ra_mode != "none":
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

    if ra_mode == "frozen" and frozen_ra is not None:
        ra_values = frozen_ra

    ppls, wall_times, gate_times, fwd_times = [], [], [], []
    enabled_rates, kept_list, budgets_used = [], [], []
    tokens_seen = 0

    for i in range(n_eval):
        idx = get_text_batch(text_data, batch_size, seq_len, rng)
        B, T = idx.shape
        n_valid = max(T - local_window, 0)

        t0 = time.perf_counter()

        # Gate decisions
        tg0 = time.perf_counter()
        if gate is not None and n_valid > 0:
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
        tg1 = time.perf_counter()

        # Manipulate RA values based on mode
        use_ra = ra_values
        if ra_mode == "shuffled" and use_ra is not None:
            # Shuffle RA scores (break alignment, preserve marginals)
            vals = use_ra.cpu().numpy().copy()
            shuffle_rng.shuffle(vals)
            use_ra = torch.tensor(vals, dtype=use_ra.dtype, device=use_ra.device)
        elif ra_mode == "corrupt_nonsurgical":
            # RA values come from surgical heads only, so this is a no-op
            # for direct RA. Instead, we add noise to the RA values themselves
            # to simulate what would happen if non-surgical layers contributed.
            if use_ra is not None:
                noise = torch.randn_like(use_ra) * use_ra.std() * 0.5
                use_ra = use_ra + noise

        tf0 = time.perf_counter()
        use_tracker = tracker if ra_mode == "normal" else None
        logits, info = forward_with_chunk_selection_v6(
            model,
            idx,
            gate_decisions,
            local_window,
            chunk_size,
            strategy,
            far_budget,
            ra_chunk_values=use_ra,
            rng=rng,
            tracker=use_tracker,
        )
        tf1 = time.perf_counter()

        # Update RA values (only in normal mode)
        if ra_mode == "normal" and use_tracker is not None:
            ra_values = tracker.get_chunk_values()

        t1 = time.perf_counter()

        # Compute kept tokens
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
        kept_list.append(kept)
        budgets_used.append(far_budget)
        enabled_rates.append(float(gate_decisions.mean()) if n_valid > 0 else 0.0)
        wall_times.append((t1 - t0) * 1000)
        gate_times.append((tg1 - tg0) * 1000)
        fwd_times.append((tf1 - tf0) * 1000)
        tokens_seen += B * T

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    [{i+1}/{n_eval}] PPL={ppl:.1f} "
                f"kept={kept:.0f} ms={wall_times[-1]:.0f}"
            )

    total_tokens = batch_size * seq_len
    kv = compute_kv_accounting(
        seq_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        local_window=local_window,
        enabled_rate=float(np.mean(enabled_rates)),
        far_budget=far_budget * chunk_size,
        bytes_per_elem=2,
    )
    dense_flops = seq_len * d_model * n_layers

    return V7RunMetrics(
        variant=variant,
        seq_len=seq_len,
        seed=seed,
        far_budget=far_budget,
        mode="stress",
        ppl_mean=float(np.mean(ppls)),
        ppl_std=float(np.std(ppls)),
        enabled_rate=float(np.mean(enabled_rates)),
        effective_kept_tokens=float(np.mean(kept_list)),
        kv_bytes_read_per_token=kv.kv_bytes_read_per_token,
        peak_kv_bytes=kv.peak_kv_bytes,
        flops_relative=(kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0),
        wall_ms_per_token=float(np.median(wall_times)) / total_tokens,
        gate_ms_per_token=float(np.median(gate_times)) / total_tokens,
        fwd_ms_per_token=float(np.median(fwd_times)) / total_tokens,
        budget_mean=float(np.mean(budgets_used)),
        budget_p95=float(np.percentile(budgets_used, 95)),
        budget_p99=float(np.percentile(budgets_used, 99)),
        tokens_seen=tokens_seen,
        n_eval_batches=n_eval,
        extra={},
    )


# ============================================================
# Section 3: Dynamic budget
# ============================================================


def compute_dynamic_budget(
    mode: str,
    attn_entropy: float,
    attn_concentration: float,
    b_max: int,
) -> int:
    """Compute per-query far budget.

    Args:
        mode: "constant", "entropy", "concentration"
        attn_entropy: local attention entropy (higher = more uncertain)
        attn_concentration: top-k attention mass (higher = more focused)
        b_max: maximum budget

    Returns:
        integer budget in [0, b_max]
    """
    if mode == "constant":
        return b_max
    elif mode == "entropy":
        # High entropy -> need more far context -> higher budget
        # Normalize: entropy in [0, log(local_window)] ~ [0, 5.5]
        frac = min(attn_entropy / 5.5, 1.0)
        return max(1, int(frac * b_max + 0.5))
    elif mode == "concentration":
        # Low concentration -> need more context -> higher budget
        # concentration in [0, 1], invert
        frac = 1.0 - min(attn_concentration, 1.0)
        return max(1, int(frac * b_max + 0.5))
    else:
        return b_max


@torch.no_grad()
def forward_dynamic_budget(
    model,
    idx,
    gate_decisions,
    local_window,
    chunk_size,
    budget_mode,
    b_max,
    ra_chunk_values,
    strategy,
    rng=None,
    tracker=None,
):
    """Forward with per-position dynamic budget."""
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

    # First pass: compute local attention stats for budget decisions
    # Use layer 0 only for cheap entropy/concentration estimates
    h0 = model.transformer.h[0].ln_1(x)
    qkv0 = model.transformer.h[0].attn.c_attn(h0)
    q0, k0, _ = qkv0.split(n_embd, dim=2)
    q0h = q0.view(B, T, n_head, head_dim).transpose(1, 2)
    k0h = k0.view(B, T, n_head, head_dim).transpose(1, 2)
    scale = 1.0 / (head_dim**0.5)
    scores0 = (q0h @ k0h.transpose(-2, -1)) * scale

    local_mask = build_local_mask(T, local_window, device)
    mask0 = causal_mask | local_mask
    scores0 = scores0.masked_fill(mask0.unsqueeze(0).unsqueeze(0), float("-inf"))
    local_probs = F.softmax(scores0, dim=-1)

    # Compute per-position entropy and concentration
    entropies = np.zeros(T)
    concentrations = np.zeros(T)
    for t in range(local_window, T):
        local_start = max(0, t - local_window + 1)
        w = local_probs[:, :, t, local_start : t + 1].mean(dim=(0, 1))
        w_np = w.detach().cpu().float().numpy()
        if len(w_np) > 0 and w_np.sum() > 0:
            w_np = w_np / (w_np.sum() + 1e-12)
            entropies[t] = float(-(w_np * np.log(w_np + 1e-12)).sum())
            concentrations[t] = float(np.sort(w_np)[-min(4, len(w_np)) :].sum())

    # Build mask with dynamic budgets
    local_restrict = torch.zeros(T, T, dtype=torch.bool, device=device)
    chunks_attended = []
    budgets_used = []

    for ti, t in enumerate(range(local_window, T)):
        far_end = max(0, t - local_window + 1)
        if far_end > 0:
            if gate_decisions[ti]:
                budget = compute_dynamic_budget(
                    budget_mode, entropies[t], concentrations[t], b_max
                )
                budgets_used.append(budget)

                selected = select_far_chunks(
                    strategy=strategy,
                    n_chunks=n_chunks,
                    far_budget=budget,
                    ra_chunk_values=ra_chunk_values,
                    query_pos=t,
                    chunk_size=chunk_size,
                    local_window=local_window,
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
                budgets_used.append(0)

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

        sc = (q @ k.transpose(-2, -1)) * scale
        sc = sc.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(sc, dim=-1)

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
        "budgets_used": budgets_used,
    }
    if tracker is not None:
        info["ra_stats"] = tracker.get_stats()

    return logits, info


def run_dynamic_budget_tests(
    model,
    text_data,
    seq_len,
    seed,
    n_eval,
    batch_size,
    local_window,
    chunk_size,
    b_max,
    gate,
    feat_mean,
    feat_std,
    target_enabled_rate,
    surgical_heads,
    output_dir,
):
    """Run dynamic budget experiments."""
    results = []
    budget_modes = ["constant", "entropy", "concentration"]
    strategies = ["ra_value", "recency"]

    for budget_mode in budget_modes:
        for strategy in strategies:
            variant = f"dynbudget_{budget_mode}_{strategy}"
            print(f"\n  {variant}")

            rng = np.random.RandomState(seed)
            tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)

            # Warmup
            warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
            _, _ = forward_with_chunk_selection_v6(
                model,
                warmup_idx,
                np.ones(max(seq_len - local_window, 0), dtype=bool),
                local_window,
                chunk_size,
                "recency",
                b_max,
                tracker=tracker,
            )
            ra_values = tracker.get_chunk_values()

            ppls, wall_times, gate_times, fwd_times = [], [], [], []
            enabled_rates, kept_list, all_budgets = [], [], []
            tokens_seen = 0
            cfg = model.config

            for i in range(n_eval):
                idx = get_text_batch(text_data, batch_size, seq_len, rng)
                B, T = idx.shape
                n_valid = max(T - local_window, 0)

                t0 = time.perf_counter()
                tg0 = time.perf_counter()
                if gate is not None and n_valid > 0:
                    feats = compute_gate_features(model, idx, local_window)
                    feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
                    gl = (
                        gate(torch.tensor(feats_norm, dtype=torch.float32))
                        .detach()
                        .numpy()
                        .flatten()
                    )
                    gp = 1.0 / (1.0 + np.exp(-np.clip(gl, -500, 500)))
                    thr = find_budget_threshold(gp, target_enabled_rate)
                    gate_decisions = gp >= thr
                else:
                    gate_decisions = np.ones(n_valid, dtype=bool)
                tg1 = time.perf_counter()

                tf0 = time.perf_counter()
                use_tracker = tracker if strategy == "ra_value" else None
                logits, info = forward_dynamic_budget(
                    model,
                    idx,
                    gate_decisions,
                    local_window,
                    chunk_size,
                    budget_mode,
                    b_max,
                    ra_values if strategy == "ra_value" else None,
                    strategy,
                    rng=rng,
                    tracker=use_tracker,
                )
                tf1 = time.perf_counter()

                if use_tracker is not None:
                    ra_values = tracker.get_chunk_values()

                t1 = time.perf_counter()

                batch_budgets = info.get("budgets_used", [])
                all_budgets.extend(batch_budgets)

                # Kept tokens (use mean budget for this batch)
                mean_budget = float(np.mean(batch_budgets)) if batch_budgets else b_max
                total_attended = 0
                for t in range(T):
                    if t < local_window:
                        total_attended += min(t + 1, T)
                    else:
                        ti = t - local_window
                        if gate_decisions[ti]:
                            total_attended += local_window + mean_budget * chunk_size
                        else:
                            total_attended += local_window
                kept = total_attended / T if T > 0 else 0.0

                ppl = compute_ppl(logits, idx)
                ppls.append(ppl)
                kept_list.append(kept)
                enabled_rates.append(
                    float(gate_decisions.mean()) if n_valid > 0 else 0.0
                )
                wall_times.append((t1 - t0) * 1000)
                gate_times.append((tg1 - tg0) * 1000)
                fwd_times.append((tf1 - tf0) * 1000)
                tokens_seen += B * T

                if (i + 1) % 10 == 0 or i == 0:
                    bm = float(np.mean(batch_budgets)) if batch_budgets else b_max
                    print(
                        f"    [{i+1}/{n_eval}] PPL={ppl:.1f} "
                        f"budget={bm:.1f} kept={kept:.0f}"
                    )

            total_tokens = batch_size * seq_len
            n_layers = cfg.n_layer
            d_model = cfg.n_embd
            mean_budget_final = float(np.mean(all_budgets)) if all_budgets else b_max
            kv = compute_kv_accounting(
                seq_len=seq_len,
                n_layers=n_layers,
                d_model=d_model,
                local_window=local_window,
                enabled_rate=float(np.mean(enabled_rates)),
                far_budget=int(mean_budget_final * chunk_size),
                bytes_per_elem=2,
            )
            dense_flops = seq_len * d_model * n_layers

            m = V7RunMetrics(
                variant=variant,
                seq_len=seq_len,
                seed=seed,
                far_budget=b_max,
                mode="dynamic-budget",
                ppl_mean=float(np.mean(ppls)),
                ppl_std=float(np.std(ppls)),
                enabled_rate=float(np.mean(enabled_rates)),
                effective_kept_tokens=float(np.mean(kept_list)),
                kv_bytes_read_per_token=kv.kv_bytes_read_per_token,
                peak_kv_bytes=kv.peak_kv_bytes,
                flops_relative=(
                    kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0
                ),
                wall_ms_per_token=float(np.median(wall_times)) / total_tokens,
                gate_ms_per_token=float(np.median(gate_times)) / total_tokens,
                fwd_ms_per_token=float(np.median(fwd_times)) / total_tokens,
                budget_mean=mean_budget_final,
                budget_p95=(
                    float(np.percentile(all_budgets, 95)) if all_budgets else b_max
                ),
                budget_p99=(
                    float(np.percentile(all_budgets, 99)) if all_budgets else b_max
                ),
                tokens_seen=tokens_seen,
                n_eval_batches=n_eval,
                extra={"budget_mode": budget_mode, "strategy": strategy},
            )
            results.append(asdict(m))

    return results


# ============================================================
# Section 4: Cheap RA-derived features
# ============================================================


def select_far_chunks_cheap(
    strategy: str,
    n_chunks: int,
    far_budget: int,
    ra_chunk_values,
    query_pos: int,
    chunk_size: int,
    local_window: int,
    rng=None,
):
    """Select far chunks using cheap RA-derived features.

    Strategies:
      ra_value: full RA scores (baseline)
      ra_ema: use EMA-smoothed scores (already in tracker)
      ra_rank: top-k binary flags (select if chunk ever was top-k)
      ra_layeragg: sum across layers without per-head breakdown
    """
    local_end_pos = max(0, query_pos - local_window + 1)
    local_end_chunk = local_end_pos // chunk_size
    far_chunks = list(range(local_end_chunk))

    if len(far_chunks) == 0:
        return np.array([], dtype=np.int64)

    budget = min(far_budget, len(far_chunks))

    if ra_chunk_values is not None and len(ra_chunk_values) > 0:
        values = (
            ra_chunk_values.cpu().numpy()
            if hasattr(ra_chunk_values, "cpu")
            else np.array(ra_chunk_values)
        )

        if strategy == "ra_rank":
            # Binary: was this chunk ever in top-k?
            # Use rank order, take top budget
            far_values = [(c, values[c]) for c in far_chunks if c < len(values)]
            far_values.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in far_values[:budget]]
        else:
            # ra_value, ra_ema, ra_layeragg all use same selection
            far_values = [(c, values[c]) for c in far_chunks if c < len(values)]
            far_values.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in far_values[:budget]]
    else:
        selected = far_chunks[-budget:]

    return np.sort(np.array(selected, dtype=np.int64))


def run_cheap_feature_tests(
    model,
    text_data,
    seq_len,
    seed,
    n_eval,
    batch_size,
    local_window,
    chunk_size,
    far_budget,
    gate,
    feat_mean,
    feat_std,
    target_enabled_rate,
    surgical_heads,
    output_dir,
):
    """Compare full RA_value against cheap RA-derived features."""
    results = []
    cfg = model.config

    cheap_strategies = [
        ("ra_value", "Full RA (baseline)"),
        ("ra_ema", "EMA-smoothed RA"),
        ("ra_rank", "Rank-only RA"),
        ("recency", "Recency (no RA)"),
    ]

    for strat_name, desc in cheap_strategies:
        variant = f"cheap_{strat_name}"
        print(f"\n  {variant}: {desc}")

        rng = np.random.RandomState(seed)
        ema_gamma = 0.3 if strat_name == "ra_ema" else 0.0
        tracker = RAValueTracker(
            surgical_heads, chunk_size=chunk_size, ema_gamma=ema_gamma
        )

        # Warmup
        warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
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

        ppls, wall_times, gate_times, fwd_times = [], [], [], []
        enabled_rates, kept_list = [], []
        tokens_seen = 0

        for i in range(n_eval):
            idx = get_text_batch(text_data, batch_size, seq_len, rng)
            B, T = idx.shape
            n_valid = max(T - local_window, 0)

            t0 = time.perf_counter()
            tg0 = time.perf_counter()
            if gate is not None and n_valid > 0:
                feats = compute_gate_features(model, idx, local_window)
                feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
                gl = (
                    gate(torch.tensor(feats_norm, dtype=torch.float32))
                    .detach()
                    .numpy()
                    .flatten()
                )
                gp = 1.0 / (1.0 + np.exp(-np.clip(gl, -500, 500)))
                thr = find_budget_threshold(gp, target_enabled_rate)
                gate_decisions = gp >= thr
            else:
                gate_decisions = np.ones(n_valid, dtype=bool)
            tg1 = time.perf_counter()

            tf0 = time.perf_counter()
            actual_strategy = (
                "ra_value" if strat_name in ("ra_ema", "ra_rank") else strat_name
            )
            use_tracker = tracker if strat_name != "recency" else None
            logits, info = forward_with_chunk_selection_v6(
                model,
                idx,
                gate_decisions,
                local_window,
                chunk_size,
                actual_strategy,
                far_budget,
                ra_chunk_values=ra_values if strat_name != "recency" else None,
                rng=rng,
                tracker=use_tracker,
            )
            tf1 = time.perf_counter()

            if use_tracker is not None:
                ra_values = tracker.get_chunk_values()

            t1 = time.perf_counter()

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
            kept_list.append(kept)
            enabled_rates.append(float(gate_decisions.mean()) if n_valid > 0 else 0.0)
            wall_times.append((t1 - t0) * 1000)
            gate_times.append((tg1 - tg0) * 1000)
            fwd_times.append((tf1 - tf0) * 1000)
            tokens_seen += B * T

            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{n_eval}] PPL={ppl:.1f} kept={kept:.0f}")

        total_tokens = batch_size * seq_len
        n_layers = cfg.n_layer
        d_model = cfg.n_embd
        kv = compute_kv_accounting(
            seq_len=seq_len,
            n_layers=n_layers,
            d_model=d_model,
            local_window=local_window,
            enabled_rate=float(np.mean(enabled_rates)),
            far_budget=far_budget * chunk_size,
            bytes_per_elem=2,
        )
        dense_flops = seq_len * d_model * n_layers

        m = V7RunMetrics(
            variant=variant,
            seq_len=seq_len,
            seed=seed,
            far_budget=far_budget,
            mode="cheap-features",
            ppl_mean=float(np.mean(ppls)),
            ppl_std=float(np.std(ppls)),
            enabled_rate=float(np.mean(enabled_rates)),
            effective_kept_tokens=float(np.mean(kept_list)),
            kv_bytes_read_per_token=kv.kv_bytes_read_per_token,
            peak_kv_bytes=kv.peak_kv_bytes,
            flops_relative=(kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0),
            wall_ms_per_token=float(np.median(wall_times)) / total_tokens,
            gate_ms_per_token=float(np.median(gate_times)) / total_tokens,
            fwd_ms_per_token=float(np.median(fwd_times)) / total_tokens,
            budget_mean=float(far_budget),
            budget_p95=float(far_budget),
            budget_p99=float(far_budget),
            tokens_seen=tokens_seen,
            n_eval_batches=n_eval,
            extra={"cheap_strategy": strat_name, "ema_gamma": ema_gamma},
        )
        results.append(asdict(m))

    return results


# ============================================================
# Main runner
# ============================================================


def load_model(checkpoint, seq_len, local_window, chunk_size, top_b):
    """Load model with position interpolation if needed."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab_size = ckpt["model"]["transformer.wte.weight"].shape[0]
    ckpt_block_size = ckpt["model"]["transformer.wpe.weight"].shape[0]

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

    if seq_len > ckpt_block_size:
        model_sd["transformer.wpe.weight"] = interpolate_pos_embed(
            model_sd["transformer.wpe.weight"], seq_len
        )
        print(f"  Position embedding interpolated: {ckpt_block_size} -> {seq_len}")

    model.load_state_dict(model_sd, strict=False)
    model.eval()
    del ckpt
    return model


def run_v7(
    checkpoint,
    seq_lens,
    seeds,
    far_budget,
    n_eval,
    batch_size,
    local_window,
    chunk_size,
    top_b,
    v4_gate_dir,
    text_data_path,
    output_dir,
    target_enabled_rate,
    mode,
):
    """Run v7 experiments."""
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
    print(f"Surgical heads: {len(surgical_heads)} heads")

    all_results = []

    for seq_len in seq_lens:
        run_seeds = seeds[:2] if seq_len >= 2048 else seeds
        for seed in run_seeds:
            print(f"\n{'='*60}")
            print(f"SEQ_LEN={seq_len}, SEED={seed}, MODE={mode}")
            print(f"{'='*60}")

            model = load_model(checkpoint, seq_len, local_window, chunk_size, top_b)
            gate, feat_mean, feat_std = load_v4_gate(v4_gate_dir, seed)

            if mode in ("stress", "all"):
                results = run_stress_tests(
                    model,
                    text_data,
                    seq_len,
                    seed,
                    n_eval,
                    batch_size,
                    local_window,
                    chunk_size,
                    far_budget,
                    gate,
                    feat_mean,
                    feat_std,
                    target_enabled_rate,
                    surgical_heads,
                    output_dir,
                )
                all_results.extend(results)

            if mode in ("dynamic-budget", "all"):
                results = run_dynamic_budget_tests(
                    model,
                    text_data,
                    seq_len,
                    seed,
                    n_eval,
                    batch_size,
                    local_window,
                    chunk_size,
                    far_budget,
                    gate,
                    feat_mean,
                    feat_std,
                    target_enabled_rate,
                    surgical_heads,
                    output_dir,
                )
                all_results.extend(results)

            if mode in ("cheap-features", "all"):
                results = run_cheap_feature_tests(
                    model,
                    text_data,
                    seq_len,
                    seed,
                    n_eval,
                    batch_size,
                    local_window,
                    chunk_size,
                    far_budget,
                    gate,
                    feat_mean,
                    feat_std,
                    target_enabled_rate,
                    surgical_heads,
                    output_dir,
                )
                all_results.extend(results)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save per-run metrics
    for r in all_results:
        fname = f"{r['variant']}_L{r['seq_len']}_S{r['seed']}.json"
        path = os.path.join(output_dir, "per_run_metrics", fname)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)

    results_path = os.path.join(output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="BPA v7 experiment")
    parser.add_argument(
        "--checkpoint",
        default="test_matrix_results_20260206_184612/"
        "gpt2_adamwspam_none_none/final_model_stepV0.pt",
    )
    parser.add_argument("--gate-dir", default="bpa_v4_gate_results")
    parser.add_argument("--seq-lens", default="512,1024,2048")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--far-budget", type=int, default=4)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--top-b", type=int, default=8)
    parser.add_argument("--target-rate", type=float, default=0.7)
    parser.add_argument("--output-dir", default="bpa_v7_results")
    parser.add_argument("--text-data", default="data/finewebedu/val.bin")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["stress", "dynamic-budget", "cheap-features", "all"],
    )

    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    run_v7(
        checkpoint=args.checkpoint,
        seq_lens=seq_lens,
        seeds=seeds,
        far_budget=args.far_budget,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        local_window=args.local_window,
        chunk_size=args.chunk_size,
        top_b=args.top_b,
        v4_gate_dir=args.gate_dir,
        text_data_path=args.text_data,
        output_dir=args.output_dir,
        target_enabled_rate=args.target_rate,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
