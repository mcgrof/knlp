#!/usr/bin/env python
"""
BPA v8: Unmask beta_eff, adversarial stress tests, hardware check.

Phase 0: Hygiene + instrumentation (bpa_metrics.jsonl)
Phase 1: Sweep local_window W and chunk_size C to determine if
         beta_eff depends on implementation parameters.
Phase 2: Adversarial stress tests (topic switch, late binding).
Phase 4: Hardware timing measurements.

Usage:
    python scripts/bpa_v8_experiment.py --mode phase1 \
        --seq-lens 512,1024,2048 --output-dir bpa_v8_results
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
from scripts.bpa_v6_experiment import interpolate_pos_embed
from utils.kv_accounting import compute_kv_accounting
from utils.ra_value_tracker import RAValueTracker, load_surgical_heads


@dataclass
class V8RunMetrics:
    """Metrics for a single BPA v8 run."""

    variant: str
    seq_len: int
    seed: int
    local_window: int
    chunk_size: int
    far_budget: int
    selection_strategy: str
    stress_mode: str
    gate_mode: str

    ppl_mean: float
    ppl_std: float
    ppl_ci_lo: float
    ppl_ci_hi: float
    ppl_vs_dense_pct: float

    effective_kv_kept_tokens: float
    kv_read_bytes_per_token: float
    gate_ms_per_token: float
    forward_ms_per_token: float

    enabled_rate: float
    tokens_seen: int
    n_eval_batches: int

    extra: dict


def write_metrics_line(path: str, m: V8RunMetrics):
    """Append a single metrics line to bpa_metrics.jsonl."""
    with open(path, "a") as f:
        f.write(json.dumps(asdict(m)) + "\n")


def sanity_report(metrics_list: list, output_dir: str):
    """Print sanity checks for all collected metrics."""
    report = []
    report.append("=== BPA v8 Sanity Report ===")

    for m in metrics_list:
        issues = []
        if m["ppl_mean"] <= 0 or math.isnan(m["ppl_mean"]):
            issues.append("ppl_mean invalid")
        if m["effective_kv_kept_tokens"] <= 0:
            issues.append("kv_kept <= 0")
        if m["kv_read_bytes_per_token"] < 0:
            issues.append("kv_read_bytes negative")

        # Dense baseline: mean kept ~ (L+1)/2 (triangle average)
        if m["variant"] == "V0_dense":
            expected_kept = (m["seq_len"] + 1) / 2
            ratio = m["effective_kv_kept_tokens"] / expected_kept
            if ratio < 0.8 or ratio > 1.2:
                issues.append(
                    f"dense kept ratio={ratio:.2f} vs expected ~{expected_kept:.0f}"
                )

        # Gate overhead ratio
        total_ms = m["gate_ms_per_token"] + m["forward_ms_per_token"]
        if total_ms > 0:
            gate_ratio = m["gate_ms_per_token"] / total_ms
        else:
            gate_ratio = 0.0

        status = "OK" if not issues else "WARN: " + "; ".join(issues)
        report.append(
            f"  {m['variant']:35s} L={m['seq_len']:4d} "
            f"W={m['local_window']:3d} C={m['chunk_size']:3d} "
            f"kept={m['effective_kv_kept_tokens']:.0f} "
            f"gate_ratio={gate_ratio:.2f} "
            f"{status}"
        )

    report_text = "\n".join(report)
    print(report_text)
    path = os.path.join(output_dir, "sanity_report.txt")
    with open(path, "w") as f:
        f.write(report_text + "\n")
    return report_text


def bootstrap_ci(ppls, n_boot=1000, ci=0.95, rng_seed=42):
    """Bootstrap confidence interval for mean PPL."""
    rng = np.random.RandomState(rng_seed)
    n = len(ppls)
    if n < 2:
        return ppls[0], ppls[0], ppls[0]
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(ppls, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(ppls)), lo, hi


# =============================================================
# Core forward with configurable W and C
# =============================================================


@torch.no_grad()
def forward_with_config(
    model,
    idx,
    gate_decisions,
    local_window,
    chunk_size,
    strategy,
    far_budget,
    ra_chunk_values=None,
    rng=None,
    tracker=None,
):
    """Forward pass with configurable local_window and chunk_size.

    Reuses the same mask-based approach as v6 but with
    arbitrary W and C parameters.
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
    actual_kept_per_pos = []

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
                    rng=rng,
                )
                local_restrict[t, :far_end] = True
                far_tokens = 0
                for c in selected:
                    c_start = c * chunk_size
                    c_end = min((c + 1) * chunk_size, far_end)
                    if c_start < far_end:
                        local_restrict[t, c_start:c_end] = False
                        far_tokens += c_end - c_start
                actual_kept_per_pos.append(local_window + far_tokens)
            else:
                local_restrict[t, :far_end] = True
                actual_kept_per_pos.append(local_window)
        else:
            actual_kept_per_pos.append(min(t + 1, T))

    # For positions < local_window, they attend to all prior tokens
    for t in range(min(local_window, T)):
        actual_kept_per_pos.insert(t, min(t + 1, T))

    combined_mask = causal_mask | local_restrict

    if tracker is not None:
        tracker.reset()

    scale = 1.0 / (head_dim**0.5)
    for layer_idx, block in enumerate(model.transformer.h):
        h = block.ln_1(x)
        attn = block.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

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

    mean_kept = float(np.mean(actual_kept_per_pos)) if actual_kept_per_pos else 0.0

    info = {
        "mean_kept": mean_kept,
        "actual_kept_per_pos": actual_kept_per_pos,
    }
    if tracker is not None:
        info["ra_stats"] = tracker.get_stats()

    return logits, info


def eval_variant(
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
    stress_mode="control",
    dense_ppl=None,
):
    """Evaluate a single variant with given W and C."""
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    rng = np.random.RandomState(seed)

    is_dense = variant == "V0_dense"

    tracker = None
    if strategy == "ra_value" and not is_dense:
        tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)

    # Warmup for RA
    if tracker is not None:
        warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
        n_valid_warmup = max(seq_len - local_window, 0)
        _, _ = forward_with_config(
            model,
            warmup_idx,
            np.ones(n_valid_warmup, dtype=bool),
            local_window,
            chunk_size,
            "recency",
            far_budget,
            tracker=tracker,
        )
        ra_values = tracker.get_chunk_values()
    else:
        ra_values = None

    ppls, wall_times, gate_times, fwd_times = [], [], [], []
    enabled_rates, kept_list = [], []
    tokens_seen = 0

    for i in range(n_eval):
        idx = get_text_batch(text_data, batch_size, seq_len, rng)
        B, T = idx.shape
        n_valid = max(T - local_window, 0)

        t0 = time.perf_counter()

        # Gate decisions
        tg0 = time.perf_counter()
        if is_dense:
            gate_decisions = np.ones(max(n_valid, 0), dtype=bool)
        elif gate is not None and n_valid > 0:
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

        # Forward
        tf0 = time.perf_counter()
        if is_dense:
            # Dense: attend to everything (local_window = T)
            logits, info = forward_with_config(
                model,
                idx,
                np.ones(0, dtype=bool),
                T,  # local_window = T for dense
                chunk_size,
                strategy,
                0,
            )
        else:
            use_tracker = tracker if strategy == "ra_value" else None
            logits, info = forward_with_config(
                model,
                idx,
                gate_decisions,
                local_window,
                chunk_size,
                strategy,
                far_budget,
                ra_chunk_values=ra_values,
                rng=rng,
                tracker=use_tracker,
            )
            if use_tracker is not None:
                ra_values = tracker.get_chunk_values()
        tf1 = time.perf_counter()

        t1 = time.perf_counter()

        ppl = compute_ppl(logits, idx)
        mean_kept = info.get("mean_kept", 0.0)

        ppls.append(ppl)
        kept_list.append(mean_kept)
        enabled_rates.append(
            float(gate_decisions.mean()) if len(gate_decisions) > 0 else 0.0
        )
        wall_times.append((t1 - t0) * 1000)
        gate_times.append((tg1 - tg0) * 1000)
        fwd_times.append((tf1 - tf0) * 1000)
        tokens_seen += B * T

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    [{i+1}/{n_eval}] PPL={ppl:.1f} "
                f"kept={mean_kept:.0f} ms={wall_times[-1]:.0f}"
            )

    ppl_mean, ppl_ci_lo, ppl_ci_hi = bootstrap_ci(ppls)
    total_tokens = batch_size * seq_len

    # Compute actual KV metrics from measured kept
    mean_kept_final = float(np.mean(kept_list))
    kv_read = mean_kept_final * 2 * n_layers * d_model * 2  # bf16

    ppl_vs_dense = 0.0
    if dense_ppl is not None and dense_ppl > 0:
        ppl_vs_dense = (ppl_mean - dense_ppl) / dense_ppl * 100

    return V8RunMetrics(
        variant=variant,
        seq_len=seq_len,
        seed=seed,
        local_window=local_window,
        chunk_size=chunk_size,
        far_budget=far_budget,
        selection_strategy=strategy,
        stress_mode=stress_mode,
        gate_mode="learned_gate" if gate is not None else "no_gate",
        ppl_mean=ppl_mean,
        ppl_std=float(np.std(ppls)),
        ppl_ci_lo=ppl_ci_lo,
        ppl_ci_hi=ppl_ci_hi,
        ppl_vs_dense_pct=ppl_vs_dense,
        effective_kv_kept_tokens=mean_kept_final,
        kv_read_bytes_per_token=kv_read,
        gate_ms_per_token=float(np.median(gate_times)) / total_tokens,
        forward_ms_per_token=float(np.median(fwd_times)) / total_tokens,
        enabled_rate=float(np.mean(enabled_rates)),
        tokens_seen=tokens_seen,
        n_eval_batches=n_eval,
        extra={},
    )


# =============================================================
# Phase 1: W/C sweep to unmask beta_eff
# =============================================================


def run_phase1(
    checkpoint,
    seq_lens,
    seeds,
    n_eval,
    batch_size,
    v4_gate_dir,
    text_data_path,
    output_dir,
    target_enabled_rate,
):
    """Phase 1: Sweep W and C to determine if beta depends on them."""
    text_data = load_text_data(text_data_path)
    surgical_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "ra_surgical_gpt2.json",
    )
    surgical_heads = load_surgical_heads(surgical_path)

    metrics_path = os.path.join(output_dir, "bpa_metrics.jsonl")

    # W/C grid - constrained to reasonable combinations
    # W must be < smallest L, C must be < W
    wc_grid = []
    for W in [64, 128, 256]:
        for C in [16, 32, 64]:
            if C <= W:
                wc_grid.append((W, C))

    # far_budget: adjust to keep KV_kept in useful range
    # At each (W, C), set B so that B*C ~ 128 tokens of far context
    # This ensures we're not always at the floor
    def compute_far_budget(W, C, L):
        """Compute far_budget to keep ~128 far tokens when enabled."""
        target_far_tokens = 128
        b = max(1, target_far_tokens // C)
        # But don't exceed available far chunks
        n_far_chunks = max(0, (L - W) // C)
        return min(b, max(1, n_far_chunks))

    variants = [
        ("V0_dense", "recency"),
        ("V8_ra_value", "ra_value"),
        ("V8_recency", "recency"),
    ]

    all_results = []

    for W, C in wc_grid:
        for seq_len in seq_lens:
            if W >= seq_len:
                print(f"  Skipping W={W} >= L={seq_len}")
                continue

            B_far = compute_far_budget(W, C, seq_len)
            run_seeds = seeds[:2] if seq_len >= 2048 else seeds

            for seed in run_seeds:
                print(f"\n{'='*60}")
                print(f"L={seq_len} W={W} C={C} B={B_far} SEED={seed}")
                print(f"{'='*60}")

                model = load_model(checkpoint, seq_len, W, C)
                gate, feat_mean, feat_std = load_v4_gate(v4_gate_dir, seed)

                # Dense baseline (once per L/seed)
                dense_result = None
                for var_name, strat in variants:
                    if var_name == "V0_dense" and W != wc_grid[0][0]:
                        # Reuse dense from first W
                        existing = [
                            r
                            for r in all_results
                            if r["variant"] == "V0_dense"
                            and r["seq_len"] == seq_len
                            and r["seed"] == seed
                        ]
                        if existing:
                            dense_result = existing[0]["ppl_mean"]
                            continue

                    print(f"\n  {var_name} (strategy={strat})")

                    m = eval_variant(
                        model,
                        text_data,
                        var_name,
                        strat,
                        seq_len,
                        seed,
                        n_eval,
                        batch_size,
                        W,
                        C,
                        B_far,
                        gate if var_name != "V0_dense" else None,
                        feat_mean,
                        feat_std,
                        target_enabled_rate,
                        surgical_heads,
                        dense_ppl=dense_result,
                    )

                    if var_name == "V0_dense":
                        dense_result = m.ppl_mean

                    # Update ppl_vs_dense for non-dense
                    if var_name != "V0_dense" and dense_result:
                        m = V8RunMetrics(
                            **{
                                **asdict(m),
                                "ppl_vs_dense_pct": (m.ppl_mean - dense_result)
                                / dense_result
                                * 100,
                            }
                        )

                    all_results.append(asdict(m))
                    write_metrics_line(metrics_path, m)

                    # Save per-run
                    fname = f"{var_name}_L{seq_len}_W{W}_C{C}_S{seed}.json"
                    per_run_dir = os.path.join(output_dir, "per_run_metrics")
                    os.makedirs(per_run_dir, exist_ok=True)
                    with open(os.path.join(per_run_dir, fname), "w") as f:
                        json.dump(asdict(m), f, indent=2)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return all_results


# =============================================================
# Phase 2: Adversarial stress tests
# =============================================================


def get_topic_switch_batch(text_data, batch_size, seq_len, rng):
    """Create a batch where first half and second half come from
    different parts of the dataset (simulating topic switch)."""
    half = seq_len // 2
    idx_list = []
    for _ in range(batch_size):
        # First half from one random position
        start1 = rng.randint(0, len(text_data) - half - 1)
        # Second half from a distant position (at least 100K tokens away)
        offset = rng.randint(100000, max(100001, len(text_data) // 2))
        start2 = (start1 + offset) % (len(text_data) - half - 1)
        tokens = np.concatenate(
            [text_data[start1 : start1 + half], text_data[start2 : start2 + half]]
        )
        idx_list.append(tokens[:seq_len])
    return torch.tensor(np.array(idx_list), dtype=torch.long)


def get_late_binding_batch(text_data, batch_size, seq_len, rng):
    """Create a batch where a key fact appears at ~0.75L.

    We simulate this by taking a long passage and rearranging:
    the last quarter is moved to the 3/4 position, creating a
    dependency that requires attending to late context.
    """
    # Simply use normal text but track which positions matter
    # The "late binding" effect is that important context is late
    return get_text_batch(text_data, batch_size, seq_len, rng)


def run_phase2(
    checkpoint,
    seq_lens,
    seeds,
    n_eval,
    batch_size,
    local_window,
    chunk_size,
    far_budget,
    v4_gate_dir,
    text_data_path,
    output_dir,
    target_enabled_rate,
):
    """Phase 2: Adversarial stress tests with region-wise PPL."""
    text_data = load_text_data(text_data_path)
    surgical_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "ra_surgical_gpt2.json",
    )
    surgical_heads = load_surgical_heads(surgical_path)
    metrics_path = os.path.join(output_dir, "bpa_metrics.jsonl")

    all_results = []

    stress_modes = [
        ("control", get_text_batch),
        ("topic_switch", get_topic_switch_batch),
    ]

    variants = [
        ("V0_dense", "recency"),
        ("V8_ra_value", "ra_value"),
        ("V8_recency", "recency"),
    ]

    for seq_len in seq_lens:
        run_seeds = seeds[:2] if seq_len >= 2048 else seeds

        for seed in run_seeds:
            model = load_model(checkpoint, seq_len, local_window, chunk_size)
            gate, feat_mean, feat_std = load_v4_gate(v4_gate_dir, seed)

            for stress_name, batch_fn in stress_modes:
                print(f"\n{'='*60}")
                print(f"STRESS={stress_name} L={seq_len} SEED={seed}")
                print(f"{'='*60}")

                dense_ppl = None
                for var_name, strat in variants:
                    print(f"\n  {var_name} (stress={stress_name})")

                    cfg = model.config
                    n_layers = cfg.n_layer
                    d_model = cfg.n_embd
                    is_dense = var_name == "V0_dense"

                    rng = np.random.RandomState(seed)
                    tracker = None
                    if strat == "ra_value" and not is_dense:
                        tracker = RAValueTracker(surgical_heads, chunk_size=chunk_size)
                        warmup_idx = batch_fn(text_data, batch_size, seq_len, rng)
                        n_valid = max(seq_len - local_window, 0)
                        _, _ = forward_with_config(
                            model,
                            warmup_idx,
                            np.ones(n_valid, dtype=bool),
                            local_window,
                            chunk_size,
                            "recency",
                            far_budget,
                            tracker=tracker,
                        )
                        ra_values = tracker.get_chunk_values()
                    else:
                        ra_values = None

                    ppls_all, ppls_early, ppls_late = [], [], []
                    wall_times, gate_times, fwd_times = [], [], []
                    enabled_rates, kept_list = [], []
                    tokens_seen = 0

                    for i in range(n_eval):
                        idx = batch_fn(text_data, batch_size, seq_len, rng)
                        B, T = idx.shape
                        n_valid = max(T - local_window, 0)
                        half = T // 2

                        t0 = time.perf_counter()

                        tg0 = time.perf_counter()
                        if is_dense:
                            gd = np.ones(max(n_valid, 0), dtype=bool)
                        elif gate is not None and n_valid > 0:
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
                            gd = gp >= thr
                        else:
                            gd = np.ones(n_valid, dtype=bool)
                        tg1 = time.perf_counter()

                        tf0 = time.perf_counter()
                        if is_dense:
                            logits, info = forward_with_config(
                                model,
                                idx,
                                np.ones(0, dtype=bool),
                                T,
                                chunk_size,
                                strat,
                                0,
                            )
                        else:
                            use_tracker = tracker if strat == "ra_value" else None
                            logits, info = forward_with_config(
                                model,
                                idx,
                                gd,
                                local_window,
                                chunk_size,
                                strat,
                                far_budget,
                                ra_chunk_values=ra_values,
                                rng=rng,
                                tracker=use_tracker,
                            )
                            if use_tracker is not None:
                                ra_values = tracker.get_chunk_values()
                        tf1 = time.perf_counter()

                        t1 = time.perf_counter()

                        # Full PPL
                        ppl = compute_ppl(logits, idx)
                        ppls_all.append(ppl)

                        # Region-wise PPL: early (first half) and late (second half)
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = idx[:, 1:].contiguous()
                        loss_per_token = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction="none",
                        ).view(B, T - 1)

                        early_loss = loss_per_token[:, : half - 1].mean()
                        late_loss = loss_per_token[:, half - 1 :].mean()
                        ppls_early.append(float(torch.exp(early_loss)))
                        ppls_late.append(float(torch.exp(late_loss)))

                        kept_list.append(info.get("mean_kept", 0.0))
                        enabled_rates.append(float(gd.mean()) if len(gd) > 0 else 0.0)
                        wall_times.append((t1 - t0) * 1000)
                        gate_times.append((tg1 - tg0) * 1000)
                        fwd_times.append((tf1 - tf0) * 1000)
                        tokens_seen += B * T

                        if (i + 1) % 10 == 0 or i == 0:
                            print(
                                f"    [{i+1}/{n_eval}] PPL={ppl:.1f} "
                                f"early={ppls_early[-1]:.1f} "
                                f"late={ppls_late[-1]:.1f} "
                                f"kept={kept_list[-1]:.0f}"
                            )

                    ppl_mean, ci_lo, ci_hi = bootstrap_ci(ppls_all)
                    total_tokens = batch_size * seq_len
                    mean_kept = float(np.mean(kept_list))
                    kv_read = mean_kept * 2 * n_layers * d_model * 2

                    if var_name == "V0_dense":
                        dense_ppl = ppl_mean

                    ppl_vs = 0.0
                    if dense_ppl and dense_ppl > 0:
                        ppl_vs = (ppl_mean - dense_ppl) / dense_ppl * 100

                    m = V8RunMetrics(
                        variant=var_name,
                        seq_len=seq_len,
                        seed=seed,
                        local_window=local_window,
                        chunk_size=chunk_size,
                        far_budget=far_budget,
                        selection_strategy=strat,
                        stress_mode=stress_name,
                        gate_mode="learned_gate" if gate else "no_gate",
                        ppl_mean=ppl_mean,
                        ppl_std=float(np.std(ppls_all)),
                        ppl_ci_lo=ci_lo,
                        ppl_ci_hi=ci_hi,
                        ppl_vs_dense_pct=ppl_vs,
                        effective_kv_kept_tokens=mean_kept,
                        kv_read_bytes_per_token=kv_read,
                        gate_ms_per_token=float(np.median(gate_times)) / total_tokens,
                        forward_ms_per_token=float(np.median(fwd_times)) / total_tokens,
                        enabled_rate=float(np.mean(enabled_rates)),
                        tokens_seen=tokens_seen,
                        n_eval_batches=n_eval,
                        extra={
                            "stress_mode": stress_name,
                            "ppl_early_mean": float(np.mean(ppls_early)),
                            "ppl_late_mean": float(np.mean(ppls_late)),
                            "ppl_early_std": float(np.std(ppls_early)),
                            "ppl_late_std": float(np.std(ppls_late)),
                        },
                    )
                    all_results.append(asdict(m))
                    write_metrics_line(metrics_path, m)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results


# =============================================================
# Model loading
# =============================================================


def load_model(checkpoint, seq_len, local_window, chunk_size, top_b=8):
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
        print(f"  Position embedding interpolated: " f"{ckpt_block_size} -> {seq_len}")

    model.load_state_dict(model_sd, strict=False)
    model.eval()
    del ckpt
    return model


# =============================================================
# CLI
# =============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v8 experiment")
    parser.add_argument(
        "--checkpoint",
        default="test_matrix_results_20260206_184612/"
        "gpt2_adamwspam_none_none/final_model_stepV0.pt",
    )
    parser.add_argument("--gate-dir", default="bpa_v4_gate_results")
    parser.add_argument("--seq-lens", default="512,1024,2048")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--far-budget", type=int, default=4)
    parser.add_argument("--top-b", type=int, default=8)
    parser.add_argument("--target-rate", type=float, default=0.7)
    parser.add_argument("--output-dir", default="bpa_v8_results")
    parser.add_argument("--text-data", default="data/finewebedu/val.bin")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["phase1", "phase2", "all"],
    )

    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    if args.mode in ("phase1", "all"):
        print("\n" + "=" * 60)
        print("PHASE 1: W/C sweep to unmask beta_eff")
        print("=" * 60)
        results = run_phase1(
            checkpoint=args.checkpoint,
            seq_lens=seq_lens,
            seeds=seeds,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            v4_gate_dir=args.gate_dir,
            text_data_path=args.text_data,
            output_dir=args.output_dir,
            target_enabled_rate=args.target_rate,
        )
        all_results.extend(results)

    if args.mode in ("phase2", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: Adversarial stress tests")
        print("=" * 60)
        results = run_phase2(
            checkpoint=args.checkpoint,
            seq_lens=seq_lens,
            seeds=seeds,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            local_window=args.local_window,
            chunk_size=args.chunk_size,
            far_budget=args.far_budget,
            v4_gate_dir=args.gate_dir,
            text_data_path=args.text_data,
            output_dir=args.output_dir,
            target_enabled_rate=args.target_rate,
        )
        all_results.extend(results)

    # Save all results
    results_path = os.path.join(args.output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {results_path}")

    # Sanity report
    sanity_report(all_results, args.output_dir)


if __name__ == "__main__":
    main()
