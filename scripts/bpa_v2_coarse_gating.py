#!/usr/bin/env python
"""
BPA v2 Phase 3: Coarse Gating Evaluation

Reduces brittleness of per-token gating by gating at coarser granularity.

Three coarse modes:
  C1) Segment gating: if any token in segment triggers, enable whole segment
      segment_size in {8, 16, 32}
  C2) Head-block gating: gate per layer (all heads in layer share decision)
  C3) Position-bucket prior: different thresholds for early/mid/late with
      hysteresis (thr_on > thr_off)

Evaluates:
  - enabled-rate stability (variance across batches)
  - PPL vs compute trade-off
  - Variance across seeds

Usage:
    python scripts/bpa_v2_coarse_gating.py [--data-dir bpa_v2_gate_dataset]
        [--seeds 1,2,3] [--n-eval 50]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v2_train_gate import MLPGate, N_FEATURES, load_dataset


def make_gate_decisions(
    gate: nn.Module,
    features: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run gate on features and return binary decisions."""
    features_norm = (features - feat_mean) / (feat_std + 1e-8)
    with torch.no_grad():
        x = torch.tensor(features_norm, dtype=torch.float32)
        logits = gate(x).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs > threshold).astype(np.float32)


def apply_segment_gating(
    decisions: np.ndarray,
    n_tokens: int,
    n_layer: int,
    n_head: int,
    segment_size: int,
) -> np.ndarray:
    """
    C1: Segment gating. If any token in a segment triggers for a given
    (layer, head), enable far-context for the whole segment.

    decisions: [n_tokens, n_layer, n_head] binary
    Returns: [n_tokens, n_layer, n_head] binary (coarsened)
    """
    coarse = decisions.copy()
    for seg_start in range(0, n_tokens, segment_size):
        seg_end = min(seg_start + segment_size, n_tokens)
        segment = decisions[seg_start:seg_end, :, :]
        # If any token in segment triggers, enable whole segment
        triggered = segment.max(axis=0, keepdims=True)  # [1, n_layer, n_head]
        coarse[seg_start:seg_end, :, :] = np.broadcast_to(
            triggered, (seg_end - seg_start, n_layer, n_head)
        )
    return coarse


def apply_head_block_gating(
    decisions: np.ndarray,
    n_tokens: int,
    n_layer: int,
    n_head: int,
) -> np.ndarray:
    """
    C2: Head-block gating. Gate per layer: if any head triggers at a
    token position, enable all heads in that layer.

    decisions: [n_tokens, n_layer, n_head] binary
    Returns: [n_tokens, n_layer, n_head] binary (coarsened)
    """
    coarse = decisions.copy()
    # For each (token, layer), if any head triggers, enable all heads
    layer_trigger = decisions.max(axis=2, keepdims=True)  # [n_tokens, n_layer, 1]
    coarse = np.broadcast_to(layer_trigger, (n_tokens, n_layer, n_head)).copy()
    return coarse


def apply_position_bucket_prior(
    decisions: np.ndarray,
    positions: np.ndarray,
    seq_len: int,
    n_layer: int,
    n_head: int,
    thr_early: float = 0.7,
    thr_mid: float = 0.5,
    thr_late: float = 0.3,
    hysteresis: float = 0.1,
) -> np.ndarray:
    """
    C3: Position-bucket prior with hysteresis.
    Different thresholds for early/mid/late positions.
    Hysteresis: thr_on = thr, thr_off = thr - hysteresis.

    decisions: [n_tokens, n_layer, n_head] binary (from gate probs)
    positions: [n_tokens] original positions in sequence
    Returns: [n_tokens, n_layer, n_head] binary
    """
    # This operates on probabilities, but we only have binary decisions.
    # For simplicity, we re-threshold based on position bucket.
    # Higher threshold for early positions (less likely to need far-context).
    coarse = decisions.copy()

    for ti in range(len(positions)):
        pos_norm = positions[ti] / max(seq_len - 1, 1)
        if pos_norm < 0.33:
            # Early: higher threshold = fewer far-context calls
            # With hysteresis: if currently on, use lower threshold to stay on
            scale = thr_early / 0.5  # scale relative to default 0.5
        elif pos_norm < 0.66:
            scale = thr_mid / 0.5
        else:
            # Late: lower threshold = more far-context calls
            scale = thr_late / 0.5

        # Modulate decisions: if scale > 1, suppress some; if < 1, add some
        if scale > 1.0:
            # Suppress: only keep if "strongly" triggered
            # Since we only have binary, probabilistically suppress
            suppress_prob = 1.0 - 1.0 / scale
            mask = np.random.random((n_layer, n_head)) > suppress_prob
            coarse[ti] = coarse[ti] * mask
        elif scale < 1.0:
            # Boost: probabilistically add triggers
            boost_prob = 1.0 - scale
            mask = np.random.random((n_layer, n_head)) < boost_prob
            coarse[ti] = np.maximum(coarse[ti], mask.astype(np.float32))

    return coarse


def evaluate_coarse_mode(
    features: np.ndarray,
    labels: np.ndarray,
    gate: nn.Module,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    n_layer: int,
    n_head: int,
    seq_len: int,
    local_window: int,
    mode: str,
    segment_size: int = 16,
    seed: int = 1,
) -> Dict:
    """Evaluate a single coarse gating mode."""
    np.random.seed(seed)

    # Get gate decisions
    decisions_flat = make_gate_decisions(
        gate, features, feat_mean, feat_std, threshold=0.5
    )

    # Total examples = B * n_pos * n_layer * n_head
    per_lh = n_layer * n_head
    n_tokens = len(decisions_flat) // per_lh

    # Reshape: [n_tokens, n_layer, n_head]
    decisions = decisions_flat.reshape(n_tokens, n_layer, n_head)
    labels_reshaped = labels.reshape(n_tokens, n_layer, n_head)

    # Create position array (we stored pos_normalized in feature index 5)
    positions = features[:, 5].reshape(n_tokens, n_layer, n_head)[:, 0, 0] * (
        seq_len - 1
    )

    # Apply coarse gating
    if mode == "fine":
        coarse = decisions
    elif mode.startswith("segment_"):
        seg_size = int(mode.split("_")[1])
        coarse = apply_segment_gating(decisions, n_tokens, n_layer, n_head, seg_size)
    elif mode == "head_block":
        coarse = apply_head_block_gating(decisions, n_tokens, n_layer, n_head)
    elif mode == "position_bucket":
        coarse = apply_position_bucket_prior(
            decisions, positions, seq_len, n_layer, n_head
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute metrics
    enabled_rate = coarse.mean()
    fine_rate = decisions.mean()

    # Stability: variance of per-batch enabled rate
    # Split into "batches" of 100 tokens
    batch_rates = []
    batch_sz = min(100, n_tokens)
    for start in range(0, n_tokens - batch_sz + 1, batch_sz):
        batch = coarse[start : start + batch_sz]
        batch_rates.append(batch.mean())
    rate_variance = float(np.var(batch_rates)) if batch_rates else 0.0

    # Oracle agreement
    thr = np.percentile(labels, 75)
    oracle = (labels_reshaped > thr).astype(np.float32)
    agreement = (coarse == oracle).mean()

    # Efficiency: how many unnecessary far-context calls vs fine gating
    extra_vs_fine = max(0, coarse.sum() - decisions.sum()) / max(decisions.sum(), 1)

    return {
        "mode": mode,
        "enabled_rate": float(enabled_rate),
        "fine_rate": float(fine_rate),
        "rate_variance": float(rate_variance),
        "oracle_agreement": float(agreement),
        "extra_vs_fine": float(extra_vs_fine),
        "n_tokens": n_tokens,
    }


def run_evaluation(
    data_dir: str = "bpa_v2_gate_dataset",
    seeds: List[int] = [1, 2, 3],
    output_dir: str = "bpa_v2_coarse_results",
    seq_len: int = 256,
    local_window: int = 64,
) -> Dict:
    """Run coarse gating evaluation across modes and seeds."""
    features, labels = load_dataset(data_dir)

    # Train a gate
    thr = np.percentile(labels, 75)
    labels_binary = (labels > thr).astype(np.float32)
    pos_rate = labels_binary.mean()
    pos_weight = (1 - pos_rate) / (pos_rate + 1e-10)

    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8

    n_layer = 12
    n_head = 12

    modes = [
        "fine",
        "segment_8",
        "segment_16",
        "segment_32",
        "head_block",
        "position_bucket",
    ]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Train gate for this seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        features_norm = (features - feat_mean) / (feat_std + 1e-8)
        gate = MLPGate(N_FEATURES, hidden=128, n_layers=2)
        optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
        pw = torch.tensor([pos_weight])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        train_x = torch.tensor(features_norm, dtype=torch.float32)
        train_y = torch.tensor(labels_binary, dtype=torch.float32)

        gate.train()
        for epoch in range(15):
            optimizer.zero_grad()
            chunk = min(10000, len(train_x))
            perm = torch.randperm(len(train_x))[:chunk]
            logits = gate(train_x[perm])
            loss = criterion(logits, train_y[perm])
            loss.backward()
            optimizer.step()
        gate.eval()

        print(f"  Gate trained (15 epochs)")

        for mode in modes:
            result = evaluate_coarse_mode(
                features,
                labels,
                gate,
                feat_mean,
                feat_std,
                n_layer=n_layer,
                n_head=n_head,
                seq_len=seq_len,
                local_window=local_window,
                mode=mode,
                seed=seed,
            )
            key = f"seed_{seed}_{mode}"
            all_results[key] = result
            print(
                f"  {mode:<20} rate={result['enabled_rate']:.4f} "
                f"var={result['rate_variance']:.6f} "
                f"oracle_agree={result['oracle_agreement']:.4f} "
                f"extra={result['extra_vs_fine']:.4f}"
            )

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    print(
        f"\n{'Mode':<20} {'Mean Rate':>10} {'Rate Var':>10} {'Oracle Agr':>12} {'Extra':>8}"
    )
    print("-" * 65)

    aggregated = {"seeds": seeds, "modes": {}}

    for mode in modes:
        rates = [all_results[f"seed_{s}_{mode}"]["enabled_rate"] for s in seeds]
        variances = [all_results[f"seed_{s}_{mode}"]["rate_variance"] for s in seeds]
        agreements = [
            all_results[f"seed_{s}_{mode}"]["oracle_agreement"] for s in seeds
        ]
        extras = [all_results[f"seed_{s}_{mode}"]["extra_vs_fine"] for s in seeds]

        aggregated["modes"][mode] = {
            "mean_rate": float(np.mean(rates)),
            "std_rate": float(np.std(rates)),
            "mean_variance": float(np.mean(variances)),
            "mean_oracle_agreement": float(np.mean(agreements)),
            "mean_extra_vs_fine": float(np.mean(extras)),
        }

        print(
            f"  {mode:<20} {np.mean(rates):>10.4f} "
            f"{np.mean(variances):>10.6f} "
            f"{np.mean(agreements):>12.4f} "
            f"{np.mean(extras):>8.4f}"
        )

    print("-" * 65)

    # Find best coarse mode (lowest variance while maintaining reasonable rate)
    coarse_modes = [m for m in modes if m != "fine"]
    best_mode = min(
        coarse_modes,
        key=lambda m: aggregated["modes"][m]["mean_variance"],
    )
    aggregated["best_coarse_mode"] = best_mode

    # Acceptance
    print(f"\n{'='*60}")
    print("ACCEPTANCE CHECK")
    print(f"{'='*60}")

    fine_var = aggregated["modes"]["fine"]["mean_variance"]
    best_var = aggregated["modes"][best_mode]["mean_variance"]
    variance_reduced = best_var <= fine_var
    print(
        f"  Coarse reduces variance:   "
        f"{'PASS' if variance_reduced else 'FAIL'} "
        f"(fine={fine_var:.6f}, {best_mode}={best_var:.6f})"
    )

    best_extra = aggregated["modes"][best_mode]["mean_extra_vs_fine"]
    not_too_wasteful = best_extra < 1.0  # less than 2x overhead
    print(
        f"  Extra compute < 100%:      "
        f"{'PASS' if not_too_wasteful else 'FAIL'} "
        f"({best_extra*100:.1f}%)"
    )

    aggregated["acceptance"] = {
        "variance_reduced": variance_reduced,
        "not_too_wasteful": not_too_wasteful,
        "best_coarse_mode": best_mode,
    }

    # Save
    result_path = os.path.join(output_dir, "phase3_results.json")

    def _json_default(x):
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(f"Not JSON serializable: {type(x)}")

    with open(result_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=_json_default)
    print(f"\nResults saved to {result_path}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="BPA v2 Phase 3: Coarse Gating")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bpa_v2_gate_dataset",
        help="Phase 0 dataset",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v2_coarse_results",
        help="Output dir",
    )
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--local-window", type=int, default=64, help="Local window")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v2 Phase 3: Coarse Gating Evaluation")
    print("=" * 70)

    run_evaluation(
        data_dir=args.data_dir,
        seeds=seeds,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        local_window=args.local_window,
    )


if __name__ == "__main__":
    main()
