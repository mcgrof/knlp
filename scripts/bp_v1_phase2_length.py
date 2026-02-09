#!/usr/bin/env python
"""
BPA v1 Phase 2: Long Context Test

Test whether boundary_pressure remains predictive as context length increases.

Context lengths:
- L = 512 (short)
- L = 1024 (medium)
- L = 2048 (long, if memory permits)

Key questions:
- Does correlation direction remain stable across L?
- Does threshold need to be L-dependent?
- Does KL-selectivity remain strong?

Acceptance criteria (R2):
- Correlation direction stable across L (no sign flips)
- Either same threshold works, OR calibration is simple and monotonic
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA, ConditionalImpactTracker


def compute_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    corr, pval = scipy_stats.spearmanr(x, y)
    return float(corr) if not np.isnan(corr) else 0.0, float(pval)


def compute_roc_auc(signal: np.ndarray, target: np.ndarray, percentile: float = 75) -> float:
    """Compute ROC-AUC for predicting target > threshold."""
    threshold = np.percentile(target, percentile)
    labels = (target > threshold).astype(int)

    if len(np.unique(labels)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(labels, signal)
        return float(auc)
    except ValueError:
        return 0.5


def create_model(seq_len: int, model_size: str = "124M") -> GPT2_BPA:
    """Create BPA model with given sequence length."""
    configs = {
        "124M": dict(n_layer=12, n_head=12, n_embd=768),
        "355M": dict(n_layer=24, n_head=16, n_embd=1024),
    }

    params = configs[model_size]

    # Scale local_window with sequence length (keep ~10%)
    local_window = max(64, seq_len // 8)

    cfg = BPAConfig(
        block_size=seq_len,
        vocab_size=50304,
        n_layer=params["n_layer"],
        n_head=params["n_head"],
        n_embd=params["n_embd"],
        local_window=local_window,
        chunk_size=64,
        top_b=4,
    )

    model = GPT2_BPA(cfg)
    model.eval()
    return model


def run_length_test(
    seq_len: int,
    model_size: str = "124M",
    n_samples: int = 20,
    n_positions: int = 6,
    n_heads_per_sample: int = 4,
    seed: int = 1,
    threshold: float = 0.1,
) -> Dict:
    """
    Run boundary_pressure test for a given sequence length.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_model(seq_len, model_size)
    cfg = model.config
    n_layer = cfg.n_layer
    n_head = cfg.n_head

    print(f"\nContext Length: L={seq_len}")
    print(f"  local_window={cfg.local_window}, model={model_size}")

    tracker = ConditionalImpactTracker(
        n_layer=n_layer,
        n_head=n_head,
        seq_len=seq_len,
        heads_per_eval=n_heads_per_sample,
        positions_per_eval=n_positions,
    )

    boundary_pressures = []
    delta_kls = []
    enabled_kls = []
    disabled_kls = []

    # Use smaller batch for longer sequences to save memory
    batch_size = 2 if seq_len <= 1024 else 1

    print(f"Collecting {n_samples} samples...")
    for sample_idx in range(n_samples):
        if (sample_idx + 1) % 5 == 0:
            print(f"  Sample {sample_idx + 1}/{n_samples}")

        idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        heads = tracker.get_heads_to_measure()
        positions = tracker.sample_positions(local_window=cfg.local_window)

        try:
            signals = model.compute_conditional_signals(idx, positions=positions)
            kl_results = model.compute_conditional_impact_kl(idx, positions, heads)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at sample {sample_idx}, reducing batch")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            raise

        for (pos, layer_idx, head_idx), delta_kl in kl_results.items():
            pos_idx = positions.index(pos) if pos in positions else 0
            bp = signals["boundary_pressure"][:, pos_idx, layer_idx, head_idx].mean().item()

            boundary_pressures.append(bp)
            delta_kls.append(delta_kl)

            if bp > threshold:
                enabled_kls.append(delta_kl)
            else:
                disabled_kls.append(delta_kl)

            tracker.update(pos, layer_idx, head_idx, delta_kl)

    boundary_pressures = np.array(boundary_pressures)
    delta_kls = np.array(delta_kls)

    n_measurements = len(delta_kls)
    print(f"\nCollected {n_measurements} measurements")

    # Compute metrics
    corr, pval = compute_spearman(boundary_pressures, delta_kls)
    auc = compute_roc_auc(boundary_pressures, delta_kls, percentile=75)

    enabled_mean = np.mean(enabled_kls) if enabled_kls else 0
    disabled_mean = np.mean(disabled_kls) if disabled_kls else 1e-10
    kl_ratio = enabled_mean / (disabled_mean + 1e-10)
    enabled_pct = len(enabled_kls) / n_measurements if n_measurements > 0 else 0

    # Compute optimal threshold (for calibration analysis)
    # Find threshold that gives ~50% enabled rate
    if len(boundary_pressures) > 0:
        p50_threshold = float(np.percentile(boundary_pressures, 50))
    else:
        p50_threshold = threshold

    results = {
        "seq_len": seq_len,
        "model_size": model_size,
        "local_window": cfg.local_window,
        "seed": seed,
        "n_measurements": n_measurements,
        "correlation": {
            "spearman_r": corr,
            "pvalue": pval,
            "roc_auc_p75": auc,
        },
        "kl_selectivity": {
            "threshold": threshold,
            "enabled_pct": enabled_pct,
            "kl_ratio": float(kl_ratio),
        },
        "threshold_calibration": {
            "p50_threshold": p50_threshold,
            "bp_mean": float(np.mean(boundary_pressures)) if len(boundary_pressures) > 0 else 0,
            "bp_std": float(np.std(boundary_pressures)) if len(boundary_pressures) > 0 else 0,
        },
    }

    return results


def run_multi_length(
    lengths: List[int],
    model_size: str,
    seeds: List[int],
    n_samples: int = 15,
) -> Dict:
    """Run length test across multiple context lengths and seeds."""

    all_results = {}

    for L in lengths:
        print(f"\n{'#'*60}")
        print(f"# CONTEXT LENGTH: L={L}")
        print(f"{'#'*60}")

        length_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            result = run_length_test(
                seq_len=L,
                model_size=model_size,
                n_samples=n_samples,
                seed=seed,
            )
            length_results.append(result)

        # Aggregate across seeds
        corrs = [r["correlation"]["spearman_r"] for r in length_results]
        aucs = [r["correlation"]["roc_auc_p75"] for r in length_results]
        kl_ratios = [r["kl_selectivity"]["kl_ratio"] for r in length_results]
        thresholds = [r["threshold_calibration"]["p50_threshold"] for r in length_results]

        signs = [1 if c > 0 else -1 for c in corrs if c != 0]
        sign_flip = len(set(signs)) > 1 if signs else False

        all_results[str(L)] = {
            "seq_len": L,
            "correlation": {
                "mean_spearman_r": float(np.mean(corrs)),
                "std_spearman_r": float(np.std(corrs)),
                "mean_auc": float(np.mean(aucs)),
                "sign_flip": bool(sign_flip),
            },
            "kl_selectivity": {
                "mean_kl_ratio": float(np.mean(kl_ratios)),
            },
            "threshold_calibration": {
                "mean_p50_threshold": float(np.mean(thresholds)),
                "std_p50_threshold": float(np.std(thresholds)),
            },
            "seed_results": length_results,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="BPA v1 Phase 2: Long Context Test")
    parser.add_argument("--lengths", type=str, default="512,1024", help="Context lengths to test")
    parser.add_argument("--model", type=str, default="124M", choices=["124M", "355M"])
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds to test")
    parser.add_argument("--samples", type=int, default=10, help="Samples per seed")
    args = parser.parse_args()

    lengths = [int(L) for L in args.lengths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v1 Phase 2: Long Context Test")
    print("=" * 70)
    print("\nAcceptance criteria (R2):")
    print("  - Correlation direction stable across L (no sign flips)")
    print("  - Threshold calibration is simple and monotonic")
    print(f"\nTesting lengths: {lengths}")
    print(f"Model: {args.model}")
    print()

    results = run_multi_length(lengths, args.model, seeds, n_samples=args.samples)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Length':<10} {'Mean ρ':>10} {'Mean AUC':>10} {'KL Ratio':>10} {'P50 Thr':>10} {'Flip':>8}")
    print("-" * 70)

    for L_str, res in results.items():
        L = int(L_str)
        corr = res["correlation"]["mean_spearman_r"]
        auc = res["correlation"]["mean_auc"]
        kl_ratio = res["kl_selectivity"]["mean_kl_ratio"]
        thr = res["threshold_calibration"]["mean_p50_threshold"]
        flip = "YES" if res["correlation"]["sign_flip"] else "no"
        print(f"L={L:<7} {corr:>10.4f} {auc:>10.4f} {kl_ratio:>10.2f} {thr:>10.4f} {flip:>8}")

    print("-" * 70)

    # Check threshold monotonicity
    sorted_lengths = sorted(results.keys(), key=int)
    thresholds = [results[L]["threshold_calibration"]["mean_p50_threshold"] for L in sorted_lengths]

    is_monotonic = all(thresholds[i] <= thresholds[i+1] * 1.5 for i in range(len(thresholds)-1))
    any_sign_flip = any(results[L]["correlation"]["sign_flip"] for L in sorted_lengths)

    # Verdict
    if any_sign_flip:
        print("\nVERDICT: R2 FAILS - sign flip detected across lengths")
        print("boundary_pressure loses predictive power at longer L")
    elif is_monotonic:
        print("\nVERDICT: R2 PASSES - correlation stable, threshold roughly monotonic")
        print("Proceed to Phase 3 (cheap signal)")
    else:
        print("\nVERDICT: R2 MARGINAL - correlation stable but threshold calibration brittle")
        print("May need per-length threshold calibration")

    # Save results
    os.makedirs("bpa_v1_results", exist_ok=True)
    with open("bpa_v1_results/phase2_length.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to bpa_v1_results/phase2_length.json")


if __name__ == "__main__":
    main()
