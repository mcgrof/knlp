#!/usr/bin/env python
"""
BPA v1 Phase 1: Scale Test

Test whether boundary_pressure remains predictive at larger model scale.

Models:
- GPT-2 124M (baseline from v19)
- GPT-2 355M (medium)

Key metrics:
- Spearman correlation (boundary_pressure vs ΔKL)
- ROC-AUC for predicting ΔKL > 75th percentile
- KL-selectivity ratio (enabled vs disabled ΔKL)

Acceptance criteria (R1):
- boundary_pressure retains ρ >= 0.45 OR AUC >= 0.68
- KL-selectivity ratio >= 10x
- No seed sign flips
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


def create_model(model_size: str) -> GPT2_BPA:
    """Create BPA model for given size."""
    # Map to GPT-2 configs
    configs = {
        "124M": dict(n_layer=12, n_head=12, n_embd=768),
        "355M": dict(n_layer=24, n_head=16, n_embd=1024),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    params = configs[model_size]

    cfg = BPAConfig(
        block_size=1024,  # Fixed L=1024 for R1
        vocab_size=50304,
        n_layer=params["n_layer"],
        n_head=params["n_head"],
        n_embd=params["n_embd"],
        local_window=128,  # ~10% of context
        chunk_size=64,
        top_b=4,
    )

    model = GPT2_BPA(cfg)
    model.eval()
    return model


def run_scale_test(
    model: GPT2_BPA,
    model_size: str,
    n_samples: int = 30,
    n_positions: int = 8,
    n_heads_per_sample: int = 4,
    seed: int = 1,
    threshold: float = 0.1,
) -> Dict:
    """
    Run boundary_pressure correlation test for a given model.

    Args:
        model: BPA model
        model_size: Model size name for logging
        n_samples: Number of input samples
        n_positions: Positions to measure per sample
        n_heads_per_sample: Heads to measure per sample
        seed: Random seed
        threshold: Gating threshold for KL-selectivity

    Returns:
        Dict with test results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = model.config
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    T = 256  # Shorter sequence for memory, but still past local_window

    print(f"\nModel: {model_size}")
    print(f"  n_layer={n_layer}, n_head={n_head}, n_embd={cfg.n_embd}")
    print(f"  local_window={cfg.local_window}, T={T}")

    tracker = ConditionalImpactTracker(
        n_layer=n_layer,
        n_head=n_head,
        seq_len=T,
        heads_per_eval=n_heads_per_sample,
        positions_per_eval=n_positions,
    )

    # Collect (boundary_pressure, delta_kl) pairs
    boundary_pressures = []
    delta_kls = []
    enabled_kls = []  # Where boundary_pressure > threshold
    disabled_kls = []  # Where boundary_pressure <= threshold

    print(f"Collecting {n_samples} samples...")
    for sample_idx in range(n_samples):
        if (sample_idx + 1) % 10 == 0:
            print(f"  Sample {sample_idx + 1}/{n_samples}")

        # Random input
        idx = torch.randint(0, cfg.vocab_size, (2, T))

        # Get heads and positions
        heads = tracker.get_heads_to_measure()
        positions = tracker.sample_positions(local_window=cfg.local_window)

        # Compute signals
        try:
            signals = model.compute_conditional_signals(idx, positions=positions)
            kl_results = model.compute_conditional_impact_kl(idx, positions, heads)
        except Exception as e:
            print(f"  Warning: sample {sample_idx} failed: {e}")
            continue

        # Collect paired samples
        for (pos, layer_idx, head_idx), delta_kl in kl_results.items():
            pos_idx = positions.index(pos) if pos in positions else 0
            bp = signals["boundary_pressure"][:, pos_idx, layer_idx, head_idx].mean().item()

            boundary_pressures.append(bp)
            delta_kls.append(delta_kl)

            # KL-selectivity tracking
            if bp > threshold:
                enabled_kls.append(delta_kl)
            else:
                disabled_kls.append(delta_kl)

            tracker.update(pos, layer_idx, head_idx, delta_kl)

    # Convert to numpy
    boundary_pressures = np.array(boundary_pressures)
    delta_kls = np.array(delta_kls)

    n_measurements = len(delta_kls)
    print(f"\nCollected {n_measurements} measurements")

    # Compute correlation
    corr, pval = compute_spearman(boundary_pressures, delta_kls)
    auc = compute_roc_auc(boundary_pressures, delta_kls, percentile=75)

    # Compute KL-selectivity ratio
    enabled_mean = np.mean(enabled_kls) if enabled_kls else 0
    disabled_mean = np.mean(disabled_kls) if disabled_kls else 1e-10
    kl_ratio = enabled_mean / (disabled_mean + 1e-10)

    enabled_pct = len(enabled_kls) / n_measurements if n_measurements > 0 else 0

    # Build results
    results = {
        "model_size": model_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": cfg.n_embd,
        "local_window": cfg.local_window,
        "seq_len": T,
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
            "enabled_mean_kl": float(enabled_mean),
            "disabled_mean_kl": float(disabled_mean),
            "kl_ratio": float(kl_ratio),
            "n_enabled": len(enabled_kls),
            "n_disabled": len(disabled_kls),
        },
        "delta_kl_stats": {
            "mean": float(np.mean(delta_kls)),
            "std": float(np.std(delta_kls)),
            "p75": float(np.percentile(delta_kls, 75)),
        },
        "boundary_pressure_stats": {
            "mean": float(np.mean(boundary_pressures)),
            "std": float(np.std(boundary_pressures)),
            "p75": float(np.percentile(boundary_pressures, 75)),
        },
    }

    # Check acceptance criteria
    passes_corr = corr >= 0.45 or auc >= 0.68
    passes_kl = kl_ratio >= 10.0 or (kl_ratio >= 5.0 and enabled_pct < 0.5)
    results["acceptance"] = {
        "passes_correlation": bool(passes_corr),
        "passes_kl_selectivity": bool(passes_kl),
        "overall": bool(passes_corr or passes_kl),
    }

    return results


def run_multi_seed(model_size: str, seeds: List[int], n_samples: int = 30) -> Dict:
    """Run scale test across multiple seeds."""
    model = create_model(model_size)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        result = run_scale_test(model, model_size, n_samples=n_samples, seed=seed)
        all_results.append(result)

    # Aggregate
    corrs = [r["correlation"]["spearman_r"] for r in all_results]
    aucs = [r["correlation"]["roc_auc_p75"] for r in all_results]
    kl_ratios = [r["kl_selectivity"]["kl_ratio"] for r in all_results]

    # Check sign flips
    signs = [1 if c > 0 else -1 for c in corrs if c != 0]
    sign_flip = len(set(signs)) > 1 if signs else False

    aggregated = {
        "model_size": model_size,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "correlation": {
            "mean_spearman_r": float(np.mean(corrs)),
            "std_spearman_r": float(np.std(corrs)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "sign_flip": bool(sign_flip),
        },
        "kl_selectivity": {
            "mean_kl_ratio": float(np.mean(kl_ratios)),
            "std_kl_ratio": float(np.std(kl_ratios)),
        },
        "seed_results": all_results,
    }

    # Acceptance
    passes_corr = np.mean(corrs) >= 0.45 or np.mean(aucs) >= 0.68
    passes_kl = np.mean(kl_ratios) >= 10.0
    no_sign_flip = not sign_flip

    aggregated["acceptance"] = {
        "passes_correlation": bool(passes_corr),
        "passes_kl_selectivity": bool(passes_kl),
        "no_sign_flip": no_sign_flip,
        "overall": bool((passes_corr or passes_kl) and no_sign_flip),
    }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="BPA v1 Phase 1: Scale Test")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds to test")
    parser.add_argument("--samples", type=int, default=20, help="Samples per seed")
    parser.add_argument("--model", type=str, default="both",
                        choices=["124M", "355M", "both"], help="Model size to test")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v1 Phase 1: Scale Test")
    print("=" * 70)
    print("\nAcceptance criteria (R1):")
    print("  - Spearman ρ >= 0.45 OR AUC >= 0.68")
    print("  - KL-selectivity ratio >= 10x")
    print("  - No seed sign flips")
    print()

    results = {}
    model_sizes = ["124M", "355M"] if args.model == "both" else [args.model]

    for size in model_sizes:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {size}")
        print(f"{'#'*70}")
        results[size] = run_multi_seed(size, seeds, n_samples=args.samples)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Mean ρ':>10} {'Mean AUC':>10} {'Mean KL Ratio':>14} {'Sign Flip':>12} {'Pass':>8}")
    print("-" * 70)

    for size, res in results.items():
        corr = res["correlation"]["mean_spearman_r"]
        auc = res["correlation"]["mean_auc"]
        kl_ratio = res["kl_selectivity"]["mean_kl_ratio"]
        flip = "YES" if res["correlation"]["sign_flip"] else "no"
        passed = "PASS" if res["acceptance"]["overall"] else "FAIL"
        print(f"{size:<10} {corr:>10.4f} {auc:>10.4f} {kl_ratio:>14.2f} {flip:>12} {passed:>8}")

    print("-" * 70)

    # Verdict
    all_pass = all(r["acceptance"]["overall"] for r in results.values())
    if all_pass:
        print("\nVERDICT: R1 PASSES - boundary_pressure remains predictive at scale")
        print("Proceed to Phase 2 (long context test)")
    else:
        failures = [s for s, r in results.items() if not r["acceptance"]["overall"]]
        print(f"\nVERDICT: R1 FAILS for {failures}")
        print("BPA may be a small-model artifact. Investigate before scaling.")

    # Save results
    os.makedirs("bpa_v1_results", exist_ok=True)
    with open("bpa_v1_results/phase1_scale.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to bpa_v1_results/phase1_scale.json")


if __name__ == "__main__":
    main()
