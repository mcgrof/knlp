#!/usr/bin/env python
"""
RGSA v19 Phase 2: Correlation Analysis

Correlate conditional signals with ΔKL to determine if any
cheap predictor can identify when far-context access matters.

Metrics:
- Spearman rank correlation
- ROC-AUC for predicting ΔKL > threshold
- Stability across random seeds
"""

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

from gpt2.rgsa import GPT2_RGSA, RGSAConfig, ConditionalImpactTracker


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
        return 0.5  # Can't compute AUC with single class

    try:
        auc = roc_auc_score(labels, signal)
        return float(auc)
    except ValueError:
        return 0.5


def run_correlation_analysis(
    model: GPT2_RGSA,
    n_samples: int = 50,
    n_positions: int = 8,
    n_heads_per_sample: int = 4,
    seed: int = 1,
) -> Dict:
    """
    Run correlation analysis between signals and conditional ΔKL.

    Args:
        model: RGSA model
        n_samples: Number of input samples to collect
        n_positions: Positions to measure per sample
        n_heads_per_sample: Heads to measure per sample
        seed: Random seed

    Returns:
        Dict with correlation results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = model.config
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    T = 128  # Sequence length

    # Create tracker for round-robin head selection
    tracker = ConditionalImpactTracker(
        n_layer=n_layer,
        n_head=n_head,
        seq_len=T,
        heads_per_eval=n_heads_per_sample,
        positions_per_eval=n_positions,
    )

    # Collect (signal, delta_kl) pairs
    samples = {
        "query_norm": [],
        "boundary_pressure": [],
        "attn_variance": [],
        "local_entropy": [],
        "delta_kl": [],
        "position": [],
        "layer": [],
        "head": [],
    }

    print(f"Collecting {n_samples} samples...")
    for sample_idx in range(n_samples):
        if (sample_idx + 1) % 10 == 0:
            print(f"  Sample {sample_idx + 1}/{n_samples}")

        # Random input
        idx = torch.randint(0, cfg.vocab_size, (2, T))

        # Get heads and positions
        heads = tracker.get_heads_to_measure()
        positions = tracker.sample_positions(local_window=cfg.local_window)

        # Compute conditional signals
        signals = model.compute_conditional_signals(idx, positions=positions)

        # Compute conditional ΔKL
        delta_kls = model.compute_conditional_impact_kl(idx, positions, heads)

        # Collect paired samples
        for (pos, layer_idx, head_idx), delta_kl in delta_kls.items():
            pos_idx = positions.index(pos) if pos in positions else 0

            samples["query_norm"].append(
                signals["query_norm"][:, pos_idx, layer_idx].mean().item()
            )
            samples["boundary_pressure"].append(
                signals["boundary_pressure"][:, pos_idx, layer_idx, head_idx].mean().item()
            )
            samples["attn_variance"].append(
                signals["attn_variance"][:, pos_idx, layer_idx].mean().item()
            )
            samples["local_entropy"].append(
                signals["local_entropy"][:, pos_idx].mean().item()
            )
            samples["delta_kl"].append(delta_kl)
            samples["position"].append(pos)
            samples["layer"].append(layer_idx)
            samples["head"].append(head_idx)

            # Update tracker EMA
            tracker.update(pos, layer_idx, head_idx, delta_kl)

    # Convert to numpy
    for k, v in samples.items():
        samples[k] = np.array(v)

    print(f"\nCollected {len(samples['delta_kl'])} (position, head) measurements")

    # Compute correlations
    signal_names = ["query_norm", "boundary_pressure", "attn_variance", "local_entropy"]
    delta_kl = samples["delta_kl"]

    results = {
        "n_samples": len(delta_kl),
        "seed": seed,
        "delta_kl_stats": {
            "mean": float(np.mean(delta_kl)),
            "std": float(np.std(delta_kl)),
            "max": float(np.max(delta_kl)),
            "min": float(np.min(delta_kl)),
            "p75": float(np.percentile(delta_kl, 75)),
        },
        "correlations": {},
    }

    print("\nCorrelation Analysis:")
    print("-" * 60)
    print(f"{'Signal':<20} {'Spearman r':>12} {'p-value':>10} {'AUC':>8}")
    print("-" * 60)

    for signal_name in signal_names:
        signal = samples[signal_name]

        # Spearman correlation
        corr, pval = compute_spearman(signal, delta_kl)

        # ROC-AUC at 75th percentile
        auc = compute_roc_auc(signal, delta_kl, percentile=75)

        results["correlations"][signal_name] = {
            "spearman_r": corr,
            "pvalue": pval,
            "roc_auc_p75": auc,
            "significant": pval < 0.05 and abs(corr) > 0.1,
        }

        sig_marker = "*" if pval < 0.05 else ""
        print(f"{signal_name:<20} {corr:>12.4f} {pval:>10.4f} {auc:>8.3f} {sig_marker}")

    print("-" * 60)

    # Check tracker statistics
    bucket_stats = tracker.get_bucket_statistics()
    results["bucket_stats"] = {k: v.item() for k, v in bucket_stats.items()}
    results["is_conditional"] = tracker.is_conditional()

    print("\nPosition Bucket Statistics (ΔKL):")
    for name, val in bucket_stats.items():
        print(f"  {name}: {val.item():.6f}")
    print(f"\nIs conditional (late >> early): {results['is_conditional']}")

    # Determine best predictor
    best_signal = None
    best_score = 0
    for name, data in results["correlations"].items():
        score = abs(data["spearman_r"]) + (data["roc_auc_p75"] - 0.5)
        if score > best_score:
            best_score = score
            best_signal = name

    results["best_predictor"] = {
        "signal": best_signal,
        "spearman_r": results["correlations"][best_signal]["spearman_r"],
        "roc_auc": results["correlations"][best_signal]["roc_auc_p75"],
        "passes_threshold": (
            abs(results["correlations"][best_signal]["spearman_r"]) > 0.3
            or results["correlations"][best_signal]["roc_auc_p75"] > 0.65
        ),
    }

    return results


def run_multi_seed_analysis(model: GPT2_RGSA, seeds: List[int]) -> Dict:
    """Run correlation analysis across multiple seeds."""
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        result = run_correlation_analysis(model, n_samples=30, seed=seed)
        all_results.append(result)

    # Aggregate correlations across seeds
    signal_names = list(all_results[0]["correlations"].keys())
    aggregated = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "signal_stats": {},
    }

    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"{'Signal':<20} {'Mean r':>10} {'Std r':>10} {'Sign Flip':>12}")
    print("-" * 60)

    for signal in signal_names:
        corrs = [r["correlations"][signal]["spearman_r"] for r in all_results]
        aucs = [r["correlations"][signal]["roc_auc_p75"] for r in all_results]

        # Check for sign flips
        signs = [1 if c > 0 else -1 for c in corrs if c != 0]
        sign_flip = len(set(signs)) > 1 if signs else False

        aggregated["signal_stats"][signal] = {
            "mean_corr": float(np.mean(corrs)),
            "std_corr": float(np.std(corrs)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "sign_flip": bool(sign_flip),
            "stable": bool(not sign_flip and np.std(corrs) < 0.2),
        }

        flip_str = "YES" if sign_flip else "no"
        print(f"{signal:<20} {np.mean(corrs):>10.4f} {np.std(corrs):>10.4f} {flip_str:>12}")

    print("-" * 60)

    # Find best stable signal
    best = None
    best_score = 0
    for signal, stats in aggregated["signal_stats"].items():
        if stats["stable"]:
            score = abs(stats["mean_corr"]) + (stats["mean_auc"] - 0.5)
            if score > best_score:
                best_score = score
                best = signal

    aggregated["best_stable_signal"] = best
    aggregated["verdict"] = (
        "VIABLE_SIGNAL" if best and aggregated["signal_stats"][best]["mean_auc"] > 0.55
        else "NO_VIABLE_SIGNAL"
    )

    return aggregated


def main():
    print("RGSA v19 Phase 2: Correlation Analysis")
    print("=" * 60)

    # Create model
    cfg = RGSAConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        local_window=64,
        chunk_size=32,
        top_b=4,
    )

    model = GPT2_RGSA(cfg)
    model.eval()

    # Run multi-seed analysis
    seeds = [1, 2, 3]
    results = run_multi_seed_analysis(model, seeds)

    # Print verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"Best stable signal: {results['best_stable_signal']}")
    print(f"Verdict: {results['verdict']}")

    if results["verdict"] == "NO_VIABLE_SIGNAL":
        print("\nNo conditional signal correlates reliably with ΔKL.")
        print("This suggests head importance is NOT fundamentally conditional,")
        print("or the correlations are too weak to be usable.")
    else:
        best = results["best_stable_signal"]
        stats = results["signal_stats"][best]
        print(f"\nSignal '{best}' shows promise:")
        print(f"  Mean correlation: {stats['mean_corr']:.4f}")
        print(f"  Mean AUC: {stats['mean_auc']:.4f}")
        print(f"  Stable across seeds: {stats['stable']}")

    # Save results
    os.makedirs("rgsa_v19_results", exist_ok=True)
    with open("rgsa_v19_results/phase2_correlation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to rgsa_v19_results/phase2_correlation.json")


if __name__ == "__main__":
    main()
