#!/usr/bin/env python
"""
RGSA v18 Phase 2: Find Predictors

Correlate usage signals (far_mass, attn_entropy, max_weight) with impact_kl
to determine which cheap signal best predicts head importance.

Metrics:
- Spearman rank correlation
- Top-K overlap (e.g., top-10 heads)
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.rgsa import GPT2_RGSA, RGSAConfig, HeadMetrics, ImpactMetrics


def compute_spearman_correlation(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[float, float]:
    """Compute Spearman rank correlation between two tensors."""
    x_flat = x.view(-1).cpu().numpy()
    y_flat = y.view(-1).cpu().numpy()
    corr, pvalue = scipy_stats.spearmanr(x_flat, y_flat)
    return float(corr), float(pvalue)


def compute_topk_overlap(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> float:
    """Compute overlap between top-k indices of two tensors."""
    x_flat = x.view(-1)
    y_flat = y.view(-1)

    _, x_topk = torch.topk(x_flat, min(k, x_flat.numel()))
    _, y_topk = torch.topk(y_flat, min(k, y_flat.numel()))

    x_set = set(x_topk.tolist())
    y_set = set(y_topk.tolist())

    overlap = len(x_set & y_set) / k
    return overlap


def run_correlation_analysis(
    model: GPT2_RGSA,
    idx: torch.Tensor,
    n_batches: int = 3,
) -> Dict:
    """
    Run correlation analysis between usage metrics and impact_kl.

    Args:
        model: RGSA model
        idx: Input token indices [B, T]
        n_batches: Number of batches to average over

    Returns:
        Dict with correlation results
    """
    n_layer = model.config.n_layer
    n_head = model.config.n_head

    # Accumulate metrics over batches
    far_mass_accum = torch.zeros(n_layer, n_head)
    attn_entropy_accum = torch.zeros(n_layer, n_head)
    max_weight_accum = torch.zeros(n_layer, n_head)
    impact_kl_accum = torch.zeros(n_layer, n_head)

    print(f"Computing metrics over {n_batches} batches...")
    for batch_idx in range(n_batches):
        print(f"  Batch {batch_idx + 1}/{n_batches}")

        # Randomize input slightly to get different attention patterns
        if batch_idx > 0:
            idx = torch.randint(0, model.config.vocab_size, idx.shape, device=idx.device)

        # Compute head metrics
        head_metrics = model.compute_head_metrics(idx)
        far_mass_accum += head_metrics.far_mass.cpu()
        attn_entropy_accum += head_metrics.attn_entropy.cpu()
        max_weight_accum += head_metrics.max_weight.cpu()

        # Compute impact KL for all heads
        impact_metrics = model.compute_drop_impact_kl(idx, heads_to_measure=None)
        impact_kl_accum += impact_metrics.impact_kl.cpu()

    # Average
    far_mass = far_mass_accum / n_batches
    attn_entropy = attn_entropy_accum / n_batches
    max_weight = max_weight_accum / n_batches
    impact_kl = impact_kl_accum / n_batches

    print("\nComputing correlations...")

    # Compute correlations with impact_kl
    results = {"usage_metrics": {}, "correlations": {}, "topk_overlap": {}}

    # Store raw values
    results["usage_metrics"]["far_mass"] = {
        "mean": far_mass.mean().item(),
        "std": far_mass.std().item(),
        "values": far_mass.tolist(),
    }
    results["usage_metrics"]["attn_entropy"] = {
        "mean": attn_entropy.mean().item(),
        "std": attn_entropy.std().item(),
        "values": attn_entropy.tolist(),
    }
    results["usage_metrics"]["max_weight"] = {
        "mean": max_weight.mean().item(),
        "std": max_weight.std().item(),
        "values": max_weight.tolist(),
    }
    results["usage_metrics"]["impact_kl"] = {
        "mean": impact_kl.mean().item(),
        "std": impact_kl.std().item(),
        "values": impact_kl.tolist(),
    }

    # Spearman correlations
    signals = [
        ("far_mass", far_mass),
        ("attn_entropy", attn_entropy),
        ("max_weight", max_weight),
        ("concentration", 1.0 / (max_weight + 1e-8)),  # Inverted max_weight
    ]

    for name, signal in signals:
        corr, pvalue = compute_spearman_correlation(signal, impact_kl)
        results["correlations"][name] = {
            "spearman_r": corr,
            "pvalue": pvalue,
            "significant": pvalue < 0.05,
        }
        print(f"  {name}: r={corr:.4f}, p={pvalue:.4f}")

    # Top-K overlap
    print("\nTop-K overlap with impact_kl:")
    for k in [5, 10, 15]:
        for name, signal in signals:
            overlap = compute_topk_overlap(signal, impact_kl, k=k)
            key = f"{name}_top{k}"
            results["topk_overlap"][key] = overlap
        print(
            f"  top-{k}: far_mass={results['topk_overlap'][f'far_mass_top{k}']:.2f}, "
            f"entropy={results['topk_overlap'][f'attn_entropy_top{k}']:.2f}, "
            f"max_weight={results['topk_overlap'][f'max_weight_top{k}']:.2f}"
        )

    # Determine best predictor
    correlations = results["correlations"]
    best_signal = max(
        [s for s in correlations.keys()],
        key=lambda s: abs(correlations[s]["spearman_r"]),
    )
    results["best_predictor"] = {
        "signal": best_signal,
        "spearman_r": correlations[best_signal]["spearman_r"],
        "recommendation": (
            f"Use {best_signal} as predictor "
            f"(r={correlations[best_signal]['spearman_r']:.4f})"
        ),
    }

    return results


def main():
    print("RGSA v18 Phase 2: Find Predictors")
    print("=" * 60)

    # Create model with reasonable config
    cfg = RGSAConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        local_window=128,
        chunk_size=32,
        top_b=4,
    )

    model = GPT2_RGSA(cfg)
    model.eval()

    # Create input
    B, T = 4, 256
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    # Run analysis
    results = run_correlation_analysis(model, idx, n_batches=3)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBest predictor: {results['best_predictor']['signal']}")
    print(f"Spearman r: {results['best_predictor']['spearman_r']:.4f}")
    print(f"Recommendation: {results['best_predictor']['recommendation']}")

    # Show correlation table
    print("\nCorrelation Table:")
    print(f"{'Signal':<20} {'Spearman r':<12} {'p-value':<12} {'Significant'}")
    print("-" * 60)
    for name, data in results["correlations"].items():
        print(
            f"{name:<20} {data['spearman_r']:>10.4f}   "
            f"{data['pvalue']:>10.4f}   {'*' if data['significant'] else ''}"
        )

    # Save results
    os.makedirs("rgsa_v18_results", exist_ok=True)
    with open("rgsa_v18_results/phase2_correlation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to rgsa_v18_results/phase2_correlation.json")


if __name__ == "__main__":
    main()
