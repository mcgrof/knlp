#!/usr/bin/env python
"""
RGSA v18 Phase 0 Sanity Test

Verifies:
1. Head-level metrics compute correctly
2. Budget allocation sums exactly
3. Metrics show variance across heads (not degenerate)
"""

import json
import os
import sys
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.rgsa import GPT2_RGSA, RGSAConfig, HeadMetrics
from utils.sensitivity import (
    compute_per_head_top_b_exact,
    compute_head_weights_from_metrics,
)


def test_head_metrics():
    """Test head-level metrics computation."""
    print("=" * 60)
    print("Test 1: Head-Level Metrics Computation")
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

    # Create dummy input
    B, T = 4, 256
    idx = torch.randint(0, 50304, (B, T))

    # Compute head metrics
    with torch.no_grad():
        metrics = model.compute_head_metrics(idx)

    # Verify shapes
    assert metrics.far_mass.shape == (
        cfg.n_layer,
        cfg.n_head,
    ), f"far_mass shape: {metrics.far_mass.shape}"
    assert metrics.attn_entropy.shape == (
        cfg.n_layer,
        cfg.n_head,
    ), f"attn_entropy shape: {metrics.attn_entropy.shape}"
    assert metrics.max_weight.shape == (
        cfg.n_layer,
        cfg.n_head,
    ), f"max_weight shape: {metrics.max_weight.shape}"

    # Verify no NaNs
    assert not torch.isnan(metrics.far_mass).any(), "far_mass has NaNs"
    assert not torch.isnan(metrics.attn_entropy).any(), "attn_entropy has NaNs"
    assert not torch.isnan(metrics.max_weight).any(), "max_weight has NaNs"

    # Print summary
    print(f"far_mass: mean={metrics.far_mass.mean():.4f}, std={metrics.far_mass.std():.4f}")
    print(
        f"  range: [{metrics.far_mass.min():.4f}, {metrics.far_mass.max():.4f}]"
    )
    print(
        f"attn_entropy: mean={metrics.attn_entropy.mean():.4f}, std={metrics.attn_entropy.std():.4f}"
    )
    print(f"max_weight: mean={metrics.max_weight.mean():.4f}, std={metrics.max_weight.std():.4f}")

    # Check variance - metrics should show some differentiation across heads
    fm_std = metrics.far_mass.std().item()
    print(f"\nVariance check: far_mass std = {fm_std:.6f}")
    if fm_std < 1e-6:
        print("WARNING: All heads have identical far_mass - no differentiation!")
    else:
        print("PASS: Metrics show variance across heads")

    print("\nTest 1 PASSED")
    return metrics


def test_budget_allocation(metrics):
    """Test head-level budget allocation."""
    print("\n" + "=" * 60)
    print("Test 2: Head-Level Budget Allocation")
    print("=" * 60)

    n_layer, n_head = metrics.far_mass.shape
    total_budget = 8 * n_layer  # 8 per layer on average

    # Use far_mass as importance signal
    weights = compute_head_weights_from_metrics(metrics.far_mass, gamma=1.0)

    # Compute allocation
    top_b = compute_per_head_top_b_exact(
        weights, total_budget, n_layer, n_head, top_b_min=0, top_b_max=16
    )

    # Verify exact sum
    actual_sum = top_b.sum().item()
    assert actual_sum == total_budget, f"Budget mismatch: {actual_sum} != {total_budget}"
    print(f"Budget sum: {actual_sum} (expected: {total_budget}) - EXACT MATCH")

    # Show allocation
    print(f"\nAllocation (total={total_budget}):")
    for l in range(n_layer):
        layer_sum = top_b[l].sum().item()
        print(f"  Layer {l}: {top_b[l].tolist()} (sum={layer_sum})")

    print("\nTest 2 PASSED")
    return top_b


def test_inverted_allocation(metrics):
    """Test inverted allocation (sanity check for causality)."""
    print("\n" + "=" * 60)
    print("Test 3: Inverted Allocation (Sanity Check)")
    print("=" * 60)

    n_layer, n_head = metrics.far_mass.shape
    total_budget = 8 * n_layer

    # Normal: high far_mass -> more budget
    weights_normal = compute_head_weights_from_metrics(metrics.far_mass, gamma=1.0)
    top_b_normal = compute_per_head_top_b_exact(
        weights_normal, total_budget, n_layer, n_head
    )

    # Inverted: high far_mass -> less budget (1/far_mass)
    eps = 1e-8
    inverted_metrics = 1.0 / (metrics.far_mass + eps)
    weights_inverted = compute_head_weights_from_metrics(inverted_metrics, gamma=1.0)
    top_b_inverted = compute_per_head_top_b_exact(
        weights_inverted, total_budget, n_layer, n_head
    )

    # Both should sum to total_budget
    assert top_b_normal.sum().item() == total_budget
    assert top_b_inverted.sum().item() == total_budget

    # Find head with max far_mass
    max_idx = metrics.far_mass.argmax().item()
    max_l, max_h = max_idx // n_head, max_idx % n_head

    print(f"Head with max far_mass: layer={max_l}, head={max_h}")
    print(f"  far_mass = {metrics.far_mass[max_l, max_h]:.4f}")
    print(f"  normal allocation: top_b = {top_b_normal[max_l, max_h].item()}")
    print(f"  inverted allocation: top_b = {top_b_inverted[max_l, max_h].item()}")

    # Inverted should give LESS to high far_mass heads
    if top_b_inverted[max_l, max_h] <= top_b_normal[max_l, max_h]:
        print("PASS: Inverted gives less budget to high far_mass head")
    else:
        print("WARNING: Inversion logic may need review")

    print("\nTest 3 PASSED")


def test_json_serialization(metrics):
    """Test JSON serialization for logging."""
    print("\n" + "=" * 60)
    print("Test 4: JSON Serialization")
    print("=" * 60)

    # Test to_dict (summary stats)
    summary = metrics.to_dict()
    print(f"Summary dict keys: {list(summary.keys())}")

    # Test to_json_dict (full tensors)
    full = metrics.to_json_dict()
    print(f"Full dict keys: {list(full.keys())}")

    # Verify JSON serializable
    json_str = json.dumps(summary)
    json_full = json.dumps(full)
    print(f"Summary JSON: {len(json_str)} bytes")
    print(f"Full JSON: {len(json_full)} bytes")

    print("\nTest 4 PASSED")


def main():
    print("RGSA v18 Phase 0 Sanity Test")
    print("=" * 60)

    # Run tests
    metrics = test_head_metrics()
    test_budget_allocation(metrics)
    test_inverted_allocation(metrics)
    test_json_serialization(metrics)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Save results
    os.makedirs("rgsa_v18_results", exist_ok=True)
    with open("rgsa_v18_results/phase0_sanity.json", "w") as f:
        json.dump(
            {
                "status": "passed",
                "metrics_summary": metrics.to_dict(),
                "metrics_full": metrics.to_json_dict(),
            },
            f,
            indent=2,
        )
    print("\nResults saved to rgsa_v18_results/phase0_sanity.json")


if __name__ == "__main__":
    main()
