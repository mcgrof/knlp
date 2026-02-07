#!/usr/bin/env python3
"""Test variance-weighted allocation in RGSA."""

import os
import sys
import json
import torch

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpt2.rgsa import GPT2_RGSA, RGSAConfig
from utils.sensitivity import (
    compute_variance_weights,
    compute_per_layer_top_b,
)


def test_uniform_allocation():
    """Test that uniform allocation (top_b_per_layer=None) works."""
    print("=" * 60)
    print("Testing uniform allocation")
    print("=" * 60)

    config = RGSAConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        top_b=8,
        top_b_per_layer=None,  # Uniform
        variance_alpha=0.0,
    )

    model = GPT2_RGSA(config)

    # Verify all layers have same top_b
    top_bs = [block.attn.top_b for block in model.transformer.h]
    print(f"Per-layer top_b: {top_bs}")

    assert all(t == 8 for t in top_bs), "All layers should have top_b=8"
    print("PASS: Uniform allocation works")


def test_variance_allocation():
    """Test that variance-weighted allocation applies per-layer top_b."""
    print("\n" + "=" * 60)
    print("Testing variance-weighted allocation")
    print("=" * 60)

    # Create per-layer top_b with variance
    top_b_per_layer = [4, 6, 8, 10, 12, 14]

    config = RGSAConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        top_b=8,  # Base (fallback)
        top_b_per_layer=top_b_per_layer,
        variance_alpha=1.0,
    )

    model = GPT2_RGSA(config)

    # Verify layers have correct per-layer top_b
    actual_top_bs = [block.attn.top_b for block in model.transformer.h]
    print(f"Expected top_b_per_layer: {top_b_per_layer}")
    print(f"Actual per-layer top_b: {actual_top_bs}")

    assert (
        actual_top_bs == top_b_per_layer
    ), f"Per-layer top_b mismatch: {actual_top_bs} != {top_b_per_layer}"
    print("PASS: Variance-weighted allocation works")


def test_variance_weights_computation():
    """Test variance weight computation from sensitivity."""
    print("\n" + "=" * 60)
    print("Testing variance weight computation")
    print("=" * 60)

    # Simulate sensitivity: earlier layers more sensitive
    S_layer = torch.tensor([12.0, 10.0, 8.0, 6.0, 4.0, 2.0])

    # alpha=1.0: linear
    weights_linear = compute_variance_weights(S_layer, alpha=1.0)
    print(f"S_layer: {S_layer.tolist()}")
    print(f"Weights (alpha=1.0): {[round(w, 4) for w in weights_linear.tolist()]}")

    # Higher sensitivity should get higher weight
    assert weights_linear[0] > weights_linear[-1], "Higher S should get higher weight"

    # Compute per-layer top_b
    top_b_per_layer = compute_per_layer_top_b(
        weights_linear,
        top_b_base=8,
        n_layer=6,
        top_b_min=2,
        top_b_max=16,
    )
    print(f"Computed top_b_per_layer: {top_b_per_layer}")

    # More sensitive layers should get more budget
    assert (
        top_b_per_layer[0] >= top_b_per_layer[-1]
    ), "More sensitive layers should get more budget"
    print("PASS: Variance weight computation works")


def test_model_forward():
    """Test that model forward pass works with variance allocation."""
    print("\n" + "=" * 60)
    print("Testing model forward pass with variance allocation")
    print("=" * 60)

    top_b_per_layer = [4, 6, 8, 10, 12, 14]

    config = RGSAConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        top_b=8,
        top_b_per_layer=top_b_per_layer,
        variance_alpha=1.0,
        block_size=512,  # Larger block size
    )

    model = GPT2_RGSA(config)
    model.eval()

    # Run forward pass with full sequence
    seq_len = 256
    x = torch.randint(0, config.vocab_size, (2, seq_len))
    with torch.no_grad():
        logits, loss = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")

    # Check shape - RGSA may return only last position in inference
    # The important thing is no crash and valid output
    assert logits.ndim == 3, f"Expected 3D output, got {logits.ndim}D"
    assert logits.shape[0] == 2, f"Batch size mismatch: {logits.shape[0]} != 2"
    assert logits.shape[-1] == config.vocab_size, f"Vocab size mismatch"
    print("PASS: Forward pass works with variance allocation")


def test_sensitivity_json_integration():
    """Test loading sensitivity from JSON and computing allocation."""
    print("\n" + "=" * 60)
    print("Testing sensitivity JSON integration")
    print("=" * 60)

    # Create test sensitivity file
    test_data = {
        "step": 100,
        "tokens_seen": 100000,
        "n_layer": 6,
        "S_layer": [12.0, 10.0, 8.0, 6.0, 4.0, 2.0],
        "summary": {},
        "layer_param_map": {},
    }

    test_path = "/tmp/test_sensitivity_integration.json"
    with open(test_path, "w") as f:
        json.dump(test_data, f)

    # Load and compute allocation
    from utils.sensitivity import compute_top_b_per_layer_from_file

    top_b_per_layer = compute_top_b_per_layer_from_file(
        test_path,
        top_b_base=8,
        alpha=1.0,
        top_b_min=2,
        top_b_max=16,
    )
    print(f"Loaded S_layer: {test_data['S_layer']}")
    print(f"Computed top_b_per_layer: {top_b_per_layer}")

    # More sensitive layers should get more budget
    assert (
        top_b_per_layer[0] >= top_b_per_layer[-1]
    ), "More sensitive layers should get more budget"
    print("PASS: Sensitivity JSON integration works")

    # Cleanup
    os.remove(test_path)


if __name__ == "__main__":
    test_uniform_allocation()
    test_variance_allocation()
    test_variance_weights_computation()
    test_model_forward()
    test_sensitivity_json_integration()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
