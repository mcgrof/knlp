#!/usr/bin/env python3
"""Test sensitivity extraction from Adam optimizer state."""

import os
import sys
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sensitivity import (
    extract_sensitivity,
    save_sensitivity_json,
    compute_variance_weights,
    compute_per_layer_top_b,
    load_sensitivity_json,
)


class DummyRGSAModel(nn.Module):
    """Simple model to test sensitivity extraction."""

    def __init__(self, n_layer=12, n_embd=768):
        super().__init__()
        self.n_layer = n_layer
        self.config = type("Config", (), {"n_layer": n_layer})()

        # Simulate GPT-2 attention structure
        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attn": nn.ModuleDict(
                                    {
                                        "c_attn": nn.Linear(n_embd, 3 * n_embd),
                                        "c_proj": nn.Linear(n_embd, n_embd),
                                    }
                                ),
                                "mlp": nn.Linear(n_embd, 4 * n_embd),  # Not attention
                            }
                        )
                        for _ in range(n_layer)
                    ]
                )
            }
        )

    def forward(self, x):
        return x


def test_sensitivity_extraction():
    """Test basic sensitivity extraction."""
    print("=" * 60)
    print("Testing sensitivity extraction")
    print("=" * 60)

    # Create model and optimizer
    model = DummyRGSAModel(n_layer=12, n_embd=768)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate some training steps to populate exp_avg_sq
    print("\nSimulating training steps...")
    for step in range(10):
        x = torch.randn(2, 768)
        # Simulate gradient computation by manually computing on attention params
        for i, block in enumerate(model.transformer["h"]):
            # Make later layers have higher gradients (simulate sensitivity)
            scale = 1.0 + i * 0.2
            block.attn.c_attn.weight.grad = (
                torch.randn_like(block.attn.c_attn.weight) * scale
            )
            block.attn.c_proj.weight.grad = (
                torch.randn_like(block.attn.c_proj.weight) * scale
            )
            block.mlp.weight.grad = torch.randn_like(block.mlp.weight) * 0.1

        optimizer.step()
        optimizer.zero_grad()

    # Extract sensitivity
    print("\nExtracting sensitivity...")
    sensitivity = extract_sensitivity(model, optimizer)

    print(f"\nS_layer shape: {sensitivity['S_layer'].shape}")
    print(f"S_layer values: {sensitivity['S_layer'].tolist()}")
    print(f"Number of mapped params: {len(sensitivity['param_sensitivity'])}")

    # Verify later layers have higher sensitivity
    S = sensitivity["S_layer"]
    print(f"\nSensitivity ratio (layer 11 / layer 0): {S[11] / S[0]:.2f}")
    assert S[11] > S[0], "Expected later layers to have higher sensitivity"
    print("PASS: Later layers have higher sensitivity as expected")

    return sensitivity


def test_variance_weights():
    """Test variance weight computation."""
    print("\n" + "=" * 60)
    print("Testing variance weights")
    print("=" * 60)

    # Create synthetic sensitivity with known pattern
    S_layer = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )

    # Test alpha=0 (uniform)
    weights_uniform = compute_variance_weights(S_layer, alpha=0.0)
    print(f"\nalpha=0.0 (uniform): {weights_uniform.tolist()}")
    assert torch.allclose(
        weights_uniform, torch.ones(12) / 12, atol=1e-6
    ), "alpha=0 should give uniform weights"
    print("PASS: alpha=0 gives uniform weights")

    # Test alpha=1 (linear)
    weights_linear = compute_variance_weights(S_layer, alpha=1.0)
    print(f"alpha=1.0 (linear): {weights_linear.tolist()}")
    assert weights_linear[-1] > weights_linear[0], "Higher S should get higher weight"
    print("PASS: alpha=1 gives higher weights to sensitive layers")

    # Test alpha=0.5
    weights_sqrt = compute_variance_weights(S_layer, alpha=0.5)
    print(f"alpha=0.5 (sqrt): {weights_sqrt.tolist()}")
    ratio_linear = weights_linear[-1] / weights_linear[0]
    ratio_sqrt = weights_sqrt[-1] / weights_sqrt[0]
    assert ratio_sqrt < ratio_linear, "alpha=0.5 should have smaller ratio than alpha=1"
    print("PASS: alpha=0.5 has intermediate skew")


def test_per_layer_top_b():
    """Test per-layer top_b computation."""
    print("\n" + "=" * 60)
    print("Testing per-layer top_b allocation")
    print("=" * 60)

    # Create weights with varying sensitivity
    S_layer = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    weights = compute_variance_weights(S_layer, alpha=1.0)

    # Compute per-layer top_b
    top_b_per_layer = compute_per_layer_top_b(
        weights, top_b_base=8, n_layer=12, top_b_min=2, top_b_max=16
    )
    print(f"\nPer-layer top_b: {top_b_per_layer}")
    print(f"Sum of top_b: {sum(top_b_per_layer)}")
    print(f"Expected sum (12 * 8): {12 * 8}")

    # Verify structure
    assert len(top_b_per_layer) == 12, "Should have 12 values"
    assert all(2 <= t <= 16 for t in top_b_per_layer), "All values should be in [2, 16]"
    assert (
        top_b_per_layer[-1] >= top_b_per_layer[0]
    ), "Higher sensitivity layers should get more budget"
    print("PASS: Per-layer top_b allocation works correctly")


def test_json_save_load():
    """Test JSON save/load."""
    print("\n" + "=" * 60)
    print("Testing JSON save/load")
    print("=" * 60)

    # Create sensitivity data
    sensitivity = {
        "S_layer": torch.tensor([1.0, 2.0, 3.0]),
        "param_sensitivity": {"layer.0": 1.0},
        "layer_param_map": {0: ["layer.0"]},
        "n_layer": 3,
    }

    # Save
    filepath = "/tmp/test_sensitivity.json"
    save_sensitivity_json(sensitivity, filepath, step=100, tokens_seen=10000)
    print(f"\nSaved to: {filepath}")

    # Load
    loaded = load_sensitivity_json(filepath)
    print(f"Loaded S_layer: {loaded['S_layer'].tolist()}")
    assert torch.allclose(
        loaded["S_layer"], sensitivity["S_layer"]
    ), "Loaded should match saved"
    print("PASS: JSON save/load works correctly")

    # Cleanup
    os.remove(filepath)


if __name__ == "__main__":
    test_sensitivity_extraction()
    test_variance_weights()
    test_per_layer_top_b()
    test_json_save_load()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
