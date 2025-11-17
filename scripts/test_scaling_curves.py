#!/usr/bin/env python3
"""
Test script for scaling curves visualization
"""

from lib.scaling_curves import show_scaling_curves

# Test with GPT-2 124M configuration
print("GPT-2 124M (12 layers, d_model=768, mlp_dim=3072):")
print(show_scaling_curves(
    n_layers=12,
    d_model=768,
    mlp_dim=3072,
    param_count=124_000_000
))

print("\n" + "=" * 80 + "\n")

# Test with a different configuration
print("Deeper model (32 layers, d_model=1024, mlp_dim=4096):")
print(show_scaling_curves(
    n_layers=32,
    d_model=1024,
    mlp_dim=4096,
    param_count=500_000_000
))
