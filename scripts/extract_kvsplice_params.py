#!/usr/bin/env python3
"""Extract KVSplice scale and shift parameters from checkpoint."""

import sys
import torch
import torch.nn.functional as F
import numpy as np


def extract_kvsplice_params(checkpoint_path):
    """Extract scale and shift values from KVSplice layers."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" not in checkpoint:
        print("ERROR: Checkpoint does not contain 'model' state dict")
        return

    state_dict = checkpoint["model"]

    # Find all KVSplice transform parameters
    scale_keys = [k for k in state_dict.keys() if "kvsplice.transform_scale" in k]
    shift_keys = [k for k in state_dict.keys() if "kvsplice.transform_shift" in k]

    if not scale_keys:
        print("No KVSplice parameters found in checkpoint")
        return

    print(f"\nFound {len(scale_keys)} KVSplice layers\n")

    all_scales = []
    all_shifts = []

    for scale_key, shift_key in zip(sorted(scale_keys), sorted(shift_keys)):
        layer_name = scale_key.split(".")[0]  # e.g., "blocks.0"

        scale_raw = state_dict[scale_key]
        shift = state_dict[shift_key]

        # Apply softplus to get actual scale values
        scale = F.softplus(scale_raw).cpu().numpy()
        shift = shift.cpu().numpy()

        all_scales.append(scale)
        all_shifts.append(shift)

        print(f"{layer_name}:")
        print(f"  Scale: mean={scale.mean():.4f}, std={scale.std():.4f}, "
              f"min={scale.min():.4f}, max={scale.max():.4f}")
        print(f"  Shift: mean={shift.mean():.4f}, std={shift.std():.4f}, "
              f"min={shift.min():.4f}, max={shift.max():.4f}")

        # Check for potential pruning candidates (very low scale = unimportant)
        low_scale_threshold = 0.1
        low_scale_dims = (scale < low_scale_threshold).sum()
        low_scale_pct = 100 * low_scale_dims / len(scale)
        print(f"  Low scale dims (<{low_scale_threshold}): {low_scale_dims}/{len(scale)} ({low_scale_pct:.1f}%)")

        # Check for high importance dims
        high_scale_threshold = 10.0
        high_scale_dims = (scale > high_scale_threshold).sum()
        high_scale_pct = 100 * high_scale_dims / len(scale)
        print(f"  High scale dims (>{high_scale_threshold}): {high_scale_dims}/{len(scale)} ({high_scale_pct:.1f}%)")
        print()

    # Global statistics across all layers
    all_scales = np.array(all_scales)  # [n_layers, d_in]
    all_shifts = np.array(all_shifts)

    print("=" * 60)
    print("GLOBAL STATISTICS (averaged across all layers)")
    print("=" * 60)

    avg_scale = all_scales.mean(axis=0)  # [d_in]
    avg_shift = all_shifts.mean(axis=0)

    print(f"\nScale (averaged across {len(all_scales)} layers):")
    print(f"  Mean: {avg_scale.mean():.4f}")
    print(f"  Std:  {avg_scale.std():.4f}")
    print(f"  Min:  {avg_scale.min():.4f}")
    print(f"  Max:  {avg_scale.max():.4f}")

    print(f"\nShift (averaged across {len(all_shifts)} layers):")
    print(f"  Mean: {avg_shift.mean():.4f}")
    print(f"  Std:  {avg_shift.std():.4f}")
    print(f"  Min:  {avg_shift.min():.4f}")
    print(f"  Max:  {avg_shift.max():.4f}")

    # Pruning potential analysis
    low_scale_dims = (avg_scale < 0.1).sum()
    low_scale_pct = 100 * low_scale_dims / len(avg_scale)

    print(f"\n" + "=" * 60)
    print("PRUNING POTENTIAL ANALYSIS")
    print("=" * 60)
    print(f"\nDimensions with low importance (avg scale < 0.1):")
    print(f"  Count: {low_scale_dims}/{len(avg_scale)} ({low_scale_pct:.1f}%)")

    if low_scale_pct > 5:
        print(f"\n⚠️  {low_scale_pct:.1f}% of dimensions have very low scale!")
        print("  These dimensions could potentially be pruned with minimal")
        print("  impact on model quality, reducing the latent dimension.")
    else:
        print("\n✓ All dimensions appear important (low pruning potential)")

    # Show distribution of scale values
    print(f"\nScale distribution (percentiles):")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(avg_scale, p)
        print(f"  {p:2d}th percentile: {val:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_kvsplice_params.py <checkpoint.pt>")
        sys.exit(1)

    extract_kvsplice_params(sys.argv[1])
