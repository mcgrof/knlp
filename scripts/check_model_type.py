#!/usr/bin/env python3
"""
Check GPT-2 model type from checkpoint.

Determines whether a checkpoint contains:
- Standard GPT-2 (no KV compression)
- MLA (6x KV compression via latent)
- MLA+KVSplice (12x KV compression)

Usage:
    python scripts/check_model_type.py <checkpoint.pt>
"""

import sys
import torch


def check_model_type(checkpoint_path):
    """Check model type from checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" not in checkpoint:
        print("ERROR: Checkpoint does not contain 'model' state dict")
        return 1

    keys = list(checkpoint["model"].keys())

    print(f"\nFirst 20 checkpoint keys:")
    for k in keys[:20]:
        print(f"  {k}")

    # Check model type based on state dict keys
    all_keys_str = " ".join(keys)

    if any("kvsplice" in k for k in keys):
        print("\n" + "="*60)
        print("✓ MODEL TYPE: MLA+KVSplice")
        print("  - 12x KV cache compression")
        print("  - Uses learned compression on top of MLA latent")
        print("  - Expected memory savings: ~92%")
        print("="*60)
        return 0

    elif any("blocks.0.attn.W_q" in k for k in keys):
        print("\n" + "="*60)
        print("✓ MODEL TYPE: MLA (Multi-head Latent Attention)")
        print("  - 6x KV cache compression")
        print("  - Shared latent for K and V projections")
        print("  - Expected memory savings: ~83%")
        print("="*60)
        return 0

    elif any("transformer.wte" in k for k in keys):
        print("\n" + "="*60)
        print("✗ MODEL TYPE: Standard GPT-2")
        print("  - NO KV cache compression")
        print("  - Full multi-head attention")
        print("  - Expected memory savings: 0%")
        print("="*60)
        return 0

    else:
        print("\n" + "="*60)
        print("? MODEL TYPE: Unknown")
        print("  - Could not determine model type from checkpoint keys")
        print("="*60)
        return 1


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_model_type.py <checkpoint.pt>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    sys.exit(check_model_type(checkpoint_path))


if __name__ == "__main__":
    main()
