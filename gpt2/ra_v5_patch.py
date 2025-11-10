"""
Unified RA Patching for GPT-2

Replaces standard attention with UnifiedRAttention (direct layout emission).
Matches baseline speed (1.33ms) while providing architectural benefits:
- Reciprocity: Q can attend to K's context and vice versa
- Learned gates: Per-head w_std, w_rec control reciprocity usage
- Zero overhead: Direct folded layout emission

The key principle: we match baseline speed by splitting the per-head
dimension D into (D_std + R) and using a fused projection to emit
a folded layout [Q_std | K_low] and [K_std | Q_low], so reciprocal
attention is computed inside the same SDPA call without increasing
FLOPs. RA becomes a reparameterization of attention, not an extra cost.

Usage:
    from ra_v5_patch import patch_gpt2_with_ra_v5
    model = patch_gpt2_with_ra_v5(model, R=4, dropout=0.1)
"""

import sys
import os

# Add parent directory to path to import unified_ra
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from unified_ra import UnifiedRAttention


def patch_gpt2_with_ra_v5(model, R=4, dropout=0.1, use_self_restart=False):
    """
    Replace all attention modules in GPT-2 with Unified RA.

    Args:
        model: GPT-2 model to patch
        R: Reciprocal rank (default 4, validated optimal)
        dropout: Dropout probability
        use_self_restart: Enable self-restart mechanism (default False)

    Returns:
        Patched model
    """
    n_head = model.config.n_head
    n_embd = model.config.n_embd
    block_size = model.config.block_size

    restart_str = " + Self-Restart" if use_self_restart else ""
    print(f"Patching GPT-2 with Unified RA (R={R}){restart_str}...")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # Replace the attention module with Unified RA
        original_attn = block.attn

        # Create Unified RA module
        unified_ra_attn = UnifiedRAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            R=R,
            dropout=dropout,
            use_self_restart=use_self_restart,
        )

        # Copy over any bias if it exists
        if hasattr(original_attn, "bias") and hasattr(unified_ra_attn, "bias"):
            unified_ra_attn.bias = original_attn.bias

        # Replace the attention module
        block.attn = unified_ra_attn

        print(f"  Layer {i}: Standard Attention → Unified RA{restart_str}")

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with Unified RA{restart_str}")

    return model


def analyze_ra_v5_gates(model):
    """
    Analyze learned gate values across all Unified RA layers.

    Returns:
        Dictionary with gate statistics
    """
    w_std_values = []
    w_rec_values = []

    for block in model.transformer.h:
        if hasattr(block.attn, "w_std") and hasattr(block.attn, "w_rec"):
            w_std_values.extend(block.attn.w_std.detach().cpu().tolist())
            w_rec_values.extend(block.attn.w_rec.detach().cpu().tolist())

    if not w_std_values:
        return None

    import torch as t

    w_std_tensor = t.tensor(w_std_values)
    w_rec_tensor = t.tensor(w_rec_values)

    return {
        "w_std_mean": w_std_tensor.mean().item(),
        "w_std_min": w_std_tensor.min().item(),
        "w_std_max": w_std_tensor.max().item(),
        "w_std_std": w_std_tensor.std().item(),
        "w_rec_mean": w_rec_tensor.mean().item(),
        "w_rec_min": w_rec_tensor.min().item(),
        "w_rec_max": w_rec_tensor.max().item(),
        "w_rec_std": w_rec_tensor.std().item(),
    }
