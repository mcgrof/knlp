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
from unified_ra import UnifiedRAttention, ReciprocalMLP


def patch_gpt2_with_ra_v5(
    model, R=4, dropout=0.1, use_self_restart=False, per_head_gates=False
):
    """
    Replace all attention modules in GPT-2 with Unified RA.

    Args:
        model: GPT-2 model to patch
        R: Reciprocal rank (default 4, validated optimal)
        dropout: Dropout probability
        use_self_restart: Enable self-restart mechanism (default False)
        per_head_gates: Use per-head gates instead of per-layer (default False)

    Returns:
        Patched model
    """
    n_head = model.config.n_head
    n_embd = model.config.n_embd
    block_size = model.config.block_size

    restart_str = " + Self-Restart" if use_self_restart else ""
    gate_str = "per-head" if per_head_gates else "per-layer"
    print(f"Patching GPT-2 with Unified RA (R={R}, {gate_str} gates){restart_str}...")

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
            per_head_gates=per_head_gates,
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


def patch_gpt2_with_rmlp(
    model,
    expansion=4,
    R_ff=64,
    dropout=0.0,
    use_mixer=False,
    use_gates=False,
    tie_up_low=False,
):
    """
    Replace all MLP modules in GPT-2 with Reciprocal MLP.

    Args:
        model: GPT-2 model to patch
        expansion: MLP expansion ratio (default 4)
        R_ff: Reciprocal rank for MLP (default 64)
        dropout: Dropout probability
        use_mixer: Add 1x1 mixer on h_low
        use_gates: Add per-token learned gates
        tie_up_low: Tie up_low to transposed subset of up_std

    Returns:
        Patched model
    """
    n_embd = model.config.n_embd

    print(f"Patching GPT-2 with R-MLP (R_ff={R_ff}, expansion={expansion})...")
    if use_mixer:
        print("  - Mixer enabled (1x1 linear on h_low)")
    if use_gates:
        print("  - Per-token gates enabled (discoverability)")
    if tie_up_low:
        print("  - Weight tying enabled (up_low tied to up_std^T)")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # Replace the MLP module with Reciprocal MLP
        original_mlp = block.mlp

        # Create Reciprocal MLP module
        reciprocal_mlp = ReciprocalMLP(
            n_embd=n_embd,
            expansion=expansion,
            R_ff=R_ff,
            dropout=dropout,
            use_mixer=use_mixer,
            use_gates=use_gates,
            tie_up_low=tie_up_low,
        )

        # Replace the MLP module
        block.mlp = reciprocal_mlp

        print(f"  Layer {i}: Standard MLP → R-MLP")

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with R-MLP")

    return model


def patch_gpt2_with_unified_ra_and_rmlp(
    model,
    R=4,
    attn_dropout=0.1,
    use_self_restart=False,
    mlp_expansion=4,
    R_ff=64,
    mlp_dropout=0.0,
    use_mixer=False,
    use_gates=False,
    tie_up_low=False,
    per_head_gates=False,
):
    """
    Replace both attention and MLP modules in GPT-2 with Unified RA + R-MLP.

    Args:
        model: GPT-2 model to patch
        R: Reciprocal rank for attention (default 4)
        attn_dropout: Attention dropout probability
        use_self_restart: Enable self-restart for attention
        mlp_expansion: MLP expansion ratio (default 4)
        R_ff: Reciprocal rank for MLP (default 64)
        mlp_dropout: MLP dropout probability
        use_mixer: Add 1x1 mixer on h_low
        use_gates: Add per-token learned gates
        tie_up_low: Tie up_low to transposed subset of up_std
        per_head_gates: Use per-head gates for RA (default False=per-layer)

    Returns:
        Patched model
    """
    # First patch attention with Unified RA
    model = patch_gpt2_with_ra_v5(
        model,
        R=R,
        dropout=attn_dropout,
        use_self_restart=use_self_restart,
        per_head_gates=per_head_gates,
    )

    # Then patch MLP with R-MLP
    model = patch_gpt2_with_rmlp(
        model,
        expansion=mlp_expansion,
        R_ff=R_ff,
        dropout=mlp_dropout,
        use_mixer=use_mixer,
        use_gates=use_gates,
        tie_up_low=tie_up_low,
    )

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


def analyze_rmlp_gates(model):
    """
    Analyze learned gate values across all R-MLP layers.

    Returns:
        Dictionary with gate statistics
    """
    w_std_values = []
    w_rec_values = []
    gate_alpha_values = []

    for block in model.transformer.h:
        if hasattr(block.mlp, "w_std") and hasattr(block.mlp, "w_rec"):
            w_std_values.append(block.mlp.w_std.detach().cpu().item())
            w_rec_values.append(block.mlp.w_rec.detach().cpu().item())

            if hasattr(block.mlp, "gate_alpha") and block.mlp.gate_alpha is not None:
                gate_alpha_values.append(block.mlp.gate_alpha.detach().cpu().item())

    if not w_std_values:
        return None

    import torch as t

    w_std_tensor = t.tensor(w_std_values)
    w_rec_tensor = t.tensor(w_rec_values)

    stats = {
        "w_std_mean": w_std_tensor.mean().item(),
        "w_std_min": w_std_tensor.min().item(),
        "w_std_max": w_std_tensor.max().item(),
        "w_std_std": w_std_tensor.std().item(),
        "w_rec_mean": w_rec_tensor.mean().item(),
        "w_rec_min": w_rec_tensor.min().item(),
        "w_rec_max": w_rec_tensor.max().item(),
        "w_rec_std": w_rec_tensor.std().item(),
    }

    if gate_alpha_values:
        gate_alpha_tensor = t.tensor(gate_alpha_values)
        stats.update(
            {
                "gate_alpha_mean": gate_alpha_tensor.mean().item(),
                "gate_alpha_min": gate_alpha_tensor.min().item(),
                "gate_alpha_max": gate_alpha_tensor.max().item(),
                "gate_alpha_std": gate_alpha_tensor.std().item(),
            }
        )

    return stats


def patch_gpt2_with_kv_pruning(
    model, k_keep=391, recency=64, learn_ratio=False, dropout=0.1
):
    """
    Replace all attention modules in GPT-2 with KV-pruned attention.

    Args:
        model: GPT-2 model to patch
        k_keep: Number of tokens to keep (default: 391 for golden ratio)
        recency: Number of recent tokens to always keep (default: 64)
        learn_ratio: If True, learn optimal pruning ratio (default: False)
        dropout: Dropout probability

    Returns:
        Patched model
    """
    from unified_ra import PrunedKVAttention

    n_head = model.config.n_head
    n_embd = model.config.n_embd
    block_size = model.config.block_size

    ratio_str = "learned" if learn_ratio else f"k={k_keep}"
    print(f"Patching GPT-2 with KV cache pruning ({ratio_str}, recency={recency})...")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # Replace the attention module
        pruned_attn = PrunedKVAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            k_keep=k_keep,
            recency=recency,
            learn_ratio=learn_ratio,
            dropout=dropout,
        )

        # Replace the attention module
        block.attn = pruned_attn

        print(f"  Layer {i}: Standard Attention → KV-Pruned Attention")

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with KV-pruned attention")

    return model
