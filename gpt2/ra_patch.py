"""
RA Patching for GPT-2

Replaces standard attention with ReciprocalAttention (direct layout emission).
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
    from ra_patch import patch_gpt2_with_ra_v5
    model = patch_gpt2_with_ra_v5(model, R=4, dropout=0.1)
"""

import sys
import os

# Add parent directory to path to import ra
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from ra import ReciprocalAttention, ReciprocalMLP


def patch_gpt2_with_ra_v5(
    model, R=4, dropout=0.1, use_self_restart=False, per_head_gates=False
):
    """
    Replace all attention modules in GPT-2 with RA.

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
    print(f"Patching GPT-2 with RA (R={R}, {gate_str} gates){restart_str}...")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # Replace the attention module with RA
        original_attn = block.attn

        # Create RA module
        unified_ra_attn = ReciprocalAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            R=R,
            dropout=dropout,
            use_self_restart=use_self_restart,
            per_head_gates=per_head_gates,
        )

        # Move to same device as original module
        device = next(original_attn.parameters()).device
        unified_ra_attn = unified_ra_attn.to(device)

        # Copy over any bias if it exists
        if hasattr(original_attn, "bias") and hasattr(unified_ra_attn, "bias"):
            unified_ra_attn.bias = original_attn.bias

        # Replace the attention module
        block.attn = unified_ra_attn

        print(f"  Layer {i}: Standard Attention → RA{restart_str}")

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with RA{restart_str}")

    return model


def patch_gpt2_with_rmlp(
    model,
    expansion=4,
    R_ff=1152,
    dropout=0.0,
    attn_scale_init=1.0,
    tie_to_attn_proj=False,
    skip_rec_init=-3.0,
):
    """
    Replace all MLP modules in GPT-2 with Reciprocal MLP and wrap blocks
    to enable attention injection.

    Args:
        model: GPT-2 model to patch
        expansion: MLP expansion ratio (default 4)
        R_ff: Reciprocal rank for MLP (default 1152, golden ratio)
        dropout: Dropout probability
        attn_scale_init: Initial attention mixing scale α
        tie_to_attn_proj: Tie up_low weights to attention c_proj
        skip_rec_init: Initial value for the reciprocal skip gate logit.

    Returns:
        Patched model
    """
    from ra import AttentionAwareMLP_Block

    n_embd = model.config.n_embd

    print(f"Patching GPT-2 with R-MLP (R_ff={R_ff}, expansion={expansion})...")
    if tie_to_attn_proj:
        print("  - Strong weight tying enabled (up_low ↔ attn.c_proj)")
    print(f"  - Attention injection enabled (α_init={attn_scale_init})")
    print(f"  - Learned geometric gates: w_std, w_rec")
    print(f"  - Reciprocal skip gate init: {skip_rec_init}")

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
            attn_scale_init=attn_scale_init,
            tie_to_attn_proj=tie_to_attn_proj,
            skip_rec_init=skip_rec_init,
        )

        # Move to same device as original module
        device = next(original_mlp.parameters()).device
        reciprocal_mlp = reciprocal_mlp.to(device)

        # Set up weight tying reference if requested
        if tie_to_attn_proj:
            # Link to attention output projection
            reciprocal_mlp._attn_proj_ref = block.attn.c_proj

        # Replace the MLP module
        block.mlp = reciprocal_mlp

        print(f"  Layer {i}: Standard MLP → R-MLP")

    # Wrap all blocks to enable attention-aware MLP forward
    print("Wrapping blocks to enable attention injection...")
    for i in range(len(model.transformer.h)):
        original_block = model.transformer.h[i]
        wrapped_block = AttentionAwareMLP_Block(original_block)

        # Move to same device
        device = next(original_block.parameters()).device
        wrapped_block = wrapped_block.to(device)

        model.transformer.h[i] = wrapped_block

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with R-MLP + attention injection")

    return model


def patch_gpt2_with_unified_ra_and_rmlp(
    model,
    R=4,
    attn_dropout=0.1,
    use_self_restart=False,
    mlp_expansion=4,
    R_ff=1152,
    mlp_dropout=0.0,
    attn_scale_init=1.0,
    tie_to_attn_proj=False,
    per_head_gates=False,
    skip_rec_init=-3.0,
):
    """
    Replace both attention and MLP modules in GPT-2 with RA + R-MLP.

    Args:
        model: GPT-2 model to patch
        R: Reciprocal rank for attention (default 4)
        attn_dropout: Attention dropout probability
        use_self_restart: Enable self-restart for attention
        mlp_expansion: MLP expansion ratio (default 4)
        R_ff: Reciprocal rank for MLP (default 1152, golden ratio)
        mlp_dropout: MLP dropout probability
        attn_scale_init: Initial attention mixing scale α
        tie_to_attn_proj: Tie up_low weights to attention c_proj
        per_head_gates: Use per-head gates for RA (default False=per-layer)
        skip_rec_init: Initial value for the R-MLP reciprocal skip gate logit.

    Returns:
        Patched model
    """
    # First patch attention with RA
    model = patch_gpt2_with_ra_v5(
        model,
        R=R,
        dropout=attn_dropout,
        use_self_restart=use_self_restart,
        per_head_gates=per_head_gates,
    )

    # Then patch MLP with R-MLP (includes attention injection)
    model = patch_gpt2_with_rmlp(
        model,
        expansion=mlp_expansion,
        R_ff=R_ff,
        dropout=mlp_dropout,
        attn_scale_init=attn_scale_init,
        tie_to_attn_proj=tie_to_attn_proj,
        skip_rec_init=skip_rec_init,
    )

    return model


def analyze_ra_v5_gates(model):
    """
    Analyze learned gate values across all RA layers.

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

    for block in model.transformer.h:
        if hasattr(block.mlp, "w_std") and hasattr(block.mlp, "w_rec"):
            w_std_values.append(block.mlp.w_std.detach().cpu().item())
            w_rec_values.append(block.mlp.w_rec.detach().cpu().item())

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
    from ra import PrunedKVAttention

    n_head = model.config.n_head
    n_embd = model.config.n_embd
    block_size = model.config.block_size

    ratio_str = "learned" if learn_ratio else f"k={k_keep}"
    print(f"Patching GPT-2 with KV cache pruning ({ratio_str}, recency={recency})...")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # Get device from original attention module
        original_attn = block.attn
        device = next(original_attn.parameters()).device

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

        # Move to same device as original module
        pruned_attn = pruned_attn.to(device)

        # Replace the attention module
        block.attn = pruned_attn

        print(f"  Layer {i}: Standard Attention → KV-Pruned Attention")

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with KV-pruned attention")

    return model


def patch_gpt2_with_gate_informed_kv_pruning(
    model,
    k_keep=391,
    recency=64,
    dropout=0.1,
    prune_mode="v_only",
    beta=1.0,
    mlp_expansion=4,
    R_ff=1152,
    mlp_dropout=0.0,
    attn_scale_init=1.0,
    tie_to_attn_proj=False,
):
    """
    Replace attention and MLP with gate-informed KV pruning + R-MLP.

    This creates a feedback loop:
    - R-MLP learns to compensate for attention via reciprocal pathway
    - R-MLP gates (w_rec, α) indicate attention confidence
    - KV pruning reads gates and modulates pruning aggressiveness
    - High confidence → prune more, low confidence → prune less

    Args:
        model: GPT-2 model to patch
        k_keep: Base number of tokens to keep (before modulation)
        recency: Number of recent tokens to always keep
        dropout: Attention dropout probability
        prune_mode: "v_only", "kv_scores_reuse", or "legacy"
        beta: Gate modulation strength (higher = stronger effect)
        mlp_expansion: MLP expansion ratio (default 4)
        R_ff: Reciprocal rank for MLP (default 1152)
        mlp_dropout: MLP dropout probability
        attn_scale_init: Initial attention mixing scale α
        tie_to_attn_proj: Tie up_low weights to attention c_proj

    Returns:
        Patched model
    """
    from ra import GateInformedKVAttention, ReciprocalMLP, AttentionAwareMLP_Block

    n_embd = model.config.n_embd
    n_head = model.config.n_head
    block_size = model.config.block_size

    print(f"Patching GPT-2 with gate-informed KV pruning + R-MLP (beta={beta})...")
    print(f"  - Base k_keep={k_keep}, recency={recency}, mode={prune_mode}")
    print(f"  - R-MLP: R_ff={R_ff}, expansion={mlp_expansion}")
    print(f"  - Adaptive pruning based on w_rec × α (attention confidence)")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # 1. Replace attention with gate-informed KV pruning
        gate_attn = GateInformedKVAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            k_keep=k_keep,
            recency=recency,
            dropout=dropout,
            prune_mode=prune_mode,
            beta=beta,
        )

        # Move to same device
        device = next(block.attn.parameters()).device
        gate_attn = gate_attn.to(device)

        # Replace attention
        block.attn = gate_attn

        # 2. Replace MLP with R-MLP
        reciprocal_mlp = ReciprocalMLP(
            n_embd=n_embd,
            expansion=mlp_expansion,
            R_ff=R_ff,
            dropout=mlp_dropout,
            attn_scale_init=attn_scale_init,
            tie_to_attn_proj=tie_to_attn_proj,
        )

        reciprocal_mlp = reciprocal_mlp.to(device)

        # Set up weight tying if requested
        if tie_to_attn_proj:
            reciprocal_mlp._attn_proj_ref = gate_attn.c_proj

        # Replace MLP
        block.mlp = reciprocal_mlp

        # 3. Link attention to MLP for gate reading
        gate_attn.set_mlp_reference(reciprocal_mlp)

        print(f"  Layer {i}: Gate-informed KV pruning + R-MLP (linked)")

    # 4. Wrap all blocks to enable attention injection
    print("Wrapping blocks to enable attention injection...")
    for i in range(len(model.transformer.h)):
        original_block = model.transformer.h[i]
        wrapped_block = AttentionAwareMLP_Block(original_block)

        device = next(original_block.parameters()).device
        wrapped_block = wrapped_block.to(device)

        model.transformer.h[i] = wrapped_block

    num_layers = len(model.transformer.h)
    print(f"Successfully patched {num_layers} layers with gate-informed pruning")

    return model
