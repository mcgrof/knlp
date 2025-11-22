"""
RA Patching for GPT-2

Replaces standard attention with RABlock (shared QKV with head groups + routing).

The key principle: partition heads into FULL and RA groups, use a router based
on |x - E(x)| to decide compute tier per token. Same code path for baseline
(phase1=True) and RA-routed (phase1=False) ensures fair comparison.

Usage:
    from gpt2.ra_patch import patch_gpt2_with_ra
    model, scheduler = patch_gpt2_with_ra(model, ra_head_frac=0.25)
"""

import sys
import os

# Add parent directory to path to import ra
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from ra import RAConfig, RABlock, WarmupScheduler


class RATransformerBlock(nn.Module):
    """
    Wrapper that replaces GPT-2 block's attention with RABlock.

    Preserves the original MLP and residual structure.
    """

    def __init__(self, original_block, ra_block, embedding_layer):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.attn = ra_block
        self.ln_2 = original_block.ln_2
        self.mlp = original_block.mlp
        self.embedding = embedding_layer

    def forward(self, x, input_ids=None):
        """
        Args:
            x: [B, T, D] hidden state
            input_ids: [B, T] token ids for embedding lookup

        Returns:
            out: [B, T, D] output hidden state
        """
        # Get embeddings for routing
        e_tok = None
        if input_ids is not None:
            e_tok = self.embedding(input_ids)

        # Attention sublayer with RA routing
        attn_out = self.attn(x, e_tok=e_tok)
        x = x + attn_out

        # MLP sublayer (standard)
        x = x + self.mlp(self.ln_2(x))

        return x


def patch_gpt2_with_ra(
    model,
    ra_head_frac=0.25,
    router_hidden=16,
    router_bias_full=-1.0,
    warmup_loss_drop=0.15,
    tie_ra_proj=True,
    dropout=0.0,
    enable_routing=True,
):
    """
    Replace all attention modules in GPT-2 with RABlock.

    Args:
        model: GPT-2 model to patch
        ra_head_frac: Fraction of heads for RA group (0 < frac < 1)
        router_hidden: Router MLP hidden dimension
        router_bias_full: Initial bias on FULL/BOTH logits (negative = discourage)
        warmup_loss_drop: Relative loss drop to trigger phase 2
        tie_ra_proj: Initialize RA projection from FULL projection
        dropout: Attention dropout probability
        enable_routing: If False, keep all blocks in phase1 (baseline mode)

    Returns:
        (model, scheduler): Patched model and WarmupScheduler instance
    """
    n_head = model.config.n_head
    n_embd = model.config.n_embd
    block_size = model.config.block_size

    # Create config
    cfg = RAConfig(
        d_model=n_embd,
        n_heads=n_head,
        block_size=block_size,
        ra_head_frac=ra_head_frac,
        router_hidden=router_hidden,
        router_bias_full=router_bias_full,
        warmup_loss_drop=warmup_loss_drop,
        tie_ra_proj=tie_ra_proj,
        dropout=dropout,
    )

    # Get embedding layer
    embedding_layer = model.transformer.wte

    # Compute head group sizes for logging
    n_ra = max(1, int(round(ra_head_frac * n_head)))
    n_ra = min(n_ra, n_head - 1)
    n_full = n_head - n_ra

    mode_str = "RA with routing" if enable_routing else "Baseline (all FULL)"
    print(f"Patching GPT-2 with {mode_str}...")
    print(f"  Heads: {n_full} FULL + {n_ra} RA = {n_head} total")
    print(f"  Router: hidden={router_hidden}, bias_full={router_bias_full}")
    print(f"  Warmup: phase2 after {warmup_loss_drop*100:.0f}% loss drop")

    # Iterate through all transformer blocks
    for i, block in enumerate(model.transformer.h):
        # Create RABlock
        ra_block = RABlock(cfg, layer_idx=i)

        # Move to same device as original
        device = next(block.attn.parameters()).device
        ra_block = ra_block.to(device)

        # If baseline mode, keep in phase1 permanently
        if not enable_routing:
            ra_block.phase1 = True

        # Wrap the block
        wrapped = RATransformerBlock(block, ra_block, embedding_layer)
        wrapped = wrapped.to(device)

        model.transformer.h[i] = wrapped

        print(f"  Layer {i}: → RABlock (FULL={n_full}, RA={n_ra})")

    num_layers = len(model.transformer.h)
    print(f"Patched {num_layers} layers")

    # Create scheduler
    scheduler = WarmupScheduler(
        threshold=warmup_loss_drop,
        min_evals=2,
    )

    return model, scheduler


def set_ra_phase(model, phase1):
    """
    Set phase for all RABlocks in model.

    Args:
        model: Patched GPT-2 model
        phase1: True for warmup (no routing), False for routing enabled
    """
    phase_str = "Phase 1 (warmup)" if phase1 else "Phase 2 (routing)"
    print(f"Setting all layers to {phase_str}")

    for block in model.transformer.h:
        if hasattr(block, "attn") and hasattr(block.attn, "set_phase"):
            block.attn.set_phase(phase1)


def get_router_stats(model):
    """
    Get router distribution statistics from model.

    Returns:
        dict with mean probabilities for each compute tier
    """
    # This would require running a forward pass with tracking
    # For now, return placeholder
    return {
        "p_none": 0.0,
        "p_ra": 0.0,
        "p_full": 0.0,
        "p_both": 0.0,
    }


def compute_total_penalty(model, x, input_ids):
    """
    Compute total compute penalty across all layers.

    Args:
        model: Patched GPT-2 model
        x: [B, T, D] hidden states
        input_ids: [B, T] token ids

    Returns:
        penalty: Scalar tensor
    """
    e_tok = model.transformer.wte(input_ids)
    total_penalty = torch.tensor(0.0, device=x.device)

    for block in model.transformer.h:
        if hasattr(block, "attn") and hasattr(block.attn, "compute_penalty"):
            total_penalty = total_penalty + block.attn.compute_penalty(x, e_tok)

    return total_penalty / len(model.transformer.h)
