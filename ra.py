#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Reciprocal Attention (RA) with Compute Routing

==============================================================================
INDUCTIVE BIAS: THE NORTH STAR
==============================================================================

This architecture is built on a simple observation: not all tokens need the
same computational effort. Some tokens are "easy" - their meaning is largely
determined by their identity alone. Others are "hard" - context dramatically
reshapes their role.

The key insight: the distance |x - E(x)| between a token's contextual hidden
state x and its static embedding E(x) measures "contextual hardness".

  - Small |x - E(x)|: Token stayed near its embedding. Context didn't change
    its meaning much. Use cheap compute (skip or RA-only).

  - Large |x - E(x)|: Token moved far from its embedding. It's highly
    context-dependent and deserves full attention.

This gives us a FLOP-cheap signal for routing compute without T^2 overhead.

==============================================================================
ARCHITECTURE OVERVIEW
==============================================================================

1. SHARED QKV ATTENTION WITH HEAD GROUPS
   - Single QKV projection for all heads (same as baseline GPT-2)
   - Heads partitioned into FULL group (front) and RA group (tail)
   - RA group uses fewer heads = cheaper projection and merge
   - Both groups computed in single SDPA call (kernel-efficient)

2. CONTEXT ROUTER
   - Computes |x - E(x)| per token (FLOP-cheap: O(BTD), no T^2)
   - Tiny MLP maps features to 4-way probabilities: NONE/RA/FULL/BOTH
   - Compute penalty discourages overuse of expensive paths
   - Router learns when cheap attention suffices

3. WEIGHT TYING WITH EMBEDDINGS
   - RA output projection initialized from attention output projection
   - Creates learned relationship: RA captures compressed view
   - Since output_embeddings = E.T in modern models, RA and E span
     related subspaces, encouraging semantic coherence

4. WARMUP PHASE
   - Phase 1: Full attention only (router OFF, RA OFF)
   - Wait until |x - E(x)| becomes meaningful (~15% loss drop)
   - Phase 2: Enable router and RA heads
   - This ensures router sees useful signals from the start

==============================================================================
DESIGN RATIONALE
==============================================================================

Why fewer heads instead of smaller dimensions?
  - Simpler GPU kernel scheduling (uniform head_dim across all heads)
  - torch.compile sees cleaner static shapes
  - RA's cheapness comes from less output projection work, not weird shapes

Why route at the token level?
  - Different tokens genuinely need different compute
  - Punctuation, common words = easy; rare tokens, operators = hard
  - Per-token routing captures this without per-head complexity

Why |x - E(x)| specifically?
  - E(x) is the "default meaning" - context-free identity
  - x is the contextual representation - what the model thinks it means
  - The gap measures how much context bent the token's role
  - Cheap to compute: just subtraction and norm, no attention needed

Why 4-way routing (NONE/RA/FULL/BOTH)?
  - NONE: Skip token-mixing entirely (just residual)
  - RA: Cheap attention with fewer heads
  - FULL: Standard attention for hard tokens
  - BOTH: "Clutch mode" when both perspectives help
  - Gives model flexibility to learn optimal routing

==============================================================================
KVSPLICE COMPATIBILITY
==============================================================================

This architecture preserves symmetry needed for KV cache compression:
  - RA heads use same QK^T geometry as FULL heads
  - Head groups are cleanly partitioned (no per-head branching)
  - Router decisions don't affect attention pattern structure
  - Future work: use router confidence to inform KV pruning aggressiveness
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Enable TF32 for better GPU utilization
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


# =============================================================================
# SECTION 1: Configuration
# =============================================================================


@dataclass
class RAConfig:
    """
    Configuration for Reciprocal Attention with routing.

    Core dimensions
    ---------------
    d_model:        Model embedding dimension.
    n_heads:        Total number of attention heads.
    block_size:     Maximum sequence length.

    RA head allocation
    ------------------
    ra_head_frac:   Fraction of heads allocated to RA group (0 < frac < 1).
                    Example: 0.25 means 25% of heads are RA heads.
                    Remaining heads are FULL heads.

    Router settings
    ---------------
    router_hidden:  Hidden dimension of router MLP.
    router_bias_full: Initial bias on FULL/BOTH logits (negative = discourage).

    Training schedule
    -----------------
    warmup_loss_drop: Relative loss drop to trigger Phase 2 (default: 0.15).
                      Router and RA are disabled until eval loss drops by this
                      fraction from initial value.

    Weight tying
    ------------
    tie_ra_proj:    If True, initialize RA output projection from FULL projection.
                    Encourages RA to learn compressed representation.

    Other
    -----
    dropout:        Attention dropout probability.
    """

    d_model: int = 768
    n_heads: int = 12
    block_size: int = 1024
    ra_head_frac: float = 0.25
    router_hidden: int = 16
    router_bias_full: float = -1.0
    warmup_loss_drop: float = 0.15
    tie_ra_proj: bool = True
    dropout: float = 0.0


# =============================================================================
# SECTION 2: Context Shift Gate
# =============================================================================


class ContextShiftGate(nn.Module):
    """
    Compute |x - E(x)| as a per-token contextual hardness signal.

    This is the core of our routing inductive bias. The L2 norm of the
    difference between contextual hidden state and static embedding measures
    how much context has reshaped the token's representation.

    Cost: O(BTD) - just subtraction and reduction, no T^2 terms.
    """

    def __init__(self, detach_embedding: bool = True, eps: float = 1e-6):
        super().__init__()
        self.detach_embedding = detach_embedding
        self.eps = eps

    def forward(self, x: torch.Tensor, e_tok: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, T, D] contextual hidden state
            e_tok: [B, T, D] static embeddings for same tokens

        Returns:
            shift_norm: [B, T] L2 norm of (x - e_tok) per token
        """
        if self.detach_embedding:
            e_tok = e_tok.detach()

        delta = x - e_tok
        shift_norm = torch.sqrt(delta.pow(2).sum(dim=-1) + self.eps)
        return shift_norm


# =============================================================================
# SECTION 3: Four-Way Router
# =============================================================================


class ContextRouter(nn.Module):
    """
    Route tokens to compute tiers based on contextual hardness.

    Maps (x, e_tok, shift_norm) -> 4-way probabilities:
      - p_none: Skip token-mixing (just residual)
      - p_ra:   Use RA heads only (cheap)
      - p_full: Use FULL heads only (expensive)
      - p_both: Use both RA and FULL (clutch mode)

    Features are FLOP-cheap per-token reductions:
      - shift_norm = |x - E(x)|
      - ||x||, ||E(x)||, <x, E(x)>

    The router MLP is tiny: features -> hidden -> 4 logits -> softmax.
    Initial bias discourages FULL/BOTH to encourage cheap paths.
    """

    def __init__(self, cfg: RAConfig):
        super().__init__()
        self.eps = 1e-6

        # Feature dimensions: shift_norm + ||x|| + ||E|| + <x,E>
        feat_dim = 4

        # Tiny router MLP
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, cfg.router_hidden),
            nn.GELU(),
            nn.Linear(cfg.router_hidden, 4),
        )

        # Bias FULL and BOTH logits to discourage expensive compute initially
        with torch.no_grad():
            # Order: [NONE, RA, FULL, BOTH]
            self.mlp[-1].bias.data[2] += cfg.router_bias_full
            self.mlp[-1].bias.data[3] += cfg.router_bias_full

    def forward(
        self,
        x: torch.Tensor,
        e_tok: torch.Tensor,
        shift_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          [B, T, D] contextual hidden state
            e_tok:      [B, T, D] static embeddings (usually detached)
            shift_norm: [B, T] from ContextShiftGate

        Returns:
            probs: dict with keys "none", "ra", "full", "both"
                   each value is [B, T, 1] probability tensor
        """
        # Build features
        norm_x = torch.sqrt(x.pow(2).sum(dim=-1) + self.eps)
        norm_e = torch.sqrt(e_tok.pow(2).sum(dim=-1) + self.eps)
        dot = (x * e_tok).sum(dim=-1)

        feats = torch.stack([shift_norm, norm_x, norm_e, dot], dim=-1)  # [B, T, 4]

        # Route
        logits = self.mlp(feats)  # [B, T, 4]
        probs = F.softmax(logits, dim=-1)

        return {
            "none": probs[..., 0:1],
            "ra": probs[..., 1:2],
            "full": probs[..., 2:3],
            "both": probs[..., 3:4],
        }


# =============================================================================
# SECTION 4: Shared QKV Attention with Head Groups
# =============================================================================


class RAAttention(nn.Module):
    """
    Shared QKV attention with FULL and RA head groups.

    Structure:
      - Single QKV projection (same as baseline GPT-2)
      - Heads split into FULL group (indices 0:n_full) and RA group (n_full:)
      - Single SDPA call computes attention for all heads
      - Separate output projections for each group

    Why this design:
      - Kernel-efficient: one attention computation, uniform head_dim
      - RA cheapness comes from fewer heads = smaller output projection
      - Clean partitioning enables future KV cache optimizations

    Returns both out_full and out_ra for router to mix downstream.
    """

    def __init__(self, cfg: RAConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # Compute head group sizes
        n_ra = max(1, int(round(cfg.ra_head_frac * cfg.n_heads)))
        n_ra = min(n_ra, cfg.n_heads - 1)  # At least 1 FULL head
        n_full = cfg.n_heads - n_ra

        self.n_ra = n_ra
        self.n_full = n_full

        # Head slices
        self.full_slice = slice(0, n_full)
        self.ra_slice = slice(n_full, n_full + n_ra)

        # Projection dimensions
        self.d_full = n_full * self.head_dim
        self.d_ra = n_ra * self.head_dim

        # QKV projection (same as baseline)
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)

        # Output projections for each group
        self.c_proj_full = nn.Linear(self.d_full, cfg.d_model, bias=False)
        self.c_proj_ra = nn.Linear(self.d_ra, cfg.d_model, bias=False)

        # Optional weight tying: init RA proj from FULL proj
        if cfg.tie_ra_proj:
            with torch.no_grad():
                # Copy tail columns of full projection to RA
                # This encourages RA to be a compressed view
                full_w = self.c_proj_full.weight  # [d_model, d_full]
                ra_w = self.c_proj_ra.weight  # [d_model, d_ra]
                ra_w.copy_(full_w[:, -self.d_ra :])

        self.dropout_p = cfg.dropout

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D] input
            need_weights: return attention probs for analysis

        Returns:
            out_full: [B, T, D] output from FULL head group
            out_ra:   [B, T, D] output from RA head group
            attn_probs: [B, H, T, T] if need_weights else None
        """
        B, T, D = x.shape

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, H, T, head_dim]
        def to_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # Use SDPA for efficient attention
        if need_weights:
            # Manual path for attention weights (debugging only)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(mask, float("-inf"))
            attn_probs = F.softmax(scores, dim=-1)
            if self.dropout_p > 0 and self.training:
                attn_probs = F.dropout(attn_probs, p=self.dropout_p)
            out = torch.matmul(attn_probs, v)
        else:
            # Fast SDPA path
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
            attn_probs = None

        # Split head groups
        full_heads = out[:, self.full_slice, :, :]  # [B, n_full, T, head_dim]
        ra_heads = out[:, self.ra_slice, :, :]  # [B, n_ra, T, head_dim]

        # Merge and project
        full_merged = full_heads.transpose(1, 2).contiguous().view(B, T, self.d_full)
        ra_merged = ra_heads.transpose(1, 2).contiguous().view(B, T, self.d_ra)

        out_full = self.c_proj_full(full_merged)
        out_ra = self.c_proj_ra(ra_merged)

        if need_weights:
            return out_full, out_ra, attn_probs
        return out_full, out_ra, None


# =============================================================================
# SECTION 5: Routed Mixing
# =============================================================================


class RoutedMixer(nn.Module):
    """
    Mix outputs from compute tiers according to router probabilities.

    out = p_none * x + p_ra * out_ra + p_full * out_full + p_both * (out_ra + out_full)/2

    This is research-mode: both RA and FULL are computed. Future optimization
    can use router probabilities to skip head groups entirely.
    """

    def forward(
        self,
        x: torch.Tensor,
        out_ra: torch.Tensor,
        out_full: torch.Tensor,
        probs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, T, D] input (residual)
            out_ra:   [B, T, D] RA head group output
            out_full: [B, T, D] FULL head group output
            probs:    dict of [B, T, 1] probabilities

        Returns:
            out: [B, T, D] mixed output
        """
        out_both = 0.5 * (out_ra + out_full)

        return (
            probs["none"] * x
            + probs["ra"] * out_ra
            + probs["full"] * out_full
            + probs["both"] * out_both
        )


# =============================================================================
# SECTION 6: RA Block (Complete Sublayer)
# =============================================================================


class RABlock(nn.Module):
    """
    Complete RA attention sublayer with optional routing.

    Supports two modes:
      - Baseline mode (phase1=True or router disabled): Use all heads as FULL
      - RA mode (phase1=False): Route between FULL and RA based on |x - E(x)|

    Same code path for both modes ensures fair comparison.
    """

    def __init__(self, cfg: RAConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        # Core attention
        self.attn = RAAttention(cfg)

        # Routing components
        self.shift_gate = ContextShiftGate()
        self.router = ContextRouter(cfg)
        self.mixer = RoutedMixer()

        # Layer norm (pre-norm architecture)
        self.ln = nn.LayerNorm(cfg.d_model)

        # Phase tracking
        self.phase1 = True  # Start in warmup

    def set_phase(self, phase1: bool):
        """Switch between warmup (phase1=True) and routing (phase1=False)."""
        self.phase1 = phase1

    def forward(
        self,
        x: torch.Tensor,
        e_tok: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:     [B, T, D] input hidden state
            e_tok: [B, T, D] static embeddings (required for phase 2)

        Returns:
            out: [B, T, D] attention sublayer output (before residual add)
        """
        # Pre-norm
        x_norm = self.ln(x)

        # Get both head group outputs
        out_full, out_ra, _ = self.attn(x_norm)

        if self.phase1:
            # Warmup: use only FULL heads
            # Combine outputs with ratio matching head allocation
            # This ensures smooth transition to phase 2
            alpha = self.attn.n_ra / self.attn.n_heads
            out = (1 - alpha) * out_full + alpha * out_ra
        else:
            # Phase 2: route based on contextual hardness
            if e_tok is None:
                raise ValueError("e_tok required for phase 2 routing")

            shift = self.shift_gate(x_norm, e_tok)
            probs = self.router(x_norm, e_tok, shift)
            out = self.mixer(x_norm, out_ra, out_full, probs)

        return out

    def compute_penalty(
        self,
        x: torch.Tensor,
        e_tok: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cost penalty for training loss.

        Returns mean of (p_full + p_both) to discourage expensive paths.
        Only meaningful in phase 2.
        """
        if self.phase1:
            return torch.tensor(0.0, device=x.device)

        x_norm = self.ln(x)
        shift = self.shift_gate(x_norm, e_tok)
        probs = self.router(x_norm, e_tok, shift)

        cost = probs["full"] + probs["both"]
        return cost.mean()


# =============================================================================
# SECTION 7: KVSplice-Compatible Attention (Preserved for Future Work)
# =============================================================================


class KVSpliceAttention(nn.Module):
    """
    Attention with KV cache pruning support.

    This preserves the KVSplice idea: use attention patterns to inform
    intelligent KV cache compression. The symmetry in our RA design
    (same QK^T geometry for all heads) enables future integration.

    Key insight: router confidence can inform pruning aggressiveness.
    High-confidence easy tokens can have more aggressive KV pruning.

    TODO: Integrate router probabilities with KV pruning decisions.
    """

    def __init__(
        self,
        cfg: RAConfig,
        prune_ratio: float = 0.5,
        recency_window: int = 64,
    ):
        super().__init__()
        self.cfg = cfg
        self.prune_ratio = prune_ratio
        self.recency_window = recency_window

        assert cfg.d_model % cfg.n_heads == 0
        self.head_dim = cfg.d_model // cfg.n_heads

        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input
            importance_scores: [B, T] optional per-token importance for pruning

        Returns:
            out: [B, T, D] attention output
        """
        B, T, D = x.shape
        H = self.cfg.n_heads

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t):
            return t.view(B, T, H, self.head_dim).transpose(1, 2)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # Compute attention with optional KV pruning
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))

        # KV pruning: keep top-k keys per query
        if self.prune_ratio < 1.0 and T > self.recency_window:
            k_keep = max(
                self.recency_window,
                int(T * self.prune_ratio),
            )

            # Always keep recent tokens
            recent_mask = torch.zeros(T, T, dtype=torch.bool, device=x.device)
            recent_mask[:, -self.recency_window :] = True

            # Select top-k by score for non-recent positions
            _, topk_idx = scores.topk(k_keep, dim=-1)
            topk_mask = torch.zeros_like(scores, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)

            # Combine masks
            keep_mask = recent_mask | topk_mask
            scores = scores.masked_fill(~keep_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(out)


# =============================================================================
# SECTION 8: Warmup Scheduler
# =============================================================================


class WarmupScheduler:
    """
    Track eval loss and trigger phase transition.

    Phase 1 -> Phase 2 when:
      1. At least min_evals completed
      2. Relative loss drop >= threshold

    Usage:
        scheduler = WarmupScheduler(threshold=0.15)

        # At each eval
        if scheduler.should_transition(eval_loss):
            for block in model.blocks:
                block.set_phase(phase1=False)
    """

    def __init__(
        self,
        threshold: float = 0.15,
        min_evals: int = 2,
    ):
        self.threshold = threshold
        self.min_evals = min_evals
        self.initial_loss = None
        self.eval_count = 0
        self.transitioned = False

    def should_transition(self, eval_loss: float) -> bool:
        """
        Check if we should transition to phase 2.

        Args:
            eval_loss: Current evaluation loss

        Returns:
            True if this call triggers the transition
        """
        if self.transitioned:
            return False

        if self.initial_loss is None:
            self.initial_loss = eval_loss
            self.eval_count = 1
            return False

        self.eval_count += 1

        if self.eval_count < self.min_evals:
            return False

        rel_drop = (self.initial_loss - eval_loss) / self.initial_loss

        if rel_drop >= self.threshold:
            self.transitioned = True
            return True

        return False

    def get_progress(self, eval_loss: float) -> float:
        """Get progress toward transition (0 to 1)."""
        if self.initial_loss is None or self.initial_loss == 0:
            return 0.0
        rel_drop = (self.initial_loss - eval_loss) / self.initial_loss
        return min(1.0, rel_drop / self.threshold)
