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
from torch.utils.checkpoint import checkpoint


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
    warmup_loss_drop: float = 0.05
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
# SECTION 3: Two-Way Router
# =============================================================================


class ContextRouter(nn.Module):
    """
    Route tokens to compute tiers based on contextual hardness.

    Maps (x, e_tok, shift_norm) -> 2-way probabilities:
      - p_ra:   Use RA heads only (cheap, fewer heads)
      - p_full: Use FULL heads only (expensive, more heads)

    Features are FLOP-cheap per-token reductions:
      - shift_norm = |x - E(x)|
      - ||x||, ||E(x)||, <x, E(x)>

    The router MLP is tiny: features -> hidden -> 2 logits -> softmax.
    Initial bias discourages FULL to encourage cheap RA path.
    """

    def __init__(self, cfg: RAConfig):
        super().__init__()
        self.eps = 1e-6

        # Feature dimensions: shift_norm + ||x|| + ||E|| + <x,E>
        feat_dim = 4

        # Tiny router MLP: 2-way routing (RA vs FULL)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, cfg.router_hidden),
            nn.GELU(),
            nn.Linear(cfg.router_hidden, 2),
        )

        # Bias FULL logit to discourage expensive compute initially
        with torch.no_grad():
            # Order: [RA, FULL]
            self.mlp[-1].bias.data[1] += cfg.router_bias_full

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
            probs: dict with keys "ra", "full"
                   each value is [B, T, 1] probability tensor
        """
        # Build features
        norm_x = torch.sqrt(x.pow(2).sum(dim=-1) + self.eps)
        norm_e = torch.sqrt(e_tok.pow(2).sum(dim=-1) + self.eps)
        dot = (x * e_tok).sum(dim=-1)

        feats = torch.stack([shift_norm, norm_x, norm_e, dot], dim=-1)  # [B, T, 4]

        # Route
        logits = self.mlp(feats)  # [B, T, 2]
        probs = F.softmax(logits, dim=-1)

        return {
            "ra": probs[..., 0:1],
            "full": probs[..., 1:2],
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
        compute_full: bool = True,
        compute_ra: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D] input
            need_weights: return attention probs for analysis
            compute_full: whether to compute FULL head group output
            compute_ra: whether to compute RA head group output

        Returns:
            out_full: [B, T, D] output from FULL head group (or None)
            out_ra:   [B, T, D] output from RA head group (or None)
            attn_probs: [B, H, T, T] if need_weights else None
        """
        B, T, D = x.shape

        # Determine which heads to compute
        if compute_full and compute_ra:
            head_slice = slice(0, self.n_heads)
            n_compute = self.n_heads
        elif compute_full:
            head_slice = self.full_slice
            n_compute = self.n_full
        elif compute_ra:
            head_slice = self.ra_slice
            n_compute = self.n_ra
        else:
            raise ValueError("Must compute at least one head group")

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, H, T, head_dim]
        def to_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # Select only needed heads for computation
        if not (compute_full and compute_ra):
            q = q[:, head_slice, :, :]
            k = k[:, head_slice, :, :]
            v = v[:, head_slice, :, :]

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

        # Process outputs based on what was computed
        out_full = None
        out_ra = None

        if compute_full and compute_ra:
            # Both computed - split head groups
            full_heads = out[:, self.full_slice, :, :]
            ra_heads = out[:, self.ra_slice, :, :]

            full_merged = (
                full_heads.transpose(1, 2).contiguous().view(B, T, self.d_full)
            )
            ra_merged = ra_heads.transpose(1, 2).contiguous().view(B, T, self.d_ra)

            out_full = self.c_proj_full(full_merged)
            out_ra = self.c_proj_ra(ra_merged)
        elif compute_full:
            # Only FULL heads computed
            full_merged = out.transpose(1, 2).contiguous().view(B, T, self.d_full)
            out_full = self.c_proj_full(full_merged)
        elif compute_ra:
            # Only RA heads computed
            ra_merged = out.transpose(1, 2).contiguous().view(B, T, self.d_ra)
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

    out = p_ra * out_ra + p_full * out_full

    This is research-mode: both RA and FULL are computed. Future optimization
    can use router probabilities to skip head groups entirely.
    """

    def forward(
        self,
        out_ra: torch.Tensor,
        out_full: torch.Tensor,
        probs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            out_ra:   [B, T, D] RA head group output
            out_full: [B, T, D] FULL head group output
            probs:    dict of [B, T, 1] probabilities

        Returns:
            out: [B, T, D] mixed output
        """
        return probs["ra"] * out_ra + probs["full"] * out_full


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
            out = self.mixer(out_ra, out_full, probs)

        return out

    def compute_penalty(
        self,
        x: torch.Tensor,
        e_tok: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cost penalty for training loss.

        Returns mean of p_full to discourage expensive path.
        Only meaningful in phase 2.
        """
        if self.phase1:
            return torch.tensor(0.0, device=x.device)

        x_norm = self.ln(x)
        shift = self.shift_gate(x_norm, e_tok)
        probs = self.router(x_norm, e_tok, shift)

        return probs["full"].mean()


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


# =============================================================================
# SECTION 8: RA-MLA with Token-Latent Cache (TL-cache)
# =============================================================================


@dataclass
class RA_MLA_Config:
    """
    Configuration for RA-MLA (Reciprocal Attention with Multi-head Latent Attention).

    This combines:
    - MLA's latent compression: single latent decompresses to Q, K, V
    - RA's bidirectional attention: alternating Q@K.T and K@Q.T across layers
    - KVSplice compatibility: balanced reciprocal paths for better compression

    Core dimensions
    ---------------
    d_model:      Model embedding dimension
    n_heads:      Number of attention heads
    head_dim:     Dimension per head (d_model // n_heads)
    d_latent:     Latent dimension for TL-cache (compressed representation)
    block_size:   Maximum sequence length
    n_layers:     Total number of layers (for global alternation normalization)

    RoPE settings
    -------------
    rope_theta:   Base for rotary position embeddings (default: 10000.0)

    Other
    -----
    dropout:      Attention dropout probability
    """

    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64
    d_latent: int = 256  # Compressed latent dimension
    block_size: int = 1024
    n_layers: int = 12
    rope_theta: float = 10000.0
    dropout: float = 0.0


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for attention."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for efficiency
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for positions up to seq_len."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to Q and K.

    Args:
        q, k: [B, H, T, D] query and key tensors
        cos, sin: [T, D//2] position embeddings

    Returns:
        Rotated q and k tensors
    """

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Expand for broadcasting: [1, 1, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Need to tile cos/sin to match head_dim
    cos = cos.repeat(1, 1, 1, 2)
    sin = sin.repeat(1, 1, 1, 2)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


class RA_MLA_Flash(nn.Module):
    """
    Reciprocal Attention with Multi-head Latent Attention and Flash compatibility.

    Key innovations:
    1. TL-cache (Token-Latent cache): Single compressed latent decompresses to Q, K, V
    2. Learned alternation: Network learns which layers use standard vs reciprocal
    3. Global normalization: Layer alternation logits sum to 1 via softmax (Markov chain)
    4. Flash attention: Compatible via argument swapping (arg1 @ arg2.T)
    5. RoPE: Position information preserved in attention computation

    The Markov chain reciprocity constraint ensures balanced bidirectional flow:
    - Half the layers (by probability mass) use Q@K.T
    - Half use K@Q.T
    - This creates optimal conditions for KVSplice compression

    Cache efficiency:
    - Standard attention: cache K, V per layer
    - MLA: cache latent per layer (smaller)
    - RA-MLA: same latent works for both directions
    """

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        alternation_logits: nn.Parameter,
    ):
        """
        Args:
            cfg: Configuration dataclass
            layer_idx: Index of this layer (0 to n_layers-1)
            alternation_logits: Shared parameter [n_layers] for global normalization
        """
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.alternation_logits = alternation_logits

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Input projection to latent space
        self.to_latent = nn.Linear(cfg.d_model, cfg.d_latent)

        # Decompress latent to Q, K, V
        # Each head needs head_dim for Q, K, V
        qkv_dim = 3 * cfg.n_heads * cfg.head_dim
        self.from_latent = nn.Linear(cfg.d_latent, qkv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        # Dropout
        self.attn_dropout = nn.Dropout(cfg.dropout)

        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def get_alternation_prob(self) -> torch.Tensor:
        """
        Get this layer's probability of using reciprocal attention.

        Uses sigmoid for independent per-layer decision.
        Balance is enforced via regularization loss in RA_MLA_Model.
        """
        return torch.sigmoid(self.alternation_logits[self.layer_idx])

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with TL-cache and learned alternation.

        Args:
            x: [B, T, D] input tensor
            cache: Optional [B, T_cache, d_latent] cached latents
            use_cache: Whether to return updated cache

        Returns:
            out: [B, T, D] output tensor
            new_cache: Optional [B, T_total, d_latent] updated cache
        """
        B, T, D = x.shape

        # Project to latent space (TL-cache)
        latent = self.to_latent(x)  # [B, T, d_latent]

        # Handle cache
        if cache is not None:
            full_latent = torch.cat([cache, latent], dim=1)
            T_total = full_latent.shape[1]
        else:
            full_latent = latent
            T_total = T

        # Decompress to Q, K, V
        qkv = self.from_latent(full_latent)  # [B, T_total, 3*H*head_dim]
        qkv = qkv.view(B, T_total, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T_total, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # For generation, only use last T queries
        if cache is not None:
            q = q[:, :, -T:, :]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            # Only rotate the new positions for q
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Determine attention direction based on learned alternation
        # During training: use probability for soft decision (Gumbel-softmax style)
        # During inference: use hard decision
        p_recip = self.get_alternation_prob()

        if self.training:
            # Straight-through estimator: hard forward, soft backward
            use_reciprocal = (p_recip > 0.5).float()
            use_reciprocal = use_reciprocal - p_recip.detach() + p_recip
        else:
            use_reciprocal = (p_recip > 0.5).float()

        # Flash attention via argument swapping
        # Standard: Q @ K.T @ V
        # Reciprocal: K @ Q.T @ V (swap Q and K roles)
        if use_reciprocal > 0.5:
            # Reciprocal: K plays role of Q, Q plays role of K
            # For causal attention: we need to be careful about masking
            # K@Q.T gives [B, H, T_k, T_q], we want to attend from K positions to Q
            attn_out = F.scaled_dot_product_attention(
                k[:, :, -T:, :] if cache is not None else k,
                q,
                v,
                is_causal=(cache is None),  # Only causal for non-cached
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        else:
            # Standard: Q @ K.T @ V
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=(cache is None),
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )

        # Merge heads and project output
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, H, head_dim]
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # Return cache if requested
        new_cache = full_latent if use_cache else None

        return out, new_cache


class RA_MLA_Model(nn.Module):
    """
    Container for multiple RA_MLA_Flash layers with shared alternation logits.

    The shared alternation_logits parameter ensures global softmax normalization
    across all layers, creating Markov chain reciprocity where approximately
    half the probability mass goes to standard attention and half to reciprocal.
    """

    def __init__(self, cfg: RA_MLA_Config):
        super().__init__()
        self.cfg = cfg

        # Shared alternation logits for all layers
        # Initialize with alternating pattern: odd layers start as reciprocal
        # This ensures balanced standard/RA attention from the start
        init_logits = torch.zeros(cfg.n_layers)
        for i in range(cfg.n_layers):
            init_logits[i] = (
                1.0 if i % 2 == 1 else -1.0
            )  # sigmoid(1)≈0.73, sigmoid(-1)≈0.27
        self.alternation_logits = nn.Parameter(init_logits)

        # Create layers
        self.layers = nn.ModuleList(
            [RA_MLA_Flash(cfg, i, self.alternation_logits) for i in range(cfg.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward through all layers.

        Args:
            x: [B, T, D] input
            cache: Optional list of [B, T_cache, d_latent] per layer
            use_cache: Whether to return updated caches

        Returns:
            out: [B, T, D] output
            new_caches: Optional list of updated caches
        """
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        return x, new_caches

    def get_alternation_distribution(self) -> torch.Tensor:
        """Get the learned alternation probabilities for all layers."""
        return torch.sigmoid(self.alternation_logits)

    def get_layer_directions(self) -> list:
        """Get which direction each layer uses (for debugging)."""
        probs = self.get_alternation_distribution()
        return ["reciprocal" if p > 0.5 else "standard" for p in probs]

    def balance_loss(self) -> torch.Tensor:
        """
        Regularization loss to ensure balanced standard/reciprocal attention.

        Encourages the sum of reciprocal probabilities to equal n_layers/2,
        creating Markov chain reciprocity for optimal KVSplice compression.

        Returns:
            Scalar loss penalizing deviation from 50/50 balance
        """
        probs = self.get_alternation_distribution()
        target = self.cfg.n_layers / 2.0
        actual = probs.sum()
        return (actual - target) ** 2

    def get_balance_stats(self) -> dict:
        """Get statistics about current alternation balance."""
        probs = self.get_alternation_distribution()
        n_recip = (probs > 0.5).sum().item()
        return {
            "n_reciprocal": n_recip,
            "n_standard": self.cfg.n_layers - n_recip,
            "prob_sum": probs.sum().item(),
            "target_sum": self.cfg.n_layers / 2.0,
        }


# =============================================================================
# SECTION 9: MLA Baseline (without Reciprocal Alternation)
# =============================================================================


class MLA_Flash(nn.Module):
    """
    Multi-head Latent Attention baseline (no reciprocal alternation).

    This is a simpler version of RA_MLA_Flash that always uses standard
    Q@K.T attention. Used for ablation testing to measure the impact of:
    - MLA alone (latent compression)
    - vs RA_MLA (compression + bidirectional flow)

    Features:
    - TL-cache: Single latent decompresses to Q, K, V
    - Flash attention compatible via SDPA
    - RoPE support
    """

    def __init__(self, cfg: RA_MLA_Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Input projection to latent space
        self.to_latent = nn.Linear(cfg.d_model, cfg.d_latent)

        # Decompress latent to Q, K, V
        qkv_dim = 3 * cfg.n_heads * cfg.head_dim
        self.from_latent = nn.Linear(cfg.d_latent, qkv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with TL-cache, always using standard attention."""
        B, T, D = x.shape

        # Project to latent space
        latent = self.to_latent(x)

        # Handle cache
        if cache is not None:
            full_latent = torch.cat([cache, latent], dim=1)
            T_total = full_latent.shape[1]
        else:
            full_latent = latent
            T_total = T

        # Decompress to Q, K, V
        qkv = self.from_latent(full_latent)
        qkv = qkv.view(B, T_total, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if cache is not None:
            q = q[:, :, -T:, :]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Standard attention: Q @ K.T @ V
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(cache is None),
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        new_cache = full_latent if use_cache else None
        return out, new_cache


class MLA_Model(nn.Module):
    """
    Container for MLA_Flash layers (baseline without reciprocal alternation).

    Use for ablation testing:
    - GPT-2 baseline vs MLA: measures latent compression benefit
    - MLA vs RA_MLA: measures reciprocal alternation benefit
    """

    def __init__(self, cfg: RA_MLA_Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([MLA_Flash(cfg, i) for i in range(cfg.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward through all layers."""
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        return x, new_caches


# =============================================================================
# SECTION 10: FIM-Guided KVSplice (Fisher Information Matrix Compression)
# =============================================================================


class FIMKVSplice(nn.Module):
    """
    Fisher Information Matrix guided KV compression.

    Compresses K/V along the temporal dimension using a basis derived from
    the Fisher Information Matrix of attention. This preserves the temporal
    directions that are most important for the attention distribution, as
    defined by the SPDA (Scaled Dot-Product Attention as EOT) result.

    Mathematical basis:
    - F = diag(p) - p @ p.T is the FIM for attention distribution p
    - Eigendecomposition F = U @ Λ @ U.T gives temporal "Fisher modes"
    - Top-r eigenvectors U_r span information-critical temporal directions
    - Compression: C = U_r.T @ K, Reconstruction: K_hat = U_r @ C

    This is more principled than PCA because it preserves information
    structure (what attention cares about) rather than just variance.
    """

    def __init__(
        self,
        max_seq_len: int,
        rank: int,
        n_heads: int = 1,
        per_head: bool = False,
    ):
        """
        Initialize FIM-guided KV compression.

        Args:
            max_seq_len: Maximum sequence length T (calibration T)
            rank: Number of FIM modes to keep (r < T)
            n_heads: Number of attention heads
            per_head: If True, use per-head basis; else shared across heads
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.n_heads = n_heads
        self.per_head = per_head

        # FIM basis: U_r ∈ R^{T×r} for each head (or shared)
        # Initialized as truncated identity (top-r positions)
        if per_head:
            # [H, T, r] - per-head temporal basis
            basis = torch.zeros(n_heads, max_seq_len, rank)
            for h in range(n_heads):
                basis[h, :rank, :] = torch.eye(rank)
            self.register_buffer("fim_basis", basis)
        else:
            # [T, r] - shared basis
            basis = torch.zeros(max_seq_len, rank)
            basis[:rank, :] = torch.eye(rank)
            self.register_buffer("fim_basis", basis)

        self._calibrated = False

    def calibrate(
        self,
        attn_probs: torch.Tensor,
        head_idx: Optional[int] = None,
        n_samples: int = 512,
    ):
        """
        Calibrate FIM basis from attention probabilities.

        Computes the average Fisher Information Matrix over samples and
        extracts top-r eigenvectors as the compression basis.

        Args:
            attn_probs: [N, T, T] or [B, H, T, T] attention probabilities
            head_idx: If per_head, which head to calibrate (None = all)
            n_samples: Max samples to use for FIM estimation
        """
        if attn_probs.dim() == 4:
            # [B, H, T, T] format
            B, H, T, _ = attn_probs.shape
            if self.per_head and head_idx is not None:
                # Calibrate single head
                p = attn_probs[:, head_idx].reshape(B * T, T)
            else:
                # Aggregate all heads
                p = attn_probs.reshape(B * H * T, T)
        else:
            # [N, T, T] format
            N, T, _ = attn_probs.shape
            p = attn_probs.reshape(N * T, T)

        # Subsample for efficiency
        if p.size(0) > n_samples:
            idx = torch.randperm(p.size(0), device=p.device)[:n_samples]
            p = p[idx]
        N_eff = p.size(0)

        # Compute average FIM: F = mean_i(diag(p_i) - p_i @ p_i.T)
        p_mean = p.mean(0)
        F = torch.diag(p_mean) - (p.T @ p) / N_eff

        # Eigendecomposition (eigvalsh returns ascending order)
        eigvals, eigvecs = torch.linalg.eigh(F)

        # Take top-r eigenvectors (largest eigenvalues = descending)
        U_r = eigvecs[:, -self.rank :].flip(-1)  # [T, r]

        # Update basis
        if self.per_head and head_idx is not None:
            self.fim_basis[head_idx] = U_r
        else:
            self.fim_basis.copy_(U_r)

        self._calibrated = True

    def compress(self, k: torch.Tensor, head_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compress K along temporal dimension using FIM basis.

        Args:
            k: [B, T, d_head] or [T, d_head] key tensor
            head_idx: Which head's basis to use (if per_head)

        Returns:
            c: [B, r, d_head] or [r, d_head] compressed representation
        """
        if self.per_head and head_idx is not None:
            U = self.fim_basis[head_idx]  # [T, r]
        else:
            U = self.fim_basis  # [T, r]

        # Handle sequence length mismatch
        T = k.shape[-2]
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} > calibrated max {self.max_seq_len}")
        elif T < self.max_seq_len:
            U = U[:T]  # Truncate basis

        # C = U.T @ K: [r, T] @ [T, d_head] = [r, d_head]
        if k.dim() == 3:
            # [B, T, d_head] -> [B, r, d_head]
            # U.T is [r, T], k is [B, T, d_head]
            c = torch.einsum("rt,btd->brd", U.T, k)
        else:
            # [T, d_head] -> [r, d_head]
            c = U.T @ k

        return c

    def decompress(
        self, c: torch.Tensor, seq_len: int, head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decompress from FIM coefficients back to full K.

        Args:
            c: [B, r, d_head] or [r, d_head] compressed representation
            seq_len: Target sequence length T
            head_idx: Which head's basis to use (if per_head)

        Returns:
            k_hat: [B, T, d_head] or [T, d_head] reconstructed keys
        """
        if self.per_head and head_idx is not None:
            U = self.fim_basis[head_idx]  # [T, r]
        else:
            U = self.fim_basis  # [T, r]

        # Truncate basis if needed
        if seq_len < self.max_seq_len:
            U = U[:seq_len]

        # K_hat = U @ C: [T, r] @ [r, d_head] = [T, d_head]
        if c.dim() == 3:
            # [B, r, d_head] -> [B, T, d_head]
            # U is [T, r], c is [B, r, d_head]
            k_hat = torch.einsum("tr,brd->btd", U, c)
        else:
            # [r, d_head] -> [T, d_head]
            k_hat = U @ c

        return k_hat

    def forward(self, k: torch.Tensor, head_idx: Optional[int] = None) -> torch.Tensor:
        """
        Full compress-decompress cycle (for training with reconstruction loss).

        Args:
            k: [B, T, d_head] key tensor
            head_idx: Which head's basis to use

        Returns:
            k_hat: [B, T, d_head] reconstructed keys
        """
        T = k.shape[-2]
        c = self.compress(k, head_idx)
        return self.decompress(c, T, head_idx)

    def get_reconstruction_error(
        self, k: torch.Tensor, head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Compute reconstruction MSE for monitoring."""
        with torch.no_grad():
            k_hat = self.forward(k, head_idx)
            return F.mse_loss(k_hat, k)

    def get_compression_stats(self) -> dict:
        """Get compression statistics for logging."""
        return {
            "max_seq_len": self.max_seq_len,
            "rank": self.rank,
            "compression_ratio": self.rank / self.max_seq_len,
            "memory_reduction": 1.0 - (self.rank / self.max_seq_len),
            "calibrated": self._calibrated,
            "per_head": self.per_head,
        }


# Legacy LearnedKVSplice for RA_MLA_KVSplice compatibility
# TODO: Update RA_MLA_KVSplice to use FIMKVSplice for temporal compression


class LearnedKVSplice(nn.Module):
    """
    Legacy differentiable KVSplice for end-to-end training.

    Compresses in feature dimension (d_in → d_compressed).
    For the principled FIM-based temporal compression, use FIMKVSplice.
    """

    def __init__(self, d_in: int, d_compressed: int):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed

        # Learned monotonic transform
        self.transform_scale = nn.Parameter(torch.ones(d_in))
        self.transform_shift = nn.Parameter(torch.zeros(d_in))

        # Learned low-rank projection
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

        # Initialize as approximate inverse
        nn.init.orthogonal_(self.compress.weight)
        with torch.no_grad():
            self.expand.weight.copy_(self.compress.weight.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transformed = x * F.softplus(self.transform_scale) + self.transform_shift
        compressed = self.compress(x_transformed)
        decompressed = self.expand(compressed)
        return (decompressed - self.transform_shift) / (
            F.softplus(self.transform_scale) + 1e-6
        )

    def compress_only(self, x: torch.Tensor) -> torch.Tensor:
        x_transformed = x * F.softplus(self.transform_scale) + self.transform_shift
        return self.compress(x_transformed)

    def decompress_only(self, compressed: torch.Tensor) -> torch.Tensor:
        decompressed = self.expand(compressed)
        return (decompressed - self.transform_shift) / (
            F.softplus(self.transform_scale) + 1e-6
        )

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.mse_loss(self.forward(x), x)

    def get_compression_stats(self) -> dict:
        return {
            "d_in": self.d_in,
            "d_compressed": self.d_compressed,
            "compression_ratio": self.d_compressed / self.d_in,
            "memory_reduction": 1.0 - (self.d_compressed / self.d_in),
        }


class RA_MLA_KVSplice(nn.Module):
    """
    RA-MLA with learned KVSplice compression.

    Extends RA_MLA_Flash with end-to-end learned compression that mirrors
    the geometry transformation + PCA approach of KVSplice, but is fully
    differentiable for training.

    The model learns to produce latents that compress well through the
    spline+PCA bottleneck, rather than hoping post-hoc compression works.
    """

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        alternation_logits: nn.Parameter,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.alternation_logits = alternation_logits

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Learned KVSplice compression
        d_compressed = int(cfg.d_latent * compression_ratio)
        self.kvsplice = LearnedKVSplice(cfg.d_latent, d_compressed)
        self.d_compressed = d_compressed

        # Input projection to latent space
        self.to_latent = nn.Linear(cfg.d_model, cfg.d_latent)

        # Decompress latent to Q, K, V
        qkv_dim = 3 * cfg.n_heads * cfg.head_dim
        self.from_latent = nn.Linear(cfg.d_latent, qkv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        # Track reconstruction error for metrics
        self._last_reconstruction_error = None

    def get_alternation_prob(self) -> torch.Tensor:
        """Get this layer's probability of using reciprocal attention."""
        return torch.sigmoid(self.alternation_logits[self.layer_idx])

    def get_kvsplice_metrics(self) -> dict:
        """Get KVSplice metrics for this layer."""
        metrics = self.kvsplice.get_compression_stats()
        if self._last_reconstruction_error is not None:
            metrics["reconstruction_error"] = self._last_reconstruction_error
        return metrics

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with learned KVSplice compression.

        Cache stores compressed latents (d_compressed instead of d_latent).
        """
        B, T, D = x.shape

        # Project to latent space
        latent_orig = self.to_latent(x)  # [B, T, d_latent]

        # Apply KVSplice bottleneck (learn compressible representations)
        latent = self.kvsplice(latent_orig)

        # Track reconstruction error (compute occasionally to avoid overhead)
        if self.training and torch.rand(1).item() < 0.01:  # 1% of steps
            self._last_reconstruction_error = self.kvsplice.get_reconstruction_error(
                latent_orig
            ).item()

        # Handle cache (stored in compressed form)
        if cache is not None:
            # Decompress cached latents
            cache_decompressed = self.kvsplice.decompress_only(cache)
            full_latent = torch.cat([cache_decompressed, latent], dim=1)
            T_total = full_latent.shape[1]
        else:
            full_latent = latent
            T_total = T

        # Decompress to Q, K, V
        qkv = self.from_latent(full_latent)
        qkv = qkv.view(B, T_total, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if cache is not None:
            q = q[:, :, -T:, :]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Determine attention direction
        p_recip = self.get_alternation_prob()

        if self.training:
            use_reciprocal = (p_recip > 0.5).float()
            use_reciprocal = use_reciprocal - p_recip.detach() + p_recip
        else:
            use_reciprocal = (p_recip > 0.5).float()

        # Flash attention
        if use_reciprocal > 0.5:
            attn_out = F.scaled_dot_product_attention(
                k[:, :, -T:, :] if cache is not None else k,
                q,
                v,
                is_causal=(cache is None),
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=(cache is None),
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # Store compressed cache
        if use_cache:
            new_cache = self.kvsplice.compress_only(full_latent)
        else:
            new_cache = None

        return out, new_cache


class RA_MLA_KVSplice_Model(nn.Module):
    """
    Container for RA_MLA_KVSplice layers with learned compression.

    Ablation step 3: measures benefit of end-to-end learned KVSplice
    over base RA_MLA architecture.

    The balanced alternation + learned compression creates optimal
    conditions for inference-time cache efficiency.
    """

    def __init__(self, cfg: RA_MLA_Config, compression_ratio: float = 0.5):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio

        # Shared alternation logits
        init_logits = torch.zeros(cfg.n_layers)
        for i in range(cfg.n_layers):
            init_logits[i] = 1.0 if i % 2 == 1 else -1.0
        self.alternation_logits = nn.Parameter(init_logits)

        # Create layers
        self.layers = nn.ModuleList(
            [
                RA_MLA_KVSplice(cfg, i, self.alternation_logits, compression_ratio)
                for i in range(cfg.n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward through all layers."""
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        return x, new_caches

    def get_alternation_distribution(self) -> torch.Tensor:
        """Get the learned alternation probabilities for all layers."""
        return torch.sigmoid(self.alternation_logits)

    def get_layer_directions(self) -> list:
        """Get which direction each layer uses."""
        probs = self.get_alternation_distribution()
        return ["reciprocal" if p > 0.5 else "standard" for p in probs]

    def balance_loss(self) -> torch.Tensor:
        """Regularization loss for balanced alternation."""
        probs = self.get_alternation_distribution()
        target = self.cfg.n_layers / 2.0
        return (probs.sum() - target) ** 2

    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        return {
            "d_latent": self.cfg.d_latent,
            "d_compressed": d_compressed,
            "compression_ratio": self.compression_ratio,
            "cache_reduction": f"{(1 - self.compression_ratio) * 100:.1f}%",
        }


# =============================================================================
# SECTION 10.5: SBA (Symmetric Bidirectional Attention)
# =============================================================================


class SBA_Flash(nn.Module):
    """
    Symmetric Bidirectional Attention (SBA) / B-EOT Attention.

    Computes BOTH forward (Q→K) and reverse (K→Q) attention in each layer,
    mixing outputs with a learned per-layer alpha:

        attn_out = α * attn_fwd + (1-α) * attn_rev

    This implements dual entropic optimal transport (B-EOT), providing:
    - Dual advantage gradients from both directions
    - Averaged Fisher Information: F = αF_fwd + (1-α)F_rev
    - Smoother optimization landscape
    - Better KVSplice compression due to symmetric geometry

    Mathematical basis: "Scaled Dot-Product Attention as One-Sided Entropic
    Optimal Transport" (August 2025) shows attention solves EOT. SBA solves
    a convex combination of two EOT problems simultaneously.

    Implementation uses single matmul + two softmaxes (row-wise and column-wise)
    for ~1.5× compute cost instead of 2×.

    TODO: Triton kernel fusion for optimal performance (see math-shit/triton.jit)
    """

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        alpha_init: float = 0.5,
        kv_mode: str = "separate",
    ):
        """
        Initialize SBA attention layer.

        Args:
            cfg: Model configuration
            layer_idx: Layer index
            alpha_init: Initial mixing parameter (0.5 = equal mix)
            kv_mode: QKV projection mode
                - "separate": Independent Q, K, V projections (default)
                - "shared_skew": Q = W_shared + W_skew, K = W_shared - W_skew
                - "k_eq_v": K = V (key-value tying)
        """
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.kv_mode = kv_mode

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # TL-cache: project to latent space
        self.to_latent = nn.Linear(cfg.d_model, cfg.d_latent)

        # QKV projections based on mode
        qk_dim = cfg.n_heads * cfg.head_dim
        v_dim = cfg.n_heads * cfg.head_dim

        if kv_mode == "separate":
            # Standard: separate Q, K, V projections
            qkv_dim = 3 * qk_dim
            self.from_latent = nn.Linear(cfg.d_latent, qkv_dim)
        elif kv_mode == "shared_skew":
            # Q = W_shared + W_skew, K = W_shared - W_skew
            self.W_shared = nn.Linear(cfg.d_latent, qk_dim, bias=False)
            self.W_skew = nn.Linear(cfg.d_latent, qk_dim, bias=False)
            self.W_v = nn.Linear(cfg.d_latent, v_dim, bias=False)
        elif kv_mode == "k_eq_v":
            # Q from shared+skew, K = V
            self.W_shared = nn.Linear(cfg.d_latent, qk_dim, bias=False)
            self.W_skew = nn.Linear(cfg.d_latent, qk_dim, bias=False)
            self.W_v = nn.Linear(cfg.d_latent, v_dim, bias=False)
        else:
            raise ValueError(f"Unknown kv_mode: {kv_mode}")

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        # Learned mixing parameter alpha ∈ (0, 1)
        alpha_init_clamped = max(1e-3, min(1 - 1e-3, alpha_init))
        alpha_logit = math.log(alpha_init_clamped / (1 - alpha_init_clamped))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit))

        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        # Track metrics
        self._last_alpha = None

    def get_alpha(self) -> torch.Tensor:
        """Get this layer's mixing parameter."""
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with symmetric bidirectional attention.

        Uses single matmul for scores, then row-wise and column-wise softmax.
        """
        B, T, D = x.shape

        # Project to latent space (TL-cache)
        latent = self.to_latent(x)  # [B, T, d_latent]

        # Handle cache
        if cache is not None:
            full_latent = torch.cat([cache, latent], dim=1)
            T_total = full_latent.shape[1]
        else:
            full_latent = latent
            T_total = T

        # Decompress to Q, K, V based on kv_mode
        if self.kv_mode == "separate":
            qkv = self.from_latent(full_latent)
            qkv = qkv.view(B, T_total, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
            q, k, v = qkv[0], qkv[1], qkv[2]
        elif self.kv_mode == "shared_skew":
            # Q = W_shared + W_skew, K = W_shared - W_skew
            shared = self.W_shared(full_latent)
            skew = self.W_skew(full_latent)
            q_flat = shared + skew
            k_flat = shared - skew
            v_flat = self.W_v(full_latent)
            # Reshape to [B, H, T, D]
            q = q_flat.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_flat.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
            v = v_flat.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
        elif self.kv_mode == "k_eq_v":
            # Q from shared+skew, K = V
            shared = self.W_shared(full_latent)
            skew = self.W_skew(full_latent)
            q_flat = shared + skew
            v_flat = self.W_v(full_latent)
            # Reshape to [B, H, T, D]
            q = q_flat.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
            v = v_flat.view(B, T_total, self.n_heads, self.head_dim).transpose(1, 2)
            k = v  # K = V tying

        if cache is not None:
            q = q[:, :, -T:, :]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Get mixing parameter
        alpha = self.get_alpha()
        self._last_alpha = alpha.item()

        # Compute attention with gradient checkpointing to save memory
        # The full [B, H, T, T] attention matrix is memory-intensive
        T_q = q.shape[2]
        T_k = k.shape[2]

        def _compute_attention(q, k, v, alpha):
            # Compute scores once: Q @ K.T -> [B, H, T_q, T_k]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask for forward direction
            if cache is None:
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, dtype=torch.bool, device=q.device), diagonal=1
                )
                scores_masked = scores.masked_fill(causal_mask, float("-inf"))
            else:
                scores_masked = scores

            # Forward attention: row-wise softmax (Q→K)
            attn_fwd = F.softmax(scores_masked, dim=-1)

            # Reverse attention: column-wise softmax (K→Q)
            attn_rev = F.softmax(scores, dim=-2)

            # Apply dropout
            if self.training and self.cfg.dropout > 0:
                attn_fwd = F.dropout(attn_fwd, p=self.cfg.dropout, training=True)
                attn_rev = F.dropout(attn_rev, p=self.cfg.dropout, training=True)

            # Mix attention patterns
            attn_mixed = alpha * attn_fwd + (1 - alpha) * attn_rev

            # Apply attention to values
            return torch.matmul(attn_mixed, v)

        # Use gradient checkpointing during training to save memory
        if self.training:
            attn_out = checkpoint(
                _compute_attention, q, k, v, alpha, use_reentrant=False
            )
        else:
            attn_out = _compute_attention(q, k, v, alpha)

        # Merge heads and project output
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, H, head_dim]
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # Return cache if requested
        new_cache = full_latent if use_cache else None

        return out, new_cache


class SBA_Model(nn.Module):
    """
    Container for multiple SBA_Flash layers.

    Provides aggregate metrics for monitoring alpha distribution across layers.
    """

    def __init__(self, cfg: RA_MLA_Config, alpha_init: float = 0.5):
        super().__init__()
        self.cfg = cfg

        # Create layers with per-layer alpha
        self.layers = nn.ModuleList(
            [SBA_Flash(cfg, i, alpha_init) for i in range(cfg.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward through all SBA layers."""
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        return x, new_caches

    def get_alpha_distribution(self) -> torch.Tensor:
        """Get alpha values for all layers."""
        return torch.tensor([layer.get_alpha().item() for layer in self.layers])

    def get_layer_types(self) -> list:
        """Classify layers as forward-dominant, reverse-dominant, or mixed."""
        alphas = self.get_alpha_distribution()
        types = []
        for a in alphas:
            if a > 0.7:
                types.append("forward")
            elif a < 0.3:
                types.append("reverse")
            else:
                types.append("mixed")
        return types

    def get_sba_metrics(self) -> dict:
        """Get SBA metrics for W&B logging."""
        alphas = self.get_alpha_distribution()
        metrics = {
            "sba/alpha_mean": alphas.mean().item(),
            "sba/alpha_std": alphas.std().item(),
            "sba/alpha_min": alphas.min().item(),
            "sba/alpha_max": alphas.max().item(),
            "sba/forward_dominant_layers": (alphas > 0.7).sum().item(),
            "sba/reverse_dominant_layers": (alphas < 0.3).sum().item(),
            "sba/mixed_layers": ((alphas >= 0.3) & (alphas <= 0.7)).sum().item(),
        }
        # Per-layer alphas
        for i, a in enumerate(alphas):
            metrics[f"sba/alpha_layer_{i}"] = a.item()
        return metrics


# =============================================================================
# SECTION 11: Full GPT-2 Models with MLA Attention
# =============================================================================


class MLABlock(nn.Module):
    """Transformer block with MLA attention + MLP."""

    def __init__(self, cfg: RA_MLA_Config, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_Flash(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLP: 4x expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        # Attention with residual
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class RAMLABlock(nn.Module):
    """Transformer block with RA_MLA attention + MLP."""

    def __init__(
        self, cfg: RA_MLA_Config, layer_idx: int, alternation_logits: nn.Parameter
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = RA_MLA_Flash(cfg, layer_idx, alternation_logits)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLP: 4x expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class RAMLAKVBlock(nn.Module):
    """Transformer block with RA_MLA_KVSplice attention + MLP."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        alternation_logits: nn.Parameter,
        compression_ratio: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = RA_MLA_KVSplice(
            cfg, layer_idx, alternation_logits, compression_ratio
        )
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLP: 4x expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class MLAGPT(nn.Module):
    """Full GPT-2 model with MLA attention."""

    def __init__(self, cfg: RA_MLA_Config, vocab_size: int = 50257):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([MLABlock(cfg, i) for i in range(cfg.n_layers)])

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """
        Compute Fisher spectrum metrics for selected layers.

        Args:
            x: Input tensor [B, T]
            layer_indices: Which layers to analyze (default: [0, n_layers//2, -1])
            n_samples: Samples per head for eigenvalue computation
            topk: Number of top eigenvalues to log

        Returns:
            Dictionary of Fisher metrics for W&B logging
        """
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass to capture attention probabilities
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h, _ = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "MLA_Flash", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_Flash layer."""
        B, T, _ = x.shape

        # Project to latent
        latent = attn.to_latent(x)

        # Decompress to Q, K, V
        qkv = attn.from_latent(latent)
        qkv = qkv.view(B, T, 3, attn.n_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores_masked = scores.masked_fill(causal_mask, float("-inf"))

        # Standard attention probs
        attn_probs = F.softmax(scores_masked, dim=-1)

        return attn_probs  # [B, H, T, T]


class RAMLAGPT(nn.Module):
    """Full GPT-2 model with RA+MLA attention."""

    def __init__(self, cfg: RA_MLA_Config, vocab_size: int = 50257):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Shared alternation logits
        init_logits = torch.zeros(cfg.n_layers)
        for i in range(cfg.n_layers):
            init_logits[i] = 1.0 if i % 2 == 1 else -1.0
        self.alternation_logits = nn.Parameter(init_logits)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [RAMLABlock(cfg, i, self.alternation_logits) for i in range(cfg.n_layers)]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_alternation_distribution(self):
        return torch.sigmoid(self.alternation_logits)

    def balance_loss(self):
        probs = self.get_alternation_distribution()
        target = self.cfg.n_layers / 2.0
        return (probs.sum() - target) ** 2

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """Compute Fisher spectrum metrics for selected layers."""
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h, _ = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "RA_MLA_Flash", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from RA_MLA_Flash layer."""
        B, T, _ = x.shape

        latent = attn.to_latent(x)

        qkv = attn.from_latent(latent)
        qkv = qkv.view(B, T, 3, attn.n_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores_masked = scores.masked_fill(causal_mask, float("-inf"))

        # Get attention probs based on alternation direction
        p_recip = attn.get_alternation_prob()

        if p_recip > 0.5:
            attn_probs = F.softmax(scores, dim=-2)
        else:
            attn_probs = F.softmax(scores_masked, dim=-1)

        return attn_probs


class RAMLAKV_GPT(nn.Module):
    """Full GPT-2 model with RA+MLA+KVSplice attention."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Shared alternation logits
        init_logits = torch.zeros(cfg.n_layers)
        for i in range(cfg.n_layers):
            init_logits[i] = 1.0 if i % 2 == 1 else -1.0
        self.alternation_logits = nn.Parameter(init_logits)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                RAMLAKVBlock(cfg, i, self.alternation_logits, compression_ratio)
                for i in range(cfg.n_layers)
            ]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_alternation_distribution(self):
        return torch.sigmoid(self.alternation_logits)

    def balance_loss(self):
        probs = self.get_alternation_distribution()
        target = self.cfg.n_layers / 2.0
        return (probs.sum() - target) ** 2

    def get_compression_stats(self):
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        return {
            "d_latent": self.cfg.d_latent,
            "d_compressed": d_compressed,
            "compression_ratio": self.compression_ratio,
            "cache_reduction": f"{(1 - self.compression_ratio) * 100:.1f}%",
        }

    def get_kvsplice_metrics(self) -> dict:
        """
        Get comprehensive KVSplice metrics for logging to W&B.

        Returns metrics for:
        - Overall compression stats
        - Per-layer reconstruction errors
        - Attention pattern statistics
        """
        metrics = {}

        # Overall compression stats
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        metrics["kvsplice/compression_ratio"] = self.compression_ratio
        metrics["kvsplice/d_latent"] = self.cfg.d_latent
        metrics["kvsplice/d_compressed"] = d_compressed
        metrics["kvsplice/memory_reduction_pct"] = (1 - self.compression_ratio) * 100

        # Collect per-layer reconstruction errors
        reconstruction_errors = []
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, "_last_reconstruction_error"):
                error = block.attn._last_reconstruction_error
                if error is not None:
                    reconstruction_errors.append(error)
                    metrics[f"kvsplice/layer_{i}_recon_error"] = error

        # Aggregate reconstruction error
        if reconstruction_errors:
            metrics["kvsplice/avg_reconstruction_error"] = sum(
                reconstruction_errors
            ) / len(reconstruction_errors)
            metrics["kvsplice/max_reconstruction_error"] = max(reconstruction_errors)
            metrics["kvsplice/min_reconstruction_error"] = min(reconstruction_errors)

        # Alternation distribution
        probs = self.get_alternation_distribution()
        metrics["kvsplice/reciprocal_layers"] = (probs > 0.5).sum().item()
        metrics["kvsplice/standard_layers"] = (probs <= 0.5).sum().item()
        metrics["kvsplice/alternation_balance"] = probs.sum().item() / self.cfg.n_layers

        return metrics

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """
        Compute Fisher spectrum metrics for selected layers.

        Runs a forward pass capturing attention probabilities, then computes
        Fisher Information Matrix eigenvalues to measure curvature geometry.

        Args:
            x: Input tensor [B, T]
            layer_indices: Which layers to analyze (default: [0, n_layers//2, -1])
            n_samples: Samples per head for eigenvalue computation
            topk: Number of top eigenvalues to log

        Returns:
            Dictionary of Fisher metrics for W&B logging
        """
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass to capture attention probabilities
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                # Get attention probabilities for this layer
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            # Normal forward through block
            h, _ = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "RA_MLA_KVSplice", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract attention probabilities from RA_MLA_KVSplice layer.

        Computes Q @ K.T and applies softmax based on alternation direction.
        """
        B, T, _ = x.shape

        # Project to latent and through KVSplice
        latent = attn.to_latent(x)
        latent = attn.kvsplice(latent)

        # Decompress to Q, K, V
        qkv = attn.from_latent(latent)
        qkv = qkv.view(B, T, 3, attn.n_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores_masked = scores.masked_fill(causal_mask, float("-inf"))

        # Get attention probs based on alternation direction
        p_recip = attn.get_alternation_prob()

        if p_recip > 0.5:
            # Reciprocal: softmax on dim=-2 (column-wise)
            attn_probs = F.softmax(scores, dim=-2)
        else:
            # Standard: softmax on dim=-1 (row-wise)
            attn_probs = F.softmax(scores_masked, dim=-1)

        return attn_probs  # [B, H, T, T]


class SBABlock(nn.Module):
    """Transformer block with SBA attention + MLP."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        alpha_init: float = 0.5,
        kv_mode: str = "separate",
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = SBA_Flash(cfg, layer_idx, alpha_init, kv_mode)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLP: 4x expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class SBAGPT(nn.Module):
    """Full GPT-2 model with SBA (Symmetric Bidirectional Attention)."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        vocab_size: int = 50257,
        alpha_init: float = 0.5,
        kv_mode: str = "separate",
    ):
        super().__init__()
        self.cfg = cfg
        self.kv_mode = kv_mode

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [SBABlock(cfg, i, alpha_init, kv_mode) for i in range(cfg.n_layers)]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_alpha_distribution(self) -> torch.Tensor:
        """Get alpha values for all layers."""
        return torch.tensor([block.attn.get_alpha().item() for block in self.blocks])

    def get_sba_metrics(self) -> dict:
        """Get SBA metrics for W&B logging."""
        alphas = self.get_alpha_distribution()
        metrics = {
            "sba/alpha_mean": alphas.mean().item(),
            "sba/alpha_std": alphas.std().item(),
            "sba/alpha_min": alphas.min().item(),
            "sba/alpha_max": alphas.max().item(),
            "sba/forward_dominant_layers": (alphas > 0.7).sum().item(),
            "sba/reverse_dominant_layers": (alphas < 0.3).sum().item(),
            "sba/mixed_layers": ((alphas >= 0.3) & (alphas <= 0.7)).sum().item(),
        }
        # Per-layer alphas
        for i, a in enumerate(alphas):
            metrics[f"sba/alpha_layer_{i}"] = a.item()
        return metrics

    def get_kv_mode(self) -> str:
        """Get the KV mode used by this model."""
        return self.kv_mode

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """
        Compute Fisher spectrum metrics for selected layers.

        Runs a forward pass capturing attention probabilities, then computes
        Fisher Information Matrix eigenvalues to measure curvature geometry.

        Args:
            x: Input tensor [B, T]
            layer_indices: Which layers to analyze (default: [0, n_layers//2, -1])
            n_samples: Samples per head for eigenvalue computation
            topk: Number of top eigenvalues to log

        Returns:
            Dictionary of Fisher metrics for W&B logging
        """
        if layer_indices is None:
            # Early, middle, late layers
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass to capture attention probabilities
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                # Get attention probabilities for this layer
                ln_out = block.ln_1(h)
                attn_probs = self._get_attn_probs(block.attn, ln_out)
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            # Normal forward through block
            h, _ = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "SBA_Flash", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract attention probabilities from SBA_Flash layer.

        This does a partial forward pass to get the mixed attention probabilities.
        """
        B, T, _ = x.shape

        # Compress to latent
        latent = attn.to_latent(x)

        # Decompress to Q, K, V
        if attn.kv_mode == "separate":
            qkv = attn.from_latent(latent)
            qkv = qkv.view(B, T, 3, attn.n_heads, attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        elif attn.kv_mode == "shared_skew":
            shared = attn.W_shared(latent)
            skew = attn.W_skew(latent)
            q = (shared + skew).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = (shared - skew).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        elif attn.kv_mode == "k_eq_v":
            shared = attn.W_shared(latent)
            skew = attn.W_skew(latent)
            q = (shared + skew).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
            v = attn.W_v(latent).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = v
        else:
            return None

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores_masked = scores.masked_fill(causal_mask, float("-inf"))

        # Get attention probabilities
        alpha = attn.get_alpha()
        attn_fwd = F.softmax(scores_masked, dim=-1)
        attn_rev = F.softmax(scores, dim=-2)
        attn_mixed = alpha * attn_fwd + (1 - alpha) * attn_rev

        return attn_mixed  # [B, H, T, T]


# =============================================================================
# SECTION 14: Fisher Information Matrix Utilities
# =============================================================================


def compute_fisher_spectrum(
    attn_probs: torch.Tensor, n_samples: int = 64
) -> torch.Tensor:
    """
    Compute Fisher Information Matrix eigenvalues from attention probabilities.

    The SPDA paper shows that the Hessian of log-sum-exp (softmax) equals the
    Fisher Information Matrix: F = diag(p) - p @ p.T

    This captures the curvature of the attention geometry - eigenvalues reveal
    which directions in score space are high-information vs low-information.

    Args:
        attn_probs: [B, T, T] attention probabilities for a single head
        n_samples: Maximum number of query positions to sample

    Returns:
        eigvals: [T] eigenvalues sorted ascending
    """
    B, T, _ = attn_probs.shape
    device = attn_probs.device

    # Flatten to [B*T_q, T_k] - each row is a query's attention distribution
    p = attn_probs.reshape(B * T, T)

    # Subsample for efficiency (O(T^3) eigendecomposition)
    if p.size(0) > n_samples:
        idx = torch.randperm(p.size(0), device=device)[:n_samples]
        p = p[idx]
    N = p.size(0)

    # Compute average F = mean_i (diag(p_i) - p_i @ p_i.T)
    # Vectorized: F = diag(p.mean(0)) - (p.T @ p) / N
    p_mean = p.mean(0)
    F = torch.diag(p_mean) - (p.T @ p) / N

    # Symmetric PSD => use eigvalsh (sorted ascending)
    eigvals = torch.linalg.eigvalsh(F)
    return eigvals


def compute_fisher_metrics(
    attn_probs: torch.Tensor,
    layer_idx: int,
    n_samples: int = 64,
    topk: int = 8,
) -> dict:
    """
    Compute Fisher spectrum metrics for attention probabilities.

    These metrics reveal the information geometry of attention (SPDA paper):
    - eigmax: largest eigenvalue = maximum curvature direction
    - trace: sum of eigenvalues = total Fisher information
    - cond: condition number = ratio of max to min curvature

    Lower eigmax and better conditioning indicate smoother optimization.
    SBA tends to produce smoother curvature due to averaging F_fwd and F_rev.

    Args:
        attn_probs: [B, H, T, T] attention probabilities
        layer_idx: Layer index for metric naming
        n_samples: Samples per head for efficiency
        topk: Number of top eigenvalues to log

    Returns:
        Dictionary of Fisher metrics for W&B logging
    """
    B, H, T, _ = attn_probs.shape
    metrics = {}

    # Aggregate metrics across heads
    eigmax_vals = []
    trace_vals = []
    cond_vals = []
    energy_r8_vals = []
    energy_r16_vals = []
    decay_vals = []

    for h in range(H):
        eigvals = compute_fisher_spectrum(attn_probs[:, h], n_samples=n_samples)
        eigvals = eigvals.detach().cpu()

        eigmax = float(eigvals[-1])
        trace = float(eigvals.sum())
        cond = float(eigvals[-1] / (eigvals[0].abs() + 1e-8))

        eigmax_vals.append(eigmax)
        trace_vals.append(trace)
        cond_vals.append(cond)

        # Per-head metrics
        metrics[f"fisher/layer{layer_idx}/head{h}/eigmax"] = eigmax
        metrics[f"fisher/layer{layer_idx}/head{h}/trace"] = trace
        metrics[f"fisher/layer{layer_idx}/head{h}/cond"] = cond

        # Top-k eigenvalues
        topk_vals = eigvals[-topk:]
        for i, v in enumerate(topk_vals):
            metrics[f"fisher/layer{layer_idx}/head{h}/eig_top{i}"] = float(v)

        # Energy vs rank: E(r) = Σ_{i=T-r+1..T} λ_i / Σ λ_i
        # Useful for picking KVsplice-FIM compression rank
        if trace > 1e-8:
            energy_r8 = float(eigvals[-8:].sum() / trace) if len(eigvals) >= 8 else 1.0
            energy_r16 = (
                float(eigvals[-16:].sum() / trace) if len(eigvals) >= 16 else 1.0
            )
        else:
            energy_r8 = 1.0
            energy_r16 = 1.0
        energy_r8_vals.append(energy_r8)
        energy_r16_vals.append(energy_r16)

        # Spectral decay: eigmax / eig_5th (how concentrated is the spectrum)
        if len(eigvals) >= 5:
            decay = float(eigvals[-1] / (eigvals[-5].abs() + 1e-8))
        else:
            decay = 1.0
        decay_vals.append(decay)

        metrics[f"fisher/layer{layer_idx}/head{h}/energy_r8"] = energy_r8
        metrics[f"fisher/layer{layer_idx}/head{h}/energy_r16"] = energy_r16
        metrics[f"fisher/layer{layer_idx}/head{h}/decay"] = decay

    # Layer aggregates (mean)
    metrics[f"fisher/layer{layer_idx}/eigmax_mean"] = sum(eigmax_vals) / H
    metrics[f"fisher/layer{layer_idx}/trace_mean"] = sum(trace_vals) / H
    metrics[f"fisher/layer{layer_idx}/cond_mean"] = sum(cond_vals) / H
    metrics[f"fisher/layer{layer_idx}/energy_r8_mean"] = sum(energy_r8_vals) / H
    metrics[f"fisher/layer{layer_idx}/energy_r16_mean"] = sum(energy_r16_vals) / H
    metrics[f"fisher/layer{layer_idx}/decay_mean"] = sum(decay_vals) / H

    # Layer aggregates (std) - stability fingerprint for RA vs SBA
    if H > 1:
        import statistics

        metrics[f"fisher/layer{layer_idx}/eigmax_std"] = statistics.stdev(eigmax_vals)
        metrics[f"fisher/layer{layer_idx}/trace_std"] = statistics.stdev(trace_vals)
        metrics[f"fisher/layer{layer_idx}/cond_std"] = statistics.stdev(cond_vals)

    return metrics
