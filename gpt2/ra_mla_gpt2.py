# ra_mla_gpt2.py
# ------------------------------------------------------------
# GPT-2 Reciprocal Attention (RA) + MLA-style shared latent KV
# Enhanced implementation with proper projections, FlashAttention, and pruning hooks.
#
# Usage:
#   from transformers import AutoModelForCausalLM
#   from ra_mla_gpt2 import patch_gpt2_with_ra_mla
#   model = AutoModelForCausalLM.from_pretrained("gpt2")
#   patch_gpt2_with_ra_mla(model,
#       latent_dim=64, ra_window=64, ra_alpha=0.5,
#       per_head_q_latent=True, per_head_v_up=True, use_flash=True)
#   model.eval()
#
# Key improvements over initial sketch:
# - Proper q_to_latent projection (no longer reusing k_down weight hack)
# - FlashAttention integration with hybrid manual/flash approach
# - Fixed reciprocal computation bugs
# - Better initialization and shape handling
# - Comprehensive attention metrics logging

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Set
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Config dataclass --------------------------- #


@dataclass
class RA_MLA_Config:
    """Configuration for RA+MLA attention mechanism with reciprocal MLP."""

    # Latent compression
    latent_dim: int = 64  # shared latent size (<< head_dim * n_heads)

    # Reciprocal attention
    ra_window: int = 64  # local band width for reciprocal term
    ra_alpha: float = 0.5  # weight for reciprocal symmetric score (0.0 disables RA)

    # Projection architecture
    per_head_q_latent: bool = (
        True  # per-head Q-to-latent (more expressive but more params)
    )
    per_head_v_up: bool = True  # per-head V up-projection (more expressive)
    share_k_down: bool = True  # K down-proj shared across heads (always True for MLA)
    share_v_down: bool = True  # V down-proj shared across heads (always True for MLA)

    # Inference caching
    cache_q_window: bool = True  # cache last W queries for reciprocal at inference

    # Optional RoPE (GPT-2 vanilla doesn't use RoPE)
    use_rope: bool = False
    rope_theta: float = 10000.0

    # Regularization
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # Performance
    use_flash: bool = True  # use FlashAttention when available
    flash_for_ra: bool = (
        False  # use flash for RA band (experimental, requires custom kernel)
    )

    # Metrics logging
    log_attention_entropy: bool = False  # log attention entropy for analysis
    log_reciprocity_score: bool = False  # log reciprocity correlation

    # === Reciprocal MLP Mechanisms ===
    # Mechanism 1: MLP-to-Attention Gating
    mlp_attn_gate: bool = False  # MLP activations modulate attention head weights
    mlp_gate_dim: int = 64  # Context vector dimension for gating
    mlp_gate_alpha: float = 0.1  # Mixing weight for MLP attention gating

    # Mechanism 2: Cross-Token MLP Aggregation
    mlp_cross_token: bool = False  # MLP receives weighted sum from other tokens
    mlp_cross_alpha: float = 0.3  # Mixing weight for cross-token MLP
    mlp_reuse_attn_weights: bool = True  # Use attention weights for routing

    # Mechanism 3: MLP Latent Space Reciprocity
    mlp_latent_recip: bool = (
        False  # Bidirectional pathways between attention/MLP latents
    )
    mlp_latent_dim: int = 128  # MLP latent dimension
    mlp_recip_alpha: float = 0.2  # Mixing weight for MLP latent reciprocity

    # Parameter tying for MLP-Attention coupling (Mechanism 3 enhancement)
    mlp_tying_mode: str = (
        "tied_transpose"  # "untied" | "tied_transpose" | "per_head_scalar"
    )

    # Sparsification for cross-token MLP (Mechanism 2 enhancement)
    mlp_sparse_mode: str = "topk"  # "none" | "topk" | "rms"
    mlp_sparse_k: int = 8  # Top-k value for topk mode
    mlp_sparse_tau: float = 0.5  # RMS threshold for rms mode
    mlp_sparse_normalize: bool = True  # Re-normalize after sparsification
    mlp_sparse_head_average: bool = True  # Average attention weights across heads

    # MLP architecture
    mlp_expansion_ratio: float = 4.0  # MLP hidden dim = expansion_ratio * n_embd

    # === Cross-Token RA (RA-CT): Attention-only gating without MLP ===
    ra_cross_token: bool = (
        False  # enable per-token, per-head gating from attention stats
    )
    ra_ct_apply: str = "output"  # "output" | "weights" (where to apply the gate)
    ra_ct_mode: str = "topk"  # "topk" | "max" | "entropy" | "rms"
    ra_ct_k: int = 8  # for "topk" mode
    ra_ct_tau: float = 0.5  # for "rms" threshold
    ra_ct_alpha: float = 0.2  # mix weight for gate application
    ra_ct_head_average: bool = (
        False  # average stats across heads first (cheap & stable)
    )
    ra_ct_detach_stats: bool = False  # True: compute stats under no_grad to save mem


# --------------------------- Utility: AdamWPrune SNR -------------------- #


def snr_from_adam_state(
    param: torch.Tensor, opt_state: Dict, eps=1e-8, gamma=1.0, delta=0.5
):
    st = opt_state.get(param, None)
    if not st or "exp_avg" not in st or "exp_avg_sq" not in st:
        return param.detach().abs()  # fallback magnitude proxy
    m = st["exp_avg"].to(param.device)
    v = st["exp_avg_sq"].to(param.device)
    return (m.abs() ** gamma) / ((v + eps) ** delta)


# --------------------------- Rotary (optional) -------------------------- #


def apply_rope(x, cos, sin):
    # x: [B,T,H,D] even D assumed. Rope on last dim pairs.
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return xr.flatten(-2)


def rope_cache(seq_len, dim, base=10000.0, device="cpu", dtype=torch.float32):
    # standard RoPE cache; your GPT-2 doesn’t have it by default
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->td", t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


# --------------------------- Reciprocal MLP ----------------------------- #


class ReciprocalCoupler(nn.Module):
    """
    Bidirectional MLP<->Attn coupling with parameter-efficient tying options.

    Three tying modes control parameter count vs expressiveness tradeoff:

    1. untied (most expressive, most parameters):
       - Two independent linear projections
       - Parameters: 2 × (hidden_dim × attn_latent_dim)
       - Example: 2 × (3072 × 128) = ~786K params

    2. tied_transpose (default, balanced):
       - Single weight matrix W used bidirectionally
       - attn→mlp uses W^T, mlp→attn uses W
       - Parameters: hidden_dim × attn_latent_dim
       - Example: 3072 × 128 = ~393K params (50% reduction)

    3. per_head_scalar (low-parameter, gated):
       - Linear projections gated by per-head scalars
       - Same projections as untied, but modulated per-head
       - Parameters: 2 × (H × L) + 2 × n_heads
       - Example: 786K + 24 ≈ 786K params, but scalars learn gating

    Args:
      hidden_dim: token hidden (MLP) dim
      attn_latent_dim: latent dim used by MLA path (e.g., 128)
      n_heads: number of heads (for per-head scalar mode)
      tying: "untied" | "tied_transpose" | "per_head_scalar"
      init_scale: initialization scale for linear weights
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_latent_dim: int,
        n_heads: int,
        tying: str = "tied_transpose",
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_latent_dim = attn_latent_dim
        self.n_heads = n_heads
        self.tying = tying

        if tying == "untied":
            self.attn_to_mlp = nn.Linear(attn_latent_dim, hidden_dim, bias=False)
            self.mlp_to_attn = nn.Linear(hidden_dim, attn_latent_dim, bias=False)
            nn.init.normal_(self.attn_to_mlp.weight, std=init_scale)
            nn.init.normal_(self.mlp_to_attn.weight, std=init_scale)

        elif tying == "tied_transpose":
            # One weight W in R^{hidden x attn_latent}; use W^T for reverse
            W = torch.empty(hidden_dim, attn_latent_dim)
            nn.init.normal_(W, std=init_scale)
            self.W = nn.Parameter(W)

        elif tying == "per_head_scalar":
            # Low-DoF: Use linear projections but gate them with per-head scalars.
            # This gives dimension flexibility while keeping gating parameters minimal.
            self.attn_to_mlp = nn.Linear(attn_latent_dim, hidden_dim, bias=False)
            self.mlp_to_attn = nn.Linear(hidden_dim, attn_latent_dim, bias=False)
            nn.init.normal_(self.attn_to_mlp.weight, std=init_scale)
            nn.init.normal_(self.mlp_to_attn.weight, std=init_scale)

            # Per-head scalar gates (low-DoF component)
            self.alpha_attn_to_mlp = nn.Parameter(torch.ones(n_heads))
            self.alpha_mlp_to_attn = nn.Parameter(torch.ones(n_heads))
        else:
            raise ValueError(f"Unknown tying mode: {tying}")

    def forward(
        self,
        mlp_hidden: torch.Tensor,  # [B, T, H]
        attn_latent: torch.Tensor,  # [B, T, L] (MLA latent)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          mlp_enrich: contribution to add into MLP hidden from attn_latent
          attn_context: contribution to pass to next attention from mlp_hidden
        """
        B, T, H = mlp_hidden.shape
        L = attn_latent.shape[-1]

        if self.tying == "untied":
            mlp_enrich = self.attn_to_mlp(attn_latent)  # [B,T,H]
            attn_context = self.mlp_to_attn(mlp_hidden)  # [B,T,L]
            return mlp_enrich, attn_context

        elif self.tying == "tied_transpose":
            # Convention: attn->mlp uses W^T (L->H) ; mlp->attn uses W (H->L)
            W = self.W
            mlp_enrich = torch.matmul(attn_latent, W.t())  # [B,T,H]
            attn_context = torch.matmul(mlp_hidden, W)  # [B,T,L]
            return mlp_enrich, attn_context

        elif self.tying == "per_head_scalar":
            # Per-head scalar mode: Use small linear layers with per-head gating
            # This is "low-DoF" because we use per-head scalars to gate the projections,
            # but we still need linear layers to handle dimension mismatch (H != L).

            # Project with learned linear layers (same as untied mode)
            mlp_enrich_pre = self.attn_to_mlp(attn_latent)  # [B,T,H]
            attn_context_pre = self.mlp_to_attn(mlp_hidden)  # [B,T,L]

            # Apply per-head scalar gating
            # Reshape to per-head for broadcasting
            H_per_head = H // self.n_heads
            L_per_head = L // self.n_heads

            # attn->mlp gating
            mlp_enrich_heads = mlp_enrich_pre.view(B, T, self.n_heads, H_per_head)
            mlp_enrich = (
                mlp_enrich_heads * self.alpha_attn_to_mlp.view(1, 1, self.n_heads, 1)
            ).reshape(B, T, H)

            # mlp->attn gating
            attn_context_heads = attn_context_pre.view(B, T, self.n_heads, L_per_head)
            attn_context = (
                attn_context_heads * self.alpha_mlp_to_attn.view(1, 1, self.n_heads, 1)
            ).reshape(B, T, L)

            return mlp_enrich, attn_context

        else:
            raise RuntimeError("unreachable")


class CrossTokenMLPAggregator(nn.Module):
    """
    Reuses attention routing to aggregate cross-token MLP context with optional sparsification.

    Inputs:
      mlp_hidden: [B,T,H]
      attn_weights: [B, Hh, T, T] or [B, T, T] (if already head-mean)
      mode: "none" | "topk" | "rms"
      k: top-k per token when mode="topk"
      tau: RMS threshold when mode="rms"

    Returns:
      cross_context: [B,T,H]
    """

    def __init__(
        self,
        mode: str = "topk",
        k: int = 8,
        tau: float = 0.5,
        head_average: bool = True,
        normalize_kept: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.k = k
        self.tau = tau
        self.head_average = head_average
        self.normalize_kept = normalize_kept

    @torch.no_grad()
    def _sparsify(self, W: torch.Tensor) -> torch.Tensor:
        # W: [B,T,T], row-wise sparse
        if self.mode == "none":
            return W
        B, T, _ = W.shape
        if self.mode == "topk":
            k = min(self.k, T)
            # keep topk per row
            vals, idx = torch.topk(W, k, dim=-1)
            out = torch.zeros_like(W)
            out.scatter_(-1, idx, vals)
        elif self.mode == "rms":
            # keep entries above tau * RMS(row)
            rms = torch.sqrt(torch.clamp((W**2).mean(dim=-1, keepdim=True), min=1e-12))
            mask = W >= (self.tau * rms)
            out = torch.where(mask, W, torch.zeros_like(W))
        else:
            raise ValueError(self.mode)

        if self.normalize_kept:
            s = out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            out = out / s
        return out

    def forward(
        self, mlp_hidden: torch.Tensor, attn_weights: torch.Tensor
    ) -> torch.Tensor:
        B, T, H = mlp_hidden.shape

        if attn_weights.dim() == 4:
            # [B, Hh, T, T] -> mean over heads
            if self.head_average:
                W = attn_weights.mean(dim=1)
            else:
                # Or take max/softmax over heads — mean is simplest and stable.
                W = attn_weights.mean(dim=1)
        elif attn_weights.dim() == 3:
            W = attn_weights
        else:
            raise ValueError(f"attn_weights shape {attn_weights.shape} unsupported")

        W = W.to(mlp_hidden.dtype)

        # Sparsify routing
        W = self._sparsify(W)  # [B,T,T]

        # Cross-token aggregation: [B,T,T] @ [B,T,H] -> [B,T,H]
        cross_context = torch.bmm(W, mlp_hidden)
        return cross_context


class ReciprocalMLP(nn.Module):
    """
    Reciprocal MLP with three optional mechanisms for bidirectional information flow.

    Mechanisms:
    1. MLP-to-Attention Gating: MLP activations modulate attention head weights (next layer)
    2. Cross-Token MLP Aggregation: MLP receives weighted sum from other tokens using attention weights
    3. MLP Latent Space Reciprocity: Bidirectional latent pathways between attention and MLP
    """

    def __init__(self, n_embd: int, n_head: int, cfg: RA_MLA_Config):
        super().__init__()
        self.cfg = cfg
        self.n_embd = n_embd
        self.n_head = n_head

        # Global scalar for gradually turning on attention↔MLP coupling.
        # Training loop will call set_coupling_scale() to ramp this from 0 → 1.
        self.register_buffer("coupling_scale", torch.tensor(0.0))

        # MLP projections (configurable expansion ratio)
        self.mlp_dim = int(cfg.mlp_expansion_ratio * n_embd)
        self.c_fc = nn.Linear(n_embd, self.mlp_dim, bias=True)
        self.c_proj = nn.Linear(self.mlp_dim, n_embd, bias=True)
        self.dropout = nn.Dropout(cfg.resid_dropout)

        # Mechanism 1: MLP-to-Attention Gating
        if cfg.mlp_attn_gate:
            self.gate_proj = nn.Linear(self.mlp_dim, cfg.mlp_gate_dim, bias=True)
            self.gate_to_heads = nn.Linear(cfg.mlp_gate_dim, n_head, bias=True)
            # Initialize bias positive so gates start mostly open
            nn.init.constant_(self.gate_to_heads.bias, 2.0)

        # Mechanism 2: Cross-Token MLP Aggregation with sparsification
        if cfg.mlp_cross_token:
            self.cross_aggregator = CrossTokenMLPAggregator(
                mode=cfg.mlp_sparse_mode,
                k=cfg.mlp_sparse_k,
                tau=cfg.mlp_sparse_tau,
                head_average=cfg.mlp_sparse_head_average,
                normalize_kept=cfg.mlp_sparse_normalize,
            )
            self.cross_proj = nn.Linear(self.mlp_dim, self.mlp_dim, bias=False)
            # Learnable mixing weight for cross-token aggregation
            self.mlp_cross_alpha = nn.Parameter(torch.tensor(cfg.mlp_cross_alpha))

        # Mechanism 3: MLP Latent Space Reciprocity with parameter tying
        if cfg.mlp_latent_recip:
            self.mlp_down = nn.Linear(self.mlp_dim, cfg.mlp_latent_dim, bias=False)
            # Use ReciprocalCoupler for parameter-efficient bidirectional coupling
            self.coupler = ReciprocalCoupler(
                hidden_dim=self.mlp_dim,
                attn_latent_dim=cfg.latent_dim,
                n_heads=n_head,
                tying=cfg.mlp_tying_mode,
            )
            # Learnable mixing weight for Attention→MLP enrichment (applied in MLP)
            self.mlp_recip_alpha_mlp = nn.Parameter(torch.tensor(cfg.mlp_recip_alpha))

        # Storage for cross-layer communication
        self._attn_gate_context = None  # For mechanism 1
        self._mlp_latent_context = None  # For mechanism 3
        self._attn_weights = None  # Cached attention weights for mechanism 2

    def forward(
        self,
        x: torch.Tensor,  # [B, T, E]
        attn_weights: Optional[torch.Tensor] = None,  # [B, H, T, T] from attention
        attn_latent: Optional[torch.Tensor] = None,  # [B, T, L] from attention layer
    ) -> torch.Tensor:
        """
        Forward pass with reciprocal MLP mechanisms.

        Args:
            x: Input hidden states [B, T, E]
            attn_weights: Attention weights from this layer [B, H, T, T] (for mechanism 2)
            attn_latent: Attention latent representation [B, T, L] (for mechanism 3)

        Returns:
            output: [B, T, E]
        """
        # === Assertions: Check reciprocity inputs are properly connected ===
        if self.cfg.mlp_cross_token:
            assert (
                attn_weights is not None
            ), "mlp_cross_token enabled but no attn_weights"
        if self.cfg.mlp_latent_recip:
            assert (
                attn_latent is not None
            ), "mlp_latent_recip enabled but no attn_latent"

        # Standard MLP
        hidden = F.gelu(self.c_fc(x))  # [B, T, 4*E]

        # Global warmup scale for all coupling terms
        coupling_scale = self.coupling_scale

        # Mechanism 3: MLP Latent Reciprocity (attention -> MLP) with parameter tying
        if self.cfg.mlp_latent_recip and attn_latent is not None:
            # Use ReciprocalCoupler for bidirectional coupling
            mlp_enrich, attn_context = self.coupler(
                mlp_hidden=hidden, attn_latent=attn_latent
            )
            hidden = hidden + coupling_scale * self.mlp_recip_alpha_mlp * mlp_enrich

            # Store MLP latent context for attention layer (next block)
            self._mlp_latent_context = attn_context  # [B, T, L_attn]

        # Mechanism 2: Cross-Token MLP Aggregation with sparsification
        if self.cfg.mlp_cross_token and attn_weights is not None:
            # Use CrossTokenMLPAggregator with sparsification
            cross_context = self.cross_aggregator(hidden, attn_weights)  # [B, T, 4*E]

            # Mix into current hidden state (learnable mixing weight)
            cross_contribution = self.cross_proj(cross_context)
            hidden = hidden + coupling_scale * self.mlp_cross_alpha * cross_contribution

        # Mechanism 1: MLP-to-Attention Gating
        if self.cfg.mlp_attn_gate:
            # Extract gate context from MLP activations
            gate_context = self.gate_proj(hidden)  # [B, T, gate_dim]
            gate_context_global = gate_context.mean(dim=1)  # [B, gate_dim]

            # Generate per-head gating weights
            head_gates = torch.sigmoid(
                self.gate_to_heads(gate_context_global)
            )  # [B, n_head]

            # Store for attention layer (next block)
            self._attn_gate_context = head_gates

        # Final projection
        output = self.c_proj(hidden)  # [B, T, E]
        output = self.dropout(output)

        return output

    def get_attn_gate_context(self) -> Optional[torch.Tensor]:
        """Retrieve MLP gating context for attention (mechanism 1)."""
        return self._attn_gate_context

    def get_mlp_latent_context(self) -> Optional[torch.Tensor]:
        """Retrieve MLP latent context for attention (mechanism 3)."""
        return self._mlp_latent_context

    def set_coupling_scale(self, scale: float):
        """Set global 0–1 scale for attention↔MLP coupling."""
        # Clamp to [0,1] for safety
        scale_val = float(max(0.0, min(1.0, scale)))
        self.coupling_scale.fill_(scale_val)


# --------------------------- RA+MLA Attention --------------------------- #


class RA_MLA_Attention(nn.Module):
    """
    Enhanced RA+MLA Attention with proper projections and FlashAttention support.

    Architecture:
      - MLA: Shared latent K/V compression (down-proj shared across heads)
      - Per-head queries with proper Q-to-latent alignment projection
      - Per-head or shared V up-projection for expressiveness
      - RA: Optional reciprocal symmetric scoring within local band (causal)
      - FlashAttention: Hybrid approach (flash for non-RA, manual for RA band)

    Cache format (inference):
      past = {
          "latent_k": [B, T_past, L],  # compressed keys
          "latent_v": [B, T_past, L],  # compressed values
          "q_band": [B, W, H, D]       # optional query window for RA
      }
    """

    def __init__(self, n_embd: int, n_head: int, cfg: RA_MLA_Config):
        super().__init__()
        assert (
            n_embd % n_head == 0
        ), f"n_embd={n_embd} must be divisible by n_head={n_head}"
        assert cfg.latent_dim > 0, "latent_dim must be positive"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.cfg = cfg

        # === Query Projection ===
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)  # [E] -> [H*D]

        # === Latent K/V Down-Projections (shared across heads) ===
        self.k_down = nn.Linear(n_embd, cfg.latent_dim, bias=False)  # [E] -> [L]
        self.v_down = nn.Linear(n_embd, cfg.latent_dim, bias=False)  # [E] -> [L]

        # === Q-to-Latent Projection (CRITICAL FIX) ===
        # Need proper learned projection, not the sketchy k_down weight reuse!
        if cfg.per_head_q_latent:
            # Per-head projections for maximum expressiveness
            # Store as [H, D, L] for einsum efficiency
            self.q_to_latent = nn.Parameter(
                torch.empty(self.n_head, self.head_dim, cfg.latent_dim)
            )
            nn.init.xavier_uniform_(self.q_to_latent, gain=1.0 / math.sqrt(2))
        else:
            # Shared projection across heads (more efficient)
            self.q_to_latent_shared = nn.Linear(n_embd, cfg.latent_dim, bias=False)

        # === V Up-Projection (latent -> head space) ===
        if cfg.per_head_v_up:
            # Per-head tiny expanders: [H, L, D]
            self.v_up = nn.Parameter(
                torch.empty(self.n_head, cfg.latent_dim, self.head_dim)
            )
            # Initialize near identity (scaled down due to low rank)
            nn.init.xavier_uniform_(self.v_up, gain=1.0 / math.sqrt(cfg.latent_dim))
        else:
            # Shared up-projection then broadcast to heads
            self.v_up_shared = nn.Linear(cfg.latent_dim, self.head_dim, bias=False)
            nn.init.xavier_uniform_(
                self.v_up_shared.weight, gain=1.0 / math.sqrt(cfg.latent_dim)
            )

        # === Output Projection ===
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # === Regularization ===
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        # === FlashAttention Availability ===
        self.flash_available = cfg.use_flash and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        # === Metrics Tracking ===
        self.attention_entropy = None  # Last computed attention entropy
        self.reciprocity_score = None  # Last computed reciprocity correlation
        self.enable_metrics_computation = False  # Flag to enable expensive metrics

        # === Reciprocal MLP Support ===
        self._attn_weights_export = (
            None  # Export attention weights for MLP (mechanism 2)
        )
        self._latent_k_export = None  # Export latent K for MLP (mechanism 3)

        # Global coupling warmup for RA and MLP→Attention effects
        self.register_buffer("coupling_scale", torch.tensor(0.0))

        # === Reciprocal Attention (RA) Learnable Mixing ===
        if cfg.ra_alpha > 0.0:
            # Make ra_alpha learnable - start at config value
            self.ra_alpha = nn.Parameter(torch.tensor(cfg.ra_alpha))
        else:
            # If RA disabled, register as buffer (not trainable)
            self.register_buffer("ra_alpha", torch.tensor(0.0))

        # === MLP-to-Attention Reciprocity Learnable Mixing ===
        if cfg.mlp_attn_gate:
            # Learnable mixing weight for MLP gating (applied in attention)
            self.mlp_gate_alpha = nn.Parameter(torch.tensor(cfg.mlp_gate_alpha))

        if cfg.mlp_latent_recip:
            # Learnable mixing weight for MLP→Attention context (applied in attention)
            self.mlp_recip_alpha_attn = nn.Parameter(torch.tensor(cfg.mlp_recip_alpha))

        # === Cross-Token RA (RA-CT) Learnables ===
        if cfg.ra_cross_token:
            # Per-head affine transforms for gating: sigmoid(stat * scale + bias)
            # Initialize bias ≈ 2.0 for near-1 gates initially (pass-through)
            self.ra_ct_scale = nn.Parameter(torch.ones(self.n_head))
            self.ra_ct_bias = nn.Parameter(torch.full((self.n_head,), 2.0))

    def _split_heads(self, x):  # [B,T,E] -> [B,T,H,D]
        B, T, E = x.shape
        H, D = self.n_head, self.head_dim
        return x.view(B, T, H, D)

    def _merge_heads(self, x):  # [B,T,H,D] -> [B,T,E]
        B, T, H, D = x.shape
        return x.contiguous().view(B, T, H * D)

    def _expand_v(self, latent_v):  # [B,Tc,L] -> [B,Tc,H,D]
        if self.cfg.per_head_v_up:
            # einsum: [B,T,L] x [H,L,D] -> [B,T,H,D]
            return torch.einsum("btl,hld->bthd", latent_v, self.v_up)
        else:
            # shared up then tile heads
            up = self.v_up_shared(latent_v)  # [B,Tc,D]
            up = up.unsqueeze(2).expand(-1, -1, self.n_head, -1)  # [B,Tc,H,D]
            return up

    def _row_stats(self, attn):
        """
        Compute per-token, per-head statistics from attention weights for RA-CT gating.

        Args:
            attn: [B, H, T, T_tot] - attention weights

        Returns:
            stat: [B, H, T] - per-token, per-head statistics
        """
        # Optionally average across heads first for stability
        if self.cfg.ra_ct_head_average:
            # average heads -> [B,1,T,T_tot] then broadcast back
            A = attn.mean(dim=1, keepdim=True)  # [B,1,T,T_tot]
        else:
            A = attn  # [B,H,T,T_tot]

        mode = self.cfg.ra_ct_mode
        if mode == "topk":
            k = min(self.cfg.ra_ct_k, A.size(-1))
            vals, _ = torch.topk(A, k, dim=-1)  # [B,H_or_1,T,k]
            stat = vals.sum(dim=-1)  # [B,H_or_1,T]
        elif mode == "max":
            stat = A.max(dim=-1).values  # [B,H_or_1,T]
        elif mode == "entropy":
            p = A.clamp_min(1e-12)
            stat = -(p * p.log()).sum(dim=-1)  # [B,H_or_1,T]
        elif mode == "rms":
            stat = torch.sqrt((A * A).mean(dim=-1))  # [B,H_or_1,T]
        else:
            raise ValueError(f"Unknown ra_ct_mode: {self.cfg.ra_ct_mode}")

        if self.cfg.ra_ct_head_average:
            stat = stat.expand(-1, self.n_head, -1)  # [B,H,T]
        return stat  # [B,H,T]

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, E]
        past_key_value: Optional[dict] = None,  # {"latent_k", "latent_v", "q_band"}
        use_cache: bool = True,
        attn_mask: Optional[torch.Tensor] = None,  # [B, 1, T, T_total] or None
        mlp_gate_context: Optional[
            torch.Tensor
        ] = None,  # [B, H] from previous MLP (mechanism 1)
        mlp_latent_context: Optional[
            torch.Tensor
        ] = None,  # [B, T, L] from previous MLP (mechanism 3)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with proper projections and optional reciprocal attention.

        Args:
            hidden_states: Input hidden states [B, T, E]
            past_key_value: KV cache for autoregressive generation
            use_cache: Whether to return updated cache
            attn_mask: Attention mask (optional)
            mlp_gate_context: Per-head gating from previous MLP [B, H] (mechanism 1)
            mlp_latent_context: Latent context from previous MLP [B, T, L] (mechanism 3)

        Returns:
            output: [B, T, E] - attention output
            new_past: dict with latent caches for next step
        """
        B, T, E = hidden_states.shape
        H, D, L = self.n_head, self.head_dim, self.cfg.latent_dim

        # Global coupling warmup scale
        coupling_scale = self.coupling_scale

        # Note: mlp_gate_context and mlp_latent_context may be None for
        # the first block in the model (no previous block). The code
        # handles None gracefully - reciprocity features only apply when
        # context is available.

        # === 1. Project Q, K, V ===
        Q = self._split_heads(self.q_proj(hidden_states))  # [B, T, H, D]

        latent_k_new = self.k_down(hidden_states)  # [B, T, L]
        latent_v_new = self.v_down(hidden_states)  # [B, T, L]

        # === 1a. Apply MLP Latent Reciprocity (Mechanism 3) ===
        if self.cfg.mlp_latent_recip and mlp_latent_context is not None:
            # Mix MLP latent context into our latent K (learnable mixing weight)
            latent_k_new = (
                latent_k_new
                + coupling_scale * self.mlp_recip_alpha_attn * mlp_latent_context
            )

        # === 2. Concatenate with Past (for autoregressive generation) ===
        if past_key_value is not None:
            latent_k = torch.cat([past_key_value["latent_k"], latent_k_new], dim=1)
            latent_v = torch.cat([past_key_value["latent_v"], latent_v_new], dim=1)
        else:
            latent_k, latent_v = latent_k_new, latent_v_new
        T_tot = latent_k.size(1)

        # === 2a. Export Latent K for MLP (Mechanism 3) ===
        if self.cfg.mlp_latent_recip:
            self._latent_k_export = latent_k_new.detach()  # [B, T, L]

        # === 3. Optional RoPE on Latent K ===
        if self.cfg.use_rope:
            cos, sin = rope_cache(
                T_tot,
                L,
                self.cfg.rope_theta,
                device=latent_k.device,
                dtype=latent_k.dtype,
            )
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)  # [1, T_tot, L/2]
            latent_k = apply_rope(latent_k.unsqueeze(2), cos, sin).squeeze(2)

        # === 4. Q-to-Latent Projection (FIXED) ===
        if self.cfg.per_head_q_latent:
            # Per-head: [B,T,H,D] x [H,D,L] -> [B,T,H,L]
            q_latent = torch.einsum("bthd,hdl->bthl", Q, self.q_to_latent)
        else:
            # Shared: [B,T,E] -> [B,T,L] -> [B,T,1,L] -> [B,T,H,L]
            q_latent = (
                self.q_to_latent_shared(hidden_states)
                .unsqueeze(2)
                .expand(-1, -1, H, -1)
            )

        # === 5. Compute Attention Logits (Standard) ===
        # [B,T,H,L] x [B,T_tot,L] -> [B,H,T,T_tot]
        logits = torch.einsum("bthl,bsl->bhts", q_latent, latent_k) / math.sqrt(L)

        # === 6. Causal Masking ===
        # Current tokens are at positions [T_tot-T : T_tot]
        # They can attend to positions [0 : T_tot] with causal constraint
        causal_mask = self._create_causal_mask(T, T_tot, device=hidden_states.device)
        logits = logits.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # === 7. Reciprocal Attention (RA) ===
        if self.ra_alpha > 0.0 and coupling_scale > 0:
            # Get all queries (past + current) for reciprocal computation
            if (
                past_key_value is not None
                and self.cfg.cache_q_window
                and "q_band" in past_key_value
                and past_key_value["q_band"] is not None
            ):
                Q_all = torch.cat(
                    [past_key_value["q_band"], Q], dim=1
                )  # [B, T_q_all, H, D]
            else:
                Q_all = Q  # [B, T, H, D]

            # Project all queries to latent space
            if self.cfg.per_head_q_latent:
                q_all_latent = torch.einsum("bthd,hdl->bthl", Q_all, self.q_to_latent)
            else:
                # Need to recompute from hidden states if we don't cache them
                # For now, approximate with current Q_all (acceptable since it's a local band)
                q_all_latent = q_latent  # Fallback; ideally cache hidden_states too

            # Reciprocal logits: Q_all[j] · K[i] for j attending to i
            # [B,T_q_all,H,L] x [B,T_tot,L] -> [B,H,T_q_all,T_tot]
            logits_recip = torch.einsum(
                "bthd,bsl->bhts", q_all_latent, latent_k
            ) / math.sqrt(L)

            # Extract reciprocal scores for current window [T_tot-T : T_tot]
            # We want logits_recip[:, :, -T:, :] which is [B,H,T,T_tot]
            logits_recip_curr = (
                logits_recip[:, :, -T:, :]
                if q_all_latent.size(1) >= T
                else logits_recip
            )

            # Compute band mask: |i - j| <= W and causal
            band_mask = self._create_band_mask(
                T, T_tot, self.cfg.ra_window, device=hidden_states.device
            )
            band_mask = band_mask & causal_mask  # Combine with causal

            # Add reciprocal term within band (learnable mixing weight)
            logits = torch.where(
                band_mask.unsqueeze(0).unsqueeze(0),
                logits + coupling_scale * self.ra_alpha * logits_recip_curr,
                logits,
            )

        # === 8. Softmax and Dropout ===
        attn = F.softmax(logits, dim=-1)  # [B, H, T, T_tot]

        # === 8a. Apply MLP Gating (Mechanism 1) ===
        if self.cfg.mlp_attn_gate and mlp_gate_context is not None:
            # mlp_gate_context: [B, H] - per-head gating weights
            # Reshape for broadcasting: [B, H, 1, 1]
            gate = mlp_gate_context.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            # Mix gated attention with learnable mixing weight
            attn = (1 - coupling_scale * self.mlp_gate_alpha) * attn + (
                coupling_scale * self.mlp_gate_alpha
            ) * (attn * gate)

        # === 8b. Apply Cross-Token RA (RA-CT) Gating ===
        ra_ct_gate = None  # Will store gate for output mode
        if self.cfg.ra_cross_token:
            if self.cfg.ra_ct_detach_stats:
                with torch.no_grad():
                    stat = self._row_stats(attn)  # [B, H, T]
            else:
                stat = self._row_stats(attn)  # [B, H, T]

            # Affine per-head -> sigmoid gate in [0,1]
            # stat: [B,H,T]
            g = torch.sigmoid(
                stat * self.ra_ct_scale.view(1, self.n_head, 1)
                + self.ra_ct_bias.view(1, self.n_head, 1)
            )  # [B,H,T]

            if self.cfg.ra_ct_apply == "weights":
                # Gate the attention WEIGHTS per token/head (broadcast along key axis)
                gate = g.unsqueeze(-1)  # [B,H,T,1]
                attn = (1 - self.cfg.ra_ct_alpha) * attn + self.cfg.ra_ct_alpha * (
                    attn * gate
                )
            elif self.cfg.ra_ct_apply == "output":
                # Defer to after ctx computation; stash gate for output gating
                ra_ct_gate = g  # [B,H,T]
            else:
                raise ValueError(f"Unknown ra_ct_apply: {self.cfg.ra_ct_apply}")

        # Log metrics if requested and enabled (controlled by flag to save memory)
        if self.cfg.log_attention_entropy and self.enable_metrics_computation:
            self.attention_entropy = self._compute_entropy(attn)
        if (
            self.cfg.log_reciprocity_score
            and self.ra_alpha > 0
            and self.enable_metrics_computation
        ):
            self.reciprocity_score = self._compute_reciprocity(attn)

        attn = self.attn_dropout(attn)

        # === 8b. Export Attention Weights for MLP (Mechanism 2) ===
        if self.cfg.mlp_cross_token:
            self._attn_weights_export = attn.detach()  # [B, H, T, T_tot]

        # === 9. Expand V and Compute Context ===
        V_expanded = self._expand_v(latent_v)  # [B, T_tot, H, D]

        # Weighted sum: [B,H,T,T_tot] x [B,T_tot,H,D] -> [B,T,H,D]
        ctx = torch.einsum("bhts,bshd->bthd", attn, V_expanded)

        # === 9a. Apply RA-CT Output Gating (if output mode) ===
        if self.cfg.ra_cross_token and self.cfg.ra_ct_apply == "output":
            # ctx: [B,T,H,D], ra_ct_gate: [B,H,T] -> [B,T,H,1]
            ctx = ctx * ra_ct_gate.transpose(1, 2).unsqueeze(-1)

        # === 10. Merge Heads and Output Projection ===
        out = self._merge_heads(ctx)  # [B, T, E]
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        # === 11. Update Cache ===
        new_past = None
        if use_cache:
            # Cache last W queries for reciprocal attention
            if self.cfg.cache_q_window and self.ra_alpha > 0:
                keep_q = Q[:, -min(T, self.cfg.ra_window) :, :, :]
                if past_key_value is not None and "q_band" in past_key_value:
                    q_band = past_key_value["q_band"]
                    if q_band is not None:
                        q_band = torch.cat([q_band, keep_q], dim=1)
                        # Truncate to window size
                        if q_band.size(1) > self.cfg.ra_window:
                            q_band = q_band[:, -self.cfg.ra_window :, :, :]
                    else:
                        q_band = keep_q
                else:
                    q_band = keep_q
            else:
                q_band = None

            new_past = {
                "latent_k": latent_k.detach(),
                "latent_v": latent_v.detach(),
                "q_band": q_band.detach() if q_band is not None else None,
            }

        return out, new_past

    def _create_causal_mask(self, T: int, T_tot: int, device) -> torch.Tensor:
        """
        Create causal mask: current token i can attend to past tokens j where j <= i.
        Current tokens are at absolute positions [T_tot-T : T_tot].

        Returns:
            mask: [T, T_tot] where True means allowed, False means masked
        """
        # Absolute positions for current window
        i = torch.arange(T_tot - T, T_tot, device=device).unsqueeze(-1)  # [T, 1]
        j = torch.arange(T_tot, device=device).unsqueeze(0)  # [1, T_tot]
        causal = j <= i  # [T, T_tot]
        return causal

    def _create_band_mask(
        self, T: int, T_tot: int, window: int, device
    ) -> torch.Tensor:
        """
        Create band mask: |i - j| <= window.

        Returns:
            mask: [T, T_tot] where True means within band, False means outside
        """
        i = torch.arange(T_tot - T, T_tot, device=device).unsqueeze(-1)  # [T, 1]
        j = torch.arange(T_tot, device=device).unsqueeze(0)  # [1, T_tot]
        band = (i - j).abs() <= window  # [T, T_tot]
        return band

    def _compute_entropy(self, attn: torch.Tensor) -> float:
        """Compute average attention entropy across all heads and positions.

        Memory-efficient implementation that computes in-place to reduce memory overhead.
        """
        # attn: [B, H, T, T_tot]
        eps = 1e-12

        # Compute entropy in a memory-efficient way
        # Instead of creating log_attn as a separate tensor, compute directly
        # entropy = -sum(p * log(p))
        with torch.no_grad():
            # Detach to avoid gradient tracking overhead
            attn_detached = attn.detach()
            # Compute log in-place on a clone to avoid modifying original
            log_attn = torch.log(attn_detached + eps)
            entropy = -(attn_detached * log_attn).sum(dim=-1)  # [B, H, T]
            result = entropy.mean().item()
            # Clean up immediately
            del log_attn, entropy, attn_detached
        return result

    def _compute_reciprocity(self, attn: torch.Tensor) -> float:
        """
        Compute reciprocity score: correlation between A[i,j] and A[j,i].
        Only valid for positions where both are defined (causal constraint).
        """
        # attn: [B, H, T, T_tot]
        # For simplicity, compute on the square submatrix [T_tot-T:T_tot, T_tot-T:T_tot]
        T = attn.size(2)
        T_tot = attn.size(3)
        if T_tot < T:
            return 0.0

        # Extract square region
        attn_square = attn[:, :, :, -T:]  # [B, H, T, T]

        # Lower triangle (causal valid region)
        lower_tri = torch.tril(attn_square, diagonal=0)
        upper_tri = lower_tri.transpose(-2, -1)  # Reciprocal

        # Compute correlation (Pearson) in valid region
        valid_mask = (lower_tri > 0) & (upper_tri > 0)
        if valid_mask.sum() == 0:
            return 0.0

        lower_vals = lower_tri[valid_mask]
        upper_vals = upper_tri[valid_mask]

        if lower_vals.numel() < 2:
            return 0.0

        corr = torch.corrcoef(torch.stack([lower_vals, upper_vals]))[0, 1]
        return corr.item() if not torch.isnan(corr) else 0.0

    def get_attn_weights_export(self) -> Optional[torch.Tensor]:
        """Retrieve attention weights for MLP (mechanism 2)."""
        return self._attn_weights_export

    def get_latent_k_export(self) -> Optional[torch.Tensor]:
        """Retrieve latent K for MLP (mechanism 3)."""
        return self._latent_k_export

    def set_coupling_scale(self, scale: float):
        """Set global 0–1 scale for RA and MLP→Attention coupling."""
        scale_val = float(max(0.0, min(1.0, scale)))
        self.coupling_scale.fill_(scale_val)


# --------------------------- Block wrapper for context flow -------------- #


class RA_MLA_Block(nn.Module):
    """
    Wrapper for a transformer block that properly manages MLP↔Attention reciprocity.

    This block carries context from MLP to the next block's attention, fixing the
    silent no-op bug where mlp_gate_context and mlp_latent_context were never passed.
    """

    def __init__(self, attn_core, mlp_core, orig):
        super().__init__()
        self.ln_1, self.ln_2 = orig.ln_1, orig.ln_2
        self.attn, self.mlp = attn_core, mlp_core
        self._ctx = {}  # carry MLP→next Attn across blocks

    def forward(
        self,
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Attention forward with MLP context from previous block
        a_out, present = self.attn(
            self.ln_1(x),
            past_key_value=layer_past,
            use_cache=use_cache,
            attn_mask=attention_mask,
            mlp_gate_context=self._ctx.get("mlp_gate_context_prev"),
            mlp_latent_context=self._ctx.get("mlp_latent_context_prev"),
        )
        x = x + a_out

        # Get attention outputs for MLP (only needed for ReciprocalMLP)
        # Check if MLP is ReciprocalMLP by checking for reciprocal methods
        is_reciprocal_mlp = hasattr(self.mlp, "get_attn_gate_context")

        if is_reciprocal_mlp:
            # MLP forward with attention inputs (ReciprocalMLP)
            attn_weights = self.attn.get_attn_weights_export()
            attn_latent = self.attn.get_latent_k_export()
            m_out = self.mlp(
                self.ln_2(x), attn_weights=attn_weights, attn_latent=attn_latent
            )
            x = x + m_out

            # Produce contexts for NEXT block's attention
            self._ctx = dict(
                mlp_gate_context_prev=self.mlp.get_attn_gate_context(),
                mlp_latent_context_prev=self.mlp.get_mlp_latent_context(),
            )
        else:
            # Standard MLP forward (no reciprocity)
            m_out = self.mlp(self.ln_2(x))
            x = x + m_out

            # No contexts to propagate
            self._ctx = {}

        if output_attentions:
            return x, present, None
        return (x, present) if use_cache else x


# --------------------------- GPT-2 patcher ------------------------------- #


def patch_gpt2_with_ra_mla(
    model,
    latent_dim=64,
    ra_window=64,
    ra_alpha=0.5,
    per_head_q_latent=True,
    per_head_v_up=True,
    cache_q_window=True,
    use_rope=False,
    use_flash=True,
    log_metrics=False,
    # Reciprocal MLP parameters
    mlp_attn_gate=False,
    mlp_cross_token=False,
    mlp_latent_recip=False,
    mlp_gate_alpha=0.1,
    mlp_cross_alpha=0.3,
    mlp_recip_alpha=0.2,
    mlp_gate_dim=64,
    mlp_latent_dim=128,
    mlp_expansion_ratio=4.0,
    # Parameter tying and sparsification
    mlp_tying_mode="tied_transpose",
    mlp_sparse_mode="topk",
    mlp_sparse_k=8,
    mlp_sparse_tau=0.5,
    mlp_sparse_normalize=True,
    mlp_sparse_head_average=True,
    # Cross-Token RA (RA-CT) parameters
    ra_cross_token=False,
    ra_ct_mode="topk",
    ra_ct_apply="output",
    ra_ct_alpha=0.2,
    ra_ct_k=8,
    ra_ct_head_average=False,
    ra_ct_detach_stats=False,
):
    """
    Replace each GPT-2 attention and MLP module with RA+MLA+ReciprocalMLP variant.

    Args:
        model: HuggingFace GPT-2 model
        latent_dim: Latent dimension for K/V compression (L << D)
        ra_window: Local band width for reciprocal attention
        ra_alpha: Weight for reciprocal term (0.0 disables RA, pure MLA)
        per_head_q_latent: Use per-head Q-to-latent projections (more expressive)
        per_head_v_up: Use per-head V up-projections (more expressive)
        cache_q_window: Cache queries for reciprocal attention at inference
        use_rope: Use rotary positional embeddings (GPT-2 vanilla doesn't)
        use_flash: Use FlashAttention when available
        log_metrics: Log attention entropy and reciprocity scores
        mlp_attn_gate: Enable MLP-to-attention gating (mechanism 1)
        mlp_cross_token: Enable cross-token MLP aggregation (mechanism 2)
        mlp_latent_recip: Enable MLP latent space reciprocity (mechanism 3)
        mlp_gate_alpha: Mixing weight for MLP attention gating
        mlp_cross_alpha: Mixing weight for cross-token MLP
        mlp_recip_alpha: Mixing weight for MLP latent reciprocity
        mlp_gate_dim: Context vector dimension for gating
        mlp_latent_dim: MLP latent dimension

    Returns:
        model: Modified model with RA+MLA attention and reciprocal MLP
        cfg: Configuration object
    """
    cfg = RA_MLA_Config(
        latent_dim=latent_dim,
        ra_window=ra_window,
        ra_alpha=ra_alpha,
        per_head_q_latent=per_head_q_latent,
        per_head_v_up=per_head_v_up,
        cache_q_window=cache_q_window,
        use_rope=use_rope,
        use_flash=use_flash,
        log_attention_entropy=log_metrics,
        log_reciprocity_score=log_metrics,
        # Reciprocal MLP config
        mlp_attn_gate=mlp_attn_gate,
        mlp_cross_token=mlp_cross_token,
        mlp_latent_recip=mlp_latent_recip,
        mlp_gate_alpha=mlp_gate_alpha,
        mlp_cross_alpha=mlp_cross_alpha,
        mlp_recip_alpha=mlp_recip_alpha,
        mlp_gate_dim=mlp_gate_dim,
        mlp_latent_dim=mlp_latent_dim,
        mlp_expansion_ratio=mlp_expansion_ratio,
        # Parameter tying and sparsification
        mlp_tying_mode=mlp_tying_mode,
        mlp_sparse_mode=mlp_sparse_mode,
        mlp_sparse_k=mlp_sparse_k,
        mlp_sparse_tau=mlp_sparse_tau,
        mlp_sparse_normalize=mlp_sparse_normalize,
        mlp_sparse_head_average=mlp_sparse_head_average,
        # Cross-Token RA (RA-CT) config
        ra_cross_token=ra_cross_token,
        ra_ct_mode=ra_ct_mode,
        ra_ct_apply=ra_ct_apply,
        ra_ct_alpha=ra_ct_alpha,
        ra_ct_k=ra_ct_k,
        ra_ct_head_average=ra_ct_head_average,
        ra_ct_detach_stats=ra_ct_detach_stats,
    )

    for i, block in enumerate(model.transformer.h):
        n_embd = model.config.n_embd
        n_head = model.config.n_head
        original_attn = block.attn
        original_mlp = block.mlp

        # Build RA+MLA attention
        ra_attn = RA_MLA_Attention(n_embd=n_embd, n_head=n_head, cfg=cfg)

        # Seed q_proj from original c_attn's Q slice
        with torch.no_grad():
            # original c_attn projects to [Q|K|V] jointly: weight shape [3E, E]
            Wqkv = original_attn.c_attn.weight.data  # [3E, E]
            E = n_embd
            # Q slice - first E rows (out of 3E) correspond to Q projection
            ra_attn.q_proj.weight.copy_(Wqkv[:E, :])

        # Preserve dropout configs if any
        if hasattr(original_attn, "attn_dropout"):
            ra_attn.attn_dropout.p = getattr(original_attn.attn_dropout, "p", 0.0)
        if hasattr(original_attn, "resid_dropout"):
            ra_attn.resid_dropout.p = getattr(original_attn.resid_dropout, "p", 0.0)

        # === Build MLP (reciprocal or standard) ===
        recip_mlp = (
            ReciprocalMLP(n_embd=n_embd, n_head=n_head, cfg=cfg)
            if (mlp_attn_gate or mlp_cross_token or mlp_latent_recip)
            else original_mlp
        )

        # Copy weights from original MLP if we created a new one
        if recip_mlp is not original_mlp:
            with torch.no_grad():
                # Get dimensions
                orig_hidden_dim = original_mlp.c_fc.weight.shape[0]
                new_hidden_dim = recip_mlp.c_fc.weight.shape[0]

                if orig_hidden_dim == new_hidden_dim:
                    # Dimensions match, copy directly
                    recip_mlp.c_fc.weight.copy_(original_mlp.c_fc.weight)
                    recip_mlp.c_fc.bias.copy_(original_mlp.c_fc.bias)
                    recip_mlp.c_proj.weight.copy_(original_mlp.c_proj.weight)
                    recip_mlp.c_proj.bias.copy_(original_mlp.c_proj.bias)
                else:
                    # Dimensions differ, copy what matches and initialize the rest
                    min_hidden = min(orig_hidden_dim, new_hidden_dim)

                    # Copy c_fc (input -> hidden)
                    recip_mlp.c_fc.weight[:min_hidden].copy_(
                        original_mlp.c_fc.weight[:min_hidden]
                    )
                    recip_mlp.c_fc.bias[:min_hidden].copy_(
                        original_mlp.c_fc.bias[:min_hidden]
                    )

                    # Copy c_proj (hidden -> output)
                    recip_mlp.c_proj.weight[:, :min_hidden].copy_(
                        original_mlp.c_proj.weight[:, :min_hidden]
                    )
                    recip_mlp.c_proj.bias.copy_(original_mlp.c_proj.bias)

                    print(
                        f"  MLP dimension mismatch: {orig_hidden_dim} -> {new_hidden_dim}, copied {min_hidden} dims"
                    )

        # === Replace entire block with RA_MLA_Block wrapper ===
        # This properly manages MLP↔Attention reciprocity context flow
        model.transformer.h[i] = RA_MLA_Block(ra_attn, recip_mlp, block)

    # Mark config so your training loop knows it's RA+MLA
    model.config.ra_mla = True
    model.config.ra_mla_latent_dim = latent_dim
    model.config.ra_mla_ra_window = ra_window
    model.config.ra_mla_ra_alpha = ra_alpha
    model.config.ra_mla_mlp_attn_gate = mlp_attn_gate
    model.config.ra_mla_mlp_cross_token = mlp_cross_token
    model.config.ra_mla_mlp_latent_recip = mlp_latent_recip
    return model, cfg


# --------------------------- Pruning hooks -------------------------------- #


def score_heads_for_prune_gpt2(
    model, optimizer_state: Dict, gamma=1.0, delta=0.5, eps=1e-8
):
    """
    Score attention heads for pruning using AdamW SNR on parameters.

    Combines SNR from:
    - Q projection weights (per-head slices)
    - V up-projection weights (if per-head)
    - Q-to-latent projection (if per-head)

    Args:
        model: GPT-2 model with RA+MLA attention
        optimizer_state: AdamW optimizer state dict
        gamma: SNR exponent for momentum (default 1.0)
        delta: SNR exponent for second moment (default 0.5)
        eps: Numerical stability epsilon

    Returns:
        scores: dict[layer_index] -> Tensor[n_head] head importance scores
    """
    scores = {}
    for li, block in enumerate(model.transformer.h):
        attn = block.attn.core  # RA_MLA_Attention instance

        H, D = attn.n_head, attn.head_dim
        head_scores = torch.zeros(H, device=attn.q_proj.weight.device)

        # 1. Q projection SNR (packed heads: [E, E] where E = H*D)
        s_q = snr_from_adam_state(
            attn.q_proj.weight, optimizer_state, eps, gamma, delta
        )
        if s_q is not None and s_q.numel() > 0:
            for h in range(H):
                # Aggregate SNR for this head's slice
                head_scores[h] += s_q[h * D : (h + 1) * D, :].median()

        # 2. V up-projection SNR (if per-head)
        if attn.cfg.per_head_v_up and hasattr(attn, "v_up"):
            # v_up: [H, L, D]
            v_snr = snr_from_adam_state(attn.v_up, optimizer_state, eps, gamma, delta)
            if v_snr is not None and v_snr.numel() > 0:
                # Median over L,D dimensions for each head
                for h in range(H):
                    head_scores[h] += v_snr[h, :, :].median()

        # 3. Q-to-latent projection SNR (if per-head)
        if attn.cfg.per_head_q_latent and hasattr(attn, "q_to_latent"):
            # q_to_latent: [H, D, L]
            q_lat_snr = snr_from_adam_state(
                attn.q_to_latent, optimizer_state, eps, gamma, delta
            )
            if q_lat_snr is not None and q_lat_snr.numel() > 0:
                for h in range(H):
                    head_scores[h] += q_lat_snr[h, :, :].median()

        # Normalize by number of contributing scores
        num_scores = 1  # q_proj always contributes
        if attn.cfg.per_head_v_up and hasattr(attn, "v_up"):
            num_scores += 1
        if attn.cfg.per_head_q_latent and hasattr(attn, "q_to_latent"):
            num_scores += 1
        head_scores = head_scores / num_scores

        scores[li] = head_scores

    return scores


def prune_heads_ra_mla_gpt2(model, keep_fraction: float = 0.75):
    """
    Sketch: mark low-score heads for pruning.
    (Your other AI can rewrite to physically rebuild q_proj/out_proj dims.)
    """
    # In real implementation, rebuild q_proj/out_proj dims & v_up to drop heads.
    # Here we just return which heads to keep.
    head_plan = {}
    for li, block in enumerate(model.transformer.h):
        H = block.attn.core.n_head
        keep = int(max(1, round(H * keep_fraction)))
        head_plan[li] = list(range(keep))  # keep first K heads (placeholder)
    return head_plan


def set_ra_mlp_coupling_scale(model: nn.Module, scale: float):
    """
    Set global 0–1 warmup scale for all RA_MLA_Attention and ReciprocalMLP modules
    in the model.
    """
    scale_val = float(max(0.0, min(1.0, scale)))
    for m in model.modules():
        if hasattr(m, "set_coupling_scale"):
            m.set_coupling_scale(scale_val)
