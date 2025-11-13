# ra_lens_gpt2.py
# ------------------------------------------------------------
# GPT-2 with Lens-Gated Attention: RA + Discoverability + Route-Gated MLP
#
# ==============================================================================
# CORE INSIGHT: Mathematical Reciprocity is Just a Transpose
# ==============================================================================
#
# Standard attention computes directed affinity scores:
#   S = Q @ K^T    # [B, H, T, T]
#   S[i,j] = <Q_i, K_j> measures how much query i wants key j
#
# But S is asymmetric: S[i,j] ≠ S[j,i] in general.
#
# RECIPROCAL ATTENTION adds symmetric mutual affinity:
#   S_rec = S^T    # Just transpose! FREE operation, no extra GEMM
#   S_rec[i,j] = S[j,i] = <Q_j, K_i>
#
# Final scores with lens mixing:
#   logits = w_std * S + w_rec * S_rec + w_disc * d
#   where w = softmax(gates) per head (always sum to 1 for stability)
#
# Visual example (T=4):
#   S = [[a, b, c, d],      S_rec = [[a, e, i, m],
#        [e, f, g, h],               [b, f, j, n],
#        [i, j, k, l],               [c, g, k, o],
#        [m, n, o, p]]               [d, h, l, p]]
#
# Diagonal is symmetric (self-attention), off-diagonal captures mutual affinity.
# If token i strongly attends to j (large S[i,j]), boost j's attention to i.
#
# ==============================================================================
# KEY INNOVATIONS
# ==============================================================================
#
# 1. **Reciprocity (RA)**: S_rec = S^T
#    - Zero compute cost (transpose is free)
#    - Captures mutual token affinity
#    - Helps tokens "find each other" bidirectionally
#
# 2. **Discoverability**: d_j = <K_j, u_h>
#    - Column bias: tokens can "broadcast" importance
#    - Tiny parameter cost (n_head * head_dim)
#    - Allows important tokens to be found regardless of query
#
# 3. **Lens Gates**: w = softmax([w_std, w_rec, w_disc])
#    - Scale stability: weights always sum to 1 per head
#    - Learnable mixing: model learns optimal balance
#    - Init bias toward standard attention (w_std≈0.8)
#
# 4. **Route Gate**: g = sigmoid(route_raw + bias)
#    - Learns attention vs MLP ratio
#    - g≈1.0: attention-heavy (traditional 4:1, large KV cache)
#    - g≈0.3: MLP-heavy (closer to 1:2.5, small KV cache)
#    - GOAL: Reduce KV cache size at inference by shifting to MLP
#
# 5. **MLP Context Summary**: mlp_input = concat([x, attn_out])
#    - MLP sees what attention computed
#    - Enables MLP to participate with cross-token information
#    - No extra projections (compute-neutral)
#
# ==============================================================================
# PARAMETER EFFICIENCY
# ==============================================================================
#
# Standard transformer block:
#   - Attention: 4*E^2 params (Q/K/V/O projections)
#   - MLP: 8*E^2 params (2 layers, 4x expansion)
#   - Total: 12*E^2
#
# Lens-gated block adds:
#   - Lens gates: 3*H params (~36 for GPT-2)
#   - Discoverability u_h: H*D params (~768 for GPT-2)
#   - Route gate: 1 param per block
#   - Low-rank MLP context (optional): E*R + R*mult*E (491K with R=128)
#   - Overhead: 0.01% (mechanisms 1-4), 7% with low-rank context
#
# Compute overhead: ZERO extra GEMMs beyond standard transformer
#
# ==============================================================================
# USAGE
# ==============================================================================
#
#   from transformers import AutoModelForCausalLM
#   from ra_lens_gpt2 import patch_gpt2_with_lens_attention
#
#   model = AutoModelForCausalLM.from_pretrained("gpt2")
#   patch_gpt2_with_lens_attention(model,
#       use_reciprocity=True,
#       use_discoverability=True,
#       use_route_gate=True,
#       mlp_use_ctx_summary=True)
#
#   # Train with route gate annealing to learn optimal KV cache size
#   for step in training_loop:
#       loss.backward()
#       optimizer.step()
#       # Optional: anneal route gate toward MLP over time
#       if step > warmup_steps:
#           model.adjust_route_bias(delta=-0.0001)  # Gradual shift to MLP
#
# ==============================================================================

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- Config dataclass --------------------------- #


@dataclass
class LensConfig:
    """Configuration for lens-gated attention mechanism."""

    # Model dimensions
    d_model: int = 768
    n_head: int = 12
    head_dim: int = 64

    # Lens gates for attention scoring
    use_reciprocity: bool = True  # Mix S + w_rec * S^T
    use_discoverability: bool = True  # Add column bias from u_h vectors
    share_u_across_layers: bool = False  # Share u_h across layers

    # Lens gate initialization (before softmax)
    init_gate_std: float = 2.2  # Bias toward standard attention initially
    init_gate_rec: float = 0.6  # Reciprocity starts small
    init_gate_disc: float = 0.1  # Discoverability starts very small

    # K/V compression (MLA-style) for parameter efficiency
    use_kv_compression: bool = False  # Compress K/V via low-rank factorization
    kv_latent_dim: int = 128  # Latent dimension for K/V compression

    # Windowing
    causal: bool = True
    ra_window: Optional[int] = None  # Optional local attention band

    # MLP configuration
    mlp_expansion_ratio: float = 4.0  # Standard transformer: 4x
    mlp_use_ctx_summary: bool = True  # Feed attention output to MLP
    mlp_ctx_detach: bool = True  # Stop gradient on ctx summary
    mlp_ctx_rank: int = 128  # Low-rank bottleneck for context projection (cheap!)
    mlp_ctx_conductor: bool = (
        False  # Only use context when route_gate < 0.5 (MLP-heavy)
    )
    mlp_disabled: bool = False  # Disable MLP entirely (attention-only ablation)

    # Route gate (learn attention vs MLP ratio)
    use_route_gate: bool = True  # Enable route gating
    init_route_gate: float = 2.2  # sigmoid≈0.9 (attention-heavy initially)
    # Annealing: Gradually shift from attention toward MLP during training
    route_anneal_start: int = 2000  # Step to start annealing
    route_anneal_end: int = 10000  # Step to finish annealing
    route_anneal_target: float = -1.0  # Target: sigmoid(-1.0)≈0.27 (MLP-heavy)

    # Regularization
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # Performance
    use_flash: bool = True  # Use FlashAttention when available

    # Metrics logging
    log_attention_entropy: bool = False


# --------------------------- Lens-Gated Attention --------------------------- #


class LensGatedAttention(nn.Module):
    """
    Lens-gated attention with reciprocity and discoverability.

    Scoring: logits = w_std*S + w_rec*S^T + w_disc*d_col
    where w = softmax(gates) per head ensures scale stability.
    """

    def __init__(self, cfg: LensConfig):
        super().__init__()
        self.cfg = cfg
        E, H, D = cfg.d_model, cfg.n_head, cfg.head_dim
        assert E == H * D, f"d_model={E} must equal n_head*head_dim={H}*{D}"

        self.n_head = H
        self.head_dim = D

        # Q/K/V projections (optionally compressed for parameter efficiency)
        self.q_proj = nn.Linear(E, H * D, bias=False)  # Q always full rank

        if cfg.use_kv_compression:
            # Compressed K/V via low-rank factorization (MLA-style)
            # Saves params: 2*E*H*D -> 2*(E*R + R*H*D)
            # Example: 2*768*768 = 1.18M -> 2*(768*128 + 128*768) = 392K
            # Savings: 788K per layer (9.5M total for 12 layers)
            R = cfg.kv_latent_dim
            self.k_down = nn.Linear(E, R, bias=False)  # E -> R
            self.k_up = nn.Linear(R, H * D, bias=False)  # R -> H*D
            self.v_down = nn.Linear(E, R, bias=False)
            self.v_up = nn.Linear(R, H * D, bias=False)
            self.k_proj = None  # Mark as compressed
            self.v_proj = None
        else:
            # Standard full-rank K/V
            self.k_proj = nn.Linear(E, H * D, bias=False)
            self.v_proj = nn.Linear(E, H * D, bias=False)
            self.k_down = None
            self.k_up = None
            self.v_down = None
            self.v_up = None

        self.o_proj = nn.Linear(H * D, E, bias=False)

        # Per-head lens gates: [w_std, w_rec, w_disc]
        # Softmax ensures stable mixing (weights sum to 1)
        self.gates = nn.Parameter(torch.zeros(H, 3))
        with torch.no_grad():
            self.gates[:, 0] = cfg.init_gate_std
            self.gates[:, 1] = cfg.init_gate_rec
            self.gates[:, 2] = cfg.init_gate_disc

        # Per-head discoverability vectors u_h ∈ ℝ^D
        if cfg.use_discoverability:
            n_vecs = 1 if cfg.share_u_across_layers else H
            self.u = nn.Parameter(torch.randn(n_vecs, D) * 0.02)
        else:
            self.register_parameter("u", None)

        # Regularization
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        # FlashAttention availability
        self.flash_available = cfg.use_flash and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if self.flash_available:
            print(f"⚠️  WARNING: Flash attention available but NOT IMPLEMENTED in lens-gated attention!")
            print(f"    All configurations (including baseline L0) use open-coded attention.")

        # Cached masks
        self._cached_masks = {}

    def _causal_band_mask(self, T, device, dtype):
        """Create causal mask with optional band window."""
        key = (T, device)
        if key in self._cached_masks:
            return self._cached_masks[key]

        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        if self.cfg.ra_window is not None:
            idx = torch.arange(T, device=device)
            band = (idx[None, :] - idx[:, None]).abs() <= int(self.cfg.ra_window)
            mask = mask & band

        self._cached_masks[key] = mask
        return mask

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, E]

        Returns:
            out: [B, T, E]
            attn_weights: [B, H, T, T] or None
        """
        B, T, E = hidden_states.shape
        H, D = self.n_head, self.head_dim

        # Projections: [B, T, E] -> [B, H, T, D]
        Q = self.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2)

        if self.cfg.use_kv_compression:
            # Compressed K/V: E -> R -> H*D
            K = self.k_up(self.k_down(hidden_states)).view(B, T, H, D).transpose(1, 2)
            V = self.v_up(self.v_down(hidden_states)).view(B, T, H, D).transpose(1, 2)
        else:
            # Standard full-rank K/V
            K = self.k_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
            V = self.v_proj(hidden_states).view(B, T, H, D).transpose(1, 2)

        # Base scores (single GEMM)
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

        # Per-head softmax lens gates
        w = F.softmax(self.gates, dim=-1)  # [H, 3]
        w_std = w[:, 0].view(1, H, 1, 1)
        w_rec = w[:, 1].view(1, H, 1, 1)
        w_disc = w[:, 2].view(1, H, 1, 1)

        logits = w_std * S

        # Reciprocity lens: S^T (pure mathematical reciprocity)
        if self.cfg.use_reciprocity:
            logits = logits + w_rec * S.transpose(-2, -1)

        # Discoverability lens: column bias from u_h
        if self.cfg.use_discoverability:
            # u: [1 or H, D] -> broadcast to [H, D]
            u = self.u if (self.u.shape[0] == H) else self.u.expand(H, -1)
            # d_raw: [B, H, T] = <K[b,h,t,:], u[h,:]>
            d_raw = torch.einsum("bhtd,hd->bht", K, u)
            # Zero-mean for stability
            d = d_raw - d_raw.mean(dim=-1, keepdim=True)  # [B, H, T]
            d = d.unsqueeze(-2)  # [B, H, 1, T] - column bias
            logits = logits + w_disc * d

        # Causal masking
        if self.cfg.causal:
            mask = self._causal_band_mask(T, logits.device, torch.bool)  # [T, T]
            logits = logits.masked_fill(~mask, float("-inf"))

        # Attention weights and aggregation
        attn = F.softmax(logits, dim=-1)  # [B, H, T, T]
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, V)  # [B, H, T, D]

        # Merge heads and output projection
        out = ctx.transpose(1, 2).contiguous().view(B, T, E)  # [B, T, E]
        out = self.o_proj(out)
        out = self.resid_dropout(out)

        return out, attn


# --------------------------- Gated MLP --------------------------- #


class GatedMLP(nn.Module):
    """
    Gated MLP with optional low-rank context summary.

    Key insight: Keep param count LOW via low-rank factorization.

    Instead of: ctx_proj: E → mult*E (2.36M params for GPT-2!)
    Use: ctx_down: E → R, ctx_up: R → mult*E (only 491K params for R=128)

    This gives MLP access to cross-token context without bloat.

    Optional conductor mode: Only use context when route_gate < 0.5
    (MLP-heavy mode). When attention-heavy, MLP doesn't need help.
    """

    def __init__(self, cfg: LensConfig):
        super().__init__()
        E = cfg.d_model
        mult = int(cfg.mlp_expansion_ratio)

        # Standard MLP dimensions (compute-neutral, matches GPT-2)
        self.fc1 = nn.Linear(E, mult * E)
        self.fc2 = nn.Linear(mult * E, E)

        # Optional: LOW-RANK context projection (parameter efficient!)
        if cfg.mlp_use_ctx_summary:
            R = cfg.mlp_ctx_rank  # Bottleneck dimension (e.g., 128)
            # Two-stage projection: E → R → mult*E
            self.ctx_down = nn.Linear(E, R, bias=False)  # E × R
            self.ctx_up = nn.Linear(R, mult * E, bias=False)  # R × mult*E
            # Total: E×R + R×mult*E (e.g., 768×128 + 128×3072 = 491K params)
            # vs full rank: E×mult*E (e.g., 768×3072 = 2.36M params)
            # Savings: 5× reduction!

            # Learnable blending weight
            self.ctx_alpha = nn.Parameter(torch.tensor(0.1))  # Start small
        else:
            self.ctx_down = None
            self.ctx_up = None
            self.ctx_alpha = None

        self.cfg = cfg
        self.route_gate_value = None  # Set by block during forward

    def forward(
        self,
        H: torch.Tensor,
        ctx_summary: Optional[torch.Tensor] = None,
        route_gate: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            H: [B, T, E] - hidden states (token-wise)
            ctx_summary: [B, T, E] or None - attention output (cross-token context)
            route_gate: float or None - current route gate value (for conductor mode)

        Returns:
            mlp_out: [B, T, E]
        """
        # Standard MLP forward (token-wise, matches GPT-2)
        h = self.fc1(H)  # [B, T, mult*E]
        h = F.gelu(h)  # Standard GELU activation (no gating)

        # Optional: blend in attention context (low-rank projection)
        if (
            self.cfg.mlp_use_ctx_summary
            and ctx_summary is not None
            and self.ctx_down is not None
        ):
            # Conductor mode: only use context when MLP-heavy (route_gate < 0.5)
            use_context = True
            if self.cfg.mlp_ctx_conductor and route_gate is not None:
                use_context = route_gate < 0.5  # Only when MLP needs to compete

            if use_context:
                # Low-rank projection: E → R → mult*E
                ctx_h = self.ctx_down(ctx_summary)  # [B, T, R]
                ctx_h = self.ctx_up(ctx_h)  # [B, T, mult*E]

                # Mix: mostly token-wise MLP, small amount of cross-token context
                alpha = self.ctx_alpha.clamp(0.0, 1.0)  # Keep in [0,1]
                h = (1 - alpha) * h + alpha * ctx_h

        # Output projection
        return self.fc2(h)


# --------------------------- Lens Block (Transformer Block) --------------------------- #


class LensBlock(nn.Module):
    """
    Transformer block with lens-gated attention and route-gated residual mixing.

    Route gate learns to blend attention vs MLP contributions:
    - g≈1.0: attention-heavy (traditional, large KV cache)
    - g≈0.3: MLP-heavy (smaller KV cache, better inference efficiency)
    """

    def __init__(self, cfg: LensConfig):
        super().__init__()
        self.attn = LensGatedAttention(cfg)
        self.mlp = GatedMLP(cfg)
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.ln_mlp = nn.LayerNorm(cfg.d_model)

        # Route gate: learns attention vs MLP ratio
        if cfg.use_route_gate:
            self.route_gate_raw = nn.Parameter(torch.tensor(cfg.init_route_gate))
            # Buffer for external scheduling (annealing)
            self.register_buffer("route_bias_add", torch.tensor(0.0))
        else:
            # No route gating - standard residual connections
            self.register_buffer("route_gate_raw", torch.tensor(10.0))  # sigmoid≈1
            self.register_buffer("route_bias_add", torch.tensor(0.0))

        self.cfg = cfg

    @property
    def route_gate(self) -> torch.Tensor:
        """Route gate in [0,1] with optional external bias."""
        return torch.sigmoid(self.route_gate_raw + self.route_bias_add)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            H: [B, T, E]

        Returns:
            out: [B, T, E]
            stats: dict with metrics
        """
        B, T, E = H.shape

        # Attention branch
        attn_out, attn_weights = self.attn(H)  # [B, T, E], [B, H, T, T]
        H1 = H + self.ln_attn(attn_out)

        # MLP branch (if not disabled)
        if self.cfg.mlp_disabled:
            # Attention-only ablation
            out = H1
            stats = {"route_gate": 1.0, "mlp_disabled": True}
        else:
            # Route gate value (for conductor mode)
            g = self.route_gate
            g_val = g.item() if isinstance(g, torch.Tensor) else g

            # MLP branch with optional context summary
            if self.cfg.mlp_use_ctx_summary:
                ctx_summary = attn_out.detach() if self.cfg.mlp_ctx_detach else attn_out
            else:
                ctx_summary = None

            # Pass route_gate to MLP for conductor mode
            mlp_out = self.mlp(
                H1, ctx_summary=ctx_summary, route_gate=g_val
            )  # [B, T, E]
            H2 = H1 + self.ln_mlp(mlp_out)

            # Route-gated residual mixing
            # Blend: H + g*(attention contribution) + (1-g)*(MLP contribution)
            out = H + g * (H1 - H) + (1 - g) * (H2 - H1)

            # Compute metrics
            stats = {"route_gate": g_val}

        if self.cfg.log_attention_entropy and attn_weights is not None:
            ent = self._row_entropy(attn_weights).mean().item()
            stats["attn_entropy"] = ent

        return out, stats

    @staticmethod
    def _row_entropy(attn: torch.Tensor) -> torch.Tensor:
        """Compute entropy per row (query position), average over heads."""
        eps = 1e-12
        p = attn.clamp_min(eps)
        ent = -(p * p.log()).sum(dim=-1)  # [B, H, T]
        return ent.mean(dim=1)  # [B, T]


# --------------------------- Route Gate Annealing --------------------------- #


def compute_route_bias_for_step(step: int, cfg: LensConfig) -> float:
    """
    Compute route gate bias for current training step.

    Gradually shifts from attention-heavy to MLP-heavy:
    - Before anneal_start: bias = 0 (stay at init)
    - During annealing: linear interpolation
    - After anneal_end: bias = final target

    Args:
        step: Current training step
        cfg: LensConfig with annealing parameters

    Returns:
        bias: Value to add to route_gate_raw
    """
    if step < cfg.route_anneal_start:
        return 0.0
    elif step >= cfg.route_anneal_end:
        # Full annealing: target - init
        return cfg.route_anneal_target - cfg.init_route_gate
    else:
        # Linear interpolation
        progress = (step - cfg.route_anneal_start) / (
            cfg.route_anneal_end - cfg.route_anneal_start
        )
        target_bias = cfg.route_anneal_target - cfg.init_route_gate
        return progress * target_bias


def apply_route_annealing(model, step: int, cfg: LensConfig) -> None:
    """
    Apply route gate annealing to all blocks in model.

    Call this during training loop to gradually shift toward MLP.

    Args:
        model: GPT-2 model with lens-gated blocks
        step: Current training step
        cfg: LensConfig with annealing parameters
    """
    bias = compute_route_bias_for_step(step, cfg)
    for module in model.modules():
        # Check for both LensBlock and LensBlockWrapper
        if isinstance(module, LensBlock):
            module.route_bias_add.fill_(bias)
        elif isinstance(module, LensBlockWrapper):
            # LensBlockWrapper stores route_bias_add in the MLP
            if hasattr(module.mlp, "route_bias_add"):
                module.mlp.route_bias_add.fill_(bias)


def get_mean_route_gate(model) -> float:
    """Get mean route gate value across all blocks."""
    gates = []
    for module in model.modules():
        # Check for both LensBlock and LensBlockWrapper
        if isinstance(module, LensBlock) and not module.cfg.mlp_disabled:
            gates.append(module.route_gate.item())
        elif isinstance(module, LensBlockWrapper):
            # LensBlockWrapper stores route_gate in the MLP
            if (
                hasattr(module.mlp, "get_route_gate")
                and not module.lens_config.mlp_disabled
            ):
                g = module.mlp.get_route_gate()
                gates.append(g.item() if isinstance(g, torch.Tensor) else g)
    return sum(gates) / len(gates) if gates else 0.0


# --------------------------- Helper Functions --------------------------- #


def visualize_reciprocity(
    S: torch.Tensor, show_first_head: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Visualize reciprocity by comparing S and S^T.

    Key insight: S[i,j] = how much i attends to j (directed)
                 S^T[i,j] = S[j,i] = how much j attends to i (reciprocal)

    Asymmetry shows where reciprocity adds value:
    - Large S[i,j] but small S[j,i]: i needs j, but j ignores i
    - Reciprocity boosts j's attention to i for mutual communication

    Args:
        S: Attention scores [B, H, T, T] before softmax
        show_first_head: Show stats for first head only

    Returns:
        dict with asymmetry metrics
    """
    if show_first_head:
        S = S[:1, :1]  # [1, 1, T, T]

    S_rec = S.transpose(-2, -1)  # S^T
    asymmetry = (S - S_rec).abs()  # |S - S^T|
    symmetry_score = 1.0 - (asymmetry.mean() / (S.abs().mean() + 1e-8))

    return {
        "S": S,
        "S_reciprocal": S_rec,
        "asymmetry": asymmetry,
        "symmetry_score": symmetry_score.item(),  # 1.0 = perfectly symmetric
    }


def analyze_lens_gates(model) -> Dict[str, torch.Tensor]:
    """
    Analyze lens gate weights across all layers.

    Shows how model distributes attention between:
    - w_std: Standard directed attention
    - w_rec: Reciprocal symmetric attention
    - w_disc: Discoverability broadcast

    Returns:
        dict with per-layer gate weights
    """
    layer_gates = []
    for name, module in model.named_modules():
        if isinstance(module, LensGatedAttention):
            w = F.softmax(module.gates, dim=-1)  # [H, 3]
            layer_gates.append(w.detach().cpu())

    if not layer_gates:
        return {}

    gates_tensor = torch.stack(layer_gates)  # [n_layers, H, 3]
    mean_gates = gates_tensor.mean(dim=[0, 1])  # [3] - average across layers and heads

    return {
        "gates_per_layer": gates_tensor,
        "mean_w_std": mean_gates[0].item(),
        "mean_w_rec": mean_gates[1].item(),
        "mean_w_disc": mean_gates[2].item(),
        "std_w_std": gates_tensor[:, :, 0].std().item(),
        "std_w_rec": gates_tensor[:, :, 1].std().item(),
        "std_w_disc": gates_tensor[:, :, 2].std().item(),
    }


def analyze_route_gates(model) -> Dict[str, float]:
    """
    Analyze route gate values across all layers.

    Route gate controls attention vs MLP balance:
    - g≈1.0: Attention-heavy (traditional 4:1 ratio, large KV cache)
    - g≈0.5: Balanced (2:1 ratio)
    - g≈0.3: MLP-heavy (1:1.4 ratio, small KV cache)

    Returns:
        dict with route gate statistics
    """
    route_gates = []
    for name, module in model.named_modules():
        # Check for both LensBlock and LensBlockWrapper
        if isinstance(module, LensBlock):
            g = module.route_gate.item()
            route_gates.append(g)
        elif isinstance(module, LensBlockWrapper):
            # LensBlockWrapper stores route_gate in the MLP
            if (
                hasattr(module.mlp, "get_route_gate")
                and not module.lens_config.mlp_disabled
            ):
                g = module.mlp.get_route_gate()
                route_gates.append(g.item() if isinstance(g, torch.Tensor) else g)

    if not route_gates:
        return {}

    route_tensor = torch.tensor(route_gates)
    return {
        "mean_route_gate": route_tensor.mean().item(),
        "std_route_gate": route_tensor.std().item(),
        "min_route_gate": route_tensor.min().item(),
        "max_route_gate": route_tensor.max().item(),
        "route_gates_per_layer": route_gates,
    }


def analyze_lens_gates(model) -> Dict[str, float]:
    """
    Analyze lens gate values (w_std, w_rec, w_disc) across all layers.

    Lens gates control per-head blending of attention mechanisms:
    - w_std: Standard attention (Q @ K^T)
    - w_rec: Reciprocity (S^T)
    - w_disc: Discoverability (column bias d)

    Returns:
        dict with lens gate statistics (mean across all heads and layers)
    """
    w_std_list = []
    w_rec_list = []
    w_disc_list = []

    for name, module in model.named_modules():
        # Check for both LensGatedAttention and wrapped versions
        if isinstance(module, LensGatedAttention):
            # gates: [H, 3] - per-head softmax gates
            with torch.no_grad():
                w = F.softmax(module.gates, dim=-1)  # [H, 3]
                w_std_list.extend(w[:, 0].cpu().tolist())  # Standard
                w_rec_list.extend(w[:, 1].cpu().tolist())  # Reciprocity
                w_disc_list.extend(w[:, 2].cpu().tolist())  # Discoverability

    if not w_std_list:
        return {}

    # Compute statistics
    w_std_tensor = torch.tensor(w_std_list)
    w_rec_tensor = torch.tensor(w_rec_list)
    w_disc_tensor = torch.tensor(w_disc_list)

    return {
        # Standard attention weights
        "w_std_mean": w_std_tensor.mean().item(),
        "w_std_std": w_std_tensor.std().item(),
        "w_std_min": w_std_tensor.min().item(),
        "w_std_max": w_std_tensor.max().item(),
        # Reciprocity weights
        "w_rec_mean": w_rec_tensor.mean().item(),
        "w_rec_std": w_rec_tensor.std().item(),
        "w_rec_min": w_rec_tensor.min().item(),
        "w_rec_max": w_rec_tensor.max().item(),
        # Discoverability weights
        "w_disc_mean": w_disc_tensor.mean().item(),
        "w_disc_std": w_disc_tensor.std().item(),
        "w_disc_min": w_disc_tensor.min().item(),
        "w_disc_max": w_disc_tensor.max().item(),
    }


def estimate_kv_cache_savings(
    route_gate: float, baseline_ratio: float = 4.0
) -> Dict[str, float]:
    """
    Estimate KV cache memory savings from route gate learning.

    Key insight: Attention requires KV caching, MLP does not.
    Lower route gate → more MLP reliance → smaller KV cache needed.

    Args:
        route_gate: Current route gate value (0-1)
        baseline_ratio: Baseline MLP:Attention ratio (default 4.0)

    Returns:
        dict with cache size estimates
    """
    # Effective ratio: higher MLP weight → can reduce attention compute
    # This is a simplified model - actual savings depend on inference strategy
    effective_attn_weight = route_gate
    effective_mlp_weight = 1.0 - route_gate

    # Estimate relative KV cache size
    # At g=1.0 (attention-heavy): 100% cache needed
    # At g=0.3 (MLP-heavy): ~30% cache needed (aggressive)
    relative_cache_size = effective_attn_weight

    # Memory savings vs baseline
    memory_savings = 1.0 - relative_cache_size

    return {
        "route_gate": route_gate,
        "effective_attn_weight": effective_attn_weight,
        "effective_mlp_weight": effective_mlp_weight,
        "relative_kv_cache_size": relative_cache_size,
        "estimated_memory_savings": memory_savings,  # 0.0-1.0
    }


def print_architecture_summary(model) -> None:
    """
    Print human-readable summary of lens-gated architecture.

    Shows:
    - Lens gate weights (how attention is mixed)
    - Route gate values (attention vs MLP balance)
    - KV cache size estimates
    - Parameter overhead
    """
    print("=" * 70)
    print("Lens-Gated Architecture Summary")
    print("=" * 70)

    # Lens gates
    lens_stats = analyze_lens_gates(model)
    if lens_stats:
        print("\nLens Gates (Attention Mixing):")
        print(
            f"  w_std (standard):      {lens_stats['mean_w_std']:.3f} ± {lens_stats['std_w_std']:.3f}"
        )
        print(
            f"  w_rec (reciprocity):   {lens_stats['mean_w_rec']:.3f} ± {lens_stats['std_w_rec']:.3f}"
        )
        print(
            f"  w_disc (discoverability): {lens_stats['mean_w_disc']:.3f} ± {lens_stats['std_w_disc']:.3f}"
        )

    # Route gates
    route_stats = analyze_route_gates(model)
    if route_stats:
        print("\nRoute Gates (Attention vs MLP):")
        print(
            f"  Mean route gate:       {route_stats['mean_route_gate']:.3f} ± {route_stats['std_route_gate']:.3f}"
        )
        print(
            f"  Range:                 [{route_stats['min_route_gate']:.3f}, {route_stats['max_route_gate']:.3f}]"
        )

        # KV cache estimate
        mean_g = route_stats["mean_route_gate"]
        cache_stats = estimate_kv_cache_savings(mean_g)
        print(f"\nKV Cache Estimates:")
        print(f"  Relative cache size:   {cache_stats['relative_kv_cache_size']:.1%}")
        print(
            f"  Memory savings:        {cache_stats['estimated_memory_savings']:.1%} vs baseline"
        )

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    print("=" * 70)


# --------------------------- GPT-2 Patching Infrastructure --------------------------- #


class LensBlockWrapper(nn.Module):
    """
    Wrapper that replaces GPT-2 attention and MLP with lens-gated versions.
    Preserves original layer norms and block interface.
    """

    def __init__(self, lens_attn, lens_mlp, orig_block):
        super().__init__()
        # Preserve original layer norms
        self.ln_1 = orig_block.ln_1
        self.ln_2 = orig_block.ln_2
        # Use lens-gated attention and MLP
        self.attn = lens_attn
        self.mlp = lens_mlp
        self.lens_config = lens_attn.cfg

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
        """
        Forward pass compatible with HuggingFace GPT-2 block interface.
        """
        # Attention branch
        attn_out, attn_weights = self.attn(self.ln_1(x))

        # Route gate for blending
        g = (
            self.mlp.get_route_gate()
            if hasattr(self.mlp, "get_route_gate")
            else torch.tensor(0.9, device=x.device)
        )
        g_val = g.item() if isinstance(g, torch.Tensor) else g

        # Apply attention residual with route gating
        H_attn = x + attn_out

        # MLP branch with optional context summary
        if self.lens_config.mlp_use_ctx_summary:
            ctx_summary = (
                attn_out.detach() if self.lens_config.mlp_ctx_detach else attn_out
            )
        else:
            ctx_summary = None

        # Pass route_gate to MLP for conductor mode
        mlp_out = self.mlp(self.ln_2(H_attn), ctx_summary=ctx_summary, route_gate=g_val)

        # Apply MLP residual with route gating
        H_out = H_attn + mlp_out

        # Route-gated blending (optional)
        if self.lens_config.use_route_gate:
            H_out = x + g * (H_attn - x) + (1 - g) * (H_out - H_attn)

        # HuggingFace interface compatibility
        present = None  # We don't cache KV yet (TODO for inference)
        if output_attentions:
            return H_out, present, attn_weights
        return (H_out, present) if use_cache else H_out


def patch_gpt2_with_lens_attention(
    model,
    use_reciprocity: bool = True,
    use_discoverability: bool = True,
    use_kv_compression: bool = False,
    kv_latent_dim: int = 128,
    use_route_gate: bool = True,
    mlp_use_ctx_summary: bool = True,
    mlp_ctx_detach: bool = True,
    mlp_ctx_rank: int = 128,
    mlp_ctx_conductor: bool = False,
    mlp_disabled: bool = False,
    init_gate_std: float = 2.2,
    init_gate_rec: float = 0.6,
    init_gate_disc: float = 0.1,
    init_route_gate: float = 2.2,
    ra_window: Optional[int] = None,
    mlp_expansion_ratio: float = 4.0,
    log_attention_entropy: bool = False,
):
    """
    Patch a GPT-2 model with lens-gated attention blocks.

    Args:
        model: HuggingFace GPT-2 model
        use_reciprocity: Enable reciprocal attention (S + w_rec*S^T)
        use_discoverability: Enable column bias from u_h vectors
        use_kv_compression: Compress K/V via low-rank factorization (parameter-neutral)
        kv_latent_dim: Latent dimension for K/V compression (default 128)
        use_route_gate: Enable route gating (learn attention vs MLP ratio)
        mlp_use_ctx_summary: Feed attention output to MLP
        mlp_ctx_detach: Stop gradient on context summary
        mlp_ctx_rank: Low-rank bottleneck dimension for MLP context (default 128)
        mlp_ctx_conductor: Only use context when route_gate < 0.5 (MLP-heavy)
        mlp_disabled: Disable MLP entirely (attention-only ablation)
        init_gate_std: Initial gate value for standard attention
        init_gate_rec: Initial gate value for reciprocity
        init_gate_disc: Initial gate value for discoverability
        init_route_gate: Initial route gate value
        ra_window: Optional local attention window
        mlp_expansion_ratio: MLP expansion ratio (default 4.0)
        log_attention_entropy: Enable attention entropy logging

    Returns:
        model: Patched model with lens-gated blocks
        cfg: LensConfig object
    """
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    # Create config
    cfg = LensConfig(
        d_model=n_embd,
        n_head=n_head,
        head_dim=head_dim,
        use_reciprocity=use_reciprocity,
        use_discoverability=use_discoverability,
        use_kv_compression=use_kv_compression,
        kv_latent_dim=kv_latent_dim,
        use_route_gate=use_route_gate,
        mlp_use_ctx_summary=mlp_use_ctx_summary,
        mlp_ctx_detach=mlp_ctx_detach,
        mlp_ctx_rank=mlp_ctx_rank,
        mlp_ctx_conductor=mlp_ctx_conductor,
        mlp_disabled=mlp_disabled,
        init_gate_std=init_gate_std,
        init_gate_rec=init_gate_rec,
        init_gate_disc=init_gate_disc,
        init_route_gate=init_route_gate,
        ra_window=ra_window,
        mlp_expansion_ratio=mlp_expansion_ratio,
        log_attention_entropy=log_attention_entropy,
    )

    print(f"Patching GPT-2 with lens-gated attention:")
    print(f"  - Reciprocity: {use_reciprocity}")
    print(f"  - Discoverability: {use_discoverability}")
    print(f"  - Route gate: {use_route_gate}")
    print(f"  - MLP context summary: {mlp_use_ctx_summary}")
    print(f"  - MLP expansion ratio: {mlp_expansion_ratio}")

    # Patch each transformer block
    for i, block in enumerate(model.transformer.h):
        original_attn = block.attn
        original_mlp = block.mlp

        # Create lens-gated attention
        lens_attn = LensGatedAttention(cfg)

        # Copy weights from original attention
        with torch.no_grad():
            # Original c_attn projects to [Q|K|V] jointly: weight shape [3*n_embd, n_embd]
            Wqkv = original_attn.c_attn.weight.data  # [3*n_embd, n_embd]
            E = n_embd

            # Copy Q projection (always full-rank)
            lens_attn.q_proj.weight.copy_(Wqkv[:E, :])  # Q slice

            if cfg.use_kv_compression:
                # Compressed K/V: Initialize via SVD factorization of original weights
                Wk = Wqkv[E : 2 * E, :]  # [E, E]
                Wv = Wqkv[2 * E :, :]  # [E, E]
                R = cfg.kv_latent_dim

                # SVD: W ≈ U @ S @ V^T, keep top R components
                # For W = [E, E], low-rank factorization: W ≈ U[:, :R] @ S[:R, :R] @ V^T[:R, :]
                Uk, Sk, Vk = torch.svd(Wk)
                Uv, Sv, Vv = torch.svd(Wv)

                # k_down: [R, E], k_up: [E, R]
                # SVD: W = U @ S @ V^T ≈ U[:,:R] @ S[:R,:R] @ V[:,:R]^T
                # Split: W ≈ (U[:,:R] @ sqrt(S)) @ (sqrt(S) @ V[:,:R]^T)
                lens_attn.k_down.weight.copy_(
                    (Sk[:R].sqrt().unsqueeze(1) * Vk[:, :R].T)
                )  # [R, E]
                lens_attn.k_up.weight.copy_((Uk[:, :R] * Sk[:R].sqrt()))  # [E, R]

                lens_attn.v_down.weight.copy_(
                    (Sv[:R].sqrt().unsqueeze(1) * Vv[:, :R].T)
                )
                lens_attn.v_up.weight.copy_((Uv[:, :R] * Sv[:R].sqrt()))
            else:
                # Standard full-rank K/V
                lens_attn.k_proj.weight.copy_(Wqkv[E : 2 * E, :])  # K slice
                lens_attn.v_proj.weight.copy_(Wqkv[2 * E :, :])  # V slice

            # Copy output projection
            lens_attn.o_proj.weight.copy_(original_attn.c_proj.weight.data)

        # Preserve dropout configs
        if hasattr(original_attn, "attn_dropout"):
            lens_attn.attn_dropout.p = getattr(original_attn.attn_dropout, "p", 0.0)
        if hasattr(original_attn, "resid_dropout"):
            lens_attn.resid_dropout.p = getattr(original_attn.resid_dropout, "p", 0.0)

        # Create gated MLP
        lens_mlp = GatedMLP(cfg)

        # Copy weights from original MLP
        with torch.no_grad():
            # Original MLP dimensions
            orig_hidden_dim = original_mlp.c_fc.weight.shape[0]
            new_hidden_dim = lens_mlp.fc1.weight.shape[0]
            new_input_dim = lens_mlp.fc1.weight.shape[1]

            # Determine how much we can copy
            E = n_embd
            copy_input_dim = E  # We can only copy the base embedding part
            copy_hidden_dim = min(orig_hidden_dim, new_hidden_dim)

            # Copy fc1 (input -> hidden)
            lens_mlp.fc1.weight[:copy_hidden_dim, :copy_input_dim].copy_(
                original_mlp.c_fc.weight[:copy_hidden_dim, :]
            )
            lens_mlp.fc1.bias[:copy_hidden_dim].copy_(
                original_mlp.c_fc.bias[:copy_hidden_dim]
            )

            # Copy fc2 (hidden -> output)
            lens_mlp.fc2.weight[:, :copy_hidden_dim].copy_(
                original_mlp.c_proj.weight[:, :copy_hidden_dim]
            )
            lens_mlp.fc2.bias.copy_(original_mlp.c_proj.bias)

            if orig_hidden_dim != new_hidden_dim or new_input_dim != E:
                print(
                    f"  Block {i}: MLP dims {E}→{orig_hidden_dim}→{E} to {new_input_dim}→{new_hidden_dim}→{E}"
                )

        # Add route gate to MLP wrapper (make it a proper method)
        lens_mlp.route_gate_raw = nn.Parameter(torch.tensor(init_route_gate))
        lens_mlp.register_buffer("route_bias_add", torch.tensor(0.0))

        # Add method to compute route gate
        def get_route_gate(self):
            return torch.sigmoid(self.route_gate_raw + self.route_bias_add)

        lens_mlp.get_route_gate = get_route_gate.__get__(lens_mlp, GatedMLP)

        # Replace block with wrapped version
        model.transformer.h[i] = LensBlockWrapper(lens_attn, lens_mlp, block)

    # Mark config so training loop knows it's lens-gated
    model.config.lens_gated = True
    model.config.lens_use_reciprocity = use_reciprocity
    model.config.lens_use_discoverability = use_discoverability
    model.config.lens_use_route_gate = use_route_gate
    model.config.lens_mlp_expansion_ratio = mlp_expansion_ratio

    print(f"✓ Patched {len(model.transformer.h)} blocks successfully")
    return model, cfg


# --------------------------- Quick Test --------------------------- #


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E = 2, 32, 768
    H, D = 12, 64
    x = torch.randn(B, T, E)

    cfg = LensConfig(
        d_model=E,
        n_head=H,
        head_dim=D,
        use_reciprocity=True,
        use_discoverability=True,
        use_route_gate=True,
        mlp_use_ctx_summary=True,
    )

    block = LensBlock(cfg)
    y, stats = block(x)
    print(f"out: {y.shape}")
    print(f"route_gate: {stats['route_gate']:.3f}")
    if "attn_entropy" in stats:
        print(f"attn_entropy: {stats['attn_entropy']:.3f}")
