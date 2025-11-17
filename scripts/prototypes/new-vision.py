# ra_rec_disc_gated_gpt2.py
# Minimal, compute-neutral RA + Discoverability + Route-Gated MLP/Attention for GPT-2-style blocks
# Requires: torch>=2.0

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

HELP = r"""
===============================================================================
RA + Discoverability + Route-Gated Transformer Block (Compute-Neutral)
===============================================================================
Rationale (short):
- Standard attention score S_ij = <Q_i, K_j> is a directed bilinear form: S_ij ≠ S_ji in general.
- Reciprocity: Use Sᵀ to emphasize mutual (i↔j) affinity with a tiny per-head gate (no extra GEMM).
- Discoverability: Per-token column bias d_j lets “information want to be found” (tokens can broadcast).
- Route-gated residual: Blend Attention vs MLP outputs with a small gate to anneal reliance on attention,
  letting you shrink KV cache at inference without increasing big-O compute.

Math:
  Q = H W_Q,  K = H W_K,  V = H W_V
  S = (Q Kᵀ) / √D      (scores, [B,H,T,T])
  S_rec = Sᵀ           (reciprocity hint)
  d_j  = <K_j, u_h>    (per-head discoverability from K with tiny vector u_h∈R^D)  → column bias

Per-head lens mixture (softmax-normalized for scale stability):
  logits = w_std*S + w_rec*S_rec + w_disc*d     where w • 1 = 1, w ≥ 0

Causality:
  Apply standard lower-triangular (and optional local band) mask AFTER composing logits.

Compute:
  - Only one QKᵀ GEMM as usual. Sᵀ is a transpose; d is a single einsum/dot with u_h.
  - No change to O(B·H·T²·D). Memory overhead negligible.

Training schedule (suggested):
  - Init lens gates per head near [w_std, w_rec, w_disc] ≈ [0.9, 0.08, 0.02].
  - Keep discoverability small (w_disc ≤ 0.2) to avoid hubness; zero-mean d over T.
  - Route gate on residuals: start ≈0.9 (attention-heavy), anneal toward 0–0.3 to bias MLP.
  - Monitor: attention entropy↑, mean attn distance↓, ppl under KV window W (64–256) at eval.

Ablations:
  - Baseline: set use_reciprocity=False, use_discoverability=False, route_bias=+∞ (gate≈1).
  - RA only: reciprocity=True, discoverability=False.
  - Disc only: reciprocity=False, discoverability=True.
  - Hybrid (recommended): both on with softmax lens gates.

API notes:
  - Route bias hook (route_bias_add) allows trainer to push the gate over time without changing weights.
  - Share u_h across layers or keep per-layer; both are cheap (H*D params per layer).
===============================================================================
"""

@dataclass
class LensConfig:
    d_model: int
    n_head: int
    head_dim: int
    causal: bool = True
    ra_window: int | None = None     # optional local band window (tokens)
    use_reciprocity: bool = True
    use_discoverability: bool = True
    share_u_across_layers: bool = False
    # init gates (before softmax) — bias toward standard attention early
    init_gate_std: float = 2.2
    init_gate_rec: float = 0.6
    init_gate_disc: float = 0.1
    # MLP expansion multiple
    mlp_mult: int = 4
    # enable cheap context-summary side feature into MLP
    mlp_use_ctx_summary: bool = True
    mlp_ctx_detach: bool = True  # stop-grad on ctx summary for stability


class LensGatedAttention(nn.Module):
    """
    Lens-gated attention:
      logits = w_std * S + w_rec * S^T + w_disc * d_col
    where w = softmax(gates) per head. d_col is a column bias from K via per-head vector u_h.
    """
    def __init__(self, cfg: LensConfig):
        super().__init__()
        self.cfg = cfg
        E, H, D = cfg.d_model, cfg.n_head, cfg.head_dim

        self.q_proj = nn.Linear(E, H * D, bias=False)
        self.k_proj = nn.Linear(E, H * D, bias=False)
        self.v_proj = nn.Linear(E, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, E, bias=False)

        # per-head raw gates → softmax → [w_std, w_rec, w_disc]
        self.gates = nn.Parameter(torch.zeros(H, 3))
        with torch.no_grad():
            self.gates[:, 0] = cfg.init_gate_std
            self.gates[:, 1] = cfg.init_gate_rec
            self.gates[:, 2] = cfg.init_gate_disc

        # per-head discoverability vectors u_h ∈ ℝ^D
        if cfg.use_discoverability:
            self.u = nn.Parameter(torch.randn((1 if cfg.share_u_across_layers else cfg.n_head), cfg.head_dim) * 0.02)
        else:
            self.register_parameter("u", None)

        # cached band mask buffer (created on first forward per device/shape)
        self._cached_masks = {}

    def _causal_band_mask(self, T, device, dtype):
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

    def forward(self, hidden_states: torch.Tensor):
        B, T, E = hidden_states.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        assert E == H * D, "d_model must equal n_head * head_dim for this minimal impl."

        # Projections
        Q = self.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        K = self.k_proj(hidden_states).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        V = self.v_proj(hidden_states).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]

        # Base scores (single GEMM)
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)           # [B,H,T,T]

        # Per-head softmax gates (stable mixture)
        w = F.softmax(self.gates, dim=-1)                                 # [H,3]
        w_std, w_rec, w_disc = (w[:, 0].view(1, H, 1, 1),
                                w[:, 1].view(1, H, 1, 1),
                                w[:, 2].view(1, H, 1, 1))

        logits = w_std * S

        # Reciprocity lens
        if self.cfg.use_reciprocity:
            logits = logits + w_rec * S.transpose(-2, -1)

        # Discoverability lens (column bias)
        if self.cfg.use_discoverability:
            # u: [1 or H, D] → broadcast to [H,D]
            u = self.u if (self.u.shape[0] == H) else self.u.expand(H, -1)
            # d_raw: [B,H,T] = <K[b,h,t,:], u[h,:]>
            d_raw = torch.einsum('bhtd,hd->bht', K, u)
            # zero-mean across sequence for stability
            d = d_raw - d_raw.mean(dim=-1, keepdim=True)                  # [B,H,T]
            d = d.unsqueeze(-2)                                           # [B,H,1,T] column bias
            logits = logits + w_disc * d

        # Masking
        if self.cfg.causal:
            mask = self._causal_band_mask(T, logits.device, torch.bool)   # [T,T]
            logits = logits.masked_fill(~mask, float('-inf'))

        # Attention and aggregation
        attn = F.softmax(logits, dim=-1)                                  # [B,H,T,T]
        ctx = torch.matmul(attn, V)                                       # [B,H,T,D]

        # Merge heads
        out = ctx.transpose(1, 2).contiguous().view(B, T, E)              # [B,T,E]
        out = self.o_proj(out)                                            # [B,T,E]

        # return also light-weight diagnostics if needed
        return out, attn


class GatedMLP(nn.Module):
    """MLP with optional cheap context-summary tap and channel gate."""
    def __init__(self, cfg: LensConfig):
        super().__init__()
        E = cfg.d_model
        mult = cfg.mlp_mult
        in_dim = E + (E if cfg.mlp_use_ctx_summary else 0)

        self.norm = nn.LayerNorm(E)
        self.fc1 = nn.Linear(in_dim, mult * E)
        self.fc_gate = nn.Linear(in_dim, mult * E)    # tiny extra gate
        self.fc2 = nn.Linear(mult * E, E)
        self.cfg = cfg

    def forward(self, H: torch.Tensor, ctx_summary: torch.Tensor | None = None):
        # ctx_summary: [B,T,E] or None
        x = self.norm(H)
        if self.cfg.mlp_use_ctx_summary and (ctx_summary is not None):
            x_in = torch.cat([x, ctx_summary], dim=-1)
        else:
            x_in = x
        h = self.fc1(x_in)
        g = torch.sigmoid(self.fc_gate(x_in))
        h = F.gelu(h) * g
        return self.fc2(h)


class RARecDiscBlock(nn.Module):
    """
    GPT-2 style block with lens-gated attention and route-gated residual mixing.
    Route gate is a single scalar per block; trainer can add an external bias to anneal.
    """
    def __init__(self, cfg: LensConfig):
        super().__init__()
        self.attn = LensGatedAttention(cfg)
        self.mlp = GatedMLP(cfg)
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.ln_mlp = nn.LayerNorm(cfg.d_model)

        # route gate (attention vs MLP) — single scalar per block
        self.route_gate_raw = nn.Parameter(torch.tensor(2.2))  # sigmoid≈0.9 start
        self.register_buffer("route_bias_add", torch.tensor(0.0))  # trainer can update per step

        self.cfg = cfg

    @property
    def route_gate(self):
        # gate in [0,1]; add external bias for schedules
        return torch.sigmoid(self.route_gate_raw + self.route_bias_add)

    def forward(self, H: torch.Tensor):
        B, T, E = H.shape

        attn_out, attn_weights = self.attn(H)              # [B,T,E], [B,H,T,T]
        H1 = H + self.ln_attn(attn_out)

        # Optional cheap context summary for MLP (mean over heads of attn @ V already in attn_out;
        # to keep it compute-neutral, reuse attn_out as proxy)
        if self.cfg.mlp_use_ctx_summary:
            ctx_summary = attn_out.detach() if self.cfg.mlp_ctx_detach else attn_out
        else:
            ctx_summary = None

        mlp_out = self.mlp(H1, ctx_summary=ctx_summary)    # [B,T,E]
        H2 = H1 + self.ln_mlp(mlp_out)

        # Route-gated blend (can be applied instead at residual join; this version is simple & robust)
        g = self.route_gate
        # Blend original H with each branch’s contribution
        # (alternative: blend branches before adding to H)
        out = H + g * (H1 - H) + (1 - g) * (H2 - H1)

        return out, {"attn_entropy": self._row_entropy(attn_weights).mean().item(),
                     "route_gate": g.item()}

    @staticmethod
    def _row_entropy(attn: torch.Tensor):
        # entropy per row (query position), average over heads
        eps = 1e-12
        p = attn.clamp_min(eps)
        ent = -(p * p.log()).sum(dim=-1)              # [B,H,T]
        return ent.mean(dim=1)                        # [B,T]


# -------------------------
# Quick wiring helper (example GPT-2-small-like)
# -------------------------
def build_block(d_model=768, n_head=12, head_dim=64, **overrides):
    cfg = LensConfig(d_model=d_model, n_head=n_head, head_dim=head_dim, **overrides)
    return RARecDiscBlock(cfg)


# -------------------------
# Tiny smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E = 2, 32, 768
    H, D = 12, 64
    x = torch.randn(B, T, E)

    block = build_block(d_model=E, n_head=H, head_dim=D,
                        ra_window=None,
                        use_reciprocity=True,
                        use_discoverability=True,
                        mlp_use_ctx_summary=True)

    y, stats = block(x)
    print("out:", y.shape, "entropy≈", round(stats["attn_entropy"], 3), "route_gate≈", round(stats["route_gate"], 3))

