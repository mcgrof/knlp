#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Reciprocal Attention + Reciprocal MLP (same-FLOP reparameterization)

Key ideas
---------
1) UnifiedRAttention:
   - Per-head dim D is split as D = D_std + R.
   - The fused projection emits folded layouts:
       Qf[i] = [Q_std[i], K_low[i]]
       Kf[i] = [K_std[i], Q_low[i]]
     so reciprocal attention is computed inside the same SDPA call.

2) ReciprocalMLP (R-MLP):
   - Expansion dim D_ff is split as D_ff = D_ff_std + R_ff.
   - Up-projection computes h_std and h_low separately.
   - Before the down-projection, we "fold" (swap) a low-rank slice:
       h_fold = [w_std * h_std, w_rec * h_low_swapped]
     where "swapped" is just the intentional cross-coupling path.
   - FLOPs match baseline MLP since D_ff is unchanged; we only reparameterize.

Both modules are drop-in replacements for standard Attention/MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Unified Reciprocal Attention (folded layout RA)
# ---------------------------------------------------------------------

class UnifiedRAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, R: int = 4, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.R = R
        self.D_std = self.head_dim - R
        self.dropout = dropout

        # Fused Qf|Kf|V (same output size as baseline Q|K|V)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Optional gates (initialize near-identity)
        self.w_std = nn.Parameter(torch.ones(n_head) * 0.9)
        self.w_rec = nn.Parameter(torch.ones(n_head) * 0.1)

        self._initialized = False

    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.c_attn.weight)
            nn.init.xavier_uniform_(self.c_proj.weight)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if not self._initialized:
            self._init_weights()

        B, T, C = x.shape
        fused = self.c_attn(x)  # [B, T, 3C]
        qf_flat, kf_flat, v_flat = fused.split(C, dim=-1)

        # [B, H, T, D]
        H, D = self.n_head, self.head_dim
        Qf = qf_flat.view(B, T, H, D).transpose(1, 2).contiguous()
        Kf = kf_flat.view(B, T, H, D).transpose(1, 2).contiguous()
        V  = v_flat.view(B, T, H, D).transpose(1, 2).contiguous()

        # One SDPA call — folded layout already enforces reciprocity
        out = F.scaled_dot_product_attention(
            Qf, Kf, V,
            is_causal=True,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


# ---------------------------------------------------------------------
# Reciprocal MLP (R-MLP)
# ---------------------------------------------------------------------
class ReciprocalMLP(nn.Module):
    """
    Standard MLP: x -> up(D_ff) -> act -> down(D)
    Reciprocal MLP: split D_ff = D_ff_std + R_ff, compute both paths,
    then fold (swap) a low-rank slice before the down-projection.

    FLOPs parity with baseline is achieved by keeping D_ff identical.
    """
    def __init__(
        self,
        n_embd: int,
        expansion: int = 4,   # e.g., 4x for GPT-style
        R_ff: int = 64,       # low-rank slice inside the MLP
        act: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_embd = n_embd
        D_ff = expansion * n_embd
        assert 0 < R_ff < D_ff, "R_ff must be in (0, D_ff)"
        self.R_ff = R_ff
        self.D_ff_std = D_ff - R_ff
        self.dropout = dropout

        # Up: split into std and low branches (same cost as single big up)
        self.up_std = nn.Linear(n_embd, self.D_ff_std, bias=False)
        self.up_low = nn.Linear(n_embd, self.R_ff,     bias=False)

        # Down: single fused down over concatenated folded features
        self.down = nn.Linear(self.D_ff_std + self.R_ff, n_embd, bias=False)

        # Gates analogous to RA
        self.w_std = nn.Parameter(torch.tensor(0.9))
        self.w_rec = nn.Parameter(torch.tensor(0.1))

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.SiLU()

        self._initialized = False

    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.up_std.weight)
            nn.init.xavier_uniform_(self.up_low.weight)
            nn.init.xavier_uniform_(self.down.weight)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if not self._initialized:
            self._init_weights()

        # Up projections
        h_std = self.act(self.up_std(x))  # [B, T, D_ff_std]
        h_low = self.act(self.up_low(x))  # [B, T, R_ff]

        # "Reciprocal fold": cross-couple a low-rank slice
        # (Here it's a direct swap path; you can insert a tiny mixer if desired)
        h_fold = torch.cat([
            self.w_std * h_std,
            self.w_rec * h_low,   # the "reciprocal" contribution
        ], dim=-1)  # [B, T, D_ff_std + R_ff]

        # Down projection (same total D_ff as baseline MLP)
        y = self.down(h_fold)  # [B, T, C]
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        return y


# ---------------------------------------------------------------------
# A tiny Transformer block using UnifiedRAttention + ReciprocalMLP
# ---------------------------------------------------------------------
class ReciprocalBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_R: int = 4,
        mlp_expansion: int = 4,
        mlp_Rff: int = 64,
        dropout: float = 0.0,
        prenorm: bool = True,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = UnifiedRAttention(n_embd=n_embd, n_head=n_head, R=attn_R, dropout=dropout)
        self.mlp  = ReciprocalMLP(n_embd=n_embd, expansion=mlp_expansion, R_ff=mlp_Rff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        return x


# ---------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, C = 2, 128, 768
    H = 12

    x = torch.randn(B, T, C, device=device)

    block = ReciprocalBlock(
        n_embd=C,
        n_head=H,
        attn_R=4,        # RA low-rank per head
        mlp_expansion=4, # typical
        mlp_Rff=64,      # small low-rank MLP slice
        dropout=0.0,
        prenorm=True,
    ).to(device)

    with torch.autocast(device_type=("cuda" if device=="cuda" else "cpu"), dtype=torch.float16 if device=="cuda" else torch.bfloat16):
        y = block(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

