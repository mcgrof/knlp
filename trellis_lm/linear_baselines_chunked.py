"""Exact chunked DeltaNet / Gated-DeltaNet — the fast, faithful baseline kernel.

The naive per-token loop in linear_baselines.py is correct but ~90x too slow to
scale (T sequential tiny kernels). The linear delta rule IS exact-chunkable, so
this computes the *same* recurrence with matmuls over chunks of C tokens (T/C
outer steps instead of T), no approximation.

Recurrence (matching DeltaNetMixer, S is [D_v, D_k], write-before-read, keys
L2-normalised):
    S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T
    o_t = S_t q_t
with a_t = 1 for plain DeltaNet, a_t = sigmoid gate for Gated DeltaNet.

Derivation (plain): write the rank-1 update value u_t = b_t(v_t - S_{t-1} k_t),
so S_t = S_in + sum_{s<=t} u_s k_s^T within a chunk and
    u_t + sum_{s<t} (b_t k_t . k_s) u_s = b_t (v_t - S_in k_t),
i.e. (I + M) U = b (V - K S_in^T), M = tril(diag(b) K K^T, -1). Then
    O = Q S_in^T + tril(Q K^T, 0) U,   S_out = S_in + U^T K.
Gated: substitute S_bar_t = S_t / a_t (a_t = cumprod of the gate in-chunk); in
S_bar-space it is the plain rule with values v_t / a_t, outputs scaled back by
a_t, and S_out scaled by a_C. chunk_size=16 keeps a_C bounded so the 1/a_t
rescale stays fp32-stable for realistic gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .trellis_mixer import RMSNorm


def chunked_delta_rule(q, k, v, beta, alpha=None, chunk_size=16):
    """q,k,v: [B,H,T,D]; beta: [B,H,T]; alpha: [B,H,T] or None. Returns o [B,H,T,D]."""
    B, H, T, D = q.shape
    in_dtype = q.dtype
    q, k, v, beta = (t.float() for t in (q, k, v, beta))
    if alpha is not None:
        alpha = alpha.float()
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta = F.pad(beta, (0, pad))
        if alpha is not None:
            alpha = F.pad(alpha, (0, pad), value=1.0)
    Tp = T + pad
    nC = Tp // chunk_size
    q, k, v = (t.view(B, H, nC, chunk_size, D) for t in (q, k, v))
    beta = beta.view(B, H, nC, chunk_size)
    if alpha is not None:
        alpha = alpha.view(B, H, nC, chunk_size)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)  # [D_v, D_k]
    eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
    outs = []
    for c in range(nC):
        Kc, Vc, Qc, Bc = k[:, :, c], v[:, :, c], q[:, :, c], beta[:, :, c]
        KKt = Kc @ Kc.transpose(-1, -2)  # [B,H,C,C]: k_t . k_s
        QKt = Qc @ Kc.transpose(-1, -2)  # q_t . k_s
        Sk = Kc @ S.transpose(-1, -2)  # [B,H,C,D] rows = (S k_t)^T
        if alpha is not None:
            # Numerically-stable gated form: keep the decay as bounded ratios
            # a_t/a_s (<=1 for s<=t) inside the matmuls, never the 1/a rescale.
            la = torch.cumsum(torch.log(alpha[:, :, c].clamp_min(1e-6)), dim=-1)
            a = torch.exp(la)  # cumprod of the gate, in (0,1]
            ratio = torch.exp((la[..., :, None] - la[..., None, :]).clamp_max(0.0))
            M = torch.tril(Bc[..., None] * ratio * KKt, -1)
            rhs = Bc[..., None] * (Vc - a[..., None] * Sk)
            U = torch.linalg.solve_triangular(
                eye + M, rhs, upper=False, unitriangular=True
            )
            Oc = (
                a[..., None] * (Qc @ S.transpose(-1, -2))
                + torch.tril(ratio * QKt, 0) @ U
            )
            ratioC = torch.exp((la[..., -1:] - la).clamp_max(0.0))  # a_C / a_s
            S = (
                a[..., -1][..., None, None] * S
                + (ratioC[..., None] * U).transpose(-1, -2) @ Kc
            )
        else:
            M = torch.tril(Bc[..., None] * KKt, -1)
            rhs = Bc[..., None] * (Vc - Sk)
            U = torch.linalg.solve_triangular(
                eye + M, rhs, upper=False, unitriangular=True
            )
            Oc = Qc @ S.transpose(-1, -2) + torch.tril(QKt, 0) @ U
            S = S + U.transpose(-1, -2) @ Kc
        outs.append(Oc)
    o = torch.cat(outs, dim=2)[:, :, :T]
    return o.to(in_dtype)


class DeltaNetMixerChunked(nn.Module):
    """Same parametrisation as DeltaNetMixer but the exact chunked kernel."""

    def __init__(self, cfg: TrellisConfig, gated: bool, chunk_size: int = 16):
        super().__init__()
        self.cfg, self.gated, self.chunk_size = cfg, gated, chunk_size
        H, Dh, d = cfg.n_heads, cfg.d_head, cfg.d_model
        self.H, self.D = H, Dh
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * Dh, bias=False)
        self.k_proj = nn.Linear(d, H * Dh, bias=False)
        self.v_proj = nn.Linear(d, H * Dh, bias=False)
        self.beta_proj = nn.Linear(d, H, bias=True)
        if gated:
            self.alpha_proj = nn.Linear(d, H, bias=True)
        self.out_proj = nn.Linear(H * Dh, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def _heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.D).permute(0, 2, 1, 3)

    def forward(self, x, training: bool = True):
        B, T, d = x.shape
        h = self.norm(x)
        q = self._heads(self.q_proj(h))
        k = F.normalize(self._heads(self.k_proj(h)), dim=-1)
        v = self._heads(self.v_proj(h))
        beta = torch.sigmoid(self.beta_proj(h)).permute(0, 2, 1)
        alpha = (
            torch.sigmoid(self.alpha_proj(h)).permute(0, 2, 1) if self.gated else None
        )
        y = chunked_delta_rule(q, k, v, beta, alpha, self.chunk_size)
        y = y.permute(0, 2, 1, 3).reshape(B, T, self.H * self.D)
        return self.drop(self.out_proj(y))
