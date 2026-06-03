"""TrellisMixer — two-pass bounded-memory sequence mixing sublayer.

Replaces self-attention. Produces its output directly from the compressed
memory state (no [B,H,T,T] mask). Returns the sublayer delta; the block adds
the residual.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .activations import get_activation
from .trellis_memory import run_trellis_memory


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.w


class CausalDWConv1d(nn.Module):
    """Depthwise causal conv over time on a [B,H,T,D] tensor (per-channel)."""

    def __init__(self, channels, kernel):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=True)

    def forward(self, x):  # x: [B,H,T,D]
        B, H, T, D = x.shape
        xt = x.permute(0, 1, 3, 2).reshape(B, H * D, T)     # [B, C=H*D, T]
        xt = F.pad(xt, (self.kernel - 1, 0))                # left pad = causal
        out = self.conv(xt)                                 # [B, C, T]
        out = out.reshape(B, H, D, T).permute(0, 1, 3, 2)   # [B,H,T,D]
        return out


class TrellisMixer(nn.Module):
    def __init__(self, cfg: TrellisConfig):
        super().__init__()
        self.cfg = cfg
        H, D, M, d = cfg.n_heads, cfg.d_head, cfg.n_slots, cfg.d_model
        self.H, self.D, self.M = H, D, M
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, H * D, bias=False)
        self.v_proj = nn.Linear(d, H * D, bias=False)
        self.alpha_proj = nn.Linear(d, H * M, bias=False)
        beta_out = H if cfg.beta_mode == "scalar_per_head" else H * M
        self.beta_proj = nn.Linear(d, beta_out, bias=True)
        # gamma positive per head via softplus(raw); init so softplus(raw)=gamma_init
        raw0 = math.log(math.expm1(cfg.gamma_init))
        self.gamma_raw = nn.Parameter(torch.full((H,), raw0))
        self.out_proj = nn.Linear(H * D, d, bias=False)
        self.post_gate = cfg.post_gate
        if cfg.post_gate:
            self.gate_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.use_conv = cfg.use_short_conv_qk
        if self.use_conv:
            self.q_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
            self.k_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
        self.phi = get_activation(cfg.activation)
        self.f = get_activation(cfg.activation)
        self.alpha_act = get_activation(cfg.alpha_mode)

    def _heads(self, x, last):  # [B,T,H*last] -> [B,H,T,last]
        B, T, _ = x.shape
        return x.view(B, T, self.H, last).permute(0, 2, 1, 3)

    def forward(self, x, training: bool = True):
        cfg = self.cfg
        B, T, d = x.shape
        h = self.norm(x)
        q = self._heads(self.q_proj(h), self.D)   # [B,H,T,D]
        k = self._heads(self.k_proj(h), self.D)
        v = self._heads(self.v_proj(h), self.D)
        if self.use_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
        alpha = self._heads(self.alpha_proj(h), self.M)        # [B,H,T,M]
        alpha = self.alpha_act(alpha)
        if cfg.beta_mode == "scalar_per_head":
            beta = torch.sigmoid(self.beta_proj(h)).view(B, T, self.H, 1).permute(0, 2, 1, 3)
        else:
            beta = torch.sigmoid(self._heads(self.beta_proj(h), self.M))  # [B,H,T,M]
        if not cfg.forget_gate:
            beta = torch.ones_like(beta)
        gamma = F.softplus(self.gamma_raw)        # [H], positive

        # key pass: write=k, read=q -> yhat [B,H,T,M]
        yhat = run_trellis_memory(k, q, alpha, beta, gamma, self.phi, "M_q", training)
        r = self.f(yhat)                          # [B,H,T,M]
        # value pass: write=v, read=r -> y [B,H,T,D]
        y = run_trellis_memory(v, r, alpha, beta, gamma, self.phi, "M_T_r", training)

        y = y.permute(0, 2, 1, 3).reshape(B, T, self.H * self.D)   # merge heads
        out = self.out_proj(y)
        if self.post_gate:
            out = out * F.silu(self.gate_proj(h))
        return self.drop(out)
