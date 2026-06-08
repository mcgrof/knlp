"""fla-backed DeltaNet / Gated-DeltaNet mixers — the SOTA Triton chunked-scan
kernels (flash-linear-attention) for the linear baselines.

Motivation: our hand-rolled `chunked_delta_rule` carries a Python chunk loop
with a `solve_triangular` per chunk, which benchmarks ~100x slower than fla's
Triton kernel at core-op level on an A100 (delta 80.9k vs 7,770k tok/s; gated
54.4k vs 5,818k). The matched-token scaling ladder spends most of its wall clock
in the linear baselines; using fla brings them to near-dense speed so the ladder
(and the 125M paper-repro) is affordable, and so the long pole becomes Trellis
(which has no parallel scan) rather than our slow baseline kernel.

fla is only present on the CUDA pods (torch 2.6 + triton 3.2 + fla-core 0.4.2).
On monster/prune (no fla) `build_linear_baseline` falls back to the bmm
`DeltaNetMixerChunked`, so the same code runs everywhere — only the speed
differs, never the math family.

Convention match to our existing baselines: keys L2-normalised (fla's
`use_qk_l2norm_in_kernel=True`), beta = sigmoid(beta_proj) in (0,1), and for the
gated variant the forget gate is log-space g = logsigmoid(alpha_proj) <= 0 so
a_t = exp(g_t) = sigmoid(alpha_proj) in (0,1), identical to the `alpha` of
`DeltaNetMixerChunked`. fla expects [B,T,H,D] (head-second), not our [B,H,T,D].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .trellis_mixer import RMSNorm

try:  # only available on the CUDA pods
    from fla.ops import chunk_delta_rule, chunk_gated_delta_rule

    HAS_FLA = True
except Exception:  # pragma: no cover - monster/prune path
    HAS_FLA = False


class FLADeltaNetMixer(nn.Module):
    """DeltaNet / Gated-DeltaNet via fla's chunked-scan Triton kernel."""

    def __init__(self, cfg: TrellisConfig, gated: bool):
        super().__init__()
        if not HAS_FLA:
            raise RuntimeError("fla not installed; use DeltaNetMixerChunked")
        self.cfg = cfg
        self.gated = gated
        H, D, d = cfg.n_heads, cfg.d_head, cfg.d_model
        self.H, self.D = H, D
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, H * D, bias=False)
        self.v_proj = nn.Linear(d, H * D, bias=False)
        self.beta_proj = nn.Linear(d, H, bias=True)
        if gated:
            self.alpha_proj = nn.Linear(d, H, bias=True)
        self.out_proj = nn.Linear(H * D, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def _heads(self, x):  # [B,T,H*D] -> [B,T,H,D] (fla layout)
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.D)

    def forward(self, x, training: bool = True):
        B, T, d = x.shape
        h = self.norm(x)
        # fla's chunked-scan kernel is bf16/fp16 only; the rest of the ladder may
        # run fp32 (Trellis needs it). Cast q/k/v/beta/g to bf16 for the fla op
        # and bring the output back to x.dtype. bf16 is the standard/faithful
        # precision for DeltaNet/GatedDeltaNet anyway (matches the fla paper).
        bf = torch.bfloat16
        q = self._heads(self.q_proj(h)).to(bf)
        k = self._heads(self.k_proj(h)).to(bf)
        v = self._heads(self.v_proj(h)).to(bf)
        beta = torch.sigmoid(self.beta_proj(h)).to(bf)  # [B,T,H] in (0,1)
        if self.gated:
            g = F.logsigmoid(self.alpha_proj(h).float()).to(bf)  # log-decay <=0
            o = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                use_qk_l2norm_in_kernel=True,
                head_first=False,
            )
        else:
            o = chunk_delta_rule(
                q,
                k,
                v,
                beta,
                use_qk_l2norm_in_kernel=True,
                head_first=False,
            )
        if isinstance(o, tuple):
            o = o[0]
        y = o.reshape(B, T, self.H * self.D).to(x.dtype)
        return self.drop(self.out_proj(y))
