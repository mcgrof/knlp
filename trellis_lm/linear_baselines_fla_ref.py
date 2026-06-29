"""Kernel-fair DeltaNet / Gated-DeltaNet baselines using fla's REFERENCE layers.

Why this exists. `linear_baselines_fla.FLADeltaNetMixer` wraps the fla Triton
chunked-scan op directly, but in a *minimal* mixer: q/k/v/beta projections, L2
keys, log-sigmoid gate, and nothing else. The published DeltaNet / Gated
DeltaNet architectures carry three extra components our minimal mixer drops: a
depthwise short causal convolution on q/k/v (conv_size 4), a learned qk
normalisation, and -- for the gated variant -- an output gate. Those components
are part of what the baseline *is*; benchmarking Trellis against a stripped-down
cousin would flatter Trellis. So the headline baselines (`delta_ref`,
`gated_delta_ref`) use fla's own `fla.layers.DeltaNet` / `fla.layers.GatedDeltaNet`
reference modules -- the exact architecture the fla authors ship -- so nobody can
argue the cousin was crippled. The minimal `delta`/`gated_delta` kinds remain as
a Tier-B speed/ablation reference.

Width is matched to the rest of the ladder: `make_cfg` always sets
n_heads=d_model//64 and d_head=64, so inner_dim == d_model and expand_k =
expand_v = 1.0. The reference layers therefore differ from the minimal mixer and
from dense/Trellis ONLY by the conv + qk-norm + (gate) components, never by head
width -- the architecture is the isolated variable.

Short-conv kernel. fla's `ShortConvolution` prefers the `causal-conv1d` CUDA
kernel; when that extension is absent it raises unless `use_fast_conv1d=False`.
The PyTorch fallback computes the identical depthwise causal convolution (just
without the fused kernel), so we force the fallback at import time. This changes
conv *speed*, never conv *math* -- the architecture stays faithful.

fla is CUDA-only (Triton). On a host without fla, importing this module raises;
the ladder only references the `*_ref` kinds on the CUDA pod.
"""

from __future__ import annotations

import inspect
import os
import torch
import torch.nn as nn

from .config import TrellisConfig
from .trellis_mixer import RMSNorm

try:  # fla reference layers live in the full flash-linear-attention package
    import fla.modules.convolution as _fla_conv

    # Default to the PyTorch short-conv fallback (identical math; no
    # causal-conv1d CUDA build required), but allow paid CUDA pods to opt into a
    # faster FLA backend after a preflight.  Keep this as an import-time patch so
    # the reference layers' internally constructed ShortConvolution instances
    # pick it up.
    _ORIG_SHORTCONV_INIT = _fla_conv.ShortConvolution.__init__

    def _shortconv_init_cpu_safe(self, *a, **kw):
        backend = os.environ.get("TRELLIS_FLA_REF_CONV_BACKEND")
        params = inspect.signature(_ORIG_SHORTCONV_INIT).parameters
        if backend:
            if "backend" in params:
                kw["backend"] = backend
            elif backend not in ("torch", "fallback", "cpu_safe"):
                kw["use_fast_conv1d"] = True
        elif "backend" in params:
            kw.setdefault("backend", "torch")
        else:
            kw["use_fast_conv1d"] = False
        return _ORIG_SHORTCONV_INIT(self, *a, **kw)

    _fla_conv.ShortConvolution.__init__ = _shortconv_init_cpu_safe

    from fla.layers import DeltaNet, GatedDeltaNet

    HAS_FLA_REF = True
except Exception:  # pragma: no cover - non-fla host
    HAS_FLA_REF = False


class FLARefMixer(nn.Module):
    """Wrap fla's reference DeltaNet / GatedDeltaNet as a ladder mixer.

    Pre-norms with the ladder's RMSNorm, runs the reference layer (which owns its
    own q/k/v/beta/(g) projections, short conv, qk-norm, (gate), and output
    projection), and returns the mixer delta -- the block adds the residual.
    """

    def __init__(self, cfg: TrellisConfig, gated: bool):
        super().__init__()
        if not HAS_FLA_REF:
            raise RuntimeError(
                "fla reference layers unavailable; use delta/gated_delta"
            )
        self.cfg = cfg
        self.gated = gated
        H, D, d = cfg.n_heads, cfg.d_head, cfg.d_model
        self.H, self.D = H, D
        self.norm = RMSNorm(d)
        if gated:
            # head_dim fixes the per-head key/value width; expand_v=1 keeps the
            # value width equal to the key width (matched to delta/dense/Trellis,
            # not the fla default expand_v=2 which would widen value state).
            self.layer = GatedDeltaNet(
                hidden_size=d,
                num_heads=H,
                head_dim=D,
                expand_v=1.0,
                use_gate=True,
                use_short_conv=True,
                conv_size=cfg.conv_kernel,
            )
        else:
            # expand_k=expand_v=1.0 -> head_k_dim=head_v_dim=d/H=D (== d_head).
            self.layer = DeltaNet(
                hidden_size=d,
                num_heads=H,
                expand_k=1.0,
                expand_v=1.0,
                use_gate=False,
                use_short_conv=True,
                conv_size=cfg.conv_kernel,
                qk_norm="l2",
            )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, training: bool = True):
        h = self.norm(x)
        # fla's chunk(_gated)_delta_rule kernel asserts bf16/fp16 (no fp32). Run
        # the whole reference layer under a bf16 autocast so its projections and
        # the Triton scan compute in bf16 while the fp32 MASTER weights stay
        # fp32 -- identical precision treatment to the minimal FLADeltaNetMixer,
        # which also always runs the kernel in bf16 (the faithful DeltaNet
        # precision). A no-op when the ladder already runs under --amp-bf16.
        with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16):
            out = self.layer(h)
        y = out[0] if isinstance(out, tuple) else out
        return self.drop(y.to(x.dtype))

    def state_bytes_per_seq(self) -> int:
        """Recurrent (fast-weight) state bytes for ONE sequence, bf16.

        DeltaNet/GatedDeltaNet keep a per-head [head_k_dim, head_v_dim] matrix.
        With expand_k=expand_v=1.0 and head_dim=D that is H * D * D elements.
        """
        return self.H * self.D * self.D * 2


def build_linear_baseline_ref(cfg: TrellisConfig, gated: bool):
    """LM whose mixer is the fla REFERENCE DeltaNet / GatedDeltaNet layer.

    Mirrors `linear_baselines.build_linear_baseline` exactly (same block, head,
    init, forward, loss) but swaps the minimal fla mixer for the reference layer.
    """
    from .model import SwiGLU, _LMBase
    import torch.nn.functional as F

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = FLARefMixer(cfg, gated)
            self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

        def forward(self, x, training=True):
            x = x + self.mixer(x, training=training)
            x = x + self.mlp(x)
            return x

    class LinearBaselineRefLM(_LMBase):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self._init_head()
            self.blocks = nn.ModuleList([_Block() for _ in range(cfg.n_layers)])
            # init only the scaffolding (head/embedding/MLP); leave fla reference
            # layers with their own (paper) initialisation untouched.
            self._init_scaffold()

        def _init_scaffold(self):
            for blk in self.blocks:
                blk.mlp.apply(self._init_linear)
            self._init_linear(self.lm_head)
            if hasattr(self, "wte"):
                nn.init.normal_(self.wte.weight, std=0.02)

        @staticmethod
        def _init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, idx, labels=None, training=None):
            if training is None:
                training = self.training
            x = self.wte(idx)
            for blk in self.blocks:
                x = blk(x, training=training)
            x = self.norm_f(x)
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return logits, loss

        def memory_state_bytes(self, batch_size: int) -> int:
            c = self.cfg
            per_seq = c.n_layers * c.n_heads * c.d_head * c.d_head * 2  # bf16
            return batch_size * per_seq

    return LinearBaselineRefLM()
