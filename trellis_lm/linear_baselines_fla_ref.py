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

Short-conv kernel. Older fla builds prefer the `causal-conv1d` CUDA kernel; when
that extension is absent they raise unless `use_fast_conv1d=False`. Newer fla
builds expose a `backend` argument and default to Triton, which does not require
`causal-conv1d`. The wrapper preserves that default unless an experiment
explicitly sets `TRELLIS_FLA_REF_CONV_BACKEND`.

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

    # Keep the older use_fast_conv1d=False fallback, but do not invent a
    # backend name for newer FLA builds.  FLA 0.4.x already defaults to Triton;
    # setting backend="torch" makes the reference row unavailable.
    _ORIG_SHORTCONV_INIT = _fla_conv.ShortConvolution.__init__

    def _shortconv_init_cpu_safe(self, *a, **kw):
        backend = os.environ.get("TRELLIS_FLA_REF_CONV_BACKEND")
        params = inspect.signature(_ORIG_SHORTCONV_INIT).parameters
        if backend:
            if "backend" in params:
                kw["backend"] = backend
            elif backend not in ("torch", "fallback", "cpu_safe"):
                kw["use_fast_conv1d"] = True
        elif "backend" not in params:
            kw["use_fast_conv1d"] = False
        return _ORIG_SHORTCONV_INIT(self, *a, **kw)

    _fla_conv.ShortConvolution.__init__ = _shortconv_init_cpu_safe

    from fla.layers import DeltaNet, GatedDeltaNet

    try:
        from fla.layers import GatedDeltaProduct
    except Exception:  # pragma: no cover - older fla without DeltaProduct
        GatedDeltaProduct = None

    HAS_FLA_REF = True
except Exception:  # pragma: no cover - non-fla host
    HAS_FLA_REF = False
    GatedDeltaProduct = None


def _fla_ref_short_conv() -> bool:
    """Short-conv is DISABLED by default for the mechanism comparison (the
    recurrence is the isolated variable). Set TRELLIS_FLA_REF_SHORT_CONV=1 for
    the author-default appendix row."""
    return os.environ.get("TRELLIS_FLA_REF_SHORT_CONV", "0") == "1"


def _construct_fla(cls, *, critical=(), gate=None, **want):
    """Instantiate an fla layer while respecting the INSTALLED signature.

    fla layers accept **kwargs and would silently swallow a mis-named argument,
    so a fidelity-critical setting (num_householder, allow_neg_eigval, ...) could
    be dropped and the row would degrade with no error. This resolves the gate
    kwarg to whichever the class accepts (`use_output_gate` on DeltaProduct vs
    `use_gate` on Gated-DeltaNet), drops kwargs the class does not declare, and
    RAISES if any name in `critical` is not in the signature. The effective
    kwargs are stashed on the module for the on-pod fidelity audit.
    """
    params = inspect.signature(cls.__init__).parameters
    if gate is not None:
        if "use_output_gate" in params:
            want["use_output_gate"] = gate
        elif "use_gate" in params:
            want["use_gate"] = gate
        else:
            raise RuntimeError(f"{cls.__name__}: no output-gate kwarg in signature")
    for name in critical:
        if name not in params:
            raise RuntimeError(
                f"{cls.__name__}: fidelity-critical kwarg '{name}' absent from "
                f"installed signature -- refusing to run a silently degraded row"
            )
    effective = {k: v for k, v in want.items() if k in params}
    dropped = {k: v for k, v in want.items() if k not in params}
    layer = cls(**effective)
    layer._fla_effective_kwargs = effective
    layer._fla_dropped_kwargs = dropped
    return layer


class FLARefMixer(nn.Module):
    """Wrap fla's reference DeltaNet / GatedDeltaNet as a ladder mixer.

    Pre-norms with the ladder's RMSNorm, runs the reference layer (which owns its
    own q/k/v/beta/(g) projections, short conv, qk-norm, (gate), and output
    projection), and returns the mixer delta -- the block adds the residual.
    """

    def __init__(self, cfg: TrellisConfig, gated: bool, num_householder: int = 1):
        super().__init__()
        if not HAS_FLA_REF:
            raise RuntimeError(
                "fla reference layers unavailable; use delta/gated_delta"
            )
        self.cfg = cfg
        self.gated = gated
        self.num_householder = int(num_householder)
        H, D, d = cfg.n_heads, cfg.d_head, cfg.d_model
        self.H, self.D = H, D
        self.norm = RMSNorm(d)
        short_conv = _fla_ref_short_conv()
        if self.num_householder > 1:
            # DeltaProduct: n_h Householder products / token -> diagonal+rank-n_h
            # transition. Needs beta in (0,2) (allow_neg_eigval) or state-tracking
            # fails. UNTESTED on W7900 (no FLA); validated on the H100 pod.
            if GatedDeltaProduct is None:
                raise RuntimeError(
                    "fla GatedDeltaProduct unavailable; upgrade flash-linear-attention"
                )
            # num_householder + allow_neg_eigval ARE the DeltaProduct mechanism;
            # fail loud if the installed fla doesn't accept them (2a).
            self.layer = _construct_fla(
                GatedDeltaProduct,
                critical=("num_householder", "allow_neg_eigval"),
                gate=True,
                hidden_size=d,
                num_heads=H,
                head_dim=D,
                expand_v=1.0,
                num_householder=self.num_householder,
                use_short_conv=short_conv,
                conv_size=cfg.conv_kernel,
                allow_neg_eigval=True,
            )
        elif gated:
            # head_dim fixes the per-head key/value width; expand_v=1 keeps the
            # value width equal to the key width (matched to delta/dense/Trellis,
            # not the fla default expand_v=2 which would widen value state).
            self.layer = _construct_fla(
                GatedDeltaNet,
                gate=True,
                hidden_size=d,
                num_heads=H,
                head_dim=D,
                expand_v=1.0,
                use_short_conv=short_conv,
                conv_size=cfg.conv_kernel,
            )
        else:
            # expand_k=expand_v=1.0 -> head_k_dim=head_v_dim=d/H=D (== d_head).
            self.layer = _construct_fla(
                DeltaNet,
                gate=False,
                hidden_size=d,
                num_heads=H,
                expand_k=1.0,
                expand_v=1.0,
                use_short_conv=short_conv,
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


def build_linear_baseline_ref(
    cfg: TrellisConfig, gated: bool, num_householder: int = 1
):
    """LM whose mixer is the fla REFERENCE DeltaNet / GatedDeltaNet /
    GatedDeltaProduct layer.

    Mirrors `linear_baselines.build_linear_baseline` exactly (same block, head,
    init, forward, loss) but swaps the minimal fla mixer for the reference layer.
    num_householder>1 selects GatedDeltaProduct.
    """
    from .model import SwiGLU, _LMBase
    import torch.nn.functional as F

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = FLARefMixer(cfg, gated, num_householder=num_householder)
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
