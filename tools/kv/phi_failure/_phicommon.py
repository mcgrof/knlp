"""Phi-failure shared helpers: import the tier-1 core, detect partial-RoPE, split rotary vs
pass-through subspaces. Phi-2 uses partial rotary (only the first rotary_dim head channels are
rotated); the rest pass through. That mixed K distribution is a prime FP8-failure suspect, so we
need to measure and quantize the two subspaces separately."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tier2_bias_fp8"
    ),
)
import k_bias_common as kbc  # noqa: E402
import _t2common as t2  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def rotary_split(model):
    """Return (rotary_dim, head_dim). rotary_dim==head_dim means full RoPE (no pass-through)."""
    cfg = model.config
    n_q = getattr(cfg, "num_attention_heads")
    hidden = getattr(cfg, "hidden_size")
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_q)
    prf = getattr(cfg, "partial_rotary_factor", None)
    rotary_dim = getattr(cfg, "rotary_dim", None)
    if rotary_dim is None and prf is not None:
        rotary_dim = int(head_dim * prf)
    if rotary_dim is None:
        rotary_dim = head_dim  # full RoPE
    return int(rotary_dim), int(head_dim)


def has_qk_norm(model):
    cfg = model.config
    return bool(
        getattr(cfg, "qk_layernorm", False) or getattr(cfg, "use_qk_norm", False)
    )


def stats(x):
    """x: flattenable tensor. amax/p99/p999/p95/rms/std/outlier-count."""
    a = x.float().abs().flatten()
    s = a.sort().values
    n = s.numel()

    def pct(p):
        return s[min(n - 1, int(p * n))].item()

    rms = x.float().pow(2).mean().sqrt().item()
    p99 = pct(0.99)
    amax = a.max().item()
    return dict(
        amax=amax,
        p999=pct(0.999),
        p99=p99,
        p95=pct(0.95),
        rms=rms,
        std=x.float().std().item(),
        max_over_p99=amax / max(p99, 1e-9),
        p999_over_p99=pct(0.999) / max(p99, 1e-9),
        outlier_count=int((a > 4 * rms).sum().item()),
    )


class SubspaceKHarness:
    """Quantize post-RoPE K with rotary/pass-through awareness. mode:
      'split_scale'     -- separate FP8 scale for rotary vs pass-through subspace
      'rotary_only'     -- quantize only the rotary subspace (pass-through BF16)
      'passthrough_only'-- quantize only the pass-through subspace (rotary BF16)
    V quantized per v_spec. Isolates the partial-RoPE mixed-distribution hypothesis."""

    def __init__(self, model, infos, rotary_dim, mode, v_spec="fp8:per_tensor"):
        self.model = model
        self.infos = infos
        self.rd = rotary_dim
        self.mode = mode
        self.v = kbc.parse_spec(v_spec)
        self.impl = model.config._attn_implementation
        self.by_mod = {id(i["attn_module"]): i for i in infos}
        self.orig = None

    def _q(self, x):
        return kbc._quant_lastdims(x, "fp8", 8, "per_tensor", 128, False)

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        orig = self.orig
        h = self

        def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
            if id(module) in h.by_mod:
                rot, pas = k[..., : h.rd], k[..., h.rd :]
                if h.mode == "split_scale":
                    k = torch.cat(
                        [h._q(rot), h._q(pas) if pas.numel() else pas], dim=-1
                    )
                elif h.mode == "rotary_only":
                    k = torch.cat([h._q(rot), pas], dim=-1)
                elif h.mode == "passthrough_only":
                    k = torch.cat([rot, h._q(pas) if pas.numel() else pas], dim=-1)
                if h.v["fmt"] is not None:
                    v = kbc._quant_lastdims(
                        v, h.v["fmt"], h.v["bits"], h.v["layout"], h.v["group"], False
                    )
            return orig(
                module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
            )

        ALL_ATTENTION_FUNCTIONS[self.impl] = hook
        return self

    def remove(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self.orig is not None:
            ALL_ATTENTION_FUNCTIONS[self.impl] = self.orig
            self.orig = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()
