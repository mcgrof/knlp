"""Mechanism discriminators for the FP8 KV-cache failure: is a model's K8 collapse driven by the
K-BIAS (a per-channel DC offset) or by the partial-RoPE PASS-THROUGH SUBSPACE (a different K
distribution across the un-rotated channels)? Two probes, both at the validated quant points:

  - SubspaceKHarness: split post-RoPE K into rotary (first rotary_dim) vs pass-through channels and
    FP8-quantize ONE subspace at a time. If pass-through-only K8 collapses while rotary-only stays
    tolerant, the pass-through distribution carries the fragility (partial-RoPE mechanism).
  - prebias recovery: subtract the K-bias, FP8 the residual, add it back. High recovery => bias
    driven. For GPT-NeoX the fused QKV is interleaved per-head so the contiguous-slice prebias does
    not apply -- InterleavedPrebiasHarness reshapes [.., n_head, 3*hd] and addresses the K slab
    [hd:2hd] (layout verified empirically: setting that slab moves K, not Q/V).

GPU-free verifiable on tiny CPU models (see run_mechanism --self-test).
"""

import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402


def _q_fp8(x):
    return kbc._quant_lastdims(x, "fp8", 8, "per_tensor", 128, False)


class SubspaceKHarness:
    """Post-RoPE K-FP8 restricted to a subspace. mode: 'full' | 'rotary_only' | 'passthrough_only'.
    V left bf16. rotary_dim splits the head; pass-through is channels [rotary_dim:]."""

    def __init__(self, model, infos, rotary_dim, mode):
        self.model = model
        self.rd = rotary_dim
        self.mode = mode
        self.impl = model.config._attn_implementation
        self.by_mod = {id(i["attn_module"]): i for i in infos}
        self.orig = None

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        orig = self.orig
        h = self

        def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
            if id(module) in h.by_mod:
                rot, pas = k[..., : h.rd], k[..., h.rd :]
                if h.mode == "full":
                    k = _q_fp8(k)
                elif h.mode == "rotary_only":
                    k = torch.cat([_q_fp8(rot), pas], dim=-1)
                elif h.mode == "passthrough_only":
                    k = torch.cat([rot, _q_fp8(pas) if pas.numel() else pas], dim=-1)
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


class InterleavedPrebiasHarness:
    """Pre-bias FP8-K for GPT-NeoX-style interleaved fused QKV. Hooks query_key_value: reshape the
    output to [.., n_head, 3*head_dim], subtract the per-head K-bias from the K slab [hd:2hd],
    FP8-quantize that residual, add the bias back -- the pre-RoPE faithful fix, layout-correct for
    the interleaved packing (verified: the K slab is [hd:2hd] per head). V left bf16."""

    def __init__(self, model, infos):
        self.model = model
        self.infos = infos
        self.handles = []

    def install(self):
        for info in self.infos:
            qkv = info.get("qkv_proj")
            if qkv is None or qkv.bias is None:
                continue
            nh, hd = info["n_q_heads"], info["head_dim"]
            kbias = (
                qkv.bias.detach().view(nh, 3 * hd)[:, hd : 2 * hd].clone()
            )  # [nh, hd]
            self.handles.append(qkv.register_forward_hook(self._mk_hook(nh, hd, kbias)))
        return self

    @staticmethod
    def _mk_hook(nh, hd, kbias):
        @torch.no_grad()
        def hook(module, inp, out):
            shp = out.shape
            o = out.view(*shp[:-1], nh, 3 * hd)
            k = o[..., hd : 2 * hd]
            b = kbias.to(k.dtype).to(k.device)
            kq = _q_fp8(k - b) + b
            o = o.clone()
            o[..., hd : 2 * hd] = kq
            return o.view(*shp)

        return hook

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()
