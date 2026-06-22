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


def kbias_per_head(info):
    """Per-(kv head, channel) K-bias [n_kv, head_dim], for contiguous AND interleaved fused QKV
    (GPT-NeoX). None if biasless. Pass-through channels [rotary_dim:] are RoPE-invariant so the raw
    bias there equals its post-RoPE-interface contribution (used by passthrough_prebias + the
    bias-L2-mass split)."""
    hd = info["head_dim"]
    if info.get("fused_interleaved"):
        qkv = info.get("qkv_proj")
        if qkv is None or qkv.bias is None:
            return None
        nh = info["n_q_heads"]  # GPT-NeoX is MHA (n_kv == n_q)
        return qkv.bias.detach().view(nh, 3 * hd)[:, hd : 2 * hd].clone()
    kb = info.get("k_bias")
    if kb is None:
        return None
    return kb.detach().view(info["n_kv_heads"], hd).clone()


class SubspaceKHarness:
    """Post-RoPE K-FP8 restricted to a subspace. mode: 'full' | 'rotary_only' | 'passthrough_only' |
    'passthrough_prebias'. V left bf16. rotary_dim splits the head; pass-through is [rotary_dim:].

    passthrough_prebias quantizes ONLY the pass-through channels AFTER subtracting their K-bias --
    the cell that breaks the bias/subspace confound: if passthrough_only is high but
    passthrough_prebias is low, the fragility is a BIAS sitting in the un-rotated tail, not the
    pass-through distribution. pass_bias_by_layer: {layer_idx: [n_kv, n_passthrough]} or None.
    """

    def __init__(
        self, model, infos, rotary_dim, mode, pass_bias_by_layer=None, mask=None
    ):
        self.model = model
        self.rd = rotary_dim
        self.mode = mode
        self.pass_bias = pass_bias_by_layer or {}
        # mask: bool [head_dim] selecting which channels to FP8 ('mask_only') -- used for the random
        # 32/48 partition control (a random same-size channel set vs the true pass-through tail).
        self.mask = mask
        self.impl = model.config._attn_implementation
        self.by_mod = {id(i["attn_module"]): i for i in infos}
        self.orig = None

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        orig = self.orig
        h = self

        def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
            info = h.by_mod.get(id(module))
            if info is not None:
                rot, pas = k[..., : h.rd], k[..., h.rd :]
                if h.mode == "full":
                    k = _q_fp8(k)
                elif h.mode == "rotary_only":
                    k = torch.cat([_q_fp8(rot), pas], dim=-1)
                elif h.mode == "passthrough_only":
                    k = torch.cat([rot, _q_fp8(pas) if pas.numel() else pas], dim=-1)
                elif h.mode == "passthrough_prebias":
                    pb = h.pass_bias.get(info["layer_idx"])
                    if pas.numel() and pb is not None:
                        b = pb.to(k.dtype).to(k.device)[None, :, None, :]
                        pas = _q_fp8(pas - b) + b
                    elif pas.numel():
                        pas = _q_fp8(pas)
                    k = torch.cat([rot, pas], dim=-1)
                elif h.mode == "mask_only" and h.mask is not None:
                    m = h.mask.to(k.device)
                    kq = _q_fp8(k)
                    k = torch.where(
                        m, kq, k
                    )  # FP8 only the masked channels, rest native
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


def control_bias(true_bias, kind, seed=0):
    """Build a control bias from the true K-bias [n*hd]: 'true' | 'zero' | 'random_same_norm' |
    'permuted'. random_same_norm and permuted preserve the L2 norm but destroy the per-channel
    placement -- the negative controls that prove pre-bias recovery is BIAS-SPECIFIC, not just any
    same-norm subtraction. Deterministic via a seeded generator."""
    # build on CPU (the seeded CPU Generator forces CPU tensors); the harness moves the result to
    # the model's device. Doing the norm/permute on a CUDA bias with CPU-generated indices/noise
    # would be a device mismatch -- a GPU-only bug the CPU self-test cannot see.
    b = true_bias.detach().float().cpu()
    if kind == "true":
        return b.clone()
    if kind == "zero":
        return torch.zeros_like(b)
    g = torch.Generator().manual_seed(seed)
    if kind == "random_same_norm":
        r = torch.randn(b.shape, generator=g)
        return r * (b.norm() / r.norm().clamp(min=1e-8))
    if kind == "permuted":
        return b[torch.randperm(b.numel(), generator=g)]
    raise ValueError(kind)


class ControlledPrebiasHarness:
    """Pre-bias FP8-K that subtracts a SPECIFIED bias vector (not the model's true bias), then adds
    it back: quant(K - b_ctrl) + b_ctrl at the k_proj / contiguous-fused-K output (pre-RoPE). The
    negative control for C2 -- if a random-same-norm or channel-permuted bias recovers as well as the
    true bias, the 'pre-bias fix' is not bias-specific. bias_by_layer: {layer_idx: [n_kv*hd]}.
    (Separate k_proj and contiguous fused QKV; NOT interleaved GPT-NeoX.)"""

    def __init__(self, model, infos, bias_by_layer, k_spec):
        self.model = model
        self.infos = infos
        self.bias = bias_by_layer
        self.k = k_spec
        self.handles = []

    def install(self):
        for info in self.infos:
            b = self.bias.get(info["layer_idx"])
            if b is None:
                continue
            if info.get("fused"):
                proj, sl = info.get("qkv_proj"), info.get("k_slice")
            else:
                proj, sl = info.get("k_proj"), None
            if proj is None:
                continue
            self.handles.append(proj.register_forward_hook(self._mk(sl, b)))
        return self

    def _mk(self, sl, b):
        spec = self.k

        @torch.no_grad()
        def hook(module, inp, out):
            if sl is None:
                bb = b.to(out.dtype).to(out.device)
                return (
                    kbc._quant_lastdims(
                        out - bb,
                        spec["fmt"],
                        spec["bits"],
                        spec["layout"],
                        spec["group"],
                        False,
                    )
                    + bb
                )
            s0, s1 = sl
            ks = out[..., s0:s1]
            bb = b.to(ks.dtype).to(ks.device)
            out = out.clone()
            out[..., s0:s1] = (
                kbc._quant_lastdims(
                    ks - bb,
                    spec["fmt"],
                    spec["bits"],
                    spec["layout"],
                    spec["group"],
                    False,
                )
                + bb
            )
            return out

        return hook

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()
