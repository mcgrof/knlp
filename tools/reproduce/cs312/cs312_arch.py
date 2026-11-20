#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Hyper-connections, manifold hyper-connections and LayerRoPE on the course d8.

These three were measured on a 32-layer, width-128 TinyStories backbone in
`research/layerrope_beta`. Nothing applied them to the CS312 course model, so
this is a port, and a port is exactly where a silent mismatch would produce
plausible numbers that mean nothing.

Two rules follow from that. The arithmetic is **vendored verbatim** from the
reference (`tile_kernels/torch/mhc.py` and the LayerRoPE author's
`complex_rotate`) rather than re-derived, and `tests/cs312/test_arch.py`
asserts this module reproduces the reference bit-for-bit on random inputs.
Only the plumbing into the course architecture is new.

What each name means here:

* **LayerRoPE** is not a change to positional RoPE. It is a learned depth-wise
  complex gain that replaces the RMSNorm affine weight at the block's norm
  sites: gamma_l = gamma (*)_c exp(r(l) + i theta(l, j)), with
  r(l) = alpha + beta log(l + 1). The course model has no peri-norm, so there
  are two sites, `in_attn` and `in_mlp`, matching the reference when
  `peri_norm` is false.
* **HC** expands the residual stream into `MULT` parallel copies and learns,
  per layer and per site, how to mix them into the branch input and how to
  write the branch output back.
* **mHC** is HC with the carry matrix pushed onto the doubly-stochastic
  manifold by Sinkhorn normalisation. That constraint is the only difference.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

MULT = 4
ROUTE_EPS = 1e-5
RMS_NORM_EPS = 1e-6


# ───────────────────────── vendored reference arithmetic ─────────────────────────
# Copied from tile_kernels/torch/mhc.py and the LayerRoPE author's module.
# test_arch.py checks these against the originals; do not "clean them up".

def expand_to_mhc_ref(hidden, mhc_mult):
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()


def sinkhorn_normalize_ref(x, repeat=10, eps=1e-6):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_norm_fn_ref(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps):
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.flatten(2, 3).float()
    assert mhc_fn.dtype == residual.dtype == torch.float
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        'mbk,nbk->mbn',
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
    return mixes.view(*residual.shape[:2], -1)


def mhc_pre_split_mixes_ref(input_mixes, mhc_scale, mhc_base, mhc_mult,
                            mhc_post_mult_value, mhc_pre_eps):
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat([
        mhc_scale[0].expand(mhc_mult),
        mhc_scale[1].expand(mhc_mult),
        mhc_scale[2].expand(mhc_mult * mhc_mult),
    ])
    input_mixes = input_mixes * mhc_scale + mhc_base
    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (input_mixes[:, :, mhc_mult:2 * mhc_mult].sigmoid()
                      * mhc_post_mult_value).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult:].view(a, b, mhc_mult, mhc_mult)
    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x, mix):
    return (x * mix).sum(-2).bfloat16()


def mhc_post_ref(x, residual, post_layer_mix, comb_res_mix):
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def complex_rotate(x, mag, theta):
    """Multiply the pairs (x[2j] + i x[2j+1]) by exp(mag + i theta[j]), in fp32."""
    x = x.to(torch.float32)
    u, v = x[..., 0::2], x[..., 1::2]
    r = torch.exp(mag.to(torch.float32))
    cos, sin = torch.cos(theta.to(torch.float32)), torch.sin(theta.to(torch.float32))
    return torch.stack([r * (u * cos - v * sin), r * (u * sin + v * cos)], dim=-1).flatten(-2)


# ───────────────────────────────── LayerRoPE ─────────────────────────────────
class LayerRoPE(nn.Module):
    """Depth-wise complex gain, shared across layers, two sites (no peri-norm)."""

    sites = ("in_attn", "in_mlp")

    def __init__(self, hidden_size, alpha=0.0, beta=0.0, base_freq=100.0):
        super().__init__()
        self.base_freq = float(base_freq)
        for s in self.sites:
            setattr(self, f"{s}_alpha", nn.Parameter(torch.tensor(float(alpha))))
            setattr(self, f"{s}_beta", nn.Parameter(torch.tensor(float(beta))))
        for s in self.sites:
            setattr(self, f"{s}_gamma", nn.Parameter(torch.ones(hidden_size)))
        # the intercept is tied to the rotation intercept, as in the bracketed runs
        for s in self.sites:
            setattr(self, f"{s}_alpha_rot", nn.Parameter(torch.tensor(float(alpha))))
            setattr(self, f"{s}_beta_rot", nn.Parameter(torch.tensor(float(beta))))

    def forward(self, site, log_depth):
        def param(name):
            return getattr(self, f"{site}_{name}").to(torch.float32)

        u = torch.as_tensor(log_depth, dtype=torch.float32,
                            device=param("gamma").device)
        mag = param("alpha") + param("beta") * u
        theta = torch.exp(param("alpha_rot") + param("beta_rot") * u)
        gamma = param("gamma")
        d = gamma.shape[0]
        j = torch.arange(d // 2, dtype=torch.float32, device=gamma.device)
        base = torch.tensor(self.base_freq, dtype=torch.float32, device=gamma.device)
        return complex_rotate(gamma, mag, theta * torch.pow(base, -2.0 * j / d))


class GainedRMSNorm(nn.Module):
    """The course RMSNorm with its affine weight supplied from outside."""

    def __init__(self, eps=RMS_NORM_EPS):
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states, gain):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return gain.to(input_dtype) * hidden_states.to(input_dtype)


# ───────────────────────────── hyper-connections ─────────────────────────────
class Routing(nn.Module):
    """Per layer and site: how to read the MULT streams and how to write back."""

    def __init__(self, index, hidden_size, mult=MULT, manifold=True):
        super().__init__()
        self.mult, self.manifold = mult, manifold
        width = mult * hidden_size
        self.phi = nn.Parameter(torch.zeros(2 * mult + mult * mult, width))
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.scale = nn.Parameter(torch.full((3,), 0.01))
        pre = torch.full((mult,), -1.0)
        pre[index % mult] = 1.0
        carry = torch.full((mult, mult), -1.0)
        carry.fill_diagonal_(1.0)
        self.base = nn.Parameter(torch.cat((pre, torch.zeros(mult), carry.flatten())))

    def forward(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            mixes = mhc_pre_norm_fn_ref(state, self.phi, self.norm_weight, ROUTE_EPS)
            pre, post, logits = mhc_pre_split_mixes_ref(
                mixes, self.scale, self.base, self.mult, 2.0, 0.0
            )
            # the manifold constraint is the only thing separating mHC from HC
            carry = (sinkhorn_normalize_ref(logits, repeat=20, eps=1e-6)
                     if self.manifold else logits)
        return pre, post, carry


# ───────────────────────────── the patched model ─────────────────────────────
def _forward_with_arch(self, input_ids, attention_mask=None, position_ids=None):
    """LlamaModel.forward rewritten for LayerRoPE gains and/or MULT streams."""
    batch_size, seq_len = input_ids.shape
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    hidden_states = self.embed_tokens(input_ids)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    routes = getattr(self, "cs312_routes", None)
    lrope = getattr(self, "cs312_layerrope", None)
    # The vendored mHC helpers return bf16, because the reference model is bf16
    # throughout. The course model keeps fp32 parameters, so the reference
    # arithmetic is preserved exactly and only the boundary is cast back.
    work_dtype = hidden_states.dtype
    if routes is not None:
        hidden_states = expand_to_mhc_ref(hidden_states, MULT)

    for idx, layer in enumerate(self.layers):
        log_depth = math.log(idx + 1.0)
        for site_index, site in enumerate(("attn", "mlp")):
            if routes is None:
                branch_input = hidden_states
            else:
                pre, post, carry = routes[2 * idx + site_index](hidden_states)
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    branch_input = mhc_pre_apply_mix_ref(hidden_states, pre).to(work_dtype)
            if site == "attn":
                gain = (lrope("in_attn", log_depth) if lrope is not None
                        else layer.input_layernorm.weight)
                normed = layer.cs312_norm(branch_input, gain)
                branch = layer.self_attn(
                    normed, position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )
            else:
                gain = (lrope("in_mlp", log_depth) if lrope is not None
                        else layer.post_attention_layernorm.weight)
                normed = layer.cs312_norm(branch_input, gain)
                branch = layer.mlp(normed)
            branch = F.dropout(branch, p=layer.dropout, training=layer.training)
            if routes is None:
                hidden_states = hidden_states + branch
            else:
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    hidden_states = mhc_post_ref(
                        branch, hidden_states, post, carry).to(work_dtype)
    if routes is not None:
        hidden_states = hidden_states.float().mean(-2).to(work_dtype)
    return self.norm(hidden_states)


VARIANTS = {
    "baseline":        dict(hc=False, manifold=False, layerrope=False),
    "hc":              dict(hc=True,  manifold=False, layerrope=False),
    "mhc":             dict(hc=True,  manifold=True,  layerrope=False),
    "layerrope":       dict(hc=False, manifold=False, layerrope=True),
    "mhc_layerrope":   dict(hc=True,  manifold=True,  layerrope=True),
}


def build_arch_model(
    model_config,
    *,
    dtype,
    qk_norm,
    tie_word_embeddings,
    dropout,
    variant="baseline",
    layerrope_alpha=0.0,
    layerrope_beta=0.0,
    layerrope_base_freq=100.0,
    **_ignored,
):
    """Course model_builder: the stock model with the named residual variant."""
    from modeling import AutoregressiveLM

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    model = AutoregressiveLM(
        model_config,
        dtype=dtype,
        qk_norm=qk_norm,
        tie_word_embeddings=tie_word_embeddings,
        dropout=dropout,
    )
    if variant == "baseline":
        return model

    inner = model.model
    hidden = model_config.hidden_size
    for layer in inner.layers:
        layer.cs312_norm = GainedRMSNorm(RMS_NORM_EPS).to(next(layer.parameters()).device)
    if spec["layerrope"]:
        inner.cs312_layerrope = LayerRoPE(
            hidden, alpha=layerrope_alpha, beta=layerrope_beta,
            base_freq=layerrope_base_freq,
        )
        # LayerRoPE supplies the gain, so the stock affine weights are retired
        for layer in inner.layers:
            layer.input_layernorm.weight.requires_grad_(False)
            layer.post_attention_layernorm.weight.requires_grad_(False)
    if spec["hc"]:
        depth = len(inner.layers)
        inner.cs312_routes = nn.ModuleList([
            Routing(i, hidden, MULT, manifold=spec["manifold"])
            for i in range(2 * depth)
        ])
    inner.forward = _forward_with_arch.__get__(inner, type(inner))
    return model
