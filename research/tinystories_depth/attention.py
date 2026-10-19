# SPDX-License-Identifier: MIT
"""Shared causal attention for QK normalization and once-on-write cache rounding."""

import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from .model import author


def normalize(x):
    """Accumulate L2 norms in FP32 (FP64 for the independent gradient checks)."""
    w = x.double() if x.dtype == torch.float64 else x.float()
    return (w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(x.dtype)


def attend(q, k, v, gain=None, past_len=0):
    kwargs = dict(dropout_p=0.0)
    if gain is not None:
        # BF16 Q times a BF16 scalar; FP32 master scalar keeps its gradient.
        q = q * gain.to(dtype=q.dtype)
        kwargs["scale"] = 1.0
    if past_len:
        kwargs["attn_mask"] = torch.arange(k.shape[-2], device=q.device)[None, :] <= (
            past_len + torch.arange(q.shape[-2], device=q.device)[:, None]
        )
    else:
        kwargs["is_causal"] = True
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
    if output_attentions:
        raise ValueError("Unpadded causal attention only")
    b, length, _ = hidden_states.shape
    q = self._shape(self.q_proj(hidden_states), length, b)
    k = self._shape(self.k_proj(hidden_states), length, b)
    v = self._shape(self.v_proj(hidden_states), length, b)
    raw_q, raw_k = q, k
    gain = getattr(self, "qknorm_gain", None)
    if gain is not None:
        q, k = normalize(q), normalize(k)
    past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
    if position_ids is None:
        position_ids = torch.arange(past_len, past_len + length, device=q.device)[None]
    cos, sin = self.rotary_emb(v, seq_len=past_len + length)
    q, k = author.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    if attention_mask is not None and attention_mask.shape != (
        b,
        1,
        length,
        past_len + length,
    ):
        raise ValueError("Unexpected author mask shape")
    policy = getattr(self, "_kv_policy", None)
    if policy is not None:
        from .kv_cache import fakequant

        k = fakequant(k, policy.k_bits, policy.method)
        v = fakequant(v, policy.v_bits, policy.method)
        policy.calls += 1
    if past_key_value is not None:
        k = torch.cat((past_key_value[0], k), dim=2)
        v = torch.cat((past_key_value[1], v), dim=2)
    observer = getattr(self, "_qk_observer", None)
    if observer is not None:
        observer(raw_q, raw_k, q, k, gain, past_len)
    out = attend(q, k, v, gain, past_len)
    out = out.transpose(1, 2).reshape(b, length, self.hidden_size)
    return self.o_proj(out), None, (k, v) if use_cache else None


def enable(model, length=256):
    """Add constant scalar parameters after shared weights, without drawing RNG."""
    if length <= 1:
        raise ValueError("QKNorm initialization needs context length above one")
    for block in model.lm.model.layers:
        attn = block.self_attn
        attn.register_parameter(
            "qknorm_gain",
            nn.Parameter(torch.tensor(math.log2(length * length - length))),
        )
        attn.forward = types.MethodType(forward, attn)
