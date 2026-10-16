"""Simulate token-local KV-cache rounding without changing model weights."""

import contextlib
import math
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from .model import author


def fakequant(x, bits, method="symmetric"):
    """One FP32 scale per [batch, head, token]; no future-token statistics.

    Quantize in FP32 and restore the incoming compute dtype. 16 means exact
    identity (the input is BF16). Symmetric uses 2**bits-1 signed levels.
    Affine uses all 2**bits levels with a floating per-vector minimum;
    unlike an integral zero-point it represents both endpoints exactly.
    This is dequantized simulation, with no packed allocation or speed claim.
    """
    if bits == 16:
        return x
    if bits not in (2, 3, 4, 8):
        raise ValueError(bits)
    if method in ("fp4_e2m1", "fp8_e4m3fn"):
        from .float_quant import scaled_float

        if bits != (4 if method == "fp4_e2m1" else 8):
            raise ValueError("Floating cache format and bit width disagree")
        return scaled_float(x, bits)
    w = x.float()
    if method == "symmetric":
        qmax = 2 ** (bits - 1) - 1
        scale = w.abs().amax(-1, keepdim=True) / qmax
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        result = torch.round(w / safe).clamp(-qmax, qmax) * scale
    elif method == "affine":
        lower = w.amin(-1, keepdim=True)
        scale = (w.amax(-1, keepdim=True) - lower) / (2**bits - 1)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        result = torch.round((w - lower) / safe).clamp(0, 2**bits - 1) * scale + lower
    else:
        raise ValueError(method)
    return result.to(x.dtype)


class CachePolicy:
    def __init__(self, k_bits=16, v_bits=16, method="symmetric"):
        self.set(k_bits, v_bits, method)
        self.calls = 0

    def set(self, k_bits, v_bits, method):
        self.k_bits, self.v_bits, self.method = k_bits, v_bits, method


def cache_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
    """Original author attention order, adding Q/DQ at the cache write.

    The original attention always uses is_causal=True, which is incorrect
    for offset single-token decode. This adapter uses an offset mask only
    for cached decode. All measured full-window calls keep the author's
    original is_causal=True path, RoPE, SDPA and projection order.
    """
    if output_attentions:
        raise ValueError("Unpadded causal evaluation only")
    bsz, q_len, _ = hidden_states.size()
    q = self._shape(self.q_proj(hidden_states), q_len, bsz)
    k = self._shape(self.k_proj(hidden_states), q_len, bsz)
    v = self._shape(self.v_proj(hidden_states), q_len, bsz)
    past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(v, seq_len=past_len + q_len)
    q, k = author.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    if attention_mask is not None and attention_mask.shape != (
        bsz,
        1,
        q_len,
        past_len + q_len,
    ):
        raise ValueError("Unexpected author mask shape")
    # As in the author path, the prebuilt mask is ignored and SDPA applies
    # causality. This experiment supplies unpadded token windows only.
    policy = self._kv_policy
    k = fakequant(k, policy.k_bits, policy.method)
    v = fakequant(v, policy.v_bits, policy.method)
    policy.calls += 1
    if past_key_value is not None:
        # The saved values already went through Q/DQ exactly once on write.
        k = torch.cat((past_key_value[0], k), dim=2)
        v = torch.cat((past_key_value[1], v), dim=2)
    cache = (k, v) if use_cache else None
    if past_len:
        mask = torch.arange(k.shape[-2], device=q.device).unsqueeze(0) <= (
            past_len + torch.arange(q_len, device=q.device).unsqueeze(1)
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    else:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    out = out.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    return self.o_proj(out), None, cache


@contextlib.contextmanager
def install(model, policy):
    blocks = model.lm.model.layers
    originals = []
    for block in blocks:
        attn = block.self_attn
        originals.append(attn.forward)
        attn._kv_policy = policy
        attn.forward = types.MethodType(cache_forward, attn)
    try:
        yield
    finally:
        for block, original in zip(blocks, originals):
            block.self_attn.forward = original
            del block.self_attn._kv_policy


def bytes_per_token(case, depth=32, heads=2, head_dim=64):
    payload = depth * heads * head_dim * (case["k_bits"] + case["v_bits"]) / 8
    # Actual simulator arithmetic uses FP32 scales and minima. Report a
    # hypothetical packed layout with those same metadata precisions.
    metadata = (
        sum(
            (
                0
                if case[side + "_bits"] == 16
                else 8 if case["method"] == "affine" else 4
            )
            for side in ("k", "v")
        )
        * depth
        * heads
    )
    return {
        "payload_bytes": payload,
        "scale_and_offset_bytes": metadata,
        "ideal_total_bytes": payload + metadata,
        "bf16_total_bytes": depth * heads * head_dim * 4,
        "actual_storage": "BF16 dequantized tensors; no physical compression",
    }


def cases():
    """Return both quantizers' full 5x5 K/V grid and one shared clean cell.

    Unequal K/V bit widths are independent of the quantizer's codebook:
    both the signed symmetric and floating-offset affine methods sweep
    every pair. Preserve the original 13-cell order for old report readers.
    """
    rows = [dict(name="clean", k_bits=16, v_bits=16, method="symmetric")]
    for method in ("symmetric", "affine"):
        for bits in (8, 4, 3, 2):
            rows.append(
                dict(
                    name=f"{method}-K{bits}V{bits}",
                    k_bits=bits,
                    v_bits=bits,
                    method=method,
                )
            )
    for k, v in ((8, 4), (4, 8), (16, 4), (4, 16)):
        rows.append(
            dict(name=f"symmetric-K{k}V{v}", k_bits=k, v_bits=v, method="symmetric")
        )
    present = {(r["method"], r["k_bits"], r["v_bits"]) for r in rows}
    for method in ("symmetric", "affine"):
        for k in (16, 8, 4, 3, 2):
            for v in (16, 8, 4, 3, 2):
                if (k, v) == (16, 16) or (method, k, v) in present:
                    continue
                rows.append(
                    dict(name=f"{method}-K{k}V{v}", k_bits=k, v_bits=v, method=method)
                )
    return rows


@torch.no_grad()
def evaluate(model, tokens, case, device, window=256, batch_size=32):
    """Evaluate adjacent causal windows, subtracting no external baseline."""
    if window < 1 or batch_size < 1:
        raise ValueError("Window and batch size must be positive")
    count = (len(tokens) - 1) // window
    if count < 1:
        raise ValueError("Expected at least one complete token window")
    policy = (
        None
        if case["name"] == "clean"
        else CachePolicy(case["k_bits"], case["v_bits"], case["method"])
    )
    context = contextlib.nullcontext() if policy is None else install(model, policy)
    losses = []
    began = time.perf_counter()
    with context:
        for start in range(0, count, batch_size):
            indices = (
                np.arange(start, min(start + batch_size, count))[:, None] * window
                + np.arange(window + 1)[None, :]
            )
            batch = torch.as_tensor(
                np.array(tokens[indices], dtype=np.int64), device=device
            )
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
            ce = (
                F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    batch[:, 1:].reshape(-1),
                    reduction="none",
                )
                .reshape(len(batch), window)
                .mean(-1)
            )
            values = ce.double().cpu().tolist()
            if not all(math.isfinite(x) for x in values):
                raise FloatingPointError("Nonfinite cache evaluation loss")
            losses.extend(values)
    if device == "cuda":
        torch.cuda.synchronize()
    expected_calls = model.depth * math.ceil(count / batch_size)
    if policy is not None and policy.calls != expected_calls:
        raise RuntimeError("The cache adapter did not run at every attention layer")
    return dict(
        ce=float(np.mean(losses)),
        window_ce=losses,
        windows=count,
        predicted_tokens=count * window,
        seconds=time.perf_counter() - began,
    )


@torch.no_grad()
def parity(model, ids, device):
    """Require exact noquant logits and an active quantization intervention."""
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        original = model(ids)
        with install(model, CachePolicy()):
            unchanged = model(ids)
        with install(model, CachePolicy(2, 2)):
            changed = model(ids)
    if not torch.equal(original, unchanged):
        raise RuntimeError("The noquant cache adapter changed model logits")
    if torch.equal(original, changed):
        raise RuntimeError("The cache quantizer did not affect model logits")
    return dict(noquant_bit_exact=True, quantization_active=True)
