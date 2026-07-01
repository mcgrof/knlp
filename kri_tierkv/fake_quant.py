# SPDX-License-Identifier: GPL-2.0
"""Asymmetric KV fake quantization (storage simulation).

This simulates the memory-traffic and quality effect of storing the KV cache at
reduced precision: quantize to an integer grid, then dequantize before attention.
It is a fake-quant study, not a speed claim. On A100 (Ampere) there is no native
FP8 throughput -- native FP8 Transformer Engine support is Hopper/H100 -- so
these routines never assert a speedup; they measure what precision costs.

Policy defaults follow the KV-protection Pareto: K precision is the dominant
lever, so K16/V8 (v8_only) is the safe default and any sub-16-bit K (k8v4) is
opt-in and flagged unsafe. Keys are quantized per-channel, values per-token,
which is the asymmetric split that survives.
"""

from __future__ import annotations


def _quant_dequant(t, bits: int, axis: int):
    """Symmetric-ish affine quantize-dequantize along `axis` (per-slice min/max).
    bits=16 is a pass-through. Requires torch.
    """
    import torch

    if bits >= 16:
        return t
    qmax = (1 << bits) - 1
    f = t.float()
    mn = f.amin(dim=axis, keepdim=True)
    mx = f.amax(dim=axis, keepdim=True)
    scale = (mx - mn).clamp(min=1e-8) / qmax
    q = ((f - mn) / scale).round().clamp(0, qmax)
    deq = q * scale + mn
    return deq.to(t.dtype)


def fake_quant_kv(k, v, k_bits: int, v_bits: int):
    """Round-trip K and V through the given bit-widths.

    K is quantized per channel (last dim held, quantize over tokens axis=2 -> use
    per-channel by reducing over tokens), V per token (reduce over head_dim). The
    exact axis choice mirrors the asymmetric KIVI-style split: keys per-channel,
    values per-token. k/v shaped (batch, n_kv_heads, seq_len, head_dim).
    """
    # keys: per-channel -> statistics over the token axis (dim=2)
    kq = _quant_dequant(k, k_bits, axis=2)
    # values: per-token -> statistics over the head_dim axis (dim=3)
    vq = _quant_dequant(v, v_bits, axis=3)
    return kq, vq


def quant_error(orig, quantized) -> float:
    """Mean relative L2 error introduced by the round-trip. Requires torch."""
    import torch

    o = orig.float()
    q = quantized.float()
    num = (o - q).norm()
    den = o.norm().clamp(min=1e-8)
    return float((num / den).item())
