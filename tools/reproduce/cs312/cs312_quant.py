# SPDX-License-Identifier: MIT
"""Fake-quantize the KV cache for the CS312 precision lane.

This file is staged as an untracked file into the pinned course checkout
root and named by TrainConfig.model_builder, so the pinned-commit check
still passes. It patches F.scaled_dot_product_attention to fake-quantize
the key and value tensors (the cache contents) right where attention
reads them, with a straight-through gradient, so a model trains to
tolerate a quantized cache (quantization-aware training). The same patch
serves eval-only post-hoc quantization of an already-trained model.

K and V bit-widths and formats are independent parameters, which covers
both the symmetric case (k_bits == v_bits) and the asymmetric case
(for example K16/V8). Quantization is per-token: one scale per vector
over the head-dimension axis, matching how a real KV cache is quantized.
Only K and V are touched; the query stays at compute precision, as it is
never cached.
"""

import threading

import torch
import torch.nn.functional as F

# Representable magnitude of each fp8 format, used to scale into range.
_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_FP8_DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


class _QuantConfig:
    def __init__(self):
        self.enabled = False
        self.k_bits = 16
        self.v_bits = 16
        self.k_format = "e4m3"
        self.v_format = "e4m3"
        self.granularity = "per_token"

    def set(self, k_bits, v_bits, k_format, v_format, granularity="per_token"):
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.k_format = str(k_format)
        self.v_format = str(v_format)
        self.granularity = str(granularity)
        self.enabled = self.k_bits < 16 or self.v_bits < 16


_CFG = _QuantConfig()
_ORIG_SDPA = None
_LOCK = threading.Lock()


def fake_quant(x, bits, fmt="e4m3", granularity="per_token"):
    """Symmetric fake-quant with a straight-through gradient.

    bits >= 16 is an exact identity. bits == 8 uses the named fp8 format;
    any smaller width uses a symmetric integer grid (int4 -> 15 levels,
    int3 -> 7, int2 -> ternary {-a,0,a}). Granularity "per_token" gives
    each vector over the head-dim axis its own scale (how a real KV cache
    is quantized); "per_tensor" uses one scale for the whole tensor, the
    harsher setting where key outliers bite.
    """
    if bits >= 16:
        return x
    work = x.float()
    if granularity == "per_tensor":
        amax = work.abs().amax().clamp(min=1e-12)
    else:
        amax = work.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    if bits == 8:
        fp8_max = _FP8_MAX[fmt]
        scale = fp8_max / amax
        q = (work * scale).to(_FP8_DTYPE[fmt]).float() / scale
    else:
        qmax = float(2 ** (bits - 1) - 1)  # int4 -> 7
        scale = amax / qmax
        q = torch.clamp(torch.round(work / scale), -qmax, qmax) * scale
    q = q.to(x.dtype)
    # Forward uses the quantized value; backward passes the gradient
    # straight through, so the optimizer can adapt to the quantization.
    return x + (q - x).detach()


def _quant_sdpa(query, key, value, *args, **kwargs):
    if _CFG.enabled:
        key = fake_quant(key, _CFG.k_bits, _CFG.k_format, _CFG.granularity)
        value = fake_quant(value, _CFG.v_bits, _CFG.v_format, _CFG.granularity)
    return _ORIG_SDPA(query, key, value, *args, **kwargs)


def install():
    """Patch F.scaled_dot_product_attention once, idempotently."""
    global _ORIG_SDPA
    with _LOCK:
        if _ORIG_SDPA is None:
            _ORIG_SDPA = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = _quant_sdpa


def configure(k_bits=16, v_bits=16, k_format="e4m3", v_format="e4m3",
              granularity="per_token"):
    install()
    _CFG.set(k_bits, v_bits, k_format, v_format, granularity)
    return _CFG


def build_quant_model(
    model_config,
    *,
    dtype,
    qk_norm,
    tie_word_embeddings,
    dropout,
    k_bits=16,
    v_bits=16,
    k_format="e4m3",
    v_format="e4m3",
    granularity="per_token",
    **_ignored,
):
    """Course model_builder: install the KV-quant hook, build the stock model."""
    from modeling import AutoregressiveLM

    configure(k_bits, v_bits, k_format, v_format, granularity)
    return AutoregressiveLM(
        model_config,
        dtype=dtype,
        qk_norm=qk_norm,
        tie_word_embeddings=tie_word_embeddings,
        dropout=dropout,
    )
