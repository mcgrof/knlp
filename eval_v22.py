#!/usr/bin/env python
"""
BPA v22 Evaluation: Scale overhead engineering + larger model scaling +
bandwidth-bound proof.

Builds on v21's finding that g=32 k=4 with theory ranking achieves
kv_ratio=0.3203 PASS@3% on Qwen2.5-0.5B. Tests whether scale metadata
overhead can be reduced for small group sizes, whether the recipe scales
to larger models, and whether bandwidth-bound regimes show latency gains.

Phases:
  0: Lock current best + regression guard
  1: Scale overhead engineering (token-window sharing, scale quant, head sharing)
  2: Larger model replication (Qwen2.5-1.5B)
  3: Bandwidth-bound regime proof
  4: Theory update + final deliverables

Usage:
    python eval_v22.py --phase 0
    python eval_v22.py --phase 1
    python eval_v22.py --phase 2 --model qwen15b
    python eval_v22.py --phase 3
    python eval_v22.py --phase 4
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from scripts.bpa_v11_bench import (
    DTYPE,
    compute_ppl,
    get_cpu_rss_mb,
    get_gpu_mem,
    get_text_batch,
    gpu_sync,
    load_validation_tokens,
    reset_gpu_mem,
)
from backends.base import DenseBackend, V14StepStats
from backends.quant import (
    QuantBackend,
    quantize_int4_block,
    quantize_int8_symmetric,
    dequantize_int4_block,
    dequantize_int8_symmetric,
)
from eval_v15 import (
    V15Result,
    apply_quality_gating,
    build_scoreboard,
    gpu_preflight,
    load_model,
    run_single_eval,
)
from eval_v16 import (
    MixedPrecisionBackend,
    build_schedules,
    run_backend_sweep,
    run_dense_baselines,
    save_results,
)
from eval_v21 import (
    GroupedMixedBackend,
    build_k_schedule,
    check_pass,
    compute_kv_bytes_per_token,
    eval_config,
    load_theory_ranking,
    max_delta,
)


# ============================================================
# Extended byte accounting for scale-amortized configurations
# ============================================================


def compute_kv_bytes_amortized(
    n_layers,
    n_kv_heads,
    head_dim,
    payload_bits,
    group_size,
    scale_dtype_bits=16,
    scale_sharing_window=1,
    k_int8_layers=0,
):
    """Compute KV bytes with amortized scale overhead.

    When scale_sharing_window=S, scales are shared across S consecutive
    tokens, reducing per-token scale cost by factor of S.

    Args:
        scale_sharing_window: Number of tokens sharing each scale factor.
            S=1 means per-token scales (no amortization).
            S=4 means one scale per 4 tokens.
    """
    n_int4_layers = n_layers - k_int8_layers
    scale_bytes = scale_dtype_bits // 8
    S = max(1, scale_sharing_window)

    dense_bytes_per_layer = 2 * n_kv_heads * head_dim * 2
    dense_total = n_layers * dense_bytes_per_layer

    if payload_bits == 16:
        return {
            "dense_bytes_per_token": dense_total,
            "total_bytes_per_token": dense_total,
            "kv_ratio": 1.0,
            "scale_overhead_pct": 0.0,
            "scale_sharing_window": S,
        }

    # INT8 layers
    int8_payload = 2 * n_kv_heads * head_dim * 1
    int8_scale = 2 * n_kv_heads * 1 * scale_bytes
    int8_bytes_per_layer = int8_payload + int8_scale

    if payload_bits == 8:
        total = n_layers * int8_bytes_per_layer
        return {
            "dense_bytes_per_token": dense_total,
            "total_bytes_per_token": total,
            "kv_ratio": round(total / dense_total, 6),
            "scale_overhead_pct": 0.0,
            "scale_sharing_window": 1,
        }

    # INT4 with group quant + amortized scales
    g = group_size if group_size is not None else head_dim
    n_groups = math.ceil(head_dim / g)
    int4_payload = 2 * n_kv_heads * head_dim * 0.5
    # Amortized: each scale serves S tokens
    int4_scale_per_token = 2 * n_kv_heads * n_groups * scale_bytes / S
    int4_bytes_per_layer = int4_payload + int4_scale_per_token

    total = k_int8_layers * int8_bytes_per_layer + n_int4_layers * int4_bytes_per_layer
    overhead = (
        int4_scale_per_token / int4_bytes_per_layer * 100
        if int4_bytes_per_layer > 0
        else 0
    )

    return {
        "dense_bytes_per_token": dense_total,
        "int4_payload_per_layer": int4_payload,
        "int4_scale_per_token_per_layer": round(int4_scale_per_token, 4),
        "int4_bytes_per_layer": round(int4_bytes_per_layer, 4),
        "total_bytes_per_token": round(total, 4),
        "kv_ratio": round(total / dense_total, 6),
        "scale_overhead_pct": round(overhead, 2),
        "scale_sharing_window": S,
    }


def compute_kv_bytes_scale_compressed(
    n_layers,
    n_kv_heads,
    head_dim,
    payload_bits,
    group_size,
    scale_dtype_bits=8,
    k_int8_layers=0,
):
    """Compute KV bytes with compressed scale representation.

    Uses INT8 or smaller dtype for scales instead of FP16.
    """
    return compute_kv_bytes_amortized(
        n_layers,
        n_kv_heads,
        head_dim,
        payload_bits,
        group_size,
        scale_dtype_bits=scale_dtype_bits,
        scale_sharing_window=1,
        k_int8_layers=k_int8_layers,
    )


# ============================================================
# Scale-amortized mixed-precision backend
# ============================================================


class AmortizedScaleMixedBackend:
    """Mixed-precision with token-window scale amortization.

    For INT4 layers, computes quantization scales once per S-token window
    and reuses them. This reduces scale storage but may degrade quality
    due to scale drift across tokens within the window.

    scale_mode controls how scales are computed:
      "post_rope": compute on post-RoPE K (naive, may drift)
      "pre_rope": compute on pre-RoPE K norms (better stability)
      "norm_invariant": compute on |K| norms which are RoPE-invariant
    """

    def __init__(
        self, layer_bits, group_size=4, scale_window=4, scale_mode="post_rope"
    ):
        self.layer_bits = layer_bits
        self.group_size = group_size
        self.scale_window = scale_window
        self.scale_mode = scale_mode
        n8 = sum(1 for b in layer_bits if b == 8)
        self._name = f"amort_g{group_size}_S{scale_window}_{scale_mode}_k{n8}"

    @property
    def name(self):
        return self._name

    def description(self):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        return (
            f"Amortized g={self.group_size} S={self.scale_window} "
            f"{self.scale_mode} {n8}xINT8"
        )

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        n4 = sum(1 for b in self.layer_bits if b == 4)
        print(
            f"    {self.name} calibrated: {n4} INT4(g={self.group_size},"
            f"S={self.scale_window}), {n8} INT8"
        )

    def _quantize_int4_amortized(self, x, group_size, scale_window):
        """Per-group INT4 with amortized scales across token windows.

        Within each window of S tokens, the first token's scales are
        used for all tokens in the window. This simulates S-token
        scale sharing.
        """
        B, H, T, D = x.shape
        g = group_size
        n_groups = (D + g - 1) // g
        pad = n_groups * g - D
        if pad > 0:
            x = F.pad(x, (0, pad))
        x_g = x.reshape(B, H, T, n_groups, g)

        S = scale_window
        # Compute scales at window boundaries, reuse within window
        # Window boundaries: 0, S, 2S, ...
        n_windows = (T + S - 1) // S

        # Compute per-token scales
        amax_all = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_all = amax_all / 7.0  # [B, H, T, n_groups, 1]

        if S > 1:
            # Amortize: use window-leader scale for all tokens in window
            # Take max scale within each window for safety
            scale_windowed = torch.zeros_like(scale_all)
            for w in range(n_windows):
                t_start = w * S
                t_end = min((w + 1) * S, T)
                window_max = scale_all[:, :, t_start:t_end].amax(dim=2, keepdim=True)
                scale_windowed[:, :, t_start:t_end] = window_max
            scale_used = scale_windowed
        else:
            scale_used = scale_all

        x_q = (x_g / scale_used).round().clamp(-8, 7).to(torch.int8)
        return x_q, scale_used, D

    def _dequantize_int4_amortized(self, x_q, scale, orig_D):
        x_hat = x_q.float() * scale
        B, H, T, ng, gs = x_hat.shape
        x_hat = x_hat.reshape(B, H, T, ng * gs)
        return x_hat[:, :, :, :orig_D]

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        del out
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        cache_len = past[0][0].shape[2]

        t0 = time.perf_counter()
        has_compressed = False
        n_full = cache_len
        n_compressed = 0

        if cache_len > self.W_min + self.W_sink:
            far_end = cache_len - self.W_min
            n_far = far_end - self.W_sink

            if n_far > 0:
                new_cache = DynamicCache()
                for li in range(n_layers):
                    k, v = past[li]
                    k_sink = k[:, :, : self.W_sink, :]
                    v_sink = v[:, :, : self.W_sink, :]
                    k_far = k[:, :, self.W_sink : far_end, :]
                    v_far = v[:, :, self.W_sink : far_end, :]
                    k_near = k[:, :, far_end:, :]
                    v_near = v[:, :, far_end:, :]

                    bits = self.layer_bits[li]
                    if bits == 8:
                        k_q, k_s = quantize_int8_symmetric(k_far)
                        k_hat = dequantize_int8_symmetric(k_q, k_s).to(dtype)
                        v_q, v_s = quantize_int8_symmetric(v_far)
                        v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
                    else:
                        k_q, k_s, k_D = self._quantize_int4_amortized(
                            k_far, self.group_size, self.scale_window
                        )
                        k_hat = self._dequantize_int4_amortized(k_q, k_s, k_D).to(dtype)
                        v_q, v_s, v_D = self._quantize_int4_amortized(
                            v_far, self.group_size, self.scale_window
                        )
                        v_hat = self._dequantize_int4_amortized(v_q, v_s, v_D).to(dtype)

                    k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
                    v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
                    new_cache.update(k_new, v_new, li)

                past = new_cache
                has_compressed = True
                n_full = self.W_sink + self.W_min
                n_compressed = n_far

        compress_ms = (time.perf_counter() - t0) * 1000

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            pos_ids = None
            if has_compressed:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )
            with torch.no_grad():
                out = model(
                    next_token,
                    past_key_values=past,
                    position_ids=pos_ids,
                    use_cache=True,
                )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            step_stats.append(
                V14StepStats(
                    kv_kept=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


class ScaleQuantMixedBackend:
    """Mixed-precision with quantized scale representation.

    Instead of FP16 scales, uses INT8 log-scale representation:
    scale = 2^(q / alpha), where q is INT8 and alpha is a fixed constant.
    This halves scale storage from 2 bytes to 1 byte per scale.
    """

    def __init__(self, layer_bits, group_size=4, alpha=16.0):
        self.layer_bits = layer_bits
        self.group_size = group_size
        self.alpha = alpha
        n8 = sum(1 for b in layer_bits if b == 8)
        self._name = f"scalequant_g{group_size}_k{n8}"

    @property
    def name(self):
        return self._name

    def description(self):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        return f"ScaleQuant g={self.group_size} alpha={self.alpha} {n8}xINT8"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        n4 = sum(1 for b in self.layer_bits if b == 4)
        print(
            f"    {self.name} calibrated: {n4} INT4(g={self.group_size},"
            f"logscale), {n8} INT8"
        )

    def _quantize_scale_int8(self, scale_fp):
        """Quantize FP16 scales to INT8 log-scale."""
        log_scale = torch.log2(scale_fp.clamp(min=1e-30))
        q = (log_scale * self.alpha).round().clamp(-128, 127).to(torch.int8)
        return q

    def _dequantize_scale_int8(self, q):
        """Reconstruct scales from INT8 log-scale."""
        return torch.pow(2.0, q.float() / self.alpha)

    def _quantize_int4_logscale(self, x, group_size):
        """INT4 quantization with INT8 log-scale storage."""
        B, H, T, D = x.shape
        g = group_size
        n_groups = (D + g - 1) // g
        pad = n_groups * g - D
        if pad > 0:
            x = F.pad(x, (0, pad))
        x_g = x.reshape(B, H, T, n_groups, g)
        amax = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_fp = amax / 7.0

        # Quantize scale to INT8
        scale_q = self._quantize_scale_int8(scale_fp)
        scale_recon = self._dequantize_scale_int8(scale_q).unsqueeze(-1)

        # Quantize payload using reconstructed scale
        x_q = (x_g / scale_recon).round().clamp(-8, 7).to(torch.int8)
        return x_q, scale_recon, D

    def _dequantize_int4_logscale(self, x_q, scale, orig_D):
        x_hat = x_q.float() * scale
        B, H, T, ng, gs = x_hat.shape
        x_hat = x_hat.reshape(B, H, T, ng * gs)
        return x_hat[:, :, :, :orig_D]

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        del out
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        cache_len = past[0][0].shape[2]

        t0 = time.perf_counter()
        has_compressed = False
        n_full = cache_len
        n_compressed = 0

        if cache_len > self.W_min + self.W_sink:
            far_end = cache_len - self.W_min
            n_far = far_end - self.W_sink

            if n_far > 0:
                new_cache = DynamicCache()
                for li in range(n_layers):
                    k, v = past[li]
                    k_sink = k[:, :, : self.W_sink, :]
                    v_sink = v[:, :, : self.W_sink, :]
                    k_far = k[:, :, self.W_sink : far_end, :]
                    v_far = v[:, :, self.W_sink : far_end, :]
                    k_near = k[:, :, far_end:, :]
                    v_near = v[:, :, far_end:, :]

                    bits = self.layer_bits[li]
                    if bits == 8:
                        k_q, k_s = quantize_int8_symmetric(k_far)
                        k_hat = dequantize_int8_symmetric(k_q, k_s).to(dtype)
                        v_q, v_s = quantize_int8_symmetric(v_far)
                        v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
                    else:
                        k_q, k_s, k_D = self._quantize_int4_logscale(
                            k_far, self.group_size
                        )
                        k_hat = self._dequantize_int4_logscale(k_q, k_s, k_D).to(dtype)
                        v_q, v_s, v_D = self._quantize_int4_logscale(
                            v_far, self.group_size
                        )
                        v_hat = self._dequantize_int4_logscale(v_q, v_s, v_D).to(dtype)

                    k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
                    v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
                    new_cache.update(k_new, v_new, li)

                past = new_cache
                has_compressed = True
                n_full = self.W_sink + self.W_min
                n_compressed = n_far

        compress_ms = (time.perf_counter() - t0) * 1000

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            pos_ids = None
            if has_compressed:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )
            with torch.no_grad():
                out = model(
                    next_token,
                    past_key_values=past,
                    position_ids=pos_ids,
                    use_cache=True,
                )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            step_stats.append(
                V14StepStats(
                    kv_kept=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


class SharedScaleMixedBackend:
    """Mixed-precision with scales shared across KV heads.

    Instead of per-head scales, compute a single scale shared across
    all KV heads within each layer. Reduces scale count by H factor.
    """

    def __init__(self, layer_bits, group_size=4):
        self.layer_bits = layer_bits
        self.group_size = group_size
        n8 = sum(1 for b in layer_bits if b == 8)
        self._name = f"sharedscale_g{group_size}_k{n8}"

    @property
    def name(self):
        return self._name

    def description(self):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        return f"SharedScale g={self.group_size} {n8}xINT8"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        n4 = sum(1 for b in self.layer_bits if b == 4)
        print(
            f"    {self.name} calibrated: {n4} INT4(g={self.group_size},"
            f"shared), {n8} INT8"
        )

    def _quantize_int4_shared(self, x, group_size):
        """INT4 with scales shared across heads."""
        B, H, T, D = x.shape
        g = group_size
        n_groups = (D + g - 1) // g
        pad = n_groups * g - D
        if pad > 0:
            x = F.pad(x, (0, pad))
        x_g = x.reshape(B, H, T, n_groups, g)
        # Per-head scales
        amax_perhead = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        # Share across heads: take max across head dim
        amax_shared = amax_perhead.amax(dim=1, keepdim=True)
        scale = amax_shared / 7.0
        x_q = (x_g / scale).round().clamp(-8, 7).to(torch.int8)
        return x_q, scale, D

    def _dequantize_int4_shared(self, x_q, scale, orig_D):
        x_hat = x_q.float() * scale
        B, H, T, ng, gs = x_hat.shape
        x_hat = x_hat.reshape(B, H, T, ng * gs)
        return x_hat[:, :, :, :orig_D]

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        del out
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        cache_len = past[0][0].shape[2]

        t0 = time.perf_counter()
        has_compressed = False
        n_full = cache_len
        n_compressed = 0

        if cache_len > self.W_min + self.W_sink:
            far_end = cache_len - self.W_min
            n_far = far_end - self.W_sink

            if n_far > 0:
                new_cache = DynamicCache()
                for li in range(n_layers):
                    k, v = past[li]
                    k_sink = k[:, :, : self.W_sink, :]
                    v_sink = v[:, :, : self.W_sink, :]
                    k_far = k[:, :, self.W_sink : far_end, :]
                    v_far = v[:, :, self.W_sink : far_end, :]
                    k_near = k[:, :, far_end:, :]
                    v_near = v[:, :, far_end:, :]

                    bits = self.layer_bits[li]
                    if bits == 8:
                        k_q, k_s = quantize_int8_symmetric(k_far)
                        k_hat = dequantize_int8_symmetric(k_q, k_s).to(dtype)
                        v_q, v_s = quantize_int8_symmetric(v_far)
                        v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
                    else:
                        k_q, k_s, k_D = self._quantize_int4_shared(
                            k_far, self.group_size
                        )
                        k_hat = self._dequantize_int4_shared(k_q, k_s, k_D).to(dtype)
                        v_q, v_s, v_D = self._quantize_int4_shared(
                            v_far, self.group_size
                        )
                        v_hat = self._dequantize_int4_shared(v_q, v_s, v_D).to(dtype)

                    k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
                    v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
                    new_cache.update(k_new, v_new, li)

                past = new_cache
                has_compressed = True
                n_full = self.W_sink + self.W_min
                n_compressed = n_far

        compress_ms = (time.perf_counter() - t0) * 1000

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            pos_ids = None
            if has_compressed:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )
            with torch.no_grad():
                out = model(
                    next_token,
                    past_key_values=past,
                    position_ids=pos_ids,
                    use_cache=True,
                )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            step_stats.append(
                V14StepStats(
                    kv_kept=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


# ============================================================
# Phase 0: Lock current best + regression guard
# ============================================================


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Regression guard on known-good configs."""
    outdir = os.path.join(args.outdir, "phase0")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 0: Lock Current Best + Regression Guard")
    print("=" * 60)

    theory_ranking = load_theory_ranking(args.theory_path)

    # Dense baselines at all L
    print("\n" + "=" * 60)
    print("Dense baselines")
    print("=" * 60)
    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    configs = OrderedDict()

    # 1. g=32 k=4 (current best from v21)
    sched_k4 = build_k_schedule(theory_ranking, 4)
    configs["g32_k4"] = {
        "backend": GroupedMixedBackend(layer_bits=sched_k4, group_size=32),
        "expected_ratio": 0.3203,
        "expected_pass": True,
    }

    # 2. S2_manual_k6 (v16 baseline)
    sched_k6 = build_k_schedule(theory_ranking, 6)
    configs["S2_k6"] = {
        "backend": MixedPrecisionBackend(layer_bits=sched_k6),
        "expected_ratio": 0.333,
        "expected_pass": True,
    }

    # 3. INT8-all
    sched_int8 = [8] * model_config["n_layers"]
    configs["INT8_all"] = {
        "backend": GroupedMixedBackend(layer_bits=sched_int8, group_size=32),
        "expected_ratio": 0.5156,
        "expected_pass": True,
    }

    results = {}
    all_pass = True

    for name, cfg in configs.items():
        be = cfg["backend"]
        print(f"\n  Testing: {name}")

        evals = eval_config(
            be,
            model,
            token_data,
            valid_L,
            args.seeds,
            args.device,
            max_ctx,
            model_config,
            dense_ppls,
            args.decode_steps,
        )

        p3 = check_pass(evals, 3.0)
        md = max_delta(evals)

        results[name] = {
            "evals": evals,
            "pass_3pct": p3,
            "max_delta": round(md, 2),
            "expected_pass": cfg["expected_pass"],
        }

        status = "PASS" if p3 else "FAIL"
        print(f"    max_delta={md:+.2f}% {status}")

        if p3 != cfg["expected_pass"]:
            print(f"    REGRESSION: expected pass={cfg['expected_pass']}, got {p3}")
            all_pass = False

    if all_pass:
        print("\n  All regression checks PASSED. Proceeding.")
    else:
        print("\n  REGRESSION DETECTED. Investigate before continuing!")

    with open(os.path.join(outdir, "regression_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 0,
        "version": "v22",
        "gpu_info": gpu_info,
        "all_pass": all_pass,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 0 complete. Saved to {outdir}/")
    return dense_ppls


# ============================================================
# Phase 1: Scale overhead engineering
# ============================================================


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Test 3 scale-overhead strategies."""
    outdir = os.path.join(args.outdir, "phase1")
    art_dir = os.path.join(args.outdir, "artifacts", "v22")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 1: Scale Overhead Engineering")
    print("=" * 60)

    theory_ranking = load_theory_ranking(args.theory_path)
    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    # Step 1a: Analytical byte accounting for all strategies
    print("\n  Step 1a: Byte accounting for scale strategies")
    accounting = OrderedDict()

    # Baseline: g=32 k=4 (no amortization)
    accounting["baseline_g32_k4"] = compute_kv_bytes_per_token(
        n_layers, n_kv_heads, head_dim, 4, 32, k_int8_layers=4
    )

    # Strategy 1: Token-window amortization for g=4 and g=8
    for g in [4, 8]:
        for S in [1, 2, 4, 8, 16]:
            for k in [2, 3, 4]:
                name = f"amort_g{g}_S{S}_k{k}"
                accounting[name] = compute_kv_bytes_amortized(
                    n_layers,
                    n_kv_heads,
                    head_dim,
                    4,
                    g,
                    scale_sharing_window=S,
                    k_int8_layers=k,
                )

    # Strategy 2: Scale quantization (INT8 logscale) for g=4 and g=8
    for g in [4, 8]:
        for k in [2, 3, 4]:
            name = f"scalequant_g{g}_k{k}"
            accounting[name] = compute_kv_bytes_scale_compressed(
                n_layers,
                n_kv_heads,
                head_dim,
                4,
                g,
                scale_dtype_bits=8,
                k_int8_layers=k,
            )

    # Strategy 3: Shared scales (reduce by H=n_kv_heads factor)
    # Byte model: same as reducing n_groups by H factor in scale count
    for g in [4, 8]:
        for k in [2, 3, 4]:
            name = f"sharedscale_g{g}_k{k}"
            # Shared scales: scale cost divided by n_kv_heads
            r = compute_kv_bytes_amortized(
                n_layers,
                n_kv_heads,
                head_dim,
                4,
                g,
                scale_sharing_window=1,
                k_int8_layers=k,
            )
            # Adjust for shared scales: scale_per_token reduced by H
            # For this model H=2, so scales halved
            r["note"] = f"scales shared across {n_kv_heads} KV heads"
            accounting[name] = r

    # Print summary table of configs beating 0.333
    print(f"\n  {'Config':<30s} {'ratio':>8s} {'scale%':>8s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")
    for name, r in accounting.items():
        ratio = r.get("kv_ratio", 1.0)
        if ratio <= 0.35:
            print(
                f"  {name:<30s} {ratio:>8.4f} "
                f"{r.get('scale_overhead_pct', 0):>8.1f}%"
            )

    # Identify configs worth testing (ratio < 0.333)
    candidates = []
    for name, r in accounting.items():
        ratio = r.get("kv_ratio", 1.0)
        if ratio < 0.333 and "baseline" not in name:
            candidates.append((name, r))
    candidates.sort(key=lambda x: x[1]["kv_ratio"])

    print(f"\n  Configs with ratio < 0.333: {len(candidates)}")
    for name, r in candidates[:10]:
        print(f"    {name}: ratio={r['kv_ratio']:.4f}")

    with open(os.path.join(art_dir, "scale_overhead_study.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "accounting": accounting,
                "candidates": [c[0] for c in candidates],
            },
            f,
            indent=2,
        )

    # Step 1b: Quality evaluation of viable strategies
    print(f"\n  Step 1b: Quality evaluation at L=8192 (quick screen)")

    dense_results_8k, dense_ppls_8k = run_dense_baselines(
        model,
        token_data,
        [8192],
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    # Test Strategy 1: Amortized g=4 with various S, k=4
    strategy1_results = {}
    for S in [1, 2, 4, 8, 16]:
        sched = build_k_schedule(theory_ranking, 4)
        be = AmortizedScaleMixedBackend(
            layer_bits=sched, group_size=4, scale_window=S, scale_mode="post_rope"
        )
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        name = f"amort_g4_S{S}_k4"
        print(f"\n    Testing: {name}")

        evals = {}
        for seed in args.seeds:
            r = run_single_eval(
                be,
                model,
                token_data,
                8192,
                args.decode_steps,
                seed,
                args.device,
                max_ctx,
                model_config,
            )
            dense_ref = dense_ppls_8k.get((8192, "r1", seed), r.ppl)
            delta = (r.ppl - dense_ref) / dense_ref * 100
            evals[f"L8192_s{seed}"] = {
                "ppl": round(r.ppl, 4),
                "delta_pct": round(delta, 2),
            }
            print(f"      s={seed}: delta={delta:+.2f}%")

        md = max(abs(e["delta_pct"]) for e in evals.values())
        acct = accounting.get(name, {})
        strategy1_results[name] = {
            "S": S,
            "g": 4,
            "k": 4,
            "evals_8k": evals,
            "max_delta_8k": round(md, 2),
            "kv_ratio": acct.get("kv_ratio"),
        }

    # Test Strategy 1 with g=8
    for S in [1, 2, 4, 8]:
        sched = build_k_schedule(theory_ranking, 4)
        be = AmortizedScaleMixedBackend(
            layer_bits=sched, group_size=8, scale_window=S, scale_mode="post_rope"
        )
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        name = f"amort_g8_S{S}_k4"
        print(f"\n    Testing: {name}")

        evals = {}
        for seed in args.seeds:
            r = run_single_eval(
                be,
                model,
                token_data,
                8192,
                args.decode_steps,
                seed,
                args.device,
                max_ctx,
                model_config,
            )
            dense_ref = dense_ppls_8k.get((8192, "r1", seed), r.ppl)
            delta = (r.ppl - dense_ref) / dense_ref * 100
            evals[f"L8192_s{seed}"] = {
                "ppl": round(r.ppl, 4),
                "delta_pct": round(delta, 2),
            }
            print(f"      s={seed}: delta={delta:+.2f}%")

        md = max(abs(e["delta_pct"]) for e in evals.values())
        acct_key = f"amort_g8_S{S}_k4"
        acct = accounting.get(acct_key, {})
        strategy1_results[name] = {
            "S": S,
            "g": 8,
            "k": 4,
            "evals_8k": evals,
            "max_delta_8k": round(md, 2),
            "kv_ratio": acct.get("kv_ratio"),
        }

    # Test Strategy 2: Scale quantization for g=4 k=4
    strategy2_results = {}
    for g in [4, 8]:
        sched = build_k_schedule(theory_ranking, 4)
        be = ScaleQuantMixedBackend(layer_bits=sched, group_size=g)
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        name = f"scalequant_g{g}_k4"
        print(f"\n    Testing: {name}")

        evals = {}
        for seed in args.seeds:
            r = run_single_eval(
                be,
                model,
                token_data,
                8192,
                args.decode_steps,
                seed,
                args.device,
                max_ctx,
                model_config,
            )
            dense_ref = dense_ppls_8k.get((8192, "r1", seed), r.ppl)
            delta = (r.ppl - dense_ref) / dense_ref * 100
            evals[f"L8192_s{seed}"] = {
                "ppl": round(r.ppl, 4),
                "delta_pct": round(delta, 2),
            }
            print(f"      s={seed}: delta={delta:+.2f}%")

        md = max(abs(e["delta_pct"]) for e in evals.values())
        acct = accounting.get(name, {})
        strategy2_results[name] = {
            "g": g,
            "k": 4,
            "evals_8k": evals,
            "max_delta_8k": round(md, 2),
            "kv_ratio": acct.get("kv_ratio"),
        }

    # Test Strategy 3: Shared scales for g=4 k=4
    strategy3_results = {}
    for g in [4, 8]:
        sched = build_k_schedule(theory_ranking, 4)
        be = SharedScaleMixedBackend(layer_bits=sched, group_size=g)
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        name = f"sharedscale_g{g}_k4"
        print(f"\n    Testing: {name}")

        evals = {}
        for seed in args.seeds:
            r = run_single_eval(
                be,
                model,
                token_data,
                8192,
                args.decode_steps,
                seed,
                args.device,
                max_ctx,
                model_config,
            )
            dense_ref = dense_ppls_8k.get((8192, "r1", seed), r.ppl)
            delta = (r.ppl - dense_ref) / dense_ref * 100
            evals[f"L8192_s{seed}"] = {
                "ppl": round(r.ppl, 4),
                "delta_pct": round(delta, 2),
            }
            print(f"      s={seed}: delta={delta:+.2f}%")

        md = max(abs(e["delta_pct"]) for e in evals.values())
        acct = accounting.get(name, {})
        strategy3_results[name] = {
            "g": g,
            "k": 4,
            "evals_8k": evals,
            "max_delta_8k": round(md, 2),
            "kv_ratio": acct.get("kv_ratio"),
        }

    # Step 1c: Validate survivors at 16K/32K
    all_8k = {}
    all_8k.update(strategy1_results)
    all_8k.update(strategy2_results)
    all_8k.update(strategy3_results)

    # Identify survivors: max_delta_8k <= 5%
    survivors = [
        (name, cfg) for name, cfg in all_8k.items() if cfg["max_delta_8k"] <= 5.0
    ]
    survivors.sort(key=lambda x: x[1].get("kv_ratio") or 1.0)

    print(f"\n  Step 1c: Validating {len(survivors)} survivors at 16K/32K")

    if survivors:
        dense_results_all, dense_ppls_all = run_dense_baselines(
            model,
            token_data,
            valid_L,
            args.decode_steps,
            args.seeds,
            args.device,
            max_ctx,
            model_config,
        )

    validated = {}
    for name, cfg in survivors:
        g = cfg["g"]
        k = cfg["k"]
        S = cfg.get("S", 1)
        sched = build_k_schedule(theory_ranking, k)

        print(f"\n    Validating: {name}")

        # Recreate correct backend type
        if "amort" in name:
            be = AmortizedScaleMixedBackend(
                layer_bits=sched,
                group_size=g,
                scale_window=S,
                scale_mode="post_rope",
            )
        elif "scalequant" in name:
            be = ScaleQuantMixedBackend(layer_bits=sched, group_size=g)
        elif "sharedscale" in name:
            be = SharedScaleMixedBackend(layer_bits=sched, group_size=g)
        else:
            be = GroupedMixedBackend(layer_bits=sched, group_size=g)

        full_evals = dict(cfg["evals_8k"])

        for L in [16384, 32768]:
            if L not in valid_L:
                continue
            be.configure(L, model_config)
            be.calibrate(model, token_data, L, args.device, model_config)

            for seed in args.seeds:
                r = run_single_eval(
                    be,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                )
                dense_ref = dense_ppls_all.get((L, "r1", seed), r.ppl)
                delta = (r.ppl - dense_ref) / dense_ref * 100
                full_evals[f"L{L}_s{seed}"] = {
                    "ppl": round(r.ppl, 4),
                    "delta_pct": round(delta, 2),
                }
                Lk = f"{L // 1024}K"
                print(f"      L={Lk} s={seed}: delta={delta:+.2f}%")

        p3 = check_pass(full_evals, 3.0)
        p1 = check_pass(full_evals, 1.0)
        md = max_delta(full_evals)

        validated[name] = {
            **cfg,
            "full_evals": full_evals,
            "pass_allL_3pct": p3,
            "pass_allL_1pct": p1,
            "max_delta_allL": round(md, 2),
        }

        status = "PASS" if p3 else "FAIL"
        print(f"      -> {status} max_delta={md:+.2f}%")

    # Also test reduced k for best strategies
    best_strat = None
    best_ratio = 1.0
    for name, v in validated.items():
        if v.get("pass_allL_3pct") and (v.get("kv_ratio") or 1.0) < best_ratio:
            best_ratio = v.get("kv_ratio") or 1.0
            best_strat = name

    reduced_k = {}
    if best_strat and "g4" in best_strat:
        base_cfg = validated[best_strat]
        S = base_cfg.get("S", 1)
        g = base_cfg["g"]
        print(f"\n  Step 1d: Testing reduced k for best strategy {best_strat}")

        for k in [2, 3]:
            sched = build_k_schedule(theory_ranking, k)
            if "amort" in best_strat:
                be = AmortizedScaleMixedBackend(
                    layer_bits=sched,
                    group_size=g,
                    scale_window=S,
                    scale_mode="post_rope",
                )
            elif "scalequant" in best_strat:
                be = ScaleQuantMixedBackend(layer_bits=sched, group_size=g)
            elif "sharedscale" in best_strat:
                be = SharedScaleMixedBackend(layer_bits=sched, group_size=g)
            else:
                be = GroupedMixedBackend(layer_bits=sched, group_size=g)

            rk_name = best_strat.replace("k4", f"k{k}")
            print(f"\n    Testing: {rk_name}")

            evals = eval_config(
                be,
                model,
                token_data,
                valid_L,
                args.seeds,
                args.device,
                max_ctx,
                model_config,
                dense_ppls_all,
                args.decode_steps,
            )

            p3 = check_pass(evals, 3.0)
            md_val = max_delta(evals)
            acct_key = rk_name.replace(f"_k{k}", f"_k{k}")
            acct = accounting.get(
                f"amort_g{g}_S{S}_k{k}", accounting.get(f"scalequant_g{g}_k{k}", {})
            )

            reduced_k[rk_name] = {
                "g": g,
                "k": k,
                "S": S,
                "evals": evals,
                "pass_3pct": p3,
                "max_delta": round(md_val, 2),
                "kv_ratio": acct.get("kv_ratio"),
            }

            status = "PASS" if p3 else "FAIL"
            print(f"      -> {status} max_delta={md_val:+.2f}%")

    # Save all results
    phase1_out = {
        "accounting_summary": {
            k: {
                "kv_ratio": v.get("kv_ratio"),
                "scale_overhead_pct": v.get("scale_overhead_pct"),
            }
            for k, v in accounting.items()
            if v.get("kv_ratio", 1.0) < 0.4
        },
        "strategy1_8k": strategy1_results,
        "strategy2_8k": strategy2_results,
        "strategy3_8k": strategy3_results,
        "validated": validated,
        "reduced_k": reduced_k,
        "best_strategy": best_strat,
    }

    with open(os.path.join(outdir, "phase1_results.json"), "w") as f:
        json.dump(phase1_out, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "version": "v22",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 1 complete. Saved to {outdir}/")


# ============================================================
# Phase 2: Larger model replication
# ============================================================


def compute_residual_ratio(model, token_data, device_str, model_config, n_layers):
    """Compute residual_ratio proxy for layer sensitivity on new model.

    Uses C3_residual_ratio from v19: ratio of quantization residual norm
    to clean activation norm per layer.
    """
    rng = np.random.RandomState(42)
    cal_len = 2048
    idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(idx, use_cache=True)
        past = out.past_key_values
    del out

    ratios = []
    for li in range(n_layers):
        k, v = past[li]
        # INT4 g=32 quantization
        B, H, T, D = k.shape
        g = 32
        n_groups = (D + g - 1) // g
        pad = n_groups * g - D

        def quant_err(x):
            xp = F.pad(x, (0, pad)) if pad > 0 else x
            xg = xp.reshape(B, H, T, n_groups, g)
            amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            sc = amax / 7.0
            xq = (xg / sc).round().clamp(-8, 7)
            xh = (xq * sc).reshape(B, H, T, n_groups * g)
            xh = xh[:, :, :, :D]
            return ((x.float() - xh.float()) ** 2).mean().sqrt().item()

        k_err = quant_err(k)
        v_err = quant_err(v)
        k_norm = k.float().norm().item() / math.sqrt(k.numel())
        v_norm = v.float().norm().item() / math.sqrt(v.numel())

        ratio = (k_err + v_err) / max(k_norm + v_norm, 1e-10)
        ratios.append({"layer": li, "residual_ratio": round(ratio, 6)})

    # Sort by ratio descending (most sensitive first)
    ratios.sort(key=lambda x: x["residual_ratio"], reverse=True)
    ranking = [r["layer"] for r in ratios]
    return ranking, ratios


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Replicate best schedule on larger model."""
    outdir = os.path.join(args.outdir, "phase2")
    art_dir = os.path.join(args.outdir, "artifacts", "v22")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 2: Larger Model Replication")
    print(f"  Model: {args.model}")
    print("=" * 60)

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    print(
        f"  Architecture: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}"
    )

    # Step 2a: Compute sensitive layers using residual_ratio proxy
    print("\n  Step 2a: Computing layer sensitivity via residual_ratio")
    ranking, ratios = compute_residual_ratio(
        model, token_data, args.device, model_config, n_layers
    )
    print(f"    Top-6 sensitive layers: {ranking[:6]}")
    for r in ratios[:6]:
        print(f"      Layer {r['layer']}: ratio={r['residual_ratio']:.6f}")

    # Step 2b: Byte accounting for this model
    print("\n  Step 2b: Byte accounting")
    acct_table = OrderedDict()
    for k in [0, 2, 4, 6]:
        name = f"g32_k{k}"
        acct_table[name] = compute_kv_bytes_per_token(
            n_layers, n_kv_heads, head_dim, 4, 32, k_int8_layers=k
        )
        print(f"    {name}: kv_ratio={acct_table[name]['kv_ratio']:.4f}")

    acct_table["INT8_all"] = compute_kv_bytes_per_token(
        n_layers, n_kv_heads, head_dim, 8, k_int8_layers=n_layers
    )
    print(f"    INT8_all: kv_ratio={acct_table['INT8_all']['kv_ratio']:.4f}")

    # Step 2c: Dense baselines
    # For larger model, limit L if OOM risk
    test_L = [L for L in valid_L if L <= max_ctx]
    # Try 32K but catch OOM
    if 32768 in test_L:
        try:
            torch.cuda.empty_cache()
            # Quick memory check
            mem_free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            # Rough estimate: 1.5B at 32K needs ~7GB model + ~3GB KV
            if mem_free < 15 * 1024**3:
                print(f"    WARNING: Only {mem_free/1024**3:.1f}GB free, skipping 32K")
                test_L = [L for L in test_L if L <= 16384]
        except Exception:
            pass

    print(f"\n  Step 2c: Dense baselines at L={test_L}")
    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        test_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    # Step 2d: Test g=32 k=4 with model-specific ranking
    results = {}

    for k in [4, 6]:
        sched = build_k_schedule(ranking, k, n_layers=n_layers)
        be = GroupedMixedBackend(layer_bits=sched, group_size=32)
        name = f"g32_k{k}_residual"

        print(f"\n  Testing: {name}")
        evals = eval_config(
            be,
            model,
            token_data,
            test_L,
            args.seeds,
            args.device,
            max_ctx,
            model_config,
            dense_ppls,
            args.decode_steps,
        )

        p3 = check_pass(evals, 3.0)
        md = max_delta(evals)
        ratio = acct_table[f"g32_k{k}"]["kv_ratio"]

        results[name] = {
            "k": k,
            "ranking": ranking[:k],
            "evals": evals,
            "pass_3pct": p3,
            "max_delta": round(md, 2),
            "kv_ratio": ratio,
        }

        status = "PASS" if p3 else "FAIL"
        print(f"    max_delta={md:+.2f}% {status} kv_ratio={ratio:.4f}")

    # Step 2e: Test with transferred 0.5B ranking [0,8,1,2]
    # Layers 0,8,1,2 from 0.5B — adapt if n_layers differs
    transferred_ranking = [0, 8, 1, 2, 3, 4, 21, 20, 9, 11, 16, 5]
    # Clip to valid range
    transferred_ranking = [l for l in transferred_ranking if l < n_layers]

    for k in [4, 6]:
        sched = build_k_schedule(transferred_ranking, k, n_layers=n_layers)
        be = GroupedMixedBackend(layer_bits=sched, group_size=32)
        name = f"g32_k{k}_transferred"

        print(f"\n  Testing: {name}")
        evals = eval_config(
            be,
            model,
            token_data,
            test_L,
            args.seeds,
            args.device,
            max_ctx,
            model_config,
            dense_ppls,
            args.decode_steps,
        )

        p3 = check_pass(evals, 3.0)
        md = max_delta(evals)
        ratio = acct_table[f"g32_k{k}"]["kv_ratio"]

        results[name] = {
            "k": k,
            "ranking": transferred_ranking[:k],
            "evals": evals,
            "pass_3pct": p3,
            "max_delta": round(md, 2),
            "kv_ratio": ratio,
        }

        status = "PASS" if p3 else "FAIL"
        print(f"    max_delta={md:+.2f}% {status} kv_ratio={ratio:.4f}")

    # Step 2f: INT8-all baseline
    sched_int8 = [8] * n_layers
    be_int8 = GroupedMixedBackend(layer_bits=sched_int8, group_size=32)
    name = "INT8_all"
    print(f"\n  Testing: {name}")
    evals = eval_config(
        be_int8,
        model,
        token_data,
        test_L,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
        args.decode_steps,
    )
    p3 = check_pass(evals, 3.0)
    md = max_delta(evals)
    results[name] = {
        "k": n_layers,
        "evals": evals,
        "pass_3pct": p3,
        "max_delta": round(md, 2),
        "kv_ratio": acct_table["INT8_all"]["kv_ratio"],
    }
    print(f"    max_delta={md:+.2f}% {'PASS' if p3 else 'FAIL'}")

    # k/D comparison
    print(f"\n  k/D Comparison:")
    print(f"    0.5B: k*=4, D=24, k/D={4/24:.4f}, kv_ratio=0.3203")
    for name, res in results.items():
        if "g32" in name and res.get("pass_3pct"):
            k = res["k"]
            kd = k / n_layers
            print(
                f"    {args.model}: {name} k={k}, D={n_layers}, "
                f"k/D={kd:.4f}, kv_ratio={res['kv_ratio']:.4f}"
            )

    # Save results
    phase2_out = {
        "model": args.model,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "test_L": test_L,
        "layer_sensitivity": ratios,
        "ranking": ranking,
        "accounting": acct_table,
        "results": results,
    }

    with open(os.path.join(art_dir, "larger_model_replication.json"), "w") as f:
        json.dump(phase2_out, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "version": "v22",
        "model": args.model,
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 2 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Phase 3: Bandwidth-bound regime proof
# ============================================================


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Bandwidth-bound regime profiling."""
    outdir = os.path.join(args.outdir, "phase3")
    art_dir = os.path.join(args.outdir, "artifacts", "v22")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 3: Bandwidth-bound Regime Proof")
    print("=" * 60)

    theory_ranking = load_theory_ranking(args.theory_path)
    n_layers = model_config["n_layers"]

    # Test at L=16384 with increasing batch sizes
    L = 16384
    decode_steps = min(args.decode_steps, 512)

    batch_sizes = [1, 4, 8]
    # Check if 16 is feasible
    mem_free = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    )
    if mem_free > 30 * 1024**3:
        batch_sizes.append(16)

    configs = OrderedDict()

    # INT8 all
    sched_int8 = [8] * n_layers
    configs["INT8_all"] = GroupedMixedBackend(layer_bits=sched_int8, group_size=32)

    # g32 k=4
    sched_k4 = build_k_schedule(theory_ranking, 4, n_layers=n_layers)
    configs["g32_k4"] = GroupedMixedBackend(layer_bits=sched_k4, group_size=32)

    # Dense baseline
    configs["dense"] = DenseBackend()

    profile_results = {}

    for batch_size in batch_sizes:
        print(f"\n  Batch size = {batch_size}")
        profile_results[f"batch_{batch_size}"] = {}

        for name, be in configs.items():
            be.configure(L, model_config)
            be.calibrate(model, token_data, L, args.device, model_config)

            # Run 3 trials, measure total time
            times = []
            for trial in range(3):
                torch.cuda.empty_cache()
                gpu_sync(args.device)

                rng = np.random.RandomState(trial)
                idx = get_text_batch(token_data, batch_size, L + decode_steps, rng).to(
                    args.device
                )
                prefix = idx[:, :L]
                continuation = idx[:, L:]

                try:
                    gpu_sync(args.device)
                    t0 = time.perf_counter()
                    logits, stats = be.run_decode(
                        model, prefix, continuation, args.device, max_ctx
                    )
                    gpu_sync(args.device)
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    del logits, stats
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"    {name} batch={batch_size}: OOM")
                        torch.cuda.empty_cache()
                        break
                    raise

            if times:
                avg_ms = np.mean(times) * 1000
                ms_per_tok = avg_ms / decode_steps
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2

                profile_results[f"batch_{batch_size}"][name] = {
                    "avg_total_ms": round(avg_ms, 2),
                    "ms_per_token": round(ms_per_tok, 2),
                    "peak_gpu_mb": round(peak_mb, 1),
                    "n_trials": len(times),
                }
                print(f"    {name}: {ms_per_tok:.2f} ms/tok, " f"peak={peak_mb:.0f}MB")

    # Compute speedup ratios
    speedup_analysis = {}
    for batch_key, batch_data in profile_results.items():
        if "dense" in batch_data and "g32_k4" in batch_data:
            dense_ms = batch_data["dense"]["ms_per_token"]
            g32_ms = batch_data["g32_k4"]["ms_per_token"]
            int8_ms = batch_data.get("INT8_all", {}).get("ms_per_token", dense_ms)

            speedup_analysis[batch_key] = {
                "dense_ms": dense_ms,
                "g32_k4_ms": g32_ms,
                "int8_ms": int8_ms,
                "g32_vs_dense_speedup": (
                    round(dense_ms / g32_ms, 3) if g32_ms > 0 else None
                ),
                "g32_vs_int8_speedup": (
                    round(int8_ms / g32_ms, 3) if g32_ms > 0 else None
                ),
                "bandwidth_bound": g32_ms < dense_ms * 0.95,
            }

    print(f"\n  Speedup Analysis:")
    for batch_key, sa in speedup_analysis.items():
        bw = "YES" if sa["bandwidth_bound"] else "NO"
        print(
            f"    {batch_key}: dense={sa['dense_ms']:.2f} "
            f"g32_k4={sa['g32_k4_ms']:.2f} "
            f"speedup={sa.get('g32_vs_dense_speedup', 'N/A')}x "
            f"bandwidth_bound={bw}"
        )

    bw_profile = {
        "model": args.model,
        "L": L,
        "decode_steps": decode_steps,
        "profiles": profile_results,
        "speedup_analysis": speedup_analysis,
    }

    with open(os.path.join(art_dir, "bandwidth_bound_profile.json"), "w") as f:
        json.dump(bw_profile, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "version": "v22",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 3 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Phase 4: Theory update + final deliverables
# ============================================================


def run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 4: Theory update and final deliverables."""
    outdir = os.path.join(args.outdir, "phase4")
    art_dir = os.path.join(args.outdir, "artifacts", "v22")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 4: Theory Update + Final Deliverables")
    print("=" * 60)

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    # Step 4a: Measure effective sigma for g=32 and g=4
    print("\n  Step 4a: Measuring per-layer sigma")
    rng = np.random.RandomState(0)
    cal_len = 4096
    idx = get_text_batch(token_data, 1, cal_len, rng).to(args.device)

    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(idx, use_cache=True)
        past = out.past_key_values
    del out

    sigma_results = {"g32": [], "g4": []}

    for li in range(n_layers):
        k, v = past[li]
        B, H, T, D = k.shape

        for g, key in [(32, "g32"), (4, "g4")]:
            n_groups = (D + g - 1) // g
            pad = n_groups * g - D

            def compute_sigma(x):
                xp = F.pad(x, (0, pad)) if pad > 0 else x
                xg = xp.reshape(B, H, T, n_groups, g)
                amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                sc = amax / 7.0
                xq = (xg / sc).round().clamp(-8, 7)
                xh = (xq * sc).reshape(B, H, T, n_groups * g)[:, :, :, :D]
                return ((x.float() - xh.float()) ** 2).mean().sqrt().item()

            k_sigma = compute_sigma(k)
            v_sigma = compute_sigma(v)
            total = math.sqrt(k_sigma**2 + v_sigma**2)
            sigma_results[key].append(
                {
                    "layer": li,
                    "k_sigma": round(k_sigma, 6),
                    "v_sigma": round(v_sigma, 6),
                    "total": round(total, 6),
                }
            )

    # Theory prediction: k* = ceil(sum_over_layers(sigma4 > threshold))
    # Threshold from v20: total unprotected noise < ~7.6
    sigma_g32 = sorted(sigma_results["g32"], key=lambda x: x["total"], reverse=True)
    sigma_g4 = sorted(sigma_results["g4"], key=lambda x: x["total"], reverse=True)

    print(f"\n  Top-6 layers by sigma (g=32):")
    for s in sigma_g32[:6]:
        print(f"    Layer {s['layer']}: sigma={s['total']:.6f}")

    print(f"\n  Top-6 layers by sigma (g=4):")
    for s in sigma_g4[:6]:
        print(f"    Layer {s['layer']}: sigma={s['total']:.6f}")

    # Total sigma comparison
    total_g32 = sum(s["total"] for s in sigma_g32)
    total_g4 = sum(s["total"] for s in sigma_g4)
    reduction = (1 - total_g4 / total_g32) * 100 if total_g32 > 0 else 0
    print(
        f"\n  Total sigma: g=32={total_g32:.4f}, g=4={total_g4:.4f} ({reduction:.1f}% reduction)"
    )

    theory = {
        "sigma_g32": sigma_results["g32"],
        "sigma_g4": sigma_results["g4"],
        "total_sigma_g32": round(total_g32, 6),
        "total_sigma_g4": round(total_g4, 6),
        "sigma_reduction_pct": round(reduction, 2),
    }

    # Step 4b: Load all phase results and build deliverables
    print("\n  Step 4b: Building deliverables")

    # Load phase results
    p0_path = os.path.join(args.outdir, "phase0", "regression_results.json")
    p1_path = os.path.join(args.outdir, "phase1", "phase1_results.json")
    p2_path = os.path.join(art_dir, "larger_model_replication.json")
    p3_path = os.path.join(art_dir, "bandwidth_bound_profile.json")
    scale_path = os.path.join(art_dir, "scale_overhead_study.json")

    phase_data = {}
    for name, path in [
        ("p0", p0_path),
        ("p1", p1_path),
        ("p2", p2_path),
        ("p3", p3_path),
        ("scale", scale_path),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                phase_data[name] = json.load(f)
        else:
            phase_data[name] = {}

    # Build scoreboard
    scoreboard = {
        "model": "Qwen2.5-0.5B",
        "experiment": "BPA v22: Scale overhead + larger model + bandwidth",
        "baselines": {
            "dense": {"kv_ratio": 1.0, "pass_3pct": "all"},
            "INT8_all": {"kv_ratio": 0.5156, "pass_3pct": "all"},
            "S2_k6": {"kv_ratio": 0.333, "pass_3pct": "all"},
            "g32_k4_v21": {"kv_ratio": 0.3203, "pass_3pct": "all"},
        },
        "phase1_scale_engineering": {},
        "phase2_larger_model": {},
        "phase3_bandwidth": {},
        "theory_update": theory,
    }

    # Populate phase1 results
    p1 = phase_data.get("p1", {})
    for name, v in p1.get("validated", {}).items():
        scoreboard["phase1_scale_engineering"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta_allL"),
            "pass_3pct": v.get("pass_allL_3pct"),
        }
    for name, v in p1.get("reduced_k", {}).items():
        scoreboard["phase1_scale_engineering"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta"),
            "pass_3pct": v.get("pass_3pct"),
        }

    # Populate phase2 results
    p2 = phase_data.get("p2", {})
    for name, v in p2.get("results", {}).items():
        scoreboard["phase2_larger_model"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta"),
            "pass_3pct": v.get("pass_3pct"),
            "k": v.get("k"),
        }

    # Populate phase3 results
    p3 = phase_data.get("p3", {})
    scoreboard["phase3_bandwidth"] = p3.get("speedup_analysis", {})

    with open("bpa_v22_scoreboard.json", "w") as f:
        json.dump(scoreboard, f, indent=2)

    # Save theory to artifacts
    with open(os.path.join(art_dir, "theory_update.json"), "w") as f:
        json.dump(theory, f, indent=2)

    print("\n  Deliverables written:")
    print("    - bpa_v22_scoreboard.json")
    print("    (Final report and branch tree to be written manually)")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "version": "v22",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 4 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="BPA v22: Scale overhead + larger model + bandwidth"
    )
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--model", default="qwen05b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--L",
        nargs="+",
        type=int,
        default=[8192, 16384, 32768],
    )
    parser.add_argument("--outdir", default="results/v22")
    parser.add_argument(
        "--theory_path",
        default="results/v20/artifacts/v20/theory_fit.json",
    )
    args = parser.parse_args()

    gpu_info = gpu_preflight(args.device)
    model, tokenizer, max_ctx, model_config = load_model(args.model, args.device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)
    valid_L = [L for L in args.L if L <= max_ctx]

    if args.phase == 0:
        run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 1:
        run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 2:
        run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 3:
        run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 4:
        run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
