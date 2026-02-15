#!/usr/bin/env python
"""
BPA v20 Evaluation: Break the k-floor.

Formalizes INT4 accumulation theory, then tests three attack paths
(A: reduce accumulation, B: reduce per-layer damage, C: structural)
to determine whether k* (minimum INT8 layers for PASS) can be lowered
below the v16/v19 floor of k=6.

Phases:
  0: Baseline lock + k-sweep curve (quality vs k)
  1: Theory: accumulation model + lower bound on k
  2: Path A interventions (per-head, V-only INT8, INT6 approx)
  3: Path B interventions (per-channel quant, learned scales)
  4: Path C interventions (rope_complex stacking, rescaling)
  5: Unified comparison + final deliverables

Usage:
    python eval_v20.py --phase 0
    python eval_v20.py --phase 1
    ...
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
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


# ============================================================
# Oracle ranking utility
# ============================================================


def load_oracle_ranking(oracle_path):
    """Load oracle per-layer sensitivity from v19 artifacts.

    Returns list of layers sorted from MOST sensitive to LEAST
    sensitive (descending oracle delta).
    """
    with open(oracle_path) as f:
        data = json.load(f)
    ranked = data["int4_ranked_tolerant_to_sensitive"]
    # ranked is tolerant -> sensitive; reverse for sensitive -> tolerant
    layers_sens_desc = [e["layer"] for e in reversed(ranked)]
    deltas = {e["layer"]: e["ppl_delta_pct"] for e in ranked}
    return layers_sens_desc, deltas


def build_k_schedule(oracle_ranking, k, n_layers=24):
    """Build INT4/INT8 schedule: top-k sensitive layers get INT8."""
    schedule = [4] * n_layers
    for i in range(min(k, n_layers)):
        schedule[oracle_ranking[i]] = 8
    return schedule


# ============================================================
# V-only INT8 backend (Path A3)
# ============================================================


class VOnlyINT8Backend:
    """Mixed-precision with V at INT8, K stays at INT4 for top-k layers."""

    def __init__(self, v_int8_layers, block_size=32):
        self.v_int8_layers = set(v_int8_layers)
        self.block_size = block_size
        self._name = f"Vonly_INT8_k{len(v_int8_layers)}"

    @property
    def name(self):
        return self._name

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n_v8 = len(self.v_int8_layers)
        n_layers = model_config["n_layers"]
        print(f"    {self.name} calibrated: {n_layers} K=INT4, {n_v8} V=INT8")

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        elem = 2

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
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
                total_bytes = 0
                for li in range(n_layers):
                    k, v = past[li]
                    k_sink = k[:, :, : self.W_sink, :]
                    v_sink = v[:, :, : self.W_sink, :]
                    k_far = k[:, :, self.W_sink : far_end, :]
                    v_far = v[:, :, self.W_sink : far_end, :]
                    k_near = k[:, :, far_end:, :]
                    v_near = v[:, :, far_end:, :]

                    # K always INT4
                    k_q, k_s = quantize_int4_block(k_far, self.block_size)
                    k_hat = dequantize_int4_block(k_q, k_s, self.block_size).to(dtype)
                    k_bytes = k_far.numel() // 2  # 4-bit

                    # V: INT8 for selected layers, INT4 otherwise
                    if li in self.v_int8_layers:
                        v_q, v_s = quantize_int8_symmetric(v_far)
                        v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
                        v_bytes = v_far.numel()  # 8-bit
                    else:
                        v_q, v_s = quantize_int4_block(v_far, self.block_size)
                        v_hat = dequantize_int4_block(v_q, v_s, self.block_size).to(
                            dtype
                        )
                        v_bytes = v_far.numel() // 2

                    total_bytes += k_bytes + v_bytes

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
                    step=step,
                    cache_len=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                    decode_ms=0,
                    ppl_step=0,
                    kv_bytes_ratio=0,
                    peak_gpu_mb=0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


# ============================================================
# Per-channel INT4 backend (Path B2)
# ============================================================


class PerChannelINT4Backend:
    """INT4 with per-channel (per-head-dim) scales instead of per-block."""

    def __init__(self, layer_bits, group_size=8):
        self.layer_bits = layer_bits
        self.group_size = group_size
        self._name = f"perchan_g{group_size}"

    @property
    def name(self):
        return self._name

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n4 = sum(1 for b in self.layer_bits if b == 4)
        n8 = sum(1 for b in self.layer_bits if b == 8)
        print(f"    {self.name} calibrated: {n4} INT4(g={self.group_size}), {n8} INT8")

    def _quantize_int4_perchannel(self, x, group_size):
        """Per-channel-group INT4 quantization.

        Instead of block-wise (across contiguous elements), this
        quantizes in groups along the head_dim, giving each group
        its own scale factor. More accurate for structured data.
        """
        B, H, T, D = x.shape
        n_groups = (D + group_size - 1) // group_size
        pad = n_groups * group_size - D
        if pad > 0:
            x = F.pad(x, (0, pad))
        x_g = x.reshape(B, H, T, n_groups, group_size)
        amax = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / 7.0
        x_q = (x_g / scale).round().clamp(-8, 7).to(torch.int8)
        return x_q, scale, D

    def _dequantize_int4_perchannel(self, x_q, scale, orig_D):
        x_hat = x_q.float() * scale
        B, H, T, ng, gs = x_hat.shape
        x_hat = x_hat.reshape(B, H, T, ng * gs)
        return x_hat[:, :, :, :orig_D]

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
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
                        # Per-channel-group INT4
                        k_q, k_s, k_D = self._quantize_int4_perchannel(
                            k_far, self.group_size
                        )
                        k_hat = self._dequantize_int4_perchannel(k_q, k_s, k_D).to(
                            dtype
                        )
                        v_q, v_s, v_D = self._quantize_int4_perchannel(
                            v_far, self.group_size
                        )
                        v_hat = self._dequantize_int4_perchannel(v_q, v_s, v_D).to(
                            dtype
                        )

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
                    step=step,
                    cache_len=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                    decode_ms=0,
                    ppl_step=0,
                    kv_bytes_ratio=0,
                    peak_gpu_mb=0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


# ============================================================
# KV Rescaling backend (Path C3)
# ============================================================


class RescaledQuantBackend:
    """Quantize KV then rescale to match clean statistics (mean/var)."""

    def __init__(self, layer_bits, block_size=32):
        self.layer_bits = layer_bits
        self.block_size = block_size
        self._name = "rescaled_quant"

    @property
    def name(self):
        return self._name

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n4 = sum(1 for b in self.layer_bits if b == 4)
        n8 = sum(1 for b in self.layer_bits if b == 8)
        print(f"    {self.name} calibrated: {n4} INT4, {n8} INT8 (with rescale)")

    def _rescale_to_match(self, x_hat, x_clean):
        """Rescale x_hat per-head to match x_clean mean/std."""
        # Per-head stats across T dimension
        clean_mean = x_clean.mean(dim=2, keepdim=True)
        clean_std = x_clean.std(dim=2, keepdim=True).clamp(min=1e-6)
        hat_mean = x_hat.mean(dim=2, keepdim=True)
        hat_std = x_hat.std(dim=2, keepdim=True).clamp(min=1e-6)
        return (x_hat - hat_mean) * (clean_std / hat_std) + clean_mean

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
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
                        k_q, k_s = quantize_int4_block(k_far, self.block_size)
                        k_hat = dequantize_int4_block(k_q, k_s, self.block_size).to(
                            dtype
                        )
                        v_q, v_s = quantize_int4_block(v_far, self.block_size)
                        v_hat = dequantize_int4_block(v_q, v_s, self.block_size).to(
                            dtype
                        )
                        # Rescale INT4 to match clean stats
                        k_hat = self._rescale_to_match(k_hat, k_far)
                        v_hat = self._rescale_to_match(v_hat, v_far)

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
                    step=step,
                    cache_len=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                    decode_ms=0,
                    ppl_step=0,
                    kv_bytes_ratio=0,
                    peak_gpu_mb=0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


# ============================================================
# Evaluation helper: run one backend at one (L, seed)
# ============================================================


def quick_eval(backend, model, token_data, L, seed, device_str, max_ctx, model_config):
    """Run a single eval, return ppl and stats."""
    result = run_single_eval(
        backend, model, token_data, L, 256, seed, device_str, max_ctx, model_config
    )
    return result


def eval_at_all_L(
    backend,
    model,
    token_data,
    valid_L,
    seeds,
    device_str,
    max_ctx,
    model_config,
    dense_ppls,
):
    """Evaluate backend at all L and seeds, return summary dict."""
    results = {}
    for L in valid_L:
        for seed in seeds:
            r = run_single_eval(
                backend,
                model,
                token_data,
                L,
                256,
                seed,
                device_str,
                max_ctx,
                model_config,
            )
            key = f"L{L}_s{seed}"
            dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
            delta = (r.ppl - dense_ref) / dense_ref * 100 if dense_ref > 0 else 0
            results[key] = {
                "ppl": round(r.ppl, 4),
                "delta_pct": round(delta, 2),
                "pass_1pct": abs(delta) <= 1.0,
                "pass_3pct": abs(delta) <= 3.0,
                "p50_ms": round(r.p50_ms, 2),
                "kv_bytes_ratio": round(r.kv_bytes_ratio, 4),
            }
    return results


# ============================================================
# Phase 0: Baseline lock + k-sweep
# ============================================================


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Baseline lock + k-sweep curve."""
    outdir = os.path.join(args.outdir, "phase0")
    art_dir = os.path.join(args.outdir, "artifacts", "v20")
    ksweep_dir = os.path.join(art_dir, "k_sweeps")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ksweep_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 0: Baseline lock + k-sweep curve")
    print("=" * 60)

    # 0.1 Baselines
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

    int8_be = QuantBackend()
    schedules = build_schedules(args.sensitivity_path)
    s2_sched = schedules["S2"]
    s2_be = MixedPrecisionBackend(layer_bits=s2_sched)
    s2_be._name = "S2_manual"

    backends = [("INT8", int8_be), ("S2_manual", s2_be)]
    comp_results = run_backend_sweep(
        backends,
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
    )

    all_results = dense_results + comp_results
    save_results(all_results, dense_ppls, outdir, 0, {"gpu_info": gpu_info})

    # 0.2 k-sweep using oracle ranking
    oracle_ranking, oracle_deltas = load_oracle_ranking(args.oracle_path)
    k_values = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]
    k_curve = {"oracle_ranking": oracle_ranking, "k_results": {}}

    for k in k_values:
        schedule = build_k_schedule(oracle_ranking, k)
        be = MixedPrecisionBackend(layer_bits=schedule)
        be._name = f"oracle_k{k}"
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        k_data = {"k": k, "schedule": schedule, "evals": {}}

        # Quick eval at L=8192 for all seeds
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
            dense_ref = dense_ppls.get(f"8192_{seed}", r.ppl)
            delta = (r.ppl - dense_ref) / dense_ref * 100
            k_data["evals"][f"8K_s{seed}"] = {
                "ppl": round(r.ppl, 4),
                "delta_pct": round(delta, 2),
                "pass_3pct": abs(delta) <= 3.0,
                "pass_1pct": abs(delta) <= 1.0,
            }
            print(f"  k={k:2d} L=8K s={seed}: PPL={r.ppl:.4f} delta={delta:+.2f}%")

        # Validate at L=16K and 32K for key k values
        if k in [0, 2, 4, 6, 8, 12]:
            for L in [16384, 32768]:
                if L not in valid_L:
                    continue
                be2 = MixedPrecisionBackend(layer_bits=schedule)
                be2._name = f"oracle_k{k}"
                be2.configure(L, model_config)
                be2.calibrate(model, token_data, L, args.device, model_config)
                for seed in args.seeds:
                    r = run_single_eval(
                        be2,
                        model,
                        token_data,
                        L,
                        args.decode_steps,
                        seed,
                        args.device,
                        max_ctx,
                        model_config,
                    )
                    Lk = f"{L // 1024}K"
                    dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
                    delta = (r.ppl - dense_ref) / dense_ref * 100
                    k_data["evals"][f"{Lk}_s{seed}"] = {
                        "ppl": round(r.ppl, 4),
                        "delta_pct": round(delta, 2),
                        "pass_3pct": abs(delta) <= 3.0,
                        "pass_1pct": abs(delta) <= 1.0,
                    }
                    print(
                        f"  k={k:2d} L={Lk} s={seed}: PPL={r.ppl:.4f} delta={delta:+.2f}%"
                    )

        k_curve["k_results"][str(k)] = k_data

    # Compute k* (minimum k for all PASS@3% across all evals)
    for tol_name, tol in [("3pct", 3.0), ("1pct", 1.0)]:
        for k in sorted(k_values):
            kd = k_curve["k_results"][str(k)]
            all_pass = all(abs(e["delta_pct"]) <= tol for e in kd["evals"].values())
            if all_pass:
                k_curve[f"k_star_{tol_name}"] = k
                print(f"\n  k* for tol={tol}%: {k}")
                break
        else:
            k_curve[f"k_star_{tol_name}"] = f">{max(k_values)}"
            print(f"\n  k* for tol={tol}%: >{max(k_values)}")

    with open(os.path.join(ksweep_dir, "k_curve.json"), "w") as f:
        json.dump(k_curve, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 0,
        "version": "v20",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 0 complete. Saved to {outdir}/ and {ksweep_dir}/")


# ============================================================
# Phase 1: Theory — accumulation model
# ============================================================


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Fit accumulation theory and predict k*."""
    art_dir = os.path.join(args.outdir, "artifacts", "v20")
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 1: Accumulation theory + k-floor bound")
    print("=" * 60)

    # Load k-curve from Phase 0
    kcurve_path = os.path.join(art_dir, "k_sweeps", "k_curve.json")
    with open(kcurve_path) as f:
        k_curve = json.load(f)

    # Load oracle per-layer deltas
    oracle_ranking, oracle_deltas = load_oracle_ranking(args.oracle_path)

    # Measure residual distortion per layer under INT4
    n_layers = model_config["n_layers"]
    print("\n  Measuring per-layer residual distortion under INT4...")

    rng = np.random.RandomState(0)
    cal_len = 4096
    idx = get_text_batch(token_data, 1, cal_len, rng).to(args.device)

    with torch.no_grad():
        out = model(idx, use_cache=True)
        past_clean = out.past_key_values

    sigma_int4 = []
    sigma_int8 = []
    for li in range(n_layers):
        k, v = past_clean[li]
        # INT4 error
        k_q4, k_s4 = quantize_int4_block(k)
        k_hat4 = dequantize_int4_block(k_q4, k_s4).to(k.dtype)
        v_q4, v_s4 = quantize_int4_block(v)
        v_hat4 = dequantize_int4_block(v_q4, v_s4).to(v.dtype)
        err4 = ((k - k_hat4) ** 2).mean().item() + ((v - v_hat4) ** 2).mean().item()
        sigma_int4.append(math.sqrt(err4))

        # INT8 error
        k_q8, k_s8 = quantize_int8_symmetric(k)
        k_hat8 = dequantize_int8_symmetric(k_q8, k_s8).to(k.dtype)
        v_q8, v_s8 = quantize_int8_symmetric(v)
        v_hat8 = dequantize_int8_symmetric(v_q8, v_s8).to(v.dtype)
        err8 = ((k - k_hat8) ** 2).mean().item() + ((v - v_hat8) ** 2).mean().item()
        sigma_int8.append(math.sqrt(err8))

    # Theory: accumulation model
    # Under all-INT4 with k protected layers at INT8:
    # Total noise ~ sum_{l not protected} sigma4_l + sum_{l protected} sigma8_l
    # = sum_all(sigma4_l) - sum_protected(sigma4_l - sigma8_l)
    # Must be <= B (budget threshold for tol%)
    #
    # k* = min k such that sum of top-k (sigma4 - sigma8) removes enough noise
    #
    # Sort layers by (sigma4 - sigma8) descending = same as sorting by
    # INT4 damage since INT8 is essentially lossless

    delta_sigma = [sigma_int4[i] - sigma_int8[i] for i in range(n_layers)]
    total_noise_all_int4 = sum(sigma_int4)
    total_noise_all_int8 = sum(sigma_int8)

    # Sort by delta descending (most benefit from upgrading)
    layer_by_benefit = sorted(
        range(n_layers), key=lambda i: delta_sigma[i], reverse=True
    )

    # Compute cumulative noise reduction as we upgrade layers
    cumulative_reduction = []
    cum = 0
    for i, li in enumerate(layer_by_benefit):
        cum += delta_sigma[li]
        remaining = total_noise_all_int4 - cum
        cumulative_reduction.append(
            {
                "k": i + 1,
                "layer_upgraded": li,
                "delta_sigma": round(delta_sigma[li], 6),
                "remaining_noise": round(remaining, 6),
                "noise_ratio_vs_int8": (
                    round(remaining / total_noise_all_int8, 4)
                    if total_noise_all_int8 > 0
                    else 0
                ),
            }
        )

    # Fit threshold B from k-curve
    # For each k in k_curve, find max |delta_pct| across 8K evals
    k_results = k_curve["k_results"]
    k_to_max_delta = {}
    for k_str, kd in k_results.items():
        k_val = int(k_str)
        max_d = max(abs(e["delta_pct"]) for e in kd["evals"].values())
        k_to_max_delta[k_val] = max_d

    # Map k -> remaining noise from theory
    k_to_remaining = {}
    k_to_remaining[0] = total_noise_all_int4
    cum = 0
    for i, li in enumerate(layer_by_benefit):
        cum += delta_sigma[li]
        k_to_remaining[i + 1] = total_noise_all_int4 - cum

    # Fit linear model: max_delta_pct ≈ alpha * remaining_noise + beta
    ks_both = sorted(set(k_to_max_delta.keys()) & set(k_to_remaining.keys()))
    if len(ks_both) >= 3:
        x_fit = np.array([k_to_remaining[k] for k in ks_both])
        y_fit = np.array([k_to_max_delta[k] for k in ks_both])
        # Least squares: y = a*x + b
        A_mat = np.column_stack([x_fit, np.ones_like(x_fit)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y_fit, rcond=None)
        alpha_fit, beta_fit = coeffs
    else:
        alpha_fit, beta_fit = 1.0, 0.0

    # Predict k* from theory
    # Find k where alpha * remaining_noise + beta <= tol
    predicted_k = {}
    for tol_name, tol in [("3pct", 3.0), ("1pct", 1.0)]:
        threshold_noise = (tol - beta_fit) / alpha_fit if alpha_fit > 0 else 0
        found = False
        for cr in cumulative_reduction:
            if cr["remaining_noise"] <= threshold_noise:
                predicted_k[tol_name] = cr["k"]
                found = True
                break
        if not found:
            predicted_k[tol_name] = n_layers

    # Observed k* from Phase 0
    observed_k = {}
    for tol_name in ["3pct", "1pct"]:
        val = k_curve.get(f"k_star_{tol_name}", ">12")
        observed_k[tol_name] = val

    theory_fit = {
        "model": "additive_noise_accumulation",
        "n_layers": n_layers,
        "total_noise_all_int4": round(total_noise_all_int4, 6),
        "total_noise_all_int8": round(total_noise_all_int8, 6),
        "per_layer_sigma_int4": [round(s, 6) for s in sigma_int4],
        "per_layer_sigma_int8": [round(s, 6) for s in sigma_int8],
        "per_layer_delta_sigma": [round(d, 6) for d in delta_sigma],
        "layer_upgrade_order": layer_by_benefit,
        "cumulative_reduction": cumulative_reduction,
        "linear_fit": {
            "alpha": round(alpha_fit, 4),
            "beta": round(beta_fit, 4),
            "description": "max_delta_pct ≈ alpha * remaining_noise + beta",
        },
        "predicted_k_star": predicted_k,
        "observed_k_star": observed_k,
        "k_to_max_delta": {str(k): round(d, 2) for k, d in k_to_max_delta.items()},
    }

    with open(os.path.join(art_dir, "theory_fit.json"), "w") as f:
        json.dump(theory_fit, f, indent=2)

    # Write theory markdown
    theory_md = f"""# Accumulation Theory: k-floor Bound

## Model

Quantization introduces per-layer noise epsilon_l to KV cache.
The total output distortion accumulates approximately additively
across layers (confirmed empirically in v15):

  Delta_total ≈ sum_l w_l * epsilon_l

where w_l ≈ 1 (amplification factor, close to unity by v12 finding).

## Noise Estimates

| Precision | Total noise (RMS) | Per-layer mean |
|-----------|-------------------|----------------|
| INT4      | {total_noise_all_int4:.4f}        | {total_noise_all_int4/n_layers:.6f}           |
| INT8      | {total_noise_all_int8:.4f}        | {total_noise_all_int8/n_layers:.6f}           |

INT4/INT8 noise ratio: {total_noise_all_int4/total_noise_all_int8:.1f}x

## Per-layer sigma (top 6 by delta):

| Layer | sigma4    | sigma8    | delta     |
|-------|-----------|-----------|-----------|
"""
    for i in range(min(6, n_layers)):
        li = layer_by_benefit[i]
        theory_md += (
            f"| {li:5d} | {sigma_int4[li]:.6f} | {sigma_int8[li]:.6f} "
            f"| {delta_sigma[li]:.6f} |\n"
        )

    theory_md += f"""
## Fitted Bound

Linear relationship between remaining noise and PPL degradation:

  max_delta_pct ≈ {alpha_fit:.4f} * remaining_noise + {beta_fit:.4f}

For PASS@3%: need remaining_noise <= {(3.0 - beta_fit) / alpha_fit:.4f}
For PASS@1%: need remaining_noise <= {(1.0 - beta_fit) / alpha_fit:.4f}

## Predicted vs Observed k*

| Tolerance | Predicted k* | Observed k* |
|-----------|-------------|-------------|
| 3%        | {predicted_k['3pct']}           | {observed_k['3pct']}           |
| 1%        | {predicted_k['1pct']}           | {observed_k['1pct']}           |

## Implications

To reduce k*, interventions must either:
1. Reduce sigma4 per layer (Path B: better quantization)
2. Reduce the number of layers contributing noise (Path A: selective protection)
3. Reduce amplification w_l (Path C: structural)

The theory predicts that sigma4 reduction by factor X reduces k* by
approximately X layers (since the top layers dominate).
"""

    with open(os.path.join(art_dir, "theory_k_floor.md"), "w") as f:
        f.write(theory_md)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "version": "v20",
        "gpu_info": gpu_info,
    }
    outdir = os.path.join(args.outdir, "phase1")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Theory fit: alpha={alpha_fit:.4f}, beta={beta_fit:.4f}")
    print(f"  Predicted k*@3%: {predicted_k['3pct']}, observed: {observed_k['3pct']}")
    print(f"  Predicted k*@1%: {predicted_k['1pct']}, observed: {observed_k['1pct']}")
    print(f"\nPhase 1 complete. Saved to {art_dir}/")


# ============================================================
# Phase 2: Path A — Reduce accumulation
# ============================================================


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Path A interventions."""
    outdir = os.path.join(args.outdir, "phase2")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 2: Path A — Reduce accumulation")
    print("=" * 60)

    # Load oracle ranking and dense baselines
    oracle_ranking, oracle_deltas = load_oracle_ranking(args.oracle_path)
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

    n_layers = model_config["n_layers"]
    results = {}

    # A3: V-only INT8 — keep K at INT4, upgrade V to INT8 for top-k
    print("\n  A3: V-only INT8 (K stays INT4)")
    a3_results = {}
    for k in [2, 4, 6, 8]:
        v_layers = oracle_ranking[:k]
        be = VOnlyINT8Backend(v_int8_layers=v_layers)
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        k_data = {"k": k, "v_int8_layers": v_layers, "evals": {}}
        for L in valid_L:
            be2 = VOnlyINT8Backend(v_int8_layers=v_layers)
            be2.configure(L, model_config)
            be2.calibrate(model, token_data, L, args.device, model_config)
            for seed in args.seeds:
                r = run_single_eval(
                    be2,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                )
                Lk = f"{L // 1024}K"
                dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
                delta = (r.ppl - dense_ref) / dense_ref * 100
                k_data["evals"][f"{Lk}_s{seed}"] = {
                    "ppl": round(r.ppl, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                    "kv_bytes_ratio": round(r.kv_bytes_ratio, 4),
                }
                print(f"    A3 k={k} L={Lk} s={seed}: delta={delta:+.2f}%")

        a3_results[f"k{k}"] = k_data
    results["A3_vonly_int8"] = a3_results

    # A2: Approximate INT6 (use group_size=4 INT4 as tighter quant)
    # This gives more scale factors = less noise per element
    print("\n  A2: Approximate INT6 (tight group INT4, g=4)")
    a2_results = {}
    for k in [0, 2, 4, 6]:
        schedule = build_k_schedule(oracle_ranking, k)
        be = PerChannelINT4Backend(layer_bits=schedule, group_size=4)
        be._name = f"tightgroup_k{k}"
        be.configure(8192, model_config)
        be.calibrate(model, token_data, 8192, args.device, model_config)

        k_data = {"k": k, "evals": {}}
        for L in valid_L:
            be2 = PerChannelINT4Backend(layer_bits=schedule, group_size=4)
            be2._name = f"tightgroup_k{k}"
            be2.configure(L, model_config)
            be2.calibrate(model, token_data, L, args.device, model_config)
            for seed in args.seeds:
                r = run_single_eval(
                    be2,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                )
                Lk = f"{L // 1024}K"
                dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
                delta = (r.ppl - dense_ref) / dense_ref * 100
                k_data["evals"][f"{Lk}_s{seed}"] = {
                    "ppl": round(r.ppl, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                    "kv_bytes_ratio": round(r.kv_bytes_ratio, 4),
                }
                print(f"    A2 k={k} L={Lk} s={seed}: delta={delta:+.2f}%")

        a2_results[f"k{k}"] = k_data
    results["A2_tight_group"] = a2_results

    with open(os.path.join(outdir, "path_a_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "version": "v20",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 2 complete. Saved to {outdir}/")


# ============================================================
# Phase 3: Path B — Reduce per-layer damage
# ============================================================


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Path B interventions."""
    outdir = os.path.join(args.outdir, "phase3")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 3: Path B — Reduce per-layer damage")
    print("=" * 60)

    oracle_ranking, oracle_deltas = load_oracle_ranking(args.oracle_path)
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

    results = {}

    # B2: Per-channel INT4 (group_size sweep)
    print("\n  B2: Per-channel INT4 quantization")
    b2_results = {}
    for gs in [8, 16, 32]:
        gs_data = {}
        for k in [0, 2, 4, 6]:
            schedule = build_k_schedule(oracle_ranking, k)
            be = PerChannelINT4Backend(layer_bits=schedule, group_size=gs)
            be._name = f"perchan_g{gs}_k{k}"

            k_data = {"k": k, "group_size": gs, "evals": {}}
            for L in valid_L:
                be2 = PerChannelINT4Backend(layer_bits=schedule, group_size=gs)
                be2._name = f"perchan_g{gs}_k{k}"
                be2.configure(L, model_config)
                be2.calibrate(model, token_data, L, args.device, model_config)
                for seed in args.seeds:
                    r = run_single_eval(
                        be2,
                        model,
                        token_data,
                        L,
                        args.decode_steps,
                        seed,
                        args.device,
                        max_ctx,
                        model_config,
                    )
                    Lk = f"{L // 1024}K"
                    dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
                    delta = (r.ppl - dense_ref) / dense_ref * 100
                    k_data["evals"][f"{Lk}_s{seed}"] = {
                        "ppl": round(r.ppl, 4),
                        "delta_pct": round(delta, 2),
                        "pass_3pct": abs(delta) <= 3.0,
                        "pass_1pct": abs(delta) <= 1.0,
                    }
                    print(f"    B2 g={gs} k={k} L={Lk} s={seed}: delta={delta:+.2f}%")
            gs_data[f"k{k}"] = k_data
        b2_results[f"g{gs}"] = gs_data
    results["B2_perchannel"] = b2_results

    with open(os.path.join(outdir, "path_b_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "version": "v20",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 3 complete. Saved to {outdir}/")


# ============================================================
# Phase 4: Path C — Structural modifications
# ============================================================


def run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 4: Path C interventions."""
    outdir = os.path.join(args.outdir, "phase4")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 4: Path C — Structural modifications")
    print("=" * 60)

    oracle_ranking, oracle_deltas = load_oracle_ranking(args.oracle_path)
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

    results = {}

    # C3: KV rescaling (match clean stats after quant)
    print("\n  C3: KV rescaling post-quantization")
    c3_results = {}
    for k in [0, 2, 4, 6]:
        schedule = build_k_schedule(oracle_ranking, k)
        be = RescaledQuantBackend(layer_bits=schedule)
        be._name = f"rescaled_k{k}"

        k_data = {"k": k, "evals": {}}
        for L in valid_L:
            be2 = RescaledQuantBackend(layer_bits=schedule)
            be2._name = f"rescaled_k{k}"
            be2.configure(L, model_config)
            be2.calibrate(model, token_data, L, args.device, model_config)
            for seed in args.seeds:
                r = run_single_eval(
                    be2,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                )
                Lk = f"{L // 1024}K"
                dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
                delta = (r.ppl - dense_ref) / dense_ref * 100
                k_data["evals"][f"{Lk}_s{seed}"] = {
                    "ppl": round(r.ppl, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                }
                print(f"    C3 k={k} L={Lk} s={seed}: delta={delta:+.2f}%")
        c3_results[f"k{k}"] = k_data
    results["C3_rescaled"] = c3_results

    # C2: Attention logit clamping (clamp QK to max magnitude)
    # This is inference-only: just clamp the attention logits
    # We implement it by clamping K values to reduce their norm
    print("\n  C2: K-value norm clamping (reduce amplification)")
    c2_results = {}
    for k in [0, 2, 4, 6]:
        schedule = build_k_schedule(oracle_ranking, k)
        be = NormClampBackend(layer_bits=schedule, clamp_percentile=0.99)
        be._name = f"normclamp_k{k}"

        k_data = {"k": k, "evals": {}}
        for L in valid_L:
            be2 = NormClampBackend(layer_bits=schedule, clamp_percentile=0.99)
            be2._name = f"normclamp_k{k}"
            be2.configure(L, model_config)
            be2.calibrate(model, token_data, L, args.device, model_config)
            for seed in args.seeds:
                r = run_single_eval(
                    be2,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                )
                Lk = f"{L // 1024}K"
                dense_ref = dense_ppls.get(f"{L}_{seed}", r.ppl)
                delta = (r.ppl - dense_ref) / dense_ref * 100
                k_data["evals"][f"{Lk}_s{seed}"] = {
                    "ppl": round(r.ppl, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                }
                print(f"    C2 k={k} L={Lk} s={seed}: delta={delta:+.2f}%")
        c2_results[f"k{k}"] = k_data
    results["C2_normclamp"] = c2_results

    with open(os.path.join(outdir, "path_c_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "version": "v20",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 4 complete. Saved to {outdir}/")


# ============================================================
# NormClamp backend (Path C2)
# ============================================================


class NormClampBackend:
    """Quantize then clamp KV norms to reduce outlier amplification."""

    def __init__(self, layer_bits, block_size=32, clamp_percentile=0.99):
        self.layer_bits = layer_bits
        self.block_size = block_size
        self.clamp_pct = clamp_percentile
        self._name = "normclamp"

    @property
    def name(self):
        return self._name

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n4 = sum(1 for b in self.layer_bits if b == 4)
        n8 = sum(1 for b in self.layer_bits if b == 8)
        print(
            f"    {self.name} calibrated: {n4} INT4, {n8} INT8 (clamp@{self.clamp_pct})"
        )

    def _clamp_norms(self, x):
        """Clamp per-token norms to percentile threshold."""
        # x: [B, H, T, D]
        norms = x.norm(dim=-1, keepdim=True)  # [B, H, T, 1]
        threshold = torch.quantile(norms.flatten(), self.clamp_pct)
        scale = torch.where(
            norms > threshold,
            threshold / norms.clamp(min=1e-8),
            torch.ones_like(norms),
        )
        return x * scale

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
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
                        # Clamp before quantization to reduce outlier damage
                        k_clamped = self._clamp_norms(k_far)
                        v_clamped = self._clamp_norms(v_far)
                        k_q, k_s = quantize_int4_block(k_clamped, self.block_size)
                        k_hat = dequantize_int4_block(k_q, k_s, self.block_size).to(
                            dtype
                        )
                        v_q, v_s = quantize_int4_block(v_clamped, self.block_size)
                        v_hat = dequantize_int4_block(v_q, v_s, self.block_size).to(
                            dtype
                        )

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
                    step=step,
                    cache_len=n_full + n_compressed + step + 1,
                    n_compressed=n_compressed,
                    n_full=n_full + step + 1,
                    compress_ms=compress_ms if step == 0 else 0,
                    decode_ms=0,
                    ppl_step=0,
                    kv_bytes_ratio=0,
                    peak_gpu_mb=0,
                )
            )

        logits = torch.cat(all_logits, dim=1)
        return logits, step_stats


# ============================================================
# Phase 5: Unified comparison
# ============================================================


def run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 5: Unified comparison + summary."""
    outdir = os.path.join(args.outdir, "phase5")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 5: Unified comparison")
    print("=" * 60)

    # Collect results from all phases
    base_dir = args.outdir
    summary = {"methods": {}}

    # Load Phase 0 k-curve
    kcurve_path = os.path.join(base_dir, "artifacts", "v20", "k_sweeps", "k_curve.json")
    if os.path.exists(kcurve_path):
        with open(kcurve_path) as f:
            k_curve = json.load(f)
        summary["baseline_k_star_3pct"] = k_curve.get("k_star_3pct", "N/A")
        summary["baseline_k_star_1pct"] = k_curve.get("k_star_1pct", "N/A")

    # Load Phase 1 theory
    theory_path = os.path.join(base_dir, "artifacts", "v20", "theory_fit.json")
    if os.path.exists(theory_path):
        with open(theory_path) as f:
            theory = json.load(f)
        summary["theory"] = {
            "predicted_k_3pct": theory["predicted_k_star"]["3pct"],
            "predicted_k_1pct": theory["predicted_k_star"]["1pct"],
            "alpha": theory["linear_fit"]["alpha"],
            "beta": theory["linear_fit"]["beta"],
        }

    # Helper: find k* from intervention results
    def find_k_star(intervention_data, tol=3.0):
        """Find minimum k where all evals pass tolerance."""
        for k_key in sorted(intervention_data.keys(), key=lambda x: int(x[1:])):
            kd = intervention_data[k_key]
            all_pass = all(abs(e["delta_pct"]) <= tol for e in kd["evals"].values())
            if all_pass:
                return kd["k"]
        return ">max_tested"

    # Load Phase 2 Path A
    p2_path = os.path.join(base_dir, "phase2", "path_a_results.json")
    if os.path.exists(p2_path):
        with open(p2_path) as f:
            p2 = json.load(f)
        for method_name, method_data in p2.items():
            k3 = find_k_star(method_data, 3.0)
            k1 = find_k_star(method_data, 1.0)
            summary["methods"][method_name] = {
                "path": "A",
                "k_star_3pct": k3,
                "k_star_1pct": k1,
            }

    # Load Phase 3 Path B
    p3_path = os.path.join(base_dir, "phase3", "path_b_results.json")
    if os.path.exists(p3_path):
        with open(p3_path) as f:
            p3 = json.load(f)
        for method_name, method_data in p3.items():
            # Per-channel has nested group_size structure
            if method_name == "B2_perchannel":
                for gs_key, gs_data in method_data.items():
                    k3 = find_k_star(gs_data, 3.0)
                    k1 = find_k_star(gs_data, 1.0)
                    summary["methods"][f"B2_{gs_key}"] = {
                        "path": "B",
                        "k_star_3pct": k3,
                        "k_star_1pct": k1,
                    }

    # Load Phase 4 Path C
    p4_path = os.path.join(base_dir, "phase4", "path_c_results.json")
    if os.path.exists(p4_path):
        with open(p4_path) as f:
            p4 = json.load(f)
        for method_name, method_data in p4.items():
            k3 = find_k_star(method_data, 3.0)
            k1 = find_k_star(method_data, 1.0)
            summary["methods"][method_name] = {
                "path": "C",
                "k_star_3pct": k3,
                "k_star_1pct": k1,
            }

    with open(os.path.join(outdir, "unified_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n  Unified k* Comparison:")
    print(f"  {'Method':<25s} {'Path':>4s} {'k*@3%':>6s} {'k*@1%':>6s}")
    print(f"  {'-'*25} {'-'*4} {'-'*6} {'-'*6}")
    print(
        f"  {'S2_baseline':<25s} {'--':>4s} {str(summary.get('baseline_k_star_3pct', '?')):>6s} {str(summary.get('baseline_k_star_1pct', '?')):>6s}"
    )
    for mname, mdata in summary["methods"].items():
        print(
            f"  {mname:<25s} {mdata['path']:>4s} {str(mdata['k_star_3pct']):>6s} {str(mdata['k_star_1pct']):>6s}"
        )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 5,
        "version": "v20",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 5 complete. Saved to {outdir}/")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v20: Break the k-floor")
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
    parser.add_argument("--outdir", default="results/v20")
    parser.add_argument(
        "--sensitivity_path",
        default="results/v15/phase4/layer_sensitivity.json",
    )
    parser.add_argument(
        "--oracle_path",
        default="results/v19/artifacts/bitterkv/oracle_empirical.json",
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
    elif args.phase == 5:
        run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
