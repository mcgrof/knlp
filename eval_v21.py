#!/usr/bin/env python
"""
BPA v21 Evaluation: Exploit g=4, beat kv_ratio=0.33.

Combines tight-group INT4 (g=4) with mixed-precision k-floor scheduling
to find configurations that beat S2_k6's kv_ratio~0.333 while maintaining
PASS@3% at all sequence lengths up to 32K.

Key hypothesis: g=4 reduces per-layer INT4 noise by ~8x (v20 finding),
potentially enabling k=2-3 to PASS at all L, yielding kv_ratio < 0.333
with scale overhead included.

Phases:
  0: True KV bytes accounting (include scale metadata overhead)
  1: Reproduce key v20 sanity points
  2: Core (k, g) grid search
  3: Failure attribution (if k<4 fails)
  4: Minimal next knob (K-specific treatment or learned scales)
  5: Final recommendation + deliverables

Usage:
    python eval_v21.py --phase 0
    python eval_v21.py --phase 1
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
# True KV bytes accounting
# ============================================================


def compute_kv_bytes_per_token(
    n_layers,
    n_kv_heads,
    head_dim,
    payload_bits,
    group_size=None,
    scale_dtype_bits=16,
    has_zero_point=False,
    zero_point_bits=0,
    k_int8_layers=0,
):
    """Compute TRUE KV cache bytes per token including scale overhead.

    Args:
        n_layers: number of transformer layers
        n_kv_heads: number of KV heads per layer
        head_dim: dimension per head
        payload_bits: quantized payload bits (4 or 8)
        group_size: quantization group size (None for per-tensor/INT8)
        scale_dtype_bits: bits for scale factors (16 for fp16)
        has_zero_point: whether zero-point is stored
        zero_point_bits: bits for zero-point (if used)
        k_int8_layers: number of layers kept at INT8 (protected)

    Returns:
        dict with bytes breakdown
    """
    n_int4_layers = n_layers - k_int8_layers
    scale_bytes = scale_dtype_bits // 8

    # Dense baseline: bf16/fp16 — pure payload, no metadata
    # Per layer per token: 2(K,V) * n_kv_heads * head_dim * 2(bytes)
    dense_bytes_per_layer = 2 * n_kv_heads * head_dim * 2
    dense_total = n_layers * dense_bytes_per_layer

    # Handle dense passthrough (payload_bits=16)
    if payload_bits == 16:
        return {
            "dense_bytes_per_token": dense_total,
            "int8_bytes_per_layer": 0,
            "int4_bytes_per_layer": 0,
            "n_int8_layers": 0,
            "n_int4_layers": 0,
            "total_bytes_per_token": dense_total,
            "kv_ratio": 1.0,
            "int4_payload_per_layer": 0,
            "int4_scale_per_layer": 0,
            "scale_overhead_pct": 0.0,
        }

    # INT8 symmetric: per-token-per-head scale (shape [..., 1])
    # Payload: 2(K,V) * n_kv_heads * head_dim * 1 byte
    # Scale: 2(K,V) * n_kv_heads * 1 * scale_bytes
    int8_payload = 2 * n_kv_heads * head_dim * 1
    int8_scale = 2 * n_kv_heads * 1 * scale_bytes
    int8_bytes_per_layer = int8_payload + int8_scale

    # Handle all-INT8 case
    if payload_bits == 8:
        total_bytes = n_layers * int8_bytes_per_layer
        return {
            "dense_bytes_per_token": dense_total,
            "int8_bytes_per_layer": int8_bytes_per_layer,
            "int4_bytes_per_layer": 0,
            "n_int8_layers": n_layers,
            "n_int4_layers": 0,
            "total_bytes_per_token": total_bytes,
            "kv_ratio": round(total_bytes / dense_total, 6),
            "int4_payload_per_layer": 0,
            "int4_scale_per_layer": 0,
            "scale_overhead_pct": 0.0,
        }

    # INT4 with group quantization
    # Payload: 2(K,V) * n_kv_heads * head_dim * 0.5 bytes
    # Scale: 2(K,V) * n_kv_heads * ceil(head_dim/g) * scale_bytes
    g = group_size if group_size is not None else head_dim
    n_groups = math.ceil(head_dim / g)
    int4_payload = 2 * n_kv_heads * head_dim * 0.5
    int4_scale = 2 * n_kv_heads * n_groups * scale_bytes
    int4_zp = 0
    if has_zero_point and zero_point_bits > 0:
        int4_zp = 2 * n_kv_heads * n_groups * (zero_point_bits // 8)
    int4_bytes_per_layer = int4_payload + int4_scale + int4_zp

    # Mixed INT4/INT8 total
    total_bytes = (
        k_int8_layers * int8_bytes_per_layer + n_int4_layers * int4_bytes_per_layer
    )
    kv_ratio = total_bytes / dense_total

    return {
        "dense_bytes_per_token": dense_total,
        "int8_bytes_per_layer": int8_bytes_per_layer,
        "int4_bytes_per_layer": int4_bytes_per_layer,
        "n_int8_layers": k_int8_layers,
        "n_int4_layers": n_int4_layers,
        "total_bytes_per_token": total_bytes,
        "kv_ratio": round(kv_ratio, 6),
        "int4_payload_per_layer": int4_payload,
        "int4_scale_per_layer": int4_scale,
        "scale_overhead_pct": round(
            (
                int4_scale / int4_bytes_per_layer * 100
                if int4_bytes_per_layer > 0
                else 0
            ),
            2,
        ),
    }


def run_phase0(args, **kwargs):
    """Phase 0: True KV bytes accounting."""
    art_dir = os.path.join(args.outdir, "artifacts", "v21")
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 0: True KV Bytes Accounting")
    print("=" * 60)

    # Qwen2.5-0.5B parameters
    n_layers = 24
    n_kv_heads = 2
    head_dim = 64

    configs = OrderedDict()

    # Dense baseline
    configs["dense_bf16"] = {
        "payload_bits": 16,
        "group_size": None,
        "k_int8_layers": 0,
        "description": "Dense bf16 baseline",
    }

    # INT8 all layers
    configs["INT8_all"] = {
        "payload_bits": 8,
        "group_size": None,
        "k_int8_layers": 24,
        "description": "All layers INT8 symmetric",
    }

    # INT4 g=32 with various k
    for k in [0, 1, 2, 3, 4, 6]:
        configs[f"INT4_g32_k{k}"] = {
            "payload_bits": 4,
            "group_size": 32,
            "k_int8_layers": k,
            "description": f"INT4 g=32, {k} INT8 layers",
        }

    # INT4 g=8 with various k
    for k in [0, 2, 4]:
        configs[f"INT4_g8_k{k}"] = {
            "payload_bits": 4,
            "group_size": 8,
            "k_int8_layers": k,
            "description": f"INT4 g=8, {k} INT8 layers",
        }

    # INT4 g=4 with various k
    for k in [0, 1, 2, 3, 4, 6]:
        configs[f"INT4_g4_k{k}"] = {
            "payload_bits": 4,
            "group_size": 4,
            "k_int8_layers": k,
            "description": f"INT4 g=4, {k} INT8 layers",
        }

    # Per-channel (g=head_dim=64, effectively per-channel)
    configs["INT4_perchan_k4"] = {
        "payload_bits": 4,
        "group_size": 64,
        "k_int8_layers": 4,
        "description": "INT4 per-channel (g=64), 4 INT8 layers",
    }

    results = OrderedDict()
    for name, cfg in configs.items():
        r = compute_kv_bytes_per_token(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            payload_bits=cfg["payload_bits"],
            group_size=cfg.get("group_size"),
            k_int8_layers=cfg.get("k_int8_layers", 0),
        )
        r["description"] = cfg["description"]
        results[name] = r

    # Save JSON
    accounting = {
        "model": "Qwen2.5-0.5B",
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "configs": results,
    }
    with open(os.path.join(art_dir, "kv_bytes_accounting.json"), "w") as f:
        json.dump(accounting, f, indent=2)

    # Write markdown report
    md = "# True KV Bytes Accounting — BPA v21\n\n"
    md += "## Model: Qwen2.5-0.5B\n"
    md += f"- Layers: {n_layers}, KV heads: {n_kv_heads}, head_dim: {head_dim}\n"
    md += f"- Dense bf16 bytes/token: {results['dense_bf16']['dense_bytes_per_token']}\n\n"

    md += "## Accounting Table\n\n"
    md += "| Config | Bytes/token | kv_ratio | Scale overhead % | Description |\n"
    md += "|--------|-------------|----------|-----------------|-------------|\n"
    for name, r in results.items():
        md += (
            f"| {name} | {r['total_bytes_per_token']:.1f} | "
            f"{r['kv_ratio']:.4f} | "
            f"{r['scale_overhead_pct']:.1f}% | "
            f"{r['description']} |\n"
        )

    md += "\n## Key Findings\n\n"
    # Compare g=4 vs g=32 scale overhead
    g4_k0 = results.get("INT4_g4_k0", {})
    g32_k0 = results.get("INT4_g32_k0", {})
    if g4_k0 and g32_k0:
        md += f"- g=4 scale overhead: {g4_k0['scale_overhead_pct']:.1f}% "
        md += f"vs g=32: {g32_k0['scale_overhead_pct']:.1f}%\n"
        md += f"- g=4 kv_ratio (k=0): {g4_k0['kv_ratio']:.4f} "
        md += f"vs g=32 (k=0): {g32_k0['kv_ratio']:.4f}\n"

    # Find configs that beat 0.333
    md += "\n## Configs with kv_ratio < 0.333\n\n"
    for name, r in results.items():
        if r["kv_ratio"] < 0.333 and name != "dense_bf16":
            md += f"- **{name}**: kv_ratio={r['kv_ratio']:.4f}\n"

    md += "\n## Configs with kv_ratio < 0.333 (PASS candidates)\n\n"
    md += "These configs could potentially beat S2_k6 if they PASS@3%:\n\n"
    for name, r in results.items():
        if 0.20 <= r["kv_ratio"] < 0.333 and name != "dense_bf16":
            md += f"- **{name}**: kv_ratio={r['kv_ratio']:.4f}\n"

    with open(os.path.join(art_dir, "kv_bytes_accounting.md"), "w") as f:
        f.write(md)

    # Print summary
    print(f"\n  {'Config':<22s} {'B/tok':>8s} {'ratio':>8s} {'scale%':>8s}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, r in results.items():
        print(
            f"  {name:<22s} {r['total_bytes_per_token']:>8.1f} "
            f"{r['kv_ratio']:>8.4f} {r['scale_overhead_pct']:>8.1f}%"
        )

    # Highlight key point: which (k,g) combos have ratio < 0.333
    print("\n  Configs beating S2_k6 kv_ratio=0.333:")
    for name, r in results.items():
        if r["kv_ratio"] < 0.333 and name != "dense_bf16":
            print(f"    {name}: ratio={r['kv_ratio']:.4f}")

    print(f"\nPhase 0 complete. Saved to {art_dir}/")


# ============================================================
# Oracle ranking and schedule building
# ============================================================


def load_oracle_ranking(oracle_path):
    """Load oracle per-layer sensitivity from v19 artifacts."""
    with open(oracle_path) as f:
        data = json.load(f)
    ranked = data["int4_ranked_tolerant_to_sensitive"]
    layers_sens_desc = [e["layer"] for e in reversed(ranked)]
    deltas = {e["layer"]: e["ppl_delta_pct"] for e in ranked}
    return layers_sens_desc, deltas


def load_theory_ranking(theory_path):
    """Load theory-based layer upgrade order from v20 artifacts."""
    with open(theory_path) as f:
        data = json.load(f)
    return data["layer_upgrade_order"]


def build_k_schedule(ranking, k, n_layers=24):
    """Build INT4/INT8 schedule: top-k sensitive layers get INT8."""
    schedule = [4] * n_layers
    for i in range(min(k, n_layers)):
        schedule[ranking[i]] = 8
    return schedule


# ============================================================
# Mixed-precision backend with configurable group_size
# ============================================================


class GroupedMixedBackend:
    """Mixed-precision with configurable group_size for INT4 layers.

    Protected layers get INT8 (per-tensor symmetric).
    Unprotected layers get INT4 with specified group_size.
    """

    def __init__(self, layer_bits, group_size=32):
        self.layer_bits = layer_bits
        self.group_size = group_size
        n8 = sum(1 for b in layer_bits if b == 8)
        n4 = sum(1 for b in layer_bits if b == 4)
        self._name = f"mixed_g{group_size}_k{n8}"

    @property
    def name(self):
        return self._name

    def description(self):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        return f"Mixed INT4(g={self.group_size})/INT8, {n8} INT8 layers"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n8 = sum(1 for b in self.layer_bits if b == 8)
        n4 = sum(1 for b in self.layer_bits if b == 4)
        print(f"    {self.name} calibrated: {n4} INT4(g={self.group_size}), {n8} INT8")

    def _quantize_int4_grouped(self, x, group_size):
        """Per-group INT4 quantization along head_dim."""
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

    def _dequantize_int4_grouped(self, x_q, scale, orig_D):
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
                        k_q, k_s, k_D = self._quantize_int4_grouped(
                            k_far, self.group_size
                        )
                        k_hat = self._dequantize_int4_grouped(k_q, k_s, k_D).to(dtype)
                        v_q, v_s, v_D = self._quantize_int4_grouped(
                            v_far, self.group_size
                        )
                        v_hat = self._dequantize_int4_grouped(v_q, v_s, v_D).to(dtype)

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
# Evaluation helpers
# ============================================================


def eval_config(
    backend,
    model,
    token_data,
    L_list,
    seeds,
    device_str,
    max_ctx,
    model_config,
    dense_ppls,
    decode_steps=256,
):
    """Evaluate a backend at all L and seeds, return results dict."""
    results = {}
    for L in L_list:
        for seed in seeds:
            r = run_single_eval(
                backend,
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device_str,
                max_ctx,
                model_config,
            )
            key = f"L{L}_s{seed}"
            dense_ref = dense_ppls.get((L, "r1", seed), r.ppl)
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


def check_pass(evals, tol=3.0):
    """Check if all evals pass at given tolerance."""
    return all(abs(e["delta_pct"]) <= tol for e in evals.values())


def max_delta(evals):
    """Return max absolute delta across all evals."""
    return max(abs(e["delta_pct"]) for e in evals.values())


# ============================================================
# Phase 1: Reproduce v20 sanity points
# ============================================================


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Reproduce key v20 results at L=32K."""
    outdir = os.path.join(args.outdir, "phase1")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 1: Reproduce v20 Sanity Points")
    print("=" * 60)

    # Get dense baselines
    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        [32768],
        args.decode_steps,
        [0],  # single seed for sanity
        args.device,
        max_ctx,
        model_config,
    )

    theory_ranking = load_theory_ranking(args.theory_path)
    results = {}

    # Test 1: INT4 g=32 k=4 at L=32K (should be ~+1.96%)
    print("\n  Test 1: INT4 g=32 k=4 at L=32K")
    sched_k4 = build_k_schedule(theory_ranking, 4)
    be1 = MixedPrecisionBackend(layer_bits=sched_k4)
    be1._name = "sanity_g32_k4"
    be1.configure(32768, model_config)
    be1.calibrate(model, token_data, 32768, args.device, model_config)
    r1 = run_single_eval(
        be1,
        model,
        token_data,
        32768,
        args.decode_steps,
        0,
        args.device,
        max_ctx,
        model_config,
    )
    dense_ref = dense_ppls.get((32768, "r1", 0), r1.ppl)
    d1 = (r1.ppl - dense_ref) / dense_ref * 100
    results["g32_k4_32K"] = {"ppl": round(r1.ppl, 4), "delta_pct": round(d1, 2)}
    print(f"    PPL={r1.ppl:.4f} delta={d1:+.2f}% (v20 expected ~+1.96%)")

    # Test 2: INT4 g=4 k=0 at L=32K (should be ~+4.46%)
    print("\n  Test 2: INT4 g=4 k=0 at L=32K")
    sched_k0 = build_k_schedule(theory_ranking, 0)
    be2 = GroupedMixedBackend(layer_bits=sched_k0, group_size=4)
    be2.configure(32768, model_config)
    be2.calibrate(model, token_data, 32768, args.device, model_config)
    r2 = run_single_eval(
        be2,
        model,
        token_data,
        32768,
        args.decode_steps,
        0,
        args.device,
        max_ctx,
        model_config,
    )
    d2 = (r2.ppl - dense_ref) / dense_ref * 100
    results["g4_k0_32K"] = {"ppl": round(r2.ppl, 4), "delta_pct": round(d2, 2)}
    print(f"    PPL={r2.ppl:.4f} delta={d2:+.2f}% (v20 expected ~+4.46%)")

    # Test 3: INT4 g=4 k=4 at L=32K (should be ~+1.70%)
    print("\n  Test 3: INT4 g=4 k=4 at L=32K")
    be3 = GroupedMixedBackend(layer_bits=sched_k4, group_size=4)
    be3.configure(32768, model_config)
    be3.calibrate(model, token_data, 32768, args.device, model_config)
    r3 = run_single_eval(
        be3,
        model,
        token_data,
        32768,
        args.decode_steps,
        0,
        args.device,
        max_ctx,
        model_config,
    )
    d3 = (r3.ppl - dense_ref) / dense_ref * 100
    results["g4_k4_32K"] = {"ppl": round(r3.ppl, 4), "delta_pct": round(d3, 2)}
    print(f"    PPL={r3.ppl:.4f} delta={d3:+.2f}% (v20 expected ~+1.70%)")

    # Check for mismatch
    v20_expected = {"g32_k4_32K": 1.96, "g4_k0_32K": 4.46, "g4_k4_32K": 1.70}
    any_mismatch = False
    for key, exp in v20_expected.items():
        obs = abs(results[key]["delta_pct"])
        diff = abs(obs - exp)
        if diff > 0.5:
            print(
                f"\n  WARNING: {key} mismatch: observed {obs:.2f}% vs expected {exp:.2f}% (diff={diff:.2f}%)"
            )
            any_mismatch = True

    if any_mismatch:
        print("\n  MISMATCH DETECTED — investigate before proceeding!")
    else:
        print("\n  All sanity checks within 0.5% of v20. Proceeding.")

    with open(os.path.join(outdir, "sanity_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "version": "v21",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 1 complete. Saved to {outdir}/")


# ============================================================
# Phase 2: Core (k, g) grid search
# ============================================================


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Grid search over (k, g) configurations."""
    outdir = os.path.join(args.outdir, "phase2")
    art_dir = os.path.join(args.outdir, "artifacts", "v21")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 2: Core (k, g) Grid Search")
    print("=" * 60)

    theory_ranking = load_theory_ranking(args.theory_path)

    # Phase 2a: Quick screen at L=8192 (all seeds)
    print("\n  Phase 2a: Quick screen at L=8192")
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

    # Load accounting for true kv_ratio
    acct_path = os.path.join(art_dir, "kv_bytes_accounting.json")
    accounting = {}
    if os.path.exists(acct_path):
        with open(acct_path) as f:
            accounting = json.load(f)

    grid_results = {}
    survivors = []

    g_values = [32, 8, 4]
    k_values = [0, 1, 2, 3, 4, 6]

    for g in g_values:
        for k in k_values:
            config_name = f"g{g}_k{k}"
            print(f"\n    Screen: g={g} k={k}")

            schedule = build_k_schedule(theory_ranking, k)
            be = GroupedMixedBackend(layer_bits=schedule, group_size=g)
            be.configure(8192, model_config)
            be.calibrate(model, token_data, 8192, args.device, model_config)

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
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                    "p50_ms": round(r.p50_ms, 2),
                }
                print(f"      s={seed}: delta={delta:+.2f}%")

            # Get true kv_ratio from accounting
            acct_key = f"INT4_g{g}_k{k}"
            true_ratio = None
            if accounting and "configs" in accounting:
                acct_entry = accounting["configs"].get(acct_key, {})
                true_ratio = acct_entry.get("kv_ratio")

            pass_8k = check_pass(evals, 3.0)
            md = max_delta(evals)

            grid_results[config_name] = {
                "g": g,
                "k": k,
                "schedule": schedule,
                "screen_8k": evals,
                "pass_8k_3pct": pass_8k,
                "max_delta_8k": round(md, 2),
                "true_kv_ratio": true_ratio,
            }

            # Survivors: PASS@3% at 8K OR close (within 5%)
            if md <= 5.0:
                survivors.append(config_name)
                print(f"      -> SURVIVOR (max_delta={md:.2f}%)")
            else:
                print(f"      -> ELIMINATED (max_delta={md:.2f}%)")

    # Phase 2b: Validate survivors at L=16K and L=32K
    print(f"\n  Phase 2b: Validating {len(survivors)} survivors at 16K/32K")
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

    for config_name in survivors:
        cfg = grid_results[config_name]
        g = cfg["g"]
        k = cfg["k"]
        schedule = cfg["schedule"]

        print(f"\n    Validating: g={g} k={k}")
        full_evals = dict(cfg["screen_8k"])  # include 8K results

        for L in [16384, 32768]:
            if L not in valid_L:
                continue
            be = GroupedMixedBackend(layer_bits=schedule, group_size=g)
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
                Lk = f"{L // 1024}K"
                dense_ref = dense_ppls_all.get((L, "r1", seed), r.ppl)
                delta = (r.ppl - dense_ref) / dense_ref * 100
                full_evals[f"L{L}_s{seed}"] = {
                    "ppl": round(r.ppl, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                    "p50_ms": round(r.p50_ms, 2),
                }
                print(f"      L={Lk} s={seed}: delta={delta:+.2f}%")

        cfg["full_evals"] = full_evals
        cfg["pass_allL_3pct"] = check_pass(full_evals, 3.0)
        cfg["pass_allL_1pct"] = check_pass(full_evals, 1.0)
        cfg["max_delta_allL"] = round(max_delta(full_evals), 2)

    # Save grid results
    with open(os.path.join(art_dir, "search_grid.json"), "w") as f:
        json.dump(grid_results, f, indent=2)

    # Summary table
    print(f"\n  {'Config':<14s} {'ratio':>7s} {'max_d':>7s} {'P@3%':>6s} {'P@1%':>6s}")
    print(f"  {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 6}")
    for name in sorted(grid_results.keys()):
        cfg = grid_results[name]
        ratio_str = f"{cfg['true_kv_ratio']:.4f}" if cfg["true_kv_ratio"] else "N/A"
        if "full_evals" in cfg:
            md = cfg["max_delta_allL"]
            p3 = "YES" if cfg["pass_allL_3pct"] else "NO"
            p1 = "YES" if cfg["pass_allL_1pct"] else "NO"
        else:
            md = cfg["max_delta_8k"]
            p3 = "8K" if cfg["pass_8k_3pct"] else "NO"
            p1 = "?"
        print(f"  {name:<14s} {ratio_str:>7s} {md:>+7.2f}% {p3:>6s} {p1:>6s}")

    # Identify best configs
    best_pass = None
    best_ratio = 1.0
    for name, cfg in grid_results.items():
        if cfg.get("pass_allL_3pct"):
            ratio = cfg.get("true_kv_ratio", 1.0)
            if ratio and ratio < best_ratio:
                best_ratio = ratio
                best_pass = name

    if best_pass:
        print(f"\n  Best PASS@3% config: {best_pass} (kv_ratio={best_ratio:.4f})")
        if best_ratio < 0.333:
            print("  BEATS S2_k6 (0.333)!")
        else:
            print("  Does NOT beat S2_k6 (0.333)")

    with open(os.path.join(outdir, "grid_summary.json"), "w") as f:
        json.dump(
            {
                "survivors": survivors,
                "best_pass_config": best_pass,
                "best_kv_ratio": best_ratio,
                "beats_s2": best_ratio < 0.333 if best_pass else False,
            },
            f,
            indent=2,
        )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "version": "v21",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 2 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Phase 3: Failure attribution
# ============================================================


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Attribute failures for k<4 configs."""
    outdir = os.path.join(args.outdir, "phase3")
    art_dir = os.path.join(args.outdir, "artifacts", "v21")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 3: Failure Attribution")
    print("=" * 60)

    theory_ranking = load_theory_ranking(args.theory_path)

    # Load grid results from Phase 2
    grid_path = os.path.join(art_dir, "search_grid.json")
    with open(grid_path) as f:
        grid_results = json.load(f)

    # Identify ALL failing configs that were validated (have full_evals)
    # Prioritize g=32 k<4 (these are the configs that could beat S2_k6)
    failing_configs = []
    for name, cfg in grid_results.items():
        if "full_evals" in cfg and not cfg.get("pass_allL_3pct", False):
            failing_configs.append(name)
    # Sort: g=32 first (most interesting), then by k ascending
    failing_configs.sort(key=lambda n: (grid_results[n]["g"], grid_results[n]["k"]))

    if not failing_configs:
        print("  No failing g=4 k<4 configs found. Phase 3 skipped.")
        with open(os.path.join(art_dir, "failure_attribution.json"), "w") as f:
            json.dump(
                {"status": "no_failures", "message": "All g=4 k<4 configs PASS"},
                f,
                indent=2,
            )
        return

    print(f"  Failing configs to investigate: {failing_configs}")

    # Get dense baselines for all L
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

    attribution = {}

    for config_name in failing_configs:
        cfg = grid_results[config_name]
        g = cfg["g"]
        k = cfg["k"]
        schedule = cfg["schedule"]

        print(f"\n  Attributing failure: {config_name} (g={g}, k={k})")

        # Find which (L, seed) cases fail
        failed_cases = []
        if "full_evals" in cfg:
            for key, ev in cfg["full_evals"].items():
                if abs(ev["delta_pct"]) > 3.0:
                    failed_cases.append({"key": key, "delta_pct": ev["delta_pct"]})

        attribution[config_name] = {
            "g": g,
            "k": k,
            "failed_cases": failed_cases,
        }

        # Per-layer distortion analysis at the worst failing L
        # Use the worst delta case
        if failed_cases:
            worst = max(failed_cases, key=lambda x: abs(x["delta_pct"]))
            print(f"    Worst case: {worst['key']} delta={worst['delta_pct']:+.2f}%")

        # Measure per-layer quantization residual with this config
        print("    Measuring per-layer residual distortion...")
        n_layers = model_config["n_layers"]
        rng = np.random.RandomState(0)
        cal_len = 4096
        idx = get_text_batch(token_data, 1, cal_len, rng).to(args.device)

        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model(idx, use_cache=True)
            past_clean = out.past_key_values
        del out

        layer_distortion = []
        for li in range(n_layers):
            kc, vc = past_clean[li]
            bits = schedule[li]

            if bits == 8:
                k_q, k_s = quantize_int8_symmetric(kc)
                k_hat = dequantize_int8_symmetric(k_q, k_s).to(kc.dtype)
                v_q, v_s = quantize_int8_symmetric(vc)
                v_hat = dequantize_int8_symmetric(v_q, v_s).to(vc.dtype)
            else:
                # Use grouped INT4
                B, H, T, D = kc.shape
                n_groups = (D + g - 1) // g
                pad = n_groups * g - D

                def quant_grouped(x):
                    if pad > 0:
                        x = F.pad(x, (0, pad))
                    x_g = x.reshape(B, H, T, n_groups, g)
                    amax = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc = amax / 7.0
                    xq = (x_g / sc).round().clamp(-8, 7)
                    xh = (xq * sc).reshape(B, H, T, n_groups * g)
                    return xh[:, :, :, :D]

                k_hat = quant_grouped(kc.float()).to(kc.dtype)
                v_hat = quant_grouped(vc.float()).to(vc.dtype)

            k_err = ((kc.float() - k_hat.float()) ** 2).mean().item()
            v_err = ((vc.float() - v_hat.float()) ** 2).mean().item()
            total_err = math.sqrt(k_err + v_err)

            layer_distortion.append(
                {
                    "layer": li,
                    "bits": bits,
                    "k_mse": round(k_err, 8),
                    "v_mse": round(v_err, 8),
                    "total_rmse": round(total_err, 6),
                    "protected": bits == 8,
                }
            )

        # Sort by distortion
        unprotected = [ld for ld in layer_distortion if not ld["protected"]]
        unprotected_sorted = sorted(
            unprotected, key=lambda x: x["total_rmse"], reverse=True
        )

        attribution[config_name]["layer_distortion"] = layer_distortion
        attribution[config_name]["top_unprotected_by_distortion"] = [
            ld["layer"] for ld in unprotected_sorted[:6]
        ]
        attribution[config_name]["total_unprotected_noise"] = round(
            sum(ld["total_rmse"] for ld in unprotected), 6
        )

        print(
            f"    Total unprotected noise: {attribution[config_name]['total_unprotected_noise']:.4f}"
        )
        print(
            f"    Top distortion layers: {attribution[config_name]['top_unprotected_by_distortion']}"
        )

    with open(os.path.join(art_dir, "failure_attribution.json"), "w") as f:
        json.dump(attribution, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "version": "v21",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 3 complete. Saved to {art_dir}/")


# ============================================================
# Phase 4: Minimal next knob
# ============================================================


def run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 4: Try minimal interventions for nearly-passing configs."""
    outdir = os.path.join(args.outdir, "phase4")
    art_dir = os.path.join(args.outdir, "artifacts", "v21")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 4: Minimal Next Knob")
    print("=" * 60)

    theory_ranking = load_theory_ranking(args.theory_path)

    # Load grid results
    grid_path = os.path.join(art_dir, "search_grid.json")
    with open(grid_path) as f:
        grid_results = json.load(f)

    # Get dense baselines
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

    # Identify "close" configs: any g, k<6, max_delta between 3.0-5.5%
    # Focus on g=32 since those have viable kv_ratio
    close_configs = []
    for name, cfg in grid_results.items():
        if "full_evals" in cfg and not cfg.get("pass_allL_3pct", False):
            md = cfg.get("max_delta_allL", cfg.get("max_delta_8k", 100))
            if 3.0 < md <= 5.5:
                close_configs.append((name, cfg, md))
    # Sort by kv_ratio ascending (prefer configs with better compression)
    close_configs.sort(key=lambda x: x[1].get("true_kv_ratio", 1.0) or 1.0)

    if not close_configs:
        print("  No g=4 k<4 configs to improve. Phase 4 skipped.")
        with open(os.path.join(outdir, "phase4_results.json"), "w") as f:
            json.dump({"status": "skipped", "reason": "no close configs"}, f, indent=2)
        return

    print(f"  Close configs to try improving: {[c[0] for c in close_configs]}")

    # Option 1: K-specific INT8 treatment
    # For the most sensitive unprotected layers, keep K at INT8 but allow V
    # to remain at INT4 g=4. This targets the RoPE damage that v20 proved
    # dominates the noise.
    print("\n  Option 1: K-only INT8 for top sensitive layers")

    for config_name, cfg, orig_delta in close_configs:
        k_base = cfg["k"]
        print(f"\n    Starting from {config_name} (max_delta={orig_delta:.2f}%)")

        # Try adding 1 or 2 more INT8 layers (full K+V)
        for extra in [1, 2]:
            k_new = k_base + extra
            base_g = cfg["g"]
            new_name = f"g{base_g}_k{k_new}_from_{config_name}"
            schedule = build_k_schedule(theory_ranking, k_new)
            be = GroupedMixedBackend(layer_bits=schedule, group_size=base_g)

            print(f"    Testing k={k_new} (adding {extra} more INT8 layers)")
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
            p1 = check_pass(evals, 1.0)
            md = max_delta(evals)

            # Get true kv_ratio using the base config's group_size
            acct_path = os.path.join(art_dir, "kv_bytes_accounting.json")
            true_ratio = None
            if os.path.exists(acct_path):
                with open(acct_path) as f:
                    acct = json.load(f)
                acct_key = f"INT4_g{base_g}_k{k_new}"
                if "configs" in acct and acct_key in acct["configs"]:
                    true_ratio = acct["configs"][acct_key]["kv_ratio"]
                else:
                    # Compute on the fly with correct group_size
                    r = compute_kv_bytes_per_token(
                        24, 2, 64, 4, base_g, k_int8_layers=k_new
                    )
                    true_ratio = r["kv_ratio"]

            results[new_name] = {
                "k": k_new,
                "g": base_g,
                "base_config": config_name,
                "evals": evals,
                "pass_3pct": p3,
                "pass_1pct": p1,
                "max_delta": round(md, 2),
                "true_kv_ratio": true_ratio,
            }

            ratio_str = f"{true_ratio:.4f}" if true_ratio else "N/A"
            print(f"      max_delta={md:+.2f}% PASS@3%={p3} ratio={ratio_str}")

            for key, ev in sorted(evals.items()):
                print(f"        {key}: delta={ev['delta_pct']:+.2f}%")

    with open(os.path.join(outdir, "phase4_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "version": "v21",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 4 complete. Saved to {outdir}/")


# ============================================================
# Phase 5: Final recommendation + deliverables
# ============================================================


def run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 5: Final report, scoreboard, branch tree."""
    outdir = os.path.join(args.outdir, "phase5")
    art_dir = os.path.join(args.outdir, "artifacts", "v21")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 5: Final Recommendation")
    print("=" * 60)

    # Load all results
    grid_path = os.path.join(art_dir, "search_grid.json")
    acct_path = os.path.join(art_dir, "kv_bytes_accounting.json")
    attr_path = os.path.join(art_dir, "failure_attribution.json")
    p4_path = os.path.join(args.outdir, "phase4", "phase4_results.json")
    p1_path = os.path.join(args.outdir, "phase1", "sanity_results.json")

    grid_results = {}
    accounting = {}
    attribution = {}
    p4_results = {}
    p1_results = {}

    if os.path.exists(grid_path):
        with open(grid_path) as f:
            grid_results = json.load(f)
    if os.path.exists(acct_path):
        with open(acct_path) as f:
            accounting = json.load(f)
    if os.path.exists(attr_path):
        with open(attr_path) as f:
            attribution = json.load(f)
    if os.path.exists(p4_path):
        with open(p4_path) as f:
            p4_results = json.load(f)
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1_results = json.load(f)

    # Build scoreboard
    scoreboard = {
        "model": "Qwen2.5-0.5B",
        "experiment": "BPA v21: Exploit g=4, beat kv_ratio=0.33",
        "baselines": {
            "dense": {"kv_ratio": 1.0, "pass_3pct": "all"},
            "INT8_all": {"kv_ratio": 0.5208, "pass_3pct": "all"},
            "S2_manual_k6": {"kv_ratio": 0.333, "pass_3pct": "all"},
        },
        "grid_results": {},
        "phase4_results": {},
    }

    # Summarize grid
    for name, cfg in sorted(grid_results.items()):
        entry = {
            "g": cfg["g"],
            "k": cfg["k"],
            "true_kv_ratio": cfg.get("true_kv_ratio"),
            "max_delta_8k": cfg.get("max_delta_8k"),
        }
        if "full_evals" in cfg:
            entry["pass_allL_3pct"] = cfg.get("pass_allL_3pct")
            entry["pass_allL_1pct"] = cfg.get("pass_allL_1pct")
            entry["max_delta_allL"] = cfg.get("max_delta_allL")
        scoreboard["grid_results"][name] = entry

    # Summarize Phase 4
    for name, res in p4_results.items():
        if isinstance(res, dict) and "pass_3pct" in res:
            scoreboard["phase4_results"][name] = {
                "k": res["k"],
                "g": res["g"],
                "pass_3pct": res["pass_3pct"],
                "max_delta": res["max_delta"],
                "true_kv_ratio": res.get("true_kv_ratio"),
            }

    # Find overall winner
    all_passing = []
    for name, cfg in grid_results.items():
        if cfg.get("pass_allL_3pct"):
            ratio = cfg.get("true_kv_ratio", 1.0)
            if ratio:
                all_passing.append((name, ratio, cfg.get("max_delta_allL", 0)))
    for name, res in p4_results.items():
        if isinstance(res, dict) and res.get("pass_3pct"):
            ratio = res.get("true_kv_ratio", 1.0)
            if ratio:
                all_passing.append((name, ratio, res.get("max_delta", 0)))

    all_passing.sort(key=lambda x: x[1])
    if all_passing:
        winner = all_passing[0]
        scoreboard["winner"] = {
            "config": winner[0],
            "kv_ratio": winner[1],
            "max_delta": winner[2],
            "beats_s2": winner[1] < 0.333,
        }

    with open("bpa_v21_scoreboard.json", "w") as f:
        json.dump(scoreboard, f, indent=2)

    # Build final report
    report = "# BPA v21 Final Report: Exploit g=4, Beat kv_ratio=0.33\n\n"
    report += "## Summary\n\n"
    report += "v21 investigated whether tight-group INT4 (g=4) combined with\n"
    report += "fewer INT8 layers (k<4) can beat S2_k6's kv_ratio=0.333 while\n"
    report += "maintaining PASS@3% at all sequence lengths up to 32K.\n\n"

    # Phase 0 summary
    report += "## 1. True KV Bytes Accounting (Phase 0)\n\n"
    if accounting and "configs" in accounting:
        report += "| Config | Bytes/tok | kv_ratio | Scale % |\n"
        report += "|--------|-----------|----------|---------|\n"
        for name, r in accounting.get("configs", {}).items():
            report += (
                f"| {name} | {r['total_bytes_per_token']:.1f} | "
                f"{r['kv_ratio']:.4f} | {r['scale_overhead_pct']:.1f}% |\n"
            )
        report += "\n"

    # Phase 1 summary
    report += "## 2. Sanity Reproduction (Phase 1)\n\n"
    for key, val in p1_results.items():
        if isinstance(val, dict):
            report += f"- {key}: delta={val.get('delta_pct', '?')}%\n"
    report += "\n"

    # Phase 2 summary
    report += "## 3. Grid Search Results (Phase 2)\n\n"
    report += "| Config | g | k | kv_ratio | max_delta | PASS@3% |\n"
    report += "|--------|---|---|----------|-----------|--------|\n"
    for name in sorted(grid_results.keys()):
        cfg = grid_results[name]
        ratio = cfg.get("true_kv_ratio", "N/A")
        ratio_str = f"{ratio:.4f}" if isinstance(ratio, (int, float)) else str(ratio)
        if "full_evals" in cfg:
            md = cfg.get("max_delta_allL", "?")
            p3 = "YES" if cfg.get("pass_allL_3pct") else "NO"
        else:
            md = cfg.get("max_delta_8k", "?")
            p3 = "8K only" if cfg.get("pass_8k_3pct") else "NO"
        md_str = f"{md:+.2f}%" if isinstance(md, (int, float)) else str(md)
        report += (
            f"| {name} | {cfg['g']} | {cfg['k']} | {ratio_str} | {md_str} | {p3} |\n"
        )
    report += "\n"

    # Phase 3 summary
    report += "## 4. Failure Attribution (Phase 3)\n\n"
    if attribution:
        if attribution.get("status") == "no_failures":
            report += "All g=4 k<4 configs passed. No attribution needed.\n\n"
        else:
            for name, attr in attribution.items():
                if isinstance(attr, dict) and "failed_cases" in attr:
                    report += f"### {name}\n"
                    report += f"- Failed cases: {len(attr['failed_cases'])}\n"
                    report += f"- Total unprotected noise: {attr.get('total_unprotected_noise', '?')}\n"
                    report += f"- Top distortion layers: {attr.get('top_unprotected_by_distortion', [])}\n\n"

    # Phase 4 summary
    report += "## 5. Minimal Next Knob (Phase 4)\n\n"
    for name, res in p4_results.items():
        if isinstance(res, dict) and "pass_3pct" in res:
            ratio = res.get("true_kv_ratio", "N/A")
            ratio_str = (
                f"{ratio:.4f}" if isinstance(ratio, (int, float)) else str(ratio)
            )
            report += f"- **{name}**: PASS@3%={res['pass_3pct']}, "
            report += f"max_delta={res['max_delta']:+.2f}%, kv_ratio={ratio_str}\n"
    report += "\n"

    # Conclusions
    report += "## 6. Conclusions\n\n"
    if all_passing:
        winner = all_passing[0]
        report += f"**Best PASS@3% config**: {winner[0]} (kv_ratio={winner[1]:.4f})\n\n"
        if winner[1] < 0.333:
            report += "This BEATS S2_k6's kv_ratio=0.333.\n\n"
        else:
            report += "This does NOT beat S2_k6's kv_ratio=0.333.\n\n"

    report += "## 7. Recommendation for v22\n\n"
    report += "(To be filled based on results)\n"

    with open("bpa_v21_final_report.md", "w") as f:
        f.write(report)

    # Build branch tree
    tree = "# BPA v21 Branch Tree\n\n"
    tree += "## Observed Outcome\n\n"
    tree += "```\n"
    tree += "v21 Exploit g=4\n"

    if all_passing:
        winner = all_passing[0]
        if winner[1] < 0.333:
            tree += f"├── WINNER: {winner[0]} (ratio={winner[1]:.4f}) BEATS S2\n"
            tree += "└── v22: Port to larger model (1.5B/7B)\n"
        else:
            tree += f"├── Best: {winner[0]} (ratio={winner[1]:.4f}) does NOT beat S2\n"
            tree += "├── v22A: Mixed precision inside layer (per-head)\n"
            tree += "└── v22B: INT6 tier or learned scale calibration\n"
    else:
        tree += "├── No PASS@3% config found with k<4\n"
        tree += "└── v22: Training-in-the-loop quantization\n"

    tree += "```\n"

    with open("bpa_v21_branch_tree.md", "w") as f:
        f.write(tree)

    print("\n  Deliverables written:")
    print("    - bpa_v21_scoreboard.json")
    print("    - bpa_v21_final_report.md")
    print("    - bpa_v21_branch_tree.md")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 5,
        "version": "v21",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 5 complete.")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v21: Exploit g=4")
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
    parser.add_argument("--outdir", default="results/v21")
    parser.add_argument(
        "--sensitivity_path",
        default="results/v15/phase4/layer_sensitivity.json",
    )
    parser.add_argument(
        "--oracle_path",
        default="results/v19/artifacts/bitterkv/oracle_empirical.json",
    )
    parser.add_argument(
        "--theory_path",
        default="results/v20/artifacts/v20/theory_fit.json",
    )
    args = parser.parse_args()

    if args.phase == 0:
        # Phase 0 is purely analytical, no GPU needed
        run_phase0(args)
        return

    gpu_info = gpu_preflight(args.device)
    model, tokenizer, max_ctx, model_config = load_model(args.model, args.device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)
    valid_L = [L for L in args.L if L <= max_ctx]

    if args.phase == 1:
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
