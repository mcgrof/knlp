#!/usr/bin/env python
"""
BPA v16 Evaluation: Breaking the 0.5 Wall.

Phases:
  0: Reconfirm v15 baseline (dense, INT8, rope_complex rank=32)
  1: Rope_complex rank sweep {32,24,16,12,8}
  2: Mixed INT8/INT4 schedules (S1,S2,S3)
  3: Hybrid structural + mixed precision
  4: Real KV storage profiling

Usage:
    python eval_v16.py --phase 0
    python eval_v16.py --phase 1
    python eval_v16.py --phase 2
    python eval_v16.py --phase 3
    python eval_v16.py --phase 4
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch

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
from eval_v15 import (
    V15Result,
    apply_quality_gating,
    build_scoreboard,
    gpu_preflight,
    load_model,
    run_single_eval,
)


# ============================================================
# Mixed precision backend
# ============================================================


class MixedPrecisionBackend:
    """Per-layer INT8/INT4 schedule based on sensitivity."""

    def __init__(self, layer_bits, block_size=32):
        """
        Args:
            layer_bits: list of ints, one per layer (8 or 4)
            block_size: block size for INT4 quantization
        """
        self.layer_bits = layer_bits
        self.block_size = block_size
        self._name = None

    @property
    def name(self):
        if self._name:
            return self._name
        n4 = sum(1 for b in self.layer_bits if b == 4)
        n8 = sum(1 for b in self.layer_bits if b == 8)
        return f"mixed_{n4}xINT4_{n8}xINT8"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        n4 = sum(1 for b in self.layer_bits if b == 4)
        n8 = sum(1 for b in self.layer_bits if b == 8)
        print(f"    {self.name} calibrated: {n4} INT4, {n8} INT8")

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from backends.quant import (
            dequantize_int4_block,
            dequantize_int8_symmetric,
            quantize_int4_block,
            quantize_int8_symmetric,
        )
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        cache_len = past[0][0].shape[2]

        # Quantize far tokens per-layer
        t0 = time.perf_counter()
        if cache_len > self.W_min + self.W_sink:
            far_end = cache_len - self.W_min
            n_far = far_end - self.W_sink

            if n_far > 0:
                new_cache = DynamicCache()
                for li in range(n_layers):
                    k, v = past[li]
                    bits = self.layer_bits[li]

                    k_sink = k[:, :, : self.W_sink, :]
                    v_sink = v[:, :, : self.W_sink, :]
                    k_far = k[:, :, self.W_sink : far_end, :]
                    v_far = v[:, :, self.W_sink : far_end, :]
                    k_near = k[:, :, far_end:, :]
                    v_near = v[:, :, far_end:, :]

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

                    k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
                    v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
                    new_cache.update(k_new, v_new, li)

                past = new_cache
                has_compressed = True
                n_full = self.W_sink + self.W_min
                n_compressed = n_far
            else:
                has_compressed = False
                n_full = cache_len
                n_compressed = 0
        else:
            has_compressed = False
            n_full = cache_len
            n_compressed = 0

        compress_ms = (time.perf_counter() - t0) * 1000

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            pos_ids = None
            if has_compressed:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )

            out = model(
                next_token,
                past_key_values=past,
                use_cache=True,
                position_ids=pos_ids,
            )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            cache_len = past[0][0].shape[2]
            bpt = 2 * n_kv_heads * head_dim * elem

            avg_bits = np.mean(self.layer_bits)
            bytes_full = n_full * bpt * n_layers
            bytes_per_comp = 2 * n_kv_heads * head_dim * (avg_bits / 8)
            bytes_compressed = int(n_compressed * bytes_per_comp * n_layers)

            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=bytes_full,
                    kv_bytes_compressed=bytes_compressed,
                    kv_bytes_total=bytes_full + bytes_compressed,
                    compress_ms=compress_ms if step == 0 else 0,
                    n_full=n_full + step + 1,
                    n_compressed=n_compressed,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def compression_ratio(self):
        avg_bits = np.mean(self.layer_bits)
        return avg_bits / 16.0

    def description(self):
        n4 = sum(1 for b in self.layer_bits if b == 4)
        n8 = sum(1 for b in self.layer_bits if b == 8)
        return f"mixed ({n4}xINT4 + {n8}xINT8)"


# ============================================================
# Hybrid rope_complex + mixed precision
# ============================================================


class HybridRopeMixedBackend:
    """rope_complex for K + mixed INT8/INT4 for V per-layer."""

    def __init__(self, rank_frac, layer_bits, block_size=32):
        self.rank_frac = rank_frac
        self.layer_bits = layer_bits
        self.block_size = block_size
        self.rope_backend = None
        self.calibrated = False
        self._name = None

    @property
    def name(self):
        if self._name:
            return self._name
        n4 = sum(1 for b in self.layer_bits if b == 4)
        rank = int(64 * self.rank_frac)
        return f"hybrid_rope{rank}_mixed{n4}xINT4"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        from backends.rope_aware_kv import RoPEAwareKVBackend

        self.rope_backend = RoPEAwareKVBackend(mode="complex", rank_frac=self.rank_frac)
        self.rope_backend.configure(
            L, model_config, W_min=self.W_min, W_sink=self.W_sink
        )
        self.rope_backend.calibrate(model, token_data, L, device_str, model_config)
        self.calibrated = self.rope_backend.calibrated
        n4 = sum(1 for b in self.layer_bits if b == 4)
        rank = int(model_config["head_dim"] * self.rank_frac)
        print(f"    hybrid calibrated: rank={rank}, {n4} INT4 layers")

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from backends.quant import (
            dequantize_int4_block,
            dequantize_int8_symmetric,
            quantize_int4_block,
            quantize_int8_symmetric,
        )
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        dim_pairs = head_dim // 2
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

        if (
            self.calibrated
            and self.rope_backend.projections
            and cache_len > self.W_min + self.W_sink
        ):
            far_end = cache_len - self.W_min
            n_far = far_end - self.W_sink

            if n_far > 0:
                new_cache = DynamicCache()
                for li in range(n_layers):
                    k, v = past[li]
                    bits = self.layer_bits[li]

                    k_sink = k[:, :, : self.W_sink, :]
                    v_sink = v[:, :, : self.W_sink, :]
                    k_far = k[:, :, self.W_sink : far_end, :]
                    v_far = v[:, :, self.W_sink : far_end, :]
                    k_near = k[:, :, far_end:, :]
                    v_near = v[:, :, far_end:, :]

                    # K: rope_complex compression
                    k_hat = self._compress_k_complex(k_far, li, dtype)

                    # V: per-layer quantization
                    if bits == 8:
                        v_q, v_s = quantize_int8_symmetric(v_far)
                        v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
                    else:
                        v_q, v_s = quantize_int4_block(v_far, self.block_size)
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

            out = model(
                next_token,
                past_key_values=past,
                use_cache=True,
                position_ids=pos_ids,
            )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            cache_len = past[0][0].shape[2]
            bpt = 2 * n_kv_heads * head_dim * elem
            avg_rank = self.rope_backend._avg_rank()
            avg_v_bits = np.mean(self.layer_bits)

            bytes_full = n_full * bpt * n_layers
            bytes_k = int(n_compressed * n_kv_heads * avg_rank * elem * n_layers)
            bytes_v = int(
                n_compressed * n_kv_heads * head_dim * (avg_v_bits / 8) * n_layers
            )
            bytes_compressed = bytes_k + bytes_v

            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=bytes_full,
                    kv_bytes_compressed=bytes_compressed,
                    kv_bytes_total=bytes_full + bytes_compressed,
                    compress_ms=compress_ms if step == 0 else 0,
                    n_full=n_full + step + 1,
                    n_compressed=n_compressed,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def _compress_k_complex(self, k_far, li, dtype):
        """Complex-plane K compression."""
        n_kv_heads = k_far.shape[1]
        head_dim = k_far.shape[3]
        dim_pairs = head_dim // 2

        k_parts = []
        for hi in range(n_kv_heads):
            k_h = k_far[0, hi, :, :].float()
            proj = self.rope_backend.projections[li][hi]
            proj_mag = proj["proj_mag"]

            k_real = k_h[:, :dim_pairs]
            k_imag = k_h[:, dim_pairs:]
            k_complex = torch.complex(k_real, k_imag)

            k_mag = k_complex.abs()
            k_phase = k_complex.angle()

            k_mag_proj = (k_mag @ proj_mag.float().T) @ proj_mag.float()
            k_recon = torch.polar(k_mag_proj, k_phase)

            k_out = torch.cat([k_recon.real, k_recon.imag], dim=-1)
            k_parts.append(k_out.unsqueeze(0).unsqueeze(0))

        return torch.cat(k_parts, dim=1).to(dtype)

    def compression_ratio(self):
        if not self.calibrated:
            return 1.0
        avg_rank = self.rope_backend._avg_rank()
        head_dim = self.mc["head_dim"]
        avg_v_bits = np.mean(self.layer_bits)
        k_ratio = avg_rank / head_dim
        v_ratio = avg_v_bits / 16.0
        return (k_ratio + v_ratio) / 2.0

    def description(self):
        n4 = sum(1 for b in self.layer_bits if b == 4)
        rank = (
            int(64 * self.rank_frac)
            if self.mc is None
            else int(self.mc.get("head_dim", 64) * self.rank_frac)
        )
        return f"hybrid rope_complex(rank={rank}) + mixed({n4}xINT4)"


# ============================================================
# Layer schedule builders
# ============================================================


def build_schedules(sensitivity_path):
    """Build S1/S2/S3 schedules from v15 layer sensitivity."""
    with open(sensitivity_path) as f:
        data = json.load(f)

    ranked = data["int4_ranked_tolerant_to_sensitive"]
    n = len(ranked)

    # Extract layer order (tolerant -> sensitive)
    layers_by_tolerance = [e["layer"] for e in ranked]

    # S1 Conservative: bottom 25% INT4, rest INT8
    n_int4_s1 = n // 4  # 6 layers
    s1 = [8] * n
    for i in range(n_int4_s1):
        s1[layers_by_tolerance[i]] = 4

    # S2 Balanced: top 25% INT8, rest INT4
    n_int8_s2 = n // 4  # 6 layers
    s2 = [4] * n
    for i in range(n_int8_s2):
        s2[layers_by_tolerance[n - 1 - i]] = 8

    # S3 Aggressive: top 33% INT8, rest INT4
    n_int8_s3 = n // 3  # 8 layers
    s3 = [4] * n
    for i in range(n_int8_s3):
        s3[layers_by_tolerance[n - 1 - i]] = 8

    schedules = {
        "S1": s1,
        "S2": s2,
        "S3": s3,
    }

    for name, sched in schedules.items():
        n4 = sum(1 for b in sched if b == 4)
        n8 = sum(1 for b in sched if b == 8)
        print(f"  Schedule {name}: {n4} INT4, {n8} INT8")

    return schedules


# ============================================================
# Phase runners
# ============================================================


def run_backend_sweep(
    backends,
    model,
    token_data,
    valid_L,
    decode_steps,
    seeds,
    device,
    max_ctx,
    model_config,
    dense_ppls,
):
    """Run a set of backends through the eval harness."""
    all_results = []
    for backend_name, backend in backends:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend_name}")
        print("=" * 60)

        try:
            backend.configure(max(valid_L), model_config)
            backend.calibrate(model, token_data, max(valid_L), device, model_config)
        except Exception as e:
            import traceback

            print(f"  Calibration failed: {e}")
            traceback.print_exc()
            continue

        for L in valid_L:
            for seed in seeds:
                print(
                    f"  {backend_name} L={L} seed={seed}...",
                    end="",
                    flush=True,
                )
                backend.configure(L, model_config)
                try:
                    r = run_single_eval(
                        backend,
                        model,
                        token_data,
                        L,
                        decode_steps,
                        seed,
                        device,
                        max_ctx,
                        model_config,
                    )
                except Exception as e:
                    r = V15Result(
                        backend=backend_name,
                        L=L,
                        regime="r1",
                        batch_size=1,
                        seed=seed,
                        decode_steps=decode_steps,
                        ppl=float("inf"),
                        error=str(e),
                        catastrophic=True,
                    )

                # Override backend name with the tuple key so each
                # distinct config gets a unique name in results.
                r.backend = backend_name
                all_results.append(r)
                if r.error:
                    print(f" ERROR: {r.error[:60]}")
                else:
                    print(
                        f" PPL={r.ppl:.1f}"
                        f" delta={r.ppl_delta_pct:+.1f}%"
                        f" p50={r.p50_ms:.2f}ms"
                    )

    return all_results


def run_dense_baselines(
    model, token_data, valid_L, decode_steps, seeds, device, max_ctx, model_config
):
    """Run dense baselines, return results and dense_ppls dict."""
    print(f"\n{'=' * 60}")
    print("Dense baselines")
    print("=" * 60)

    dense_be = DenseBackend()
    results = []
    dense_ppls = {}
    for L in valid_L:
        for seed in seeds:
            print(f"  dense L={L} seed={seed}...", end="", flush=True)
            dense_be.configure(L, model_config)
            r = run_single_eval(
                dense_be,
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device,
                max_ctx,
                model_config,
            )
            r.ppl_dense = r.ppl
            r.passed_1pct = True
            r.passed_3pct = True
            results.append(r)
            dense_ppls[(L, "r1", seed)] = r.ppl
            print(f" PPL={r.ppl:.1f} p50={r.p50_ms:.2f}ms")

    return results, dense_ppls


def save_results(all_results, dense_ppls, outdir, phase, meta_extra=None):
    """Apply gating, build scoreboard, save."""
    apply_quality_gating(all_results, dense_ppls)

    # Print summary
    backend_names = []
    seen = set()
    for r in all_results:
        if r.backend not in seen:
            backend_names.append(r.backend)
            seen.add(r.backend)

    print(f"\n{'=' * 60}")
    print("Quality gating")
    print("=" * 60)
    for bname in backend_names:
        br = [r for r in all_results if r.backend == bname]
        n = len(br)
        n1 = sum(1 for r in br if r.passed_1pct)
        n3 = sum(1 for r in br if r.passed_3pct)
        nc = sum(1 for r in br if r.catastrophic)
        ratios = [r.kv_bytes_ratio for r in br if 0 < r.kv_bytes_ratio < 2]
        avg_r = np.mean(ratios) if ratios else 1.0
        print(
            f"  {bname:35s}: {n1}/{n} @1%  {n3}/{n} @3%"
            f"  {nc} cat  ratio={avg_r:.3f}"
        )

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    scoreboard = build_scoreboard(all_results)
    with open(os.path.join(outdir, "scoreboard.json"), "w") as f:
        json.dump(scoreboard, f, indent=2, default=str)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "version": "v16",
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(all_results)} results to {outdir}/")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v16")
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
    parser.add_argument("--outdir", default="results/v16")
    parser.add_argument(
        "--sensitivity_path",
        default="results/v15/phase4/layer_sensitivity.json",
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


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Reconfirm v15 baseline."""
    from backends.quant import QuantBackend
    from backends.rope_aware_kv import RoPEAwareKVBackend

    outdir = os.path.join(args.outdir, "phase0")
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
    rope_be = RoPEAwareKVBackend(mode="complex", rank_frac=0.5)

    backends = [
        ("quant", int8_be),
        ("rope_complex", rope_be),
    ]
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


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Rope_complex rank sweep."""
    from backends.rope_aware_kv import RoPEAwareKVBackend

    outdir = os.path.join(args.outdir, "phase1")
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

    head_dim = model_config["head_dim"]
    ranks = [32, 24, 16, 12, 8]

    backends = []
    for rank in ranks:
        frac = rank / head_dim
        be = RoPEAwareKVBackend(mode="complex", rank_frac=frac)
        backends.append((f"rope_complex_r{rank}", be))

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
    save_results(
        all_results,
        dense_ppls,
        outdir,
        1,
        {"gpu_info": gpu_info, "ranks_tested": ranks},
    )


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Mixed INT8/INT4 schedules."""
    outdir = os.path.join(args.outdir, "phase2")
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

    schedules = build_schedules(args.sensitivity_path)

    backends = []
    for sname, sched in schedules.items():
        be = MixedPrecisionBackend(layer_bits=sched)
        be._name = sname
        backends.append((sname, be))

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
    save_results(
        all_results,
        dense_ppls,
        outdir,
        2,
        {"gpu_info": gpu_info, "schedules": {k: v for k, v in schedules.items()}},
    )


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Hybrid rope_complex + mixed precision."""
    outdir = os.path.join(args.outdir, "phase3")
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

    # Load Phase 1 results to find best rank
    phase1_path = os.path.join(args.outdir, "phase1", "all_results.json")
    if os.path.exists(phase1_path):
        with open(phase1_path) as f:
            p1_results = json.load(f)
        # Find min rank that passes 3% across all seeds and L.
        # Try backend name first (new format), then description
        # (old format where all ranks share backend="rope_complex").
        best_rank = 32
        for rank in [8, 12, 16, 24, 32]:
            bname = f"rope_complex_r{rank}"
            br = [r for r in p1_results if r["backend"] == bname]
            if not br:
                # Fall back: match by description containing the rank
                tag = f"K_rank={rank}/"
                br = [r for r in p1_results if r.get("description", "").find(tag) >= 0]
            if br and all(
                r.get("passed_3pct", False) or r.get("ppl_delta_pct", 99) <= 3.0
                for r in br
            ):
                best_rank = rank
                break
        print(f"Best passing rank from Phase 1: {best_rank}")
    else:
        best_rank = 32
        print("Phase 1 results not found, using rank=32")

    head_dim = model_config["head_dim"]
    schedules = build_schedules(args.sensitivity_path)

    backends = []
    for sname, sched in schedules.items():
        be = HybridRopeMixedBackend(
            rank_frac=best_rank / head_dim,
            layer_bits=sched,
        )
        be._name = f"hybrid_r{best_rank}_{sname}"
        backends.append((f"hybrid_r{best_rank}_{sname}", be))

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
    save_results(
        all_results,
        dense_ppls,
        outdir,
        3,
        {"gpu_info": gpu_info, "best_rank": best_rank},
    )


def run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 4: Real KV storage profiling."""
    outdir = os.path.join(args.outdir, "phase4")
    os.makedirs(outdir, exist_ok=True)

    # Measure actual GPU memory for different compression configs
    from scripts.bpa_v11_bench import get_text_batch

    results = {}
    for L in valid_L:
        rng = np.random.RandomState(0)
        total_len = L + 256
        idx = get_text_batch(token_data, 1, total_len, rng).to(args.device)
        prefix = idx[:, :L]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Dense baseline memory
        with torch.no_grad():
            out = model(prefix, use_cache=True)
            past = out.past_key_values

        gpu_sync(args.device)
        dense_alloc = torch.cuda.max_memory_allocated() / 1e6
        dense_reserved = torch.cuda.max_memory_reserved() / 1e6

        # Count KV cache bytes
        kv_bytes = 0
        for li in range(len(past)):
            k, v = past[li]
            kv_bytes += k.nelement() * k.element_size()
            kv_bytes += v.nelement() * v.element_size()

        # Measure quantized sizes for uniform INT8 and INT4
        from backends.quant import quantize_int4_block, quantize_int8_symmetric

        int8_bytes = 0
        int4_bytes = 0
        for li in range(len(past)):
            k, v = past[li]
            k_q8, k_s8 = quantize_int8_symmetric(k)
            v_q8, v_s8 = quantize_int8_symmetric(v)
            int8_bytes += k_q8.nelement() * 1 + k_s8.nelement() * k_s8.element_size()
            int8_bytes += v_q8.nelement() * 1 + v_s8.nelement() * v_s8.element_size()

            k_q4, k_s4 = quantize_int4_block(k)
            v_q4, v_s4 = quantize_int4_block(v)
            # INT4: 0.5 byte per element + scales
            int4_bytes += k_q4.nelement() // 2 + k_s4.nelement() * k_s4.element_size()
            int4_bytes += v_q4.nelement() // 2 + v_s4.nelement() * v_s4.element_size()

        # Mixed-precision schedules: per-layer INT8/INT4
        schedules = build_schedules(args.sensitivity_path)
        mixed_bytes = {}
        for sname, sched in schedules.items():
            mb = 0
            for li in range(len(past)):
                k, v = past[li]
                bits = sched[li]
                if bits == 8:
                    kq, ks = quantize_int8_symmetric(k)
                    vq, vs = quantize_int8_symmetric(v)
                    mb += kq.nelement() * 1 + ks.nelement() * ks.element_size()
                    mb += vq.nelement() * 1 + vs.nelement() * vs.element_size()
                else:
                    kq, ks = quantize_int4_block(k)
                    vq, vs = quantize_int4_block(v)
                    mb += kq.nelement() // 2 + ks.nelement() * ks.element_size()
                    mb += vq.nelement() // 2 + vs.nelement() * vs.element_size()
            mixed_bytes[sname] = mb

        # Hybrid: rope_complex K (rank=24 SVD) + mixed V
        # K storage: rank coefficients only (24 floats per head per token)
        # V storage: per-layer quantized as in schedules
        head_dim = model_config["head_dim"]
        n_kv_heads = model_config["n_kv_heads"]
        n_layers = model_config["n_layers"]
        best_rank = 24
        hybrid_bytes = {}
        for sname, sched in schedules.items():
            hb = 0
            for li in range(len(past)):
                k, v = past[li]
                seq_len = k.shape[2]
                # K: store rank coefficients (rank floats per head)
                k_rank_bytes = seq_len * n_kv_heads * best_rank * 2
                hb += k_rank_bytes
                # V: per-layer quantized
                bits = sched[li]
                if bits == 8:
                    vq, vs = quantize_int8_symmetric(v)
                    hb += vq.nelement() * 1 + vs.nelement() * vs.element_size()
                else:
                    vq, vs = quantize_int4_block(v)
                    hb += vq.nelement() // 2 + vs.nelement() * vs.element_size()
            hybrid_bytes[f"hybrid_r{best_rank}_{sname}"] = hb

        entry = {
            "L": L,
            "dense_gpu_alloc_mb": round(dense_alloc, 1),
            "dense_gpu_reserved_mb": round(dense_reserved, 1),
            "kv_cache_bytes_dense": kv_bytes,
            "kv_cache_bytes_int8": int8_bytes,
            "kv_cache_bytes_int4": int4_bytes,
            "kv_ratio_int8": round(int8_bytes / kv_bytes, 4),
            "kv_ratio_int4": round(int4_bytes / kv_bytes, 4),
            "kv_cache_mb_dense": round(kv_bytes / 1e6, 2),
            "kv_cache_mb_int8": round(int8_bytes / 1e6, 2),
            "kv_cache_mb_int4": round(int4_bytes / 1e6, 2),
        }
        for sname, mb in mixed_bytes.items():
            entry[f"kv_cache_bytes_{sname}"] = mb
            entry[f"kv_ratio_{sname}"] = round(mb / kv_bytes, 4)
            entry[f"kv_cache_mb_{sname}"] = round(mb / 1e6, 2)
        for hname, hb in hybrid_bytes.items():
            entry[f"kv_cache_bytes_{hname}"] = hb
            entry[f"kv_ratio_{hname}"] = round(hb / kv_bytes, 4)
            entry[f"kv_cache_mb_{hname}"] = round(hb / 1e6, 2)

        results[L] = entry
        print(
            f"  L={L}: dense={kv_bytes / 1e6:.1f}MB"
            f" INT8={int8_bytes / 1e6:.1f}MB ({int8_bytes / kv_bytes:.3f})"
            f" INT4={int4_bytes / 1e6:.1f}MB ({int4_bytes / kv_bytes:.3f})"
        )
        for sname, mb in mixed_bytes.items():
            print(f"    {sname}: {mb / 1e6:.1f}MB ({mb / kv_bytes:.3f})")
        for hname, hb in hybrid_bytes.items():
            print(f"    {hname}: {hb / 1e6:.1f}MB ({hb / kv_bytes:.3f})")

        del past, out
        torch.cuda.empty_cache()

    # Latency comparison: decode with real quantized vs cast baseline
    print(f"\n{'=' * 60}")
    print("Latency comparison: quantize-dequantize overhead")
    print("=" * 60)

    L_test = 16384
    rng = np.random.RandomState(0)
    idx = get_text_batch(token_data, 1, L_test + 256, rng).to(args.device)
    prefix = idx[:, :L_test]
    continuation = idx[:, L_test : L_test + 256]

    dense_be = DenseBackend()
    dense_be.configure(L_test, model_config)

    from backends.quant import QuantBackend

    int8_be = QuantBackend()
    int8_be.configure(L_test, model_config)
    int8_be.calibrate(model, token_data, L_test, args.device, model_config)

    from backends.rope_aware_kv import RoPEAwareKVBackend

    rope_be = RoPEAwareKVBackend(mode="complex", rank_frac=0.5)
    rope_be.configure(L_test, model_config)
    rope_be.calibrate(model, token_data, L_test, args.device, model_config)

    # Add best hybrid (r24 + S2) to latency test
    schedules = build_schedules(args.sensitivity_path)
    s2_sched = schedules["S2"]
    mixed_s2_be = MixedPrecisionBackend(layer_bits=s2_sched)
    mixed_s2_be._name = "S2"
    mixed_s2_be.configure(L_test, model_config)
    mixed_s2_be.calibrate(model, token_data, L_test, args.device, model_config)

    hybrid_be = HybridRopeMixedBackend(
        rank_frac=24 / model_config["head_dim"],
        layer_bits=s2_sched,
    )
    hybrid_be._name = "hybrid_r24_S2"
    hybrid_be.configure(L_test, model_config)
    hybrid_be.calibrate(model, token_data, L_test, args.device, model_config)

    latency_results = {}
    for bname, be in [
        ("dense", dense_be),
        ("int8", int8_be),
        ("rope_complex", rope_be),
        ("mixed_S2", mixed_s2_be),
        ("hybrid_r24_S2", hybrid_be),
    ]:
        times = []
        for trial in range(3):
            torch.cuda.empty_cache()
            gpu_sync(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, stats = be.run_decode(
                    model, prefix, continuation, args.device, max_ctx
                )
            gpu_sync(args.device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            del logits

        avg_ms = np.mean(times) * 1000
        p50_per_tok = avg_ms / 256
        latency_results[bname] = {
            "total_ms": round(avg_ms, 1),
            "p50_per_tok_ms": round(p50_per_tok, 2),
            "trials": [round(t * 1000, 1) for t in times],
        }
        print(f"  {bname}: {avg_ms:.1f}ms total, {p50_per_tok:.2f}ms/tok")

    results["latency"] = latency_results

    with open(os.path.join(outdir, "storage_profile.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "version": "v16",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {outdir}/")


if __name__ == "__main__":
    main()
