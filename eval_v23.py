#!/usr/bin/env python
"""
BPA v23 Evaluation: Max-out W7900 — throughput story, robust amortization,
1.5B oracle sensitivity, H100 readiness dossier.

Builds on v22's finding that amort_g8_S8_k4 achieves kv_ratio=0.3073
PASS@3% on Qwen2.5-0.5B. v23 turns this into a throughput story,
stress-tests amortization knobs, and does proper empirical sensitivity
on 1.5B.

Phases:
  0: Lock current best + regression guard
  1: Throughput benchmark suite (W7900 capacity story)
  2: Amortization parameter sweep (Pareto frontier)
  3: 1.5B oracle sensitivity + k-sweep
  4: H100 readiness dossier + final deliverables

Usage:
    python eval_v23.py --phase 0
    python eval_v23.py --phase 1
    python eval_v23.py --phase 2
    python eval_v23.py --phase 3 --model qwen15b
    python eval_v23.py --phase 4
"""

import argparse
import gc
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
from eval_v21 import (
    GroupedMixedBackend,
    build_k_schedule,
    check_pass,
    compute_kv_bytes_per_token,
    eval_config,
    load_theory_ranking,
    max_delta,
)
from eval_v22 import (
    AmortizedScaleMixedBackend,
    compute_kv_bytes_amortized,
)


# ============================================================
# Phase 0: Baseline lock + reproducibility
# ============================================================


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Lock baselines and verify no regression."""
    outdir = os.path.join(args.outdir, "phase0")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 0: Lock Current Best + Regression Guard")
    print("=" * 60)

    n_layers = model_config["n_layers"]

    # Dense baselines
    print(f"\n{'=' * 60}")
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

    # Test configs: INT8_all, g32_k4, S2_k6, amort_g8_S8_k4
    theory_ranking = [0, 8, 1, 2, 3, 4, 21, 20, 9, 11, 16, 5]

    configs = OrderedDict()

    # amort_g8_S8_k4 (v22 best)
    sched_k4 = build_k_schedule(theory_ranking, 4, n_layers=n_layers)
    configs["amort_g8_S8_k4"] = {
        "backend": AmortizedScaleMixedBackend(
            layer_bits=sched_k4, group_size=8, scale_window=8
        ),
        "tol": 3.0,
    }

    # g32_k4 (v21 best)
    configs["g32_k4"] = {
        "backend": GroupedMixedBackend(layer_bits=sched_k4, group_size=32),
        "tol": 3.0,
    }

    # S2_k6
    s2_ranking = [0, 8, 2, 16, 1, 11]
    sched_s2 = build_k_schedule(s2_ranking, 6, n_layers=n_layers)
    configs["S2_k6"] = {
        "backend": MixedPrecisionBackend(layer_bits=sched_s2),
        "tol": 3.0,
    }

    # INT8_all
    configs["INT8_all"] = {
        "backend": GroupedMixedBackend(layer_bits=[8] * n_layers, group_size=32),
        "tol": 3.0,
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
        md = max_delta(evals)
        p3 = check_pass(evals, 3.0)
        results[name] = {"evals": evals, "max_delta": round(md, 2), "pass_3pct": p3}
        status = "PASS" if p3 else "FAIL"
        print(f"    max_delta={md:+.2f}% {status}")
        if not p3:
            all_pass = False

    if all_pass:
        print("\n  All regression checks PASSED. Proceeding.")
    else:
        print("\n  WARNING: Some regression checks FAILED!")

    with open(os.path.join(outdir, "regression_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 0,
        "version": "v23",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 0 complete. Saved to {outdir}/")


# ============================================================
# Phase 1: Throughput benchmark suite
# ============================================================


def estimate_kv_mem_bytes(
    n_layers, n_kv_heads, head_dim, seq_len, batch, dtype_bytes=2
):
    """Estimate KV cache memory in bytes for dense fp16."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * dtype_bytes


def run_throughput_trial(
    backend,
    model,
    token_data,
    L,
    batch,
    decode_steps,
    device_str,
    max_ctx,
    model_config,
):
    """Run a single throughput trial. Returns metrics or None on OOM."""
    rng = np.random.RandomState(0)
    n_layers = model_config["n_layers"]

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Prefill
        prefix_len = L - decode_steps
        if prefix_len < 256:
            prefix_len = 256
            decode_steps = min(decode_steps, L - prefix_len)

        total_tokens_out = 0
        total_time_ms = 0

        for b in range(batch):
            idx = get_text_batch(token_data, 1, L, rng).to(device_str)
            prefix_ids = idx[:, :prefix_len]
            cont_ids = idx[:, prefix_len : prefix_len + decode_steps]

            gpu_sync(device_str)
            t0 = time.perf_counter()

            backend.configure(L, model_config, W_min=1024, W_sink=4)
            backend.calibrate(model, token_data, L, device_str, model_config)
            logits, stats = backend.run_decode(
                model, prefix_ids, cont_ids, device_str, max_ctx
            )

            gpu_sync(device_str)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            total_tokens_out += decode_steps
            total_time_ms += elapsed_ms

            del logits, stats, idx, prefix_ids, cont_ids
            torch.cuda.empty_cache()

        peak_mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
        tokens_per_sec = (
            total_tokens_out / (total_time_ms / 1000) if total_time_ms > 0 else 0
        )
        ms_per_token = total_time_ms / total_tokens_out if total_tokens_out > 0 else 0

        return {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "ms_per_token": round(ms_per_token, 2),
            "total_tokens": total_tokens_out,
            "total_time_ms": round(total_time_ms, 2),
            "peak_mem_mb": peak_mem,
            "batch": batch,
            "L": L,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Throughput benchmark suite."""
    outdir = os.path.join(args.outdir, "phase1")
    art_dir = os.path.join(args.outdir, "artifacts", "v23")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 1: Throughput Benchmark Suite (W7900)")
    print("=" * 60)

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]
    decode_steps = args.decode_steps

    theory_ranking = [0, 8, 1, 2, 3, 4, 21, 20, 9, 11, 16, 5]
    sched_k4 = build_k_schedule(theory_ranking, 4, n_layers=n_layers)

    methods = OrderedDict()
    methods["dense"] = DenseBackend()
    methods["INT8_all"] = GroupedMixedBackend(layer_bits=[8] * n_layers, group_size=32)
    methods["g32_k4"] = GroupedMixedBackend(layer_bits=sched_k4, group_size=32)
    methods["amort_g8_S8_k4"] = AmortizedScaleMixedBackend(
        layer_bits=sched_k4, group_size=8, scale_window=8
    )

    batch_sizes = [1, 4, 8, 16]
    throughput_results = {}

    for L in valid_L:
        for batch in batch_sizes:
            batch_key = f"L{L}_b{batch}"
            throughput_results[batch_key] = {}

            for method_name, be in methods.items():
                print(f"\n  {method_name} L={L} batch={batch}...", end="", flush=True)
                result = run_throughput_trial(
                    be,
                    model,
                    token_data,
                    L,
                    batch,
                    decode_steps,
                    args.device,
                    max_ctx,
                    model_config,
                )
                if result is None:
                    print(f" OOM")
                    throughput_results[batch_key][method_name] = {"status": "OOM"}
                else:
                    print(
                        f" {result['tokens_per_sec']:.1f} tok/s, "
                        f"{result['ms_per_token']:.2f} ms/tok, "
                        f"peak={result['peak_mem_mb']}MB"
                    )
                    throughput_results[batch_key][method_name] = result

    # KV memory estimates
    kv_estimates = {}
    for L in valid_L:
        dense_kv = estimate_kv_mem_bytes(n_layers, n_kv_heads, head_dim, L, 1)
        kv_estimates[f"L{L}"] = {
            "dense_kv_bytes": dense_kv,
            "dense_kv_mb": round(dense_kv / (1024 * 1024), 2),
            "amort_g8_S8_k4_kv_mb": round(dense_kv * 0.3073 / (1024 * 1024), 2),
            "g32_k4_kv_mb": round(dense_kv * 0.3203 / (1024 * 1024), 2),
            "int8_kv_mb": round(dense_kv * 0.5156 / (1024 * 1024), 2),
        }

    # Capacity analysis: max sequences at each L
    capacity = {}
    gpu_total_mb = gpu_info.get("total_mem_gb", 48.3) * 1024
    model_mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)

    for L in valid_L:
        dense_kv_per_seq = estimate_kv_mem_bytes(
            n_layers, n_kv_heads, head_dim, L, 1
        ) / (1024 * 1024)
        available_mb = gpu_total_mb - model_mem_mb - 1024  # 1GB headroom
        dense_seqs = int(available_mb / dense_kv_per_seq) if dense_kv_per_seq > 0 else 0
        amort_seqs = (
            int(available_mb / (dense_kv_per_seq * 0.3073))
            if dense_kv_per_seq > 0
            else 0
        )
        g32_seqs = (
            int(available_mb / (dense_kv_per_seq * 0.3203))
            if dense_kv_per_seq > 0
            else 0
        )

        capacity[f"L{L}"] = {
            "dense_max_seqs": dense_seqs,
            "amort_g8_S8_k4_max_seqs": amort_seqs,
            "g32_k4_max_seqs": g32_seqs,
            "capacity_gain_vs_dense": round(amort_seqs / max(dense_seqs, 1), 2),
            "capacity_gain_vs_g32": round(amort_seqs / max(g32_seqs, 1), 2),
        }

    print(f"\n  Capacity Analysis (estimated max concurrent sequences):")
    for key, cap in capacity.items():
        print(
            f"    {key}: dense={cap['dense_max_seqs']}, "
            f"amort={cap['amort_g8_S8_k4_max_seqs']} "
            f"({cap['capacity_gain_vs_dense']:.2f}x vs dense), "
            f"g32={cap['g32_k4_max_seqs']}"
        )

    throughput_out = {
        "gpu": gpu_info,
        "model_mem_mb": model_mem_mb,
        "methods": list(methods.keys()),
        "results": throughput_results,
        "kv_estimates": kv_estimates,
        "capacity": capacity,
    }

    with open(os.path.join(art_dir, "throughput_bench.json"), "w") as f:
        json.dump(throughput_out, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "version": "v23",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 1 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Phase 2: Amortization parameter sweep
# ============================================================


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Amortization parameter sweep for Pareto frontier."""
    outdir = os.path.join(args.outdir, "phase2")
    art_dir = os.path.join(args.outdir, "artifacts", "v23")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 2: Amortization Parameter Sweep")
    print("=" * 60)

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    theory_ranking = [0, 8, 1, 2, 3, 4, 21, 20, 9, 11, 16, 5]

    # Dense baselines
    print(f"\n  Dense baselines")
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

    # Build sweep configs
    sweep_configs = []
    for g in [8, 4]:
        for S in [1, 2, 4, 8, 16]:
            for scale_mode in ["post_rope"]:
                # Only test pre_rope and norm_invariant for g=8 S=8
                modes = ["post_rope"]
                if g == 8 and S == 8:
                    modes = ["post_rope", "pre_rope", "norm_invariant"]
                if scale_mode not in modes:
                    continue
                for k in [4]:  # Fix k=4 for sweep, test k=3 on winners
                    acct = compute_kv_bytes_amortized(
                        n_layers,
                        n_kv_heads,
                        head_dim,
                        4,
                        g,
                        scale_sharing_window=S,
                        k_int8_layers=k,
                    )
                    sweep_configs.append(
                        {
                            "g": g,
                            "S": S,
                            "k": k,
                            "scale_mode": scale_mode,
                            "kv_ratio": acct["kv_ratio"],
                            "name": f"amort_g{g}_S{S}_{scale_mode}_k{k}",
                        }
                    )

    # Add scale mode variants for g=8 S=8
    for scale_mode in ["pre_rope", "norm_invariant"]:
        acct = compute_kv_bytes_amortized(
            n_layers,
            n_kv_heads,
            head_dim,
            4,
            8,
            scale_sharing_window=8,
            k_int8_layers=4,
        )
        sweep_configs.append(
            {
                "g": 8,
                "S": 8,
                "k": 4,
                "scale_mode": scale_mode,
                "kv_ratio": acct["kv_ratio"],
                "name": f"amort_g8_S8_{scale_mode}_k4",
            }
        )

    # Sort by kv_ratio ascending (best compression first)
    sweep_configs.sort(key=lambda c: c["kv_ratio"])

    print(f"\n  Sweep: {len(sweep_configs)} configs")
    for c in sweep_configs:
        print(f"    {c['name']}: kv_ratio={c['kv_ratio']:.4f}")

    # Quick screen at L=8192
    screen_L = [8192]
    screen_results = {}
    survivors = []

    print(f"\n  Quick screen at L={screen_L}")
    for cfg in sweep_configs:
        sched = build_k_schedule(theory_ranking, cfg["k"], n_layers=n_layers)
        be = AmortizedScaleMixedBackend(
            layer_bits=sched,
            group_size=cfg["g"],
            scale_window=cfg["S"],
            scale_mode=cfg["scale_mode"],
        )

        evals = eval_config(
            be,
            model,
            token_data,
            screen_L,
            args.seeds,
            args.device,
            max_ctx,
            model_config,
            dense_ppls,
            args.decode_steps,
        )

        md = max_delta(evals)
        p3 = check_pass(evals, 3.0)

        screen_results[cfg["name"]] = {
            **cfg,
            "evals_8k": evals,
            "max_delta_8k": round(md, 2),
        }

        status = "PASS" if p3 else "FAIL"
        print(f"    {cfg['name']}: max_delta={md:+.2f}% {status}")

        if md <= 4.0:  # Generous filter for validation
            survivors.append(cfg)

    # Validate survivors at full L
    print(f"\n  Validating {len(survivors)} survivors at all L")
    validated = {}

    for cfg in survivors:
        sched = build_k_schedule(theory_ranking, cfg["k"], n_layers=n_layers)
        be = AmortizedScaleMixedBackend(
            layer_bits=sched,
            group_size=cfg["g"],
            scale_window=cfg["S"],
            scale_mode=cfg["scale_mode"],
        )

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

        md = max_delta(evals)
        p3 = check_pass(evals, 3.0)

        validated[cfg["name"]] = {
            **cfg,
            "full_evals": evals,
            "max_delta_allL": round(md, 2),
            "pass_allL_3pct": p3,
            "pass_allL_1pct": check_pass(evals, 1.0),
        }

        status = "PASS" if p3 else "FAIL"
        print(
            f"    {cfg['name']}: max_delta={md:+.2f}% {status} ratio={cfg['kv_ratio']:.4f}"
        )

    # Test k=3 on best survivors that pass
    print(f"\n  Testing k=3 on passing configs")
    k3_results = {}
    for name, v in validated.items():
        if not v["pass_allL_3pct"]:
            continue
        g, S, sm = v["g"], v["S"], v["scale_mode"]
        acct3 = compute_kv_bytes_amortized(
            n_layers,
            n_kv_heads,
            head_dim,
            4,
            g,
            scale_sharing_window=S,
            k_int8_layers=3,
        )
        sched3 = build_k_schedule(theory_ranking, 3, n_layers=n_layers)
        be3 = AmortizedScaleMixedBackend(
            layer_bits=sched3,
            group_size=g,
            scale_window=S,
            scale_mode=sm,
        )
        evals3 = eval_config(
            be3,
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
        md3 = max_delta(evals3)
        p3_3 = check_pass(evals3, 3.0)
        k3_name = f"amort_g{g}_S{S}_{sm}_k3"
        k3_results[k3_name] = {
            "g": g,
            "S": S,
            "k": 3,
            "scale_mode": sm,
            "kv_ratio": acct3["kv_ratio"],
            "full_evals": evals3,
            "max_delta_allL": round(md3, 2),
            "pass_allL_3pct": p3_3,
        }
        status = "PASS" if p3_3 else "FAIL"
        print(
            f"    {k3_name}: max_delta={md3:+.2f}% {status} ratio={acct3['kv_ratio']:.4f}"
        )

    # Check for RoPE drift: does quality worsen with L?
    print(f"\n  RoPE drift check (does delta grow with L?):")
    for name, v in validated.items():
        if not v["pass_allL_3pct"]:
            continue
        deltas_by_L = {}
        for key, ev in v["full_evals"].items():
            L_val = int(key.split("_")[0][1:])
            deltas_by_L.setdefault(L_val, []).append(abs(ev["delta_pct"]))
        mean_by_L = {L: sum(ds) / len(ds) for L, ds in sorted(deltas_by_L.items())}
        drift = list(mean_by_L.values())
        is_drift = len(drift) >= 2 and drift[-1] > drift[0] * 1.5
        print(
            f"    {name}: {' -> '.join(f'L{L}:{d:.2f}%' for L, d in mean_by_L.items())}"
            f" {'DRIFT!' if is_drift else 'OK'}"
        )

    sweep_out = {
        "screen_results": screen_results,
        "validated": validated,
        "k3_results": k3_results,
    }

    with open(os.path.join(art_dir, "amortization_sweep.json"), "w") as f:
        json.dump(sweep_out, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "version": "v23",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 2 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Phase 3: 1.5B oracle sensitivity + k-sweep
# ============================================================


def run_oracle_sensitivity(
    model,
    token_data,
    device_str,
    max_ctx,
    model_config,
    dense_ppls,
    L=8192,
    seeds=None,
    decode_steps=256,
):
    """Empirical per-layer INT4 sensitivity for oracle ranking.

    For each layer, keep all others at INT8 and quantize only that
    layer's KV to INT4 g=32. Measure PPL delta.
    """
    if seeds is None:
        seeds = [0]
    n_layers = model_config["n_layers"]

    layer_scores = []
    for li in range(n_layers):
        # All INT8 except layer li which is INT4
        layer_bits = [8] * n_layers
        layer_bits[li] = 4
        be = GroupedMixedBackend(layer_bits=layer_bits, group_size=32)

        evals = eval_config(
            be,
            model,
            token_data,
            [L],
            seeds,
            device_str,
            max_ctx,
            model_config,
            dense_ppls,
            decode_steps,
        )

        md = max_delta(evals)
        mean_d = sum(abs(e["delta_pct"]) for e in evals.values()) / len(evals)

        layer_scores.append(
            {
                "layer": li,
                "max_delta": round(md, 4),
                "mean_delta": round(mean_d, 4),
            }
        )

        print(f"      Layer {li:2d}: max_delta={md:+.4f}%")

    # Sort by max_delta descending (most sensitive first)
    layer_scores.sort(key=lambda x: x["max_delta"], reverse=True)
    ranking = [s["layer"] for s in layer_scores]

    return ranking, layer_scores


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: 1.5B oracle sensitivity + k-sweep."""
    outdir = os.path.join(args.outdir, "phase3")
    art_dir = os.path.join(args.outdir, "artifacts", "v23")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 3: 1.5B Oracle Sensitivity + k-sweep")
    print(f"  Model: {args.model}")
    print("=" * 60)

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    print(
        f"  Architecture: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}"
    )

    # Limit L for memory safety on 1.5B
    test_L = [L for L in valid_L if L <= max_ctx]
    # Check 32K feasibility
    if 32768 in test_L:
        try:
            torch.cuda.empty_cache()
            mem_free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            if mem_free < 15 * 1024**3:
                print(f"    WARNING: Only {mem_free/1024**3:.1f}GB free, skipping 32K")
                test_L = [L for L in test_L if L <= 16384]
        except Exception:
            pass

    # Step 3a: Dense baselines
    print(f"\n  Step 3a: Dense baselines at L={test_L}")
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

    # Step 3b: INT8-all baseline
    print(f"\n  Step 3b: INT8-all baseline")
    be_int8 = GroupedMixedBackend(layer_bits=[8] * n_layers, group_size=32)
    int8_evals = eval_config(
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
    int8_md = max_delta(int8_evals)
    int8_p3 = check_pass(int8_evals, 3.0)
    print(f"    INT8_all: max_delta={int8_md:+.2f}% {'PASS' if int8_p3 else 'FAIL'}")

    # Step 3c: Oracle sensitivity ranking
    print(f"\n  Step 3c: Oracle sensitivity ranking (1 seed, L=8192)")
    # Use 1 seed for coarse screen to save time
    oracle_ranking, oracle_scores = run_oracle_sensitivity(
        model,
        token_data,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
        L=8192,
        seeds=[0],
        decode_steps=args.decode_steps,
    )

    print(f"\n    Oracle ranking (top-10): {oracle_ranking[:10]}")
    print(f"    Top-6 sensitive layers:")
    for s in oracle_scores[:6]:
        print(f"      Layer {s['layer']}: max_delta={s['max_delta']:+.4f}%")

    # Step 3d: Refine top-8 with 3 seeds
    print(f"\n  Step 3d: Refine top-8 layers with 3 seeds")
    top8_layers = oracle_ranking[:8]
    refined_scores = []
    for li in top8_layers:
        layer_bits = [8] * n_layers
        layer_bits[li] = 4
        be = GroupedMixedBackend(layer_bits=layer_bits, group_size=32)
        evals = eval_config(
            be,
            model,
            token_data,
            [8192],
            args.seeds,
            args.device,
            max_ctx,
            model_config,
            dense_ppls,
            args.decode_steps,
        )
        md = max_delta(evals)
        refined_scores.append({"layer": li, "max_delta_3seed": round(md, 4)})
        print(f"      Layer {li:2d}: max_delta={md:+.4f}% (3 seeds)")

    refined_scores.sort(key=lambda x: x["max_delta_3seed"], reverse=True)
    refined_ranking = [s["layer"] for s in refined_scores]
    # Merge: refined top-8 + remaining from coarse
    final_ranking = refined_ranking + [
        l for l in oracle_ranking if l not in refined_ranking
    ]

    print(f"\n    Final ranking (top-8): {final_ranking[:8]}")

    # Step 3e: k-sweep using oracle ranking
    print(f"\n  Step 3e: k-sweep with oracle ranking")

    # Byte accounting
    acct_table = OrderedDict()
    for k in [2, 3, 4, 6, 8]:
        name = f"g32_k{k}"
        acct_table[name] = compute_kv_bytes_per_token(
            n_layers, n_kv_heads, head_dim, 4, 32, k_int8_layers=k
        )

    k_results = {}
    for k in [2, 3, 4, 6, 8]:
        sched = build_k_schedule(final_ranking, k, n_layers=n_layers)
        be = GroupedMixedBackend(layer_bits=sched, group_size=32)
        name = f"g32_k{k}_oracle"

        print(f"\n    {name} (protected: {final_ranking[:k]})")
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

        md = max_delta(evals)
        p3 = check_pass(evals, 3.0)
        ratio = acct_table[f"g32_k{k}"]["kv_ratio"]

        k_results[name] = {
            "k": k,
            "ranking": final_ranking[:k],
            "evals": evals,
            "pass_3pct": p3,
            "max_delta": round(md, 2),
            "kv_ratio": ratio,
        }

        status = "PASS" if p3 else "FAIL"
        print(f"      max_delta={md:+.2f}% {status} kv_ratio={ratio:.4f}")

    # Step 3f: Test amort_g8_S8 on best passing k
    print(f"\n  Step 3f: Test amort_g8_S8 with oracle ranking")
    amort_results = {}
    for k in [4, 6]:
        sched = build_k_schedule(final_ranking, k, n_layers=n_layers)
        be = AmortizedScaleMixedBackend(layer_bits=sched, group_size=8, scale_window=8)
        acct = compute_kv_bytes_amortized(
            n_layers,
            n_kv_heads,
            head_dim,
            4,
            8,
            scale_sharing_window=8,
            k_int8_layers=k,
        )
        name = f"amort_g8_S8_k{k}_oracle"

        print(f"\n    {name} (protected: {final_ranking[:k]})")
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

        md = max_delta(evals)
        p3 = check_pass(evals, 3.0)
        ratio = acct["kv_ratio"]

        amort_results[name] = {
            "k": k,
            "ranking": final_ranking[:k],
            "evals": evals,
            "pass_3pct": p3,
            "max_delta": round(md, 2),
            "kv_ratio": ratio,
        }

        status = "PASS" if p3 else "FAIL"
        print(f"      max_delta={md:+.2f}% {status} kv_ratio={ratio:.4f}")

    # Compare with v22 transferred ranking
    print(f"\n  k/D Comparison:")
    print(f"    0.5B: k*=4, D=24, k/D={4/24:.4f}, kv_ratio=0.3203")
    k_star = None
    for k in [2, 3, 4, 6, 8]:
        name = f"g32_k{k}_oracle"
        res = k_results.get(name, {})
        if res.get("pass_3pct"):
            kd = k / n_layers
            print(
                f"    1.5B: {name} k={k}, D={n_layers}, "
                f"k/D={kd:.4f}, kv_ratio={res['kv_ratio']:.4f} PASS"
            )
            if k_star is None:
                k_star = k
        else:
            print(
                f"    1.5B: {name} k={k}, max_delta={res.get('max_delta', '?')}% FAIL"
            )

    if k_star:
        print(f"\n    k* (1.5B) = {k_star}")

    # Save
    phase3_out = {
        "model": args.model,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "test_L": test_L,
        "oracle_ranking": oracle_ranking,
        "oracle_scores": oracle_scores,
        "refined_ranking": final_ranking,
        "refined_scores": refined_scores,
        "int8_all": {
            "evals": int8_evals,
            "pass_3pct": int8_p3,
            "max_delta": round(int8_md, 2),
        },
        "k_results": k_results,
        "amort_results": amort_results,
        "k_star": k_star,
    }

    with open(os.path.join(art_dir, "model_1p5b_sensitivity_oracle.json"), "w") as f:
        json.dump(phase3_out, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "version": "v23",
        "model": args.model,
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 3 complete. Saved to {outdir}/ and {art_dir}/")


# ============================================================
# Phase 4: H100 readiness dossier + final deliverables
# ============================================================


def run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 4: H100 readiness dossier + final deliverables."""
    outdir = os.path.join(args.outdir, "phase4")
    art_dir = os.path.join(args.outdir, "artifacts", "v23")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 4: H100 Readiness Dossier + Final Deliverables")
    print("=" * 60)

    # Load all phase results
    p0_path = os.path.join(args.outdir, "phase0", "regression_results.json")
    p1_path = os.path.join(art_dir, "throughput_bench.json")
    p2_path = os.path.join(art_dir, "amortization_sweep.json")
    p3_path = os.path.join(art_dir, "model_1p5b_sensitivity_oracle.json")

    phase_data = {}
    for name, path in [
        ("p0", p0_path),
        ("p1", p1_path),
        ("p2", p2_path),
        ("p3", p3_path),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                phase_data[name] = json.load(f)
        else:
            phase_data[name] = {}

    # Build scoreboard
    scoreboard = {
        "model": "Qwen2.5-0.5B",
        "experiment": "BPA v23: Throughput + amort sweep + 1.5B oracle",
        "baselines": {
            "dense": {"kv_ratio": 1.0, "pass_3pct": "all"},
            "INT8_all": {"kv_ratio": 0.5156, "pass_3pct": "all"},
            "S2_k6": {"kv_ratio": 0.333, "pass_3pct": "all"},
            "g32_k4": {"kv_ratio": 0.3203, "pass_3pct": "all"},
            "amort_g8_S8_k4_v22": {"kv_ratio": 0.3073, "pass_3pct": "all"},
        },
        "phase1_throughput": phase_data.get("p1", {}).get("capacity", {}),
        "phase2_amort_sweep": {},
        "phase3_1p5b": {},
    }

    # Populate phase 2
    p2 = phase_data.get("p2", {})
    for name, v in p2.get("validated", {}).items():
        scoreboard["phase2_amort_sweep"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta_allL"),
            "pass_3pct": v.get("pass_allL_3pct"),
        }
    for name, v in p2.get("k3_results", {}).items():
        scoreboard["phase2_amort_sweep"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta_allL"),
            "pass_3pct": v.get("pass_allL_3pct"),
        }

    # Populate phase 3
    p3 = phase_data.get("p3", {})
    for name, v in p3.get("k_results", {}).items():
        scoreboard["phase3_1p5b"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta"),
            "pass_3pct": v.get("pass_3pct"),
            "k": v.get("k"),
        }
    for name, v in p3.get("amort_results", {}).items():
        scoreboard["phase3_1p5b"][name] = {
            "kv_ratio": v.get("kv_ratio"),
            "max_delta": v.get("max_delta"),
            "pass_3pct": v.get("pass_3pct"),
            "k": v.get("k"),
        }

    scoreboard["phase3_1p5b"]["k_star"] = p3.get("k_star")
    scoreboard["phase3_1p5b"]["oracle_ranking_top8"] = p3.get("refined_ranking", [])[:8]

    with open("bpa_v23_scoreboard.json", "w") as f:
        json.dump(scoreboard, f, indent=2)
    print("    Written: bpa_v23_scoreboard.json")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "version": "v23",
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
        description="BPA v23: Throughput + amort sweep + 1.5B oracle"
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
    parser.add_argument("--outdir", default="results/v23")
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
