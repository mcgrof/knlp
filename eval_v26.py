#!/usr/bin/env python
"""
BPA v26 Evaluation: H100 validation of O(1) k* scaling hypothesis.

Extends v24 to larger models on H100 GPU:
- Qwen2.5-7B (D=28, 4 KV heads, head_dim=128) — same D, more capacity
- Mistral-7B-v0.1 (D=32, 8 KV heads, head_dim=128) — new D data point

Phases:
  0: Dense baselines (3 seeds, L={8192, 32768})
  1: Oracle sensitivity ranking (per-layer INT4 ablation)
  2: k-sweep for k*(D,ε) determination
  3: Latency/throughput benchmark (batch sweep)
  4: Final report, scoreboard, figures

Usage:
    python eval_v26.py --phase 0 --model qwen7b
    python eval_v26.py --phase 1 --model qwen7b
    python eval_v26.py --phase 2 --model qwen7b
    python eval_v26.py --phase 0 --model mistral7b
    python eval_v26.py --phase 1 --model mistral7b
    python eval_v26.py --phase 2 --model mistral7b
    python eval_v26.py --phase 3 --model qwen7b
    python eval_v26.py --phase 4
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
    run_single_eval,
)
from eval_v16 import (
    MixedPrecisionBackend,
    build_schedules,
    run_dense_baselines,
    save_results,
)
from eval_v21 import (
    GroupedMixedBackend,
    build_k_schedule,
    check_pass,
    compute_kv_bytes_per_token,
    eval_config,
    max_delta,
)

# ============================================================
# GPU preflight — works on both CUDA (H100) and ROCm (W7900)
# ============================================================


def gpu_preflight(device_str):
    """Verify GPU and log info."""
    assert torch.cuda.is_available(), "CUDA not available"
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    hip = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)
    backend = f"hip={hip}" if hip else f"cuda={cuda_ver}"
    print(f"GPU Preflight OK: {props.name} ({total_gb:.1f}GB)")
    print(f"  torch={torch.__version__} {backend}")
    return {
        "device_name": props.name,
        "total_gb": round(total_gb, 1),
        "torch_version": torch.__version__,
        "backend": backend,
    }


# ============================================================
# DynamicCache compatibility for transformers 5.x
# ============================================================


def _cache_get_kv(past, layer_idx):
    """Get (key, value) tensors from cache, compatible with old and new API."""
    if hasattr(past, "layers"):
        # transformers 5.x: DynamicCache with DynamicLayer objects
        layer = past.layers[layer_idx]
        return layer.keys, layer.values
    else:
        # transformers 4.x: tuple of (key, value) per layer
        return past[layer_idx]


def _cache_num_layers(past):
    """Get number of layers in cache."""
    if hasattr(past, "layers"):
        return len(past.layers)
    else:
        return len(past)


class DenseBackendV26(DenseBackend):
    """Dense backend with transformers 5.x DynamicCache compatibility."""

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2  # fp16

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            all_logits.append(out.logits)

            k0, _ = _cache_get_kv(past, 0)
            cache_len = k0.shape[2]
            bpt = 2 * n_kv_heads * head_dim * elem
            total_bytes = cache_len * bpt * n_layers
            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=total_bytes,
                    kv_bytes_compressed=0,
                    kv_bytes_total=total_bytes,
                    n_full=cache_len,
                    n_compressed=0,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats


class GroupedMixedBackendV26(GroupedMixedBackend):
    """Subclass with transformers 5.x DynamicCache compatibility."""

    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        from transformers.cache_utils import DynamicCache

        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        k0, v0 = _cache_get_kv(past, 0)
        dtype = k0.dtype
        all_logits = [out.logits[:, -1:, :]]
        del out
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        cache_len = k0.shape[2]

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
                    k, v = _cache_get_kv(past, li)
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
# Model registry — extended for 7B models
# ============================================================


MODEL_REGISTRY = {
    "qwen05b": "Qwen/Qwen2.5-0.5B",
    "qwen15b": "Qwen/Qwen2.5-1.5B",
    "qwen7b": "Qwen/Qwen2.5-7B",
    "mistral7b": "mistralai/Mistral-7B-v0.1",
}


def run_dense_baselines_v26(
    model, token_data, valid_L, decode_steps, seeds, device, max_ctx, model_config
):
    """Run dense baselines using V26 backend (transformers 5.x compatible)."""
    print(f"\n{'=' * 60}")
    print("Dense baselines")
    print("=" * 60)

    dense_be = DenseBackendV26()
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


def load_model_v26(model_key, device_str):
    """Load HF model with v26 registry."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_name = MODEL_REGISTRY.get(model_key, model_key)
    print(f"Loading model {model_name}...")

    config = AutoConfig.from_pretrained(model_name)
    max_ctx = getattr(config, "max_position_embeddings", 1024)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads

    rope_theta = getattr(config, "rope_theta", 10000.0)
    rope_type = getattr(config, "rope_scaling", None)

    model_config = {
        "n_layers": n_layers,
        "hidden_size": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "rope_scaling": rope_type,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)
    model = model.to(device_str).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M max_ctx={max_ctx}")
    print(
        f"  layers={n_layers} hidden={hidden} heads={n_heads} "
        f"kv_heads={n_kv_heads} head_dim={head_dim}"
    )
    print(f"  rope_theta={rope_theta}")

    return model, tokenizer, max_ctx, model_config


# ============================================================
# Phase 0: Dense baselines
# ============================================================


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Lock dense baselines and INT8-all sanity."""
    outdir = os.path.join(args.outdir, "phase0")
    os.makedirs(outdir, exist_ok=True)

    n_layers = model_config["n_layers"]
    model_name = args.model

    print(f"\n{'=' * 60}")
    print(f"Phase 0: Dense Baselines ({model_name}, D={n_layers})")
    print("=" * 60)

    dense_results, dense_ppls = run_dense_baselines_v26(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    print(f"\n  Dense PPLs:")
    for key, ppl in sorted(dense_ppls.items()):
        L, regime, seed = key
        print(f"    L={L} seed={seed}: PPL={ppl:.4f}")

    # INT8-all baseline
    be_int8 = GroupedMixedBackendV26(layer_bits=[8] * n_layers, group_size=32)
    int8_evals = eval_config(
        be_int8,
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
    int8_md = max_delta(int8_evals)
    int8_p3 = check_pass(int8_evals, 3.0)
    int8_p1 = check_pass(int8_evals, 1.0)
    print(
        f"\n  INT8_all: max_delta={int8_md:+.2f}%"
        f" PASS@1%={int8_p1} PASS@3%={int8_p3}"
    )

    phase0_out = {
        "version": "v26",
        "model": model_name,
        "n_layers": n_layers,
        "head_dim": model_config["head_dim"],
        "n_kv_heads": model_config["n_kv_heads"],
        "dense_ppls": {f"L{L}_s{s}": ppl for (L, _, s), ppl in dense_ppls.items()},
        "int8_all": {
            "evals": int8_evals,
            "max_delta": round(int8_md, 2),
            "pass_1pct": int8_p1,
            "pass_3pct": int8_p3,
        },
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
    }

    outfile = os.path.join(outdir, f"phase0_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(phase0_out, f, indent=2)

    print(f"\nPhase 0 ({model_name}) complete. Saved to {outfile}")
    return dense_ppls


# ============================================================
# Phase 1: Oracle sensitivity ranking
# ============================================================


def measure_oracle_sensitivity(
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
    """Per-layer INT4 sensitivity (oracle ablation).

    For each layer l, quantize only that layer's KV to INT4 g=32
    while all others stay INT8. Returns sorted scores.
    """
    if seeds is None:
        seeds = [0]
    n_layers = model_config["n_layers"]

    layer_scores = []
    for li in range(n_layers):
        layer_bits = [8] * n_layers
        layer_bits[li] = 4
        be = GroupedMixedBackendV26(layer_bits=layer_bits, group_size=32)

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

    layer_scores.sort(key=lambda x: x["max_delta"], reverse=True)
    ranking = [s["layer"] for s in layer_scores]

    return ranking, layer_scores


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Oracle sensitivity ranking."""
    outdir = os.path.join(args.outdir, "phase1")
    art_dir = os.path.join(args.outdir, "artifacts", "v26")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    n_layers = model_config["n_layers"]
    model_name = args.model

    print(f"\n{'=' * 60}")
    print(f"Phase 1: Oracle Sensitivity ({model_name}, D={n_layers})")
    print("=" * 60)

    # Dense baselines at oracle L
    oracle_L = 8192
    print(f"\n  Dense baselines at L={oracle_L}")
    dense_results, dense_ppls = run_dense_baselines_v26(
        model,
        token_data,
        [oracle_L],
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    # Oracle sensitivity
    print(f"\n  Per-layer INT4 ablation at L={oracle_L}, seeds={args.seeds}")
    oracle_ranking, oracle_scores = measure_oracle_sensitivity(
        model,
        token_data,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
        L=oracle_L,
        seeds=args.seeds,
        decode_steps=args.decode_steps,
    )

    print(f"\n  Oracle ranking (top-8): {oracle_ranking[:8]}")
    for s in oracle_scores[:8]:
        print(f"    Layer {s['layer']:2d}: delta={s['max_delta']:+.4f}%")

    # Tail analysis
    all_deltas = sorted([s["max_delta"] for s in oracle_scores], reverse=True)
    for C in [1, 2, 4]:
        top_sum = sum(all_deltas[:C])
        tail_sum = sum(all_deltas[C:])
        total = top_sum + tail_sum
        tail_frac = tail_sum / max(total, 1e-10)
        print(
            f"  C={C}: top_sum={top_sum:.4f}, "
            f"tail_sum={tail_sum:.4f}, tail_frac={tail_frac:.4f}"
        )

    # Save
    phase1_out = {
        "version": "v26",
        "model": model_name,
        "n_layers": n_layers,
        "oracle_L": oracle_L,
        "seeds": args.seeds,
        "oracle_ranking": oracle_ranking,
        "oracle_scores": oracle_scores,
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
    }

    outfile = os.path.join(art_dir, f"oracle_sensitivity_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(phase1_out, f, indent=2)

    print(f"\nPhase 1 ({model_name}) complete. Saved to {outfile}")


# ============================================================
# Phase 2: k-sweep for k* determination
# ============================================================


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: k-sweep for k*(D,eps)."""
    outdir = os.path.join(args.outdir, "phase2")
    art_dir = os.path.join(args.outdir, "artifacts", "v26")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]
    model_name = args.model

    print(f"\n{'=' * 60}")
    print(f"Phase 2: k*(D,eps) Determination ({model_name}, D={n_layers})")
    print("=" * 60)

    # Load oracle ranking from Phase 1
    oracle_file = os.path.join(art_dir, f"oracle_sensitivity_{model_name}.json")
    if os.path.exists(oracle_file):
        with open(oracle_file) as f:
            p1_data = json.load(f)
        oracle_ranking = p1_data["oracle_ranking"]
        print(f"  Oracle ranking (top-8): {oracle_ranking[:8]}")
    else:
        print(f"  ERROR: No Phase 1 data for {model_name}")
        print(f"  Run: python eval_v26.py --phase 1 --model {model_name}")
        return

    # Dense baselines at full L set
    print(f"\n  Dense baselines at L={valid_L}")
    dense_results, dense_ppls = run_dense_baselines_v26(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    # k-sweep
    k_values = [0, 1, 2, 3, 4, 6, 8]
    k_values = [k for k in k_values if k <= n_layers]
    if n_layers >= 16:
        k_values.append(12)

    print(f"\n  k-sweep: k in {k_values}")
    print(f"  L in {valid_L}, seeds in {args.seeds}")

    k_results = OrderedDict()
    for k in k_values:
        sched = build_k_schedule(oracle_ranking, k, n_layers=n_layers)
        be = GroupedMixedBackendV26(layer_bits=sched, group_size=32)
        name = f"g32_k{k}"

        protected = oracle_ranking[:k] if k > 0 else []
        print(f"\n    {name} (protected: {protected})")

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
        p1 = check_pass(evals, 1.0)
        p3 = check_pass(evals, 3.0)

        acct = compute_kv_bytes_per_token(
            n_layers, n_kv_heads, head_dim, 4, 32, k_int8_layers=k
        )

        k_results[name] = {
            "k": k,
            "protected_layers": protected,
            "evals": evals,
            "max_delta": round(md, 2),
            "pass_1pct": p1,
            "pass_3pct": p3,
            "kv_ratio": acct["kv_ratio"],
        }

        print(
            f"      max_delta={md:+.2f}% "
            f"PASS@1%={p1} PASS@3%={p3} "
            f"ratio={acct['kv_ratio']:.4f}"
        )

    # Determine k*
    k_star_1pct = None
    k_star_3pct = None
    for k in sorted(k_values):
        name = f"g32_k{k}"
        res = k_results.get(name, {})
        if res.get("pass_1pct") and k_star_1pct is None:
            k_star_1pct = k
        if res.get("pass_3pct") and k_star_3pct is None:
            k_star_3pct = k

    print(f"\n  k* for {model_name} (D={n_layers}):")
    print(f"    k*(eps=1%) = {k_star_1pct}")
    print(f"    k*(eps=3%) = {k_star_3pct}")
    if k_star_3pct is not None:
        ratio_at_kstar = k_results[f"g32_k{k_star_3pct}"]["kv_ratio"]
        kd = k_star_3pct / n_layers
        print(f"    k*/D = {kd:.4f}")
        print(f"    kv_ratio at k* = {ratio_at_kstar:.4f}")

    # Save
    phase2_out = {
        "version": "v26",
        "model": model_name,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "k_star_1pct": k_star_1pct,
        "k_star_3pct": k_star_3pct,
        "k_over_D_3pct": (
            round(k_star_3pct / n_layers, 4) if k_star_3pct is not None else None
        ),
        "k_results": k_results,
        "oracle_ranking_used": oracle_ranking[:12],
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
    }

    outfile = os.path.join(art_dir, f"k_star_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(phase2_out, f, indent=2)

    print(f"\nPhase 2 ({model_name}) complete. Saved to {outfile}")


# ============================================================
# Phase 3: Latency / throughput benchmark
# ============================================================


def benchmark_latency(
    model,
    token_data,
    device_str,
    max_ctx,
    model_config,
    oracle_ranking,
    k_star,
    n_layers,
    L_list,
    batch_sizes,
    decode_steps=64,
):
    """Benchmark decode latency across batch sizes and L.

    Returns list of dicts with latency measurements.
    """
    results = []
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    methods = {
        "dense": [16] * n_layers,
        "int8_all": [8] * n_layers,
    }
    if k_star is not None:
        sched = build_k_schedule(oracle_ranking, k_star, n_layers=n_layers)
        methods[f"int4_k{k_star}"] = sched

    for L in L_list:
        if L > max_ctx:
            continue
        for method_name, layer_bits in methods.items():
            for batch in batch_sizes:
                torch.cuda.empty_cache()
                gc.collect()

                is_dense = all(b == 16 for b in layer_bits)

                print(
                    f"    L={L} batch={batch} method={method_name}...",
                    end="",
                    flush=True,
                )

                try:
                    # Generate fake prefix tokens of length L
                    rng = np.random.RandomState(0)
                    tokens = get_text_batch(token_data, 1, L + decode_steps, rng)
                    if tokens.shape[1] < L + decode_steps:
                        print(" SKIP (not enough tokens)")
                        continue
                    # Replicate for batch
                    prefix = tokens[:, :L].to(device_str)
                    if batch > 1:
                        prefix = prefix.expand(batch, -1).contiguous()
                    continuation = tokens[:, L : L + decode_steps].to(device_str)
                    if batch > 1:
                        continuation = continuation.expand(batch, -1).contiguous()

                    if is_dense:
                        # Dense: just do prefill+decode
                        be = DenseBackendV26()
                        be.configure(L, model_config)
                    else:
                        be = GroupedMixedBackendV26(
                            layer_bits=layer_bits, group_size=32
                        )

                    be.configure(L, model_config)
                    be.calibrate(model, token_data, L, device_str, model_config)

                    # Warmup
                    torch.cuda.synchronize()
                    logits, stats = be.run_decode(
                        model, prefix, continuation, device_str, max_ctx
                    )
                    del logits, stats
                    torch.cuda.synchronize()

                    # Timed runs
                    n_runs = 3
                    times = []
                    for _ in range(n_runs):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        logits, stats = be.run_decode(
                            model, prefix, continuation, device_str, max_ctx
                        )
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        times.append((t1 - t0) * 1000)
                        del logits, stats

                    median_ms = sorted(times)[len(times) // 2]
                    ms_per_token = median_ms / decode_steps

                    # Memory
                    mem_mb = torch.cuda.max_memory_allocated() / 1e6
                    torch.cuda.reset_peak_memory_stats()

                    result = {
                        "L": L,
                        "batch": batch,
                        "method": method_name,
                        "total_ms": round(median_ms, 2),
                        "ms_per_token": round(ms_per_token, 3),
                        "peak_mem_mb": round(mem_mb, 1),
                        "decode_steps": decode_steps,
                    }
                    results.append(result)
                    print(f" {ms_per_token:.2f} ms/tok, " f"peak={mem_mb:.0f}MB")

                except torch.cuda.OutOfMemoryError:
                    print(" OOM")
                    torch.cuda.empty_cache()
                    gc.collect()
                    results.append(
                        {
                            "L": L,
                            "batch": batch,
                            "method": method_name,
                            "error": "OOM",
                        }
                    )

    return results


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Latency/throughput benchmark."""
    outdir = os.path.join(args.outdir, "phase3")
    art_dir = os.path.join(args.outdir, "artifacts", "v26")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    n_layers = model_config["n_layers"]
    model_name = args.model

    print(f"\n{'=' * 60}")
    print(f"Phase 3: Latency Benchmark ({model_name}, D={n_layers})")
    print("=" * 60)

    # Load k* from Phase 2
    kstar_file = os.path.join(art_dir, f"k_star_{model_name}.json")
    if os.path.exists(kstar_file):
        with open(kstar_file) as f:
            p2_data = json.load(f)
        k_star = p2_data.get("k_star_3pct")
        oracle_ranking = p2_data.get("oracle_ranking_used", [])
        print(f"  k*(3%) = {k_star}")
    else:
        print(f"  WARNING: No Phase 2 data, using k=2")
        k_star = 2
        oracle_ranking = list(range(n_layers))

    # Benchmark config
    L_list = [8192, 16384, 32768]
    batch_sizes = [1, 4, 8, 16, 32]

    print(f"\n  L in {L_list}")
    print(f"  batch in {batch_sizes}")
    print(f"  decode_steps = {args.bench_decode_steps}")

    results = benchmark_latency(
        model,
        token_data,
        args.device,
        max_ctx,
        model_config,
        oracle_ranking,
        k_star,
        n_layers,
        L_list,
        batch_sizes,
        decode_steps=args.bench_decode_steps,
    )

    # Analysis: find bandwidth-bound regime
    print(f"\n  Latency Analysis:")
    dense_results = {
        (r["L"], r["batch"]): r["ms_per_token"]
        for r in results
        if r["method"] == "dense" and "error" not in r
    }
    compressed_results = {
        (r["L"], r["batch"]): r["ms_per_token"]
        for r in results
        if r["method"] == f"int4_k{k_star}" and "error" not in r
    }

    best_speedup = 0
    best_config = None
    for key in dense_results:
        if key in compressed_results:
            dense_ms = dense_results[key]
            comp_ms = compressed_results[key]
            speedup = (dense_ms - comp_ms) / dense_ms * 100
            L, batch = key
            print(
                f"    L={L} batch={batch}: "
                f"dense={dense_ms:.2f} comp={comp_ms:.2f} "
                f"speedup={speedup:+.1f}%"
            )
            if speedup > best_speedup:
                best_speedup = speedup
                best_config = key

    if best_config:
        print(
            f"\n  Best speedup: {best_speedup:.1f}% at "
            f"L={best_config[0]} batch={best_config[1]}"
        )
        criterion_c = best_speedup >= 10.0
        print(f"  Criterion C (>=10% latency): {'PASS' if criterion_c else 'FAIL'}")

    # Capacity analysis
    print(f"\n  Capacity Analysis:")
    for L in L_list:
        dense_mem = None
        comp_mem = None
        for r in results:
            if r.get("error"):
                continue
            if r["L"] == L and r["batch"] == 1:
                if r["method"] == "dense":
                    dense_mem = r["peak_mem_mb"]
                elif r["method"] == f"int4_k{k_star}":
                    comp_mem = r["peak_mem_mb"]
        if dense_mem and comp_mem:
            ratio = dense_mem / comp_mem
            print(
                f"    L={L}: dense={dense_mem:.0f}MB comp={comp_mem:.0f}MB ratio={ratio:.2f}x"
            )

    # Save
    phase3_out = {
        "version": "v26",
        "model": model_name,
        "n_layers": n_layers,
        "k_star_3pct": k_star,
        "benchmark_results": results,
        "best_speedup_pct": round(best_speedup, 2),
        "best_config": (
            {"L": best_config[0], "batch": best_config[1]} if best_config else None
        ),
        "criterion_c_pass": best_speedup >= 10.0,
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
    }

    outfile = os.path.join(art_dir, f"latency_benchmark_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(phase3_out, f, indent=2)

    print(f"\nPhase 3 ({model_name}) complete. Saved to {outfile}")


# ============================================================
# Phase 4: Final report and scoreboard
# ============================================================


def run_phase4(args):
    """Phase 4: Aggregate results, generate report and scoreboard."""
    art_dir = os.path.join(args.outdir, "artifacts", "v26")
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 4: Final Report and Scoreboard")
    print("=" * 60)

    # Load all model results
    models_data = {}
    for model_key in ["qwen7b", "mistral7b"]:
        kstar_file = os.path.join(art_dir, f"k_star_{model_key}.json")
        oracle_file = os.path.join(art_dir, f"oracle_sensitivity_{model_key}.json")
        latency_file = os.path.join(art_dir, f"latency_benchmark_{model_key}.json")

        data = {}
        for fname, key in [
            (kstar_file, "k_star"),
            (oracle_file, "oracle"),
            (latency_file, "latency"),
        ]:
            if os.path.exists(fname):
                with open(fname) as f:
                    data[key] = json.load(f)

        if data:
            models_data[model_key] = data

    if not models_data:
        print("  No model results found. Run phases 0-3 first.")
        return

    # O(1) scaling table — combine with v24 data
    print("\n  O(1) Scaling Evidence:")
    print(f"  {'Model':<20} {'D':>3} {'k*(3%)':>7} {'k*/D':>7} {'k*(1%)':>7}")
    print("  " + "-" * 50)

    # v24 reference data
    v24_data = [
        ("Qwen2.5-0.5B", 24, 2, None),
        ("Qwen2.5-1.5B", 28, 2, 3),
    ]
    for name, D, k3, k1 in v24_data:
        kd = k3 / D if k3 is not None else None
        print(
            f"  {name:<20} {D:>3} "
            f"{str(k3) if k3 is not None else 'N/A':>7} "
            f"{f'{kd:.4f}' if kd else 'N/A':>7} "
            f"{str(k1) if k1 is not None else 'N/A':>7}"
        )

    # v26 data
    for model_key, data in models_data.items():
        if "k_star" not in data:
            continue
        ks = data["k_star"]
        name = MODEL_REGISTRY.get(model_key, model_key)
        D = ks["n_layers"]
        k3 = ks.get("k_star_3pct")
        k1 = ks.get("k_star_1pct")
        kd = k3 / D if k3 is not None else None
        print(
            f"  {name:<20} {D:>3} "
            f"{str(k3) if k3 is not None else 'N/A':>7} "
            f"{f'{kd:.4f}' if kd else 'N/A':>7} "
            f"{str(k1) if k1 is not None else 'N/A':>7}"
        )

    # Acceptance criteria
    print(f"\n  Acceptance Criteria Assessment:")
    all_k3 = []
    for data in models_data.values():
        if "k_star" in data:
            k3 = data["k_star"].get("k_star_3pct")
            if k3 is not None:
                all_k3.append(k3)

    if all_k3:
        max_k3 = max(all_k3)
        crit_a = max_k3 <= 4
        print(
            f"  A) O(1): max k*(3%) = {max_k3} {'PASS' if crit_a else 'FAIL'} (need <=4)"
        )
    else:
        print("  A) O(1): no data")

    # kv_ratio
    for model_key, data in models_data.items():
        if "k_star" not in data:
            continue
        ks = data["k_star"]
        k3 = ks.get("k_star_3pct")
        if k3 is not None:
            kr = ks["k_results"].get(f"g32_k{k3}", {}).get("kv_ratio")
            if kr:
                crit_b = kr <= 0.30
                print(
                    f"  B) kv_ratio ({model_key}): {kr:.4f} "
                    f"{'PASS' if crit_b else 'FAIL'} (need <=0.30)"
                )

    # Latency
    for model_key, data in models_data.items():
        if "latency" in data:
            best = data["latency"].get("best_speedup_pct", 0)
            crit_c = best >= 10.0
            print(
                f"  C) Latency ({model_key}): {best:.1f}% "
                f"{'PASS' if crit_c else 'FAIL'} (need >=10%)"
            )

    # Save scoreboard
    scoreboard = {
        "version": "v26",
        "models": {},
        "timestamp": datetime.now().isoformat(),
    }
    for model_key, data in models_data.items():
        entry = {}
        if "k_star" in data:
            entry["k_star_3pct"] = data["k_star"].get("k_star_3pct")
            entry["k_star_1pct"] = data["k_star"].get("k_star_1pct")
            entry["n_layers"] = data["k_star"]["n_layers"]
        if "oracle" in data:
            entry["oracle_ranking_top8"] = data["oracle"]["oracle_ranking"][:8]
        if "latency" in data:
            entry["best_speedup_pct"] = data["latency"].get("best_speedup_pct")
            entry["criterion_c_pass"] = data["latency"].get("criterion_c_pass")
        scoreboard["models"][model_key] = entry

    sb_file = os.path.join(args.outdir, "bpa_v26_scoreboard.json")
    with open(sb_file, "w") as f:
        json.dump(scoreboard, f, indent=2)

    print(f"\n  Scoreboard saved to {sb_file}")
    print(f"\nPhase 4 complete.")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="BPA v26: H100 validation of O(1) k* scaling"
    )
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--model", default="qwen7b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--bench_decode_steps", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--L",
        nargs="+",
        type=int,
        default=[8192, 32768],
    )
    parser.add_argument("--outdir", default="results/v26")
    args = parser.parse_args()

    if args.phase == 4:
        run_phase4(args)
        return

    # GPU phases
    gpu_info = gpu_preflight(args.device)
    model, tokenizer, max_ctx, model_config = load_model_v26(args.model, args.device)

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
    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
