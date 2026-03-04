#!/usr/bin/env python3
"""BPA v28 Phase 2.1: Benchmark current Python INT4 path.

Measures per-token latency and kernel breakdown for:
1. Dense fp16 baseline (no quantization)
2. INT8 (all layers INT8)
3. Current g32 mixed precision (k* INT8, rest INT4)

Identifies where the Python overhead lives by profiling
quantization, cache manipulation, and attention separately.
"""

import gc
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/mnt/tmpfs/knlp/results/v28")
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Protocol
SEEDS = [0]
W_SINK = 4
W_MIN = 1024
GROUP_SIZE = 32
DECODE_TOKENS = 64
DATASET = "wikitext-103-raw-v1"
N_TOKENS = 500000

# Benchmark model (smallest to reduce variable time)
MODEL_KEY = "mistral_7b"
HF_NAME = "mistralai/Mistral-7B-v0.1"
D = 32
N_KV_HEADS = 8
HEAD_DIM = 128


def get_gpu_info():
    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


_TOKEN_CACHE = {}


def load_wikitext_tokens(tokenizer):
    key = id(tokenizer)
    if key not in _TOKEN_CACHE:
        from datasets import load_dataset

        ds = load_dataset("wikitext", DATASET, split="validation")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer.encode(text)
        arr = np.array(tokens[:N_TOKENS], dtype=np.int64)
        _TOKEN_CACHE[key] = arr
    return _TOKEN_CACHE[key]


def load_passage(tokenizer, L, seed=0):
    token_data = load_wikitext_tokens(tokenizer)
    seq_len = L + DECODE_TOKENS
    rng = np.random.RandomState(seed)
    start = rng.randint(0, max(1, len(token_data) - seq_len))
    batch = token_data[start : start + seq_len]
    return torch.from_numpy(batch).unsqueeze(0)


def quantize_int4_grouped(tensor, group_size=32):
    shape = tensor.shape
    hd = shape[-1]
    ng = (hd + group_size - 1) // group_size
    pd = ng * group_size
    if pd > hd:
        pad = torch.zeros(
            *shape[:-1],
            pd - hd,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=-1)
    r = tensor.reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = amax / 7.0
    q = (r / s).round().clamp(-8, 7)
    return (q * s).reshape(*shape[:-1], pd)[..., :hd]


def quantize_int8(tensor):
    amax = tensor.abs().amax().clamp(min=1e-8)
    s = amax / 127.0
    return ((tensor / s).round().clamp(-128, 127)) * s


def _cache_get_kv(past, li):
    if hasattr(past, "layers"):
        layer = past.layers[li]
        return layer.keys, layer.values
    return past[li]


def _cache_set_kv(past, li, k, v):
    if hasattr(past, "layers"):
        past.layers[li].keys = k
        past.layers[li].values = v
    else:
        past[li] = (k, v)


def cache_length(past):
    if hasattr(past, "layers"):
        return past.layers[0].keys.shape[2]
    return past[0][0].shape[2]


def n_cache_layers(past):
    if hasattr(past, "layers"):
        return len(past.layers)
    return len(past)


def apply_quantization(past, layer_bits):
    """Apply quantization to cache, return timing."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    clen = cache_length(past)
    far_end = clen - W_MIN
    if far_end > W_SINK:
        for li in range(n_cache_layers(past)):
            k, v = _cache_get_kv(past, li)
            k_s = k[:, :, :W_SINK, :]
            v_s = v[:, :, :W_SINK, :]
            k_f = k[:, :, W_SINK:far_end, :]
            v_f = v[:, :, W_SINK:far_end, :]
            k_n = k[:, :, far_end:, :]
            v_n = v[:, :, far_end:, :]
            if layer_bits[li] == 8:
                k_q = quantize_int8(k_f)
                v_q = quantize_int8(v_f)
            else:
                k_q = quantize_int4_grouped(k_f, GROUP_SIZE)
                v_q = quantize_int4_grouped(v_f, GROUP_SIZE)
            _cache_set_kv(
                past,
                li,
                torch.cat([k_s, k_q, k_n], dim=2),
                torch.cat([v_s, v_q, v_n], dim=2),
            )

    torch.cuda.synchronize()
    quant_ms = (time.perf_counter() - t0) * 1000
    return quant_ms


def benchmark_decode(model, tokenizer, L, layer_bits, label, warmup=5, n_iter=3):
    """Benchmark decode throughput with given quantization config."""
    device = next(model.parameters()).device
    results = []

    for iteration in range(n_iter):
        passage = load_passage(tokenizer, L, seed=0)
        input_ids = passage[:, :L].to(device)
        continuation = passage[:, L : L + DECODE_TOKENS].to(device)

        # Prefill
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            past = out.past_key_values
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_prefill) * 1000

        # Quantize cache
        quant_ms = 0.0
        if layer_bits is not None:
            quant_ms = apply_quantization(past, layer_bits)

        # Decode tokens
        token_latencies = []
        for t in range(DECODE_TOKENS):
            tok = continuation[:, t : t + 1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(tok, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) * 1000
            if t >= warmup:
                token_latencies.append(lat)
            past = out.past_key_values

        p50 = float(np.percentile(token_latencies, 50))
        p95 = float(np.percentile(token_latencies, 95))
        mean = float(np.mean(token_latencies))

        results.append(
            {
                "iteration": iteration,
                "prefill_ms": round(prefill_ms, 2),
                "quant_ms": round(quant_ms, 2),
                "decode_p50_ms": round(p50, 3),
                "decode_p95_ms": round(p95, 3),
                "decode_mean_ms": round(mean, 3),
                "n_tokens": len(token_latencies),
            }
        )
        del past
        torch.cuda.empty_cache()

    # Aggregate
    agg = {
        "label": label,
        "L": L,
        "prefill_ms": round(np.mean([r["prefill_ms"] for r in results]), 2),
        "quant_ms": round(np.mean([r["quant_ms"] for r in results]), 2),
        "decode_p50_ms": round(np.mean([r["decode_p50_ms"] for r in results]), 3),
        "decode_p95_ms": round(np.mean([r["decode_p95_ms"] for r in results]), 3),
        "decode_mean_ms": round(np.mean([r["decode_mean_ms"] for r in results]), 3),
        "tokens_per_sec": round(
            1000 / np.mean([r["decode_mean_ms"] for r in results]), 1
        ),
        "runs": results,
    }
    return agg


def main():
    gpu = get_gpu_info()
    print("=" * 60)
    print("BPA v28 Phase 2.1: Current Path Benchmark")
    print(f"GPU: {gpu['device_name']}")
    print(f"Model: {HF_NAME} (D={D})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("cuda").eval()
    print(f"  Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    all_benchmarks = {}

    for L in [8192, 32768]:
        print(f"\n--- L={L} ---")

        # Dense baseline (no quantization)
        print("  Dense fp16...")
        dense = benchmark_decode(model, tokenizer, L, None, "dense_fp16")
        print(
            f"    p50={dense['decode_p50_ms']:.3f}ms "
            f"({dense['tokens_per_sec']:.0f} tok/s)"
        )

        # All INT8
        print("  All INT8...")
        lb_int8 = [8] * D
        int8_all = benchmark_decode(model, tokenizer, L, lb_int8, "int8_all")
        print(
            f"    p50={int8_all['decode_p50_ms']:.3f}ms "
            f"quant={int8_all['quant_ms']:.1f}ms "
            f"({int8_all['tokens_per_sec']:.0f} tok/s)"
        )

        # All INT4
        print("  All INT4...")
        lb_int4 = [4] * D
        int4_all = benchmark_decode(model, tokenizer, L, lb_int4, "int4_all")
        print(
            f"    p50={int4_all['decode_p50_ms']:.3f}ms "
            f"quant={int4_all['quant_ms']:.1f}ms "
            f"({int4_all['tokens_per_sec']:.0f} tok/s)"
        )

        # Mixed (k*=2 with oracle)
        print("  Mixed k=2 (oracle)...")
        lb_mixed = [4] * D
        lb_mixed[0] = 8  # Layer 0 (sink)
        lb_mixed[15] = 8  # Layer 15 (second most sensitive for Mistral)
        mixed = benchmark_decode(model, tokenizer, L, lb_mixed, "mixed_k2_oracle")
        print(
            f"    p50={mixed['decode_p50_ms']:.3f}ms "
            f"quant={mixed['quant_ms']:.1f}ms "
            f"({mixed['tokens_per_sec']:.0f} tok/s)"
        )

        all_benchmarks[f"L{L}"] = {
            "dense_fp16": dense,
            "int8_all": int8_all,
            "int4_all": int4_all,
            "mixed_k2_oracle": mixed,
        }

    result = {
        "version": "v28",
        "phase": "2.1_benchmark_current_path",
        "model": MODEL_KEY,
        "hf_name": HF_NAME,
        "D": D,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "gpu_info": gpu,
        "benchmarks": all_benchmarks,
        "analysis": {
            "quant_overhead_ms_L8K": (
                all_benchmarks["L8192"]["int4_all"]["quant_ms"] - 0  # dense has 0 quant
            ),
            "quant_overhead_ms_L32K": (
                all_benchmarks["L32768"]["int4_all"]["quant_ms"] - 0
            ),
            "decode_overhead_pct_L8K": round(
                (
                    all_benchmarks["L8192"]["int4_all"]["decode_p50_ms"]
                    - all_benchmarks["L8192"]["dense_fp16"]["decode_p50_ms"]
                )
                / all_benchmarks["L8192"]["dense_fp16"]["decode_p50_ms"]
                * 100,
                2,
            ),
            "decode_overhead_pct_L32K": round(
                (
                    all_benchmarks["L32768"]["int4_all"]["decode_p50_ms"]
                    - all_benchmarks["L32768"]["dense_fp16"]["decode_p50_ms"]
                )
                / all_benchmarks["L32768"]["dense_fp16"]["decode_p50_ms"]
                * 100,
                2,
            ),
        },
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(RESULTS_ROOT, "benchmark_current_path.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for L_key in ["L8192", "L32768"]:
        b = all_benchmarks[L_key]
        print(f"\n  {L_key}:")
        for label in ["dense_fp16", "int8_all", "int4_all", "mixed_k2_oracle"]:
            d = b[label]
            print(
                f"    {label:20s}: p50={d['decode_p50_ms']:7.3f}ms "
                f"quant={d['quant_ms']:7.1f}ms "
                f"tok/s={d['tokens_per_sec']:7.0f}"
            )
    print("=" * 60)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
