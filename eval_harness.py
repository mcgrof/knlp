#!/usr/bin/env python
"""
BPA v13 Unified Evaluation Harness.

Single entrypoint for evaluating all bitter methods (0-7) plus
dense baseline. Produces JSON logs, quality gating (PASS/FAIL),
and scoreboard data.

Usage:
    python eval_harness.py \
        --model qwen05b \
        --L 1024 2048 4096 \
        --regimes r1 r2 \
        --methods dense bitter0 bitter1 \
        --tol 0.01 0.03 \
        --decode_steps 256 \
        --seeds 0 1 2 \
        --outdir results/v13_run
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

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
    kv_bytes_per_token,
    load_validation_tokens,
    reset_gpu_mem,
)
from methods.base import DenseMethod, StepStats

# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY = {
    "qwen05b": "Qwen/Qwen2.5-0.5B",
    "qwen15b": "Qwen/Qwen2.5-1.5B",
}

# ============================================================
# Method registry
# ============================================================


def get_method(name):
    """Instantiate a bitter method by name."""
    if name == "dense":
        return DenseMethod()
    elif name == "bitter0":
        from methods.bitter0 import Bitter0

        return Bitter0()
    elif name == "bitter1":
        from methods.bitter1 import Bitter1

        return Bitter1()
    elif name == "bitter2":
        from methods.bitter2 import Bitter2

        return Bitter2()
    elif name == "bitter3":
        from methods.bitter3 import Bitter3

        return Bitter3()
    elif name == "bitter4":
        from methods.bitter4 import Bitter4

        return Bitter4()
    elif name == "bitter5":
        from methods.bitter5 import Bitter5

        return Bitter5()
    elif name == "bitter6":
        from methods.bitter6 import Bitter6

        return Bitter6()
    elif name == "bitter7":
        from methods.bitter7 import Bitter7

        return Bitter7()
    else:
        raise ValueError(f"Unknown method: {name}")


# ============================================================
# GPU preflight
# ============================================================


def gpu_preflight(device_str):
    """Verify GPU availability and log device info."""
    assert torch.version.hip is not None, "ROCm/HIP not available"
    assert torch.cuda.is_available(), "CUDA not available"

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    print(f"GPU Preflight OK: {props.name} ({total_gb:.1f}GB)")
    print(f"  torch={torch.__version__} hip={torch.version.hip}")
    return {
        "device_name": props.name,
        "total_gb": round(total_gb, 1),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }


# ============================================================
# Model loading
# ============================================================


def load_model(model_key, device_str):
    """Load HF model and return model, tokenizer, config info."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_name = MODEL_REGISTRY.get(model_key, model_key)
    print(f"Loading model {model_name}...")

    config = AutoConfig.from_pretrained(model_name)
    max_ctx = getattr(config, "max_position_embeddings", None) or getattr(
        config, "n_positions", 1024
    )
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads if hidden and n_heads else None

    model_config = {
        "n_layers": n_layers,
        "hidden_size": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", 50257),
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    model = model.to(device_str)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M max_ctx={max_ctx}")
    print(
        f"  layers={n_layers} hidden={hidden} " f"heads={n_heads} kv_heads={n_kv_heads}"
    )

    return model, tokenizer, max_ctx, model_config


# ============================================================
# Single evaluation run
# ============================================================


@dataclass
class EvalResult:
    """Result from a single (method, L, regime, seed) evaluation."""

    method: str
    L: int
    regime: str
    batch_size: int
    seed: int
    decode_steps: int
    ppl: float
    ppl_dense: float = 0.0
    ppl_delta_pct: float = 0.0
    passed_1pct: bool = False
    passed_3pct: bool = False
    catastrophic: bool = False
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    tok_s: float = 0.0
    kv_bytes_proxy_per_tok: float = 0.0
    peak_gpu_alloc_mb: float = 0.0
    peak_gpu_reserved_mb: float = 0.0
    peak_cpu_rss_mb: float = 0.0
    gate_overhead_pct: float = 0.0
    tier_full: int = 0
    tier_mla: int = 0
    tier_splice: int = 0
    tier_dropped: int = 0
    kept_tokens: float = 0.0
    n_thresholds: int = 0
    learned_fraction: float = 0.0
    error: str = ""


def run_single_eval(
    method_obj,
    model,
    token_data,
    L,
    decode_steps,
    seed,
    device_str,
    max_ctx,
    model_config,
    batch_size=1,
    regime="r1",
):
    """Run a single evaluation of one method at one (L, seed, regime)."""
    rng = np.random.RandomState(seed)
    total_len = L + decode_steps

    # Get input data
    idx = get_text_batch(token_data, batch_size, total_len, rng).to(device_str)
    prefix = idx[:, :L]
    continuation = idx[:, L : L + decode_steps]

    rss_before = get_cpu_rss_mb()

    # Warmup
    if device_str != "cpu":
        with torch.no_grad():
            _ = model(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    # Configure method
    method_obj.configure(L, model_config)

    # Time the decode
    gpu_sync(device_str)
    t_start = time.perf_counter()

    try:
        all_logits, step_stats = method_obj.run_decode(
            model, prefix, continuation, device_str, max_ctx
        )
    except Exception as e:
        torch.cuda.empty_cache()
        return EvalResult(
            method=method_obj.name,
            L=L,
            regime=regime,
            batch_size=batch_size,
            seed=seed,
            decode_steps=decode_steps,
            ppl=float("inf"),
            error=str(e),
            catastrophic=True,
        )

    gpu_sync(device_str)
    total_s = time.perf_counter() - t_start

    # Compute PPL
    ppl = compute_ppl(all_logits[:, :-1, :], continuation)

    # Compute timing from step stats
    # We don't have per-step times in step_stats; use total time
    decode_ms = total_s * 1000
    p50_ms = decode_ms / decode_steps
    p95_ms = p50_ms * 1.5  # rough estimate

    # Gate overhead
    total_gate_ms = sum(s.gate_ms for s in step_stats)
    gate_pct = 100 * total_gate_ms / decode_ms if decode_ms > 0 else 0

    # KV bytes proxy
    if step_stats:
        avg_kv_bytes = np.mean([s.kv_bytes_proxy for s in step_stats])
        last_stats = step_stats[-1]
    else:
        avg_kv_bytes = 0
        last_stats = StepStats()

    # GPU/CPU memory
    rss_after = get_cpu_rss_mb()
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    # Tier counts from last step
    kept = last_stats.kv_kept if step_stats else L + decode_steps

    result = EvalResult(
        method=method_obj.name,
        L=L,
        regime=regime,
        batch_size=batch_size,
        seed=seed,
        decode_steps=decode_steps,
        ppl=float(ppl),
        p50_ms=round(p50_ms, 3),
        p95_ms=round(p95_ms, 3),
        tok_s=round(decode_steps / total_s, 1) if total_s > 0 else 0,
        kv_bytes_proxy_per_tok=round(avg_kv_bytes / max(kept, 1), 1),
        peak_gpu_alloc_mb=round(gpu_alloc, 0),
        peak_gpu_reserved_mb=round(gpu_reserved, 0),
        peak_cpu_rss_mb=round(max(rss_before, rss_after), 0),
        gate_overhead_pct=round(gate_pct, 2),
        tier_full=last_stats.tier_full,
        tier_mla=last_stats.tier_mla,
        tier_splice=last_stats.tier_splice,
        tier_dropped=last_stats.tier_dropped,
        kept_tokens=float(kept),
        n_thresholds=method_obj.n_thresholds(),
        learned_fraction=method_obj.learned_fraction(),
        catastrophic=(ppl > 1e5 or math.isnan(ppl) or math.isinf(ppl)),
    )

    # Cleanup
    del all_logits
    torch.cuda.empty_cache()

    return result


# ============================================================
# Quality gating
# ============================================================


def apply_quality_gating(results, dense_ppls):
    """Apply PASS/FAIL based on dense PPL thresholds.

    Args:
        results: list of EvalResult
        dense_ppls: dict of {(L, regime, seed): ppl}
    """
    for r in results:
        key = (r.L, r.regime, r.seed)
        if key in dense_ppls:
            r.ppl_dense = dense_ppls[key]
            if r.ppl_dense > 0:
                r.ppl_delta_pct = round(100 * (r.ppl - r.ppl_dense) / r.ppl_dense, 2)
                r.passed_1pct = r.ppl <= r.ppl_dense * 1.01
                r.passed_3pct = r.ppl <= r.ppl_dense * 1.03
            if r.ppl > r.ppl_dense * 10:
                r.catastrophic = True


# ============================================================
# Scoreboard
# ============================================================


def build_scoreboard(results):
    """Build PASS/FAIL matrix from results.

    Returns dict of {method: {L: {regime: {tol: PASS/FAIL}}}}
    """
    scoreboard = {}
    # Group by (method, L, regime)
    groups = {}
    for r in results:
        key = (r.method, r.L, r.regime)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (method, L, regime), group in groups.items():
        if method not in scoreboard:
            scoreboard[method] = {}
        if L not in scoreboard[method]:
            scoreboard[method][L] = {}
        if regime not in scoreboard[method][L]:
            scoreboard[method][L][regime] = {}

        # PASS only if ALL seeds pass and none catastrophic
        any_catastrophic = any(r.catastrophic for r in group)
        all_pass_1 = all(r.passed_1pct for r in group) and not any_catastrophic
        all_pass_3 = all(r.passed_3pct for r in group) and not any_catastrophic

        avg_ppl = np.mean([r.ppl for r in group if not r.catastrophic])
        avg_p50 = np.mean([r.p50_ms for r in group])
        avg_kept = np.mean([r.kept_tokens for r in group])

        scoreboard[method][L][regime] = {
            "pass_1pct": all_pass_1,
            "pass_3pct": all_pass_3,
            "any_catastrophic": any_catastrophic,
            "avg_ppl": round(float(avg_ppl), 2) if not np.isnan(avg_ppl) else 999999,
            "avg_p50_ms": round(float(avg_p50), 3),
            "avg_kept": round(float(avg_kept), 0),
            "n_seeds": len(group),
        }

    return scoreboard


# ============================================================
# Scaling fits
# ============================================================


def compute_scaling_fits(results):
    """Compute beta (kept~L^beta), gamma (kv_bytes~L^gamma),
    delta (latency~L^delta) for methods that PASS."""
    fits = {}

    # Group passing results by method
    method_data = {}
    for r in results:
        if r.method == "dense" or r.catastrophic:
            continue
        if not r.passed_3pct:
            continue
        if r.method not in method_data:
            method_data[r.method] = {}
        if r.L not in method_data[r.method]:
            method_data[r.method][r.L] = []
        method_data[r.method][r.L].append(r)

    for method, by_L in method_data.items():
        Ls = sorted(by_L.keys())
        if len(Ls) < 2:
            continue

        log_L = np.array([math.log(L) for L in Ls])

        # Beta: kept tokens
        log_kept = np.array(
            [math.log(max(1, np.mean([r.kept_tokens for r in by_L[L]]))) for L in Ls]
        )
        A = np.vstack([log_L, np.ones(len(log_L))]).T
        beta_result = np.linalg.lstsq(A, log_kept, rcond=None)
        beta = float(beta_result[0][0])

        # Gamma: kv_bytes_proxy
        log_bytes = np.array(
            [
                math.log(max(1, np.mean([r.kv_bytes_proxy_per_tok for r in by_L[L]])))
                for L in Ls
            ]
        )
        gamma_result = np.linalg.lstsq(A, log_bytes, rcond=None)
        gamma = float(gamma_result[0][0])

        # Delta: latency
        log_lat = np.array(
            [math.log(max(0.001, np.mean([r.p50_ms for r in by_L[L]]))) for L in Ls]
        )
        delta_result = np.linalg.lstsq(A, log_lat, rcond=None)
        delta = float(delta_result[0][0])

        fits[method] = {
            "beta": round(beta, 4),
            "gamma": round(gamma, 4),
            "delta": round(delta, 4),
            "n_L_points": len(Ls),
            "L_values": Ls,
        }

    return fits


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v13 Eval Harness")
    parser.add_argument("--model", default="qwen05b", help="Model key or HF name")
    parser.add_argument(
        "--L", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument(
        "--regimes", nargs="+", default=["r1"], help="r1=batch1, r2=batch4"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "dense",
            "bitter0",
            "bitter1",
            "bitter2",
            "bitter3",
            "bitter4",
            "bitter5",
            "bitter6",
            "bitter7",
        ],
    )
    parser.add_argument("--tol", nargs="+", type=float, default=[0.01, 0.03])
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--outdir", default=None, help="Output directory (auto-generated if not set)"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Output directory
    if args.outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = f"results/v13_run_{ts}"
    os.makedirs(args.outdir, exist_ok=True)

    # GPU preflight
    gpu_info = gpu_preflight(args.device)

    # Load model
    model, tokenizer, max_ctx, model_config = load_model(args.model, args.device)

    # Load validation data
    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    # Regime batch sizes
    regime_bs = {"r1": 1, "r2": 4}

    # Filter L to in-range only
    valid_L = [L for L in args.L if L <= max_ctx]
    if len(valid_L) < len(args.L):
        skipped = [L for L in args.L if L > max_ctx]
        print(f"Skipping out-of-range L values: {skipped}")

    # ---- Phase A: Dense baselines ----
    print("\n" + "=" * 60)
    print("Phase A: Dense baselines")
    print("=" * 60)

    all_results = []
    dense_ppls = {}

    for regime in args.regimes:
        bs = regime_bs.get(regime, 1)
        for L in valid_L:
            for seed in args.seeds:
                print(
                    f"  dense L={L} {regime}(bs={bs}) seed={seed}...",
                    end="",
                    flush=True,
                )
                method = DenseMethod()
                r = run_single_eval(
                    method,
                    model,
                    token_data,
                    L,
                    args.decode_steps,
                    seed,
                    args.device,
                    max_ctx,
                    model_config,
                    batch_size=bs,
                    regime=regime,
                )
                r.ppl_dense = r.ppl
                r.ppl_delta_pct = 0.0
                r.passed_1pct = True
                r.passed_3pct = True
                all_results.append(r)
                dense_ppls[(L, regime, seed)] = r.ppl
                print(
                    f" PPL={r.ppl:.1f} p50={r.p50_ms:.2f}ms"
                    f" gpu={r.peak_gpu_alloc_mb:.0f}MB"
                )

    # ---- Phase B: Bitter methods ----
    methods_to_run = [m for m in args.methods if m != "dense"]
    if methods_to_run:
        print("\n" + "=" * 60)
        print(f"Phase B: Methods {methods_to_run}")
        print("=" * 60)

    for method_name in methods_to_run:
        print(f"\n--- {method_name} ---")
        method_obj = get_method(method_name)

        for regime in args.regimes:
            bs = regime_bs.get(regime, 1)
            for L in valid_L:
                for seed in args.seeds:
                    print(
                        f"  {method_name} L={L} {regime}(bs={bs})" f" seed={seed}...",
                        end="",
                        flush=True,
                    )
                    try:
                        r = run_single_eval(
                            method_obj,
                            model,
                            token_data,
                            L,
                            args.decode_steps,
                            seed,
                            args.device,
                            max_ctx,
                            model_config,
                            batch_size=bs,
                            regime=regime,
                        )
                    except Exception as e:
                        print(f" ERROR: {e}")
                        r = EvalResult(
                            method=method_name,
                            L=L,
                            regime=regime,
                            batch_size=bs,
                            seed=seed,
                            decode_steps=args.decode_steps,
                            ppl=float("inf"),
                            error=str(e),
                            catastrophic=True,
                        )
                    all_results.append(r)

                    status = ""
                    if r.error:
                        status = f" ERROR: {r.error[:50]}"
                    elif r.catastrophic:
                        status = " CATASTROPHIC"
                    else:
                        status = (
                            f" PPL={r.ppl:.1f} kept={r.kept_tokens:.0f}"
                            f" p50={r.p50_ms:.2f}ms"
                        )
                    print(status)

                # Re-instantiate method for each L to reset state
                method_obj = get_method(method_name)

    # ---- Quality gating ----
    print("\n" + "=" * 60)
    print("Quality gating")
    print("=" * 60)
    apply_quality_gating(all_results, dense_ppls)

    # Count passes
    for method_name in args.methods:
        method_results = [r for r in all_results if r.method == method_name]
        n_total = len(method_results)
        n_pass1 = sum(1 for r in method_results if r.passed_1pct)
        n_pass3 = sum(1 for r in method_results if r.passed_3pct)
        n_cat = sum(1 for r in method_results if r.catastrophic)
        print(
            f"  {method_name:10s}: {n_pass1}/{n_total} @1%"
            f"  {n_pass3}/{n_total} @3%"
            f"  {n_cat} catastrophic"
        )

    # ---- Scoreboard ----
    scoreboard = build_scoreboard(all_results)

    # ---- Scaling fits ----
    fits = compute_scaling_fits(all_results)
    if fits:
        print("\nScaling fits (PASS@3% points only):")
        for method, f in fits.items():
            print(
                f"  {method}: beta={f['beta']:.3f}"
                f" gamma={f['gamma']:.3f}"
                f" delta={f['delta']:.3f}"
                f" (n={f['n_L_points']})"
            )

    # ---- Bitter lesson alignment ----
    alignment = {}
    for method_name in args.methods:
        method_obj = get_method(method_name)
        alignment[method_name] = {
            "n_thresholds": method_obj.n_thresholds(),
            "learned_fraction": method_obj.learned_fraction(),
        }

    # ---- Save results ----
    print(f"\nSaving results to {args.outdir}/")

    # All results as JSON
    results_data = [asdict(r) for r in all_results]
    with open(os.path.join(args.outdir, "all_results.json"), "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    # Scoreboard
    with open(os.path.join(args.outdir, "scoreboard.json"), "w") as f:
        json.dump(scoreboard, f, indent=2)

    # Scaling fits
    with open(os.path.join(args.outdir, "scaling_fits.json"), "w") as f:
        json.dump(fits, f, indent=2)

    # Alignment
    with open(os.path.join(args.outdir, "bitter_alignment.json"), "w") as f:
        json.dump(alignment, f, indent=2)

    # Run metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "L_values": valid_L,
        "regimes": args.regimes,
        "methods": args.methods,
        "decode_steps": args.decode_steps,
        "seeds": args.seeds,
        "gpu_info": gpu_info,
        "n_results": len(all_results),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
