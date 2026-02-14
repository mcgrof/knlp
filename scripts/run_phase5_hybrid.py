#!/usr/bin/env python
"""
Phase 5: Hybrid tier schedule evaluation.

Evaluates the hybrid_tier backend that combines:
- Full precision for sink + near tokens
- INT8 for mid context
- rope_complex for far context

Compares against dense and INT8-only baselines.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import asdict
from datetime import datetime

from backends.base import DenseBackend
from backends.hybrid_tier import HybridTierBackend
from backends.quant import QuantBackend
from backends.rope_aware_kv import RoPEAwareKVBackend
from eval_v15 import (
    V15Result,
    apply_quality_gating,
    gpu_preflight,
    load_model,
    run_single_eval,
)
from scripts.bpa_v11_bench import load_validation_tokens


def main():
    device = "cuda"
    model_key = "qwen05b"
    L_values = [4096, 8192, 16384, 32768]
    decode_steps = 256
    seeds = [0, 1, 2]
    outdir = "results/v15/phase5"

    os.makedirs(outdir, exist_ok=True)

    gpu_info = gpu_preflight(device)
    model, tokenizer, max_ctx, model_config = load_model(model_key, device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    valid_L = [L for L in L_values if L <= max_ctx]
    all_results = []
    dense_ppls = {}

    # Dense baselines
    print(f"\n{'=' * 60}")
    print("Dense baselines")
    print("=" * 60)
    dense_be = DenseBackend()
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
            all_results.append(r)
            dense_ppls[(L, "r1", seed)] = r.ppl
            print(f" PPL={r.ppl:.1f}")

    # INT8-only baseline
    print(f"\n{'=' * 60}")
    print("INT8-only baseline")
    print("=" * 60)
    int8_be = QuantBackend()
    int8_be.configure(max(valid_L), model_config)
    int8_be.calibrate(model, token_data, max(valid_L), device, model_config)
    for L in valid_L:
        for seed in seeds:
            print(f"  quant L={L} seed={seed}...", end="", flush=True)
            int8_be.configure(L, model_config)
            r = run_single_eval(
                int8_be,
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device,
                max_ctx,
                model_config,
            )
            all_results.append(r)
            print(f" PPL={r.ppl:.1f}")

    # rope_complex standalone
    print(f"\n{'=' * 60}")
    print("rope_complex standalone")
    print("=" * 60)
    rope_be = RoPEAwareKVBackend(mode="complex", rank_frac=0.5)
    rope_be.configure(max(valid_L), model_config)
    rope_be.calibrate(model, token_data, max(valid_L), device, model_config)
    for L in valid_L:
        for seed in seeds:
            print(f"  rope_complex L={L} seed={seed}...", end="", flush=True)
            rope_be.configure(L, model_config)
            r = run_single_eval(
                rope_be,
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device,
                max_ctx,
                model_config,
            )
            all_results.append(r)
            print(f" PPL={r.ppl:.1f}")

    # Hybrid tier
    print(f"\n{'=' * 60}")
    print("Hybrid tier")
    print("=" * 60)
    hybrid_be = HybridTierBackend(mid_frac=0.5, rank_frac=0.5)
    hybrid_be.configure(max(valid_L), model_config)
    hybrid_be.calibrate(model, token_data, max(valid_L), device, model_config)
    for L in valid_L:
        for seed in seeds:
            print(f"  hybrid_tier L={L} seed={seed}...", end="", flush=True)
            hybrid_be.configure(L, model_config)
            r = run_single_eval(
                hybrid_be,
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device,
                max_ctx,
                model_config,
            )
            all_results.append(r)
            print(f" PPL={r.ppl:.1f}")

    # Quality gating
    print(f"\n{'=' * 60}")
    print("Quality gating")
    print("=" * 60)
    apply_quality_gating(all_results, dense_ppls)

    for bname in ["dense", "quant", "rope_complex", "hybrid_tier"]:
        br = [r for r in all_results if r.backend == bname]
        if not br:
            continue
        n = len(br)
        n1 = sum(1 for r in br if r.passed_1pct)
        n3 = sum(1 for r in br if r.passed_3pct)
        nc = sum(1 for r in br if r.catastrophic)
        avg_ratio = np.mean([r.kv_bytes_ratio for r in br if r.kv_bytes_ratio < 1.5])
        print(
            f"  {bname:25s}: {n1}/{n} @1%  {n3}/{n} @3%"
            f"  {nc} catastrophic"
            f"  avg_kv_ratio={avg_ratio:.3f}"
        )

    # Save
    print(f"\nSaving to {outdir}/")
    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    from eval_v15 import build_scoreboard

    scoreboard = build_scoreboard(all_results)
    with open(os.path.join(outdir, "scoreboard.json"), "w") as f:
        json.dump(scoreboard, f, indent=2, default=str)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": model_key,
        "L_values": valid_L,
        "decode_steps": decode_steps,
        "seeds": seeds,
        "gpu_info": gpu_info,
        "version": "v15",
        "phase": 5,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
