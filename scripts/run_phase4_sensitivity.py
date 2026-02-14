#!/usr/bin/env python
"""
Phase 4: Layer sensitivity profiling.

Runs INT8 and INT4 on each layer individually (all others dense),
measuring PPL delta to identify sensitive vs tolerant layers.

Uses L=16384 with seeds 0,1,2 for each layer/bits combination.
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
from backends.layer_sensitivity import LayerSensitivityBackend
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
    L = 16384
    decode_steps = 256
    seeds = [0, 1, 2]
    outdir = "results/v15/phase4"

    os.makedirs(outdir, exist_ok=True)

    gpu_info = gpu_preflight(device)
    model, tokenizer, max_ctx, model_config = load_model(model_key, device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    n_layers = model_config["n_layers"]

    # Dense baseline at L=16384
    print(f"\n{'=' * 60}")
    print(f"Dense baseline at L={L}")
    print("=" * 60)

    dense_ppls = {}
    dense_results = []
    dense_be = DenseBackend()
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
        dense_ppls[(L, "r1", seed)] = r.ppl
        dense_results.append(r)
        print(f" PPL={r.ppl:.1f}")

    avg_dense_ppl = np.mean([r.ppl for r in dense_results])
    print(f"  Average dense PPL: {avg_dense_ppl:.2f}")

    # Layer sensitivity sweep
    all_results = list(dense_results)
    sensitivity = {}

    for bits in [8, 4]:
        print(f"\n{'=' * 60}")
        print(f"INT{bits} per-layer sweep")
        print("=" * 60)

        for li in range(n_layers):
            layer_ppls = []
            for seed in seeds:
                be = LayerSensitivityBackend(target_layer=li, target_bits=bits)
                be.configure(L, model_config)
                print(
                    f"  INT{bits} layer={li:2d} seed={seed}...",
                    end="",
                    flush=True,
                )
                r = run_single_eval(
                    be,
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
                layer_ppls.append(r.ppl)
                print(f" PPL={r.ppl:.1f}")

            avg_ppl = np.mean(layer_ppls)
            delta_pct = 100 * (avg_ppl - avg_dense_ppl) / avg_dense_ppl
            sensitivity[(bits, li)] = {
                "layer": li,
                "bits": bits,
                "avg_ppl": round(avg_ppl, 3),
                "ppl_delta_pct": round(delta_pct, 3),
                "ppls": [round(p, 3) for p in layer_ppls],
            }
            print(
                f"  -> layer {li:2d} INT{bits}:"
                f" avg_ppl={avg_ppl:.2f}"
                f" delta={delta_pct:+.2f}%"
            )

    # Quality gating
    apply_quality_gating(all_results, dense_ppls)

    # Build sensitivity profile
    profile = {
        "model": model_key,
        "L": L,
        "decode_steps": decode_steps,
        "seeds": seeds,
        "n_layers": n_layers,
        "avg_dense_ppl": round(avg_dense_ppl, 3),
        "int8_sensitivity": [],
        "int4_sensitivity": [],
    }

    for bits in [8, 4]:
        key = f"int{bits}_sensitivity"
        for li in range(n_layers):
            entry = sensitivity[(bits, li)]
            profile[key].append(entry)

    # Rank layers by sensitivity (INT4, since INT8 is too easy)
    int4_ranked = sorted(
        profile["int4_sensitivity"],
        key=lambda x: x["ppl_delta_pct"],
    )

    profile["int4_ranked_tolerant_to_sensitive"] = [
        {"layer": e["layer"], "ppl_delta_pct": e["ppl_delta_pct"]} for e in int4_ranked
    ]

    # Summary
    print(f"\n{'=' * 60}")
    print("Layer sensitivity summary (INT4, ranked tolerant->sensitive)")
    print("=" * 60)
    for entry in int4_ranked:
        marker = ""
        if entry["ppl_delta_pct"] > 3.0:
            marker = " SENSITIVE"
        elif entry["ppl_delta_pct"] < 1.0:
            marker = " TOLERANT"
        print(
            f"  layer {entry['layer']:2d}:"
            f" delta={entry['ppl_delta_pct']:+.2f}%{marker}"
        )

    # Count tolerant layers for adaptive schedule
    n_tolerant_int4 = sum(1 for e in int4_ranked if e["ppl_delta_pct"] < 3.0)
    n_tolerant_int8 = sum(
        1 for e in profile["int8_sensitivity"] if e["ppl_delta_pct"] < 1.0
    )
    print(f"\n  Tolerant layers (INT4 <3%): {n_tolerant_int4}/{n_layers}")
    print(f"  Tolerant layers (INT8 <1%): {n_tolerant_int8}/{n_layers}")

    # Save
    with open(os.path.join(outdir, "layer_sensitivity.json"), "w") as f:
        json.dump(profile, f, indent=2)

    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": model_key,
        "L": L,
        "n_layers": n_layers,
        "decode_steps": decode_steps,
        "seeds": seeds,
        "gpu_info": gpu_info,
        "version": "v15",
        "phase": 4,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {outdir}/")
    print(f"Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
