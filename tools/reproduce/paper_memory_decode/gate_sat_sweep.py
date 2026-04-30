#!/usr/bin/env python3
"""Saturation sweep gate: batch × context throughput grid for Qwen2.5-7B.

Runs the model through vLLM with enforce_eager=True across a grid of
batch sizes and context lengths, measuring tokens/second for each cell.
Results are written to KNLP_RESULT_PATH as JSON.

Metrics printed for parent stage (grep-able):
  SAT_SWEEP_CELLS=<int>     total (batch, ctx) cells measured
  SAT_SWEEP_FAILED=<int>    cells that failed or OOMed

Exits 0 on pass (at least 80% of cells measured), 1 on failure.

Expected runtime: 60-120 min on a single H100 SXM.
"""
from __future__ import annotations

import json
import os
import sys
import time

RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/sat_sweep_results.json")
MODEL_ID = os.environ.get("KNLP_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# Grid: batch sizes × context lengths (tokens).
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
CTX_LENGTHS = [2048, 4096, 8192, 16384, 32768]

# Measurement parameters.
WARMUP_REQUESTS = 3  # discarded
MEASURE_REQUESTS = 10  # averaged

# Minimum fraction of cells that must succeed.
MIN_CELL_FRACTION = 0.80


def _measure_cell(llm, tokenizer, batch: int, ctx: int) -> float | None:
    """Return tok/s for one (batch, ctx) cell, or None on OOM/error."""
    import torch

    prompt_ids = [1] * (ctx - 64)  # leave room for 64 output tokens
    if len(prompt_ids) < 1:
        prompt_ids = [1]
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    prompts = [prompt] * batch

    from vllm import SamplingParams

    params = SamplingParams(max_tokens=64, temperature=0.0)

    # Warmup (discarded).
    for _ in range(WARMUP_REQUESTS):
        try:
            llm.generate(prompts, params)
        except Exception:
            return None

    # Measurement.
    t0 = time.perf_counter()
    total_out = 0
    for _ in range(MEASURE_REQUESTS):
        try:
            outs = llm.generate(prompts, params)
        except Exception:
            return None
        for o in outs:
            total_out += sum(len(x.token_ids) for x in o.outputs)
    elapsed = time.perf_counter() - t0
    if elapsed <= 0 or total_out == 0:
        return None
    return total_out / elapsed


def main() -> int:
    try:
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"SKIP: required package not available: {e}", flush=True)
        return 2

    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU available", flush=True)
        return 2

    print(f"Loading {MODEL_ID} for saturation sweep ...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=max(CTX_LENGTHS),
            gpu_memory_utilization=0.90,
        )
    except Exception as e:
        print(f"ERROR: model load failed: {e}", flush=True)
        return 1

    results: list[dict] = []
    n_cells = len(BATCH_SIZES) * len(CTX_LENGTHS)
    n_failed = 0

    for ctx in CTX_LENGTHS:
        for batch in BATCH_SIZES:
            cell_id = f"B{batch}_T{ctx}"
            print(f"  measuring {cell_id} ...", flush=True)
            try:
                tps = _measure_cell(llm, tokenizer, batch, ctx)
            except Exception as e:
                print(f"  {cell_id}: exception {e}", flush=True)
                tps = None

            if tps is None:
                n_failed += 1
                print(f"  {cell_id}: FAILED/OOM", flush=True)
            else:
                print(f"  {cell_id}: {tps:.1f} tok/s", flush=True)

            results.append(
                {
                    "batch": batch,
                    "ctx": ctx,
                    "tps": tps,
                    "failed": tps is None,
                }
            )

    n_ok = n_cells - n_failed
    print(f"\nSAT_SWEEP_CELLS={n_cells}", flush=True)
    print(f"SAT_SWEEP_FAILED={n_failed}", flush=True)
    print(f"SAT_SWEEP_OK={n_ok}", flush=True)

    payload = {
        "model": MODEL_ID,
        "batch_sizes": BATCH_SIZES,
        "ctx_lengths": CTX_LENGTHS,
        "cells": results,
        "n_cells": n_cells,
        "n_ok": n_ok,
        "n_failed": n_failed,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {RESULT_PATH}", flush=True)

    if n_ok < MIN_CELL_FRACTION * n_cells:
        print(
            f"GATE FAILED: only {n_ok}/{n_cells} cells succeeded "
            f"(need ≥{MIN_CELL_FRACTION*100:.0f}%)",
            flush=True,
        )
        return 1

    print("SAT SWEEP GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
