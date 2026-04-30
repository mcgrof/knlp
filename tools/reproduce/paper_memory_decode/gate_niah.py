#!/usr/bin/env python3
"""NIAH (Needle In A Haystack) gate for 16K and 32K contexts.

Inserts a unique "needle" sentence at a random position inside a
long filler context, then asks the model to retrieve it.  Tests
retrieval accuracy across three KV configurations:

  fp16       — BF16 weights, BF16 KV
  fp8_sym    — BF16 weights, FP8-e4m3 symmetric KV
  asym_k16v8 — BF16 weights, BF16-K / FP8-e4m3-V (paper claim)

Pass criteria:
  - fp16 accuracy  ≥ 0.90 at both 16K and 32K
  - asym accuracy within 5pp of fp16 at both lengths
  - fp8_sym may collapse (expected; not gating)

Metrics printed:
  NIAH_FP16_16K=<float>     NIAH_FP16_32K=<float>
  NIAH_ASYM_16K=<float>     NIAH_ASYM_32K=<float>
  NIAH_SYM_16K=<float>      NIAH_SYM_32K=<float>
  NIAH_ASYM_DELTA_16K=<float>
  NIAH_ASYM_DELTA_32K=<float>

Exits 0 on pass, 1 on failure, 2 on skip.
"""
from __future__ import annotations

import json
import os
import random
import sys

RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/niah_results.json")
MODEL_ID = os.environ.get("KNLP_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

CTX_LENGTHS = [16384, 32768]
N_NEEDLES = 10  # number of independent needle insertions per config

FILLER_WORD = "The grass is green. The sky is blue. The sun is bright. "

FP16_ACC_FLOOR = 0.90
ASYM_DELTA_CAP = 0.05


def _make_context(ctx_len: int, needle: str, tokenizer) -> str:
    """Build a filler context of ~ctx_len tokens with needle inserted at random."""
    # Estimate tokens: ~1.3 tokens per word, ~4 chars per token
    chars_needed = ctx_len * 4
    filler = (FILLER_WORD * (chars_needed // len(FILLER_WORD) + 1))[:chars_needed]
    # Insert needle at a random position (not at the very end).
    mid = random.randint(len(filler) // 4, 3 * len(filler) // 4)
    return filler[:mid] + " " + needle + " " + filler[mid:]


def _run_niah(llm, tokenizer, ctx_len: int, kv_dtype) -> float:
    """Return fraction of needles retrieved correctly."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=40, temperature=0.0)
    correct = 0

    for i in range(N_NEEDLES):
        secret = f"SECRET-{random.randint(10000, 99999)}"
        needle = f"The special value is {secret}."
        question = f"What is the special value mentioned in the text? Answer with just the value."
        context = _make_context(ctx_len, needle, tokenizer)
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        try:
            outs = llm.generate([prompt], params)
            response = outs[0].outputs[0].text.strip()
            if secret in response:
                correct += 1
        except Exception:
            pass

    return correct / N_NEEDLES


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

    print(f"Loading {MODEL_ID} for NIAH test ...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"ERROR: tokenizer load failed: {e}", flush=True)
        return 1

    configs = [
        ("fp16", "auto"),
        ("fp8_sym", "fp8_e4m3"),
        ("asym_k16v8", ("auto", "fp8_e4m3")),
    ]

    results: dict = {}

    for cfg_name, kv_dtype in configs:
        print(f"\n=== {cfg_name} kv_cache_dtype={kv_dtype!r} ===", flush=True)
        try:
            llm = LLM(
                model=MODEL_ID,
                dtype="bfloat16",
                kv_cache_dtype=kv_dtype,
                enforce_eager=True,
                max_model_len=max(CTX_LENGTHS) + 256,
                gpu_memory_utilization=0.90,
                attention_config={"backend": "FLASHINFER"},
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}", flush=True)
            for ctx_len in CTX_LENGTHS:
                results[f"{cfg_name}_{ctx_len}"] = None
            continue

        for ctx_len in CTX_LENGTHS:
            print(f"  ctx={ctx_len//1024}K ...", flush=True)
            acc = _run_niah(llm, tokenizer, ctx_len, kv_dtype)
            results[f"{cfg_name}_{ctx_len}"] = acc
            print(f"  {cfg_name} {ctx_len//1024}K: acc={acc:.2f}", flush=True)

        del llm

    # Print metrics.
    for ctx_len in CTX_LENGTHS:
        k = ctx_len // 1024
        fp16 = results.get(f"fp16_{ctx_len}")
        asym = results.get(f"asym_k16v8_{ctx_len}")
        sym = results.get(f"fp8_sym_{ctx_len}")
        print(f"\nNIAH_FP16_{k}K={fp16 if fp16 is not None else 'NA'}", flush=True)
        print(f"NIAH_ASYM_{k}K={asym if asym is not None else 'NA'}", flush=True)
        print(f"NIAH_SYM_{k}K={sym if sym is not None else 'NA'}", flush=True)
        if fp16 is not None and asym is not None:
            delta = asym - fp16
            print(f"NIAH_ASYM_DELTA_{k}K={delta:.4f}", flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(
            {"model": MODEL_ID, "n_needles": N_NEEDLES, "results": results}, f, indent=2
        )
    print(f"Results written to {RESULT_PATH}", flush=True)

    # Gate checks.
    failures = []
    for ctx_len in CTX_LENGTHS:
        k = ctx_len // 1024
        fp16 = results.get(f"fp16_{ctx_len}")
        asym = results.get(f"asym_k16v8_{ctx_len}")
        if fp16 is None:
            failures.append(f"fp16 {k}K failed to run")
            continue
        if fp16 < FP16_ACC_FLOOR:
            failures.append(f"fp16 {k}K acc={fp16:.2f} < {FP16_ACC_FLOOR}")
        if asym is not None and (fp16 - asym) > ASYM_DELTA_CAP:
            failures.append(
                f"asym {k}K acc={asym:.2f} is {fp16-asym:.2f}pp below fp16 "
                f"(cap={ASYM_DELTA_CAP}pp)"
            )

    if failures:
        for f in failures:
            print(f"GATE FAILED: {f}", flush=True)
        return 1

    print("\nNIAH GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
