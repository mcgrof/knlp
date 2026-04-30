#!/usr/bin/env python3
"""Speculative decoding interaction gate.

Measures how n-gram speculation (k=5) interacts with KV quantization
across three 7B-class models and three batch sizes, reproducing the
composition analysis from the paper (Section: Speculative Decoding).

For each model × batch × context combination, measures:
  - baseline tok/s       (no spec, no quant)
  - spec-only tok/s      (n-gram spec, fp16 KV)
  - quant-only tok/s     (no spec, FP8-sym KV)
  - combined tok/s       (n-gram spec + FP8-sym KV)

Then computes the composition ratio:
  rho = S_combined / (S_spec / S_baseline * S_quant / S_baseline)
      = S_combined * S_baseline / (S_spec * S_quant / S_baseline)

Paper finding: rho < 1 (sub-multiplicative) when KV traffic < 3%
of total bandwidth; rho > 1 (super-multiplicative) at long context
where KV fraction grows beyond ~6%.

Metrics printed:
  SPEC_N_CONFIGS=<int>
  SPEC_SUB_MULT=<int>       configs where rho < 0.9
  SPEC_SUPER_MULT=<int>     configs where rho > 1.1
  SPEC_MEAN_RHO=<float>

Pass criterion: at least one super-multiplicative config found at
long context (T >= 8K) with small batch (B=1) for any model.
Exits 0 on pass, 1 on failure, 2 on skip.
"""
from __future__ import annotations

import json
import os
import sys
import time

RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/spec_decode_results.json")

MODELS = {
    "qwen25_7b": os.environ.get("KNLP_MODEL_QWEN25_7B", "Qwen/Qwen2.5-7B-Instruct"),
}
BATCH_SIZES = [1, 4, 16]
CTX_LENGTHS = [2048, 8192, 16384]
SPEC_K = 5
MEASURE_STEPS = 20
WARMUP_STEPS = 3


def _measure_tps(llm, prompts: list[str], max_tokens: int = 64) -> float | None:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    # Warmup.
    for _ in range(WARMUP_STEPS):
        try:
            llm.generate(prompts, params)
        except Exception:
            return None
    t0 = time.perf_counter()
    total = 0
    for _ in range(MEASURE_STEPS):
        try:
            outs = llm.generate(prompts, params)
            for o in outs:
                total += sum(len(x.token_ids) for x in o.outputs)
        except Exception:
            return None
    elapsed = time.perf_counter() - t0
    return total / elapsed if elapsed > 0 else None


def _build_prompts(tokenizer, batch: int, ctx: int) -> list[str]:
    ids = [1] * max(1, ctx - 64)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return [text] * batch


def _load_llm(model_id: str, kv_dtype, use_spec: bool, max_len: int):
    from vllm import LLM

    kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        kv_cache_dtype=kv_dtype,
        enforce_eager=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.85,
    )
    if use_spec:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": SPEC_K,
            "prompt_lookup_max": 4,
        }
    try:
        return LLM(**kwargs)
    except Exception:
        # speculative_config API may differ across vLLM versions.
        if use_spec:
            return None
        raise


def main() -> int:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"SKIP: {e}", flush=True)
        return 2

    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU available", flush=True)
        return 2

    all_results: list[dict] = []
    max_len = max(CTX_LENGTHS) + 256

    for model_key, model_id in MODELS.items():
        print(f"\n=== model: {model_key} ===", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"  ERROR loading tokenizer: {e}", flush=True)
            continue

        for ctx in CTX_LENGTHS:
            for batch in BATCH_SIZES:
                cell = {"model": model_key, "batch": batch, "ctx": ctx}
                prompts = _build_prompts(tokenizer, batch, ctx)
                print(f"  B={batch} T={ctx//1024}K", flush=True, end="")

                tps: dict[str, float | None] = {}
                conditions = [
                    ("baseline", "auto", False),
                    ("quant_only", "fp8_e4m3", False),
                    ("spec_only", "auto", True),
                    ("combined", "fp8_e4m3", True),
                ]
                for cond_name, kv_dtype, use_spec in conditions:
                    llm = _load_llm(model_id, kv_dtype, use_spec, max_len)
                    if llm is None:
                        tps[cond_name] = None
                    else:
                        tps[cond_name] = _measure_tps(llm, prompts)
                        del llm

                cell.update(tps)

                # Compute rho.
                b = tps.get("baseline")
                s = tps.get("spec_only")
                q = tps.get("quant_only")
                c = tps.get("combined")
                rho = None
                if b and s and q and c and b > 0:
                    s_spec = s / b
                    s_quant = q / b
                    denom = s_spec * s_quant
                    if denom > 0:
                        rho = (c / b) / denom
                cell["rho"] = rho
                print(
                    f"  baseline={b:.1f if b else 'NA'} "
                    f"spec={s:.1f if s else 'NA'} "
                    f"quant={q:.1f if q else 'NA'} "
                    f"combined={c:.1f if c else 'NA'} "
                    f"rho={rho:.3f if rho else 'NA'}",
                    flush=True,
                )
                all_results.append(cell)

    valid = [r for r in all_results if r.get("rho") is not None]
    n_sub = sum(1 for r in valid if r["rho"] < 0.9)
    n_super = sum(1 for r in valid if r["rho"] > 1.1)
    mean_rho = sum(r["rho"] for r in valid) / len(valid) if valid else 0.0

    print(f"\nSPEC_N_CONFIGS={len(valid)}", flush=True)
    print(f"SPEC_SUB_MULT={n_sub}", flush=True)
    print(f"SPEC_SUPER_MULT={n_super}", flush=True)
    print(f"SPEC_MEAN_RHO={mean_rho:.4f}", flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(
            {
                "results": all_results,
                "summary": {
                    "n_configs": len(valid),
                    "n_sub_mult": n_sub,
                    "n_super_mult": n_super,
                    "mean_rho": mean_rho,
                },
            },
            f,
            indent=2,
        )
    print(f"Results written to {RESULT_PATH}", flush=True)

    # Pass: at least one super-multiplicative config at long ctx + small batch.
    long_ctx_small_batch = [
        r for r in valid if r["ctx"] >= 8192 and r["batch"] == 1 and r["rho"] > 1.1
    ]
    if not long_ctx_small_batch:
        print(
            "GATE FAILED: no super-multiplicative config found at "
            "long context (T≥8K) + B=1; paper predicts rho>1.1 there",
            flush=True,
        )
        return 1

    print("SPEC DECODE GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
