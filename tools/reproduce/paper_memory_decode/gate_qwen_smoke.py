#!/usr/bin/env python3
"""Qwen2.5-7B smoke throughput test via vLLM.

Measures tok/s for each of the three KV cache configurations using a
single decode prompt.  Runs a warm-up pass before timing.

Exits 0 always (throughput is informational, not a hard gate).
Prints structured tags for the parent stage:
  FP16_TOKS=<float>
  SYM_TOKS=<float>
  ASYM_TOKS=<float>
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("FLASHINFER_EXPERIMENTAL_ASYM_PREFILL", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get("KNLP_MODEL_QWEN", "Qwen/Qwen2.5-7B-Instruct")
RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/qwen_smoke_results.json")

PROMPT = (
    "<|im_start|>user\n"
    "Explain in 3 short sentences why autoregressive transformer decode "
    "is memory-bandwidth-limited.<|im_end|>\n<|im_start|>assistant\n"
)

CONFIGS = [
    ("fp16", "auto"),
    ("fp8_sym", "fp8_e4m3"),
    ("asym_k16_v8", ("auto", "fp8_e4m3")),
]


def main():
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"vLLM import failed: {e}")
        sys.exit(1)

    sampling = SamplingParams(temperature=0.0, max_tokens=64)
    results = {}

    for name, dtype in CONFIGS:
        print(f"\n=== {name}  kv_cache_dtype={dtype!r} ===", flush=True)
        try:
            llm = LLM(
                model=MODEL,
                dtype="bfloat16",
                kv_cache_dtype=dtype,
                gpu_memory_utilization=0.85,
                max_model_len=8192,
                enforce_eager=True,
                attention_config={"backend": "FLASHINFER"},
            )
        except Exception as e:
            print(f"  LLM init failed: {type(e).__name__}: {e}", flush=True)
            results[name] = {"error": f"init: {type(e).__name__}: {e}"}
            continue

        # Warm-up pass.
        _ = llm.generate([PROMPT], sampling)
        # Timed pass.
        t0 = time.time()
        out = llm.generate([PROMPT], sampling)
        wall = time.time() - t0
        n_tok = len(out[0].outputs[0].token_ids)
        tps = n_tok / wall if wall > 0 else 0
        text = out[0].outputs[0].text
        print(f"  tokens={n_tok}  wall={wall:.3f}s  tok/s={tps:.1f}")
        print(f"  text: {text[:120]!r}")
        results[name] = {
            "tokens": n_tok,
            "wall_s": wall,
            "tok_per_s": tps,
            "text_snippet": text[:120],
        }
        del llm
        import gc, torch

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFP16_TOKS={results.get('fp16', {}).get('tok_per_s', 0):.2f}")
    print(f"SYM_TOKS={results.get('fp8_sym', {}).get('tok_per_s', 0):.2f}")
    print(f"ASYM_TOKS={results.get('asym_k16_v8', {}).get('tok_per_s', 0):.2f}")

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULT_PATH}")
    sys.exit(0)


if __name__ == "__main__":
    main()
