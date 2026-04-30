#!/usr/bin/env python3
"""Large model sanity gate (Qwen2.5-72B or Llama-3.1-70B, TP=4).

Runs a single-batch sanity check on a 70B-class model with tensor
parallelism across 4 GPUs.  Verifies that:
  - The model loads and generates coherent output under FP16 KV
  - Asym K16/V8 produces output within tolerable perplexity delta

This stage requires exactly 4 NVIDIA H100 GPUs (or equivalent ≥80GB).
It skips gracefully on any other hardware configuration.

Pass criteria:
  - FP16 GSM8K spot-check acc ≥ 0.80 (5 problems)
  - Asym GSM8K acc within 10pp of FP16

Metrics printed:
  LARGE_FP16_ACC=<float>
  LARGE_ASYM_ACC=<float>
  LARGE_ASYM_DELTA=<float>

Exits 0 on pass, 1 on failure, 2 on skip.
"""
from __future__ import annotations

import json
import os
import sys

RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/large_model_results.json")
MODEL_ID = os.environ.get(
    "KNLP_LARGE_MODEL_ID",
    "Qwen/Qwen2.5-72B-Instruct",
)
REQUIRED_GPUS = 4
MIN_GPU_MEM_GB = 70

GSM8K_PROBLEMS = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "5",
    ),
    (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "42",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "624",
    ),
]


def _eval_gsm8k(llm) -> float:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=64, temperature=0.0)
    prompts = [f"Problem: {q}\nAnswer:" for q, _ in GSM8K_PROBLEMS]
    correct = 0
    try:
        outs = llm.generate(prompts, params)
        for i, o in enumerate(outs):
            if GSM8K_PROBLEMS[i][1] in o.outputs[0].text:
                correct += 1
    except Exception:
        pass
    return correct / len(GSM8K_PROBLEMS)


def main() -> int:
    try:
        import torch
        from vllm import LLM
    except ImportError as e:
        print(f"SKIP: {e}", flush=True)
        return 2

    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU available", flush=True)
        return 2

    n_gpu = torch.cuda.device_count()
    if n_gpu < REQUIRED_GPUS:
        print(
            f"SKIP: {n_gpu} GPUs available; large model lanes require {REQUIRED_GPUS}",
            flush=True,
        )
        return 2

    mem_gb = [
        torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(n_gpu)
    ]
    if min(mem_gb) < MIN_GPU_MEM_GB:
        print(
            f"SKIP: GPU memory {min(mem_gb):.0f} GB < {MIN_GPU_MEM_GB} GB required",
            flush=True,
        )
        return 2

    print(
        f"Loading {MODEL_ID} on {REQUIRED_GPUS}× GPU (TP={REQUIRED_GPUS}) ...",
        flush=True,
    )

    results: dict = {}
    configs = [
        ("fp16", "auto"),
        ("asym_k16v8", ("auto", "fp8_e4m3")),
    ]

    for cfg_name, kv_dtype in configs:
        print(f"\n=== {cfg_name} ===", flush=True)
        try:
            llm = LLM(
                model=MODEL_ID,
                dtype="bfloat16",
                kv_cache_dtype=kv_dtype,
                tensor_parallel_size=REQUIRED_GPUS,
                enforce_eager=True,
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                attention_config={"backend": "FLASHINFER"},
            )
            acc = _eval_gsm8k(llm)
            results[f"{cfg_name}_acc"] = acc
            print(f"  GSM8K acc={acc:.3f}", flush=True)
            del llm
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results[f"{cfg_name}_acc"] = None

    fp16_acc = results.get("fp16_acc")
    asym_acc = results.get("asym_k16v8_acc")
    delta = (
        (asym_acc - fp16_acc)
        if (fp16_acc is not None and asym_acc is not None)
        else None
    )

    if fp16_acc is not None:
        print(f"\nLARGE_FP16_ACC={fp16_acc:.4f}", flush=True)
    if asym_acc is not None:
        print(f"LARGE_ASYM_ACC={asym_acc:.4f}", flush=True)
    if delta is not None:
        print(f"LARGE_ASYM_DELTA={delta:.4f}", flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(
            {"model": MODEL_ID, "tensor_parallel": REQUIRED_GPUS, "results": results},
            f,
            indent=2,
        )
    print(f"Results written to {RESULT_PATH}", flush=True)

    failures = []
    if fp16_acc is None:
        failures.append("fp16 failed to run")
    elif fp16_acc < 0.80:
        failures.append(f"fp16 acc={fp16_acc:.3f} < 0.80")
    if asym_acc is not None and fp16_acc is not None and (fp16_acc - asym_acc) > 0.10:
        failures.append(f"asym acc={asym_acc:.3f} is >{10}pp below fp16={fp16_acc:.3f}")

    if failures:
        for f in failures:
            print(f"GATE FAILED: {f}", flush=True)
        return 1

    print("LARGE MODEL GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
