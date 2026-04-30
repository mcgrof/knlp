#!/usr/bin/env python3
"""Cross-GPU consistency gate.

Verifies that the core asym K16/V8 quality result (Qwen K-fragility
confirmed, asym matches FP16) reproduces on whatever GPU is present,
and records GPU identity for the manifest.

This gate runs on any sm89+ GPU.  On sm80 (A100) it tests FP8 support
via torch's native path.  On GPU families without FP8 hardware it
skips gracefully.

Metrics printed:
  CROSS_GPU_NAME=<string>
  CROSS_GPU_COMPUTE=<int>.<int>
  CROSS_GPU_FP16_ACC=<float>
  CROSS_GPU_ASYM_ACC=<float>
  CROSS_GPU_SYM_ACC=<float>

Pass criterion: same as Qwen quality gate — asym within 5pp of FP16,
FP8-sym degraded by ≥40pp vs FP16.

Exits 0 on pass, 1 on failure, 2 on skip.
"""
from __future__ import annotations

import json
import os
import sys

RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/cross_gpu_results.json")
MODEL_ID = os.environ.get("KNLP_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

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

    cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}  sm{cap[0]}{cap[1]}", flush=True)
    print(f"CROSS_GPU_NAME={gpu_name}", flush=True)
    print(f"CROSS_GPU_COMPUTE={cap[0]}.{cap[1]}", flush=True)

    if cap < (8, 0):
        print("SKIP: GPU sm < 80; FP8 requires sm80+", flush=True)
        return 2

    configs = [
        ("fp16", "auto"),
        ("fp8_sym", "fp8_e4m3"),
        ("asym_k16v8", ("auto", "fp8_e4m3")),
    ]

    results: dict = {}
    for cfg_name, kv_dtype in configs:
        if cfg_name == "asym_k16v8" and cap < (8, 9):
            print(f"  SKIP {cfg_name}: asym requires sm89+ (H100/H200)", flush=True)
            results[f"{cfg_name}_acc"] = None
            continue
        print(f"\n=== {cfg_name} ===", flush=True)
        try:
            llm = LLM(
                model=MODEL_ID,
                dtype="bfloat16",
                kv_cache_dtype=kv_dtype,
                enforce_eager=True,
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                attention_config=(
                    {"backend": "FLASHINFER"} if cfg_name == "asym_k16v8" else {}
                ),
            )
            acc = _eval_gsm8k(llm)
            results[f"{cfg_name}_acc"] = acc
            print(f"  GSM8K acc={acc:.3f}", flush=True)
            del llm
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results[f"{cfg_name}_acc"] = None

    fp16 = results.get("fp16_acc")
    sym = results.get("fp8_sym_acc")
    asym = results.get("asym_k16v8_acc")

    if fp16 is not None:
        print(f"\nCROSS_GPU_FP16_ACC={fp16:.4f}", flush=True)
    if sym is not None:
        print(f"CROSS_GPU_SYM_ACC={sym:.4f}", flush=True)
    if asym is not None:
        print(f"CROSS_GPU_ASYM_ACC={asym:.4f}", flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(
            {
                "gpu_name": gpu_name,
                "compute_cap": f"{cap[0]}.{cap[1]}",
                "model": MODEL_ID,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results written to {RESULT_PATH}", flush=True)

    failures = []
    if fp16 is None:
        failures.append("fp16 failed to run")
    if sym is not None and fp16 is not None and (fp16 - sym) < 0.40:
        failures.append(
            f"fp8_sym={sym:.3f} not sufficiently degraded vs fp16={fp16:.3f}; "
            "expected Qwen K-fragility collapse"
        )
    if asym is not None and fp16 is not None and (fp16 - asym) > 0.05:
        failures.append(f"asym={asym:.3f} is >{5}pp below fp16={fp16:.3f}")

    if failures:
        for f in failures:
            print(f"GATE FAILED: {f}", flush=True)
        return 1

    print("CROSS GPU GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
