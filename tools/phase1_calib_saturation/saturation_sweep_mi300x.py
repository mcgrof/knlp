#!/usr/bin/env python3
"""MI300X saturation sweep — Phase 2.

Reuses the B x T grid and per-config measurement protocol from
the H100 sweep but with the MI300X-appropriate configs:

    bf16          BF16 KV cache, P0 baseline (MI300X is native BF16,
                  not FP16 — matches the rocm/vllm default dtype)
    fp8_uncalib   symmetric FP8 via --kv-cache-dtype=fp8, no PTPC
    ptpc_fp8      PTPC-FP8 (Per-Token-activation Per-Channel-weight),
                  AMD's recommended calibrated FP8 path

Asymmetric K/V (FP16-K/FP8-V) is not tested here — AITER does not
ship our asymmetric-kv-dtype branch, and the paper's story on AMD
is precisely "does WMMA's native FP8 path remove the symmetric
FP8 penalty that Hopper suffers?".

Env vars MUST be set before this script runs:
    VLLM_ROCM_USE_AITER=1
    VLLM_ROCM_USE_AITER_FP4BMM=0
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch


RESULT_DIR = Path("/workspace/results")
B_GRID = [2, 4, 8, 16, 32, 64]
T_GRID = [1024, 4096, 16384]

CONFIGS = ["bf16", "fp8_uncalib", "ptpc_fp8"]


def build_llm(model: str, config: str, max_model_len: int):
    from vllm import LLM

    common = dict(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        max_model_len=max_model_len,
    )

    if config == "bf16":
        return LLM(**common, kv_cache_dtype="auto")

    if config == "fp8_uncalib":
        return LLM(**common, kv_cache_dtype="fp8")

    if config == "ptpc_fp8":
        return LLM(**common, kv_cache_dtype="fp8", quantization="ptpc_fp8")

    raise ValueError(f"unknown config: {config}")


def measure_decode_throughput(llm, batch: int, context: int,
                               decode_tokens: int = 64):
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    prompt_ids = tok.encode("The meaning of life is", add_special_tokens=False)
    while len(prompt_ids) < context:
        prompt_ids.extend(prompt_ids)
    prompt_ids = prompt_ids[:context]

    prompts = [tok.decode(prompt_ids, skip_special_tokens=False)] * batch
    params = SamplingParams(max_tokens=decode_tokens, temperature=0.0, top_p=1.0)

    # Warmup (discarded)
    _ = llm.generate(prompts, sampling_params=params, use_tqdm=False)

    torch.cuda.synchronize()
    t0 = time.time()
    out = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in out)
    return total_out_tokens / dt


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _write(rows, path: Path, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_one_config(model: str, config: str, out_path: Path):
    max_model_len = max(T_GRID) + 128
    results = []
    try:
        print(f"=== {model} / {config} ===", flush=True)
        llm = build_llm(model, config, max_model_len)
    except Exception as e:
        err = f"build_llm: {e}"[:600]
        print(f"  {err}", flush=True)
        for B in B_GRID:
            for T in T_GRID:
                results.append(dict(
                    model=model, config=config, B=B, T=T,
                    tok_per_s=None, err=err, gpu="MI300X",
                ))
        _write(results, out_path, append=True)
        return

    for B in B_GRID:
        for T in T_GRID:
            try:
                tps = measure_decode_throughput(llm, B, T)
                print(f"  {config} B={B} T={T} -> {tps:.1f} tok/s", flush=True)
                results.append(dict(
                    model=model, config=config, B=B, T=T,
                    tok_per_s=tps, err=None, gpu="MI300X",
                ))
            except Exception as e:
                err = str(e)[:400]
                print(f"  {config} B={B} T={T} FAIL: {err}", flush=True)
                results.append(dict(
                    model=model, config=config, B=B, T=T,
                    tok_per_s=None, err=err, gpu="MI300X",
                ))
        _write(results, out_path, append=True)
        results = []

    print(f"done: {model}/{config}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--out-dir", default=str(RESULT_DIR))
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
    os.environ.setdefault("VLLM_ROCM_USE_AITER_FP4BMM", "0")

    out = Path(args.out_dir) / f"saturation_mi300x_{slug(args.model)}.jsonl"
    run_one_config(args.model, args.config, out)


if __name__ == "__main__":
    main()
