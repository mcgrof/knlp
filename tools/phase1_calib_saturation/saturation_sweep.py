#!/usr/bin/env python3.12
"""Per-model decode throughput sweep across B x T grid.

Configs (5 total after the API reality-check on vLLM 0.19):
    fp16              FP16 KV cache, P0 baseline
    fp8_uncalib       symmetric FP8, unit scales (vLLM default)
    fp8_calib         symmetric FP8, calculate_kv_scales=True
                      (vLLM's built-in runtime KV calibration)
    asym_uncalib      asymmetric FP16-K/FP8-V, unit V scales
    asym_calib        asymmetric FP16-K/FP8-V, calculate_kv_scales=True

Per-channel K calibration (KVQuant-style) would require a static
calibrated FP8 checkpoint produced by llm-compressor. vLLM 0.19
loads such checkpoints by detecting .attn.k_scale / .attn.v_scale
tensors in safetensors at model init; it does not accept external
JSON scales. That path is deferred to Phase 1b.

Output: /workspace/results/saturation_<model_slug>.jsonl
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

CONFIGS = [
    "fp16",
    "fp8_uncalib",
    "fp8_calib",
    "asym_uncalib",
    "asym_calib",
]


def build_llm(model: str, config: str, max_model_len: int):
    from vllm import LLM

    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    common = dict(
        model=model,
        dtype="float16",
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        max_model_len=max_model_len,
        attention_backend="FLASHINFER",
    )

    if config == "fp16":
        return LLM(**common, kv_cache_dtype="auto")

    if config == "fp8_uncalib":
        return LLM(**common, kv_cache_dtype="fp8_e4m3")

    if config == "fp8_calib":
        return LLM(**common, kv_cache_dtype="fp8_e4m3",
                   calculate_kv_scales=True)

    if config == "asym_uncalib":
        return LLM(**common, kv_cache_dtype="auto;fp8_e4m3")

    if config == "asym_calib":
        return LLM(**common, kv_cache_dtype="auto;fp8_e4m3",
                   calculate_kv_scales=True)

    raise ValueError(f"unknown config: {config}")


def measure_decode_throughput(llm, batch: int, context: int, decode_tokens: int = 64):
    """Steady-state decode tokens/s at (batch, context).  Measures
    prefill+decode wall time and reports total_output_tokens / dt.
    Warmup run discarded first."""
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    prompt_ids = tok.encode("The meaning of life is", add_special_tokens=False)
    while len(prompt_ids) < context:
        prompt_ids.extend(prompt_ids)
    prompt_ids = prompt_ids[:context]

    prompts = [tok.decode(prompt_ids, skip_special_tokens=False)] * batch
    params = SamplingParams(max_tokens=decode_tokens, temperature=0.0, top_p=1.0)

    # Warmup
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


def run_one_model(model: str, out_path: Path, configs_to_run):
    results = []
    max_model_len = max(T_GRID) + 128

    for config in configs_to_run:
        try:
            print(f"=== {model} / {config} ===", flush=True)
            llm = build_llm(model, config, max_model_len)
        except Exception as e:
            err = f"build_llm: {e}"[:400]
            print(f"  {err}", flush=True)
            for B in B_GRID:
                for T in T_GRID:
                    results.append(dict(
                        model=model, config=config, B=B, T=T,
                        tok_per_s=None, err=err,
                    ))
            continue

        for B in B_GRID:
            for T in T_GRID:
                try:
                    tps = measure_decode_throughput(llm, B, T)
                    print(f"  {config} B={B} T={T} -> {tps:.1f} tok/s", flush=True)
                    results.append(dict(
                        model=model, config=config, B=B, T=T,
                        tok_per_s=tps, err=None,
                    ))
                except Exception as e:
                    err = str(e)[:400]
                    print(f"  {config} B={B} T={T} FAIL: {err}", flush=True)
                    results.append(dict(
                        model=model, config=config, B=B, T=T,
                        tok_per_s=None, err=err,
                    ))

            # Persist after every batch-block so a late crash doesn't
            # erase earlier points
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    print(f"done: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=str(RESULT_DIR))
    ap.add_argument("--configs", nargs="+", default=CONFIGS,
                    help="subset of configs to run")
    args = ap.parse_args()

    out = Path(args.out_dir) / f"saturation_{slug(args.model)}.jsonl"
    run_one_model(args.model, out, args.configs)


if __name__ == "__main__":
    main()
