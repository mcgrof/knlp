#!/usr/bin/env python3.12
"""Qwen3.6-27B decode-throughput sweep across (config, B, T) grid.

Same protocol as tools/phase1_calib_saturation/saturation_sweep.py
but pinned to one large hybrid model and three configs:

    fp16          BF16/FP16 KV (vLLM "auto" -> bfloat16 here)
    fp8_sym       symmetric FP8 via kv_cache_dtype=fp8_e4m3
    asym_k16v8    asymmetric FP16-K/FP8-V via tuple API

The same prefill+decode wall-time methodology used for the other
saturation sweeps (saturation_sweep.py) is reused here so numbers
are directly comparable.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HOME", "/runpod-volume/hf_cache/huggingface")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

import torch


MODEL = "Qwen/Qwen3.6-27B"
B_GRID = [1, 2, 4, 8, 16, 32]   # 27B + 4096 ctx + B=64 may not fit; cap at 32
T_GRID = [1024, 4096, 16384]
CONFIGS = ["fp16", "fp8_sym", "asym_k16v8"]


def build_llm(config: str, max_model_len: int):
    from vllm import LLM
    common = dict(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=max_model_len,
        attention_backend="FLASHINFER",
    )
    if config == "fp16":
        return LLM(**common, kv_cache_dtype="auto")
    if config == "fp8_sym":
        return LLM(**common, kv_cache_dtype="fp8_e4m3")
    if config == "asym_k16v8":
        return LLM(**common, kv_cache_dtype=("auto", "fp8_e4m3"))
    raise ValueError(config)


def measure_decode_throughput(llm, batch: int, context: int,
                               decode_tokens: int = 64):
    """Steady-state decode tokens/s at (batch, context).

    Same prefill+decode wall-time methodology as the other sat
    sweeps; warmup run discarded.
    """
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    prompt_ids = tok.encode("The meaning of life is", add_special_tokens=False)
    while len(prompt_ids) < context:
        prompt_ids.extend(prompt_ids)
    prompt_ids = prompt_ids[:context]
    prompts = [tok.decode(prompt_ids, skip_special_tokens=False)] * batch
    params = SamplingParams(max_tokens=decode_tokens, temperature=0.0,
                            top_p=1.0)

    # Warmup
    _ = llm.generate(prompts, sampling_params=params, use_tqdm=False)

    torch.cuda.synchronize()
    t0 = time.time()
    out = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in out)
    return total_out_tokens / dt


def _write(rows, path: Path, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_one_config(config: str, out_path: Path):
    max_model_len = max(T_GRID) + 128
    print(f"=== {MODEL} / {config} ===", flush=True)
    try:
        llm = build_llm(config, max_model_len)
    except Exception as e:
        err = f"build_llm: {e}"[:600]
        print(f"  {err}", flush=True)
        rows = []
        for B in B_GRID:
            for T in T_GRID:
                rows.append(dict(model=MODEL, config=config, B=B, T=T,
                                 tok_per_s=None, err=err, gpu="H100"))
        _write(rows, out_path, append=True)
        return

    rows = []
    for B in B_GRID:
        for T in T_GRID:
            try:
                tps = measure_decode_throughput(llm, B, T)
                print(f"  {config} B={B:3d} T={T:5d} -> {tps:.1f} tok/s",
                      flush=True)
                rows.append(dict(model=MODEL, config=config, B=B, T=T,
                                 tok_per_s=tps, err=None, gpu="H100"))
            except Exception as e:
                err = str(e)[:400]
                print(f"  {config} B={B:3d} T={T:5d} FAIL: {err}",
                      flush=True)
                rows.append(dict(model=MODEL, config=config, B=B, T=T,
                                 tok_per_s=None, err=err, gpu="H100"))
        _write(rows, out_path, append=True)
        rows = []
    print(f"done: {MODEL} / {config}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--out", default="/workspace/results/q27_saturation.jsonl")
    args = ap.parse_args()
    run_one_config(args.config, Path(args.out))


if __name__ == "__main__":
    main()
