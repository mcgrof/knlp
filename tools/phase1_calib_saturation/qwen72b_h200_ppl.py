#!/usr/bin/env python3.12
"""Experiment 3 — Qwen2.5-72B on H200, FP8 activation quant.

Closes paper Limitation (1): large-model validation currently uses
weight quantization as a proxy; FP8 activation quantization at 72B
needs an H200's 141GB HBM3e.

Measures WikiText-103 perplexity at T=2048 across three configs:
    fp16              FP16 KV cache (P0 baseline)
    fp8_sym           symmetric FP8 via kv_cache_dtype=fp8_e4m3
    asym_k16_v8       asymmetric FP16-K/FP8-V (FlashInfer branch)

One config per Python process (vLLM v1 doesn't release GPU memory on del).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch


CONFIGS = ["fp16", "fp8_sym", "asym_k16_v8"]
# FP8-quantized weights fit in H200's 141GB; FP16 weights do not
# (72B*2B=144GB).  The paper claim is about KV-cache activation quant
# at 70B scale, not weight precision, so FP8 weights are fine here.
MODEL = "RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic"


def build_llm(config: str, max_model_len: int, mem_util: float):
    from vllm import LLM

    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    # RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic ships FP8 weights
    # (~72GB on disk).  The KV-cache activation-quant question is
    # independent of the weight-quant question.
    common = dict(
        model=MODEL,
        dtype="float16",
        gpu_memory_utilization=mem_util,
        enforce_eager=True,
        max_model_len=max_model_len,
        attention_backend="FLASHINFER",
    )

    # Python API path: pass asymmetric spec as tuple.  The pipe/comma
    # string form is only parsed on the CLI entry in arg_utils.py.
    if config == "fp16":
        return LLM(**common, kv_cache_dtype="auto")
    if config == "fp8_sym":
        return LLM(**common, kv_cache_dtype="fp8_e4m3")
    if config == "asym_k16_v8":
        return LLM(**common, kv_cache_dtype=("auto", "fp8_e4m3"))
    raise ValueError(config)


def wikitext_chunks(tokenizer, n_chunks: int, chunk_len: int):
    """WikiText-103 test split, chunked to fixed-length samples."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids) - chunk_len, chunk_len):
        chunks.append(ids[i:i + chunk_len])
        if len(chunks) >= n_chunks:
            break
    return chunks


def compute_ppl(llm, chunks):
    """Compute token-level NLL via logprobs over chunk continuations.

    For each chunk of N prompt tokens, vLLM's prompt_logprobs path
    returns a length-N list.  Position 0 is None.  Position i>=1 is
    a dict token_id -> Logprob for a set of tokens including the
    REALIZED token at position i (the actual token in the prompt).

    To compute per-token NLL we must index by the realized token id,
    not take "the first entry in the dict" — the dict is keyed by
    token id and its iteration order is not aligned with the prompt.
    """
    from vllm import SamplingParams

    # prompt_logprobs=0 asks vLLM to emit only the realized token's
    # logprob, which is the cheapest way to get what we need.  In
    # practice older vLLM versions require 1.  Use 1 and index.
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )

    tok = llm.get_tokenizer()
    prompts = [tok.decode(c, skip_special_tokens=False) for c in chunks]

    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    dt = time.time() - t0

    total_nll = 0.0
    total_toks = 0
    n_inf = 0
    n_nan = 0
    for out in outs:
        logprobs = out.prompt_logprobs
        if logprobs is None:
            continue
        # Use the ACTUAL prompt token ids from vLLM's tokenization of
        # the prompt string (may differ from our pre-tokenized chunk
        # if the tokenizer re-tokenizes after decode).
        prompt_ids = out.prompt_token_ids
        for pos, step in enumerate(logprobs):
            if step is None:
                continue
            target = prompt_ids[pos]
            lp_obj = step.get(target)
            if lp_obj is None:
                # Realized token wasn't emitted for this position; skip
                continue
            val = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
            if not math.isfinite(val):
                if math.isnan(val):
                    n_nan += 1
                else:
                    n_inf += 1
                continue
            total_nll += -val
            total_toks += 1
    nll_per_tok = total_nll / max(total_toks, 1)
    return dict(
        ppl=float(math.exp(nll_per_tok)),
        nll=float(nll_per_tok),
        n_tokens=total_toks,
        n_inf_skipped=n_inf,
        n_nan_skipped=n_nan,
        eval_time_s=float(dt),
    )


def run_one(config: str, n_chunks: int, chunk_len: int, mem_util: float,
            out_path: Path):
    print(f"=== Qwen2.5-72B / {config} / T={chunk_len} / N={n_chunks} ===",
          flush=True)
    try:
        llm = build_llm(config, chunk_len + 8, mem_util)
    except Exception as e:
        err = f"build_llm: {e}"[:800]
        print(f"  BUILD FAIL: {err}", flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(dict(
                model=MODEL, config=config, T=chunk_len,
                n_chunks=n_chunks, ppl=None, err=err, gpu="H200",
            )) + "\n")
        return

    tok = llm.get_tokenizer()
    chunks = wikitext_chunks(tok, n_chunks, chunk_len)
    print(f"  have {len(chunks)} chunks of {chunk_len} tokens", flush=True)

    try:
        res = compute_ppl(llm, chunks)
        print(f"  ppl={res['ppl']:.4f} n={res['n_tokens']} "
              f"t={res['eval_time_s']:.1f}s", flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(dict(
                model=MODEL, config=config, T=chunk_len,
                n_chunks=len(chunks), **res, err=None, gpu="H200",
            )) + "\n")
    except Exception as e:
        err = str(e)[:600]
        print(f"  PPL FAIL: {err}", flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(dict(
                model=MODEL, config=config, T=chunk_len,
                n_chunks=n_chunks, ppl=None, err=err, gpu="H200",
            )) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--chunk-len", type=int, default=2048)
    ap.add_argument("--n-chunks", type=int, default=128,
                    help="128 chunks at T=2048 ~= 262K tokens total")
    ap.add_argument("--mem-util", type=float, default=0.90,
                    help="H200 has 141GB; 72B in FP16 is ~144GB so we "
                         "need at least some quantization of weights, or "
                         "mem_util=0.95 and lean on swap")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_one(args.config, args.n_chunks, args.chunk_len, args.mem_util,
            out_path)


if __name__ == "__main__":
    main()
