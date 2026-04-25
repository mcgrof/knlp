#!/usr/bin/env python3.12
"""Round 1: WikiText-2 PPL + GSM8K on Qwen2.5-7B across 4 configs.

Decisive question: does static-calibrated FP8 (per-tensor) rescue Qwen
from the FP8 collapse, or does Qwen still fail like the paper claims?

One config per Python process (vLLM v1 doesn't release GPU memory cleanly
between configs).
"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HOME", "/runpod-volume/hf_cache/huggingface")

import torch


CONFIGS = ["fp16", "fp8_uncalib", "fp8_static_calib", "asym_k16v8"]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CALIB_PATH = "/workspace/calib/Qwen2.5-7B-Instruct-FP8KV-static"


def build_llm(config: str, max_model_len: int):
    from vllm import LLM

    model = CALIB_PATH if config == "fp8_static_calib" else BASE_MODEL
    common = dict(
        model=model,
        dtype="float16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=max_model_len,
        attention_backend="FLASHINFER",
    )

    if config == "fp16":
        return LLM(**common, kv_cache_dtype="auto")
    if config == "fp8_uncalib":
        return LLM(**common, kv_cache_dtype="fp8_e4m3")
    if config == "fp8_static_calib":
        # Calibrated checkpoint: vLLM auto-detects k_scale/v_scale tensors
        return LLM(**common, kv_cache_dtype="fp8_e4m3")
    if config == "asym_k16v8":
        return LLM(**common, kv_cache_dtype=("auto", "fp8_e4m3"))
    raise ValueError(config)


def wikitext_chunks(tokenizer, n_chunks: int, chunk_len: int):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids) - chunk_len, chunk_len):
        chunks.append(ids[i:i + chunk_len])
        if len(chunks) >= n_chunks:
            break
    return chunks


def compute_ppl(llm, chunks):
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    tok = llm.get_tokenizer()
    prompts = [tok.decode(c, skip_special_tokens=False) for c in chunks]
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    dt = time.time() - t0

    total_nll = 0.0; total_toks = 0; n_inf = 0; n_nan = 0
    for out in outs:
        logprobs = out.prompt_logprobs
        if logprobs is None: continue
        prompt_ids = out.prompt_token_ids
        for pos, step in enumerate(logprobs):
            if step is None: continue
            target = prompt_ids[pos]
            lp_obj = step.get(target)
            if lp_obj is None: continue
            val = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
            if not math.isfinite(val):
                if math.isnan(val): n_nan += 1
                else: n_inf += 1
                continue
            total_nll += -val
            total_toks += 1
    nll = total_nll / max(total_toks, 1)
    return dict(ppl=float(math.exp(nll)), nll=float(nll),
                n_tokens=total_toks, n_inf=n_inf, n_nan=n_nan,
                eval_time_s=float(dt))


def extract_gsm8k(s):
    m = re.search(r"####\s*([-0-9.,]+)", s)
    if m: return m.group(1).replace(",", "").strip()
    m = re.findall(r"([-0-9.,]+)", s)
    return (m[-1].replace(",", "").strip() if m else None)


def gsm8k_eval(llm, n=200, n_shot=8):
    """8-shot GSM8K with strict-match (paper's protocol)."""
    from datasets import load_dataset
    from vllm import SamplingParams

    train = list(load_dataset("gsm8k", "main", split="train"))[:n_shot]
    test = list(load_dataset("gsm8k", "main", split="test"))[:n]

    shots = "\n\n".join(
        f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        for ex in train
    )
    prompts = [
        shots + f"\n\nQuestion: {ex['question']}\nAnswer:"
        for ex in test
    ]
    refs = [extract_gsm8k(ex["answer"]) for ex in test]

    params = SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0,
                            stop=["\n\nQuestion:", "Question:"])
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    dt = time.time() - t0

    correct = 0
    for o, r in zip(outs, refs):
        pred = extract_gsm8k(o.outputs[0].text)
        if pred == r:
            correct += 1
    return dict(correct=correct, n=len(test), acc=correct/len(test),
                eval_time_s=float(dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--ppl-T", type=int, default=2048)
    ap.add_argument("--ppl-N", type=int, default=128)
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.config} ===", flush=True)
    max_len = max(args.ppl_T, 4096) + 128
    try:
        llm = build_llm(args.config, max_len)
    except Exception as e:
        err = f"build_llm: {e}"[:600]
        print(f"  BUILD FAIL: {err}", flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(dict(config=args.config, build_err=err)) + "\n")
        return

    tok = llm.get_tokenizer()

    # WikiText-2 PPL
    chunks = wikitext_chunks(tok, args.ppl_N, args.ppl_T)
    print(f"  PPL: {len(chunks)} chunks of {args.ppl_T}", flush=True)
    try:
        ppl_res = compute_ppl(llm, chunks)
        print(f"  PPL ppl={ppl_res['ppl']:.4f} nll={ppl_res['nll']:.4f} "
              f"n_toks={ppl_res['n_tokens']} t={ppl_res['eval_time_s']:.1f}s",
              flush=True)
    except Exception as e:
        ppl_res = {"err": str(e)[:400]}
        print(f"  PPL FAIL: {ppl_res['err']}", flush=True)

    # GSM8K
    print(f"  GSM8K n={args.gsm8k_n} 8-shot strict-match", flush=True)
    try:
        gsm_res = gsm8k_eval(llm, n=args.gsm8k_n, n_shot=8)
        print(f"  GSM8K acc={gsm_res['acc']:.3f} ({gsm_res['correct']}/{gsm_res['n']}) "
              f"t={gsm_res['eval_time_s']:.1f}s", flush=True)
    except Exception as e:
        gsm_res = {"err": str(e)[:400]}
        print(f"  GSM8K FAIL: {gsm_res['err']}", flush=True)

    record = dict(
        config=args.config,
        model=CALIB_PATH if args.config == "fp8_static_calib" else BASE_MODEL,
        ppl=ppl_res,
        gsm8k=gsm_res,
        T=args.ppl_T,
        gpu="H100",
    )
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
