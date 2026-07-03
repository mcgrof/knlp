#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Phase-10 serving confirmation: mirror the fake-quant FP8-KV atlas in the REAL
vLLM + FlashInfer stack, so the proxy<->serving gap is measurable.

For one (model, cell) it loads the model under the cell's kv_cache_dtype and
measures, on a fixed teacher-forced holdout:
  - ppl                : teacher-forced perplexity (from vLLM prompt_logprobs)
  - per_position_top1  : argmax token id at each holdout position
  - greedy_tokens      : a short greedy continuation from a fixed prefix
The last two let the reporter compute top1-agreement and greedy-divergence
against the native cell without any cross-process state.

Cells (mirror configs/kv/fp8_failure_serving.yaml):
  native  kv_cache_dtype = "auto"                       (bf16 K + bf16 V)
  k8v8    kv_cache_dtype = "fp8_e4m3"                    (symmetric FP8 = failure trigger)
  k16v8   kv_cache_dtype = ("auto", "fp8_e4m3")          (asym bf16 K / fp8 V = the fix)

vLLM gotchas baked in (from the v0.18 asym serve work): import vllm._C to
register _C_cache_ops, drop VLLM_BATCH_INVARIANT, pass the FlashInfer backend
via attention_config (the env var is not honored), and give --model a local
snapshot dir so vLLM's repo_utils file-list does not 429 on the HF API.

Usage:
  python serving_confirm.py --model <path-or-id> --short-name qwen25_7b \
      --cell k16v8 --holdout holdout.jsonl --out out/qwen25_7b_k16v8.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.pop("VLLM_BATCH_INVARIANT", None)
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import vllm._C  # noqa: F401  -- registers _C_cache_ops
from vllm import LLM, SamplingParams


CELLS = {
    "native": "auto",
    "k8v8": "fp8_e4m3",
    "k16v8": ("auto", "fp8_e4m3"),
}


def load_holdout(path):
    """Holdout is a jsonl of {"text": ...}; return a list of strings."""
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line)["text"])
    return docs


def build_llm(model, cell, max_len):
    kv = CELLS[cell]
    return LLM(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype=kv,
        attention_config={"backend": "FLASHINFER"},
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=max_len,
        disable_log_stats=True,
    )


def teacher_forced(llm, docs):
    """Sum token logprobs over each doc -> ppl; collect per-position argmax ids."""
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outs = llm.generate(docs, sp)
    total_lp, total_tok = 0.0, 0
    top1_ids = []
    for o in outs:
        plps = o.prompt_logprobs  # list; [0] is None (first token has no context)
        toks = o.prompt_token_ids
        doc_top1 = []
        for pos, lp_dict in enumerate(plps):
            if lp_dict is None:
                continue
            tid = toks[pos]
            # logprob of the actual token at this position
            if tid in lp_dict:
                total_lp += lp_dict[tid].logprob
                total_tok += 1
            # argmax token = the rank-0 entry
            best = min(lp_dict.values(), key=lambda x: x.rank)
            doc_top1.append(best.decoded_token if False else None)
            # store the argmax token id (find key with rank 0)
            amax = next((k for k, v in lp_dict.items() if v.rank == 1), tid)
            doc_top1[-1] = int(amax)
        top1_ids.append(doc_top1)
    ppl = math.exp(-total_lp / max(1, total_tok))
    return ppl, top1_ids, total_tok


def greedy_continue(llm, prefixes, n_new):
    sp = SamplingParams(temperature=0.0, max_tokens=n_new)
    outs = llm.generate(prefixes, sp)
    return [list(o.outputs[0].token_ids) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot dir or HF id")
    ap.add_argument("--short-name", required=True)
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--greedy-new", type=int, default=48)
    ap.add_argument("--greedy-prefixes", type=int, default=4)
    args = ap.parse_args()

    docs = load_holdout(args.holdout)
    llm = build_llm(args.model, args.cell, args.max_len)

    ppl, top1_ids, n_tok = teacher_forced(llm, docs)
    # greedy divergence probe: first ~200 chars of the first few docs as prefixes
    prefixes = [d[:400] for d in docs[: args.greedy_prefixes]]
    greedy = greedy_continue(llm, prefixes, args.greedy_new)

    result = {
        "model": args.short_name,
        "model_id": args.model,
        "cell": args.cell,
        "kv_cache_dtype": str(CELLS[args.cell]),
        "ppl": ppl,
        "n_tokens": n_tok,
        "per_position_top1": top1_ids,
        "greedy_tokens": greedy,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(
        f"[serve] {args.short_name} {args.cell}: ppl={ppl:.3f} "
        f"n_tok={n_tok} greedy_docs={len(greedy)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
