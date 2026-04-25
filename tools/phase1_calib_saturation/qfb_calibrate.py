#!/usr/bin/env python3.12
"""Static FP8 KV calibration for Qwen2.5-7B-Instruct via llm-compressor.

Produces a calibrated checkpoint that vLLM loads natively (the FP8
KV scales are stored as `.attn.k_scale` / `.attn.v_scale` tensors in
the model.safetensors files; vLLM's KV cache quant path picks them up
automatically when kv_cache_dtype=fp8_e4m3 is set).

Calibration set: WikiText-2 train, 512 sequences of 2048 tokens each
(~1M tokens; a fairly dense calibration budget).
"""
import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("HF_HOME", "/runpod-volume/hf_cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="/workspace/calib/Qwen2.5-7B-Instruct-FP8KV-static")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading WikiText-2 train, sampling {args.n_samples} sequences "
          f"of {args.seq_len} tokens ...", flush=True)
    from datasets import Dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)

    # Take consecutive non-overlapping windows
    samples = []
    for i in range(0, len(ids) - args.seq_len, args.seq_len):
        samples.append(ids[i:i + args.seq_len])
        if len(samples) >= args.n_samples:
            break
    print(f"Have {len(samples)} calibration sequences", flush=True)

    # llm-compressor wants a HF Dataset with input_ids + attention_mask
    calib_ds = Dataset.from_dict({
        "input_ids": samples,
        "attention_mask": [[1] * len(s) for s in samples],
    })

    # Static FP8 KV calibration ONLY — weights stay FP16.  Earlier
    # attempt used scheme="FP8_DYNAMIC" which silently quantized
    # weights too, producing FP8-weight + FP8-KV (wrong baseline).
    # No `targets`/`scheme` keyword means no Linear weight quant;
    # only the kv_cache_scheme is applied.
    recipe = QuantizationModifier(
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        },
    )

    # Run oneshot calibration
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        output_dir=args.out,
        max_seq_length=args.seq_len,
        num_calibration_samples=len(samples),
    )

    print(f"Calibrated checkpoint at {args.out}", flush=True)


if __name__ == "__main__":
    main()
