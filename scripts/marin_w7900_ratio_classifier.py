#!/usr/bin/env python3
"""Run the Marin ratio classifier safely on AMD W7900.

This script exists because the H100/RunPod ratio classifier path is not a good
fit for prune's shared W7900 environment.

Key W7900-specific choices:
- use `device_map=None` and explicit `model.to("cuda")` instead of
  `device_map="auto"`, which triggered bad ROCm/accelerate behavior here;
- default to a smaller prompt length and fewer prompts to avoid blowing up host
  memory on a shared workstation GPU;
- run INT6 and INT8 passes sequentially and clear memory aggressively.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

W_SINK = 4
GROUP_SIZE = 32
BITS_MAP = {"int4": 4, "int6": 6, "int8": 8}


def quantize_intn_grouped(
    tensor: torch.Tensor, n_bits: int, group_size: int = GROUP_SIZE
) -> torch.Tensor:
    flat = tensor.reshape(-1)
    pad = (group_size - flat.numel() % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -qmax - 1
    scale = groups.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    quantized = (groups / scale).round().clamp(qmin, qmax)
    dequantized = quantized * scale
    flat_out = dequantized.reshape(-1)
    return flat_out[: tensor.numel()].reshape(tensor.shape)


def quantize_tensor(tensor: torch.Tensor, quant_type: str | None) -> torch.Tensor:
    if quant_type in ("fp16", "bf16", None):
        return tensor
    return quantize_intn_grouped(tensor, BITS_MAP[quant_type], GROUP_SIZE)


def cache_get_kv(past, layer_idx: int):
    if hasattr(past, "layers"):
        return past.layers[layer_idx].keys, past.layers[layer_idx].values
    return past[layer_idx]


def cache_set_kv(past, layer_idx: int, k, v):
    if hasattr(past, "layers"):
        past.layers[layer_idx].keys = k
        past.layers[layer_idx].values = v
    else:
        past[layer_idx] = (k, v)


def n_cache_layers(past) -> int:
    if hasattr(past, "layers"):
        return len(past.layers)
    return len(past)


def ensure_mutable(past):
    return list(past) if isinstance(past, tuple) else past


def quantize_cache_keys_only(past, k_type: str):
    past = ensure_mutable(past)
    for layer_idx in range(n_cache_layers(past)):
        k, v = cache_get_kv(past, layer_idx)
        if k.shape[2] <= W_SINK:
            continue
        k_sink = k[:, :, :W_SINK, :]
        k_rest = k[:, :, W_SINK:, :]
        if k_type not in ("fp16", None):
            k_rest = quantize_tensor(k_rest, k_type)
        cache_set_kv(past, layer_idx, torch.cat([k_sink, k_rest], dim=2), v)
    return past


def run_classifier(
    model_name: str,
    prompt_len: int,
    n_prompts: int,
    dtype: str,
) -> dict:
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to("cuda")
    model.eval()

    vocab_size = tokenizer.vocab_size
    logit_errors = {}
    seed_results = []
    start = time.time()

    for k_type in ["int6", "int8"]:
        errs = []
        for seed in range(n_prompts):
            rng = np.random.RandomState(seed + 200)
            input_ids = (
                torch.tensor(
                    rng.randint(100, vocab_size - 100, (prompt_len,)),
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .to(model.device)
            )

            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values

            past_q = quantize_cache_keys_only(past_q, k_type)
            next_tok = torch.tensor(
                [[rng.randint(100, vocab_size - 100)]], dtype=torch.long
            ).to(model.device)

            with torch.no_grad():
                out_ref = model(next_tok, past_key_values=past_fp)
                out_test = model(next_tok, past_key_values=past_q)

            err = (
                (out_ref.logits[0, 0].float() - out_test.logits[0, 0].float())
                .abs()
                .max()
                .item()
            )
            errs.append(err)
            seed_results.append({"k_type": k_type, "seed": seed, "max_logit_err": err})

            del out_fp, past_fp, out_q, past_q, out_ref, out_test, input_ids, next_tok
            gc.collect()
            torch.cuda.empty_cache()

        logit_errors[k_type] = float(np.mean(errs))

    ratio = logit_errors["int6"] / max(logit_errors["int8"], 1e-8)
    result = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": dtype,
        "prompt_len": prompt_len,
        "n_prompts": n_prompts,
        "load_mode": "device_map=None_then_model.to(cuda)",
        "logit_error_INT6": logit_errors["int6"],
        "logit_error_INT8": logit_errors["int8"],
        "ratio_INT6_INT8": ratio,
        "needs_fp16_keys": bool(ratio > 3.0),
        "threshold": 3.0,
        "elapsed_s": time.time() - start,
        "seed_results": seed_results,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="marin-community/marin-8b-base",
    )
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--n-prompts", type=int, default=2)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_classifier(
        model_name=args.model_name,
        prompt_len=args.prompt_len,
        n_prompts=args.n_prompts,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
