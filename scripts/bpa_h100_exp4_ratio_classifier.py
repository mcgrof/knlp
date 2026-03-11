#!/usr/bin/env python3
"""BPA H100 Experiment 4: Ratio Classifier Validation.

Runs v52/v53 ratio classifier on H100. Tests INT6/INT8 key
logit error ratio to confirm threshold is hardware-independent.
"""

import asyncio
import json
import time

from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="bpa-h100-exp4-ratio",
    gpu=GpuGroup.ADA_80_PRO,
    workers=1,
    idle_timeout=300,
    execution_timeout_ms=0,
    dependencies=["torch", "transformers", "accelerate", "numpy"],
)
async def run_ratio_classifier(config):
    import gc
    import math
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    W_SINK = 4
    GROUP_SIZE = 32

    def quantize_intN_grouped(tensor, n_bits, group_size=32):
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

    bits_map = {"int4": 4, "int6": 6, "int8": 8}

    def quantize_tensor(tensor, quant_type, group_size=32):
        if quant_type in ("fp16", "bf16", None):
            return tensor
        return quantize_intN_grouped(
            tensor, bits_map[quant_type], group_size
        )

    def cache_get_kv(past, li):
        if hasattr(past, "layers"):
            return past.layers[li].keys, past.layers[li].values
        return past[li]

    def cache_set_kv(past, li, k, v):
        if hasattr(past, "layers"):
            past.layers[li].keys = k
            past.layers[li].values = v
        elif isinstance(past, list):
            past[li] = (k, v)

    def n_cache_layers(past):
        if hasattr(past, "layers"):
            return len(past.layers)
        return len(past)

    def ensure_mutable(past):
        return list(past) if isinstance(past, tuple) else past

    def quantize_cache_keys_only(past, k_type):
        past = ensure_mutable(past)
        for li in range(n_cache_layers(past)):
            k, v = cache_get_kv(past, li)
            if k.shape[2] > W_SINK:
                k_s, k_f = k[:, :, :W_SINK, :], k[:, :, W_SINK:, :]
                if k_type not in ("fp16", None):
                    k_f = quantize_tensor(k_f, k_type, GROUP_SIZE)
                cache_set_kv(
                    past, li,
                    torch.cat([k_s, k_f], dim=2), v,
                )
        return past

    model_name = config["model_name"]
    prompt_len = config.get("prompt_len", 2048)
    n_prompts = config.get("n_prompts", 5)

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    gpu_name = torch.cuda.get_device_name(0)
    vocab_size = tokenizer.vocab_size

    # Test INT6 and INT8 keys (V stays FP16)
    bit_tests = ["int6", "int8"]
    logit_errors = {}

    for k_type in bit_tests:
        errs = []
        for seed in range(n_prompts):
            rng = np.random.RandomState(seed + 200)
            input_ids = torch.tensor(
                rng.randint(100, vocab_size - 100, (prompt_len,)),
                dtype=torch.long,
            ).unsqueeze(0).to(model.device)

            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values

            past_q = quantize_cache_keys_only(past_q, k_type)

            next_tok = torch.tensor(
                [[rng.randint(100, vocab_size - 100)]]
            ).to(model.device)
            with torch.no_grad():
                o1 = model(next_tok, past_key_values=past_fp)
                o2 = model(next_tok, past_key_values=past_q)

            err = (
                (o1.logits[0, 0].float() - o2.logits[0, 0].float())
                .abs()
                .max()
                .item()
            )
            errs.append(err)

        logit_errors[k_type] = float(np.mean(errs))

    ratio = logit_errors["int6"] / max(logit_errors["int8"], 1e-8)
    needs_fp16 = ratio > 3.0

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "logit_error_INT6": logit_errors["int6"],
        "logit_error_INT8": logit_errors["int8"],
        "ratio_INT6_INT8": ratio,
        "needs_fp16_keys": bool(needs_fp16),
        "threshold": 3.0,
        "prompt_len": prompt_len,
        "n_prompts": n_prompts,
        "gpu": gpu_name,
    }


async def main():
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "tiiuae/falcon-7b",
        "01-ai/Yi-1.5-9B",
    ]

    all_results = []
    for model_name in models:
        short = model_name.split("/")[-1]
        print(f"\n>>> Running ratio classifier: {short}")
        t0 = time.time()
        try:
            result = await run_ratio_classifier({
                "model_name": model_name,
                "prompt_len": 2048,
                "n_prompts": 5,
            })
            elapsed = time.time() - t0
            result["elapsed_s"] = elapsed
            all_results.append(result)
            print(
                f"    INT6_err={result['logit_error_INT6']:.4f} "
                f"INT8_err={result['logit_error_INT8']:.4f} "
                f"ratio={result['ratio_INT6_INT8']:.2f} "
                f"needs_fp16={result['needs_fp16_keys']} "
                f"({elapsed:.0f}s)"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"model": model_name, "error": str(e)})

    out_path = (
        "/data/knlp-key-results/bpa-h100/json/h100_ratio_classifier.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {"experiment": "h100_ratio_classifier", "results": all_results},
            f, indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
