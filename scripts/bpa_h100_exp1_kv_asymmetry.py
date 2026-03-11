#!/usr/bin/env python3
"""BPA H100 Experiment 1: KV Precision Asymmetry Validation.

Replicates v49 key asymmetry on H100. Simple approach: fresh prefill
per config to avoid slow deep_clone of DynamicCache. Reduced to 2
prompts and logit error only (no PPL) for speed.
"""

import asyncio
import json
import time

from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="bpa-h100-exp1-kvasym",
    gpu=GpuGroup.ADA_80_PRO,
    workers=1,
    idle_timeout=300,
    execution_timeout_ms=0,
    dependencies=["torch", "transformers", "accelerate", "numpy"],
)
async def run_kv_asymmetry(config):
    import gc
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

    bits_map = {"int4": 4, "int5": 5, "int6": 6, "int7": 7, "int8": 8}

    def quantize_tensor(tensor, quant_type, group_size=32):
        if quant_type in ("fp16", "bf16", None):
            return tensor
        return quantize_intN_grouped(
            tensor, bits_map[quant_type], group_size
        )

    def cache_get_kv(past, li):
        if hasattr(past, "layers"):
            layer = past.layers[li]
            return layer.keys, layer.values
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
        if isinstance(past, tuple):
            return list(past)
        return past

    def quantize_cache(past, k_type, v_type):
        past = ensure_mutable(past)
        D = n_cache_layers(past)
        for li in range(D):
            k, v = cache_get_kv(past, li)
            if k.shape[2] > W_SINK:
                k_sink = k[:, :, :W_SINK, :]
                k_far = k[:, :, W_SINK:, :]
                v_sink = v[:, :, :W_SINK, :]
                v_far = v[:, :, W_SINK:, :]
                if k_type not in ("fp16", None):
                    k_far = quantize_tensor(k_far, k_type, GROUP_SIZE)
                if v_type not in ("fp16", None):
                    v_far = quantize_tensor(v_far, v_type, GROUP_SIZE)
                cache_set_kv(
                    past, li,
                    torch.cat([k_sink, k_far], dim=2),
                    torch.cat([v_sink, v_far], dim=2),
                )
        return past

    model_name = config["model_name"]
    bit_configs = config["bit_configs"]
    prompt_len = config.get("prompt_len", 2048)
    n_prompts = config.get("n_prompts", 2)

    cfg = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
    cfg._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    gpu_name = torch.cuda.get_device_name(0)
    vocab_size = tokenizer.vocab_size

    all_results = []

    for seed in range(n_prompts):
        sr = np.random.RandomState(seed + 100)
        input_ids = torch.tensor(
            sr.randint(100, vocab_size - 100, (prompt_len,)),
            dtype=torch.long,
        ).unsqueeze(0).to(model.device)
        next_tok = torch.tensor(
            [[sr.randint(100, vocab_size - 100)]]
        ).to(model.device)

        # FP16 baseline: fresh prefill
        with torch.no_grad():
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            o_fp = model(next_tok, past_key_values=past_fp)
            logits_fp = o_fp.logits[0, 0].float().cpu()

        del out_fp, o_fp, past_fp
        gc.collect()
        torch.cuda.empty_cache()

        for bc in bit_configs:
            k_type = bc["k_type"]
            v_type = bc["v_type"]
            label = bc["label"]

            if k_type == "fp16" and v_type == "fp16":
                all_results.append({
                    "seed": seed,
                    "label": label,
                    "k_type": k_type,
                    "v_type": v_type,
                    "logit_error": 0.0,
                })
                continue

            try:
                # Fresh prefill for each config
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values

                past_q = quantize_cache(past_q, k_type, v_type)

                with torch.no_grad():
                    o_q = model(
                        next_tok, past_key_values=past_q,
                    )
                logit_err = (
                    (logits_fp - o_q.logits[0, 0].float().cpu())
                    .abs()
                    .max()
                    .item()
                )

                all_results.append({
                    "seed": seed,
                    "label": label,
                    "k_type": k_type,
                    "v_type": v_type,
                    "logit_error": float(logit_err),
                })

                del past_q, out_q, o_q
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                all_results.append({
                    "seed": seed,
                    "label": label,
                    "error": str(e),
                })

    # Aggregate results by config label
    import collections
    grouped = collections.defaultdict(list)
    for r in all_results:
        grouped[r["label"]].append(r)

    summary = []
    for label, items in grouped.items():
        valid = [x for x in items if "error" not in x]
        if valid:
            summary.append({
                "label": label,
                "k_type": valid[0]["k_type"],
                "v_type": valid[0]["v_type"],
                "avg_logit_error": float(
                    np.mean([x["logit_error"] for x in valid])
                ),
                "n_valid": len(valid),
            })

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "prompt_len": prompt_len,
        "n_prompts": n_prompts,
        "gpu": gpu_name,
        "summary": summary,
        "raw": all_results,
    }


async def main():
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    bit_configs = [
        {"k_type": "fp16", "v_type": "fp16", "label": "K_FP16/V_FP16"},
        {"k_type": "fp16", "v_type": "int4", "label": "K_FP16/V_INT4"},
        {"k_type": "int8", "v_type": "int4", "label": "K_INT8/V_INT4"},
        {"k_type": "int6", "v_type": "int4", "label": "K_INT6/V_INT4"},
        {"k_type": "int4", "v_type": "int4", "label": "K_INT4/V_INT4"},
    ]

    all_results = []
    for model_name in models:
        short = model_name.split("/")[-1]
        print(f"\n>>> Running all configs for: {short}")
        t0 = time.time()
        try:
            result = await run_kv_asymmetry({
                "model_name": model_name,
                "bit_configs": bit_configs,
                "prompt_len": 2048,
                "n_prompts": 2,
            })
            elapsed = time.time() - t0
            result["elapsed_s"] = elapsed
            all_results.append(result)
            print(f"    Completed in {elapsed:.0f}s")
            for cfg in result["summary"]:
                print(
                    f"    {cfg['label']}: "
                    f"logit_err={cfg['avg_logit_error']:.4f}"
                )
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"model": model_name, "error": str(e)})

    out_path = (
        "/data/knlp-key-results/bpa-h100/json/h100_kv_asymmetry.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {"experiment": "h100_kv_asymmetry", "results": all_results},
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
