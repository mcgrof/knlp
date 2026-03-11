#!/usr/bin/env python3
"""BPA H100 Experiment 3: Decode Latency Measurement.

Measures time per token for autoregressive generation with
different KV cache precision configs on H100.
"""

import asyncio
import json
import time

from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="bpa-h100-exp3-latency",
    gpu=GpuGroup.ADA_80_PRO,
    workers=1,
    idle_timeout=300,
    execution_timeout_ms=0,
    dependencies=["torch", "transformers", "accelerate", "numpy"],
)
async def run_kernel_latency(config):
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

    bits_map = {"int4": 4, "int8": 8}

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

    def quantize_cache(past, k_type, v_type):
        past = ensure_mutable(past)
        for li in range(n_cache_layers(past)):
            k, v = cache_get_kv(past, li)
            if k.shape[2] > W_SINK:
                k_s, k_f = k[:, :, :W_SINK, :], k[:, :, W_SINK:, :]
                v_s, v_f = v[:, :, :W_SINK, :], v[:, :, W_SINK:, :]
                if k_type not in ("fp16", None):
                    k_f = quantize_tensor(k_f, k_type, GROUP_SIZE)
                if v_type not in ("fp16", None):
                    v_f = quantize_tensor(v_f, v_type, GROUP_SIZE)
                cache_set_kv(
                    past, li,
                    torch.cat([k_s, k_f], dim=2),
                    torch.cat([v_s, v_f], dim=2),
                )
        return past

    model_name = config["model_name"]
    bit_configs = config["bit_configs"]
    context_len = config.get("context_len", 2048)
    gen_tokens = config.get("gen_tokens", 100)
    n_repeats = config.get("n_repeats", 5)

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

    rng = np.random.RandomState(42)
    context_ids = torch.tensor(
        rng.randint(100, vocab_size - 100, (context_len,)),
        dtype=torch.long,
    ).unsqueeze(0).to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model(context_ids[:, :64], use_cache=True)
    torch.cuda.synchronize()

    all_results = []

    for bc in bit_configs:
        k_type, v_type, label = bc["k_type"], bc["v_type"], bc["label"]

        latencies = []
        for rep in range(n_repeats):
            # Prefill
            with torch.no_grad():
                out = model(context_ids, use_cache=True)
                past = out.past_key_values

            if k_type != "fp16" or v_type != "fp16":
                past = quantize_cache(past, k_type, v_type)

            # Decode tokens one at a time
            tok = out.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            for _ in range(gen_tokens):
                with torch.no_grad():
                    out_d = model(tok, past_key_values=past, use_cache=True)
                    past = out_d.past_key_values
                    tok = out_d.logits[:, -1:, :].argmax(dim=-1)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_token_ms = ((t1 - t0) / gen_tokens) * 1000.0
            latencies.append(per_token_ms)

        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)

        all_results.append({
            "label": label, "k_type": k_type, "v_type": v_type,
            "mean_ms_per_token": float(mean_lat),
            "std_ms_per_token": float(std_lat),
            "all_latencies_ms": [float(x) for x in latencies],
        })

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name, "context_len": context_len,
        "gen_tokens": gen_tokens, "n_repeats": n_repeats,
        "gpu": gpu_name, "configs": all_results,
    }


async def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    bit_configs = [
        {"k_type": "fp16", "v_type": "fp16", "label": "FP16/FP16"},
        {"k_type": "fp16", "v_type": "int4", "label": "FP16/INT4"},
        {"k_type": "int8", "v_type": "int4", "label": "INT8/INT4"},
        {"k_type": "int4", "v_type": "int4", "label": "INT4/INT4"},
    ]

    print(f"\n>>> Running latency benchmark")
    t0 = time.time()
    try:
        result = await run_kernel_latency({
            "model_name": model_name,
            "bit_configs": bit_configs,
            "context_len": 2048,
            "gen_tokens": 100,
            "n_repeats": 5,
        })
        elapsed = time.time() - t0
        result["elapsed_s"] = elapsed
        print(f"    Completed in {elapsed:.0f}s")
        for cfg in result["configs"]:
            print(
                f"    {cfg['label']}: "
                f"{cfg['mean_ms_per_token']:.2f} "
                f"+/- {cfg['std_ms_per_token']:.2f} ms/token"
            )
    except Exception as e:
        print(f"    FAILED: {e}")
        result = {"model": model_name, "error": str(e)}

    out_path = "/data/knlp-key-results/bpa-h100/json/h100_kernel_latency.json"
    with open(out_path, "w") as f:
        json.dump(
            {"experiment": "h100_kernel_latency", "result": result},
            f, indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
