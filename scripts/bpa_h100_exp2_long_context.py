#!/usr/bin/env python3
"""BPA H100 Experiment 2: Long-Context Quality (up to 64K).

Tests whether KV quantization error accumulates with context length.
Uses prefill+quantize+continue approach.
"""

import asyncio
import json
import time

from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="bpa-h100-exp2-longctx",
    gpu=GpuGroup.ADA_80_PRO,
    workers=1,
    idle_timeout=300,
    execution_timeout_ms=0,
    dependencies=["torch", "transformers", "accelerate", "numpy"],
)
async def run_long_context(config):
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
    seq_lengths = config["seq_lengths"]
    eval_len = 128

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

    all_results = []

    for seq_len in seq_lengths:
        rng = np.random.RandomState(42)
        input_ids = torch.tensor(
            rng.randint(100, vocab_size - 100, (seq_len,)),
            dtype=torch.long,
        ).unsqueeze(0).to(model.device)
        cont_ids = torch.tensor(
            rng.randint(100, vocab_size - 100, (eval_len,)),
            dtype=torch.long,
        ).unsqueeze(0).to(model.device)

        # Baseline prefill
        try:
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
        except torch.cuda.OutOfMemoryError:
            all_results.append({
                "seq_len": seq_len, "error": "OOM on baseline"
            })
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Baseline continuation PPL
        with torch.no_grad():
            out_c_fp = model(cont_ids, past_key_values=past_fp)
        logits_fp = out_c_fp.logits[:, :-1, :].float().cpu()
        targets = cont_ids[:, 1:].cpu()
        loss_fn = torch.nn.CrossEntropyLoss()
        ppl_fp = math.exp(
            loss_fn(
                logits_fp.reshape(-1, logits_fp.size(-1)),
                targets.reshape(-1),
            ).item()
        )

        all_results.append({
            "seq_len": seq_len, "label": "K_FP16/V_FP16",
            "ppl_base": ppl_fp, "ppl_quant": ppl_fp,
            "ppl_delta_pct": 0.0,
        })

        del past_fp, out_fp, out_c_fp, logits_fp
        gc.collect()
        torch.cuda.empty_cache()

        for bc in bit_configs:
            k_type, v_type, label = bc["k_type"], bc["v_type"], bc["label"]
            try:
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values

                past_q = quantize_cache(past_q, k_type, v_type)

                with torch.no_grad():
                    out_c_q = model(cont_ids, past_key_values=past_q)

                logits_q = out_c_q.logits[:, :-1, :].float().cpu()
                ppl_q = math.exp(
                    loss_fn(
                        logits_q.reshape(-1, logits_q.size(-1)),
                        targets.reshape(-1),
                    ).item()
                )
                ppl_delta = ((ppl_q - ppl_fp) / ppl_fp) * 100.0

                all_results.append({
                    "seq_len": seq_len, "label": label,
                    "k_type": k_type, "v_type": v_type,
                    "ppl_base": ppl_fp, "ppl_quant": ppl_q,
                    "ppl_delta_pct": ppl_delta,
                })

                del past_q, out_q, out_c_q, logits_q
                gc.collect()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                all_results.append({
                    "seq_len": seq_len, "label": label, "error": "OOM"
                })
                gc.collect()
                torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name, "gpu": gpu_name, "results": all_results,
    }


async def main():
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    bit_configs = [
        {"k_type": "fp16", "v_type": "int4", "label": "K_FP16/V_INT4"},
        {"k_type": "int8", "v_type": "int4", "label": "K_INT8/V_INT4"},
    ]
    seq_lengths = [4096, 8192, 16384, 32768, 65536]

    all_results = []
    for model_name in models:
        short = model_name.split("/")[-1]
        print(f"\n>>> Running long-context for: {short}")
        t0 = time.time()
        try:
            result = await run_long_context({
                "model_name": model_name,
                "bit_configs": bit_configs,
                "seq_lengths": seq_lengths,
            })
            elapsed = time.time() - t0
            result["elapsed_s"] = elapsed
            all_results.append(result)
            print(f"    Completed in {elapsed:.0f}s")
            for r in result["results"]:
                if "error" in r:
                    print(f"    L={r['seq_len']} {r.get('label','')}: "
                          f"{r['error']}")
                else:
                    print(f"    L={r['seq_len']} {r['label']}: "
                          f"ppl_delta={r['ppl_delta_pct']:.2f}%")
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"model": model_name, "error": str(e)})

    out_path = "/data/knlp-key-results/bpa-h100/json/h100_long_context.json"
    with open(out_path, "w") as f:
        json.dump(
            {"experiment": "h100_long_context", "results": all_results},
            f, indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
