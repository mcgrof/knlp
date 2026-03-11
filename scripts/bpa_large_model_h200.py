#!/usr/bin/env python3
"""BPA Large Model Validation on H200 141GB.

Tests KV quantization sensitivity at 70B+ scale to determine
whether Qwen key sensitivity persists with model size.

Tests (all in ONE job per model to avoid re-downloading):
  1. KV Precision Asymmetry (K/V bit combos)
  2. Ratio Classifier (INT6/INT8 ratio threshold)
  3. KV Cache Size Impact (memory at various seq_lens)
  4. Decode Latency (tokens/sec across configs)

Models: Qwen2.5-72B, Llama-3.1-70B (or 2-70b), Mixtral-8x7B
"""

import asyncio
import json
import os
import time

from runpod_flash import Endpoint, GpuGroup, PodTemplate

RESULTS_DIR = "/data/knlp-key-results/bpa-large-model"
JSON_DIR = f"{RESULTS_DIR}/json"
LOG_DIR = f"{RESULTS_DIR}/logs"


@Endpoint(
    name="bpa-large-h200",
    gpu=GpuGroup.HOPPER_141,
    workers=1,
    idle_timeout=600,
    execution_timeout_ms=0,
    dependencies=["torch", "transformers", "datasets", "accelerate"],
    template=PodTemplate(containerDiskInGb=200),
)
async def run_all_tests(config):
    """Single endpoint: load model once, run all 4 tests."""
    import gc
    import math
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def quantize_tensor(tensor, bits):
        if bits >= 16:
            return tensor
        qmax = 2 ** (bits - 1) - 1
        scale = tensor.abs().amax(dim=-1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)
        quantized = (tensor / scale).round().clamp(-qmax, qmax)
        return quantized * scale

    model_name = config["model_name"]
    dtype_str = config.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    # Load model once
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    load_time = time.time() - t_load

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    n_layers = len(model.model.layers)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = tokenizer.vocab_size

    mcfg = model.config
    n_kv_heads = getattr(mcfg, "num_key_value_heads", mcfg.num_attention_heads)
    head_dim = getattr(mcfg, "head_dim", mcfg.hidden_size // mcfg.num_attention_heads)

    model_info = {
        "model": model_name,
        "n_params": n_params,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype_str,
        "gpu": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
        "load_time_s": load_time,
    }

    results = {"model_info": model_info}

    # ── Test 1: KV Precision Asymmetry ─────────────────────
    seq_len = config.get("seq_len", 2048)
    n_prompts = config.get("n_prompts", 5)

    torch.manual_seed(42)
    prompts = [
        torch.randint(100, vocab_size - 100, (seq_len,)) for _ in range(n_prompts)
    ]

    kv_configs = [
        (16, 16, "K_FP16/V_FP16"),
        (16, 4, "K_FP16/V_INT4"),
        (8, 4, "K_INT8/V_INT4"),
        (4, 4, "K_INT4/V_INT4"),
    ]

    # Collect baseline logits first
    baseline_logits = []
    for prompt_ids in prompts:
        input_ids = prompt_ids.unsqueeze(0).to(model.device)
        with torch.no_grad():
            out = model(input_ids)
            baseline_logits.append(out.logits.float().cpu())
            del out
        torch.cuda.empty_cache()

    asym_results = []
    for k_bits, v_bits, label in kv_configs:
        t0 = time.time()

        if k_bits >= 16 and v_bits >= 16:
            # Baseline — reuse cached logits
            errs, agrees, ppl_deltas = [], [], []
            for p_idx, prompt_ids in enumerate(prompts):
                input_ids = prompt_ids.unsqueeze(0).to(model.device)
                targets = input_ids[:, 1:].cpu()
                logits = baseline_logits[p_idx]
                shift = logits[:, :-1, :].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss()
                ppl = math.exp(
                    loss_fn(
                        shift.view(-1, shift.size(-1)),
                        targets.view(-1),
                    ).item()
                )
                errs.append(0.0)
                agrees.append(1.0)
                ppl_deltas.append(0.0)
        else:
            # Install hooks for quantization
            state = {"active": True}
            hooks = []

            def make_hook(kb, vb):
                def hook_fn(module, args, kwargs, output):
                    if not state["active"]:
                        return output
                    if isinstance(output, tuple) and len(output) >= 3:
                        ao, aw, pkv = output[0], output[1], output[2]
                        if pkv is not None and isinstance(pkv, tuple):
                            k, v = pkv
                            return (
                                ao,
                                aw,
                                (
                                    quantize_tensor(k, kb),
                                    quantize_tensor(v, vb),
                                ),
                            )
                    return output

                return hook_fn

            for layer in model.model.layers:
                h = layer.self_attn.register_forward_hook(
                    make_hook(k_bits, v_bits), with_kwargs=True
                )
                hooks.append(h)

            errs, agrees, ppl_deltas = [], [], []
            loss_fn = torch.nn.CrossEntropyLoss()

            for p_idx, prompt_ids in enumerate(prompts):
                input_ids = prompt_ids.unsqueeze(0).to(model.device)
                state["active"] = True
                with torch.no_grad():
                    out = model(input_ids)
                    logits_q = out.logits.float().cpu()
                    del out
                state["active"] = False

                logits_b = baseline_logits[p_idx]
                err = (logits_q - logits_b).abs().mean().item()
                pred_b = logits_b.argmax(dim=-1)
                pred_q = logits_q.argmax(dim=-1)
                agree = (pred_b == pred_q).float().mean().item()

                targets = input_ids[:, 1:].cpu()
                shift_b = logits_b[:, :-1, :].contiguous()
                shift_q = logits_q[:, :-1, :].contiguous()
                ppl_b = math.exp(
                    loss_fn(
                        shift_b.view(-1, shift_b.size(-1)),
                        targets.view(-1),
                    ).item()
                )
                ppl_q = math.exp(
                    loss_fn(
                        shift_q.view(-1, shift_q.size(-1)),
                        targets.view(-1),
                    ).item()
                )
                ppl_d = ((ppl_q - ppl_b) / ppl_b) * 100.0

                errs.append(err)
                agrees.append(agree)
                ppl_deltas.append(ppl_d)

                del logits_q
                torch.cuda.empty_cache()

            for h in hooks:
                h.remove()

        asym_results.append(
            {
                "label": label,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "avg_logit_error": sum(errs) / len(errs),
                "avg_token_agreement": sum(agrees) / len(agrees),
                "avg_ppl_delta_pct": sum(ppl_deltas) / len(ppl_deltas),
                "elapsed_s": time.time() - t0,
            }
        )

    results["kv_asymmetry"] = asym_results

    # ── Test 2: Ratio Classifier ───────────────────────────
    # Reuse baseline_logits from above. Test INT6 and INT8 keys.
    logit_errors = {}
    for bits in [6, 8]:
        state = {"active": True}
        hooks = []

        def make_hook_k(kb):
            def hook_fn(module, args, kwargs, output):
                if not state["active"]:
                    return output
                if isinstance(output, tuple) and len(output) >= 3:
                    ao, aw, pkv = output[0], output[1], output[2]
                    if pkv is not None and isinstance(pkv, tuple):
                        k, v = pkv
                        return (ao, aw, (quantize_tensor(k, kb), v))
                return output

            return hook_fn

        for layer in model.model.layers:
            h = layer.self_attn.register_forward_hook(
                make_hook_k(bits), with_kwargs=True
            )
            hooks.append(h)

        errors = []
        for p_idx, prompt_ids in enumerate(prompts):
            input_ids = prompt_ids.unsqueeze(0).to(model.device)
            state["active"] = True
            with torch.no_grad():
                out = model(input_ids)
                logits_q = out.logits.float().cpu()
                del out
            state["active"] = False
            err = (logits_q - baseline_logits[p_idx]).abs().mean().item()
            errors.append(err)
            del logits_q
            torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        logit_errors[bits] = sum(errors) / len(errors)

    ratio = logit_errors[6] / max(logit_errors[8], 1e-8)
    results["ratio_classifier"] = {
        "logit_error_INT6": logit_errors[6],
        "logit_error_INT8": logit_errors[8],
        "ratio_INT6_INT8": ratio,
        "needs_fp16_keys": ratio > 3.0,
        "threshold": 3.0,
    }

    # Free baseline logits
    del baseline_logits
    gc.collect()
    torch.cuda.empty_cache()

    # ── Test 3: KV Cache Size Impact ───────────────────────
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_gb = param_bytes / 1e9

    cache_results = []
    for sl in config.get("cache_seq_lens", [2048, 4096, 8192]):
        dtype_bytes = 2
        kv_per_token_per_layer = 2 * n_kv_heads * head_dim * dtype_bytes
        kv_total_fp16 = kv_per_token_per_layer * n_layers * sl
        kv_total_fp16_gb = kv_total_fp16 / 1e9
        kv_total_int4_gb = (kv_total_fp16 * 4 / 16) / 1e9
        kv_frac = kv_total_fp16_gb / (model_size_gb + kv_total_fp16_gb)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        mem_before = torch.cuda.memory_allocated() / 1e9

        try:
            torch.manual_seed(42)
            ids = torch.randint(100, vocab_size - 100, (1, sl)).to(model.device)
            with torch.no_grad():
                _ = model(ids)
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            mem_used = mem_peak - mem_before
            status = "ok"
            del ids
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mem_peak, mem_used, status = -1, -1, "oom"
            else:
                raise
        torch.cuda.empty_cache()
        gc.collect()

        cache_results.append(
            {
                "seq_len": sl,
                "kv_cache_fp16_gb": kv_total_fp16_gb,
                "kv_cache_int4_gb": kv_total_int4_gb,
                "kv_savings_gb": kv_total_fp16_gb - kv_total_int4_gb,
                "kv_fraction_of_total_fp16": kv_frac,
                "gpu_mem_peak_gb": mem_peak,
                "gpu_mem_used_gb": mem_used,
                "status": status,
            }
        )

    results["kv_cache_size"] = {
        "model_size_gb": model_size_gb,
        "per_seq_len": cache_results,
    }

    # ── Test 4: Decode Latency ─────────────────────────────
    context_len = config.get("context_len", 2048)
    gen_tokens = config.get("gen_tokens", 50)
    n_repeats = config.get("n_repeats", 3)

    torch.manual_seed(42)
    context_ids = torch.randint(100, vocab_size - 100, (1, context_len)).to(
        model.device
    )

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            context_ids[:, :64],
            max_new_tokens=5,
            do_sample=False,
        )
    torch.cuda.synchronize()

    lat_configs = [
        (16, 16, "FP16/FP16"),
        (16, 4, "FP16/INT4"),
        (8, 4, "INT8/INT4"),
    ]

    lat_results = []
    for k_bits, v_bits, label in lat_configs:
        hooks = []
        if k_bits < 16 or v_bits < 16:

            def make_lat_hook(kb, vb):
                def hook_fn(module, args, kwargs, output):
                    if isinstance(output, tuple) and len(output) >= 3:
                        ao, aw, pkv = (
                            output[0],
                            output[1],
                            output[2],
                        )
                        if pkv is not None and isinstance(pkv, tuple):
                            k, v = pkv
                            return (
                                ao,
                                aw,
                                (
                                    quantize_tensor(k, kb),
                                    quantize_tensor(v, vb),
                                ),
                            )
                    return output

                return hook_fn

            for layer in model.model.layers:
                h = layer.self_attn.register_forward_hook(
                    make_lat_hook(k_bits, v_bits),
                    with_kwargs=True,
                )
                hooks.append(h)

        latencies = []
        for rep in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(
                    context_ids,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_s = t1 - t0
            latencies.append(
                {
                    "ms_per_token": (total_s / gen_tokens) * 1000.0,
                    "tokens_per_sec": gen_tokens / total_s,
                }
            )

        for h in hooks:
            h.remove()

        mean_ms = sum(l["ms_per_token"] for l in latencies) / len(latencies)
        mean_tps = sum(l["tokens_per_sec"] for l in latencies) / len(latencies)

        lat_results.append(
            {
                "label": label,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "mean_ms_per_token": mean_ms,
                "mean_tokens_per_sec": mean_tps,
                "all_latencies": latencies,
            }
        )

    results["decode_latency"] = lat_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


async def main():
    t_start = time.time()
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 72B/70B too large for RunPod serverless (~140GB download
    # hangs indefinitely). Use fallback models that still answer
    # the scale questions: 32B Qwen (does sensitivity persist?),
    # Mixtral-8x7B (MoE architecture).
    # Llama-70B skipped: gated + too large for serverless.
    models = [
        {
            "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "dtype": "bfloat16",
            "fallback_dtype": None,
            "fallback": None,
        },
        {
            "name": "Qwen/Qwen2.5-32B-Instruct",
            "dtype": "bfloat16",
            "fallback_dtype": None,
            "fallback": None,
        },
    ]

    grand_results = {}

    for model_info in models:
        model_name = model_info["name"]
        short = model_name.split("/")[-1]

        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*60}")

        config = {
            "model_name": model_name,
            "dtype": model_info["dtype"],
            "seq_len": 2048,
            "n_prompts": 5,
            "cache_seq_lens": [2048, 4096, 8192],
            "context_len": 2048,
            "gen_tokens": 50,
            "n_repeats": 3,
        }

        try:
            t0 = time.time()
            result = await run_all_tests(config)
            elapsed = time.time() - t0
            result["total_elapsed_s"] = elapsed
            grand_results[model_name] = result
            print(f"\n>>> {short} complete in {elapsed:.0f}s")
        except Exception as e:
            err_msg = str(e)
            print(f"\n!!! {short} FAILED: {err_msg}")

            # Try FP16 dtype fallback
            if model_info.get("fallback_dtype"):
                fb_dtype = model_info["fallback_dtype"]
                print(f">>> Trying {fb_dtype}...")
                config["dtype"] = fb_dtype
                try:
                    t0 = time.time()
                    result = await run_all_tests(config)
                    elapsed = time.time() - t0
                    result["total_elapsed_s"] = elapsed
                    grand_results[model_name] = result
                    print(f">>> {short} ({fb_dtype}) done ({elapsed:.0f}s)")
                    continue
                except Exception as e2:
                    print(f">>> {fb_dtype} also failed: {e2}")

            # Try fallback model
            if model_info.get("fallback"):
                fallback = model_info["fallback"]
                fb_short = fallback.split("/")[-1]
                print(f">>> Trying fallback: {fb_short}")
                config["model_name"] = fallback
                config["dtype"] = model_info["dtype"]
                try:
                    t0 = time.time()
                    result = await run_all_tests(config)
                    elapsed = time.time() - t0
                    result["total_elapsed_s"] = elapsed
                    grand_results[fallback] = result
                    print(f">>> {fb_short} done ({elapsed:.0f}s)")
                except Exception as e2:
                    print(f">>> {fb_short} also failed: {e2}")
                    grand_results[fallback] = {"error": str(e2)}
            else:
                grand_results[model_name] = {"error": err_msg}

        # Save per-model JSON
        safe = short.lower().replace("-", "_").replace(".", "_")
        actual_model = list(grand_results.keys())[-1]
        actual_result = grand_results[actual_model]
        if "error" not in actual_result:
            for test_name in [
                "kv_asymmetry",
                "ratio_classifier",
                "kv_cache_size",
                "decode_latency",
            ]:
                if test_name in actual_result:
                    out = f"{JSON_DIR}/large_{test_name}_{safe}.json"
                    with open(out, "w") as f:
                        json.dump(
                            {
                                "experiment": f"large_{test_name}",
                                "model_info": actual_result["model_info"],
                                "results": actual_result[test_name],
                            },
                            f,
                            indent=2,
                            default=str,
                        )
                    print(f"  Saved: {out}")

    # Combined results
    out_path = f"{JSON_DIR}/large_model_combined.json"
    with open(out_path, "w") as f:
        json.dump(grand_results, f, indent=2, default=str)
    print(f"\nCombined: {out_path}")

    total_time = time.time() - t_start
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mname, res in grand_results.items():
        short = mname.split("/")[-1]
        if "error" in res:
            print(f"\n{short}: FAILED — {res['error'][:80]}")
            continue

        info = res.get("model_info", {})
        print(
            f"\n{short} ({info.get('n_params',0)/1e9:.1f}B, "
            f"D={info.get('n_layers','?')}, "
            f"loaded in {info.get('load_time_s',0):.0f}s):"
        )

        if "kv_asymmetry" in res:
            for r in res["kv_asymmetry"]:
                print(
                    f"  {r['label']:20s} "
                    f"logit_err={r['avg_logit_error']:.4f} "
                    f"agree={r['avg_token_agreement']:.4f} "
                    f"ppl_d={r['avg_ppl_delta_pct']:+.2f}%"
                )

        if "ratio_classifier" in res:
            rc = res["ratio_classifier"]
            print(
                f"  Ratio: {rc['ratio_INT6_INT8']:.2f} "
                f"(needs_fp16={rc['needs_fp16_keys']})"
            )

        if "kv_cache_size" in res:
            cs = res["kv_cache_size"]
            for r in cs["per_seq_len"]:
                print(
                    f"  L={r['seq_len']:5d}: "
                    f"KV={r['kv_cache_fp16_gb']:.2f}GB "
                    f"({r['kv_fraction_of_total_fp16']:.1%})"
                )

        if "decode_latency" in res:
            for r in res["decode_latency"]:
                print(
                    f"  Lat {r['label']:12s}: "
                    f"{r['mean_ms_per_token']:.1f} ms/tok "
                    f"({r['mean_tokens_per_sec']:.1f} tok/s)"
                )


if __name__ == "__main__":
    asyncio.run(main())
