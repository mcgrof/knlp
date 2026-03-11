#!/usr/bin/env python3
"""BPA GPU Validation: All 4 Experiments (standalone, no runpod-flash).

Runs directly on a GPU node. Combines all experiments into one script
to minimize model load overhead. Each model loaded once, all experiments
run against it before moving to the next.

Usage: python3 bpa_gpu_validation.py [--output-dir /path/to/output]
"""

import argparse
import collections
import gc
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


BITS_MAP = {"int4": 4, "int5": 5, "int6": 6, "int7": 7, "int8": 8}


def quantize_tensor(tensor, quant_type, group_size=32):
    if quant_type in ("fp16", "bf16", None):
        return tensor
    return quantize_intN_grouped(tensor, BITS_MAP[quant_type], group_size)


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
                past,
                li,
                torch.cat([k_s, k_f], dim=2),
                torch.cat([v_s, v_f], dim=2),
            )
    return past


def quantize_cache_keys_only(past, k_type):
    past = ensure_mutable(past)
    for li in range(n_cache_layers(past)):
        k, v = cache_get_kv(past, li)
        if k.shape[2] > W_SINK:
            k_s, k_f = k[:, :, :W_SINK, :], k[:, :, W_SINK:, :]
            if k_type not in ("fp16", None):
                k_f = quantize_tensor(k_f, k_type, GROUP_SIZE)
            cache_set_kv(
                past,
                li,
                torch.cat([k_s, k_f], dim=2),
                v,
            )
    return past


def run_exp1_kv_asymmetry(model, tokenizer, vocab_size, prompt_len=1024, n_prompts=3):
    """Experiment 1: KV Precision Asymmetry."""
    print("  [Exp1] KV Asymmetry")
    t0 = time.time()

    configs = [
        {"k": "fp16", "v": "fp16", "label": "K_FP16/V_FP16"},
        {"k": "fp16", "v": "int4", "label": "K_FP16/V_INT4"},
        {"k": "int8", "v": "int4", "label": "K_INT8/V_INT4"},
        {"k": "int6", "v": "int4", "label": "K_INT6/V_INT4"},
        {"k": "int4", "v": "int4", "label": "K_INT4/V_INT4"},
    ]
    raw = []

    for seed in range(n_prompts):
        sr = np.random.RandomState(seed + 100)
        input_ids = (
            torch.tensor(
                sr.randint(100, vocab_size - 100, (prompt_len,)),
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(model.device)
        )
        next_tok = torch.tensor([[sr.randint(100, vocab_size - 100)]]).to(model.device)

        with torch.no_grad():
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            o_fp = model(next_tok, past_key_values=past_fp)
            logits_fp = o_fp.logits[0, 0].float().cpu()
        del out_fp, o_fp, past_fp
        gc.collect()
        torch.cuda.empty_cache()

        for bc in configs:
            k_t, v_t, label = bc["k"], bc["v"], bc["label"]
            if k_t == "fp16" and v_t == "fp16":
                raw.append({"seed": seed, "label": label, "logit_error": 0.0})
                continue
            try:
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values
                past_q = quantize_cache(past_q, k_t, v_t)
                with torch.no_grad():
                    o_q = model(next_tok, past_key_values=past_q)
                err = (logits_fp - o_q.logits[0, 0].float().cpu()).abs().max().item()
                raw.append(
                    {
                        "seed": seed,
                        "label": label,
                        "k_type": k_t,
                        "v_type": v_t,
                        "logit_error": float(err),
                    }
                )
                del past_q, out_q, o_q
            except Exception as e:
                raw.append({"seed": seed, "label": label, "error": str(e)})
            gc.collect()
            torch.cuda.empty_cache()

    grouped = collections.defaultdict(list)
    for r in raw:
        grouped[r["label"]].append(r)
    summary = []
    for label, items in grouped.items():
        valid = [x for x in items if "error" not in x]
        if valid:
            summary.append(
                {
                    "label": label,
                    "avg_logit_error": float(
                        np.mean([x["logit_error"] for x in valid])
                    ),
                    "n_valid": len(valid),
                }
            )

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s")
    for s in summary:
        print(f"      {s['label']}: err={s['avg_logit_error']:.4f}")

    return {
        "summary": summary,
        "raw": raw,
        "prompt_len": prompt_len,
        "n_prompts": n_prompts,
        "elapsed_s": elapsed,
    }


def run_exp2_long_context(model, tokenizer, vocab_size):
    """Experiment 2: Long-Context Quality."""
    print("  [Exp2] Long-Context Quality")
    t0 = time.time()

    configs = [
        {"k": "fp16", "v": "int4", "label": "K_FP16/V_INT4"},
        {"k": "int8", "v": "int4", "label": "K_INT8/V_INT4"},
    ]
    seq_lengths = [2048, 4096, 8192]
    eval_len = 64
    results = []
    loss_fn = torch.nn.CrossEntropyLoss()

    for seq_len in seq_lengths:
        rng = np.random.RandomState(42)
        input_ids = (
            torch.tensor(
                rng.randint(100, vocab_size - 100, (seq_len,)),
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(model.device)
        )
        cont_ids = (
            torch.tensor(
                rng.randint(100, vocab_size - 100, (eval_len,)),
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(model.device)
        )

        try:
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
                out_c_fp = model(cont_ids, past_key_values=past_fp)
            logits_fp = out_c_fp.logits[:, :-1, :].float().cpu()
            targets = cont_ids[:, 1:].cpu()
            ppl_fp = math.exp(
                loss_fn(
                    logits_fp.reshape(-1, logits_fp.size(-1)),
                    targets.reshape(-1),
                ).item()
            )
            results.append(
                {
                    "seq_len": seq_len,
                    "label": "K_FP16/V_FP16",
                    "ppl": ppl_fp,
                    "ppl_delta_pct": 0.0,
                }
            )
            del past_fp, out_fp, out_c_fp, logits_fp
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results.append(
                {"seq_len": seq_len, "label": "K_FP16/V_FP16", "error": "OOM"}
            )
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for bc in configs:
            k_t, v_t, label = bc["k"], bc["v"], bc["label"]
            try:
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values
                past_q = quantize_cache(past_q, k_t, v_t)
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
                results.append(
                    {
                        "seq_len": seq_len,
                        "label": label,
                        "k_type": k_t,
                        "v_type": v_t,
                        "ppl": ppl_q,
                        "ppl_delta_pct": ppl_delta,
                    }
                )
                del past_q, out_q, out_c_q, logits_q
            except torch.cuda.OutOfMemoryError:
                results.append({"seq_len": seq_len, "label": label, "error": "OOM"})
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s")
    for r in results:
        if "error" in r:
            print(f"      L={r['seq_len']} {r.get('label','')}: {r['error']}")
        else:
            print(
                f"      L={r['seq_len']} {r['label']}: "
                f"ppl_delta={r['ppl_delta_pct']:.2f}%"
            )

    return {"results": results, "elapsed_s": elapsed}


def run_exp3_latency(model, tokenizer, vocab_size):
    """Experiment 3: Decode Latency."""
    print("  [Exp3] Decode Latency")
    t0 = time.time()

    configs = [
        {"k": "fp16", "v": "fp16", "label": "FP16/FP16"},
        {"k": "fp16", "v": "int4", "label": "FP16/INT4"},
        {"k": "int8", "v": "int4", "label": "INT8/INT4"},
        {"k": "int4", "v": "int4", "label": "INT4/INT4"},
    ]
    context_len = 1024
    gen_tokens = 50
    n_repeats = 3

    rng = np.random.RandomState(42)
    context_ids = (
        torch.tensor(
            rng.randint(100, vocab_size - 100, (context_len,)),
            dtype=torch.long,
        )
        .unsqueeze(0)
        .to(model.device)
    )

    # Warmup
    with torch.no_grad():
        _ = model(context_ids[:, :64], use_cache=True)
    torch.cuda.synchronize()

    results = []
    for bc in configs:
        k_t, v_t, label = bc["k"], bc["v"], bc["label"]
        latencies = []
        for rep in range(n_repeats):
            with torch.no_grad():
                out = model(context_ids, use_cache=True)
                past = out.past_key_values
            if k_t != "fp16" or v_t != "fp16":
                past = quantize_cache(past, k_t, v_t)
            tok = out.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(gen_tokens):
                with torch.no_grad():
                    out_d = model(tok, past_key_values=past, use_cache=True)
                    past = out_d.past_key_values
                    tok = out_d.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            per_tok_ms = ((t2 - t1) / gen_tokens) * 1000.0
            latencies.append(per_tok_ms)

        results.append(
            {
                "label": label,
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "latencies": [float(x) for x in latencies],
            }
        )

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s")
    for c in results:
        print(f"      {c['label']}: {c['mean_ms']:.2f} +/- {c['std_ms']:.2f} ms/tok")

    return {
        "context_len": context_len,
        "gen_tokens": gen_tokens,
        "n_repeats": n_repeats,
        "configs": results,
        "elapsed_s": elapsed,
    }


def run_exp4_ratio_classifier(model, tokenizer, vocab_size):
    """Experiment 4: Ratio Classifier."""
    print("  [Exp4] Ratio Classifier")
    t0 = time.time()

    n_prompts = 3
    bit_tests = ["int6", "int8"]
    logit_errors = {}

    for k_type in bit_tests:
        errs = []
        for seed in range(n_prompts):
            rng = np.random.RandomState(seed + 200)
            input_ids = (
                torch.tensor(
                    rng.randint(100, vocab_size - 100, (1024,)),
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
            next_tok = torch.tensor([[rng.randint(100, vocab_size - 100)]]).to(
                model.device
            )
            with torch.no_grad():
                o1 = model(next_tok, past_key_values=past_fp)
                o2 = model(next_tok, past_key_values=past_q)
            err = (o1.logits[0, 0].float() - o2.logits[0, 0].float()).abs().max().item()
            errs.append(err)
            del past_fp, past_q, out_fp, out_q, o1, o2
            gc.collect()
            torch.cuda.empty_cache()

        logit_errors[k_type] = float(np.mean(errs))

    ratio = logit_errors["int6"] / max(logit_errors["int8"], 1e-8)
    needs_fp16 = ratio > 3.0

    elapsed = time.time() - t0
    print(
        f"    Done: INT6={logit_errors['int6']:.4f} "
        f"INT8={logit_errors['int8']:.4f} "
        f"ratio={ratio:.2f} needs_fp16={needs_fp16} ({elapsed:.0f}s)"
    )

    return {
        "logit_error_INT6": logit_errors["int6"],
        "logit_error_INT8": logit_errors["int8"],
        "ratio_INT6_INT8": ratio,
        "needs_fp16_keys": bool(needs_fp16),
        "threshold": 3.0,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="/tmp/bpa_results", help="Output directory"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Output: {args.output_dir}")

    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    all_output = {"gpu": gpu_name, "experiments": {}}
    t_start = time.time()

    for model_name in models:
        short = model_name.split("/")[-1]
        print(f"\n=== Loading model: {short} ===")
        t_load = time.time()

        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
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
            vocab_size = tokenizer.vocab_size
            print(f"    Loaded in {time.time() - t_load:.0f}s")
        except Exception as e:
            print(f"    FAILED to load: {e}")
            all_output["experiments"][short] = {"error": str(e)}
            continue

        model_results = {}

        try:
            model_results["exp1_kv_asymmetry"] = run_exp1_kv_asymmetry(
                model, tokenizer, vocab_size
            )
        except Exception as e:
            model_results["exp1_kv_asymmetry"] = {"error": str(e)}
            print(f"    Exp1 FAILED: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            model_results["exp2_long_context"] = run_exp2_long_context(
                model, tokenizer, vocab_size
            )
        except Exception as e:
            model_results["exp2_long_context"] = {"error": str(e)}
            print(f"    Exp2 FAILED: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            model_results["exp3_latency"] = run_exp3_latency(
                model, tokenizer, vocab_size
            )
        except Exception as e:
            model_results["exp3_latency"] = {"error": str(e)}
            print(f"    Exp3 FAILED: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            model_results["exp4_ratio_classifier"] = run_exp4_ratio_classifier(
                model, tokenizer, vocab_size
            )
        except Exception as e:
            model_results["exp4_ratio_classifier"] = {"error": str(e)}
            print(f"    Exp4 FAILED: {e}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        all_output["experiments"][short] = model_results

    all_output["total_elapsed_s"] = time.time() - t_start
    print(f"\nTotal elapsed: {all_output['total_elapsed_s']:.0f}s")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_experiments.json")
    with open(combined_path, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    main()
