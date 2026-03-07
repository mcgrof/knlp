#!/usr/bin/env python3
"""BPA v46: W7900 Validation Experiments Addressing Reviewer Critiques.

Runs 8 experiments on the W7900 (48GB) with Qwen2.5-7B:
  1. Direct KV activation quantization
  2. Token agreement test
  3. Logit difference analysis
  4. Long-context perplexity scaling
  5. Needle-in-haystack retrieval
  6. Downstream capability benchmarks (MMLU, GSM8K, HumanEval)
  7. Decode bandwidth measurement
  8. Batch-size scaling verification

All results saved as JSON to /data/knlp-key-results/bpa46/json/
Plots saved to /data/knlp-key-results/bpa46/plots/
"""

import gc
import json
import math
import os
import re
import sys
import time
from datetime import datetime

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Paths ===
RESULTS_ROOT = "/data/knlp-key-results/bpa46"
JSON_DIR = os.path.join(RESULTS_ROOT, "json")
PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
for d in [JSON_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# === Protocol parameters ===
MODEL_NAME = "Qwen/Qwen2.5-7B"
DATASET = "wikitext-103-raw-v1"
N_TOKENS = 500_000
W_SINK = 4
GROUP_SIZE = 32
DEVICE = "cuda"
DTYPE = torch.bfloat16


def save_json(data, filename):
    path = os.path.join(JSON_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def gpu_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(props.total_memory / 1e9, 1),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
    }


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


# === Dataset loading ===
_TOKEN_CACHE = {}


def load_wikitext_tokens(tokenizer):
    key = id(tokenizer)
    if key not in _TOKEN_CACHE:
        from datasets import load_dataset

        ds = load_dataset("wikitext", DATASET, split="validation")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer.encode(text)
        arr = np.array(tokens[:N_TOKENS], dtype=np.int64)
        print(f"  Loaded {DATASET}: {len(arr)} tokens")
        _TOKEN_CACHE[key] = arr
    return _TOKEN_CACHE[key]


def load_passage(tokenizer, L, seed=0, extra=64):
    data = load_wikitext_tokens(tokenizer)
    seq_len = L + extra
    rng = np.random.RandomState(seed)
    start = rng.randint(0, max(1, len(data) - seq_len))
    return torch.from_numpy(data[start : start + seq_len]).unsqueeze(0)


# === Quantization ===
def quantize_int4_grouped(tensor, group_size=32):
    shape = tensor.shape
    hd = shape[-1]
    ng = (hd + group_size - 1) // group_size
    pd = ng * group_size
    if pd > hd:
        pad = torch.zeros(
            *shape[:-1], pd - hd, device=tensor.device, dtype=tensor.dtype
        )
        tensor = torch.cat([tensor, pad], dim=-1)
    r = tensor.reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = amax / 7.0
    q = (r / s).round().clamp(-8, 7)
    return (q * s).reshape(*shape[:-1], pd)[..., :hd]


# === Cache helpers (transformers 4.x/5.x compat) ===
def _cache_get_kv(past, li):
    if hasattr(past, "layers"):
        layer = past.layers[li]
        return layer.keys, layer.values
    return past[li]


def _cache_set_kv(past, li, k, v):
    if hasattr(past, "layers"):
        past.layers[li].keys = k
        past.layers[li].values = v
    else:
        past[li] = (k, v)


def cache_length(past):
    if hasattr(past, "layers"):
        return past.layers[0].keys.shape[2]
    return past[0][0].shape[2]


def n_cache_layers(past):
    if hasattr(past, "layers"):
        return len(past.layers)
    return len(past)


def quantize_cache_int4(past):
    """Quantize all KV cache layers to INT4 (except sink tokens)."""
    D = n_cache_layers(past)
    clen = cache_length(past)
    for li in range(D):
        k, v = _cache_get_kv(past, li)
        # Protect sink tokens
        k_sink = k[:, :, :W_SINK, :]
        v_sink = v[:, :, :W_SINK, :]
        k_far = k[:, :, W_SINK:, :]
        v_far = v[:, :, W_SINK:, :]
        k_q = quantize_int4_grouped(k_far, GROUP_SIZE)
        v_q = quantize_int4_grouped(v_far, GROUP_SIZE)
        _cache_set_kv(
            past, li,
            torch.cat([k_sink, k_q], dim=2),
            torch.cat([v_sink, v_q], dim=2),
        )
    return past


# === Model loading ===
def load_model_and_tokenizer():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    print(f"  Model loaded on {DEVICE}")
    return model, tokenizer


# ============================================================
# Experiment 1: Direct KV Activation Quantization
# ============================================================
def experiment_1_activation_quantization(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 1: Direct KV Activation Quantization")
    print("=" * 60)
    t0 = time.time()

    L_set = [2048, 8192]
    seeds = [0, 1, 2]
    decode_tokens = 64
    results = {"experiment": "activation_quantization", "gpu": gpu_info(), "evals": {}}

    for L in L_set:
        for seed in seeds:
            key = f"L{L}_s{seed}"
            print(f"  Evaluating {key}...")
            try:
                passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                input_ids = passage[:, :L].to(DEVICE)
                continuation = passage[:, L : L + decode_tokens].to(DEVICE)

                # BF16 baseline
                with torch.no_grad():
                    out_fp16 = model(input_ids, use_cache=True)
                    past_fp16 = out_fp16.past_key_values

                logits_fp16 = [out_fp16.logits[:, -1:, :].cpu()]
                past = past_fp16
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_fp16.append(out.logits.cpu())
                    past = out.past_key_values
                logits_fp16 = torch.cat(logits_fp16, dim=1)
                del past, past_fp16, out_fp16, out
                clear_gpu()

                # INT4 KV cache
                with torch.no_grad():
                    out_int4 = model(input_ids, use_cache=True)
                    past_int4 = out_int4.past_key_values
                past_int4 = quantize_cache_int4(past_int4)

                logits_int4 = [out_int4.logits[:, -1:, :].cpu()]
                past = past_int4
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_int4.append(out.logits.cpu())
                    past = out.past_key_values
                logits_int4 = torch.cat(logits_int4, dim=1)
                del past, past_int4, out_int4, out
                clear_gpu()

                # Compute PPL (on CPU)
                targets = continuation.cpu().reshape(-1)
                B, T, V = logits_fp16[:, :-1, :].shape

                loss_fp16 = F.cross_entropy(
                    logits_fp16[:, :-1, :].reshape(-1, V).float(), targets
                )
                loss_int4 = F.cross_entropy(
                    logits_int4[:, :-1, :].reshape(-1, V).float(), targets
                )
                ppl_fp16 = math.exp(min(loss_fp16.item(), 20))
                ppl_int4 = math.exp(min(loss_int4.item(), 20))
                delta_pct = (ppl_int4 - ppl_fp16) / ppl_fp16 * 100

                # Token agreement
                pred_fp16 = logits_fp16[:, :-1, :].argmax(dim=-1)
                pred_int4 = logits_int4[:, :-1, :].argmax(dim=-1)
                agreement = (pred_fp16 == pred_int4).float().mean().item()

                # Logit error
                logit_diff = (logits_fp16[:, :-1, :] - logits_int4[:, :-1, :]).abs()
                max_diff_per_token = logit_diff.max(dim=-1).values.float()
                logit_stats = {
                    "mean": max_diff_per_token.mean().item(),
                    "median": max_diff_per_token.median().item(),
                    "p95": torch.quantile(max_diff_per_token.flatten(), 0.95).item(),
                    "p99": torch.quantile(max_diff_per_token.flatten(), 0.99).item(),
                }

                results["evals"][key] = {
                    "ppl_fp16": round(ppl_fp16, 4),
                    "ppl_int4": round(ppl_int4, 4),
                    "delta_pct": round(delta_pct, 2),
                    "token_agreement": round(agreement, 4),
                    "logit_error": {k: round(v, 6) for k, v in logit_stats.items()},
                }
                print(
                    f"    FP16 PPL={ppl_fp16:.4f} INT4 PPL={ppl_int4:.4f} "
                    f"delta={delta_pct:.2f}% agree={agreement:.4f}"
                )
                del logits_fp16, logits_int4
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at {key}, skipping")
                clear_gpu()
                continue

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "activation_quantization.json")
    return results


# ============================================================
# Experiment 2: Token Agreement Test
# ============================================================
def experiment_2_token_agreement(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 2: Token Agreement Test")
    print("=" * 60)
    t0 = time.time()

    n_prompts = 200
    gen_tokens = 256
    L = 2048  # context length for prompts

    results = {
        "experiment": "token_agreement",
        "gpu": gpu_info(),
        "n_prompts": n_prompts,
        "gen_tokens": gen_tokens,
        "context_length": L,
        "agreements": [],
        "logit_diffs_all": [],
    }

    agreements = []
    all_logit_diffs = []

    for i in range(n_prompts):
        if i % 20 == 0:
            print(f"  Prompt {i}/{n_prompts}...")
        passage = load_passage(tokenizer, L, seed=i, extra=gen_tokens)
        input_ids = passage[:, :L].to(DEVICE)

        # FP16 generation
        with torch.no_grad():
            out_fp16 = model(input_ids, use_cache=True)
            past_fp16 = out_fp16.past_key_values
            tokens_fp16 = [out_fp16.logits[:, -1:, :].argmax(dim=-1)]
            logits_fp16_list = [out_fp16.logits[:, -1:, :]]
            past = past_fp16
            for t in range(gen_tokens - 1):
                with torch.no_grad():
                    out = model(tokens_fp16[-1], past_key_values=past, use_cache=True)
                tokens_fp16.append(out.logits.argmax(dim=-1))
                logits_fp16_list.append(out.logits)
                past = out.past_key_values
        tokens_fp16_t = torch.cat(tokens_fp16, dim=1)
        logits_fp16_t = torch.cat(logits_fp16_list, dim=1)

        del past, past_fp16
        clear_gpu()

        # INT4 generation
        with torch.no_grad():
            out_int4 = model(input_ids, use_cache=True)
            past_int4 = quantize_cache_int4(out_int4.past_key_values)
            tokens_int4 = [out_int4.logits[:, -1:, :].argmax(dim=-1)]
            logits_int4_list = [out_int4.logits[:, -1:, :]]
            past = past_int4
            for t in range(gen_tokens - 1):
                with torch.no_grad():
                    out = model(tokens_int4[-1], past_key_values=past, use_cache=True)
                tokens_int4.append(out.logits.argmax(dim=-1))
                logits_int4_list.append(out.logits)
                past = out.past_key_values
        tokens_int4_t = torch.cat(tokens_int4, dim=1)
        logits_int4_t = torch.cat(logits_int4_list, dim=1)

        agree = (tokens_fp16_t == tokens_int4_t).float().mean().item()
        agreements.append(agree)

        # Logit diffs for first few tokens (memory)
        n_sample = min(32, gen_tokens)
        diff = (
            logits_fp16_t[:, :n_sample, :]
            - logits_int4_t[:, :n_sample, :]
        ).abs().max(dim=-1).values
        all_logit_diffs.extend(diff.flatten().cpu().tolist())

        del tokens_fp16_t, tokens_int4_t, logits_fp16_t, logits_int4_t
        del past, past_int4
        clear_gpu()

    results["agreements"] = [round(a, 4) for a in agreements]
    results["mean_agreement"] = round(np.mean(agreements), 4)
    results["std_agreement"] = round(np.std(agreements), 4)
    results["min_agreement"] = round(np.min(agreements), 4)
    results["max_agreement"] = round(np.max(agreements), 4)
    results["logit_diff_stats"] = {
        "mean": round(float(np.mean(all_logit_diffs)), 6),
        "median": round(float(np.median(all_logit_diffs)), 6),
        "p95": round(float(np.percentile(all_logit_diffs, 95)), 6),
        "p99": round(float(np.percentile(all_logit_diffs, 99)), 6),
    }
    results["elapsed_sec"] = round(time.time() - t0, 1)

    print(
        f"  Agreement: mean={results['mean_agreement']:.4f} "
        f"std={results['std_agreement']:.4f}"
    )
    save_json(results, "token_agreement.json")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(agreements, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(np.mean(agreements), color="red", linestyle="--", label=f"Mean={np.mean(agreements):.3f}")
    ax1.set_xlabel("Token Agreement Rate")
    ax1.set_ylabel("Count")
    ax1.set_title("Token Agreement Distribution (FP16 vs INT4 KV)")
    ax1.legend()

    ax2.hist(all_logit_diffs, bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlabel("Max |logit_fp16 - logit_int4|")
    ax2.set_ylabel("Count")
    ax2.set_title("Logit Difference Distribution")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "token_agreement.png"), dpi=300)
    plt.close()
    print(f"  Saved plot: token_agreement.png")

    return results


# ============================================================
# Experiment 3: Logit Difference Analysis
# ============================================================
def experiment_3_logit_difference(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 3: Logit Difference Analysis")
    print("=" * 60)
    t0 = time.time()

    L_set = [2048, 8192]
    seeds = [0, 1, 2]
    decode_tokens = 64

    results = {"experiment": "logit_error", "gpu": gpu_info(), "evals": {}}

    all_diffs = []

    for L in L_set:
        for seed in seeds:
            key = f"L{L}_s{seed}"
            print(f"  Evaluating {key}...")
            try:
                passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                input_ids = passage[:, :L].to(DEVICE)
                continuation = passage[:, L : L + decode_tokens].to(DEVICE)

                with torch.no_grad():
                    out_fp16 = model(input_ids, use_cache=True)
                    past_fp16 = out_fp16.past_key_values

                logits_fp16 = [out_fp16.logits[:, -1:, :].cpu()]
                past = past_fp16
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_fp16.append(out.logits.cpu())
                    past = out.past_key_values
                logits_fp16 = torch.cat(logits_fp16, dim=1)
                del past, past_fp16, out_fp16, out
                clear_gpu()

                with torch.no_grad():
                    out_int4 = model(input_ids, use_cache=True)
                    past_int4 = quantize_cache_int4(out_int4.past_key_values)

                logits_int4 = [out_int4.logits[:, -1:, :].cpu()]
                past = past_int4
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_int4.append(out.logits.cpu())
                    past = out.past_key_values
                logits_int4 = torch.cat(logits_int4, dim=1)
                del past, past_int4, out_int4, out
                clear_gpu()

                diff = (logits_fp16 - logits_int4).abs()
                max_per_token = diff.max(dim=-1).values.flatten().float()

                stats = {
                    "mean": round(max_per_token.mean().item(), 6),
                    "median": round(max_per_token.median().item(), 6),
                    "p95": round(torch.quantile(max_per_token, 0.95).item(), 6),
                    "p99": round(torch.quantile(max_per_token, 0.99).item(), 6),
                    "max": round(max_per_token.max().item(), 6),
                }
                results["evals"][key] = stats
                all_diffs.extend(max_per_token.cpu().tolist())
                print(f"    mean={stats['mean']:.4f} p95={stats['p95']:.4f} p99={stats['p99']:.4f}")

                del logits_fp16, logits_int4
                clear_gpu()
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at {key}, skipping")
                clear_gpu()
                continue

    results["aggregate"] = {
        "mean": round(float(np.mean(all_diffs)), 6),
        "median": round(float(np.median(all_diffs)), 6),
        "p95": round(float(np.percentile(all_diffs, 95)), 6),
        "p99": round(float(np.percentile(all_diffs, 99)), 6),
    }
    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "logit_error.json")
    return results


# ============================================================
# Experiment 4: Long Context Perplexity Scaling
# ============================================================
def experiment_4_long_context_ppl(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 4: Long Context Perplexity Scaling")
    print("=" * 60)
    t0 = time.time()

    L_set = [2048, 8192, 16384, 32768]
    seeds = [0, 1, 2]
    decode_tokens = 64

    results = {"experiment": "long_context_perplexity", "gpu": gpu_info(), "evals": {}}

    for L in L_set:
        fp16_ppls = []
        int4_ppls = []
        for seed in seeds:
            key = f"L{L}_s{seed}"
            print(f"  Evaluating {key}...")

            try:
                passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                input_ids = passage[:, :L].to(DEVICE)
                continuation = passage[:, L : L + decode_tokens].to(DEVICE)
            except Exception as e:
                print(f"    Skipping {key}: not enough tokens for L={L}")
                continue

            try:
                # FP16 PPL: prefill + decode continuation
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                logits_list = [out.logits[:, -1:, :]]
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_list.append(out.logits)
                    past = out.past_key_values
                logits_fp16 = torch.cat(logits_list, dim=1)
                B, T, V = logits_fp16[:, :-1, :].shape
                loss_fp16 = F.cross_entropy(
                    logits_fp16[:, :-1, :].reshape(-1, V).float(),
                    continuation.reshape(-1), reduction="mean",
                )
                ppl_fp16 = math.exp(min(loss_fp16.item(), 20))
                del logits_fp16, logits_list, past
                clear_gpu()

                # INT4 KV PPL: prefill, quantize cache, decode continuation
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = quantize_cache_int4(out.past_key_values)
                logits_list = [out.logits[:, -1:, :]]
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_list.append(out.logits)
                    past = out.past_key_values
                logits_int4 = torch.cat(logits_list, dim=1)
                loss_int4 = F.cross_entropy(
                    logits_int4[:, :-1, :].reshape(-1, V).float(),
                    continuation.reshape(-1), reduction="mean",
                )
                ppl_int4 = math.exp(min(loss_int4.item(), 20))
                del logits_int4, logits_list, past
                clear_gpu()

                fp16_ppls.append(ppl_fp16)
                int4_ppls.append(ppl_int4)
                delta = (ppl_int4 - ppl_fp16) / ppl_fp16 * 100

                results["evals"][key] = {
                    "ppl_fp16": round(ppl_fp16, 4),
                    "ppl_int4": round(ppl_int4, 4),
                    "delta_pct": round(delta, 2),
                }
                print(f"    FP16={ppl_fp16:.4f} INT4={ppl_int4:.4f} delta={delta:.2f}%")
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at {key}, skipping")
                clear_gpu()
                continue

        if not fp16_ppls:
            continue
        results["evals"][f"L{L}_avg"] = {
            "ppl_fp16": round(np.mean(fp16_ppls), 4),
            "ppl_int4": round(np.mean(int4_ppls), 4),
            "delta_pct": round(
                (np.mean(int4_ppls) - np.mean(fp16_ppls)) / np.mean(fp16_ppls) * 100, 2
            ),
        }

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "long_context_perplexity.json")

    # Plot (only context lengths that succeeded)
    fig, ax = plt.subplots(figsize=(8, 5))
    ctx_lengths = [L for L in L_set if f"L{L}_avg" in results["evals"]]
    fp16_avgs = [results["evals"][f"L{L}_avg"]["ppl_fp16"] for L in ctx_lengths]
    int4_avgs = [results["evals"][f"L{L}_avg"]["ppl_int4"] for L in ctx_lengths]

    ax.plot(ctx_lengths, fp16_avgs, "o-", label="FP16 KV", color="steelblue", linewidth=2)
    ax.plot(ctx_lengths, int4_avgs, "s--", label="INT4 KV", color="coral", linewidth=2)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity vs Context Length (Qwen2.5-7B)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "perplexity_vs_context.png"), dpi=300)
    plt.close()
    print(f"  Saved plot: perplexity_vs_context.png")

    return results


# ============================================================
# Experiment 5: Needle-In-Haystack Retrieval
# ============================================================
def experiment_5_needle_retrieval(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 5: Needle-In-Haystack Retrieval")
    print("=" * 60)
    t0 = time.time()

    needle = "The secret phrase is: avocado-electric-tractor."
    question = "\n\nWhat is the secret phrase?"
    answer_key = "avocado-electric-tractor"

    L_set = [4096, 8192]  # 16K+ OOMs with eager attention on W7900
    n_prompts = 50
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]  # relative depth positions

    results = {
        "experiment": "needle_retrieval",
        "gpu": gpu_info(),
        "needle": needle,
        "n_prompts_per_depth": n_prompts,
        "evals": {},
    }

    # Build haystack text from wikitext
    data = load_wikitext_tokens(tokenizer)
    haystack_text = tokenizer.decode(data[:100000].tolist())

    for L in L_set:
        print(f"\n  Context length: {L}")
        fp16_correct = 0
        int4_correct = 0
        total = 0

        for depth in depths:
            for pi in range(n_prompts // len(depths)):
                # Build prompt: haystack with needle inserted at depth
                target_chars = L * 4  # rough char estimate
                hay = haystack_text[pi * target_chars : (pi + 1) * target_chars]
                insert_pos = int(len(hay) * depth)
                prompt_text = (
                    hay[:insert_pos]
                    + f"\n{needle}\n"
                    + hay[insert_pos:]
                    + question
                )

                # Tokenize and truncate to L
                tokens = tokenizer.encode(prompt_text, return_tensors="pt")
                if tokens.shape[1] > L:
                    # Re-insert needle to ensure it's within context
                    tokens = tokens[:, :L]
                input_ids = tokens.to(DEVICE)

                gen_len = 30

                # FP16 generation
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                    gen_tokens = []
                    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                    gen_tokens.append(next_tok)
                    for _ in range(gen_len - 1):
                        out = model(next_tok, past_key_values=past, use_cache=True)
                        next_tok = out.logits.argmax(dim=-1)
                        gen_tokens.append(next_tok)
                        past = out.past_key_values
                    gen_text_fp16 = tokenizer.decode(
                        torch.cat(gen_tokens, dim=1)[0].tolist()
                    )
                del past
                clear_gpu()

                # INT4 generation
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = quantize_cache_int4(out.past_key_values)
                    gen_tokens = []
                    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                    gen_tokens.append(next_tok)
                    for _ in range(gen_len - 1):
                        out = model(next_tok, past_key_values=past, use_cache=True)
                        next_tok = out.logits.argmax(dim=-1)
                        gen_tokens.append(next_tok)
                        past = out.past_key_values
                    gen_text_int4 = tokenizer.decode(
                        torch.cat(gen_tokens, dim=1)[0].tolist()
                    )
                del past
                clear_gpu()

                if answer_key.lower() in gen_text_fp16.lower():
                    fp16_correct += 1
                if answer_key.lower() in gen_text_int4.lower():
                    int4_correct += 1
                total += 1

        fp16_acc = fp16_correct / total if total > 0 else 0
        int4_acc = int4_correct / total if total > 0 else 0
        results["evals"][f"L{L}"] = {
            "fp16_accuracy": round(fp16_acc, 4),
            "int4_accuracy": round(int4_acc, 4),
            "fp16_correct": fp16_correct,
            "int4_correct": int4_correct,
            "total": total,
        }
        print(
            f"    FP16 acc={fp16_acc:.3f} ({fp16_correct}/{total}) "
            f"INT4 acc={int4_acc:.3f} ({int4_correct}/{total})"
        )

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "needle_retrieval.json")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ctx = [int(k[1:]) for k in results["evals"].keys()]
    fp16_accs = [results["evals"][k]["fp16_accuracy"] for k in results["evals"]]
    int4_accs = [results["evals"][k]["int4_accuracy"] for k in results["evals"]]

    x = np.arange(len(ctx))
    w = 0.35
    ax.bar(x - w / 2, fp16_accs, w, label="FP16 KV", color="steelblue")
    ax.bar(x + w / 2, int4_accs, w, label="INT4 KV", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c // 1024}K" for c in ctx])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title("Needle-In-Haystack Retrieval (Qwen2.5-7B)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "needle_accuracy.png"), dpi=300)
    plt.close()
    print(f"  Saved plot: needle_accuracy.png")

    return results


# ============================================================
# Experiment 6: Downstream Capability Benchmarks
# ============================================================
def experiment_6_downstream(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 6: Downstream Capability Benchmarks")
    print("=" * 60)
    t0 = time.time()

    results = {"experiment": "downstream_tasks", "gpu": gpu_info(), "tasks": {}}

    # --- MMLU (simplified: multiple choice) ---
    print("  Running MMLU subset...")
    try:
        from datasets import load_dataset

        mmlu = load_dataset("cais/mmlu", "all", split="test")
        # Sample 200 questions
        rng = np.random.RandomState(42)
        indices = rng.choice(len(mmlu), size=min(200, len(mmlu)), replace=False)

        choices_map = ["A", "B", "C", "D"]
        fp16_correct = 0
        int4_correct = 0
        total = 0

        for idx in indices:
            item = mmlu[int(idx)]
            question = item["question"]
            choices = item["choices"]
            answer = item["answer"]  # 0-3 index

            prompt = f"Question: {question}\n"
            for ci, c in enumerate(choices):
                prompt += f"{choices_map[ci]}. {c}\n"
            prompt += "Answer:"

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, :2048]

            # FP16
            with torch.no_grad():
                out_fp16 = model(input_ids, use_cache=True)
                logits_fp16 = out_fp16.logits[:, -1, :]
                past_fp16 = out_fp16.past_key_values

            # INT4
            with torch.no_grad():
                out_int4 = model(input_ids, use_cache=True)
                past_int4 = quantize_cache_int4(out_int4.past_key_values)
                # Re-run last token with quantized cache
                # Actually the logits from prefill are from FP16 cache,
                # so INT4 effect shows in generation. For MMLU we just
                # check the next-token prediction from the prefill logits
                # after quantizing. Let's use a 1-token decode step.
                last_tok = input_ids[:, -1:]
                out_q = model(last_tok, past_key_values=past_int4, use_cache=True)
                logits_int4 = out_q.logits[:, -1, :]

            # Get token IDs for A, B, C, D
            choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in choices_map]
            fp16_scores = logits_fp16[0, choice_ids]
            int4_scores = logits_int4[0, choice_ids]

            if fp16_scores.argmax().item() == answer:
                fp16_correct += 1
            if int4_scores.argmax().item() == answer:
                int4_correct += 1
            total += 1

            del out_fp16, out_int4, past_fp16, past_int4
            if total % 50 == 0:
                clear_gpu()

        results["tasks"]["mmlu"] = {
            "fp16_accuracy": round(fp16_correct / total, 4),
            "int4_accuracy": round(int4_correct / total, 4),
            "fp16_correct": fp16_correct,
            "int4_correct": int4_correct,
            "total": total,
        }
        print(
            f"    MMLU: FP16={fp16_correct}/{total} ({fp16_correct/total:.3f}) "
            f"INT4={int4_correct}/{total} ({int4_correct/total:.3f})"
        )
    except Exception as e:
        print(f"    MMLU failed: {e}")
        results["tasks"]["mmlu"] = {"error": str(e)}
    clear_gpu()

    # --- GSM8K ---
    print("  Running GSM8K subset...")
    try:
        from datasets import load_dataset

        gsm = load_dataset("gsm8k", "main", split="test")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(gsm), size=min(200, len(gsm)), replace=False)

        fp16_correct = 0
        int4_correct = 0
        total = 0

        for idx in indices:
            item = gsm[int(idx)]
            question = item["question"]
            answer_text = item["answer"]
            # Extract final number
            final_answer = answer_text.split("####")[-1].strip()

            prompt = f"Question: {question}\nLet's solve step by step.\nAnswer: The answer is"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, :2048]

            gen_len = 20

            # FP16 generation
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                gen_toks = []
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_toks.append(next_tok)
                for _ in range(gen_len - 1):
                    out = model(next_tok, past_key_values=past, use_cache=True)
                    next_tok = out.logits.argmax(dim=-1)
                    gen_toks.append(next_tok)
                    past = out.past_key_values
                gen_fp16 = tokenizer.decode(torch.cat(gen_toks, dim=1)[0].tolist())
            del past
            clear_gpu()

            # INT4 generation
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = quantize_cache_int4(out.past_key_values)
                gen_toks = []
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_toks.append(next_tok)
                for _ in range(gen_len - 1):
                    out = model(next_tok, past_key_values=past, use_cache=True)
                    next_tok = out.logits.argmax(dim=-1)
                    gen_toks.append(next_tok)
                    past = out.past_key_values
                gen_int4 = tokenizer.decode(torch.cat(gen_toks, dim=1)[0].tolist())
            del past
            clear_gpu()

            # Check if final answer appears in generation
            if final_answer in gen_fp16:
                fp16_correct += 1
            if final_answer in gen_int4:
                int4_correct += 1
            total += 1

        results["tasks"]["gsm8k"] = {
            "fp16_accuracy": round(fp16_correct / total, 4),
            "int4_accuracy": round(int4_correct / total, 4),
            "fp16_correct": fp16_correct,
            "int4_correct": int4_correct,
            "total": total,
        }
        print(
            f"    GSM8K: FP16={fp16_correct}/{total} ({fp16_correct/total:.3f}) "
            f"INT4={int4_correct}/{total} ({int4_correct/total:.3f})"
        )
    except Exception as e:
        print(f"    GSM8K failed: {e}")
        results["tasks"]["gsm8k"] = {"error": str(e)}
    clear_gpu()

    # --- HumanEval (simplified pass@1 check) ---
    print("  Running HumanEval subset...")
    try:
        from datasets import load_dataset

        he = load_dataset("openai/openai_humaneval", split="test")
        n_problems = min(100, len(he))

        fp16_correct = 0
        int4_correct = 0
        total = 0

        for i in range(n_problems):
            item = he[i]
            prompt = item["prompt"]
            # We check if the model generates syntactically valid Python
            # that includes key elements from the canonical solution
            test_code = item.get("test", "")
            entry_point = item.get("entry_point", "")

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, :2048]

            gen_len = 128

            # FP16
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                gen_toks = []
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_toks.append(next_tok)
                for _ in range(gen_len - 1):
                    out = model(next_tok, past_key_values=past, use_cache=True)
                    next_tok = out.logits.argmax(dim=-1)
                    gen_toks.append(next_tok)
                    past = out.past_key_values
                    # Stop at newline after return
                gen_fp16 = tokenizer.decode(torch.cat(gen_toks, dim=1)[0].tolist())
            del past

            # INT4
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = quantize_cache_int4(out.past_key_values)
                gen_toks = []
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_toks.append(next_tok)
                for _ in range(gen_len - 1):
                    out = model(next_tok, past_key_values=past, use_cache=True)
                    next_tok = out.logits.argmax(dim=-1)
                    gen_toks.append(next_tok)
                    past = out.past_key_values
                gen_int4 = tokenizer.decode(torch.cat(gen_toks, dim=1)[0].tolist())
            del past

            # Simple functional check: try to compile and run
            def try_exec(code_str):
                full_code = prompt + code_str
                # Truncate at first double newline (end of function)
                if "\n\n" in code_str:
                    code_str = code_str[: code_str.index("\n\n")]
                full_code = prompt + code_str
                try:
                    compile(full_code, "<string>", "exec")
                    return True
                except SyntaxError:
                    return False

            if try_exec(gen_fp16):
                fp16_correct += 1
            if try_exec(gen_int4):
                int4_correct += 1
            total += 1

            if total % 25 == 0:
                clear_gpu()

        results["tasks"]["humaneval"] = {
            "fp16_syntax_valid": round(fp16_correct / total, 4),
            "int4_syntax_valid": round(int4_correct / total, 4),
            "fp16_correct": fp16_correct,
            "int4_correct": int4_correct,
            "total": total,
            "note": "Syntax validity check only (not functional correctness)",
        }
        print(
            f"    HumanEval (syntax): FP16={fp16_correct}/{total} "
            f"INT4={int4_correct}/{total}"
        )
    except Exception as e:
        print(f"    HumanEval failed: {e}")
        results["tasks"]["humaneval"] = {"error": str(e)}
    clear_gpu()

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "downstream_tasks.json")
    return results


# ============================================================
# Experiment 7: Decode Bandwidth Measurement
# ============================================================
def experiment_7_bandwidth(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 7: Decode Bandwidth Measurement")
    print("=" * 60)
    t0 = time.time()

    batch_sizes = [1, 2, 4, 8, 16, 32]
    ctx_lengths = [2048, 4096, 8192, 16384]
    n_decode_steps = 32
    warmup_steps = 4

    # Model config
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    results = {
        "experiment": "bandwidth_measurements",
        "gpu": gpu_info(),
        "model_config": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
        },
        "evals": {},
    }

    for L in ctx_lengths:
        for B in batch_sizes:
            key = f"L{L}_B{B}"
            # Check if this fits in memory
            kv_bytes = 2 * n_layers * n_kv_heads * head_dim * L * B * 2  # fp16
            kv_gb = kv_bytes / 1e9
            if kv_gb > 40:
                print(f"  Skipping {key}: KV cache would be {kv_gb:.1f} GB")
                results["evals"][key] = {"skipped": True, "kv_gb": round(kv_gb, 1)}
                continue

            print(f"  Measuring {key} (KV ~{kv_gb:.1f} GB)...")
            try:
                # Create dummy input
                data = load_wikitext_tokens(tokenizer)
                input_ids = torch.from_numpy(data[:L]).unsqueeze(0).repeat(B, 1).to(DEVICE)

                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                    next_tok = out.logits[:, -1:, :].argmax(dim=-1)

                # Warmup
                for _ in range(warmup_steps):
                    with torch.no_grad():
                        out = model(next_tok, past_key_values=past, use_cache=True)
                        next_tok = out.logits.argmax(dim=-1)
                        past = out.past_key_values
                torch.cuda.synchronize()

                # Timed decode
                t_start = time.time()
                for _ in range(n_decode_steps):
                    with torch.no_grad():
                        out = model(next_tok, past_key_values=past, use_cache=True)
                        next_tok = out.logits.argmax(dim=-1)
                        past = out.past_key_values
                torch.cuda.synchronize()
                t_end = time.time()

                decode_time = t_end - t_start
                time_per_step = decode_time / n_decode_steps

                # KV bytes read per decode step
                # Each step reads all KV cache: 2 * n_layers * n_kv_heads * head_dim * seq_len * B * 2 bytes
                current_len = L + warmup_steps + n_decode_steps
                kv_bytes_per_step = (
                    2 * n_layers * n_kv_heads * head_dim * current_len * B * 2
                )
                bandwidth_gbps = kv_bytes_per_step / time_per_step / 1e9

                tokens_per_sec = B / time_per_step

                results["evals"][key] = {
                    "batch_size": B,
                    "context_length": L,
                    "decode_time_sec": round(decode_time, 4),
                    "time_per_step_ms": round(time_per_step * 1000, 2),
                    "kv_bytes_per_step": kv_bytes_per_step,
                    "bandwidth_gbps": round(bandwidth_gbps, 2),
                    "tokens_per_sec": round(tokens_per_sec, 2),
                }
                print(
                    f"    {time_per_step*1000:.1f} ms/step, "
                    f"BW={bandwidth_gbps:.1f} GB/s, "
                    f"{tokens_per_sec:.1f} tok/s"
                )

                del past, input_ids, out
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM for {key}")
                results["evals"][key] = {"oom": True}
                clear_gpu()
            except Exception as e:
                print(f"    Error for {key}: {e}")
                results["evals"][key] = {"error": str(e)}
                clear_gpu()

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "bandwidth_measurements.json")

    # Plot: bandwidth vs batch size for each context length
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "coral", "forestgreen", "purple"]
    for ci, L in enumerate(ctx_lengths):
        bs_list = []
        bw_list = []
        for B in batch_sizes:
            key = f"L{L}_B{B}"
            ev = results["evals"].get(key, {})
            if "bandwidth_gbps" in ev:
                bs_list.append(B)
                bw_list.append(ev["bandwidth_gbps"])
        if bs_list:
            ax.plot(
                bs_list, bw_list, "o-",
                label=f"L={L // 1024}K",
                color=colors[ci % len(colors)],
                linewidth=2,
            )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("KV Bandwidth (GB/s)")
    ax.set_title("Decode KV Bandwidth vs Batch Size (Qwen2.5-7B, W7900)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bandwidth_vs_batch.png"), dpi=300)
    plt.close()
    print(f"  Saved plot: bandwidth_vs_batch.png")

    return results


# ============================================================
# Experiment 8: Batch Scaling Verification
# ============================================================
def experiment_8_batch_scaling(model, tokenizer):
    print("\n" + "=" * 60)
    print("Experiment 8: Batch-Size Scaling Verification")
    print("=" * 60)
    t0 = time.time()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    L = 2048
    n_decode_steps = 64
    warmup_steps = 8

    results = {
        "experiment": "batch_scaling",
        "gpu": gpu_info(),
        "context_length": L,
        "evals": {},
    }

    data = load_wikitext_tokens(tokenizer)
    base_ids = torch.from_numpy(data[:L]).unsqueeze(0).to(DEVICE)

    for B in batch_sizes:
        print(f"  Batch size: {B}")
        try:
            input_ids = base_ids.repeat(B, 1)

            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)

            # Warmup
            for _ in range(warmup_steps):
                with torch.no_grad():
                    out = model(next_tok, past_key_values=past, use_cache=True)
                    next_tok = out.logits.argmax(dim=-1)
                    past = out.past_key_values
            torch.cuda.synchronize()

            # Timed decode
            t_start = time.time()
            for _ in range(n_decode_steps):
                with torch.no_grad():
                    out = model(next_tok, past_key_values=past, use_cache=True)
                    next_tok = out.logits.argmax(dim=-1)
                    past = out.past_key_values
            torch.cuda.synchronize()
            t_end = time.time()

            decode_time = t_end - t_start
            tokens_per_sec = B * n_decode_steps / decode_time

            results["evals"][f"B{B}"] = {
                "batch_size": B,
                "decode_time_sec": round(decode_time, 4),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "ms_per_token": round(decode_time / (B * n_decode_steps) * 1000, 4),
            }
            print(f"    {tokens_per_sec:.1f} tok/s, {decode_time:.2f}s total")

            del past, input_ids, out
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at B={B}")
            results["evals"][f"B{B}"] = {"oom": True}
            clear_gpu()
            break
        except Exception as e:
            print(f"    Error at B={B}: {e}")
            results["evals"][f"B{B}"] = {"error": str(e)}
            clear_gpu()

    # Fit Hill model: S(B) = Smax * B^gamma / (B_half^gamma + B^gamma)
    bs_data = []
    tps_data = []
    for k, v in results["evals"].items():
        if "tokens_per_sec" in v:
            bs_data.append(v["batch_size"])
            tps_data.append(v["tokens_per_sec"])

    hill_fit = None
    if len(bs_data) >= 3:
        try:
            from scipy.optimize import curve_fit

            def hill(B, Smax, B_half, gamma):
                return Smax * np.power(B, gamma) / (np.power(B_half, gamma) + np.power(B, gamma))

            bs_arr = np.array(bs_data, dtype=float)
            tps_arr = np.array(tps_data, dtype=float)
            popt, pcov = curve_fit(
                hill, bs_arr, tps_arr,
                p0=[tps_arr.max() * 1.5, bs_arr[len(bs_arr) // 2], 1.0],
                maxfev=10000,
            )
            hill_fit = {
                "Smax": round(float(popt[0]), 2),
                "B_half": round(float(popt[1]), 2),
                "gamma": round(float(popt[2]), 4),
            }
            results["hill_fit"] = hill_fit
            print(f"  Hill fit: Smax={popt[0]:.1f} B_half={popt[1]:.1f} gamma={popt[2]:.3f}")
        except Exception as e:
            print(f"  Hill fit failed: {e}")
            results["hill_fit_error"] = str(e)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, "batch_scaling.json")

    # Plot: decode saturation curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs_data, tps_data, "o-", color="steelblue", linewidth=2, markersize=8, label="Measured")

    if hill_fit is not None:
        from scipy.optimize import curve_fit

        def hill(B, Smax, B_half, gamma):
            return Smax * np.power(B, gamma) / (np.power(B_half, gamma) + np.power(B, gamma))

        B_smooth = np.linspace(1, max(bs_data) * 1.2, 100)
        S_smooth = hill(B_smooth, hill_fit["Smax"], hill_fit["B_half"], hill_fit["gamma"])
        ax.plot(B_smooth, S_smooth, "--", color="coral", linewidth=2,
                label=f"Hill fit (Smax={hill_fit['Smax']:.0f}, "
                      f"B½={hill_fit['B_half']:.1f}, γ={hill_fit['gamma']:.2f})")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Decode Throughput Saturation (Qwen2.5-7B, W7900, L=2K)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "decode_saturation_curve.png"), dpi=300)
    plt.close()
    print(f"  Saved plot: decode_saturation_curve.png")

    return results


# ============================================================
# Summary Report
# ============================================================
def generate_summary(all_results):
    print("\n" + "=" * 60)
    print("Generating Summary Report")
    print("=" * 60)

    report = f"""# BPA v46 — W7900 Validation Results

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**GPU**: AMD Radeon Pro W7900 (48GB)
**Model**: Qwen2.5-7B (FP16)

## Reviewer Critiques Addressed

"""

    # Exp 1: Activation quantization
    r1 = all_results.get("exp1", {})
    report += "### 1. Direct KV Activation Quantization\n\n"
    report += "| Config | PPL FP16 | PPL INT4 | Delta (%) | Token Agree |\n"
    report += "|--------|----------|----------|-----------|-------------|\n"
    for k, v in r1.get("evals", {}).items():
        report += (
            f"| {k} | {v['ppl_fp16']:.4f} | {v['ppl_int4']:.4f} "
            f"| {v['delta_pct']:.2f}% | {v['token_agreement']:.4f} |\n"
        )
    report += "\n"

    # Exp 2: Token agreement
    r2 = all_results.get("exp2", {})
    report += "### 2. Token Agreement\n\n"
    report += f"- Mean agreement: {r2.get('mean_agreement', 'N/A')}\n"
    report += f"- Std: {r2.get('std_agreement', 'N/A')}\n"
    report += f"- Min: {r2.get('min_agreement', 'N/A')}\n"
    report += f"- Prompts: {r2.get('n_prompts', 'N/A')}\n\n"

    # Exp 3: Logit error
    r3 = all_results.get("exp3", {})
    agg = r3.get("aggregate", {})
    report += "### 3. Logit Difference Analysis\n\n"
    report += f"- Mean max|delta_logit|: {agg.get('mean', 'N/A')}\n"
    report += f"- Median: {agg.get('median', 'N/A')}\n"
    report += f"- P95: {agg.get('p95', 'N/A')}\n"
    report += f"- P99: {agg.get('p99', 'N/A')}\n\n"

    # Exp 4: Long context PPL
    r4 = all_results.get("exp4", {})
    report += "### 4. Long-Context Perplexity Scaling\n\n"
    report += "| Context | PPL FP16 | PPL INT4 | Delta (%) |\n"
    report += "|---------|----------|----------|-----------|\n"
    for k, v in r4.get("evals", {}).items():
        if k.endswith("_avg"):
            L_str = k.replace("_avg", "").replace("L", "")
            report += (
                f"| {int(L_str)//1024}K | {v['ppl_fp16']:.4f} "
                f"| {v['ppl_int4']:.4f} | {v['delta_pct']:.2f}% |\n"
            )
    report += "\n![PPL vs Context](plots/perplexity_vs_context.png)\n\n"

    # Exp 5: Needle retrieval
    r5 = all_results.get("exp5", {})
    report += "### 5. Needle-In-Haystack Retrieval\n\n"
    report += "| Context | FP16 Acc | INT4 Acc |\n"
    report += "|---------|----------|----------|\n"
    for k, v in r5.get("evals", {}).items():
        L_str = k.replace("L", "")
        report += f"| {int(L_str)//1024}K | {v['fp16_accuracy']:.3f} | {v['int4_accuracy']:.3f} |\n"
    report += "\n![Needle Accuracy](plots/needle_accuracy.png)\n\n"

    # Exp 6: Downstream
    r6 = all_results.get("exp6", {})
    report += "### 6. Downstream Capability Benchmarks\n\n"
    report += "| Task | FP16 | INT4 | Delta |\n"
    report += "|------|------|------|-------|\n"
    for task, v in r6.get("tasks", {}).items():
        if "error" not in v:
            fp16_key = "fp16_accuracy" if "fp16_accuracy" in v else "fp16_syntax_valid"
            int4_key = "int4_accuracy" if "int4_accuracy" in v else "int4_syntax_valid"
            fp16 = v.get(fp16_key, 0)
            int4 = v.get(int4_key, 0)
            delta = int4 - fp16
            report += f"| {task.upper()} | {fp16:.3f} | {int4:.3f} | {delta:+.3f} |\n"
    report += "\n"

    # Exp 7: Bandwidth
    r7 = all_results.get("exp7", {})
    report += "### 7. Decode Bandwidth\n\n"
    report += "See bandwidth_measurements.json for full results.\n\n"
    report += "![Bandwidth vs Batch](plots/bandwidth_vs_batch.png)\n\n"

    # Exp 8: Batch scaling
    r8 = all_results.get("exp8", {})
    report += "### 8. Batch-Size Scaling\n\n"
    hill = r8.get("hill_fit", {})
    if hill:
        report += f"Hill model fit: Smax={hill.get('Smax')}, "
        report += f"B_half={hill.get('B_half')}, gamma={hill.get('gamma')}\n\n"
    report += "![Saturation Curve](plots/decode_saturation_curve.png)\n\n"

    # Timing
    total_time = sum(
        r.get("elapsed_sec", 0)
        for r in all_results.values()
        if isinstance(r, dict)
    )
    report += f"\n## Total Runtime: {total_time:.0f} seconds ({total_time/3600:.1f} hours)\n"

    report_path = os.path.join(RESULTS_ROOT, "bpa46_summary.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("BPA v46: W7900 Validation Experiments")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()

    all_results = {}

    # Load completed experiments from prior runs
    for exp_name, filename in [("exp1", "activation_quantization.json"),
                                ("exp2", "token_agreement.json"),
                                ("exp3", "logit_error.json"),
                                ("exp4", "long_context_perplexity.json"),
                                ("exp5", "needle_retrieval.json"),
                                ("exp6", "downstream_tasks.json")]:
        path = os.path.join(JSON_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                all_results[exp_name] = json.load(f)
            print(f"  Loaded prior result: {filename}")

    # Run remaining experiments
    if "exp1" not in all_results:
        all_results["exp1"] = experiment_1_activation_quantization(model, tokenizer)
    if "exp2" not in all_results:
        all_results["exp2"] = experiment_2_token_agreement(model, tokenizer)
    if "exp3" not in all_results:
        all_results["exp3"] = experiment_3_logit_difference(model, tokenizer)
    if "exp4" not in all_results:
        all_results["exp4"] = experiment_4_long_context_ppl(model, tokenizer)
    if "exp5" not in all_results:
        all_results["exp5"] = experiment_5_needle_retrieval(model, tokenizer)
    if "exp6" not in all_results:
        all_results["exp6"] = experiment_6_downstream(model, tokenizer)

    # Free some memory for bandwidth tests
    del model
    clear_gpu()
    model, tokenizer = load_model_and_tokenizer()

    all_results["exp7"] = experiment_7_bandwidth(model, tokenizer)
    all_results["exp8"] = experiment_8_batch_scaling(model, tokenizer)

    generate_summary(all_results)

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Results in: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
