#!/usr/bin/env python3
"""BPA v49: Key Precision Threshold Sweep for KV Bandwidth Scaling.

Determines the minimum viable key precision while values stay at INT4.
Sweeps K through FP16, INT8, INT6, INT5, INT4 with V fixed at INT4.

Results saved to /data/knlp-key-results/bpa49/
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Paths ===
RESULTS_ROOT = "/data/knlp-key-results/bpa49"
JSON_DIR = os.path.join(RESULTS_ROOT, "json")
PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
for d in [JSON_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# === Protocol parameters ===
DATASET = "wikitext-103-raw-v1"
W_SINK = 4
GROUP_SIZE = 32
DEVICE = "cuda"
DTYPE = torch.bfloat16

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


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
    if key in _TOKEN_CACHE:
        return _TOKEN_CACHE[key]
    from datasets import load_dataset

    ds = load_dataset("wikitext", DATASET, split="test")
    text = "\n".join([x["text"] for x in ds if x["text"].strip()])
    tokens = tokenizer.encode(text)
    _TOKEN_CACHE[key] = tokens
    print(f"  Loaded {DATASET}: {len(tokens)} tokens")
    return tokens


def load_passage(tokenizer, length, seed, extra=0):
    tokens = load_wikitext_tokens(tokenizer)
    rng = np.random.RandomState(seed)
    total_need = length + extra
    start = rng.randint(0, max(1, len(tokens) - total_need))
    passage = tokens[start : start + total_need]
    return torch.tensor(passage, dtype=torch.long).unsqueeze(0)


# === Quantization functions ===
def quantize_intN_grouped(tensor, n_bits, group_size=32):
    """Generic symmetric grouped quantization for any bit width."""
    qmax = (1 << (n_bits - 1)) - 1
    qmin = -(1 << (n_bits - 1))
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
    s = amax / qmax
    q = (r / s).round().clamp(qmin, qmax)
    return (q * s).reshape(*shape[:-1], pd)[..., :hd]


def quantize_tensor(tensor, quant_type, group_size=32):
    if quant_type is None or quant_type == "fp16":
        return tensor
    bits_map = {"int4": 4, "int5": 5, "int6": 6, "int8": 8}
    if quant_type in bits_map:
        return quantize_intN_grouped(tensor, bits_map[quant_type], group_size)
    raise ValueError(f"Unknown quant type: {quant_type}")


# === Cache helpers ===
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


def n_cache_layers(past):
    if hasattr(past, "layers"):
        return len(past.layers)
    return len(past)


def quantize_cache_uniform(past, k_type, v_type, group_size=32):
    """Quantize all layers with same K and V precision."""
    D = n_cache_layers(past)
    for li in range(D):
        k, v = _cache_get_kv(past, li)
        k_sink = k[:, :, :W_SINK, :]
        v_sink = v[:, :, :W_SINK, :]
        k_far = k[:, :, W_SINK:, :]
        v_far = v[:, :, W_SINK:, :]
        if k_type is not None and k_type != "fp16":
            k_far = quantize_tensor(k_far, k_type, group_size)
        if v_type is not None and v_type != "fp16":
            v_far = quantize_tensor(v_far, v_type, group_size)
        _cache_set_kv(
            past,
            li,
            torch.cat([k_sink, k_far], dim=2),
            torch.cat([v_sink, v_far], dim=2),
        )
    return past


# === Model loading ===
def load_model_and_tokenizer(model_key):
    model_name = MODELS[model_key]
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Model loaded: {n_layers} layers, device={DEVICE}")
    return model, tokenizer, n_layers


# === Configurations ===
CONFIGS = [
    {"name": "K_FP16__V_INT4", "k_type": "fp16", "v_type": "int4"},
    {"name": "K_INT8__V_INT4", "k_type": "int8", "v_type": "int4"},
    {"name": "K_INT6__V_INT4", "k_type": "int6", "v_type": "int4"},
    {"name": "K_INT5__V_INT4", "k_type": "int5", "v_type": "int4"},
    {"name": "K_INT4__V_INT4", "k_type": "int4", "v_type": "int4"},
]

BASELINE_CONFIGS = [
    {"name": "FP16_baseline", "k_type": "fp16", "v_type": "fp16"},
    {"name": "INT8_uniform", "k_type": "int8", "v_type": "int8"},
]


def dtype_bytes(quant_type):
    """Bytes per element for a given quant type."""
    m = {"fp16": 2.0, "int8": 1.0, "int6": 0.75, "int5": 0.625, "int4": 0.5}
    return m.get(quant_type, 2.0)


# ============================================================
# Early Collapse Detection
# ============================================================
def early_collapse_check(model, tokenizer, config):
    """Quick check if a config collapses before running full eval."""
    name = config["name"]
    k_type = config["k_type"]
    v_type = config["v_type"]
    print(f"\n  Collapse check: {name}")

    L = 512
    gen_tokens = 64
    seeds = [0, 1, 2]
    ppls = []
    agrees = []
    logit_errs = []

    for seed in seeds:
        try:
            passage = load_passage(tokenizer, L, seed, extra=gen_tokens)
            input_ids = passage[:, :L].to(DEVICE)
            continuation = passage[:, L : L + gen_tokens].to(DEVICE)

            # FP16 baseline
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
            logits_fp = [out_fp.logits[:, -1:, :].cpu()]
            past = past_fp
            for t in range(gen_tokens):
                tok = continuation[:, t : t + 1]
                with torch.no_grad():
                    out = model(tok, past_key_values=past, use_cache=True)
                logits_fp.append(out.logits.cpu())
                past = out.past_key_values
            logits_fp = torch.cat(logits_fp, dim=1)
            del past, past_fp
            clear_gpu()

            # Config
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)
            logits_q = [out_q.logits[:, -1:, :].cpu()]
            past = past_q
            for t in range(gen_tokens):
                tok = continuation[:, t : t + 1]
                with torch.no_grad():
                    out = model(tok, past_key_values=past, use_cache=True)
                logits_q.append(out.logits.cpu())
                past = out.past_key_values
            logits_q = torch.cat(logits_q, dim=1)
            del past, past_q
            clear_gpu()

            # Metrics
            targets = continuation.cpu().reshape(-1)
            _, _, V_dim = logits_fp[:, :-1, :].shape
            loss_q = F.cross_entropy(
                logits_q[:, :-1, :].reshape(-1, V_dim).float(), targets
            )
            ppls.append(math.exp(min(loss_q.item(), 20)))

            pred_fp = logits_fp[:, :-1, :].argmax(dim=-1)
            pred_q = logits_q[:, :-1, :].argmax(dim=-1)
            agrees.append((pred_fp == pred_q).float().mean().item())

            logit_err = (
                (logits_fp[:, :-1, :].float() - logits_q[:, :-1, :].float())
                .abs()
                .max()
                .item()
            )
            logit_errs.append(logit_err)

            del logits_fp, logits_q
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            continue

    if not ppls:
        print(f"    COLLAPSED (OOM)")
        return True, {"name": name, "reason": "OOM"}

    mean_ppl = np.mean(ppls)
    mean_agree = np.mean(agrees)
    mean_logit_err = np.mean(logit_errs)

    collapsed = mean_agree < 0.40 or mean_ppl > 50 or mean_logit_err > 5
    status = "COLLAPSED" if collapsed else "OK"
    print(
        f"    {status}: PPL={mean_ppl:.2f} agree={mean_agree:.4f} "
        f"logit_err={mean_logit_err:.2f}"
    )

    return collapsed, {
        "name": name,
        "ppl": round(mean_ppl, 4),
        "token_agreement": round(mean_agree, 4),
        "logit_error": round(mean_logit_err, 4),
        "collapsed": collapsed,
    }


# ============================================================
# Experiment 1: Perplexity Evaluation
# ============================================================
def experiment_1_perplexity(model, tokenizer, model_key, config):
    name = config["name"]
    k_type = config["k_type"]
    v_type = config["v_type"]
    print(f"\n  PPL eval: {name}")

    L = 2048
    seeds = [0, 1, 2, 3, 4]
    decode_tokens = 64
    ppls_fp = []
    ppls_q = []

    for seed in seeds:
        try:
            passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
            input_ids = passage[:, :L].to(DEVICE)
            continuation = passage[:, L : L + decode_tokens].to(DEVICE)

            # FP16
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
            logits_fp = [out_fp.logits[:, -1:, :].cpu()]
            past = past_fp
            for t in range(decode_tokens):
                tok = continuation[:, t : t + 1]
                with torch.no_grad():
                    out = model(tok, past_key_values=past, use_cache=True)
                logits_fp.append(out.logits.cpu())
                past = out.past_key_values
            logits_fp = torch.cat(logits_fp, dim=1)
            del past, past_fp
            clear_gpu()

            # Config
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)
            logits_q = [out_q.logits[:, -1:, :].cpu()]
            past = past_q
            for t in range(decode_tokens):
                tok = continuation[:, t : t + 1]
                with torch.no_grad():
                    out = model(tok, past_key_values=past, use_cache=True)
                logits_q.append(out.logits.cpu())
                past = out.past_key_values
            logits_q = torch.cat(logits_q, dim=1)
            del past, past_q
            clear_gpu()

            targets = continuation.cpu().reshape(-1)
            _, _, V_dim = logits_fp[:, :-1, :].shape
            loss_fp = F.cross_entropy(
                logits_fp[:, :-1, :].reshape(-1, V_dim).float(), targets
            )
            loss_q = F.cross_entropy(
                logits_q[:, :-1, :].reshape(-1, V_dim).float(), targets
            )
            ppls_fp.append(math.exp(min(loss_fp.item(), 20)))
            ppls_q.append(math.exp(min(loss_q.item(), 20)))

            del logits_fp, logits_q
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            continue

    if ppls_fp:
        result = {
            "ppl_fp16": round(np.mean(ppls_fp), 4),
            "ppl_config": round(np.mean(ppls_q), 4),
            "delta_pct": round(
                (np.mean(ppls_q) - np.mean(ppls_fp)) / np.mean(ppls_fp) * 100, 4
            ),
            "n_seeds": len(ppls_fp),
        }
        print(
            f"    PPL: FP16={result['ppl_fp16']:.4f} "
            f"Config={result['ppl_config']:.4f} "
            f"delta={result['delta_pct']:.2f}%"
        )
        return result
    return None


# ============================================================
# Experiment 2: Token Agreement (200 prompts x 256 tokens)
# ============================================================
def experiment_2_token_agreement(model, tokenizer, model_key, config):
    name = config["name"]
    k_type = config["k_type"]
    v_type = config["v_type"]
    print(f"\n  Token agreement: {name}")

    L = 512
    gen_tokens = 256
    n_prompts = 200
    agrees = []

    for pi in range(n_prompts):
        if pi % 50 == 0:
            print(f"    Prompt {pi}/{n_prompts}...")
        try:
            passage = load_passage(tokenizer, L, seed=pi, extra=gen_tokens)
            input_ids = passage[:, :L].to(DEVICE)

            # FP16 generation
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past = out_fp.past_key_values
            gen_fp = [out_fp.logits[0, -1, :].argmax().item()]
            for _ in range(gen_tokens - 1):
                with torch.no_grad():
                    out = model(
                        torch.tensor([[gen_fp[-1]]]).to(DEVICE),
                        past_key_values=past,
                        use_cache=True,
                    )
                gen_fp.append(out.logits[0, -1, :].argmax().item())
                past = out.past_key_values
            del past
            clear_gpu()

            # Config generation
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)
            gen_q = [out_q.logits[0, -1, :].argmax().item()]
            past = past_q
            for _ in range(gen_tokens - 1):
                with torch.no_grad():
                    out = model(
                        torch.tensor([[gen_q[-1]]]).to(DEVICE),
                        past_key_values=past,
                        use_cache=True,
                    )
                gen_q.append(out.logits[0, -1, :].argmax().item())
                past = out.past_key_values
            del past
            clear_gpu()

            agree = sum(a == b for a, b in zip(gen_fp, gen_q)) / len(gen_fp)
            agrees.append(agree)

        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            continue

    if agrees:
        result = {
            "mean": round(np.mean(agrees), 4),
            "std": round(np.std(agrees), 4),
            "min": round(np.min(agrees), 4),
            "max": round(np.max(agrees), 4),
            "n_prompts": len(agrees),
        }
        print(f"    Agreement: mean={result['mean']:.4f} std={result['std']:.4f}")
        return result
    return None


# ============================================================
# Experiment 3: Logit Error Statistics
# ============================================================
def experiment_3_logit_error(model, tokenizer, model_key, config):
    name = config["name"]
    k_type = config["k_type"]
    v_type = config["v_type"]
    print(f"\n  Logit error: {name}")

    L = 2048
    seeds = list(range(10))
    all_max_errs = []
    all_mean_errs = []

    for seed in seeds:
        try:
            passage = load_passage(tokenizer, L, seed, extra=64)
            input_ids = passage[:, :L].to(DEVICE)
            continuation = passage[:, L : L + 64].to(DEVICE)

            # FP16
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
            logits_fp = [out_fp.logits[:, -1:, :].cpu()]
            past = past_fp
            for t in range(64):
                tok = continuation[:, t : t + 1]
                with torch.no_grad():
                    out = model(tok, past_key_values=past, use_cache=True)
                logits_fp.append(out.logits.cpu())
                past = out.past_key_values
            logits_fp = torch.cat(logits_fp, dim=1).float()
            del past, past_fp
            clear_gpu()

            # Config
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)
            logits_q = [out_q.logits[:, -1:, :].cpu()]
            past = past_q
            for t in range(64):
                tok = continuation[:, t : t + 1]
                with torch.no_grad():
                    out = model(tok, past_key_values=past, use_cache=True)
                logits_q.append(out.logits.cpu())
                past = out.past_key_values
            logits_q = torch.cat(logits_q, dim=1).float()
            del past, past_q
            clear_gpu()

            diff = (logits_fp - logits_q).abs()
            all_max_errs.append(diff.max().item())
            all_mean_errs.append(diff.mean().item())

            del logits_fp, logits_q
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            continue

    if all_max_errs:
        sorted_max = sorted(all_max_errs)
        n = len(sorted_max)
        result = {
            "max_mean": round(np.mean(all_max_errs), 4),
            "max_median": round(np.median(all_max_errs), 4),
            "max_p95": round(
                sorted_max[int(n * 0.95)] if n >= 2 else sorted_max[-1], 4
            ),
            "max_p99": round(sorted_max[min(int(n * 0.99), n - 1)], 4),
            "mean_logit_err": round(np.mean(all_mean_errs), 4),
            "n_seeds": n,
        }
        print(
            f"    Max logit err: mean={result['max_mean']:.2f} "
            f"p95={result['max_p95']:.2f} "
            f"mean_err={result['mean_logit_err']:.4f}"
        )
        return result
    return None


# ============================================================
# Experiment 4: Needle-in-Haystack Retrieval
# ============================================================
def experiment_4_needle(model, tokenizer, model_key, config):
    name = config["name"]
    k_type = config["k_type"]
    v_type = config["v_type"]
    print(f"\n  Needle retrieval: {name}")

    L = 4096
    n_prompts = 50
    needle = "The secret phrase is: avocado-electric-tractor."
    query = "What is the secret phrase?"

    accs_fp = []
    accs_q = []

    for pi in range(n_prompts):
        try:
            passage = load_passage(tokenizer, L, seed=pi + 100, extra=0)
            passage_tokens = passage[0].tolist()

            rng = np.random.RandomState(pi)
            needle_tokens = tokenizer.encode(needle)
            insert_pos = rng.randint(
                len(passage_tokens) // 4, 3 * len(passage_tokens) // 4
            )
            haystack = (
                passage_tokens[:insert_pos]
                + needle_tokens
                + passage_tokens[insert_pos:]
            )[:L]

            query_tokens = tokenizer.encode("\n\n" + query)
            input_ids = (
                torch.tensor(haystack + query_tokens, dtype=torch.long)
                .unsqueeze(0)
                .to(DEVICE)
            )

            # FP16
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
            past = out_fp.past_key_values
            gen_fp = [out_fp.logits[0, -1, :].argmax().item()]
            for _ in range(30):
                with torch.no_grad():
                    out = model(
                        torch.tensor([[gen_fp[-1]]]).to(DEVICE),
                        past_key_values=past,
                        use_cache=True,
                    )
                gen_fp.append(out.logits[0, -1, :].argmax().item())
                past = out.past_key_values
            gen_text_fp = tokenizer.decode(gen_fp)
            hit_fp = "avocado-electric-tractor" in gen_text_fp.lower()
            accs_fp.append(1.0 if hit_fp else 0.0)
            del past
            clear_gpu()

            # Config
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)
            gen_q = [out_q.logits[0, -1, :].argmax().item()]
            past = past_q
            for _ in range(30):
                with torch.no_grad():
                    out = model(
                        torch.tensor([[gen_q[-1]]]).to(DEVICE),
                        past_key_values=past,
                        use_cache=True,
                    )
                gen_q.append(out.logits[0, -1, :].argmax().item())
                past = out.past_key_values
            gen_text_q = tokenizer.decode(gen_q)
            hit_q = "avocado-electric-tractor" in gen_text_q.lower()
            accs_q.append(1.0 if hit_q else 0.0)
            del past
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            continue

    if accs_fp:
        result = {
            "fp16_acc": round(np.mean(accs_fp), 4),
            "config_acc": round(np.mean(accs_q), 4),
            "n_prompts": len(accs_fp),
        }
        print(
            f"    Needle: FP16={result['fp16_acc']:.3f} "
            f"Config={result['config_acc']:.3f}"
        )
        return result
    return None


# ============================================================
# Experiment 5: Downstream Tasks (MMLU, GSM8K, HumanEval)
# ============================================================
def experiment_5_downstream(model, tokenizer, model_key, config):
    name = config["name"]
    k_type = config["k_type"]
    v_type = config["v_type"]
    print(f"\n  Downstream tasks: {name}")

    results = {}

    # MMLU (200 questions)
    print("    MMLU...")
    try:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="test")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), size=min(200, len(ds)), replace=False)

        correct_fp = 0
        correct_q = 0
        total = 0

        for idx in indices:
            item = ds[int(idx)]
            question = item["question"]
            choices = item["choices"]
            answer = item["answer"]

            prompt = f"Question: {question}\n"
            for ci, c in enumerate(choices):
                prompt += f"{chr(65+ci)}. {c}\n"
            prompt += "Answer:"

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 2048:
                continue

            # FP16
            with torch.no_grad():
                out_fp = model(input_ids)
            logits_fp = out_fp.logits[0, -1, :]

            # Config
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past = out_q.past_key_values
            past = quantize_cache_uniform(past, k_type, v_type, GROUP_SIZE)
            with torch.no_grad():
                out_q2 = model(input_ids[:, -1:], past_key_values=past, use_cache=True)
            logits_q = out_q2.logits[0, -1, :]
            del past
            clear_gpu()

            choice_ids = [
                tokenizer.encode(chr(65 + ci))[-1] for ci in range(len(choices))
            ]
            fp_scores = logits_fp[choice_ids]
            q_scores = logits_q[choice_ids]

            if fp_scores.argmax().item() == answer:
                correct_fp += 1
            if q_scores.argmax().item() == answer:
                correct_q += 1
            total += 1

        if total > 0:
            results["mmlu"] = {
                "fp16": round(correct_fp / total, 4),
                "config": round(correct_q / total, 4),
                "total": total,
            }
            print(
                f"      MMLU: FP16={correct_fp/total:.3f} "
                f"Config={correct_q/total:.3f}"
            )
    except Exception as e:
        print(f"      MMLU error: {e}")

    # GSM8K (200 questions)
    print("    GSM8K...")
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), size=min(200, len(ds)), replace=False)

        correct_fp = 0
        correct_q = 0
        total = 0

        for idx in indices:
            item = ds[int(idx)]
            question = item["question"]
            answer_text = item["answer"]
            answer_match = re.search(r"####\s*(.+)", answer_text)
            if not answer_match:
                continue
            gold = answer_match.group(1).strip().replace(",", "")

            prompt = f"Q: {question}\nA: Let me solve step by step.\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 1024:
                continue

            for mode in ["fp16", "config"]:
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                if mode == "config":
                    past = quantize_cache_uniform(past, k_type, v_type, GROUP_SIZE)

                gen = [out.logits[0, -1, :].argmax().item()]
                for _ in range(100):
                    with torch.no_grad():
                        out2 = model(
                            torch.tensor([[gen[-1]]]).to(DEVICE),
                            past_key_values=past,
                            use_cache=True,
                        )
                    gen.append(out2.logits[0, -1, :].argmax().item())
                    past = out2.past_key_values
                    if tokenizer.eos_token_id and gen[-1] == tokenizer.eos_token_id:
                        break

                gen_text = tokenizer.decode(gen)
                nums = re.findall(r"-?\d+\.?\d*", gen_text.replace(",", ""))
                pred = nums[-1] if nums else ""

                if pred == gold:
                    if mode == "fp16":
                        correct_fp += 1
                    else:
                        correct_q += 1

                del past
                clear_gpu()

            total += 1

        if total > 0:
            results["gsm8k"] = {
                "fp16": round(correct_fp / total, 4),
                "config": round(correct_q / total, 4),
                "total": total,
            }
            print(
                f"      GSM8K: FP16={correct_fp/total:.3f} "
                f"Config={correct_q/total:.3f}"
            )
    except Exception as e:
        print(f"      GSM8K error: {e}")

    # HumanEval (100 problems)
    print("    HumanEval...")
    try:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), size=min(100, len(ds)), replace=False)

        valid_fp = 0
        valid_q = 0
        total = 0

        for idx in indices:
            item = ds[int(idx)]
            prompt = item["prompt"]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 1024:
                continue

            for mode in ["fp16", "config"]:
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                if mode == "config":
                    past = quantize_cache_uniform(past, k_type, v_type, GROUP_SIZE)

                gen = [out.logits[0, -1, :].argmax().item()]
                for _ in range(200):
                    with torch.no_grad():
                        out2 = model(
                            torch.tensor([[gen[-1]]]).to(DEVICE),
                            past_key_values=past,
                            use_cache=True,
                        )
                    gen.append(out2.logits[0, -1, :].argmax().item())
                    past = out2.past_key_values
                    if tokenizer.eos_token_id and gen[-1] == tokenizer.eos_token_id:
                        break

                gen_text = tokenizer.decode(gen)
                try:
                    compile(prompt + gen_text, "<string>", "exec")
                    if mode == "fp16":
                        valid_fp += 1
                    else:
                        valid_q += 1
                except SyntaxError:
                    pass

                del past
                clear_gpu()

            total += 1

        if total > 0:
            results["humaneval"] = {
                "fp16": round(valid_fp / total, 4),
                "config": round(valid_q / total, 4),
                "total": total,
            }
            print(
                f"      HumanEval: FP16={valid_fp/total:.3f} "
                f"Config={valid_q/total:.3f}"
            )
    except Exception as e:
        print(f"      HumanEval error: {e}")

    return results


# ============================================================
# Experiment 6: Bandwidth Analysis
# ============================================================
def compute_bandwidth_savings(configs, model_config):
    """Compute theoretical KV bandwidth for each config."""
    n_layers = model_config.num_hidden_layers
    n_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    tokens = 4096

    results = []
    fp16_bytes = tokens * n_layers * n_kv_heads * head_dim * 2.0 * 2

    for cfg in configs:
        k_bytes = dtype_bytes(cfg["k_type"])
        v_bytes = dtype_bytes(cfg["v_type"])
        config_bytes = tokens * n_layers * n_kv_heads * head_dim * (k_bytes + v_bytes)
        reduction = 1.0 - config_bytes / fp16_bytes
        results.append(
            {
                "name": cfg["name"],
                "k_type": cfg["k_type"],
                "v_type": cfg["v_type"],
                "kv_bytes_4k": round(config_bytes / 1e6, 2),
                "fp16_bytes_4k": round(fp16_bytes / 1e6, 2),
                "bandwidth_reduction_pct": round(reduction * 100, 1),
            }
        )

    return results


# ============================================================
# Plot Generation
# ============================================================
def generate_plots(all_results, model_key):
    """Generate all required plots."""
    configs_tested = [r for r in all_results if not r.get("collapsed", False)]
    if not configs_tested:
        print("  No non-collapsed configs to plot")
        return

    precision_order = {"fp16": 0, "int8": 1, "int6": 2, "int5": 3, "int4": 4}

    def sort_key(r):
        return precision_order.get(r.get("k_type", "fp16"), 99)

    configs_tested = sorted(configs_tested, key=sort_key)
    names = [r["name"] for r in configs_tested]

    # 1. PPL vs key precision
    if any("ppl" in r for r in configs_tested):
        fig, ax = plt.subplots(figsize=(10, 6))
        ppls = [r.get("ppl", {}).get("delta_pct", 0) for r in configs_tested]
        colors = [
            "green" if abs(p) < 3 else "orange" if abs(p) < 10 else "red" for p in ppls
        ]
        ax.bar(range(len(names)), ppls, color=colors)
        ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="3% threshold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("PPL Delta (%)")
        ax.set_title(f"Perplexity vs Key Precision ({model_key})")
        ax.legend()
        for i, v in enumerate(ppls):
            ax.text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"perplexity_vs_key_precision_{model_key}.png"),
            dpi=300,
        )
        plt.close()

    # 2. Token agreement vs key precision
    if any("token_agreement" in r for r in configs_tested):
        fig, ax = plt.subplots(figsize=(10, 6))
        agrees = [r.get("token_agreement", {}).get("mean", 0) for r in configs_tested]
        colors = [
            "green" if a > 0.9 else "orange" if a > 0.7 else "red" for a in agrees
        ]
        ax.bar(range(len(names)), agrees, color=colors)
        ax.axhline(
            y=0.9,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="90% threshold",
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Token Agreement")
        ax.set_title(f"Token Agreement vs Key Precision ({model_key})")
        ax.set_ylim(0, 1.05)
        ax.legend()
        for i, v in enumerate(agrees):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"token_agreement_vs_key_precision_{model_key}.png"),
            dpi=300,
        )
        plt.close()

    # 3. Logit error vs key precision
    if any("logit_error" in r for r in configs_tested):
        fig, ax = plt.subplots(figsize=(10, 6))
        errs = [
            r.get("logit_error", {}).get("mean_logit_err", 0) for r in configs_tested
        ]
        ax.bar(range(len(names)), errs, color="steelblue")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Logit Error")
        ax.set_title(f"Logit Error vs Key Precision ({model_key})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"logit_error_vs_key_precision_{model_key}.png"),
            dpi=300,
        )
        plt.close()

    # 4. Quality vs bandwidth tradeoff
    if any("bandwidth" in r for r in configs_tested):
        fig, ax = plt.subplots(figsize=(10, 6))
        bw = [
            r.get("bandwidth", {}).get("bandwidth_reduction_pct", 0)
            for r in configs_tested
        ]
        ppls = [abs(r.get("ppl", {}).get("delta_pct", 0)) for r in configs_tested]
        for i, r in enumerate(configs_tested):
            color = "green" if ppls[i] < 3 else "orange" if ppls[i] < 10 else "red"
            ax.scatter(bw[i], ppls[i], c=color, s=100, zorder=5)
            ax.annotate(
                r["name"],
                (bw[i], ppls[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
        ax.axhline(
            y=3.0,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="3% quality threshold",
        )
        ax.set_xlabel("Bandwidth Reduction (%)")
        ax.set_ylabel("|PPL Delta| (%)")
        ax.set_title(f"Quality vs Bandwidth Tradeoff ({model_key})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PLOT_DIR,
                f"quality_vs_bandwidth_tradeoff_{model_key}.png",
            ),
            dpi=300,
        )
        plt.close()


def generate_architecture_comparison(qwen_results, mistral_results):
    """Compare key precision tolerance across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, model_results, model_name in [
        (axes[0], qwen_results, "Qwen2.5-7B"),
        (axes[1], mistral_results, "Mistral-7B"),
    ]:
        if not model_results:
            ax.set_title(f"{model_name} (no data)")
            continue
        active = [r for r in model_results if not r.get("collapsed")]
        names = [r["name"] for r in active]
        ppls = [r.get("ppl", {}).get("delta_pct", 0) for r in active]
        colors = [
            "green" if abs(p) < 3 else "orange" if abs(p) < 10 else "red" for p in ppls
        ]
        ax.bar(range(len(names)), ppls, color=colors)
        ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("PPL Delta (%)")
        ax.set_title(model_name)

    plt.suptitle("Architecture Key Precision Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "architecture_key_precision_comparison.png"),
        dpi=300,
    )
    plt.close()


# ============================================================
# Report Generation
# ============================================================
def generate_report(all_model_results, bandwidth_data, start_time):
    """Generate the final summary report."""
    import glob

    elapsed = time.time() - start_time
    hours = elapsed / 3600

    lines = [
        "# BPA v49 — Key Precision Threshold Sweep",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**GPU**: {torch.cuda.get_device_name(0)}",
        "**Dtype**: BF16",
        f"**Runtime**: {hours:.1f} hours",
        "",
    ]

    for model_key, results in all_model_results.items():
        lines.append(f"## {model_key}")
        lines.append("")

        # Collapsed configs
        collapsed = [r for r in results if r.get("collapsed")]
        if collapsed:
            lines.append("### Collapsed Configurations")
            lines.append("")
            for r in collapsed:
                lines.append(
                    f"- **{r['name']}**: "
                    f"{r.get('collapse_reason', 'quality below threshold')}"
                )
            lines.append("")

        # Main results table
        active = [r for r in results if not r.get("collapsed")]
        if active:
            lines.append("### Results")
            lines.append("")
            lines.append(
                "| Config | PPL Delta (%) | Token Agree | "
                "Max Logit Err | Needle Acc | BW Reduction |"
            )
            lines.append(
                "|--------|--------------|-------------|"
                "---------------|------------|-------------|"
            )
            for r in active:
                ppl = r.get("ppl", {}).get("delta_pct", "N/A")
                agree = r.get("token_agreement", {}).get("mean", "N/A")
                logit_err = r.get("logit_error", {}).get("max_mean", "N/A")
                needle = r.get("needle", {}).get("config_acc", "N/A")
                bw = r.get("bandwidth", {}).get("bandwidth_reduction_pct", "N/A")
                lines.append(
                    f"| {r['name']} | {ppl} | {agree} | "
                    f"{logit_err} | {needle} | {bw}% |"
                )
            lines.append("")

        # Downstream tasks
        downstream = [r for r in active if r.get("downstream")]
        if downstream:
            lines.append("### Downstream Tasks")
            lines.append("")
            lines.append(
                "| Config | MMLU FP16 | MMLU Config | "
                "GSM8K FP16 | GSM8K Config | "
                "HumanEval FP16 | HumanEval Config |"
            )
            lines.append(
                "|--------|-----------|-------------|"
                "------------|--------------|"
                "----------------|-----------------|"
            )
            for r in downstream:
                ds = r["downstream"]
                mmlu_fp = ds.get("mmlu", {}).get("fp16", "N/A")
                mmlu_q = ds.get("mmlu", {}).get("config", "N/A")
                gsm_fp = ds.get("gsm8k", {}).get("fp16", "N/A")
                gsm_q = ds.get("gsm8k", {}).get("config", "N/A")
                he_fp = ds.get("humaneval", {}).get("fp16", "N/A")
                he_q = ds.get("humaneval", {}).get("config", "N/A")
                lines.append(
                    f"| {r['name']} | {mmlu_fp} | {mmlu_q} | "
                    f"{gsm_fp} | {gsm_q} | {he_fp} | {he_q} |"
                )
            lines.append("")

    # Bandwidth table
    if bandwidth_data:
        lines.append("## KV Bandwidth Savings")
        lines.append("")
        lines.append("| Config | K Type | V Type | KV Bytes (4K) | BW Reduction |")
        lines.append("|--------|--------|--------|--------------|-------------|")
        for b in bandwidth_data:
            lines.append(
                f"| {b['name']} | {b['k_type']} | {b['v_type']} | "
                f"{b['kv_bytes_4k']} MB | {b['bandwidth_reduction_pct']}% |"
            )
        lines.append("")

    # Analysis placeholder
    lines.append("## Analysis")
    lines.append("")
    lines.append("### What is the minimum viable key precision?")
    lines.append("")
    lines.append("*(See results tables above for threshold identification)*")
    lines.append("")
    lines.append("### Architecture differences in key precision tolerance")
    lines.append("")
    lines.append("*(Compare Qwen vs Mistral results)*")
    lines.append("")

    # Plots
    lines.append("## Plots")
    lines.append("")
    for png in sorted(glob.glob(os.path.join(PLOT_DIR, "*.png"))):
        fname = os.path.basename(png)
        lines.append(f"![{fname}](plots/{fname})")
    lines.append("")

    report_path = os.path.join(RESULTS_ROOT, "bpa49_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    print("=" * 60)
    print("BPA v49: Key Precision Threshold Sweep")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    print(f"GPU: {gpu_info()}")

    all_configs = BASELINE_CONFIGS + CONFIGS
    all_model_results = {}
    all_bandwidth = []

    for model_key in MODELS:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_key}")
        print(f"{'#' * 60}")

        model, tokenizer, n_layers = load_model_and_tokenizer(model_key)
        model_results = []

        for cfg in all_configs:
            print(f"\n{'=' * 60}")
            print(f"Config: {cfg['name']} ({model_key})")
            print("=" * 60)

            result = {
                "name": cfg["name"],
                "k_type": cfg["k_type"],
                "v_type": cfg["v_type"],
                "model": model_key,
            }

            # Early collapse check
            collapsed, collapse_info = early_collapse_check(model, tokenizer, cfg)
            if collapsed:
                result["collapsed"] = True
                result["collapse_info"] = collapse_info
                result["collapse_reason"] = (
                    f"PPL={collapse_info.get('ppl', 'N/A')} "
                    f"agree={collapse_info.get('token_agreement', 'N/A')} "
                    f"logit_err={collapse_info.get('logit_error', 'N/A')}"
                )
                model_results.append(result)
                save_json(result, f"collapsed_{cfg['name']}_{model_key}.json")
                continue

            # Experiment 1: PPL
            ppl_result = experiment_1_perplexity(model, tokenizer, model_key, cfg)
            if ppl_result:
                result["ppl"] = ppl_result

            # Experiment 2: Token Agreement
            agree_result = experiment_2_token_agreement(
                model, tokenizer, model_key, cfg
            )
            if agree_result:
                result["token_agreement"] = agree_result

            # Experiment 3: Logit Error
            logit_result = experiment_3_logit_error(model, tokenizer, model_key, cfg)
            if logit_result:
                result["logit_error"] = logit_result

            # Experiment 4: Needle
            needle_result = experiment_4_needle(model, tokenizer, model_key, cfg)
            if needle_result:
                result["needle"] = needle_result

            # Experiment 5: Downstream (only for sweep configs)
            if cfg in CONFIGS:
                downstream_result = experiment_5_downstream(
                    model, tokenizer, model_key, cfg
                )
                if downstream_result:
                    result["downstream"] = downstream_result

            # Experiment 6: Bandwidth
            bw_results = compute_bandwidth_savings([cfg], model.config)
            if bw_results:
                result["bandwidth"] = bw_results[0]

            model_results.append(result)
            save_json(result, f"results_{cfg['name']}_{model_key}.json")

        all_model_results[model_key] = model_results

        # Full bandwidth table
        bw_all = compute_bandwidth_savings(all_configs, model.config)
        if not all_bandwidth:
            all_bandwidth = bw_all
        save_json(bw_all, f"kv_bandwidth_savings_{model_key}.json")

        # Per-model plots
        generate_plots(model_results, model_key)

        # Save all results for this model
        save_json(model_results, f"all_results_{model_key}.json")

        # Free model
        del model, tokenizer
        clear_gpu()

    # Architecture comparison plot
    generate_architecture_comparison(
        all_model_results.get("qwen2.5-7b", []),
        all_model_results.get("mistral-7b", []),
    )

    # Final report
    generate_report(all_model_results, all_bandwidth, start_time)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("BPA v49 COMPLETE")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    print(f"Results: {RESULTS_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
