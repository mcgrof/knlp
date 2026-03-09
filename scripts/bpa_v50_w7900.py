#!/usr/bin/env python3
"""BPA v50: KV Precision Cliff Localization + Architecture Hypothesis Testing.

Stage 1: Localize Qwen key precision cliff (INT8/INT7/INT6/INT5/INT4)
Stage 2: Hypothesis testing (RoPE energy, key norms, attn entropy, KL div)
Stage 3: Rank hypotheses by correlation with precision floor
Stage 4: Predict precision floor for new models (Phi-3, Pythia-6.9b)
Stage 5: Validate predictions on new models
Stage 6: Fast heuristic estimator

Results saved to /data/knlp-key-results/bpa50/
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# === Paths ===
RESULTS_ROOT = "/data/knlp-key-results/bpa50"
JSON_DIR = os.path.join(RESULTS_ROOT, "json")
PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
HYP_DIR = os.path.join(RESULTS_ROOT, "hypothesis")
for d in [JSON_DIR, PLOT_DIR, LOG_DIR, HYP_DIR]:
    os.makedirs(d, exist_ok=True)

# === Protocol parameters ===
DATASET = "wikitext-103-raw-v1"
W_SINK = 4
GROUP_SIZE = 32
DEVICE = "cuda"
DTYPE = torch.bfloat16

MODELS_BASELINE = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}

# Gemma-7B is gated; substitute Pythia-6.9b (RoPE, open access)
MODELS_NEW = {
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
}


def save_json(data, filename, subdir=None):
    d = subdir if subdir else JSON_DIR
    path = os.path.join(d, filename)
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


# === Quantization ===
def quantize_intN_grouped(tensor, n_bits, group_size=32):
    """Generic symmetric grouped quantization for arbitrary bit widths."""
    orig_shape = tensor.shape
    flat = tensor.reshape(-1)
    pad = (group_size - flat.shape[0] % group_size) % group_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)

    qmax = (1 << (n_bits - 1)) - 1
    qmin = -(1 << (n_bits - 1))

    amax = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale = amax / qmax
    quantized = (groups / scale).round().clamp(qmin, qmax)
    dequantized = quantized * scale

    flat_out = dequantized.reshape(-1)
    if pad > 0:
        flat_out = flat_out[: orig_shape.numel()]
    return flat_out.reshape(orig_shape)


def quantize_tensor(tensor, quant_type, group_size=32):
    if quant_type == "fp16":
        return tensor
    bits_map = {"int8": 8, "int7": 7, "int6": 6, "int5": 5, "int4": 4}
    if quant_type in bits_map:
        return quantize_intN_grouped(tensor, bits_map[quant_type], group_size)
    raise ValueError(f"Unknown quant type: {quant_type}")


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
    """Quantize all KV cache layers uniformly (DynamicCache aware)."""
    D = n_cache_layers(past)
    for li in range(D):
        k, v = _cache_get_kv(past, li)
        if k.shape[2] > W_SINK:
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
def load_model(model_id):
    """Load model with eager attention, .to(DEVICE) pattern."""
    print(f"Loading {model_id}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    n_layers = config.num_hidden_layers
    print(f"  Model loaded: {n_layers} layers, device={DEVICE}")
    return model, tokenizer, config


def unload_model(model):
    del model
    clear_gpu()


# === Early Collapse Detection ===
def early_collapse_check(model, tokenizer, k_type, v_type, config_name):
    """Quick 512-token test. Returns (collapsed, info_dict)."""
    print(f"\n  Collapse check: {config_name}")
    tokens = load_wikitext_tokens(tokenizer)
    results = []
    for seed in range(3):
        rng = np.random.RandomState(seed + 100)
        start = rng.randint(0, max(1, len(tokens) - 600))
        input_ids = (
            torch.tensor(tokens[start : start + 512], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values

        if k_type != "fp16" or v_type != "fp16":
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

        # Generate 64 tokens
        agree = 0
        gen_fp = input_ids
        gen_q = input_ids
        past_fp_gen = past_fp
        past_q_gen = past_q
        for step in range(64):
            with torch.no_grad():
                o_fp = model(
                    gen_fp[:, -1:],
                    past_key_values=past_fp_gen,
                    use_cache=True,
                )
                o_q = model(
                    gen_q[:, -1:],
                    past_key_values=past_q_gen,
                    use_cache=True,
                )
            t_fp = o_fp.logits[0, -1].argmax().item()
            t_q = o_q.logits[0, -1].argmax().item()
            if t_fp == t_q:
                agree += 1
            gen_fp = torch.tensor([[t_fp]]).to(DEVICE)
            gen_q = torch.tensor([[t_q]]).to(DEVICE)
            past_fp_gen = o_fp.past_key_values
            past_q_gen = o_q.past_key_values

        # PPL on 512 tokens
        with torch.no_grad():
            out_ppl = model(input_ids)
        logits = out_ppl.logits[:, :-1, :].float().cpu()
        targets = input_ids[:, 1:].cpu()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        ppl = math.exp(loss.item())

        # Max logit error
        with torch.no_grad():
            out_fp2 = model(input_ids, use_cache=True)
            past_fp2 = out_fp2.past_key_values
            out_q2 = model(input_ids, use_cache=True)
            past_q2 = out_q2.past_key_values
        if k_type != "fp16" or v_type != "fp16":
            past_q2 = quantize_cache_uniform(past_q2, k_type, v_type, GROUP_SIZE)
        with torch.no_grad():
            next_tok = torch.tensor([[tokens[start + 512]]]).to(DEVICE)
            o1 = model(next_tok, past_key_values=past_fp2, use_cache=True)
            o2 = model(next_tok, past_key_values=past_q2, use_cache=True)
        logit_err = (
            (o1.logits[0, 0].float() - o2.logits[0, 0].float()).abs().max().item()
        )

        results.append({"ppl": ppl, "agree": agree / 64, "logit_err": logit_err})
        del past_fp, past_q, past_fp_gen, past_q_gen
        del past_fp2, past_q2
        clear_gpu()

    avg_ppl = np.mean([r["ppl"] for r in results])
    avg_agree = np.mean([r["agree"] for r in results])
    avg_logit = np.mean([r["logit_err"] for r in results])

    # v50 uses relaxed thresholds learned from v49 analysis:
    # - Token agreement is inherently low (~0.3-0.5) even for
    #   near-lossless PPL configs due to autoregressive error
    #   amplification. agree<0.5 causes massive false positives.
    # - Only flag collapse for truly catastrophic failures:
    #   agree<0.15 (random-level) or logit_err>10 (gross mismatch)
    collapsed = avg_agree < 0.15 or avg_logit > 10

    info = {
        "name": config_name,
        "ppl": round(avg_ppl, 4),
        "token_agreement": round(avg_agree, 4),
        "logit_error": round(avg_logit, 4),
        "collapsed": str(collapsed),
    }
    status = "COLLAPSED" if collapsed else "OK"
    print(
        f"    {status}: PPL={avg_ppl:.2f} agree={avg_agree:.4f} "
        f"logit_err={avg_logit:.2f}"
    )
    return collapsed, info


# === PPL Evaluation ===
def compute_ppl(model, tokenizer, k_type, v_type, config_name, n_seeds=5):
    """Compute perplexity on WikiText-103."""
    print(f"\n  PPL eval: {config_name}")
    tokens = load_wikitext_tokens(tokenizer)
    ppls_fp = []
    ppls_q = []

    for seed in range(n_seeds):
        input_ids = load_passage(tokenizer, 2048, seed).to(DEVICE)

        with torch.no_grad():
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values

        if k_type != "fp16" or v_type != "fp16":
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

        # Generate 64 tokens for PPL
        gen_logits_fp = []
        gen_logits_q = []
        gen_tokens = []

        next_tok = input_ids[:, -1:]
        p_fp = past_fp
        p_q = past_q
        for step in range(64):
            with torch.no_grad():
                o_fp = model(next_tok, past_key_values=p_fp, use_cache=True)
                o_q = model(next_tok, past_key_values=p_q, use_cache=True)
            gen_logits_fp.append(o_fp.logits[0, 0].float().cpu())
            gen_logits_q.append(o_q.logits[0, 0].float().cpu())
            t = o_fp.logits[0, 0].argmax().item()
            gen_tokens.append(t)
            next_tok = torch.tensor([[t]]).to(DEVICE)
            p_fp = o_fp.past_key_values
            p_q = o_q.past_key_values

        # Compute PPL from prefill
        logits_fp = out_fp.logits[:, :-1, :].float().cpu()
        logits_q = out_q.logits[:, :-1, :].float().cpu()
        targets = input_ids[:, 1:].cpu()

        loss_fp = F.cross_entropy(
            logits_fp.reshape(-1, logits_fp.shape[-1]),
            targets.reshape(-1),
        )
        loss_q = F.cross_entropy(
            logits_q.reshape(-1, logits_q.shape[-1]),
            targets.reshape(-1),
        )
        ppls_fp.append(math.exp(loss_fp.item()))
        ppls_q.append(math.exp(loss_q.item()))

        del past_fp, past_q, p_fp, p_q
        clear_gpu()

    ppl_fp = np.mean(ppls_fp)
    ppl_q = np.mean(ppls_q)
    delta = (ppl_q - ppl_fp) / ppl_fp * 100 if ppl_fp > 0 else 0

    result = {
        "ppl_fp16": round(ppl_fp, 4),
        "ppl_config": round(ppl_q, 4),
        "delta_pct": round(delta, 4),
        "n_seeds": n_seeds,
    }
    print(f"    PPL: FP16={ppl_fp:.4f} Config={ppl_q:.4f} " f"delta={delta:.2f}%")
    return result


# === Token Agreement ===
def compute_token_agreement(
    model, tokenizer, k_type, v_type, config_name, n_prompts=200
):
    """200 prompts x 256 generated tokens."""
    print(f"\n  Token agreement: {config_name}")
    tokens = load_wikitext_tokens(tokenizer)
    agrees = []

    for i in range(n_prompts):
        if i % 50 == 0:
            print(f"    Prompt {i}/{n_prompts}...")
        rng = np.random.RandomState(i + 200)
        start = rng.randint(0, max(1, len(tokens) - 300))
        input_ids = (
            torch.tensor(tokens[start : start + 128], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values

        if k_type != "fp16" or v_type != "fp16":
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

        agree = 0
        tok_fp = input_ids[:, -1:]
        tok_q = input_ids[:, -1:]
        p_fp = past_fp
        p_q = past_q

        for step in range(256):
            with torch.no_grad():
                o_fp = model(tok_fp, past_key_values=p_fp, use_cache=True)
                o_q = model(tok_q, past_key_values=p_q, use_cache=True)
            t_fp = o_fp.logits[0, -1].argmax().item()
            t_q = o_q.logits[0, -1].argmax().item()
            if t_fp == t_q:
                agree += 1
            tok_fp = torch.tensor([[t_fp]]).to(DEVICE)
            tok_q = torch.tensor([[t_q]]).to(DEVICE)
            p_fp = o_fp.past_key_values
            p_q = o_q.past_key_values

        agrees.append(agree / 256)
        del past_fp, past_q, p_fp, p_q
        clear_gpu()

    result = {
        "mean": round(np.mean(agrees), 4),
        "std": round(np.std(agrees), 4),
        "min": round(np.min(agrees), 4),
        "max": round(np.max(agrees), 4),
        "n_prompts": n_prompts,
    }
    print(f"    Agreement: mean={result['mean']:.4f} std={result['std']:.4f}")
    return result


# === Logit Error ===
def compute_logit_error(model, tokenizer, k_type, v_type, config_name):
    """Compute max logit error across 10 seeds."""
    print(f"\n  Logit error: {config_name}")
    tokens = load_wikitext_tokens(tokenizer)
    max_errs = []

    for seed in range(10):
        input_ids = load_passage(tokenizer, 1024, seed + 50).to(DEVICE)

        with torch.no_grad():
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values

        if k_type != "fp16" or v_type != "fp16":
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

        next_tok = input_ids[:, -1:]
        with torch.no_grad():
            o1 = model(next_tok, past_key_values=past_fp, use_cache=True)
            o2 = model(next_tok, past_key_values=past_q, use_cache=True)

        diff = (o1.logits[0, 0].float() - o2.logits[0, 0].float()).abs()
        max_errs.append(diff.max().item())

        del past_fp, past_q
        clear_gpu()

    errs = np.array(max_errs)
    result = {
        "max_mean": round(errs.mean(), 4),
        "max_median": round(np.median(errs), 4),
        "max_p95": round(np.percentile(errs, 95), 4),
        "max_p99": round(np.percentile(errs, 99), 4),
        "mean_logit_err": round(
            np.mean(
                [
                    (o1.logits[0, 0].float() - o2.logits[0, 0].float())
                    .abs()
                    .mean()
                    .item()
                ]
            ),
            4,
        ),
        "n_seeds": 10,
    }
    print(
        f"    Max logit err: mean={result['max_mean']:.2f} "
        f"p95={result['max_p95']:.2f} "
        f"mean_err={result['mean_logit_err']:.4f}"
    )
    return result


# === Needle Retrieval ===
def compute_needle(model, tokenizer, k_type, v_type, config_name):
    """Needle-in-haystack at 4K context."""
    print(f"\n  Needle retrieval: {config_name}")
    needle = "The secret phrase is: avocado-electric-tractor."
    n_prompts = 50
    correct_fp = 0
    correct_q = 0

    tokens = load_wikitext_tokens(tokenizer)
    for i in range(n_prompts):
        rng = np.random.RandomState(i + 300)
        start = rng.randint(0, max(1, len(tokens) - 4200))
        haystack_tokens = tokens[start : start + 4000]
        needle_tokens = tokenizer.encode(needle)
        insert_pos = rng.randint(100, 3800)
        full = (
            haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
        )
        full = full[:4096]
        question = " What is the secret phrase?"
        q_tokens = tokenizer.encode(question)
        input_tokens = full + q_tokens
        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

        if input_ids.shape[1] > 4200:
            input_ids = input_ids[:, :4200]

        for mode in ["fp16", "config"]:
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
            if mode == "config" and (k_type != "fp16" or v_type != "fp16"):
                past = quantize_cache_uniform(past, k_type, v_type, GROUP_SIZE)

            gen = []
            tok = input_ids[:, -1:]
            p = past
            for _ in range(30):
                with torch.no_grad():
                    o = model(tok, past_key_values=p, use_cache=True)
                t = o.logits[0, -1].argmax().item()
                gen.append(t)
                tok = torch.tensor([[t]]).to(DEVICE)
                p = o.past_key_values
                if tokenizer.eos_token_id and t == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(gen).lower()
            if "avocado" in text and "tractor" in text:
                if mode == "fp16":
                    correct_fp += 1
                else:
                    correct_q += 1

            del past, p
            clear_gpu()

    result = {
        "fp16_acc": round(correct_fp / n_prompts, 4),
        "config_acc": round(correct_q / n_prompts, 4),
        "n_prompts": n_prompts,
    }
    print(
        f"    Needle: FP16={result['fp16_acc']:.3f} "
        f"Config={result['config_acc']:.3f}"
    )
    return result


# === Bandwidth computation ===
def compute_bandwidth_savings(configs, model_config, model_name):
    """Compute KV bandwidth for each config."""
    n_layers = model_config.num_hidden_layers
    n_kv_heads = getattr(
        model_config,
        "num_key_value_heads",
        model_config.num_attention_heads,
    )
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    seq_len = 4096

    dtype_bytes = {
        "fp16": 2.0,
        "int8": 1.0,
        "int7": 0.875,
        "int6": 0.75,
        "int5": 0.625,
        "int4": 0.5,
    }

    results = []
    fp16_bytes = seq_len * n_layers * n_kv_heads * head_dim * 2.0 * 2  # K+V

    for cfg in configs:
        k_bytes = dtype_bytes.get(cfg["k_type"], 2.0)
        v_bytes = dtype_bytes.get(cfg["v_type"], 2.0)
        kv_bytes = seq_len * n_layers * n_kv_heads * head_dim * (k_bytes + v_bytes)
        reduction = (1 - kv_bytes / fp16_bytes) * 100

        results.append(
            {
                "name": cfg["name"],
                "k_type": cfg["k_type"],
                "v_type": cfg["v_type"],
                "kv_bytes_4k": round(kv_bytes / 1e6, 2),
                "fp16_bytes_4k": round(fp16_bytes / 1e6, 2),
                "bandwidth_reduction_pct": round(reduction, 1),
            }
        )

    save_json(results, f"kv_bandwidth_savings_{model_name}.json")
    return results


# ============================================================
# Stage 2: Hypothesis Testing
# ============================================================


def hypothesis_a_rope_energy(model, tokenizer, config, model_name):
    """Measure RoPE frequency energy distribution."""
    print(f"\n  Hypothesis A: RoPE frequency energy ({model_name})")
    tokens = load_wikitext_tokens(tokenizer)
    input_ids = torch.tensor(tokens[:512], dtype=torch.long).unsqueeze(0).to(DEVICE)

    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    # Extract K values from cache
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    energy_per_layer = []
    high_freq_ratios = []

    for layer_idx in range(n_layers):
        k, _ = _cache_get_kv(past, layer_idx)  # [B, n_kv_heads, T, hd]
        # Energy per dimension (cast to float for RDNA3 BF16 compat)
        k_f = k.float()
        energy = (k_f**2).mean(dim=(0, 2)).cpu().numpy()  # [n_kv_heads, hd]
        energy_per_layer.append(energy)

        # High-frequency ratio: last half of dims have higher RoPE freq
        total_e = energy.sum(axis=1)
        half = head_dim // 2
        high_e = energy[:, half:].sum(axis=1)
        ratio = high_e / (total_e + 1e-10)
        high_freq_ratios.append(ratio.mean())

    del past
    clear_gpu()

    result = {
        "model": model_name,
        "mean_high_freq_ratio": round(float(np.mean(high_freq_ratios)), 6),
        "std_high_freq_ratio": round(float(np.std(high_freq_ratios)), 6),
        "per_layer_high_freq_ratio": [round(float(r), 6) for r in high_freq_ratios],
    }
    save_json(result, f"hypothesis_a_rope_{model_name}.json", HYP_DIR)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(high_freq_ratios, "o-")
    ax.set_xlabel("Layer")
    ax.set_ylabel("High-Frequency Energy Ratio")
    ax.set_title(f"RoPE Frequency Energy - {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, f"rope_frequency_energy_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)

    print(f"    Mean high-freq ratio: {result['mean_high_freq_ratio']:.4f}")
    return result


def hypothesis_b_key_norms(model, tokenizer, config, model_name):
    """Measure key activation distribution per layer."""
    print(f"\n  Hypothesis B: Key norm distribution ({model_name})")
    tokens = load_wikitext_tokens(tokenizer)
    input_ids = torch.tensor(tokens[:512], dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    n_layers = config.num_hidden_layers
    means = []
    stds = []
    maxes = []

    for layer_idx in range(n_layers):
        k, _ = _cache_get_kv(past, layer_idx)  # [B, n_kv_heads, T, hd]
        k_norms = k.float().norm(dim=-1)  # [B, n_kv_heads, T]
        means.append(k_norms.mean().item())
        stds.append(k_norms.std().item())
        maxes.append(k_norms.max().item())

    del past
    clear_gpu()

    result = {
        "model": model_name,
        "mean_k_norm": round(np.mean(means), 4),
        "std_k_norm": round(np.mean(stds), 4),
        "max_k_norm": round(np.max(maxes), 4),
        "outlier_ratio": round(np.max(maxes) / (np.mean(means) + 1e-10), 4),
        "per_layer_mean": [round(m, 4) for m in means],
        "per_layer_std": [round(s, 4) for s in stds],
        "per_layer_max": [round(m, 4) for m in maxes],
    }
    save_json(result, f"hypothesis_b_keynorms_{model_name}.json", HYP_DIR)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(means, "o-")
    axes[0].set_title("Mean ||K||")
    axes[0].set_xlabel("Layer")
    axes[1].plot(stds, "o-")
    axes[1].set_title("Std ||K||")
    axes[1].set_xlabel("Layer")
    axes[2].plot(maxes, "o-")
    axes[2].set_title("Max ||K||")
    axes[2].set_xlabel("Layer")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Key Norm Distribution - {model_name}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, f"key_norm_distribution_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)

    print(
        f"    Mean K norm: {result['mean_k_norm']:.4f} "
        f"outlier_ratio: {result['outlier_ratio']:.4f}"
    )
    return result


def hypothesis_c_attention_entropy(model, tokenizer, config, model_name):
    """Compute attention entropy per head."""
    print(f"\n  Hypothesis C: Attention entropy ({model_name})")
    tokens = load_wikitext_tokens(tokenizer)
    input_ids = torch.tensor(tokens[:512], dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(
            input_ids,
            output_attentions=True,
            use_cache=False,
        )

    n_layers = config.num_hidden_layers
    entropies_per_layer = []

    for layer_idx in range(n_layers):
        attn = out.attentions[layer_idx]  # [B, n_heads, T, T]
        # Entropy = -sum(p * log(p))
        log_attn = torch.log(attn + 1e-10)
        entropy = -(attn * log_attn).sum(dim=-1)  # [B, n_heads, T]
        mean_entropy = entropy.mean().item()
        entropies_per_layer.append(mean_entropy)

    del out
    clear_gpu()

    result = {
        "model": model_name,
        "mean_entropy": round(np.mean(entropies_per_layer), 4),
        "std_entropy": round(np.std(entropies_per_layer), 4),
        "min_entropy": round(np.min(entropies_per_layer), 4),
        "per_layer_entropy": [round(e, 4) for e in entropies_per_layer],
    }
    save_json(result, f"hypothesis_c_entropy_{model_name}.json", HYP_DIR)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(entropies_per_layer, "o-")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title(f"Attention Entropy Distribution - {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"attention_entropy_distribution_{model_name}.png",
        ),
        dpi=300,
    )
    plt.close(fig)

    print(f"    Mean entropy: {result['mean_entropy']:.4f}")
    return result


def hypothesis_d_routing_kl(model, tokenizer, config, model_name):
    """Compute KL divergence of attention routing under INT6 quantization."""
    print(f"\n  Hypothesis D: Routing KL divergence ({model_name})")
    tokens = load_wikitext_tokens(tokenizer)
    input_ids = torch.tensor(tokens[:512], dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Run model twice to get separate cache objects
        out_fp = model(input_ids, use_cache=True)
        past_fp = out_fp.past_key_values
        out_q = model(input_ids, use_cache=True)
        past_q = out_q.past_key_values

    # Quantize only the second cache (past_fp stays pristine)
    past_q = quantize_cache_uniform(past_q, "int6", "fp16", GROUP_SIZE)

    # Compute attention weights with FP16 and INT6 keys
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = config.hidden_size // n_heads

    kl_per_layer = []

    for layer_idx in range(n_layers):
        k_fp, _ = _cache_get_kv(past_fp, layer_idx)
        k_q, _ = _cache_get_kv(past_q, layer_idx)

        # Use last token's query to compute attention
        # We already have the output; compute Q @ K^T manually
        # Approximate: use key similarity as proxy
        # k_fp: [1, n_kv_heads, T, hd]
        # Compute self-attention scores as proxy
        k_fp_f = k_fp.float()
        k_q_f = k_q.float()
        sim_fp = torch.matmul(k_fp_f, k_fp_f.transpose(-2, -1)) / math.sqrt(head_dim)
        sim_q = torch.matmul(k_q_f, k_q_f.transpose(-2, -1)) / math.sqrt(head_dim)

        p_fp = F.softmax(sim_fp[:, :, -1, :], dim=-1)  # last token
        p_q = F.softmax(sim_q[:, :, -1, :], dim=-1)

        # KL(p_fp || p_q) — clamp to avoid log(0)
        kl = (
            F.kl_div((p_q + 1e-10).log(), p_fp, reduction="none")
            .sum(dim=-1)
            .mean()
            .item()
        )
        kl_per_layer.append(kl)

    del past_fp, past_q
    clear_gpu()

    result = {
        "model": model_name,
        "mean_kl": round(np.mean(kl_per_layer), 6),
        "max_kl": round(np.max(kl_per_layer), 6),
        "per_layer_kl": [round(k, 6) for k in kl_per_layer],
    }
    save_json(result, f"hypothesis_d_kl_{model_name}.json", HYP_DIR)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kl_per_layer, "o-")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence (INT6 vs FP16)")
    ax.set_title(f"Attention Routing KL - {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, f"attention_KL_vs_precision_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)

    print(f"    Mean KL: {result['mean_kl']:.6f}")
    return result


# ============================================================
# Stage 3: Hypothesis Ranking
# ============================================================
def rank_hypotheses(hyp_results, precision_floors):
    """Rank hypotheses by correlation with precision floor."""
    print("\n=== Stage 3: Hypothesis Ranking ===")

    # Map precision floor to numeric (lower = more sensitive)
    floor_map = {
        "fp16": 16,
        "int8": 8,
        "int7": 7,
        "int6": 6,
        "int5": 5,
        "int4": 4,
        "none": 0,
    }

    models = list(precision_floors.keys())
    floors = [floor_map.get(precision_floors[m], 0) for m in models]

    rankings = []

    # Hypothesis A: high freq ratio (higher = more sensitive?)
    if all(m in hyp_results.get("rope_energy", {}) for m in models):
        vals = [hyp_results["rope_energy"][m]["mean_high_freq_ratio"] for m in models]
        corr = np.corrcoef(vals, floors)[0, 1] if len(vals) > 1 else 0
        rankings.append(
            {
                "hypothesis": "A_rope_frequency_energy",
                "metric": "mean_high_freq_ratio",
                "values": {m: round(v, 6) for m, v in zip(models, vals)},
                "correlation_with_floor": (
                    round(abs(corr), 4) if not np.isnan(corr) else 0
                ),
            }
        )

    # Hypothesis B: key norm outlier ratio
    if all(m in hyp_results.get("key_norms", {}) for m in models):
        vals = [hyp_results["key_norms"][m]["outlier_ratio"] for m in models]
        corr = np.corrcoef(vals, floors)[0, 1] if len(vals) > 1 else 0
        rankings.append(
            {
                "hypothesis": "B_key_norm_variance",
                "metric": "outlier_ratio",
                "values": {m: round(v, 4) for m, v in zip(models, vals)},
                "correlation_with_floor": (
                    round(abs(corr), 4) if not np.isnan(corr) else 0
                ),
            }
        )

    # Hypothesis C: attention entropy (lower = sharper = more sensitive?)
    if all(m in hyp_results.get("entropy", {}) for m in models):
        vals = [hyp_results["entropy"][m]["mean_entropy"] for m in models]
        corr = np.corrcoef(vals, floors)[0, 1] if len(vals) > 1 else 0
        rankings.append(
            {
                "hypothesis": "C_attention_entropy",
                "metric": "mean_entropy",
                "values": {m: round(v, 4) for m, v in zip(models, vals)},
                "correlation_with_floor": (
                    round(abs(corr), 4) if not np.isnan(corr) else 0
                ),
            }
        )

    # Hypothesis D: routing KL
    if all(m in hyp_results.get("routing_kl", {}) for m in models):
        vals = [hyp_results["routing_kl"][m]["mean_kl"] for m in models]
        corr = np.corrcoef(vals, floors)[0, 1] if len(vals) > 1 else 0
        rankings.append(
            {
                "hypothesis": "D_routing_KL_divergence",
                "metric": "mean_kl",
                "values": {m: round(v, 6) for m, v in zip(models, vals)},
                "correlation_with_floor": (
                    round(abs(corr), 4) if not np.isnan(corr) else 0
                ),
            }
        )

    # Sort by correlation
    rankings.sort(key=lambda x: x["correlation_with_floor"], reverse=True)

    for i, r in enumerate(rankings):
        print(
            f"  {i+1}. {r['hypothesis']}: " f"|corr|={r['correlation_with_floor']:.4f}"
        )

    save_json(
        {"rankings": rankings, "precision_floors": precision_floors},
        "hypothesis_ranking.json",
    )
    return rankings


# ============================================================
# Plotting
# ============================================================
def plot_precision_cliff(results, model_name):
    """Plot PPL and logit error vs key precision."""
    configs = []
    ppls = []
    logit_errs = []

    for r in results:
        if "ppl" in r and r.get("ppl"):
            configs.append(r["name"])
            ppls.append(r["ppl"]["delta_pct"])
            if "logit_error" in r and r.get("logit_error"):
                logit_errs.append(r["logit_error"]["max_mean"])
            else:
                logit_errs.append(0)

    if not configs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(len(configs)), ppls)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.set_ylabel("PPL Delta (%)")
    ax1.set_title(f"Perplexity vs Key Precision - {model_name}")
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(logit_errs)), logit_errs, color="orange")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha="right")
    ax2.set_ylabel("Max Logit Error")
    ax2.set_title(f"Logit Error vs Key Precision - {model_name}")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, f"perplexity_vs_key_precision_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)


def plot_architecture_comparison(all_results):
    """Cross-architecture precision comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, results in all_results.items():
        configs = []
        ppls = []
        for r in results:
            if "ppl" in r and r.get("ppl"):
                configs.append(r["name"])
                ppls.append(r["ppl"]["delta_pct"])
        if configs:
            ax.plot(range(len(configs)), ppls, "o-", label=model_name)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("PPL Delta (%)")
    ax.set_title("Architecture Key Precision Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, "architecture_precision_comparison.png"),
        dpi=300,
    )
    plt.close(fig)


def plot_quality_vs_bandwidth(results, bw_data, model_name):
    """Quality vs bandwidth tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bw_map = {b["name"]: b["bandwidth_reduction_pct"] for b in bw_data}

    for r in results:
        name = r["name"]
        if name in bw_map and "ppl" in r and r.get("ppl"):
            bw = bw_map[name]
            ppl_delta = abs(r["ppl"]["delta_pct"])
            ax.scatter(bw, ppl_delta, s=100, zorder=5)
            ax.annotate(
                name,
                (bw, ppl_delta),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel("Bandwidth Reduction (%)")
    ax.set_ylabel("|PPL Delta| (%)")
    ax.set_title(f"Quality vs Bandwidth - {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, f"quality_vs_bandwidth_tradeoff_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)


# ============================================================
# Summary Report
# ============================================================
def write_summary(
    stage1_results,
    hyp_results,
    rankings,
    predictions,
    validation_results,
    estimator_result,
    runtime_s,
):
    """Write bpa50_summary.md."""
    path = os.path.join(RESULTS_ROOT, "bpa50_summary.md")
    with open(path, "w") as f:
        f.write("# BPA v50 — KV Precision Cliff Localization\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("**GPU**: AMD Radeon Pro W7900\n")
        f.write("**Dtype**: BF16\n")
        f.write(f"**Runtime**: {runtime_s:.0f}s " f"({runtime_s/3600:.1f} hours)\n\n")

        # Stage 1
        f.write("## Stage 1: Key Precision Cliff (Qwen)\n\n")
        if "qwen2.5-7b" in stage1_results:
            qr = stage1_results["qwen2.5-7b"]
            f.write(
                "| Config | PPL Delta (%) | Token Agree | " "Logit Err | Collapsed |\n"
            )
            f.write(
                "|--------|--------------|-------------|" "----------|-----------|\n"
            )
            for r in qr:
                if r.get("collapsed"):
                    ci = r.get("collapse_info", {})
                    f.write(
                        f"| {r['name']} | N/A | "
                        f"{ci.get('token_agreement', 'N/A')} | "
                        f"{ci.get('logit_error', 'N/A')} | YES |\n"
                    )
                elif "ppl" in r:
                    f.write(
                        f"| {r['name']} | {r['ppl']['delta_pct']} | "
                        f"{r.get('token_agreement', {}).get('mean', 'N/A')} | "
                        f"{r.get('logit_error', {}).get('max_mean', 'N/A')} | "
                        f"NO |\n"
                    )
            f.write("\n")

        # Stage 2
        f.write("## Stage 2: Hypothesis Testing\n\n")
        for hyp_name, hyp_data in hyp_results.items():
            f.write(f"### {hyp_name}\n\n")
            for model_name, data in hyp_data.items():
                key_metric = list(data.keys())[1] if len(data) > 1 else "N/A"
                val = data.get(key_metric, "N/A")
                f.write(f"- **{model_name}**: {key_metric}={val}\n")
            f.write("\n")

        # Stage 3
        f.write("## Stage 3: Hypothesis Ranking\n\n")
        if rankings:
            f.write("| Rank | Hypothesis | |Correlation| |\n")
            f.write("|------|------------|----------------|\n")
            for i, r in enumerate(rankings):
                f.write(
                    f"| {i+1} | {r['hypothesis']} | "
                    f"{r['correlation_with_floor']} |\n"
                )
            f.write("\n")

        # Stage 4-5
        f.write("## Stage 4-5: Predictions & Validation\n\n")
        if predictions:
            f.write("| Model | Predicted Floor | Observed Floor |\n")
            f.write("|-------|----------------|----------------|\n")
            for m, pred in predictions.items():
                obs = validation_results.get(m, {}).get("observed_floor", "N/A")
                f.write(f"| {m} | {pred} | {obs} |\n")
            f.write("\n")

        # Stage 6
        f.write("## Stage 6: Fast Estimator\n\n")
        if estimator_result:
            f.write(f"Estimator uses: {estimator_result.get('method', 'N/A')}\n\n")

        # Analysis placeholder
        f.write("## Analysis\n\n")
        f.write("*(Analysis to be written after all stages complete)*\n\n")

        # Plots
        f.write("## Plots\n\n")
        for p in sorted(os.listdir(PLOT_DIR)):
            if p.endswith(".png"):
                f.write(f"![{p}](plots/{p})\n")

    print(f"\n  Report saved: {path}")
    return path


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("BPA v50 — KV Precision Cliff + Hypothesis Testing")
    print(f"Started: {datetime.now()}")
    print(f"GPU: {gpu_info()}")
    print("=" * 60)

    # Redirect stdout to log
    log_path = os.path.join(LOG_DIR, "run.log")
    import io

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    all_stage1_results = {}
    hyp_results = {
        "rope_energy": {},
        "key_norms": {},
        "entropy": {},
        "routing_kl": {},
    }

    # Known precision floors from v48/v49
    precision_floors = {
        "qwen2.5-7b": "fp16",  # Will be refined in Stage 1
        "mistral-7b": "none",  # No floor detected in v49
    }

    # ============================================================
    # Stage 1: Qwen Key Precision Cliff
    # ============================================================
    print("\n" + "=" * 60)
    print("# Stage 1: Key Precision Cliff Localization (Qwen)")
    print("=" * 60)

    model, tokenizer, config = load_model(MODELS_BASELINE["qwen2.5-7b"])

    configs_stage1 = [
        {"name": "FP16_baseline", "k_type": "fp16", "v_type": "fp16"},
        {"name": "K_FP16__V_INT4", "k_type": "fp16", "v_type": "int4"},
        {"name": "K_INT8__V_INT4", "k_type": "int8", "v_type": "int4"},
        {"name": "K_INT7__V_INT4", "k_type": "int7", "v_type": "int4"},
        {"name": "K_INT6__V_INT4", "k_type": "int6", "v_type": "int4"},
        {"name": "K_INT5__V_INT4", "k_type": "int5", "v_type": "int4"},
        {"name": "K_INT4__V_INT4", "k_type": "int4", "v_type": "int4"},
    ]

    qwen_results = []
    for cfg in configs_stage1:
        print(f"\n{'=' * 60}")
        print(f"Config: {cfg['name']} (qwen2.5-7b)")
        print(f"{'=' * 60}")

        result = {"name": cfg["name"], "k_type": cfg["k_type"], "v_type": cfg["v_type"]}

        # Early collapse detection
        collapsed, collapse_info = early_collapse_check(
            model, tokenizer, cfg["k_type"], cfg["v_type"], cfg["name"]
        )

        if collapsed:
            result["collapsed"] = True
            result["collapse_info"] = collapse_info
            result["collapse_reason"] = (
                f"PPL={collapse_info['ppl']} "
                f"agree={collapse_info['token_agreement']} "
                f"logit_err={collapse_info['logit_error']}"
            )
            save_json(
                collapse_info,
                f"collapsed_{cfg['name']}_qwen2.5-7b.json",
            )
            qwen_results.append(result)
            continue

        # Full evaluation
        result["ppl"] = compute_ppl(
            model, tokenizer, cfg["k_type"], cfg["v_type"], cfg["name"]
        )
        result["token_agreement"] = compute_token_agreement(
            model,
            tokenizer,
            cfg["k_type"],
            cfg["v_type"],
            cfg["name"],
            n_prompts=100,  # Reduced from 200 for speed
        )
        result["logit_error"] = compute_logit_error(
            model, tokenizer, cfg["k_type"], cfg["v_type"], cfg["name"]
        )
        result["needle"] = compute_needle(
            model, tokenizer, cfg["k_type"], cfg["v_type"], cfg["name"]
        )

        save_json(result, f"results_{cfg['name']}_qwen2.5-7b.json")
        qwen_results.append(result)

    # Determine Qwen precision cliff
    qwen_cliff = "fp16"
    for cfg_name, bits in [
        ("K_INT8__V_INT4", "int8"),
        ("K_INT7__V_INT4", "int7"),
        ("K_INT6__V_INT4", "int6"),
        ("K_INT5__V_INT4", "int5"),
        ("K_INT4__V_INT4", "int4"),
    ]:
        r = next((x for x in qwen_results if x["name"] == cfg_name), None)
        if r and not r.get("collapsed"):
            qwen_cliff = bits
        else:
            break
    precision_floors["qwen2.5-7b"] = qwen_cliff
    print(f"\n  Qwen precision cliff: {qwen_cliff}")

    bw_data_qwen = compute_bandwidth_savings(configs_stage1, config, "qwen2.5-7b")
    save_json(qwen_results, "all_results_qwen2.5-7b.json")
    plot_precision_cliff(qwen_results, "qwen2.5-7b")
    plot_quality_vs_bandwidth(qwen_results, bw_data_qwen, "qwen2.5-7b")

    all_stage1_results["qwen2.5-7b"] = qwen_results

    # ============================================================
    # Stage 2: Hypothesis Testing (Qwen + Mistral)
    # ============================================================
    print("\n" + "=" * 60)
    print("# Stage 2: Hypothesis Testing")
    print("=" * 60)

    # Qwen hypotheses (model already loaded)
    print("\n--- Qwen2.5-7B Hypotheses ---")
    try:
        hyp_results["rope_energy"]["qwen2.5-7b"] = hypothesis_a_rope_energy(
            model, tokenizer, config, "qwen2.5-7b"
        )
    except Exception as e:
        print(f"  Hypothesis A error (Qwen): {e}")

    try:
        hyp_results["key_norms"]["qwen2.5-7b"] = hypothesis_b_key_norms(
            model, tokenizer, config, "qwen2.5-7b"
        )
    except Exception as e:
        print(f"  Hypothesis B error (Qwen): {e}")

    try:
        hyp_results["entropy"]["qwen2.5-7b"] = hypothesis_c_attention_entropy(
            model, tokenizer, config, "qwen2.5-7b"
        )
    except Exception as e:
        print(f"  Hypothesis C error (Qwen): {e}")

    try:
        hyp_results["routing_kl"]["qwen2.5-7b"] = hypothesis_d_routing_kl(
            model, tokenizer, config, "qwen2.5-7b"
        )
    except Exception as e:
        print(f"  Hypothesis D error (Qwen): {e}")

    unload_model(model)

    # Mistral hypotheses
    print("\n--- Mistral-7B Hypotheses ---")
    model, tokenizer, config = load_model(MODELS_BASELINE["mistral-7b"])

    try:
        hyp_results["rope_energy"]["mistral-7b"] = hypothesis_a_rope_energy(
            model, tokenizer, config, "mistral-7b"
        )
    except Exception as e:
        print(f"  Hypothesis A error (Mistral): {e}")

    try:
        hyp_results["key_norms"]["mistral-7b"] = hypothesis_b_key_norms(
            model, tokenizer, config, "mistral-7b"
        )
    except Exception as e:
        print(f"  Hypothesis B error (Mistral): {e}")

    try:
        hyp_results["entropy"]["mistral-7b"] = hypothesis_c_attention_entropy(
            model, tokenizer, config, "mistral-7b"
        )
    except Exception as e:
        print(f"  Hypothesis C error (Mistral): {e}")

    try:
        hyp_results["routing_kl"]["mistral-7b"] = hypothesis_d_routing_kl(
            model, tokenizer, config, "mistral-7b"
        )
    except Exception as e:
        print(f"  Hypothesis D error (Mistral): {e}")

    unload_model(model)

    # ============================================================
    # Stage 3: Hypothesis Ranking
    # ============================================================
    rankings = rank_hypotheses(hyp_results, precision_floors)

    # ============================================================
    # Stage 4: Predict for new models
    # ============================================================
    print("\n" + "=" * 60)
    print("# Stage 4: Predict KV Precision for New Models")
    print("=" * 60)

    predictions = {}
    winning_hyp = rankings[0]["hypothesis"] if rankings else "unknown"
    print(f"  Using winning hypothesis: {winning_hyp}")

    for model_name, model_id in MODELS_NEW.items():
        print(f"\n--- Computing predictor for {model_name} ---")
        try:
            model, tokenizer, config = load_model(model_id)

            # Compute all hypothesis metrics for new model
            try:
                hyp_results["rope_energy"][model_name] = hypothesis_a_rope_energy(
                    model, tokenizer, config, model_name
                )
            except Exception as e:
                print(f"  Hypothesis A error ({model_name}): {e}")

            try:
                hyp_results["key_norms"][model_name] = hypothesis_b_key_norms(
                    model, tokenizer, config, model_name
                )
            except Exception as e:
                print(f"  Hypothesis B error ({model_name}): {e}")

            try:
                hyp_results["entropy"][model_name] = hypothesis_c_attention_entropy(
                    model, tokenizer, config, model_name
                )
            except Exception as e:
                print(f"  Hypothesis C error ({model_name}): {e}")

            try:
                hyp_results["routing_kl"][model_name] = hypothesis_d_routing_kl(
                    model, tokenizer, config, model_name
                )
            except Exception as e:
                print(f"  Hypothesis D error ({model_name}): {e}")

            # Simple prediction based on winning hypothesis
            # Compare new model's metric to known models
            if "rope" in winning_hyp.lower():
                key = "rope_energy"
                metric_key = "mean_high_freq_ratio"
            elif "key_norm" in winning_hyp.lower():
                key = "key_norms"
                metric_key = "outlier_ratio"
            elif "entropy" in winning_hyp.lower():
                key = "entropy"
                metric_key = "mean_entropy"
            elif "kl" in winning_hyp.lower():
                key = "routing_kl"
                metric_key = "mean_kl"
            else:
                key = "entropy"
                metric_key = "mean_entropy"

            new_val = hyp_results.get(key, {}).get(model_name, {}).get(metric_key, None)
            qwen_val = (
                hyp_results.get(key, {}).get("qwen2.5-7b", {}).get(metric_key, None)
            )
            mistral_val = (
                hyp_results.get(key, {}).get("mistral-7b", {}).get(metric_key, None)
            )

            if new_val is not None and qwen_val is not None and mistral_val is not None:
                # Interpolate: closer to Qwen = higher floor
                qwen_dist = abs(new_val - qwen_val)
                mistral_dist = abs(new_val - mistral_val)

                if qwen_dist < mistral_dist:
                    predictions[model_name] = "int8"
                    print(
                        f"  Predicted: {model_name} → INT8 " f"(closer to Qwen pattern)"
                    )
                else:
                    predictions[model_name] = "int4"
                    print(
                        f"  Predicted: {model_name} → INT4 "
                        f"(closer to Mistral pattern)"
                    )
            else:
                predictions[model_name] = "unknown"
                print(f"  Could not compute prediction for {model_name}")

            unload_model(model)

        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            predictions[model_name] = "error"

    save_json(
        {"predictions": predictions, "winning_hypothesis": winning_hyp},
        "predictions.json",
    )

    # ============================================================
    # Stage 5: Validation on New Models
    # ============================================================
    print("\n" + "=" * 60)
    print("# Stage 5: Validation on New Models")
    print("=" * 60)

    validation_results = {}
    validation_configs = [
        {"name": "K_INT8__V_INT4", "k_type": "int8", "v_type": "int4"},
        {"name": "K_INT7__V_INT4", "k_type": "int7", "v_type": "int4"},
        {"name": "K_INT6__V_INT4", "k_type": "int6", "v_type": "int4"},
    ]

    for model_name, model_id in MODELS_NEW.items():
        print(f"\n--- Validating {model_name} ---")
        try:
            model, tokenizer, config = load_model(model_id)
            model_results = []
            observed_floor = "int6"  # default assumption

            for cfg in validation_configs:
                print(f"\n  Config: {cfg['name']} ({model_name})")

                collapsed, collapse_info = early_collapse_check(
                    model,
                    tokenizer,
                    cfg["k_type"],
                    cfg["v_type"],
                    cfg["name"],
                )

                if collapsed:
                    model_results.append(
                        {
                            "name": cfg["name"],
                            "collapsed": True,
                            "collapse_info": collapse_info,
                        }
                    )
                    # If INT8 collapsed, floor is FP16
                    if cfg["k_type"] == "int8":
                        observed_floor = "fp16"
                    elif cfg["k_type"] == "int7" and observed_floor != "fp16":
                        observed_floor = "int8"
                    elif cfg["k_type"] == "int6" and observed_floor not in (
                        "fp16",
                        "int8",
                    ):
                        observed_floor = "int7"
                    continue

                # Quick eval (PPL + logit error only)
                ppl = compute_ppl(
                    model,
                    tokenizer,
                    cfg["k_type"],
                    cfg["v_type"],
                    cfg["name"],
                )
                logit_err = compute_logit_error(
                    model,
                    tokenizer,
                    cfg["k_type"],
                    cfg["v_type"],
                    cfg["name"],
                )

                model_results.append(
                    {
                        "name": cfg["name"],
                        "collapsed": False,
                        "ppl": ppl,
                        "logit_error": logit_err,
                    }
                )

                # Update observed floor
                if cfg["k_type"] == "int6" and not collapsed:
                    observed_floor = "int6"
                elif cfg["k_type"] == "int7" and not collapsed:
                    if observed_floor not in ("int6",):
                        observed_floor = "int7"

            # If all passed, floor might be lower
            all_passed = all(not r.get("collapsed") for r in model_results)
            if all_passed:
                observed_floor = "int6_or_lower"

            validation_results[model_name] = {
                "results": model_results,
                "observed_floor": observed_floor,
                "predicted_floor": predictions.get(model_name, "unknown"),
            }

            save_json(
                validation_results[model_name],
                f"validation_{model_name}.json",
            )

            print(
                f"\n  {model_name}: predicted={predictions.get(model_name)} "
                f"observed={observed_floor}"
            )

            # Store results for cross-model comparison
            all_stage1_results[model_name] = model_results
            precision_floors[model_name] = observed_floor

            unload_model(model)

        except Exception as e:
            print(f"  Error validating {model_name}: {e}")
            import traceback

            traceback.print_exc()
            validation_results[model_name] = {
                "error": str(e),
                "observed_floor": "error",
            }

    save_json(
        {"predicted": predictions, "validation": validation_results},
        "predicted_vs_actual_precision.json",
    )

    # ============================================================
    # Stage 6: Fast Estimator
    # ============================================================
    print("\n" + "=" * 60)
    print("# Stage 6: Fast KV Precision Estimator")
    print("=" * 60)

    estimator_result = {
        "method": "attention_entropy + key_norm_outlier_ratio",
        "description": (
            "Run 50 calibration prompts (512 tokens), measure "
            "attention entropy and key norm outlier ratio. "
            "If entropy < threshold_low AND outlier_ratio > "
            "threshold_high, predict FP16 floor. Otherwise "
            "predict INT4 floor."
        ),
        "calibration_data": {},
    }

    # Compute thresholds from known models
    for model_name in list(precision_floors.keys()):
        entropy_val = (
            hyp_results.get("entropy", {}).get(model_name, {}).get("mean_entropy", None)
        )
        outlier_val = (
            hyp_results.get("key_norms", {})
            .get(model_name, {})
            .get("outlier_ratio", None)
        )

        if entropy_val is not None and outlier_val is not None:
            estimator_result["calibration_data"][model_name] = {
                "entropy": entropy_val,
                "outlier_ratio": outlier_val,
                "precision_floor": precision_floors[model_name],
            }

    # Simple threshold: average of Qwen and Mistral
    qwen_entropy = (
        hyp_results.get("entropy", {}).get("qwen2.5-7b", {}).get("mean_entropy", None)
    )
    mistral_entropy = (
        hyp_results.get("entropy", {}).get("mistral-7b", {}).get("mean_entropy", None)
    )
    if qwen_entropy is not None and mistral_entropy is not None:
        estimator_result["entropy_threshold"] = round(
            (qwen_entropy + mistral_entropy) / 2, 4
        )
    else:
        estimator_result["entropy_threshold"] = None

    qwen_outlier = (
        hyp_results.get("key_norms", {})
        .get("qwen2.5-7b", {})
        .get("outlier_ratio", None)
    )
    mistral_outlier = (
        hyp_results.get("key_norms", {})
        .get("mistral-7b", {})
        .get("outlier_ratio", None)
    )
    if qwen_outlier is not None and mistral_outlier is not None:
        estimator_result["outlier_threshold"] = round(
            (qwen_outlier + mistral_outlier) / 2, 4
        )
    else:
        estimator_result["outlier_threshold"] = None

    save_json(estimator_result, "kv_precision_estimator.json")
    print(f"  Estimator saved with thresholds:")
    print(f"    entropy_threshold: {estimator_result.get('entropy_threshold')}")
    print(f"    outlier_threshold: {estimator_result.get('outlier_threshold')}")

    # ============================================================
    # Final: Architecture comparison plot + summary
    # ============================================================
    plot_architecture_comparison(all_stage1_results)

    # Re-rank with all models
    rankings = rank_hypotheses(hyp_results, precision_floors)

    runtime = time.time() - t0
    write_summary(
        all_stage1_results,
        hyp_results,
        rankings,
        predictions,
        validation_results,
        estimator_result,
        runtime,
    )

    print(f"\n{'=' * 60}")
    print(f"BPA v50 COMPLETE")
    print(f"Total runtime: {runtime:.0f}s ({runtime/3600:.1f} hours)")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'=' * 60}")

    log_file.close()


if __name__ == "__main__":
    main()
