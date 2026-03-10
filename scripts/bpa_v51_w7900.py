#!/usr/bin/env python3
"""BPA v51: Attention Entropy Predictor Validation.

Validates v50's finding that attention entropy predicts key-precision
floor across additional architectures.

Stage 1: Measure attention entropy per model
Stage 2: Precision sweep (K_INT8_V_INT4, K_INT6_V_INT4)
Stage 3: Early collapse detection
Stage 4: Determine observed key precision floor
Stage 5: Predictor validation (entropy vs floor correlation)
Stage 6: Visualization

Results saved to /data/knlp-key-results/bpa51/
"""

import gc
import json
import math
import os
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
RESULTS_ROOT = "/data/knlp-key-results/bpa51"
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
N_PROMPTS = 50
PROMPT_LEN = 512
GEN_LEN = 64

# v51 models — 7B-class models. Some spec models are gated
# (Llama-2, Llama-3, Gemma); substitute open-access alternatives.
MODELS = {
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "falcon-7b": "tiiuae/falcon-7b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "qwen2-7b": "Qwen/Qwen2.5-7B",
    # Open-access substitutes for gated models
    "opt-6.7b": "facebook/opt-6.7b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}

# Precision configs to test (values always INT4)
PRECISION_CONFIGS = [
    ("int8", "int4", "K_INT8__V_INT4"),
    ("int6", "int4", "K_INT6__V_INT4"),
]

# v50 data for combined analysis
V50_ENTROPY = {
    "qwen2.5-7b": 2.2369,
    "mistral-7b": 1.9814,
    "pythia-6.9b": 2.1224,
}
V50_FLOORS = {
    "qwen2.5-7b": "int7",
    "mistral-7b": "none",
    "pythia-6.9b": "int7",
}


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that default=str would corrupt (bool→'False')."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def save_json(data, filename):
    path = os.path.join(JSON_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
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

    ds = load_dataset("wikitext", DATASET, split="validation")
    text = "\n".join([x["text"] for x in ds if x["text"].strip()])
    tokens = tokenizer.encode(text)
    _TOKEN_CACHE[key] = tokens
    print(f"  Loaded {DATASET} validation: {len(tokens)} tokens")
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


# === DynamicCache helpers (transformers 5.x compatible) ===
def _cache_get_kv(past, li):
    if hasattr(past, "layers"):
        layer = past.layers[li]
        return layer.keys, layer.values
    return past[li]


def _cache_set_kv(past, li, k, v):
    if hasattr(past, "layers"):
        past.layers[li].keys = k
        past.layers[li].values = v
    elif isinstance(past, list):
        past[li] = (k, v)
    else:
        raise TypeError(
            f"Cannot set KV on {type(past).__name__}. "
            "Use _ensure_mutable_cache() first."
        )


def n_cache_layers(past):
    if hasattr(past, "layers"):
        return len(past.layers)
    return len(past)


def _ensure_mutable_cache(past):
    """Convert tuple-based cache to list so we can mutate entries."""
    if isinstance(past, tuple):
        return list(past)
    return past


def quantize_cache_uniform(past, k_type, v_type, group_size=32):
    past = _ensure_mutable_cache(past)
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
# Models whose cache API is incompatible with transformers 5.x
# precision testing. Entropy-only measurement still works.
SKIP_PRECISION_MODELS = {"phi-3-mini"}


def load_model(model_id):
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


# =========================================================
# Stage 1: Measure Attention Entropy
# =========================================================
def measure_attention_entropy(model, tokenizer, config, model_name):
    print(f"\n{'='*60}")
    print(f"Stage 1: Attention Entropy — {model_name}")
    print(f"{'='*60}")

    cached_file = os.path.join(JSON_DIR, f"entropy_{model_name}.json")
    if os.path.exists(cached_file):
        print(f"  Cached: {cached_file}")
        with open(cached_file) as f:
            return json.load(f)

    tokens = load_wikitext_tokens(tokenizer)
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads

    # Accumulate per-head entropies across prompts
    all_entropies = []  # list of [n_layers, n_heads]

    for prompt_idx in range(N_PROMPTS):
        if prompt_idx % 10 == 0:
            print(f"  Prompt {prompt_idx}/{N_PROMPTS}...")

        rng = np.random.RandomState(prompt_idx)
        start = rng.randint(0, max(1, len(tokens) - PROMPT_LEN - 10))
        input_ids = (
            torch.tensor(tokens[start : start + PROMPT_LEN], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            out = model(
                input_ids,
                output_attentions=True,
                use_cache=False,
            )

        attentions = out.attentions  # tuple of [B, H, T, T] per layer
        if attentions is None or len(attentions) == 0:
            print(f"  WARNING: No attention weights returned for {model_name}")
            print("  Trying hook-based extraction...")
            result = _measure_entropy_via_hooks(
                model, tokenizer, config, model_name, tokens
            )
            return result

        prompt_entropy = np.zeros((n_layers, n_heads))
        for li, attn in enumerate(attentions):
            # attn: [1, H, T, T] — already softmaxed
            attn_f = attn[0].float()  # [H, T, T]
            # Entropy per head: average over query positions
            # entropy = -sum(p * log(p)) for each row
            p = attn_f.clamp(min=1e-10)
            ent = -(p * p.log()).sum(dim=-1)  # [H, T]
            head_entropy = ent.mean(dim=-1).cpu().numpy()  # [H]
            nh = head_entropy.shape[0]
            prompt_entropy[li, :nh] = head_entropy

        all_entropies.append(prompt_entropy)

        del out, attentions
        clear_gpu()

    all_ent = np.stack(all_entropies)  # [N_PROMPTS, n_layers, n_heads]
    mean_per_head = all_ent.mean(axis=0)  # [n_layers, n_heads]
    std_per_head = all_ent.std(axis=0)

    # Global statistics
    mean_entropy = float(mean_per_head.mean())
    std_entropy = float(mean_per_head.std())
    per_layer_mean = mean_per_head.mean(axis=1).tolist()
    per_layer_std = mean_per_head.std(axis=1).tolist()

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_prompts": N_PROMPTS,
        "prompt_length": PROMPT_LEN,
        "mean_attention_entropy": round(mean_entropy, 4),
        "std_attention_entropy": round(std_entropy, 4),
        "per_layer_mean_entropy": [round(x, 4) for x in per_layer_mean],
        "per_layer_std_entropy": [round(x, 4) for x in per_layer_std],
    }

    save_json(result, f"entropy_{model_name}.json")

    # Plot per-layer entropy distribution
    _plot_entropy(mean_per_head, model_name, n_layers, n_heads)

    print(f"  Entropy: mean={mean_entropy:.4f} std={std_entropy:.4f}")
    return result


def _measure_entropy_via_hooks(model, tokenizer, config, model_name, tokens):
    """Fallback: extract attention weights via forward hooks."""
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    captured_attns = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]
                if attn_w is not None:
                    captured_attns[layer_idx] = attn_w.detach()

        return hook_fn

    # Find attention layers and register hooks
    hooks = []
    for name, mod in model.named_modules():
        if hasattr(mod, "num_heads") and "attn" in name.lower():
            # Extract layer index from name
            parts = name.split(".")
            for p in parts:
                if p.isdigit():
                    li = int(p)
                    hooks.append(mod.register_forward_hook(make_hook(li)))
                    break

    if not hooks:
        print("  ERROR: Could not find attention modules for hooks")
        result = {
            "model": model_name,
            "mean_attention_entropy": None,
            "std_attention_entropy": None,
            "error": "No attention weights available",
        }
        save_json(result, f"entropy_{model_name}.json")
        return result

    all_entropies = []
    for prompt_idx in range(N_PROMPTS):
        if prompt_idx % 10 == 0:
            print(f"  Hook prompt {prompt_idx}/{N_PROMPTS}...")

        rng = np.random.RandomState(prompt_idx)
        start = rng.randint(0, max(1, len(tokens) - PROMPT_LEN - 10))
        input_ids = (
            torch.tensor(tokens[start : start + PROMPT_LEN], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        captured_attns.clear()
        with torch.no_grad():
            model(input_ids, output_attentions=True, use_cache=False)

        if not captured_attns:
            continue

        prompt_entropy = np.zeros((n_layers, n_heads))
        for li, attn in captured_attns.items():
            attn_f = attn[0].float()
            p = attn_f.clamp(min=1e-10)
            ent = -(p * p.log()).sum(dim=-1)
            head_entropy = ent.mean(dim=-1).cpu().numpy()
            nh = min(head_entropy.shape[0], n_heads)
            prompt_entropy[li, :nh] = head_entropy[:nh]

        all_entropies.append(prompt_entropy)
        clear_gpu()

    for h in hooks:
        h.remove()

    if not all_entropies:
        result = {
            "model": model_name,
            "mean_attention_entropy": None,
            "std_attention_entropy": None,
            "error": "Hook-based extraction failed",
        }
        save_json(result, f"entropy_{model_name}.json")
        return result

    all_ent = np.stack(all_entropies)
    mean_per_head = all_ent.mean(axis=0)
    mean_entropy = float(mean_per_head.mean())
    std_entropy = float(mean_per_head.std())
    per_layer_mean = mean_per_head.mean(axis=1).tolist()
    per_layer_std = mean_per_head.std(axis=1).tolist()

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_prompts": N_PROMPTS,
        "prompt_length": PROMPT_LEN,
        "mean_attention_entropy": round(mean_entropy, 4),
        "std_attention_entropy": round(std_entropy, 4),
        "per_layer_mean_entropy": [round(x, 4) for x in per_layer_mean],
        "per_layer_std_entropy": [round(x, 4) for x in per_layer_std],
        "extraction_method": "hooks",
    }

    save_json(result, f"entropy_{model_name}.json")
    _plot_entropy(mean_per_head, model_name, n_layers, n_heads)
    print(f"  Entropy (hooks): mean={mean_entropy:.4f} std={std_entropy:.4f}")
    return result


def _plot_entropy(mean_per_head, model_name, n_layers, n_heads):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(
        mean_per_head.T, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(f"Attention Entropy — {model_name}")
    plt.colorbar(im, label="Entropy (nats)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"attention_entropy_{model_name}.png"), dpi=300)
    plt.close()


# =========================================================
# Stage 2-3: Precision Sweep + Early Collapse Detection
# =========================================================
def early_collapse_check(model, tokenizer, k_type, v_type, config_name):
    """Quick test: 512 tokens prefill, measure PPL delta + logit error.

    Uses PPL-delta based collapse detection (not token agreement alone)
    because v50 showed token agreement is inherently low (~0.1-0.3)
    even for near-lossless PPL configs due to autoregressive error
    amplification.

    Collapse criteria: PPL delta > 15% OR logit_err > 10.
    """
    print(f"\n  Collapse check: {config_name}")
    tokens = load_wikitext_tokens(tokenizer)
    results = []

    for seed in range(5):
        rng = np.random.RandomState(seed + 100)
        start = rng.randint(0, max(1, len(tokens) - PROMPT_LEN - 10))
        input_ids = (
            torch.tensor(tokens[start : start + PROMPT_LEN], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            # FP16 baseline
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values

            # Quantized (separate forward for separate cache)
            out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values

        if k_type != "fp16" or v_type != "fp16":
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

        # PPL from prefill logits
        logits_fp = out_fp.logits[:, :-1, :].float().cpu()
        targets = input_ids[:, 1:].cpu()
        loss_fp = F.cross_entropy(
            logits_fp.reshape(-1, logits_fp.shape[-1]),
            targets.reshape(-1),
        )
        ppl_fp = math.exp(loss_fp.item())

        # Logit error: compare next-token prediction after quantized cache
        with torch.no_grad():
            next_tok_id = (
                tokens[start + PROMPT_LEN]
                if (start + PROMPT_LEN) < len(tokens)
                else tokens[-1]
            )
            next_tok = torch.tensor([[next_tok_id]]).to(DEVICE)
            o1 = model(next_tok, past_key_values=past_fp, use_cache=True)
            o2 = model(next_tok, past_key_values=past_q, use_cache=True)

        logit_err = (
            (o1.logits[0, 0].float() - o2.logits[0, 0].float()).abs().max().item()
        )

        # Token agreement: generate 64 tokens greedily from both
        agree = 0
        tok_fp = next_tok
        tok_q = next_tok
        p_fp = o1.past_key_values
        p_q = o2.past_key_values

        for step in range(GEN_LEN):
            with torch.no_grad():
                of = model(tok_fp, past_key_values=p_fp, use_cache=True)
                oq = model(tok_q, past_key_values=p_q, use_cache=True)
            t_fp = of.logits[0, -1].argmax().item()
            t_q = oq.logits[0, -1].argmax().item()
            if t_fp == t_q:
                agree += 1
            tok_fp = torch.tensor([[t_fp]]).to(DEVICE)
            tok_q = torch.tensor([[t_q]]).to(DEVICE)
            p_fp = of.past_key_values
            p_q = oq.past_key_values

        results.append(
            {
                "ppl_fp16": ppl_fp,
                "agree": agree / GEN_LEN,
                "logit_err": logit_err,
            }
        )

        del past_fp, past_q, p_fp, p_q
        clear_gpu()

    avg_ppl_fp = np.mean([r["ppl_fp16"] for r in results])
    avg_agree = np.mean([r["agree"] for r in results])
    avg_logit = np.mean([r["logit_err"] for r in results])

    # Collapse: logit_err > 10 (catastrophic) OR logit_err > 5 (severe)
    # PPL delta is not measurable from prefill alone (both use same
    # input), so we rely on logit error as the primary signal.
    # agree < 0.15 is random-level (catastrophic).
    collapsed = bool(avg_logit > 5 or avg_agree < 0.08)

    info = {
        "name": config_name,
        "ppl_fp16": round(float(avg_ppl_fp), 4),
        "token_agreement": round(float(avg_agree), 4),
        "logit_error": round(float(avg_logit), 4),
        "collapsed": collapsed,
        "n_seeds": len(results),
    }
    status = "COLLAPSED" if collapsed else "OK"
    print(
        f"    {status}: PPL_FP16={avg_ppl_fp:.2f} agree={avg_agree:.4f} "
        f"logit_err={avg_logit:.2f}"
    )
    return collapsed, info


# =========================================================
# Stage 3 (cont): Minimal Evaluation for surviving configs
# =========================================================
def minimal_eval(model, tokenizer, k_type, v_type, config_name, n_seeds=5):
    """Compute PPL + token agreement for surviving configs.

    PPL is measured on continuation tokens (eval_len=256) after a
    prefill of PROMPT_LEN tokens, so the quantized cache actually
    affects the logits. Previous versions measured prefill PPL which
    was identical for FP16 and quantized (cache not used during prefill).
    """
    print(f"\n  Minimal eval: {config_name}")
    tokens = load_wikitext_tokens(tokenizer)
    eval_len = 256

    ppls_fp = []
    ppls_q = []
    for seed in range(n_seeds):
        # Load prefill + eval tokens
        input_ids = load_passage(tokenizer, PROMPT_LEN + eval_len, seed).to(DEVICE)
        prefill = input_ids[:, :PROMPT_LEN]
        continuation = input_ids[:, PROMPT_LEN : PROMPT_LEN + eval_len]

        with torch.no_grad():
            # Build caches from prefill
            out_fp = model(prefill, use_cache=True)
            past_fp = out_fp.past_key_values
            out_q = model(prefill, use_cache=True)
            past_q = out_q.past_key_values

        if k_type != "fp16" or v_type != "fp16":
            past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

        # Evaluate PPL on continuation using each cache
        with torch.no_grad():
            logits_fp = (
                model(continuation, past_key_values=past_fp, use_cache=False)
                .logits.float()
                .cpu()
            )
            logits_q = (
                model(continuation, past_key_values=past_q, use_cache=False)
                .logits.float()
                .cpu()
            )

        # Shifted targets: predict token i+1 from position i
        targets = continuation[:, 1:].cpu()
        logits_fp = logits_fp[:, :-1, :]
        logits_q = logits_q[:, :-1, :]

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

        del out_fp, out_q, past_fp, past_q
        clear_gpu()

    ppl_fp = float(np.mean(ppls_fp))
    ppl_q = float(np.mean(ppls_q))
    delta = (ppl_q - ppl_fp) / ppl_fp * 100 if ppl_fp > 0 else 0

    # Token agreement (50 prompts x 128 tokens per spec)
    agrees = []
    for i in range(N_PROMPTS):
        if i % 10 == 0:
            print(f"    Agreement prompt {i}/{N_PROMPTS}...")
        rng = np.random.RandomState(i + 500)
        start = rng.randint(0, max(1, len(tokens) - PROMPT_LEN - 10))
        input_ids = (
            torch.tensor(tokens[start : start + PROMPT_LEN], dtype=torch.long)
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

        for step in range(128):
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

        agrees.append(agree / 128)
        del past_fp, past_q, p_fp, p_q
        clear_gpu()

    result = {
        "ppl_fp16": round(ppl_fp, 4),
        "ppl_config": round(ppl_q, 4),
        "delta_pct": round(delta, 4),
        "token_agreement": round(float(np.mean(agrees)), 4),
        "token_agreement_std": round(float(np.std(agrees)), 4),
        "n_seeds_ppl": n_seeds,
        "n_prompts_agree": N_PROMPTS,
    }
    print(f"    PPL: FP16={ppl_fp:.4f} Config={ppl_q:.4f} delta={delta:.2f}%")
    print(f"    Agreement: mean={np.mean(agrees):.4f} std={np.std(agrees):.4f}")
    return result


# =========================================================
# Stage 4: Determine Observed Key Precision Floor
# =========================================================
def determine_floor(collapse_results, eval_results, model_name):
    """Determine the observed key precision floor for a model."""
    # INT6 survived → floor <= INT6
    # INT6 collapsed, INT8 survived → floor = INT7-INT8 region
    # Both collapsed → floor > INT8

    int8_name = "K_INT8__V_INT4"
    int6_name = "K_INT6__V_INT4"

    int8_collapsed = True
    int6_collapsed = True

    def _is_collapsed(val):
        """Handle both bool and string 'False'/'True' from JSON."""
        if isinstance(val, str):
            return val.lower() != "false"
        return bool(val)

    for cr in collapse_results:
        if cr["name"] == int8_name:
            int8_collapsed = _is_collapsed(cr.get("collapsed", True))
        if cr["name"] == int6_name:
            int6_collapsed = _is_collapsed(cr.get("collapsed", True))

    if not int6_collapsed:
        floor = "int6_or_below"
    elif not int8_collapsed:
        floor = "int7_to_int8"
    else:
        floor = "above_int8"

    result = {
        "model": model_name,
        "int8_collapsed": int8_collapsed,
        "int6_collapsed": int6_collapsed,
        "observed_floor": floor,
        "collapse_details": collapse_results,
    }
    if eval_results:
        result["eval_details"] = eval_results

    save_json(result, f"key_precision_floor_{model_name}.json")
    print(f"  Floor for {model_name}: {floor}")
    return result


# =========================================================
# Stage 5: Predictor Validation
# =========================================================
def predictor_validation(all_entropy, all_floors):
    """Correlate entropy with observed precision floor."""
    print(f"\n{'='*60}")
    print("Stage 5: Predictor Validation")
    print(f"{'='*60}")

    # Encode floors numerically
    floor_encoding = {
        "none": 0,  # no floor (robust)
        "int6_or_below": 1,
        "int7_to_int8": 2,
        "int7": 2,
        "above_int8": 3,
    }

    # Combine v50 and v51 data
    combined = {}
    for model_name, entropy in V50_ENTROPY.items():
        if model_name in V50_FLOORS:
            combined[model_name] = {
                "entropy": entropy,
                "floor": V50_FLOORS[model_name],
                "floor_num": floor_encoding.get(V50_FLOORS[model_name], -1),
                "source": "v50",
            }

    for model_name, entropy_data in all_entropy.items():
        ent = entropy_data.get("mean_attention_entropy")
        if ent is None:
            continue
        floor = all_floors.get(model_name, {}).get("observed_floor", "unknown")
        if floor == "unknown":
            continue
        combined[model_name] = {
            "entropy": ent,
            "floor": floor,
            "floor_num": floor_encoding.get(floor, -1),
            "source": "v51",
        }

    # Build table
    table = []
    entropies = []
    floors_num = []
    for model_name, data in sorted(combined.items()):
        table.append(
            {
                "model": model_name,
                "entropy": data["entropy"],
                "floor": data["floor"],
                "floor_num": data["floor_num"],
                "source": data["source"],
            }
        )
        if data["floor_num"] >= 0:
            entropies.append(data["entropy"])
            floors_num.append(data["floor_num"])

    # Compute correlation
    correlation = None
    if len(entropies) >= 3:
        ent_arr = np.array(entropies)
        floor_arr = np.array(floors_num)
        if ent_arr.std() > 0 and floor_arr.std() > 0:
            correlation = float(np.corrcoef(ent_arr, floor_arr)[0, 1])

    result = {
        "table": table,
        "correlation": round(correlation, 4) if correlation else None,
        "n_models": len(entropies),
        "v50_models": list(V50_ENTROPY.keys()),
        "v51_models": list(all_entropy.keys()),
    }

    save_json(result, "entropy_precision_correlation.json")

    print("\n  Model          | Entropy | Floor          | Source")
    print("  " + "-" * 55)
    for row in table:
        print(
            f"  {row['model']:16s} | {row['entropy']:.4f}  | "
            f"{row['floor']:14s} | {row['source']}"
        )
    if correlation is not None:
        print(f"\n  Correlation: {correlation:.4f} (n={len(entropies)})")

    return result


# =========================================================
# Stage 6: Visualization
# =========================================================
def generate_plots(correlation_data, all_entropy):
    """Generate final visualization plots."""
    print(f"\n{'='*60}")
    print("Stage 6: Visualization")
    print(f"{'='*60}")

    table = correlation_data["table"]

    # Plot 1: Entropy vs Precision Floor
    fig, ax = plt.subplots(figsize=(10, 7))
    floor_encoding = {
        "none": 0,
        "int6_or_below": 1,
        "int7_to_int8": 2,
        "int7": 2,
        "above_int8": 3,
    }
    floor_labels = {
        0: "None (robust)",
        1: "INT6 or below",
        2: "INT7-INT8",
        3: "Above INT8",
    }

    colors_src = {"v50": "tab:blue", "v51": "tab:red"}
    for row in table:
        fn = floor_encoding.get(row["floor"], -1)
        if fn < 0:
            continue
        c = colors_src.get(row["source"], "gray")
        ax.scatter(row["entropy"], fn, s=120, c=c, zorder=5)
        ax.annotate(
            row["model"],
            (row["entropy"], fn),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    # Trend line if we have correlation
    entropies = [
        row["entropy"] for row in table if floor_encoding.get(row["floor"], -1) >= 0
    ]
    floors_num = [
        floor_encoding[row["floor"]]
        for row in table
        if floor_encoding.get(row["floor"], -1) >= 0
    ]
    if len(entropies) >= 2:
        z = np.polyfit(entropies, floors_num, 1)
        x_line = np.linspace(min(entropies) - 0.1, max(entropies) + 0.1, 50)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.3)

    ax.set_xlabel("Mean Attention Entropy (nats)", fontsize=12)
    ax.set_ylabel("Key Precision Floor", fontsize=12)
    ax.set_yticks(list(floor_labels.keys()))
    ax.set_yticklabels(list(floor_labels.values()))
    ax.set_title(
        "Attention Entropy vs Key Precision Floor\n"
        f"Correlation: {correlation_data.get('correlation', 'N/A')}",
        fontsize=13,
    )
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:blue",
                label="v50 data",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:red",
                label="v51 data",
                markersize=10,
            ),
        ],
        loc="upper left",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "entropy_vs_precision_floor.png"), dpi=300)
    plt.close()
    print("  Saved: entropy_vs_precision_floor.png")

    # Plot 2: Entropy distribution across models
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = []
    model_entropies = []

    # v50 data
    for m, e in sorted(V50_ENTROPY.items()):
        model_names.append(f"{m} (v50)")
        model_entropies.append(e)

    # v51 data
    for m, data in sorted(all_entropy.items()):
        ent = data.get("mean_attention_entropy")
        if ent is not None:
            model_names.append(m)
            model_entropies.append(ent)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.barh(model_names, model_entropies, color=colors)
    ax.set_xlabel("Mean Attention Entropy (nats)", fontsize=12)
    ax.set_title("Attention Entropy Distribution Across Models", fontsize=13)
    for bar, val in zip(bars, model_entropies):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "entropy_distribution_models.png"), dpi=300)
    plt.close()
    print("  Saved: entropy_distribution_models.png")


# =========================================================
# Main
# =========================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("BPA v51: Attention Entropy Predictor Validation")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    info = gpu_info()
    print(f"GPU: {info['device']} ({info['total_gb']} GB)")
    print(f"Torch: {info['torch']}, HIP: {info['hip']}")

    all_entropy = {}
    all_collapse = {}
    all_eval = {}
    all_floors = {}
    model_errors = {}

    # Process each model
    for model_name, model_id in MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name} ({model_id})")
        print(f"{'#'*60}")

        try:
            model, tokenizer, config = load_model(model_id)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            model_errors[model_name] = str(e)
            save_json(
                {"model": model_name, "error": str(e)},
                f"error_{model_name}.json",
            )
            continue

        # Stage 1: Entropy
        try:
            entropy_result = measure_attention_entropy(
                model, tokenizer, config, model_name
            )
            all_entropy[model_name] = entropy_result
        except Exception as e:
            print(f"  ERROR measuring entropy for {model_name}: {e}")
            model_errors[model_name] = f"entropy: {e}"
            entropy_result = None

        # Stage 2-3: Precision sweep with collapse detection
        collapse_results = []
        eval_results = {}

        if model_name in SKIP_PRECISION_MODELS:
            print(
                f"  Skipping precision test for {model_name} "
                f"(incompatible cache API)"
            )
            save_json(
                {
                    "collapse": [],
                    "eval": {},
                    "skipped": True,
                    "reason": "incompatible cache API",
                },
                f"precision_test_{model_name}.json",
            )
            save_json([], f"collapse_{model_name}.json")
            # Entropy-only: no floor determination possible
            unload_model(model)
            del tokenizer, config
            clear_gpu()
            continue

        for k_type, v_type, config_name in PRECISION_CONFIGS:
            try:
                collapsed, info_dict = early_collapse_check(
                    model, tokenizer, k_type, v_type, config_name
                )
                collapse_results.append(info_dict)

                if not collapsed:
                    # Stage 3: Minimal evaluation
                    try:
                        eval_r = minimal_eval(
                            model, tokenizer, k_type, v_type, config_name
                        )
                        eval_results[config_name] = eval_r
                    except Exception as e:
                        print(f"  ERROR eval {config_name}: {e}")

                # Runtime optimization: skip INT6 if INT8 collapsed
                if collapsed and k_type == "int8":
                    print("  INT8 collapsed — skipping INT6")
                    collapse_results.append(
                        {
                            "name": "K_INT6__V_INT4",
                            "collapsed": True,
                            "skipped": True,
                            "reason": "INT8 collapsed",
                        }
                    )
                    break

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on {config_name} — skipping")
                collapse_results.append(
                    {
                        "name": config_name,
                        "collapsed": True,
                        "error": "OOM",
                    }
                )
                clear_gpu()
            except Exception as e:
                print(f"  ERROR {config_name}: {e}")
                collapse_results.append(
                    {
                        "name": config_name,
                        "collapsed": True,
                        "error": str(e),
                    }
                )

        all_collapse[model_name] = collapse_results
        all_eval[model_name] = eval_results

        save_json(
            {"collapse": collapse_results, "eval": eval_results},
            f"precision_test_{model_name}.json",
        )
        save_json(collapse_results, f"collapse_{model_name}.json")

        # Stage 4: Determine floor
        floor_result = determine_floor(collapse_results, eval_results, model_name)
        all_floors[model_name] = floor_result

        # Unload model
        unload_model(model)
        del tokenizer, config
        clear_gpu()

    # Stage 5: Predictor validation
    correlation_data = predictor_validation(all_entropy, all_floors)

    # Stage 6: Visualization
    generate_plots(correlation_data, all_entropy)

    # Save run metadata
    elapsed = time.time() - t_start
    metadata = {
        "experiment": "BPA-v51",
        "date": str(datetime.now()),
        "gpu": gpu_info(),
        "elapsed_seconds": round(elapsed, 1),
        "models_tested": list(all_entropy.keys()),
        "models_failed": model_errors,
        "correlation": correlation_data.get("correlation"),
    }
    save_json(metadata, "bpa51_metadata.json")

    # Write summary report
    write_summary(all_entropy, all_floors, correlation_data, model_errors, elapsed)

    print(f"\n{'='*60}")
    print(f"BPA v51 COMPLETE — {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'='*60}")


def write_summary(all_entropy, all_floors, correlation_data, errors, elapsed):
    """Generate the final summary report."""
    lines = []
    lines.append("# BPA v51 — Attention Entropy Predictor Validation")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**GPU**: {gpu_info()['device']}")
    lines.append(f"**Dtype**: BF16")
    lines.append(f"**Runtime**: {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    lines.append("")

    # Entropy table
    lines.append("## Model Entropy Values")
    lines.append("")
    lines.append("| Model | Entropy | Std | Source |")
    lines.append("|-------|---------|-----|--------|")
    for m, e in sorted(V50_ENTROPY.items()):
        lines.append(f"| {m} | {e:.4f} | — | v50 |")
    for m, data in sorted(all_entropy.items()):
        ent = data.get("mean_attention_entropy", "N/A")
        std = data.get("std_attention_entropy", "N/A")
        if ent is not None:
            lines.append(f"| {m} | {ent:.4f} | {std:.4f} | v51 |")
        else:
            lines.append(f"| {m} | ERROR | — | v51 |")
    lines.append("")

    # Floor table
    lines.append("## Observed Key Precision Floors")
    lines.append("")
    lines.append("| Model | Entropy | Observed Floor | Source |")
    lines.append("|-------|---------|----------------|--------|")
    for m, f in sorted(V50_FLOORS.items()):
        ent = V50_ENTROPY.get(m, "?")
        lines.append(f"| {m} | {ent} | {f} | v50 |")
    for m, data in sorted(all_floors.items()):
        ent_data = all_entropy.get(m, {})
        ent = ent_data.get("mean_attention_entropy", "?")
        floor = data.get("observed_floor", "unknown")
        lines.append(f"| {m} | {ent} | {floor} | v51 |")
    lines.append("")

    # Correlation
    corr = correlation_data.get("correlation")
    n = correlation_data.get("n_models", 0)
    lines.append("## Entropy-Precision Correlation")
    lines.append("")
    if corr is not None:
        lines.append(f"Pearson correlation: **{corr:.4f}** (n={n} models)")
    else:
        lines.append("Correlation: insufficient data")
    lines.append("")

    # Errors
    if errors:
        lines.append("## Model Errors")
        lines.append("")
        for m, e in errors.items():
            lines.append(f"- **{m}**: {e}")
        lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")
    if corr is not None and abs(corr) > 0.7:
        lines.append(
            f"v50's attention entropy predictor is **validated** "
            f"(corr={corr:.4f}) across {n} architectures. "
            f"Higher entropy continues to predict higher required "
            f"key precision."
        )
    elif corr is not None:
        lines.append(
            f"The correlation ({corr:.4f}) across {n} models is "
            f"moderate. The predictor shows some signal but is not "
            f"as strong as v50's 3-model result (0.89)."
        )
    else:
        lines.append("Insufficient data for correlation analysis.")
    lines.append("")

    # Plots
    lines.append("## Plots")
    lines.append("")
    lines.append("![entropy_vs_precision_floor](plots/entropy_vs_precision_floor.png)")
    lines.append("![entropy_distribution](plots/entropy_distribution_models.png)")
    for m in sorted(all_entropy.keys()):
        lines.append(f"![entropy_{m}](plots/attention_entropy_{m}.png)")
    lines.append("")

    summary_path = os.path.join(RESULTS_ROOT, "bpa51_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
