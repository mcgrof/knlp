#!/usr/bin/env python3
"""BPA v52: GQA Key-Sharing Quantization Sensitivity Hypothesis.

Tests whether GQA ratio (query heads per KV head) predicts KV cache
key quantization sensitivity. v49-v51 showed only Qwen models have
elevated key precision floors; v52 tests whether GQA key-sharing
amplifies quantization noise across grouped query heads.

Stage 1: Architecture census (config-only, no GPU)
Stage 2: Quick calibration (INT8/INT6/INT4 keys, V=INT4)
Stage 3: GQA spread measurement (attention weight extraction)
Stage 4: Correlation analysis
Stage 5: Binary classifier validation
Stage 6: Summary report + plots

Results saved to /data/knlp-key-results/bpa52/
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
RESULTS_ROOT = "/data/knlp-key-results/bpa52"
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

# Stage 2: quick calibration
N_CAL_PROMPTS = 10
CAL_PROMPT_LEN = 2048
GEN_LEN = 64

# Stage 3: GQA spread
N_SPREAD_PROMPTS = 10
SPREAD_PROMPT_LEN = 512

# Models — use base models for consistency with v49-v51.
# Spec says Instruct variants, but base models match prior data.
MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "falcon-7b": "tiiuae/falcon-7b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "opt-6.7b": "facebook/opt-6.7b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma-2-9b": "google/gemma-2-9b",
    "phi-2": "microsoft/phi-2",
}

# Precision configs: keys swept, values always INT4
PRECISION_CONFIGS = [
    ("int8", "int4", "K_INT8__V_INT4"),
    ("int6", "int4", "K_INT6__V_INT4"),
    ("int4", "int4", "K_INT4__V_INT4"),
]

# Prior entropy data from v50/v51 for correlation analysis
PRIOR_ENTROPY = {
    "qwen2.5-7b": 2.2571,
    "mistral-7b": 2.1294,
    "falcon-7b": 1.6012,
    "pythia-6.9b": 2.2147,
    "opt-6.7b": 2.4054,
}

# Models whose cache API is incompatible with precision testing
SKIP_PRECISION_MODELS = set()


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that default=str would corrupt (bool->'False')."""

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


def load_cached_json(filename):
    path = os.path.join(JSON_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


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


# === Quantization (same as v49-v51) ===
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
def load_model(model_id):
    print(f"Loading {model_id}...")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=True)
    config._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        token=True,
    ).to(DEVICE)
    model.eval()

    n_layers = config.num_hidden_layers
    print(f"  Model loaded: {n_layers} layers, device={DEVICE}")
    return model, tokenizer, config


def unload_model(model):
    del model
    clear_gpu()


# =========================================================
# Stage 1: Architecture Census
# =========================================================
def architecture_census():
    """Extract architecture parameters from configs (no GPU needed)."""
    print(f"\n{'='*60}")
    print("Stage 1: Architecture Census")
    print(f"{'='*60}")

    cached = load_cached_json("architecture_census.json")
    if cached:
        print("  Cached: architecture_census.json")
        return cached

    census = {}
    for model_name, model_id in MODELS.items():
        print(f"  {model_name}: ", end="", flush=True)
        try:
            cfg = AutoConfig.from_pretrained(
                model_id, trust_remote_code=True, token=True
            )
            n_heads = cfg.num_attention_heads
            n_kv = getattr(cfg, "num_key_value_heads", n_heads)
            hidden = cfg.hidden_size
            head_dim = hidden // n_heads
            rope_theta = getattr(
                cfg, "rope_theta", getattr(cfg, "rotary_emb_base", None)
            )
            n_layers = cfg.num_hidden_layers

            gqa_ratio = n_heads / n_kv if n_kv > 0 else 1.0

            census[model_name] = {
                "model_id": model_id,
                "num_attention_heads": n_heads,
                "num_key_value_heads": n_kv,
                "gqa_ratio": round(gqa_ratio, 2),
                "head_dim": head_dim,
                "rope_theta": rope_theta,
                "hidden_size": hidden,
                "num_layers": n_layers,
                "is_gqa": gqa_ratio > 1.0,
            }
            print(
                f"Q={n_heads} KV={n_kv} GQA={gqa_ratio:.1f}x "
                f"d={head_dim} layers={n_layers}"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            census[model_name] = {"model_id": model_id, "error": str(e)}

    save_json(census, "architecture_census.json")
    return census


# =========================================================
# Stage 2: Quick Calibration
# =========================================================
def quick_calibration(model, tokenizer, config, model_name):
    """Test INT8/INT6/INT4 key precision with V=INT4."""
    print(f"\n{'='*60}")
    print(f"Stage 2: Quick Calibration — {model_name}")
    print(f"{'='*60}")

    cached = load_cached_json(f"quick_calibration_{model_name}.json")
    if cached:
        print(f"  Cached: quick_calibration_{model_name}.json")
        return cached

    tokens = load_wikitext_tokens(tokenizer)
    eval_len = 256
    results = {}

    for k_type, v_type, config_name in PRECISION_CONFIGS:
        print(f"\n  Config: {config_name}")

        # Check if prior config collapsed (skip more aggressive)
        if k_type == "int6" and "K_INT8__V_INT4" in results:
            if results["K_INT8__V_INT4"].get("collapsed", False):
                print("    INT8 collapsed — skipping INT6")
                results[config_name] = {
                    "collapsed": True,
                    "skipped": True,
                    "reason": "INT8 collapsed",
                }
                continue
        if k_type == "int4":
            prev_key = "K_INT6__V_INT4"
            if prev_key in results and results[prev_key].get("collapsed", False):
                print("    INT6 collapsed — skipping INT4")
                results[config_name] = {
                    "collapsed": True,
                    "skipped": True,
                    "reason": "INT6 collapsed",
                }
                continue

        # Collapse check: 10 prompts, logit_err + token_agreement
        logit_errs = []
        agrees = []
        ppls_fp = []
        ppls_q = []

        for seed in range(N_CAL_PROMPTS):
            try:
                # Prefill
                rng = np.random.RandomState(seed + 200)
                start = rng.randint(
                    0, max(1, len(tokens) - CAL_PROMPT_LEN - eval_len - 10)
                )
                input_ids = (
                    torch.tensor(
                        tokens[start : start + CAL_PROMPT_LEN],
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                with torch.no_grad():
                    out_fp = model(input_ids, use_cache=True)
                    past_fp = out_fp.past_key_values
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values

                past_q = quantize_cache_uniform(past_q, k_type, v_type, GROUP_SIZE)

                # Logit error on next token
                next_tok_id = (
                    tokens[start + CAL_PROMPT_LEN]
                    if (start + CAL_PROMPT_LEN) < len(tokens)
                    else tokens[-1]
                )
                next_tok = torch.tensor([[next_tok_id]]).to(DEVICE)
                with torch.no_grad():
                    o1 = model(next_tok, past_key_values=past_fp, use_cache=True)
                    o2 = model(next_tok, past_key_values=past_q, use_cache=True)

                logit_err = (
                    (o1.logits[0, 0].float() - o2.logits[0, 0].float())
                    .abs()
                    .max()
                    .item()
                )
                logit_errs.append(logit_err)

                # Token agreement: 64 greedy tokens
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

                agrees.append(agree / GEN_LEN)

                # Continuation PPL (256 tokens after prefill)
                continuation = (
                    torch.tensor(
                        tokens[
                            start + CAL_PROMPT_LEN : start + CAL_PROMPT_LEN + eval_len
                        ],
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                # Rebuild caches for PPL (the previous ones were consumed
                # by generation)
                with torch.no_grad():
                    out_fp2 = model(input_ids, use_cache=True)
                    past_fp2 = out_fp2.past_key_values
                    out_q2 = model(input_ids, use_cache=True)
                    past_q2 = out_q2.past_key_values

                past_q2 = quantize_cache_uniform(past_q2, k_type, v_type, GROUP_SIZE)

                with torch.no_grad():
                    lf = (
                        model(
                            continuation,
                            past_key_values=past_fp2,
                            use_cache=False,
                        )
                        .logits.float()
                        .cpu()
                    )
                    lq = (
                        model(
                            continuation,
                            past_key_values=past_q2,
                            use_cache=False,
                        )
                        .logits.float()
                        .cpu()
                    )

                targets = continuation[:, 1:].cpu()
                loss_fp = F.cross_entropy(
                    lf[:, :-1, :].reshape(-1, lf.shape[-1]),
                    targets.reshape(-1),
                )
                loss_q = F.cross_entropy(
                    lq[:, :-1, :].reshape(-1, lq.shape[-1]),
                    targets.reshape(-1),
                )
                ppls_fp.append(math.exp(loss_fp.item()))
                ppls_q.append(math.exp(loss_q.item()))

                del (
                    past_fp,
                    past_q,
                    p_fp,
                    p_q,
                    past_fp2,
                    past_q2,
                    out_fp,
                    out_q,
                    out_fp2,
                    out_q2,
                )
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at seed {seed}")
                clear_gpu()
                break
            except Exception as e:
                print(f"    Error seed {seed}: {e}")
                continue

        if not logit_errs:
            results[config_name] = {
                "collapsed": True,
                "error": "No successful seeds",
            }
            continue

        avg_logit = float(np.mean(logit_errs))
        avg_agree = float(np.mean(agrees)) if agrees else 0.0
        avg_ppl_fp = float(np.mean(ppls_fp)) if ppls_fp else 0.0
        avg_ppl_q = float(np.mean(ppls_q)) if ppls_q else 0.0
        ppl_delta = (avg_ppl_q - avg_ppl_fp) / avg_ppl_fp * 100 if avg_ppl_fp > 0 else 0

        # v51 collapse thresholds for comparability
        collapsed = bool(avg_logit > 5 or avg_agree < 0.08)

        results[config_name] = {
            "logit_error": round(avg_logit, 4),
            "token_agreement": round(avg_agree, 4),
            "ppl_fp16": round(avg_ppl_fp, 4),
            "ppl_quant": round(avg_ppl_q, 4),
            "ppl_delta_pct": round(ppl_delta, 4),
            "collapsed": collapsed,
            "n_seeds": len(logit_errs),
        }

        status = "COLLAPSED" if collapsed else "OK"
        print(
            f"    {status}: logit_err={avg_logit:.2f} agree={avg_agree:.4f} "
            f"ppl_delta={ppl_delta:.2f}%"
        )

    # Determine floor
    int8_col = results.get("K_INT8__V_INT4", {}).get("collapsed", True)
    int6_col = results.get("K_INT6__V_INT4", {}).get("collapsed", True)
    int4_col = results.get("K_INT4__V_INT4", {}).get("collapsed", True)

    if not int4_col:
        floor = "int4_or_below"
    elif not int6_col:
        floor = "int5_to_int6"
    elif not int8_col:
        floor = "int7_to_int8"
    else:
        floor = "above_int8"

    output = {
        "model": model_name,
        "configs": results,
        "observed_floor": floor,
    }

    save_json(output, f"quick_calibration_{model_name}.json")
    print(f"  Floor: {floor}")
    return output


# =========================================================
# Stage 3: GQA Spread Measurement
# =========================================================
def measure_gqa_spread(model, tokenizer, config, model_name, gqa_ratio):
    """Measure how spread out query heads are within each KV group."""
    print(f"\n{'='*60}")
    print(f"Stage 3: GQA Spread — {model_name} (GQA={gqa_ratio}x)")
    print(f"{'='*60}")

    cached = load_cached_json(f"gqa_spread_{model_name}.json")
    if cached:
        print(f"  Cached: gqa_spread_{model_name}.json")
        return cached

    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    n_layers = config.num_hidden_layers
    group_size_gqa = n_heads // n_kv

    if group_size_gqa <= 1:
        result = {
            "model": model_name,
            "gqa_ratio": 1.0,
            "mean_spread": 1.0,
            "max_spread": 1.0,
            "spread_variance": 0.0,
            "note": "MHA — each head has its own key, trivial spread",
        }
        save_json(result, f"gqa_spread_{model_name}.json")
        print("  MHA model — trivial spread=1.0")
        return result

    tokens = load_wikitext_tokens(tokenizer)

    # Collect per-layer spread across prompts
    all_layer_spreads = []  # [n_prompts, n_layers]

    for prompt_idx in range(N_SPREAD_PROMPTS):
        if prompt_idx % 5 == 0:
            print(f"  Prompt {prompt_idx}/{N_SPREAD_PROMPTS}...")

        rng = np.random.RandomState(prompt_idx + 300)
        start = rng.randint(0, max(1, len(tokens) - SPREAD_PROMPT_LEN - 10))
        input_ids = (
            torch.tensor(tokens[start : start + SPREAD_PROMPT_LEN], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        try:
            with torch.no_grad():
                out = model(input_ids, output_attentions=True, use_cache=False)

            attentions = out.attentions
            if attentions is None or len(attentions) == 0:
                print("  WARNING: No attention weights returned")
                break

            layer_spreads = []
            for li, attn in enumerate(attentions):
                # attn: [1, n_heads, T, T]
                attn_w = attn[0].float()  # [n_heads, T, T]

                # Group query heads by KV head
                group_sims = []
                for kv_idx in range(n_kv):
                    head_start = kv_idx * group_size_gqa
                    head_end = head_start + group_size_gqa
                    group_attns = attn_w[head_start:head_end]  # [G, T, T]

                    # Flatten each head's attention pattern
                    G = group_attns.shape[0]
                    flat = group_attns.reshape(G, -1)  # [G, T*T]

                    # Pairwise cosine similarity
                    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
                    normalized = flat / norms
                    sim_matrix = normalized @ normalized.T  # [G, G]

                    # Mean of off-diagonal elements
                    mask = 1 - torch.eye(G, device=sim_matrix.device)
                    if mask.sum() > 0:
                        mean_sim = (sim_matrix * mask).sum() / mask.sum()
                        group_sims.append(mean_sim.item())

                layer_spread = float(np.mean(group_sims)) if group_sims else 1.0
                layer_spreads.append(layer_spread)

            all_layer_spreads.append(layer_spreads)

            del out, attentions
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at prompt {prompt_idx}")
            clear_gpu()
            break
        except Exception as e:
            print(f"  Error at prompt {prompt_idx}: {e}")
            continue

    if not all_layer_spreads:
        result = {
            "model": model_name,
            "gqa_ratio": gqa_ratio,
            "error": "No attention weights collected",
        }
        save_json(result, f"gqa_spread_{model_name}.json")
        return result

    spread_arr = np.array(all_layer_spreads)  # [n_prompts, n_layers]
    per_layer_mean = spread_arr.mean(axis=0).tolist()
    mean_spread = float(spread_arr.mean())
    max_spread = float(spread_arr.max())
    spread_var = float(spread_arr.var())

    result = {
        "model": model_name,
        "gqa_ratio": gqa_ratio,
        "group_size": group_size_gqa,
        "n_kv_heads": n_kv,
        "n_prompts": len(all_layer_spreads),
        "mean_spread": round(mean_spread, 4),
        "max_spread": round(max_spread, 4),
        "spread_variance": round(spread_var, 6),
        "per_layer_mean_spread": [round(x, 4) for x in per_layer_mean],
    }

    save_json(result, f"gqa_spread_{model_name}.json")
    print(
        f"  Spread: mean={mean_spread:.4f} max={max_spread:.4f} "
        f"var={spread_var:.6f}"
    )
    return result


# =========================================================
# Stage 4: Correlation Analysis
# =========================================================
def correlation_analysis(census, calibration, spread_data):
    """Correlate GQA ratio and spread with observed precision floor."""
    print(f"\n{'='*60}")
    print("Stage 4: Correlation Analysis")
    print(f"{'='*60}")

    from scipy.stats import spearmanr

    # Encode floors numerically (higher = more sensitive)
    floor_encoding = {
        "int4_or_below": 0,
        "int5_to_int6": 1,
        "int6_or_below": 1,
        "int7_to_int8": 2,
        "int7": 2,
        "above_int8": 3,
        "none": 0,
    }

    rows = []
    for model_name in MODELS:
        cen = census.get(model_name, {})
        cal = calibration.get(model_name, {})
        spr = spread_data.get(model_name, {})

        if "error" in cen or "error" in cal:
            continue

        gqa = cen.get("gqa_ratio", 1.0)
        floor = cal.get("observed_floor", "unknown")
        floor_num = floor_encoding.get(floor, -1)
        if floor_num < 0:
            continue

        mean_spread = spr.get("mean_spread", 1.0)
        entropy = PRIOR_ENTROPY.get(model_name)
        is_gqa = 1 if gqa > 1.0 else 0

        # INT8 logit error for binary classifier
        int8_logit = (
            cal.get("configs", {}).get("K_INT8__V_INT4", {}).get("logit_error", None)
        )

        rows.append(
            {
                "model": model_name,
                "gqa_ratio": gqa,
                "floor": floor,
                "floor_num": floor_num,
                "mean_spread": mean_spread,
                "entropy": entropy,
                "is_gqa": is_gqa,
                "int8_logit_err": int8_logit,
            }
        )

    if len(rows) < 3:
        result = {"error": "Too few models for correlation", "n_models": len(rows)}
        save_json(result, "correlation_analysis.json")
        return result

    floor_arr = np.array([r["floor_num"] for r in rows])
    gqa_arr = np.array([r["gqa_ratio"] for r in rows])
    spread_arr = np.array([r["mean_spread"] for r in rows])
    is_gqa_arr = np.array([r["is_gqa"] for r in rows])

    correlations = {}

    # 1. gqa_ratio alone
    if np.std(gqa_arr) > 0 and np.std(floor_arr) > 0:
        rho, p = spearmanr(gqa_arr, floor_arr)
        correlations["gqa_ratio"] = {"rho": round(rho, 4), "p": round(p, 4)}

    # 2. gqa_ratio * mean_spread
    interaction = gqa_arr * spread_arr
    if np.std(interaction) > 0:
        rho, p = spearmanr(interaction, floor_arr)
        correlations["gqa_x_spread"] = {"rho": round(rho, 4), "p": round(p, 4)}

    # 3. entropy * gqa_ratio (for models with entropy data)
    ent_rows = [r for r in rows if r["entropy"] is not None]
    if len(ent_rows) >= 3:
        ent_arr = np.array([r["entropy"] for r in ent_rows])
        ent_gqa = np.array([r["gqa_ratio"] for r in ent_rows])
        ent_floor = np.array([r["floor_num"] for r in ent_rows])
        ent_interaction = ent_arr * ent_gqa
        if np.std(ent_interaction) > 0 and np.std(ent_floor) > 0:
            rho, p = spearmanr(ent_interaction, ent_floor)
            correlations["entropy_x_gqa"] = {
                "rho": round(rho, 4),
                "p": round(p, 4),
            }

    # 4. is_gqa binary
    if np.std(is_gqa_arr) > 0:
        rho, p = spearmanr(is_gqa_arr, floor_arr)
        correlations["is_gqa_binary"] = {"rho": round(rho, 4), "p": round(p, 4)}

    # 5. entropy alone (for comparison with v50)
    if len(ent_rows) >= 3:
        rho, p = spearmanr(
            [r["entropy"] for r in ent_rows],
            [r["floor_num"] for r in ent_rows],
        )
        correlations["entropy_alone"] = {"rho": round(rho, 4), "p": round(p, 4)}

    result = {
        "models": rows,
        "correlations": correlations,
        "n_models": len(rows),
        "v50_entropy_corr_3models": 0.89,
        "v51_entropy_corr_6models": 0.29,
    }

    save_json(result, "correlation_analysis.json")

    print("\n  Predictor correlations:")
    for name, vals in correlations.items():
        print(f"    {name:20s}: rho={vals['rho']:.4f}  p={vals['p']:.4f}")

    return result


# =========================================================
# Stage 5: Binary Classifier Validation
# =========================================================
def binary_classifier(calibration):
    """Test: INT8 logit_err > 2.0 predicts 'needs FP16 keys'."""
    print(f"\n{'='*60}")
    print("Stage 5: Binary Classifier Validation")
    print(f"{'='*60}")

    predictions = []
    for model_name, cal in calibration.items():
        int8_data = cal.get("configs", {}).get("K_INT8__V_INT4", {})
        logit_err = int8_data.get("logit_error")
        if logit_err is None:
            continue

        floor = cal.get("observed_floor", "unknown")
        # Ground truth: needs FP16 if floor is int7+ (i.e., INT6 collapses)
        needs_fp16 = floor in ("int7_to_int8", "above_int8")
        predicted_fp16 = logit_err > 2.0

        predictions.append(
            {
                "model": model_name,
                "int8_logit_err": round(logit_err, 4),
                "predicted_needs_fp16": predicted_fp16,
                "actual_needs_fp16": needs_fp16,
                "correct": predicted_fp16 == needs_fp16,
            }
        )

    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    # Confusion matrix
    tp = sum(
        1 for p in predictions if p["predicted_needs_fp16"] and p["actual_needs_fp16"]
    )
    fp = sum(
        1
        for p in predictions
        if p["predicted_needs_fp16"] and not p["actual_needs_fp16"]
    )
    fn = sum(
        1
        for p in predictions
        if not p["predicted_needs_fp16"] and p["actual_needs_fp16"]
    )
    tn = sum(
        1
        for p in predictions
        if not p["predicted_needs_fp16"] and not p["actual_needs_fp16"]
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    result = {
        "threshold": 2.0,
        "predictions": predictions,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_models": total,
    }

    save_json(result, "binary_classifier_results.json")

    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0%})")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")
    for p in predictions:
        mark = "OK" if p["correct"] else "WRONG"
        print(
            f"    {p['model']:16s} logit={p['int8_logit_err']:.2f} "
            f"pred={p['predicted_needs_fp16']} "
            f"actual={p['actual_needs_fp16']} [{mark}]"
        )

    return result


# =========================================================
# Stage 6: Plots
# =========================================================
def generate_plots(census, calibration, spread_data, corr_data, classifier):
    """Generate visualization plots."""
    print(f"\n{'='*60}")
    print("Stage 6: Visualization")
    print(f"{'='*60}")

    # Plot 1: GQA ratio vs floor
    fig, ax = plt.subplots(figsize=(10, 7))
    floor_encoding = {
        "int4_or_below": 0,
        "int5_to_int6": 1,
        "int6_or_below": 1,
        "int7_to_int8": 2,
        "above_int8": 3,
    }
    floor_labels = {
        0: "INT4 or below",
        1: "INT5-INT6",
        2: "INT7-INT8",
        3: "Above INT8",
    }

    for model_name, cal in calibration.items():
        cen = census.get(model_name, {})
        gqa = cen.get("gqa_ratio", 1.0)
        floor = cal.get("observed_floor", "unknown")
        fn = floor_encoding.get(floor, -1)
        if fn < 0:
            continue

        is_gqa = gqa > 1.0
        color = "tab:red" if is_gqa else "tab:blue"
        marker = "s" if is_gqa else "o"
        ax.scatter(gqa, fn, s=120, c=color, marker=marker, zorder=5)
        ax.annotate(
            model_name,
            (gqa, fn),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    ax.set_xlabel("GQA Ratio (Q heads / KV heads)", fontsize=12)
    ax.set_ylabel("Key Precision Floor", fontsize=12)
    ax.set_yticks(list(floor_labels.keys()))
    ax.set_yticklabels(list(floor_labels.values()))
    rho = corr_data.get("correlations", {}).get("gqa_ratio", {}).get("rho", "N/A")
    ax.set_title(f"GQA Ratio vs Key Precision Floor (rho={rho})", fontsize=13)
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="tab:red",
                label="GQA",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:blue",
                label="MHA",
                markersize=10,
            ),
        ],
        loc="upper left",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gqa_ratio_vs_floor.png"), dpi=300)
    plt.close()
    print("  Saved: gqa_ratio_vs_floor.png")

    # Plot 2: Binary classifier confusion matrix
    cm = classifier.get("confusion_matrix", {})
    matrix = np.array(
        [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Tolerant", "Pred: Needs FP16"])
    ax.set_yticklabels(["True: Tolerant", "True: Needs FP16"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=20,
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    ax.set_title(
        f"Binary Classifier (logit_err > 2.0)\n"
        f"Accuracy: {classifier.get('accuracy', 0):.0%}",
        fontsize=13,
    )
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "binary_classifier_roc.png"), dpi=300)
    plt.close()
    print("  Saved: binary_classifier_roc.png")


# =========================================================
# Summary Report
# =========================================================
def write_summary(
    census, calibration, spread_data, corr_data, classifier, errors, elapsed
):
    lines = []
    lines.append("# BPA v52 — GQA Key-Sharing Quantization Sensitivity")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**GPU**: {gpu_info()['device']}")
    lines.append(f"**Runtime**: {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    lines.append("")

    # Architecture census table
    lines.append("## Architecture Census")
    lines.append("")
    lines.append(
        "| Model | Q Heads | KV Heads | GQA Ratio | Head Dim | " "RoPE theta | Layers |"
    )
    lines.append("|" + "|".join(["-------"] * 7) + "|")
    for m, c in sorted(census.items()):
        if "error" in c:
            lines.append(f"| {m} | ERROR | | | | | |")
            continue
        theta = c.get("rope_theta")
        theta_str = f"{theta:.0f}" if theta else "N/A"
        lines.append(
            f"| {m} | {c['num_attention_heads']} | "
            f"{c['num_key_value_heads']} | {c['gqa_ratio']} | "
            f"{c['head_dim']} | {theta_str} | {c['num_layers']} |"
        )
    lines.append("")

    # Quick calibration results
    lines.append("## Quick Calibration Results")
    lines.append("")
    lines.append(
        "| Model | INT8 logit_err | INT6 logit_err | INT4 logit_err | "
        "INT8 PPL% | INT6 PPL% | Floor |"
    )
    lines.append("|" + "|".join(["-------"] * 7) + "|")
    for m, cal in sorted(calibration.items()):
        configs = cal.get("configs", {})
        floor = cal.get("observed_floor", "?")

        def _val(cfg_name, key, fmt=".2f"):
            d = configs.get(cfg_name, {})
            if d.get("skipped"):
                return "skip"
            if d.get("collapsed") and not d.get("skipped"):
                v = d.get(key)
                if v is not None:
                    return f"**{v:{fmt}}**"
                return "COLL"
            v = d.get(key)
            return f"{v:{fmt}}" if v is not None else "—"

        lines.append(
            f"| {m} | {_val('K_INT8__V_INT4', 'logit_error')} | "
            f"{_val('K_INT6__V_INT4', 'logit_error')} | "
            f"{_val('K_INT4__V_INT4', 'logit_error')} | "
            f"{_val('K_INT8__V_INT4', 'ppl_delta_pct')} | "
            f"{_val('K_INT6__V_INT4', 'ppl_delta_pct')} | "
            f"{floor} |"
        )
    lines.append("")

    # GQA Spread
    lines.append("## GQA Spread Analysis")
    lines.append("")
    lines.append("| Model | GQA Ratio | Group Size | Mean Spread | Max Spread |")
    lines.append("|" + "|".join(["-------"] * 5) + "|")
    for m, s in sorted(spread_data.items()):
        gqa = s.get("gqa_ratio", 1.0)
        gs = s.get("group_size", 1)
        ms = s.get("mean_spread", 1.0)
        mx = s.get("max_spread", 1.0)
        lines.append(f"| {m} | {gqa} | {gs} | {ms:.4f} | {mx:.4f} |")
    lines.append("")

    # Correlation comparison
    lines.append("## Correlation Analysis")
    lines.append("")
    correlations = corr_data.get("correlations", {})
    lines.append("| Predictor | Spearman rho | p-value |")
    lines.append("|-----------|-------------|---------|")
    for name, vals in correlations.items():
        lines.append(f"| {name} | {vals['rho']:.4f} | {vals['p']:.4f} |")
    lines.append(f"| v50 entropy (3 models) | 0.89 | — |")
    lines.append(f"| v51 entropy (6 models) | 0.29 | — |")
    lines.append("")

    # Binary classifier
    lines.append("## Binary Classifier")
    lines.append("")
    lines.append(f"Threshold: INT8 logit_err > 2.0 predicts 'needs FP16 keys'")
    lines.append(f"Accuracy: {classifier.get('accuracy', 0):.0%}")
    cm = classifier.get("confusion_matrix", {})
    lines.append(
        f"TP={cm.get('tp',0)} FP={cm.get('fp',0)} "
        f"FN={cm.get('fn',0)} TN={cm.get('tn',0)}"
    )
    lines.append("")

    # Stop conditions
    lines.append("## Stop Condition Evaluation")
    lines.append("")

    gqa_models_floors = {}
    mha_models_floors = {}
    for m, cal in calibration.items():
        cen = census.get(m, {})
        gqa = cen.get("gqa_ratio", 1.0)
        floor = cal.get("observed_floor", "unknown")
        if gqa > 1.0:
            gqa_models_floors[m] = floor
        else:
            mha_models_floors[m] = floor

    lines.append(f"GQA models: {gqa_models_floors}")
    lines.append(f"MHA models: {mha_models_floors}")
    lines.append("")

    # Errors
    if errors:
        lines.append("## Model Errors")
        lines.append("")
        for m, e in errors.items():
            lines.append(f"- **{m}**: {e}")
        lines.append("")

    # Plots
    lines.append("## Plots")
    lines.append("")
    lines.append("![gqa_ratio_vs_floor](plots/gqa_ratio_vs_floor.png)")
    lines.append("![binary_classifier](plots/binary_classifier_roc.png)")
    lines.append("")

    summary_path = os.path.join(RESULTS_ROOT, "bpa52_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")


# =========================================================
# Main
# =========================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("BPA v52: GQA Key-Sharing Quantization Sensitivity")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    info = gpu_info()
    print(f"GPU: {info['device']} ({info['total_gb']} GB)")
    print(f"Torch: {info['torch']}, HIP: {info['hip']}")

    # Stage 1: Architecture Census
    t1 = time.time()
    census = architecture_census()
    print(f"  Stage 1 time: {time.time()-t1:.0f}s")

    calibration = {}
    spread_data = {}
    model_errors = {}

    # Stages 2-3: Per-model processing
    for model_name, model_id in MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name} ({model_id})")
        print(f"{'#'*60}")

        # Check if model had a census error
        if "error" in census.get(model_name, {}):
            print(f"  Skipping — census error")
            model_errors[model_name] = census[model_name]["error"]
            continue

        try:
            model, tokenizer, config = load_model(model_id)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            model_errors[model_name] = str(e)
            save_json(
                {"model": model_name, "error": str(e)},
                f"error_{model_name}.json",
            )
            clear_gpu()
            continue

        # Stage 2: Quick calibration
        try:
            cal_result = quick_calibration(model, tokenizer, config, model_name)
            calibration[model_name] = cal_result
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during calibration — skipping")
            model_errors[model_name] = "OOM during calibration"
            calibration[model_name] = {
                "model": model_name,
                "error": "OOM",
                "observed_floor": "unknown",
            }
            clear_gpu()
        except Exception as e:
            print(f"  ERROR calibration: {e}")
            model_errors[model_name] = f"calibration: {e}"
            calibration[model_name] = {
                "model": model_name,
                "error": str(e),
                "observed_floor": "unknown",
            }

        # Stage 3: GQA spread
        gqa_ratio = census.get(model_name, {}).get("gqa_ratio", 1.0)
        try:
            spr = measure_gqa_spread(model, tokenizer, config, model_name, gqa_ratio)
            spread_data[model_name] = spr
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during GQA spread — skipping")
            spread_data[model_name] = {
                "model": model_name,
                "gqa_ratio": gqa_ratio,
                "error": "OOM",
            }
            clear_gpu()
        except Exception as e:
            print(f"  ERROR GQA spread: {e}")
            spread_data[model_name] = {
                "model": model_name,
                "gqa_ratio": gqa_ratio,
                "error": str(e),
            }

        unload_model(model)
        del tokenizer, config
        clear_gpu()

    # Stage 4: Correlation analysis
    t4 = time.time()
    corr_data = correlation_analysis(census, calibration, spread_data)
    print(f"  Stage 4 time: {time.time()-t4:.0f}s")

    # Stage 5: Binary classifier
    t5 = time.time()
    classifier = binary_classifier(calibration)
    print(f"  Stage 5 time: {time.time()-t5:.0f}s")

    # Stage 6: Plots + Summary
    generate_plots(census, calibration, spread_data, corr_data, classifier)

    elapsed = time.time() - t_start
    metadata = {
        "experiment": "BPA-v52",
        "date": str(datetime.now()),
        "gpu": gpu_info(),
        "elapsed_seconds": round(elapsed, 1),
        "models_tested": list(calibration.keys()),
        "models_failed": model_errors,
    }
    save_json(metadata, "bpa52_metadata.json")

    write_summary(
        census,
        calibration,
        spread_data,
        corr_data,
        classifier,
        model_errors,
        elapsed,
    )

    print(f"\n{'='*60}")
    print(f"BPA v52 COMPLETE — {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
