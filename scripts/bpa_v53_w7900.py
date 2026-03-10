#!/usr/bin/env python3
"""BPA v53: Ratio Classifier Stress Test + Failure Mode Hunt.

Stress-tests the v52 ratio classifier (INT6_err / INT8_err > 2.0
predicts "needs FP16 keys") on additional ~7B models spanning
diverse architectures and GQA configurations.

Stage 1: Architecture census (config-only, no GPU)
Stage 2: Ratio classifier test (INT8/INT6 keys, V=INT4)
Stage 3: Classifier evaluation (combined v52 + v53)
Stage 4: Architecture factor analysis + plots
Stage 5: Summary report

Results saved to /data/knlp-key-results/bpa53/
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
RESULTS_ROOT = "/data/knlp-key-results/bpa53"
JSON_DIR = os.path.join(RESULTS_ROOT, "json")
PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
for d in [JSON_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# v52 results path for combined analysis
V52_JSON = "/data/knlp-key-results/bpa52/json"

# === Protocol parameters (same as v49-v52) ===
DATASET = "wikitext-103-raw-v1"
W_SINK = 4
GROUP_SIZE = 32
DEVICE = "cuda"
DTYPE = torch.bfloat16

N_CAL_PROMPTS = 10
CAL_PROMPT_LEN = 2048
GEN_LEN = 64

# v53 models -- new models NOT in v52
MODELS = {
    # Tier 1: Chinese-lab, high GQA
    "yi-1.5-9b": "01-ai/Yi-1.5-9B",
    "yi-6b": "01-ai/Yi-6B",
    "internlm2-7b": "internlm/internlm2-7b",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-base",
    # Tier 2: Broader coverage
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "stablelm-2-1.6b": "stabilityai/stablelm-2-1_6b",
    "starcoder2-7b": "bigcode/starcoder2-7b",
    # Tier 3: If accessible
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-7b": "google/gemma-7b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2-7b": "Qwen/Qwen2-7B",
}

# Precision configs: INT8 and INT6 keys, V always INT4
PRECISION_CONFIGS = [
    ("int8", "int4", "K_INT8__V_INT4"),
    ("int6", "int4", "K_INT6__V_INT4"),
]


class _NumpyEncoder(json.JSONEncoder):
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


def load_v52_json(filename):
    path = os.path.join(V52_JSON, filename)
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


# === Quantization (same as v49-v52) ===
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
            head_dim = getattr(cfg, "head_dim", hidden // n_heads)
            rope_theta = getattr(
                cfg, "rope_theta", getattr(cfg, "rotary_emb_base", None)
            )
            n_layers = cfg.num_hidden_layers
            gqa_ratio = n_heads / n_kv if n_kv > 0 else 1.0

            # Detect sliding window
            sliding = getattr(cfg, "sliding_window", None)

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
                "sliding_window": sliding,
            }
            print(
                f"Q={n_heads} KV={n_kv} GQA={gqa_ratio:.1f}x "
                f"d={head_dim} layers={n_layers}"
            )
        except Exception as e:
            err_short = str(e).split("\n")[0][:120]
            print(f"ERROR: {err_short}")
            census[model_name] = {"model_id": model_id, "error": str(e)}

    save_json(census, "architecture_census.json")
    return census


# =========================================================
# Stage 2: Ratio Classifier Test
# =========================================================
def ratio_test(model, tokenizer, config, model_name):
    """Run INT8 and INT6 key quantization, compute error ratio."""
    print(f"\n{'='*60}")
    print(f"Stage 2: Ratio Test — {model_name}")
    print(f"{'='*60}")

    cached = load_cached_json(f"ratio_test_{model_name}.json")
    if cached:
        print(f"  Cached: ratio_test_{model_name}.json")
        return cached

    tokens = load_wikitext_tokens(tokenizer)
    eval_len = 256
    results = {}

    for k_type, v_type, config_name in PRECISION_CONFIGS:
        print(f"\n  Config: {config_name}")

        logit_errs = []
        agrees = []
        ppls_fp = []
        ppls_q = []

        for seed in range(N_CAL_PROMPTS):
            try:
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

                # Continuation PPL
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

    # Compute error ratio
    int8_data = results.get("K_INT8__V_INT4", {})
    int6_data = results.get("K_INT6__V_INT4", {})
    int8_err = int8_data.get("logit_error", 0)
    int6_err = int6_data.get("logit_error", 0)
    error_ratio = int6_err / int8_err if int8_err > 0.001 else 0
    predicted_sensitive = error_ratio > 2.0

    # Determine floor from what we tested
    int8_col = int8_data.get("collapsed", True)
    int6_col = int6_data.get("collapsed", True)

    if not int6_col:
        floor = "int6_or_below"
    elif not int8_col:
        floor = "int7_to_int8"
    else:
        floor = "above_int8"

    output = {
        "model": model_name,
        "configs": results,
        "error_ratio": round(error_ratio, 4),
        "predicted_sensitive": predicted_sensitive,
        "observed_floor": floor,
    }

    save_json(output, f"ratio_test_{model_name}.json")
    print(f"  Ratio: {error_ratio:.2f}x  Predicted sensitive: {predicted_sensitive}")
    print(f"  Floor: {floor}")
    return output


# =========================================================
# Stage 2b: INT4 confirmation for ratio > 2.0 models
# =========================================================
def confirm_int4(model, tokenizer, model_name):
    """Run INT4 keys to confirm actual floor for flagged models."""
    print(f"\n  Confirming INT4 floor for {model_name}...")

    tokens = load_wikitext_tokens(tokenizer)
    eval_len = 256
    logit_errs = []
    agrees = []
    ppls_fp = []
    ppls_q = []

    for seed in range(N_CAL_PROMPTS):
        try:
            rng = np.random.RandomState(seed + 200)
            start = rng.randint(0, max(1, len(tokens) - CAL_PROMPT_LEN - eval_len - 10))
            input_ids = (
                torch.tensor(tokens[start : start + CAL_PROMPT_LEN], dtype=torch.long)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past_fp = out_fp.past_key_values
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values

            past_q = quantize_cache_uniform(past_q, "int4", "int4", GROUP_SIZE)

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
                (o1.logits[0, 0].float() - o2.logits[0, 0].float()).abs().max().item()
            )
            logit_errs.append(logit_err)

            # Token agreement
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

            # PPL
            continuation = (
                torch.tensor(
                    tokens[start + CAL_PROMPT_LEN : start + CAL_PROMPT_LEN + eval_len],
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .to(DEVICE)
            )
            with torch.no_grad():
                out_fp2 = model(input_ids, use_cache=True)
                past_fp2 = out_fp2.past_key_values
                out_q2 = model(input_ids, use_cache=True)
                past_q2 = out_q2.past_key_values
            past_q2 = quantize_cache_uniform(past_q2, "int4", "int4", GROUP_SIZE)

            with torch.no_grad():
                lf = (
                    model(continuation, past_key_values=past_fp2, use_cache=False)
                    .logits.float()
                    .cpu()
                )
                lq = (
                    model(continuation, past_key_values=past_q2, use_cache=False)
                    .logits.float()
                    .cpu()
                )
            targets = continuation[:, 1:].cpu()
            loss_fp = F.cross_entropy(
                lf[:, :-1, :].reshape(-1, lf.shape[-1]), targets.reshape(-1)
            )
            loss_q = F.cross_entropy(
                lq[:, :-1, :].reshape(-1, lq.shape[-1]), targets.reshape(-1)
            )
            ppls_fp.append(math.exp(loss_fp.item()))
            ppls_q.append(math.exp(loss_q.item()))

            del past_fp, past_q, p_fp, p_q, past_fp2, past_q2
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at seed {seed}")
            clear_gpu()
            break
        except Exception as e:
            print(f"    Error seed {seed}: {e}")
            continue

    if not logit_errs:
        return {"collapsed": True, "error": "No successful seeds"}

    avg_logit = float(np.mean(logit_errs))
    avg_agree = float(np.mean(agrees)) if agrees else 0.0
    avg_ppl_fp = float(np.mean(ppls_fp)) if ppls_fp else 0.0
    avg_ppl_q = float(np.mean(ppls_q)) if ppls_q else 0.0
    ppl_delta = (avg_ppl_q - avg_ppl_fp) / avg_ppl_fp * 100 if avg_ppl_fp > 0 else 0
    collapsed = bool(avg_logit > 5 or avg_agree < 0.08)

    result = {
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
        f"    INT4 {status}: logit_err={avg_logit:.2f} agree={avg_agree:.4f} "
        f"ppl_delta={ppl_delta:.2f}%"
    )
    return result


# =========================================================
# Stage 3: Combined Classifier Evaluation
# =========================================================
def combined_classifier_evaluation(v53_results, census):
    """Combine v52 + v53 results and evaluate ratio classifier."""
    print(f"\n{'='*60}")
    print("Stage 3: Combined Classifier Evaluation")
    print(f"{'='*60}")

    # Load v52 calibration results
    v52_census = load_v52_json("architecture_census.json") or {}
    v52_models = [
        "qwen2.5-7b",
        "mistral-7b",
        "falcon-7b",
        "pythia-6.9b",
        "opt-6.7b",
        "phi-2",
    ]

    all_rows = []

    # v52 models
    for m in v52_models:
        cal = load_v52_json(f"quick_calibration_{m}.json")
        if cal is None or "error" in cal:
            continue
        cen = v52_census.get(m, {})
        if "error" in cen:
            continue

        configs = cal.get("configs", {})
        int8_err = configs.get("K_INT8__V_INT4", {}).get("logit_error", 0)
        int6_err = configs.get("K_INT6__V_INT4", {}).get("logit_error", 0)
        ratio = int6_err / int8_err if int8_err > 0.001 else 0
        floor = cal.get("observed_floor", "unknown")

        # Ground truth: needs FP16 if floor is int7+ (INT6 collapses)
        actual_sensitive = floor in ("int7_to_int8", "above_int8")
        predicted_sensitive = ratio > 2.0

        all_rows.append(
            {
                "model": m,
                "source": "v52",
                "gqa_ratio": cen.get("gqa_ratio", 1.0),
                "rope_theta": cen.get("rope_theta"),
                "head_dim": cen.get("head_dim"),
                "num_layers": cen.get("num_layers"),
                "int8_logit_err": round(int8_err, 4),
                "int6_logit_err": round(int6_err, 4),
                "error_ratio": round(ratio, 4),
                "observed_floor": floor,
                "actual_sensitive": actual_sensitive,
                "predicted_sensitive": predicted_sensitive,
                "correct": predicted_sensitive == actual_sensitive,
            }
        )

    # v53 models
    for m, res in v53_results.items():
        if "error" in res:
            continue
        cen = census.get(m, {})
        if "error" in cen:
            continue

        ratio = res.get("error_ratio", 0)
        floor = res.get("observed_floor", "unknown")

        # For v53 we only ran INT8/INT6, so floor is approximate
        actual_sensitive = floor in ("int7_to_int8", "above_int8")
        predicted_sensitive = ratio > 2.0

        all_rows.append(
            {
                "model": m,
                "source": "v53",
                "gqa_ratio": cen.get("gqa_ratio", 1.0),
                "rope_theta": cen.get("rope_theta"),
                "head_dim": cen.get("head_dim"),
                "num_layers": cen.get("num_layers"),
                "int8_logit_err": res.get("configs", {})
                .get("K_INT8__V_INT4", {})
                .get("logit_error", 0),
                "int6_logit_err": res.get("configs", {})
                .get("K_INT6__V_INT4", {})
                .get("logit_error", 0),
                "error_ratio": round(ratio, 4),
                "observed_floor": floor,
                "actual_sensitive": actual_sensitive,
                "predicted_sensitive": predicted_sensitive,
                "correct": predicted_sensitive == actual_sensitive,
            }
        )

    total = len(all_rows)
    correct = sum(1 for r in all_rows if r["correct"])
    accuracy = correct / total if total > 0 else 0

    tp = sum(1 for r in all_rows if r["predicted_sensitive"] and r["actual_sensitive"])
    fp = sum(
        1 for r in all_rows if r["predicted_sensitive"] and not r["actual_sensitive"]
    )
    fn = sum(
        1 for r in all_rows if not r["predicted_sensitive"] and r["actual_sensitive"]
    )
    tn = sum(
        1
        for r in all_rows
        if not r["predicted_sensitive"] and not r["actual_sensitive"]
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    result = {
        "threshold": 2.0,
        "models": all_rows,
        "n_models": total,
        "n_v52": sum(1 for r in all_rows if r["source"] == "v52"),
        "n_v53": sum(1 for r in all_rows if r["source"] == "v53"),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    save_json(result, "combined_classifier_evaluation.json")

    print(
        f"\n  Combined: {total} models ({result['n_v52']} v52 + {result['n_v53']} v53)"
    )
    print(f"  Accuracy: {correct}/{total} ({accuracy:.0%})")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")
    for r in all_rows:
        mark = "OK" if r["correct"] else "WRONG"
        flag = " ***" if r["error_ratio"] > 2.0 else ""
        print(
            f"    {r['model']:20s} ratio={r['error_ratio']:.2f} "
            f"pred={r['predicted_sensitive']} "
            f"actual={r['actual_sensitive']} [{mark}]{flag}"
        )

    return result


# =========================================================
# Stage 4: Factor Analysis + Plots
# =========================================================
def factor_analysis(combined):
    """Correlate architectural features with error ratio."""
    print(f"\n{'='*60}")
    print("Stage 4: Architecture Factor Analysis")
    print(f"{'='*60}")

    from scipy.stats import spearmanr

    rows = combined.get("models", [])
    if len(rows) < 3:
        result = {"error": "Too few models", "n_models": len(rows)}
        save_json(result, "factor_correlations.json")
        return result

    ratios = np.array([r["error_ratio"] for r in rows])
    gqa = np.array([r["gqa_ratio"] for r in rows])

    # Handle None rope_theta: replace with 0 for non-RoPE models
    rope = np.array([r["rope_theta"] if r["rope_theta"] else 0 for r in rows])
    names = [r["model"] for r in rows]

    correlations = {}

    # 1. GQA ratio vs error ratio
    if np.std(gqa) > 0 and np.std(ratios) > 0:
        rho, p = spearmanr(gqa, ratios)
        correlations["gqa_ratio"] = {"rho": round(rho, 4), "p": round(p, 4)}

    # 2. RoPE theta vs error ratio (only RoPE models)
    rope_mask = rope > 0
    if rope_mask.sum() >= 3:
        rho, p = spearmanr(rope[rope_mask], ratios[rope_mask])
        correlations["rope_theta"] = {
            "rho": round(rho, 4),
            "p": round(p, 4),
            "n": int(rope_mask.sum()),
        }

    # 3. GQA * RoPE interaction
    interaction = gqa * rope
    if np.std(interaction) > 0:
        rho, p = spearmanr(interaction, ratios)
        correlations["gqa_x_rope"] = {"rho": round(rho, 4), "p": round(p, 4)}

    # 4. log(RoPE) for RoPE models
    if rope_mask.sum() >= 3:
        log_rope = np.log10(rope[rope_mask] + 1)
        rho, p = spearmanr(log_rope, ratios[rope_mask])
        correlations["log_rope_theta"] = {
            "rho": round(rho, 4),
            "p": round(p, 4),
            "n": int(rope_mask.sum()),
        }

    # 5. is_gqa binary
    is_gqa = (gqa > 1.0).astype(float)
    if np.std(is_gqa) > 0:
        rho, p = spearmanr(is_gqa, ratios)
        correlations["is_gqa_binary"] = {"rho": round(rho, 4), "p": round(p, 4)}

    result = {
        "correlations": correlations,
        "n_models": len(rows),
        "models": [
            {
                "model": r["model"],
                "gqa_ratio": r["gqa_ratio"],
                "rope_theta": r["rope_theta"],
                "error_ratio": r["error_ratio"],
            }
            for r in rows
        ],
    }
    save_json(result, "factor_correlations.json")

    print("\n  Factor correlations:")
    for name, vals in correlations.items():
        print(f"    {name:20s}: rho={vals['rho']:.4f}  p={vals['p']:.4f}")

    # === Plots ===

    # Plot 1: GQA ratio vs error ratio
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, r in enumerate(rows):
        color = "tab:red" if r["error_ratio"] > 2.0 else "tab:blue"
        marker = "s" if r["gqa_ratio"] > 1.0 else "o"
        ax.scatter(
            r["gqa_ratio"], r["error_ratio"], s=120, c=color, marker=marker, zorder=5
        )
        ax.annotate(
            r["model"],
            (r["gqa_ratio"], r["error_ratio"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Threshold (2.0)")
    rho_gqa = correlations.get("gqa_ratio", {}).get("rho", "N/A")
    ax.set_xlabel("GQA Ratio (Q heads / KV heads)", fontsize=12)
    ax.set_ylabel("Error Ratio (INT6_err / INT8_err)", fontsize=12)
    ax.set_title(
        f"GQA Ratio vs Error Ratio (rho={rho_gqa})\n"
        f"v52 + v53 combined ({len(rows)} models)",
        fontsize=13,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gqa_ratio_vs_error_ratio.png"), dpi=300)
    plt.close()
    print("  Saved: gqa_ratio_vs_error_ratio.png")

    # Plot 2: RoPE theta vs error ratio
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in rows:
        theta = r["rope_theta"] if r["rope_theta"] else 0
        color = "tab:red" if r["error_ratio"] > 2.0 else "tab:blue"
        marker = "s" if r["gqa_ratio"] > 1.0 else "o"
        ax.scatter(
            theta if theta > 0 else 1,
            r["error_ratio"],
            s=120,
            c=color,
            marker=marker,
            zorder=5,
        )
        ax.annotate(
            r["model"],
            (theta if theta > 0 else 1, r["error_ratio"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Threshold (2.0)")
    ax.set_xscale("log")
    ax.set_xlabel("RoPE theta (log scale)", fontsize=12)
    ax.set_ylabel("Error Ratio (INT6_err / INT8_err)", fontsize=12)
    rho_rope = correlations.get("rope_theta", {}).get("rho", "N/A")
    ax.set_title(
        f"RoPE theta vs Error Ratio (rho={rho_rope})\n"
        f"v52 + v53 combined ({len(rows)} models)",
        fontsize=13,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rope_theta_vs_error_ratio.png"), dpi=300)
    plt.close()
    print("  Saved: rope_theta_vs_error_ratio.png")

    # Plot 3: Combined scatter with GQA*RoPE
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in rows:
        theta = r["rope_theta"] if r["rope_theta"] else 0
        x = r["gqa_ratio"] * theta
        color = "tab:red" if r["error_ratio"] > 2.0 else "tab:blue"
        ax.scatter(
            x if x > 0 else 1,
            r["error_ratio"],
            s=120,
            c=color,
            zorder=5,
        )
        ax.annotate(
            r["model"],
            (x if x > 0 else 1, r["error_ratio"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Threshold (2.0)")
    ax.set_xscale("log")
    ax.set_xlabel("GQA Ratio * RoPE theta (log scale)", fontsize=12)
    ax.set_ylabel("Error Ratio", fontsize=12)
    rho_inter = correlations.get("gqa_x_rope", {}).get("rho", "N/A")
    ax.set_title(
        f"GQA*RoPE Interaction vs Error Ratio (rho={rho_inter})",
        fontsize=13,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gqa_rope_interaction.png"), dpi=300)
    plt.close()
    print("  Saved: gqa_rope_interaction.png")

    return result


# =========================================================
# Stage 5: Summary Report
# =========================================================
def write_summary(census, v53_results, combined, factors, model_errors, elapsed):
    lines = []
    lines.append("# BPA v53 — Ratio Classifier Stress Test")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**GPU**: {gpu_info()['device']}")
    lines.append(f"**Runtime**: {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    lines.append("")

    # Architecture census
    lines.append("## Architecture Census (v53 new models)")
    lines.append("")
    lines.append(
        "| Model | Q Heads | KV Heads | GQA Ratio | Head Dim | "
        "RoPE theta | Layers | Status |"
    )
    lines.append("|" + "|".join(["-------"] * 8) + "|")
    for m, c in sorted(census.items()):
        if "error" in c:
            err = c["error"].split("\n")[0][:40]
            lines.append(f"| {m} | — | — | — | — | — | — | {err} |")
            continue
        theta = c.get("rope_theta")
        theta_str = f"{theta:.0f}" if theta else "N/A"
        lines.append(
            f"| {m} | {c['num_attention_heads']} | "
            f"{c['num_key_value_heads']} | {c['gqa_ratio']} | "
            f"{c['head_dim']} | {theta_str} | {c['num_layers']} | OK |"
        )
    lines.append("")

    # Combined results table
    lines.append("## Combined Results (v52 + v53)")
    lines.append("")
    lines.append(
        "| Model | Source | GQA | RoPE theta | INT8 err | "
        "INT6 err | Ratio | Floor | Sensitive? |"
    )
    lines.append("|" + "|".join(["-------"] * 9) + "|")
    for r in sorted(combined.get("models", []), key=lambda x: -x["error_ratio"]):
        theta = r.get("rope_theta")
        theta_str = f"{theta:.0f}" if theta else "N/A"
        flag = " **" if r["error_ratio"] > 2.0 else ""
        lines.append(
            f"| {r['model']} | {r['source']} | {r['gqa_ratio']} | "
            f"{theta_str} | {r['int8_logit_err']:.2f} | "
            f"{r['int6_logit_err']:.2f} | {r['error_ratio']:.2f}{flag} | "
            f"{r['observed_floor']} | {r['actual_sensitive']} |"
        )
    lines.append("")

    # Classifier performance
    lines.append("## Classifier Performance")
    lines.append("")
    cm = combined.get("confusion_matrix", {})
    lines.append(f"- **Threshold**: ratio > 2.0 predicts 'needs FP16 keys'")
    lines.append(
        f"- **Models tested**: {combined.get('n_models', 0)} "
        f"({combined.get('n_v52', 0)} from v52, "
        f"{combined.get('n_v53', 0)} from v53)"
    )
    lines.append(f"- **Accuracy**: {combined.get('accuracy', 0):.0%}")
    lines.append(f"- **Precision**: {combined.get('precision', 0):.2f}")
    lines.append(f"- **Recall**: {combined.get('recall', 0):.2f}")
    lines.append(f"- **F1**: {combined.get('f1', 0):.2f}")
    lines.append(
        f"- **Confusion**: TP={cm.get('tp',0)} FP={cm.get('fp',0)} "
        f"FN={cm.get('fn',0)} TN={cm.get('tn',0)}"
    )
    lines.append("")

    # Factor analysis
    lines.append("## Factor Correlations")
    lines.append("")
    lines.append("| Factor | Spearman rho | p-value |")
    lines.append("|--------|-------------|---------|")
    for name, vals in factors.get("correlations", {}).items():
        lines.append(f"| {name} | {vals['rho']:.4f} | {vals['p']:.4f} |")
    lines.append("")

    # Stop condition evaluation
    lines.append("## Stop Condition Evaluation")
    lines.append("")
    new_flagged = [
        r
        for r in combined.get("models", [])
        if r["source"] == "v53" and r["error_ratio"] > 2.0
    ]
    if not new_flagged:
        lines.append(
            "All new models show ratio < 2.0. The classifier is robust "
            "and Qwen2.5-7B remains the only model flagged as sensitive."
        )
    else:
        names = [r["model"] for r in new_flagged]
        lines.append(
            f"New models with ratio > 2.0: {', '.join(names)}. "
            "The classifier generalizes beyond Qwen."
        )
    lines.append("")

    # Errors
    if model_errors:
        lines.append("## Model Errors")
        lines.append("")
        for m, e in model_errors.items():
            err_short = str(e).split("\n")[0][:80]
            lines.append(f"- **{m}**: {err_short}")
        lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    # Will be filled after seeing results
    lines.append("(See combined results and factor correlations above.)")
    lines.append("")

    # Plots
    lines.append("## Plots")
    lines.append("")
    lines.append("![gqa_vs_ratio](plots/gqa_ratio_vs_error_ratio.png)")
    lines.append("![rope_vs_ratio](plots/rope_theta_vs_error_ratio.png)")
    lines.append("![interaction](plots/gqa_rope_interaction.png)")
    lines.append("")

    summary_path = os.path.join(RESULTS_ROOT, "bpa53_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")
    return summary_path


# =========================================================
# Main
# =========================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("BPA v53: Ratio Classifier Stress Test")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    info = gpu_info()
    print(f"GPU: {info['device']} ({info['total_gb']} GB)")
    print(f"Torch: {info['torch']}, HIP: {info['hip']}")

    # Stage 1
    t1 = time.time()
    census = architecture_census()
    print(f"  Stage 1 time: {time.time()-t1:.0f}s")

    v53_results = {}
    model_errors = {}

    # Stage 2: Per-model ratio test
    for model_name, model_id in MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name} ({model_id})")
        print(f"{'#'*60}")

        if "error" in census.get(model_name, {}):
            print(f"  Skipping — census error")
            model_errors[model_name] = census[model_name]["error"]
            continue

        t_model = time.time()
        try:
            model, tokenizer, config = load_model(model_id)
        except Exception as e:
            err_short = str(e).split("\n")[0][:120]
            print(f"  ERROR loading: {err_short}")
            model_errors[model_name] = str(e)
            save_json(
                {"model": model_name, "error": str(e)},
                f"error_{model_name}.json",
            )
            clear_gpu()
            continue

        try:
            res = ratio_test(model, tokenizer, config, model_name)
            v53_results[model_name] = res

            # If ratio > 2.0, confirm with INT4
            if res.get("predicted_sensitive", False):
                int4_result = confirm_int4(model, tokenizer, model_name)
                res["int4_confirmation"] = int4_result
                # Update floor based on INT4 result
                if not int4_result.get("collapsed", True):
                    res["observed_floor"] = "int4_or_below"
                save_json(res, f"ratio_test_{model_name}.json")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during ratio test")
            model_errors[model_name] = "OOM"
            v53_results[model_name] = {"model": model_name, "error": "OOM"}
            clear_gpu()
        except Exception as e:
            print(f"  ERROR: {e}")
            model_errors[model_name] = str(e)
            v53_results[model_name] = {"model": model_name, "error": str(e)}

        elapsed_model = time.time() - t_model
        print(f"  Model time: {elapsed_model:.0f}s")

        unload_model(model)
        del tokenizer, config
        clear_gpu()

    # Stage 3: Combined classifier evaluation
    t3 = time.time()
    combined = combined_classifier_evaluation(v53_results, census)
    print(f"  Stage 3 time: {time.time()-t3:.0f}s")

    # Stage 4: Factor analysis
    t4 = time.time()
    factors = factor_analysis(combined)
    print(f"  Stage 4 time: {time.time()-t4:.0f}s")

    # Stage 5: Summary
    elapsed = time.time() - t_start
    metadata = {
        "experiment": "BPA-v53",
        "date": str(datetime.now()),
        "gpu": gpu_info(),
        "elapsed_seconds": round(elapsed, 1),
        "models_tested": [m for m in v53_results if "error" not in v53_results[m]],
        "models_failed": model_errors,
    }
    save_json(metadata, "bpa53_metadata.json")

    write_summary(census, v53_results, combined, factors, model_errors, elapsed)

    print(f"\n{'='*60}")
    print(f"BPA v53 COMPLETE — {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
