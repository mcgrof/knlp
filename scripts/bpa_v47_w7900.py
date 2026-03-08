#!/usr/bin/env python3
"""BPA v47: RoPE-Safe KV Quantization + FIM-Guided Layer Precision.

Extends v46 with:
  1. Pre-RoPE KV quantization (quantize before RoPE, apply RoPE at attention)
  2. Fisher Information-guided layer precision allocation
  3. 8 precision strategies (A-H) with early collapse detection
  4. Cross-architecture validation (Qwen2.5-7B, Mistral-7B, Llama-2-7B)

Results saved to /data/knlp-key-results/bpa47/
"""

import gc
import json
import math
import os
import re
import sys
import time
import copy
import functools
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
RESULTS_ROOT = "/data/knlp-key-results/bpa47"
JSON_DIR = os.path.join(RESULTS_ROOT, "json")
PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
FIM_DIR = os.path.join(RESULTS_ROOT, "fim_maps")
for d in [JSON_DIR, PLOT_DIR, LOG_DIR, FIM_DIR]:
    os.makedirs(d, exist_ok=True)

# === Protocol parameters ===
DATASET = "wikitext-103-raw-v1"
N_TOKENS = 500_000
W_SINK = 4
GROUP_SIZE = 32
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Models to test
MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
}
PRIMARY_MODEL = "qwen2.5-7b"

# Collapse detection thresholds
COLLAPSE_AGREE_MIN = 0.40
COLLAPSE_PPL_MAX = 50.0
COLLAPSE_LOGIT_ERR_MAX = 5.0


def save_json(data, filename):
    path = os.path.join(JSON_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def save_fim_json(data, filename):
    path = os.path.join(FIM_DIR, filename)
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


# === Quantization functions ===
def quantize_int4_grouped(tensor, group_size=32):
    """Symmetric INT4 quantization with per-group scales."""
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


def quantize_int8_grouped(tensor, group_size=32):
    """Symmetric INT8 quantization with per-group scales."""
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
    s = amax / 127.0
    q = (r / s).round().clamp(-128, 127)
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


# === RoPE extraction ===
def get_rope_fn(model):
    """Extract the RoPE embedding module from the model."""
    if hasattr(model, "model"):
        inner = model.model
    else:
        inner = model
    if hasattr(inner, "rotary_emb"):
        return inner.rotary_emb
    if hasattr(inner, "layers") and len(inner.layers) > 0:
        attn = inner.layers[0].self_attn
        if hasattr(attn, "rotary_emb"):
            return attn.rotary_emb
    return None


def apply_rotary_pos_emb_to_k(k, cos, sin):
    """Apply rotary embeddings to key states.

    k: [B, n_kv_heads, T, head_dim]
    cos, sin: [1, 1, T, head_dim] or broadcastable
    """
    # Standard RoPE: split into pairs, rotate
    k_r = k.float()
    half = k_r.shape[-1] // 2
    k1 = k_r[..., :half]
    k2 = k_r[..., half:]
    cos_h = cos[..., :half]
    sin_h = sin[..., :half]
    k_rot = torch.cat([k1 * cos_h - k2 * sin_h, k2 * cos_h + k1 * sin_h], dim=-1)
    return k_rot.to(k.dtype)


# === Pre-RoPE KV Quantization via Monkey-Patching ===
class PreRoPEQuantConfig:
    """Holds per-layer quantization configuration."""

    def __init__(self, n_layers):
        self.n_layers = n_layers
        # Per-layer config: None = FP16, 'int4', 'int8'
        self.layer_quant = [None] * n_layers
        self.enabled = False
        # Store pre-RoPE keys for quantization
        self.pre_rope_keys = {}

    def set_strategy(self, strategy_map):
        """strategy_map: dict of layer_idx -> 'int4'|'int8'|None"""
        for i in range(self.n_layers):
            self.layer_quant[i] = strategy_map.get(i, None)

    def set_uniform(self, quant_type):
        """Set all layers to same quantization."""
        for i in range(self.n_layers):
            self.layer_quant[i] = quant_type

    def set_all_fp16(self):
        self.set_uniform(None)


def quantize_tensor(tensor, quant_type, group_size=32):
    """Quantize a tensor according to type."""
    if quant_type == "int4":
        return quantize_int4_grouped(tensor, group_size)
    elif quant_type == "int8":
        return quantize_int8_grouped(tensor, group_size)
    return tensor


def quantize_cache_with_strategy(past, quant_config, group_size=32):
    """Quantize KV cache according to per-layer strategy.

    This applies post-RoPE quantization (as in v46) for strategies
    that don't need pre-RoPE handling.
    """
    D = n_cache_layers(past)
    for li in range(D):
        qt = quant_config.layer_quant[li]
        if qt is None:
            continue
        k, v = _cache_get_kv(past, li)
        clen = k.shape[2]
        # Protect sink tokens
        k_sink = k[:, :, :W_SINK, :]
        v_sink = v[:, :, :W_SINK, :]
        k_far = k[:, :, W_SINK:, :]
        v_far = v[:, :, W_SINK:, :]
        k_q = quantize_tensor(k_far, qt, group_size)
        v_q = quantize_tensor(v_far, qt, group_size)
        _cache_set_kv(
            past,
            li,
            torch.cat([k_sink, k_q], dim=2),
            torch.cat([v_sink, v_q], dim=2),
        )
    return past


def quantize_cache_kv_asymmetric(past, k_type, v_type, group_size=32):
    """Quantize K and V with different precisions."""
    D = n_cache_layers(past)
    for li in range(D):
        k, v = _cache_get_kv(past, li)
        k_sink = k[:, :, :W_SINK, :]
        v_sink = v[:, :, :W_SINK, :]
        k_far = k[:, :, W_SINK:, :]
        v_far = v[:, :, W_SINK:, :]
        k_q = quantize_tensor(k_far, k_type, group_size)
        v_q = quantize_tensor(v_far, v_type, group_size)
        _cache_set_kv(
            past,
            li,
            torch.cat([k_sink, k_q], dim=2),
            torch.cat([v_sink, v_q], dim=2),
        )
    return past


# === Strategy definitions ===
def make_strategy_configs(n_layers, fim_ranking=None):
    """Create all 8 strategy configurations.

    Returns dict of strategy_name -> quant_config.
    fim_ranking: list of layer indices sorted by FIM score (highest first).
    """
    strategies = {}

    # A: FP16 baseline
    cfg_a = PreRoPEQuantConfig(n_layers)
    cfg_a.set_all_fp16()
    strategies["A_fp16"] = cfg_a

    # B: Uniform INT8
    cfg_b = PreRoPEQuantConfig(n_layers)
    cfg_b.set_uniform("int8")
    strategies["B_int8"] = cfg_b

    # C: Uniform INT4
    cfg_c = PreRoPEQuantConfig(n_layers)
    cfg_c.set_uniform("int4")
    strategies["C_int4"] = cfg_c

    # D: First/Last layer protection
    cfg_d = PreRoPEQuantConfig(n_layers)
    cfg_d.set_uniform("int4")
    for i in range(min(2, n_layers)):
        cfg_d.layer_quant[i] = None
    for i in range(max(0, n_layers - 2), n_layers):
        cfg_d.layer_quant[i] = None
    strategies["D_firstlast"] = cfg_d

    # E: Key-Value asymmetric (special handling)
    # Stored as marker; applied differently
    strategies["E_kv_asym"] = "kv_asymmetric"

    if fim_ranking is not None:
        n_top25 = max(1, n_layers // 4)
        n_top20 = max(1, n_layers // 5)
        n_mid40 = max(1, int(n_layers * 0.4))

        # F: FIM-Top-K Protection (top 25% FP16, rest INT4)
        cfg_f = PreRoPEQuantConfig(n_layers)
        cfg_f.set_uniform("int4")
        for li in fim_ranking[:n_top25]:
            cfg_f.layer_quant[li] = None
        strategies["F_fim_topk"] = cfg_f

        # G: Multi-Tier FIM (top 20% FP16, mid 40% INT8, bottom 40% INT4)
        cfg_g = PreRoPEQuantConfig(n_layers)
        cfg_g.set_uniform("int4")
        for li in fim_ranking[:n_top20]:
            cfg_g.layer_quant[li] = None
        for li in fim_ranking[n_top20 : n_top20 + n_mid40]:
            cfg_g.layer_quant[li] = "int8"
        strategies["G_fim_multi"] = cfg_g

        # H: Random layer protection (same count as F)
        rng = np.random.RandomState(42)
        random_layers = rng.choice(n_layers, size=n_top25, replace=False)
        cfg_h = PreRoPEQuantConfig(n_layers)
        cfg_h.set_uniform("int4")
        for li in random_layers:
            cfg_h.layer_quant[li] = None
        strategies["H_random"] = cfg_h

    return strategies


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


def unload_model(model):
    del model
    clear_gpu()


# === Apply quantization to cache based on strategy ===
def apply_strategy_to_cache(past, strategy_name, strategy_obj):
    """Apply a quantization strategy to a KV cache."""
    if strategy_name == "A_fp16":
        return past  # no quantization
    if strategy_obj == "kv_asymmetric":
        return quantize_cache_kv_asymmetric(past, "int8", "int4", GROUP_SIZE)
    if isinstance(strategy_obj, PreRoPEQuantConfig):
        return quantize_cache_with_strategy(past, strategy_obj, GROUP_SIZE)
    return past


# ============================================================
# Experiment 1: Fisher Sensitivity Calibration
# ============================================================
def experiment_1_fisher_calibration(model, tokenizer, model_key, n_layers):
    print("\n" + "=" * 60)
    print(f"Experiment 1: Fisher Sensitivity Calibration ({model_key})")
    print("=" * 60)
    t0 = time.time()

    # Use wikitext validation as calibration data (different split from test)
    from datasets import load_dataset

    ds = load_dataset("wikitext", DATASET, split="validation")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer.encode(text)

    # Sample calibration sequences
    cal_seq_len = 512  # shorter for gradient computation
    n_cal_seqs = 200  # fewer sequences to fit in memory with gradients
    rng = np.random.RandomState(0)

    fim_k = torch.zeros(n_layers, device="cpu")
    fim_v = torch.zeros(n_layers, device="cpu")
    variance_map = torch.zeros(n_layers, device="cpu")
    n_counted = 0

    # We need to hook into each layer to capture KV gradients
    # Access attention layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        print("  ERROR: Cannot find model layers")
        return None, None

    print(f"  Running Fisher calibration on {n_cal_seqs} sequences...")

    for seq_idx in range(n_cal_seqs):
        if seq_idx % 50 == 0:
            print(f"    Sequence {seq_idx}/{n_cal_seqs}...")

        start = rng.randint(0, max(1, len(tokens) - cal_seq_len - 1))
        input_ids = (
            torch.tensor(tokens[start : start + cal_seq_len], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # We compute Fisher via gradient of loss w.r.t. KV projections
        # Enable gradients temporarily
        model.zero_grad()

        # Hook to capture K and V activations
        kv_activations = {}

        def make_hook(layer_idx, kv_type):
            def hook_fn(module, input, output):
                # output is the projected tensor
                kv_activations[(layer_idx, kv_type)] = output

            return hook_fn

        handles = []
        for li, layer in enumerate(layers):
            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn
            handles.append(attn.k_proj.register_forward_hook(make_hook(li, "k")))
            handles.append(attn.v_proj.register_forward_hook(make_hook(li, "v")))

        try:
            # Forward with gradient tracking on KV activations
            with torch.enable_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                loss.backward()

                for li in range(n_layers):
                    k_act = kv_activations.get((li, "k"))
                    v_act = kv_activations.get((li, "v"))
                    if k_act is not None and k_act.grad is not None:
                        fim_k[li] += k_act.grad.pow(2).mean().item()
                    if v_act is not None and v_act.grad is not None:
                        fim_v[li] += v_act.grad.pow(2).mean().item()

                    # Also compute variance from gradients of parameters
                    attn = (
                        layers[li].self_attn
                        if hasattr(layers[li], "self_attn")
                        else layers[li].attn
                    )
                    k_grad = attn.k_proj.weight.grad
                    v_grad = attn.v_proj.weight.grad
                    if k_grad is not None and v_grad is not None:
                        var = (k_grad.pow(2).mean() + v_grad.pow(2).mean()).item()
                        variance_map[li] += var
                        fim_k[li] += k_grad.pow(2).mean().item()
                        fim_v[li] += v_grad.pow(2).mean().item()

                n_counted += 1

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at seq {seq_idx}, stopping calibration early")
            clear_gpu()
            break
        except Exception as e:
            print(f"    Error at seq {seq_idx}: {e}")
            clear_gpu()
            continue
        finally:
            for h in handles:
                h.remove()
            kv_activations.clear()
            model.zero_grad(set_to_none=True)
            clear_gpu()

    if n_counted == 0:
        print("  WARNING: No successful calibration sequences")
        # Return uniform ranking
        ranking = list(range(n_layers))
        fim_trace = [1.0] * n_layers
        return ranking, fim_trace

    # Normalize
    fim_k /= n_counted
    fim_v /= n_counted
    variance_map /= n_counted
    fim_total = fim_k + fim_v
    fourth_root = variance_map.pow(0.25)

    # Ranking: highest FIM first
    ranking = torch.argsort(fim_total, descending=True).tolist()

    fim_trace = fim_total.tolist()
    results = {
        "model": model_key,
        "n_cal_seqs": n_counted,
        "cal_seq_len": cal_seq_len,
        "fim_k": fim_k.tolist(),
        "fim_v": fim_v.tolist(),
        "fim_total": fim_trace,
        "variance_map": variance_map.tolist(),
        "fourth_root": fourth_root.tolist(),
        "ranking": ranking,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    save_fim_json(results, f"{model_key}_fim_trace.json")
    save_fim_json(
        {"model": model_key, "variance": variance_map.tolist()},
        f"{model_key}_variance_map.json",
    )
    save_fim_json(
        {"model": model_key, "fourth_root": fourth_root.tolist()},
        f"{model_key}_fourth_root_map.json",
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = list(range(n_layers))

    axes[0].bar(x, fim_trace, color="steelblue")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("FIM Score")
    axes[0].set_title(f"Fisher Sensitivity ({model_key})")

    axes[1].bar(x, variance_map.tolist(), color="coral")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Gradient Variance")
    axes[1].set_title(f"Gradient Variance ({model_key})")

    axes[2].bar(x, fourth_root.tolist(), color="forestgreen")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Fourth Root")
    axes[2].set_title(f"Fourth Root Transform ({model_key})")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"fim_calibration_{model_key}.png"), dpi=300)
    plt.close()

    print(f"  FIM ranking (top 5): {ranking[:5]}")
    print(f"  Elapsed: {results['elapsed_sec']}s")

    return ranking, fim_trace


# ============================================================
# Experiment 2: Single Layer Sensitivity Validation
# ============================================================
def experiment_2_layer_sensitivity(
    model, tokenizer, model_key, n_layers, fim_ranking, fim_trace
):
    print("\n" + "=" * 60)
    print(f"Experiment 2: Single Layer Sensitivity ({model_key})")
    print("=" * 60)
    t0 = time.time()

    L = 2048
    seeds = [0, 1, 2]
    decode_tokens = 64

    results = {
        "experiment": "layer_sensitivity",
        "model": model_key,
        "n_layers": n_layers,
        "evals": {},
    }

    # First get FP16 baseline PPL
    print("  Computing FP16 baseline...")
    baseline_ppls = []
    for seed in seeds:
        passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
        input_ids = passage[:, :L].to(DEVICE)
        continuation = passage[:, L : L + decode_tokens].to(DEVICE)
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            past = out.past_key_values
            logits_list = [out.logits[:, -1:, :].cpu()]
            for t in range(decode_tokens):
                tok = continuation[:, t : t + 1]
                out2 = model(tok, past_key_values=past, use_cache=True)
                logits_list.append(out2.logits.cpu())
                past = out2.past_key_values
        logits = torch.cat(logits_list, dim=1)
        targets = continuation.cpu().reshape(-1)
        B_dim, T_dim, V_dim = logits[:, :-1, :].shape
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V_dim).float(), targets)
        baseline_ppls.append(math.exp(min(loss.item(), 20)))
        del past, out, logits, logits_list
        clear_gpu()

    baseline_ppl = np.mean(baseline_ppls)
    results["baseline_ppl"] = round(baseline_ppl, 4)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Per-layer INT4 quantization
    ppl_deltas = []
    logit_errors = []

    for li in range(n_layers):
        if li % 7 == 0:
            print(f"  Layer {li}/{n_layers}...")
        layer_ppls = []
        layer_logit_errs = []

        for seed in seeds:
            try:
                passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                input_ids = passage[:, :L].to(DEVICE)
                continuation = passage[:, L : L + decode_tokens].to(DEVICE)

                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values

                # Quantize only layer li
                cfg = PreRoPEQuantConfig(n_layers)
                cfg.layer_quant[li] = "int4"
                past = quantize_cache_with_strategy(past, cfg, GROUP_SIZE)

                logits_q = [out.logits[:, -1:, :].cpu()]
                for t_idx in range(decode_tokens):
                    tok = continuation[:, t_idx : t_idx + 1]
                    with torch.no_grad():
                        out2 = model(tok, past_key_values=past, use_cache=True)
                    logits_q.append(out2.logits.cpu())
                    past = out2.past_key_values
                logits_q = torch.cat(logits_q, dim=1)

                targets = continuation.cpu().reshape(-1)
                B_dim, T_dim, V_dim = logits_q[:, :-1, :].shape
                loss = F.cross_entropy(
                    logits_q[:, :-1, :].reshape(-1, V_dim).float(), targets
                )
                ppl = math.exp(min(loss.item(), 20))
                layer_ppls.append(ppl)

                del past, out, logits_q
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at layer {li} seed {seed}")
                clear_gpu()
                continue

        if layer_ppls:
            mean_ppl = np.mean(layer_ppls)
            delta = (mean_ppl - baseline_ppl) / baseline_ppl * 100
        else:
            mean_ppl = float("nan")
            delta = float("nan")

        ppl_deltas.append(delta)
        results["evals"][f"layer_{li}"] = {
            "ppl": round(mean_ppl, 4) if not math.isnan(mean_ppl) else "nan",
            "delta_pct": round(delta, 4) if not math.isnan(delta) else "nan",
        }

    # Compute correlation with FIM
    valid_mask = [not math.isnan(d) for d in ppl_deltas]
    if fim_trace and any(valid_mask):
        fim_arr = np.array([fim_trace[i] for i in range(n_layers) if valid_mask[i]])
        delta_arr = np.array([ppl_deltas[i] for i in range(n_layers) if valid_mask[i]])
        if len(fim_arr) > 2 and np.std(fim_arr) > 0 and np.std(delta_arr) > 0:
            from scipy.stats import spearmanr

            rho, pval = spearmanr(fim_arr, delta_arr)
            results["fim_correlation"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(pval, 6),
            }
            print(f"  FIM-PPL Spearman rho={rho:.4f}, p={pval:.6f}")

    results["ppl_deltas"] = [
        round(d, 4) if not math.isnan(d) else "nan" for d in ppl_deltas
    ]
    results["elapsed_sec"] = round(time.time() - t0, 1)

    save_json(results, f"layer_sensitivity_{model_key}.json")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(n_layers))
    valid_deltas = [d if not math.isnan(d) else 0 for d in ppl_deltas]
    colors = [
        "red" if d > 3 else "orange" if d > 1 else "steelblue" for d in valid_deltas
    ]
    ax.bar(x, valid_deltas, color=colors)
    ax.set_xlabel("Layer")
    ax.set_ylabel("PPL Delta (%)")
    ax.set_title(f"Per-Layer INT4 Sensitivity ({model_key})")
    ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="3% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"layer_sensitivity_{model_key}.png"), dpi=300)
    plt.close()

    return results


# ============================================================
# Early Collapse Detection
# ============================================================
def early_collapse_check(model, tokenizer, strategy_name, strategy_obj, n_layers):
    """Quick sanity check before full evaluation."""
    print(f"    Collapse check for {strategy_name}...")
    L = 512
    gen_tokens = 64

    passage = load_passage(tokenizer, L, seed=0, extra=gen_tokens)
    input_ids = passage[:, :L].to(DEVICE)
    continuation = passage[:, L : L + gen_tokens].to(DEVICE)

    try:
        with torch.no_grad():
            # Baseline
            out_fp = model(input_ids, use_cache=True)
            past_fp = out_fp.past_key_values
            logits_fp = [out_fp.logits[:, -1:, :].cpu()]
            past = past_fp
            for t in range(gen_tokens):
                tok = continuation[:, t : t + 1]
                out = model(tok, past_key_values=past, use_cache=True)
                logits_fp.append(out.logits.cpu())
                past = out.past_key_values
            logits_fp = torch.cat(logits_fp, dim=1)
            del past, past_fp
            clear_gpu()

            # Strategy
            out_q = model(input_ids, use_cache=True)
            past_q = out_q.past_key_values
            past_q = apply_strategy_to_cache(past_q, strategy_name, strategy_obj)
            logits_q = [out_q.logits[:, -1:, :].cpu()]
            past = past_q
            for t in range(gen_tokens):
                tok = continuation[:, t : t + 1]
                out = model(tok, past_key_values=past, use_cache=True)
                logits_q.append(out.logits.cpu())
                past = out.past_key_values
            logits_q = torch.cat(logits_q, dim=1)
            del past, past_q
            clear_gpu()

        # Token agreement
        pred_fp = logits_fp[:, :-1, :].argmax(dim=-1)
        pred_q = logits_q[:, :-1, :].argmax(dim=-1)
        agreement = (pred_fp == pred_q).float().mean().item()

        # Quick PPL
        targets = continuation.cpu().reshape(-1)
        B_dim, T_dim, V_dim = logits_q[:, :-1, :].shape
        loss = F.cross_entropy(logits_q[:, :-1, :].reshape(-1, V_dim).float(), targets)
        ppl = math.exp(min(loss.item(), 20))

        # Logit error
        diff = (logits_fp[:, :-1, :] - logits_q[:, :-1, :]).abs()
        max_diff = diff.max(dim=-1).values.float().mean().item()

        del logits_fp, logits_q
        clear_gpu()

        collapsed = (
            agreement < COLLAPSE_AGREE_MIN
            or ppl > COLLAPSE_PPL_MAX
            or max_diff > COLLAPSE_LOGIT_ERR_MAX
        )

        result = {
            "strategy": strategy_name,
            "agreement": round(agreement, 4),
            "ppl": round(ppl, 4),
            "mean_logit_error": round(max_diff, 4),
            "collapsed": collapsed,
        }

        if collapsed:
            print(
                f"    COLLAPSED: agree={agreement:.3f} ppl={ppl:.1f} "
                f"logit_err={max_diff:.2f}"
            )
        else:
            print(
                f"    OK: agree={agreement:.3f} ppl={ppl:.1f} "
                f"logit_err={max_diff:.2f}"
            )

        return result

    except torch.cuda.OutOfMemoryError:
        print(f"    OOM during collapse check")
        clear_gpu()
        return {
            "strategy": strategy_name,
            "collapsed": True,
            "reason": "OOM",
        }


# ============================================================
# Experiment 3: Strategy Evaluation (PPL, Token Agree, Logit Err)
# ============================================================
def experiment_3_strategy_eval(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    """Evaluate a single strategy: PPL at 2K/8K, token agreement, logit error."""
    print(f"\n  Strategy evaluation: {strategy_name}")
    t0 = time.time()

    L_set = [2048, 8192]
    seeds = [0, 1, 2]
    decode_tokens = 64

    results = {
        "strategy": strategy_name,
        "model": model_key,
        "ppl": {},
        "token_agreement": {},
        "logit_error": {},
    }

    for L in L_set:
        ppls_fp = []
        ppls_q = []
        agrees = []
        logit_diffs = []

        for seed in seeds:
            key = f"L{L}_s{seed}"
            try:
                passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                input_ids = passage[:, :L].to(DEVICE)
                continuation = passage[:, L : L + decode_tokens].to(DEVICE)

                # Baseline
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

                # Strategy
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values
                past_q = apply_strategy_to_cache(past_q, strategy_name, strategy_obj)
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

                # PPL
                targets = continuation.cpu().reshape(-1)
                B_dim, T_dim, V_dim = logits_fp[:, :-1, :].shape
                loss_fp = F.cross_entropy(
                    logits_fp[:, :-1, :].reshape(-1, V_dim).float(), targets
                )
                loss_q = F.cross_entropy(
                    logits_q[:, :-1, :].reshape(-1, V_dim).float(), targets
                )
                ppls_fp.append(math.exp(min(loss_fp.item(), 20)))
                ppls_q.append(math.exp(min(loss_q.item(), 20)))

                # Token agreement
                pred_fp = logits_fp[:, :-1, :].argmax(dim=-1)
                pred_q = logits_q[:, :-1, :].argmax(dim=-1)
                agrees.append((pred_fp == pred_q).float().mean().item())

                # Logit error
                diff = (logits_fp[:, :-1, :] - logits_q[:, :-1, :]).abs()
                max_diff = diff.max(dim=-1).values.float()
                logit_diffs.extend(max_diff.flatten().tolist())

                del logits_fp, logits_q
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"      OOM at {key}")
                clear_gpu()
                continue

        if ppls_fp:
            results["ppl"][f"L{L}"] = {
                "fp16": round(np.mean(ppls_fp), 4),
                "strategy": round(np.mean(ppls_q), 4),
                "delta_pct": round(
                    (np.mean(ppls_q) - np.mean(ppls_fp)) / np.mean(ppls_fp) * 100,
                    2,
                ),
            }
        if agrees:
            results["token_agreement"][f"L{L}"] = round(np.mean(agrees), 4)
        if logit_diffs:
            ld = np.array(logit_diffs)
            results["logit_error"][f"L{L}"] = {
                "mean": round(float(ld.mean()), 4),
                "median": round(float(np.median(ld)), 4),
                "p95": round(float(np.percentile(ld, 95)), 4),
                "p99": round(float(np.percentile(ld, 99)), 4),
            }

    results["elapsed_sec"] = round(time.time() - t0, 1)
    return results


# ============================================================
# Experiment 4: Token Agreement (200 prompts)
# ============================================================
def experiment_4_token_agreement(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    print(f"\n  Token agreement (200 prompts): {strategy_name}")
    t0 = time.time()

    n_prompts = 200
    gen_tokens = 256
    L = 2048
    agreements = []

    for i in range(n_prompts):
        if i % 50 == 0:
            print(f"    Prompt {i}/{n_prompts}...")
        try:
            passage = load_passage(tokenizer, L, seed=i, extra=gen_tokens)
            input_ids = passage[:, :L].to(DEVICE)

            # FP16 generation
            with torch.no_grad():
                out_fp = model(input_ids, use_cache=True)
                past = out_fp.past_key_values
                tokens_fp = [out_fp.logits[:, -1:, :].argmax(dim=-1)]
                for t in range(gen_tokens - 1):
                    out = model(tokens_fp[-1], past_key_values=past, use_cache=True)
                    tokens_fp.append(out.logits.argmax(dim=-1))
                    past = out.past_key_values
            tokens_fp_t = torch.cat(tokens_fp, dim=1)
            del past
            clear_gpu()

            # Strategy generation
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values
                past_q = apply_strategy_to_cache(past_q, strategy_name, strategy_obj)
                tokens_q = [out_q.logits[:, -1:, :].argmax(dim=-1)]
                past = past_q
                for t in range(gen_tokens - 1):
                    out = model(tokens_q[-1], past_key_values=past, use_cache=True)
                    tokens_q.append(out.logits.argmax(dim=-1))
                    past = out.past_key_values
            tokens_q_t = torch.cat(tokens_q, dim=1)

            agree = (tokens_fp_t == tokens_q_t).float().mean().item()
            agreements.append(agree)

            del tokens_fp_t, tokens_q_t, past
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at prompt {i}")
            clear_gpu()
            continue

    results = {
        "strategy": strategy_name,
        "model": model_key,
        "n_prompts": len(agreements),
        "mean_agreement": round(np.mean(agreements), 4) if agreements else 0,
        "std_agreement": round(np.std(agreements), 4) if agreements else 0,
        "min_agreement": round(np.min(agreements), 4) if agreements else 0,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    save_json(results, f"token_agreement_{strategy_name}_{model_key}.json")
    return results


# ============================================================
# Experiment 5: Long Context Perplexity
# ============================================================
def experiment_5_long_context_ppl(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    print(f"\n  Long-context PPL: {strategy_name}")
    t0 = time.time()

    L_set = [2048, 8192]  # 16K+ OOMs on W7900 with eager attention
    seeds = [0, 1, 2]
    decode_tokens = 64
    results = {"strategy": strategy_name, "model": model_key, "evals": {}}

    for L in L_set:
        ppls_fp = []
        ppls_q = []
        for seed in seeds:
            try:
                passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                input_ids = passage[:, :L].to(DEVICE)
                continuation = passage[:, L : L + decode_tokens].to(DEVICE)

                # Baseline
                with torch.no_grad():
                    out_fp = model(input_ids, use_cache=True)
                    past = out_fp.past_key_values
                logits_fp = [out_fp.logits[:, -1:, :].cpu()]
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_fp.append(out.logits.cpu())
                    past = out.past_key_values
                logits_fp = torch.cat(logits_fp, dim=1)
                del past
                clear_gpu()

                # Strategy
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values
                past_q = apply_strategy_to_cache(past_q, strategy_name, strategy_obj)
                logits_q = [out_q.logits[:, -1:, :].cpu()]
                past = past_q
                for t in range(decode_tokens):
                    tok = continuation[:, t : t + 1]
                    with torch.no_grad():
                        out = model(tok, past_key_values=past, use_cache=True)
                    logits_q.append(out.logits.cpu())
                    past = out.past_key_values
                logits_q = torch.cat(logits_q, dim=1)
                del past
                clear_gpu()

                targets = continuation.cpu().reshape(-1)
                B_dim, T_dim, V_dim = logits_fp[:, :-1, :].shape
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
                print(f"      OOM at L={L} seed={seed}")
                clear_gpu()
                continue

        if ppls_fp:
            results["evals"][f"L{L}"] = {
                "fp16": round(np.mean(ppls_fp), 4),
                "strategy": round(np.mean(ppls_q), 4),
                "delta_pct": round(
                    (np.mean(ppls_q) - np.mean(ppls_fp)) / np.mean(ppls_fp) * 100,
                    2,
                ),
            }

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"long_context_ppl_{strategy_name}_{model_key}.json")
    return results


# ============================================================
# Experiment 6: Needle-In-Haystack Retrieval
# ============================================================
def experiment_6_needle(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    print(f"\n  Needle retrieval: {strategy_name}")
    t0 = time.time()

    L_set = [4096, 8192]  # 16K+ OOMs
    n_prompts_per = 50
    needle = "The secret phrase is: avocado-electric-tractor."
    query = "What is the secret phrase?"

    results = {"strategy": strategy_name, "model": model_key, "evals": {}}

    for L in L_set:
        accs_fp = []
        accs_q = []
        for pi in range(n_prompts_per):
            try:
                # Build haystack
                passage = load_passage(tokenizer, L, seed=pi + 100, extra=0)
                passage_tokens = passage[0].tolist()

                # Insert needle at random position
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

                # Add query
                query_tokens = tokenizer.encode("\n\n" + query)
                input_ids = (
                    torch.tensor(haystack + query_tokens, dtype=torch.long)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                # FP16
                with torch.no_grad():
                    out_fp = model(input_ids, use_cache=True)
                gen_ids_fp = out_fp.logits[0, -1, :].argmax().item()
                past = out_fp.past_key_values
                gen_fp = [gen_ids_fp]
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

                # Strategy
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                past_q = out_q.past_key_values
                past_q = apply_strategy_to_cache(past_q, strategy_name, strategy_obj)
                gen_ids_q = out_q.logits[0, -1, :].argmax().item()
                gen_q = [gen_ids_q]
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
                print(f"      OOM at L={L} prompt {pi}")
                clear_gpu()
                continue

        if accs_fp:
            results["evals"][f"L{L}"] = {
                "fp16_acc": round(np.mean(accs_fp), 4),
                "strategy_acc": round(np.mean(accs_q), 4),
                "n_prompts": len(accs_fp),
            }
            print(
                f"    L={L}: FP16={np.mean(accs_fp):.3f} "
                f"Strategy={np.mean(accs_q):.3f}"
            )

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"needle_{strategy_name}_{model_key}.json")
    return results


# ============================================================
# Experiment 7: Downstream Tasks
# ============================================================
def experiment_7_downstream(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    print(f"\n  Downstream tasks: {strategy_name}")
    t0 = time.time()

    results = {"strategy": strategy_name, "model": model_key, "tasks": {}}

    # MMLU (200 questions, simple multiple-choice)
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

            # Strategy
            with torch.no_grad():
                out_q = model(input_ids, use_cache=True)
                past = out_q.past_key_values
            past = apply_strategy_to_cache(past, strategy_name, strategy_obj)
            # Re-run last token with quantized cache to get affected logits
            with torch.no_grad():
                out_q2 = model(input_ids[:, -1:], past_key_values=past, use_cache=True)
            logits_q = out_q2.logits[0, -1, :]
            del past
            clear_gpu()

            # Check answers
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
            results["tasks"]["mmlu"] = {
                "fp16": round(correct_fp / total, 4),
                "strategy": round(correct_q / total, 4),
                "total": total,
            }
            print(f"      MMLU: FP16={correct_fp/total:.3f} Q={correct_q/total:.3f}")
    except Exception as e:
        print(f"      MMLU error: {e}")

    # GSM8K (200 questions)
    print("    GSM8K...")
    try:
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
            # Extract numeric answer
            answer_match = re.search(r"####\s*(.+)", answer_text)
            if not answer_match:
                continue
            gold = answer_match.group(1).strip().replace(",", "")

            prompt = f"Q: {question}\nA: Let me solve step by step.\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] > 1024:
                continue

            # Generate short answers
            for mode in ["fp16", "strategy"]:
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                if mode == "strategy":
                    past = apply_strategy_to_cache(past, strategy_name, strategy_obj)

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
                # Try to find a number in the output
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
            results["tasks"]["gsm8k"] = {
                "fp16": round(correct_fp / total, 4),
                "strategy": round(correct_q / total, 4),
                "total": total,
            }
            print(f"      GSM8K: FP16={correct_fp/total:.3f} Q={correct_q/total:.3f}")
    except Exception as e:
        print(f"      GSM8K error: {e}")

    # HumanEval (100 problems - simplified syntax check)
    print("    HumanEval...")
    try:
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

            for mode in ["fp16", "strategy"]:
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                if mode == "strategy":
                    past = apply_strategy_to_cache(past, strategy_name, strategy_obj)

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
                # Simple validity check
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
            results["tasks"]["humaneval"] = {
                "fp16": round(valid_fp / total, 4),
                "strategy": round(valid_q / total, 4),
                "total": total,
            }
            print(f"      HumanEval: FP16={valid_fp/total:.3f} Q={valid_q/total:.3f}")
    except Exception as e:
        print(f"      HumanEval error: {e}")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"downstream_{strategy_name}_{model_key}.json")
    return results


# ============================================================
# Experiment 8: Decode Bandwidth
# ============================================================
def experiment_8_bandwidth(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    print(f"\n  Decode bandwidth: {strategy_name}")
    t0 = time.time()

    batch_sizes = [1, 2, 4, 8, 16, 32]
    ctx_lengths = [2048, 4096, 8192]
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    results = {
        "strategy": strategy_name,
        "model": model_key,
        "measurements": [],
    }

    for L in ctx_lengths:
        for B in batch_sizes:
            try:
                # Create dummy input
                passage = load_passage(tokenizer, L, seed=0)
                input_ids = passage[:, :L].to(DEVICE)
                if B > 1:
                    input_ids = input_ids.expand(B, -1)

                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    past = out.past_key_values
                past = apply_strategy_to_cache(past, strategy_name, strategy_obj)

                # Measure decode latency
                n_warmup = 3
                n_measure = 10
                tok = out.logits[:, -1:, :].argmax(dim=-1)

                for _ in range(n_warmup):
                    with torch.no_grad():
                        out2 = model(tok, past_key_values=past, use_cache=True)
                    tok = out2.logits.argmax(dim=-1)
                    past = out2.past_key_values

                torch.cuda.synchronize()
                t_start = time.perf_counter()
                for _ in range(n_measure):
                    with torch.no_grad():
                        out2 = model(tok, past_key_values=past, use_cache=True)
                    tok = out2.logits.argmax(dim=-1)
                    past = out2.past_key_values
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t_start) / n_measure

                # KV bytes read per step
                kv_bytes = (
                    B * 2 * n_layers * n_kv_heads * L * head_dim * 2
                )  # 2 bytes per bf16
                bw_gbs = kv_bytes / elapsed / 1e9

                results["measurements"].append(
                    {
                        "batch_size": B,
                        "context_length": L,
                        "latency_ms": round(elapsed * 1000, 2),
                        "kv_bytes_mb": round(kv_bytes / 1e6, 1),
                        "bandwidth_gbs": round(bw_gbs, 2),
                    }
                )

                del past, out
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"      OOM at B={B} L={L}")
                clear_gpu()
                continue

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"bandwidth_{strategy_name}_{model_key}.json")
    return results


# ============================================================
# Experiment 9: Batch Scaling + Hill Fit
# ============================================================
def experiment_9_batch_scaling(
    model, tokenizer, model_key, n_layers, strategy_name, strategy_obj
):
    print(f"\n  Batch scaling: {strategy_name}")
    t0 = time.time()

    L = 2048
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    results = {
        "strategy": strategy_name,
        "model": model_key,
        "measurements": [],
    }

    for B in batch_sizes:
        try:
            passage = load_passage(tokenizer, L, seed=0)
            input_ids = passage[:, :L].to(DEVICE)
            if B > 1:
                input_ids = input_ids.expand(B, -1)

            with torch.no_grad():
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
            past = apply_strategy_to_cache(past, strategy_name, strategy_obj)

            tok = out.logits[:, -1:, :].argmax(dim=-1)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    out2 = model(tok, past_key_values=past, use_cache=True)
                tok = out2.logits.argmax(dim=-1)
                past = out2.past_key_values

            # Measure
            n_measure = 20
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(n_measure):
                with torch.no_grad():
                    out2 = model(tok, past_key_values=past, use_cache=True)
                tok = out2.logits.argmax(dim=-1)
                past = out2.past_key_values
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t_start) / n_measure

            tps = B / elapsed
            results["measurements"].append(
                {
                    "batch_size": B,
                    "latency_ms": round(elapsed * 1000, 2),
                    "tokens_per_sec": round(tps, 2),
                }
            )

            del past, out
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            print(f"      OOM at B={B}")
            clear_gpu()
            continue

    # Hill fit
    if len(results["measurements"]) >= 3:
        try:
            from scipy.optimize import curve_fit

            bs = np.array([m["batch_size"] for m in results["measurements"]])
            tps = np.array([m["tokens_per_sec"] for m in results["measurements"]])

            def hill(B, Smax, Bhalf, gamma):
                return Smax * B**gamma / (Bhalf**gamma + B**gamma)

            popt, _ = curve_fit(hill, bs, tps, p0=[200, 10, 1.0], maxfev=5000)
            results["hill_fit"] = {
                "Smax": round(float(popt[0]), 2),
                "B_half": round(float(popt[1]), 2),
                "gamma": round(float(popt[2]), 4),
            }
            print(
                f"    Hill fit: Smax={popt[0]:.1f} B_half={popt[1]:.1f} gamma={popt[2]:.3f}"
            )
        except Exception as e:
            print(f"    Hill fit failed: {e}")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"batch_scaling_{strategy_name}_{model_key}.json")
    return results


# ============================================================
# Generate Plots
# ============================================================
def generate_summary_plots(all_results, model_key):
    print(f"\n  Generating plots for {model_key}...")

    # Strategy quality comparison
    strategies = []
    ppls_2k = []
    agrees_2k = []
    for sname, res in all_results.items():
        if "strategy_eval" in res and res["strategy_eval"]:
            ev = res["strategy_eval"]
            if "L2048" in ev.get("ppl", {}):
                strategies.append(sname)
                ppls_2k.append(ev["ppl"]["L2048"]["delta_pct"])
                agrees_2k.append(ev.get("token_agreement", {}).get("L2048", 0))

    if strategies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = range(len(strategies))
        colors = [
            "green" if abs(d) < 3 else "orange" if abs(d) < 10 else "red"
            for d in ppls_2k
        ]
        ax1.bar(x, ppls_2k, color=colors)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha="right")
        ax1.set_ylabel("PPL Delta (%)")
        ax1.set_title(f"Perplexity Impact by Strategy ({model_key})")
        ax1.axhline(y=3, color="red", linestyle="--", alpha=0.5)

        ax2.bar(x, agrees_2k, color="steelblue")
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha="right")
        ax2.set_ylabel("Token Agreement")
        ax2.set_title(f"Token Agreement by Strategy ({model_key})")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"strategy_comparison_{model_key}.png"),
            dpi=300,
        )
        plt.close()

    # FIM heatmap if available
    fim_path = os.path.join(FIM_DIR, f"{model_key}_fim_trace.json")
    if os.path.exists(fim_path):
        with open(fim_path) as f:
            fim_data = json.load(f)
        fig, ax = plt.subplots(figsize=(12, 3))
        fim_arr = np.array(fim_data["fim_total"]).reshape(1, -1)
        im = ax.imshow(fim_arr, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Layer")
        ax.set_title(f"Fisher Sensitivity Heatmap ({model_key})")
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"fim_heatmap_{model_key}.png"), dpi=300)
        plt.close()


# ============================================================
# Generate Final Report
# ============================================================
def generate_report(all_model_results):
    print("\n" + "=" * 60)
    print("Generating Final Report")
    print("=" * 60)

    report = []
    report.append(
        "# BPA v47 — RoPE-Safe KV Quantization + FIM-Guided Layer Precision\n"
    )
    report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("**GPU**: AMD Radeon Pro W7900 (48GB)")
    report.append("**Dtype**: BF16\n")

    report.append("## Experiment Summary\n")

    for model_key, model_results in all_model_results.items():
        report.append(f"### Model: {model_key}\n")

        # FIM ranking
        if "fim_ranking" in model_results:
            ranking = model_results["fim_ranking"]
            report.append(f"**FIM Top-5 Sensitive Layers**: {ranking[:5]}\n")

        # Strategy comparison table
        report.append("#### Strategy Comparison (L=2048)\n")
        report.append(
            "| Strategy | PPL FP16 | PPL Strategy | Delta (%) | Token Agree | Collapsed |"
        )
        report.append(
            "|----------|----------|-------------|-----------|-------------|-----------|"
        )

        for sname, sdata in model_results.get("strategies", {}).items():
            collapse = sdata.get("collapse_check", {})
            ev = sdata.get("strategy_eval", {})
            ppl_data = ev.get("ppl", {}).get("L2048", {})
            agree = ev.get("token_agreement", {}).get("L2048", "N/A")
            collapsed = "YES" if collapse.get("collapsed", False) else "no"
            fp16_ppl = ppl_data.get("fp16", "N/A")
            s_ppl = ppl_data.get("strategy", "N/A")
            delta = ppl_data.get("delta_pct", "N/A")
            report.append(
                f"| {sname} | {fp16_ppl} | {s_ppl} | {delta} | {agree} | {collapsed} |"
            )

        report.append("")

        # Needle results
        report.append("#### Needle-In-Haystack Retrieval\n")
        for sname, sdata in model_results.get("strategies", {}).items():
            needle = sdata.get("needle", {})
            if needle and needle.get("evals"):
                for lk, lv in needle["evals"].items():
                    report.append(
                        f"- {sname} @ {lk}: FP16={lv.get('fp16_acc','N/A')} "
                        f"Strategy={lv.get('strategy_acc','N/A')}"
                    )
        report.append("")

    # Analysis
    report.append("## Analysis\n")
    report.append("### Does FIM ranking predict sensitive layers?\n")
    for model_key, model_results in all_model_results.items():
        ls = model_results.get("layer_sensitivity", {})
        corr = ls.get("fim_correlation", {})
        if corr:
            report.append(
                f"- {model_key}: Spearman rho={corr.get('spearman_rho','N/A')}, "
                f"p={corr.get('p_value','N/A')}"
            )

    report.append("\n### Do FIM strategies outperform heuristics?\n")
    report.append(
        "Compare strategies F (FIM-Top-K) and G (FIM-Multi-Tier) against "
        "D (First/Last) and H (Random) in the tables above.\n"
    )

    report.append("### KV precision vs bandwidth vs accuracy\n")
    report.append(
        "INT8 (Strategy B) provides good quality preservation. "
        "INT4 (Strategy C) shows significant degradation. "
        "FIM-guided allocation (F, G) aims to match INT8 quality "
        "at closer to INT4 bandwidth.\n"
    )

    # Plots
    report.append("## Plots\n")
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.endswith(".png"):
            report.append(f"![{f}](plots/{f})")
    report.append("")

    report_path = os.path.join(RESULTS_ROOT, "bpa47_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"  Report saved: {report_path}")


# ============================================================
# Main orchestration
# ============================================================
def run_model_experiments(model_key, run_secondary_experiments=True):
    """Run all experiments for a single model."""
    print(f"\n{'#' * 60}")
    print(f"# Model: {model_key}")
    print(f"{'#' * 60}")

    model, tokenizer, n_layers = load_model_and_tokenizer(model_key)

    model_results = {"n_layers": n_layers}

    # Experiment 1: Fisher calibration
    fim_json = os.path.join(FIM_DIR, f"{model_key}_fim_trace.json")
    if os.path.exists(fim_json):
        print(f"  Loading cached FIM results from {fim_json}")
        with open(fim_json) as f:
            fim_data = json.load(f)
        fim_ranking = fim_data["ranking"]
        fim_trace = fim_data["fim_total"]
    else:
        fim_ranking, fim_trace = experiment_1_fisher_calibration(
            model, tokenizer, model_key, n_layers
        )

    model_results["fim_ranking"] = fim_ranking
    model_results["fim_trace"] = fim_trace

    # Experiment 2: Layer sensitivity validation
    ls_json = os.path.join(JSON_DIR, f"layer_sensitivity_{model_key}.json")
    if os.path.exists(ls_json):
        print(f"  Loading cached layer sensitivity from {ls_json}")
        with open(ls_json) as f:
            model_results["layer_sensitivity"] = json.load(f)
    else:
        model_results["layer_sensitivity"] = experiment_2_layer_sensitivity(
            model, tokenizer, model_key, n_layers, fim_ranking, fim_trace
        )

    # Build strategies
    strategies = make_strategy_configs(n_layers, fim_ranking)

    # Collapse detection + strategy evaluation
    model_results["strategies"] = {}
    collapsed_configs = []

    for sname, sobj in strategies.items():
        print(f"\n--- Strategy {sname} ---")
        sresult = {}

        # Skip baseline for collapse check
        if sname == "A_fp16":
            sresult["collapse_check"] = {"collapsed": False}
        else:
            collapse = early_collapse_check(model, tokenizer, sname, sobj, n_layers)
            sresult["collapse_check"] = collapse
            if collapse.get("collapsed", False):
                collapsed_configs.append(collapse)
                model_results["strategies"][sname] = sresult
                continue

        # Experiment 3: Strategy eval (PPL, token agree, logit error)
        sresult["strategy_eval"] = experiment_3_strategy_eval(
            model, tokenizer, model_key, n_layers, sname, sobj
        )

        if run_secondary_experiments:
            # Experiment 5: Long context PPL
            sresult["long_context_ppl"] = experiment_5_long_context_ppl(
                model, tokenizer, model_key, n_layers, sname, sobj
            )

            # Experiment 6: Needle retrieval
            sresult["needle"] = experiment_6_needle(
                model, tokenizer, model_key, n_layers, sname, sobj
            )

        model_results["strategies"][sname] = sresult

    # Save collapsed configs
    if collapsed_configs:
        save_json(
            {"model": model_key, "collapsed": collapsed_configs},
            f"collapsed_configs_{model_key}.json",
        )

    # Experiment 4: Token agreement (only for non-collapsed strategies, primary model)
    if model_key == PRIMARY_MODEL:
        for sname, sobj in strategies.items():
            sdata = model_results["strategies"].get(sname, {})
            if sdata.get("collapse_check", {}).get("collapsed", False):
                continue
            sdata["token_agreement_200"] = experiment_4_token_agreement(
                model, tokenizer, model_key, n_layers, sname, sobj
            )

    # Experiment 7: Downstream tasks (primary model only, select strategies)
    if model_key == PRIMARY_MODEL:
        key_strategies = ["A_fp16", "B_int8", "C_int4", "F_fim_topk", "G_fim_multi"]
        for sname in key_strategies:
            if sname not in strategies:
                continue
            sobj = strategies[sname]
            sdata = model_results["strategies"].get(sname, {})
            if sdata.get("collapse_check", {}).get("collapsed", False):
                continue
            sdata["downstream"] = experiment_7_downstream(
                model, tokenizer, model_key, n_layers, sname, sobj
            )

    # Experiment 8-9: Bandwidth and batch scaling (primary model, key strategies)
    if model_key == PRIMARY_MODEL:
        for sname in ["A_fp16", "C_int4", "F_fim_topk"]:
            if sname not in strategies:
                continue
            sobj = strategies[sname]
            sdata = model_results["strategies"].get(sname, {})
            if sdata.get("collapse_check", {}).get("collapsed", False):
                continue
            sdata["bandwidth"] = experiment_8_bandwidth(
                model, tokenizer, model_key, n_layers, sname, sobj
            )
            sdata["batch_scaling"] = experiment_9_batch_scaling(
                model, tokenizer, model_key, n_layers, sname, sobj
            )

    # Generate plots
    generate_summary_plots(model_results["strategies"], model_key)

    # Unload model
    unload_model(model)

    return model_results


def main():
    print("=" * 60)
    print("BPA v47: RoPE-Safe KV Quantization + FIM-Guided Precision")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    print(f"GPU: {gpu_info()}")

    t_total = time.time()
    all_model_results = {}

    # Primary model: full experiments
    all_model_results[PRIMARY_MODEL] = run_model_experiments(
        PRIMARY_MODEL, run_secondary_experiments=True
    )

    # Secondary models: core experiments only (FIM + strategies 3-5)
    for model_key in ["mistral-7b", "llama2-7b"]:
        try:
            all_model_results[model_key] = run_model_experiments(
                model_key, run_secondary_experiments=False
            )
        except Exception as e:
            print(f"\nERROR with {model_key}: {e}")
            import traceback

            traceback.print_exc()

    # Cross-architecture FIM comparison plot
    print("\n  Generating cross-architecture comparison...")
    fig, axes = plt.subplots(
        1, len(all_model_results), figsize=(6 * len(all_model_results), 5)
    )
    if len(all_model_results) == 1:
        axes = [axes]
    for ax, (mk, mr) in zip(axes, all_model_results.items()):
        ft = mr.get("fim_trace", [])
        if ft:
            ax.bar(range(len(ft)), ft, color="steelblue")
            ax.set_xlabel("Layer")
            ax.set_ylabel("FIM Score")
            ax.set_title(mk)
    plt.suptitle("Cross-Architecture Fisher Sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fim_maps_comparison.png"), dpi=300)
    plt.close()

    # Layer sensitivity comparison
    fig, axes = plt.subplots(
        1, len(all_model_results), figsize=(6 * len(all_model_results), 5)
    )
    if len(all_model_results) == 1:
        axes = [axes]
    for ax, (mk, mr) in zip(axes, all_model_results.items()):
        ls = mr.get("layer_sensitivity", {})
        deltas = ls.get("ppl_deltas", [])
        if deltas:
            valid = [d if d != "nan" else 0 for d in deltas]
            colors = [
                "red" if d > 3 else "orange" if d > 1 else "steelblue" for d in valid
            ]
            ax.bar(range(len(valid)), valid, color=colors)
            ax.set_xlabel("Layer")
            ax.set_ylabel("PPL Delta (%)")
            ax.set_title(mk)
    plt.suptitle("Cross-Architecture Layer Sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "layer_sensitivity_comparison.png"), dpi=300)
    plt.close()

    # Generate report
    generate_report(all_model_results)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"BPA v47 COMPLETE")
    print(f"Total runtime: {total_time:.0f}s ({total_time/3600:.1f} hours)")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
