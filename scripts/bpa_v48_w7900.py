#!/usr/bin/env python3
"""BPA v48: Cross-Layer Fisher Interaction Analysis for KV Quantization.

Extends v47 with:
  1. Cross-layer gradient covariance (interaction matrix)
  2. Interaction-guided precision allocation
  3. K vs V separate Fisher sensitivity
  4. K/V asymmetric quantization sweep
  5. Architecture interaction comparison

Results saved to /data/knlp-key-results/bpa48/
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
RESULTS_ROOT = "/data/knlp-key-results/bpa48"
JSON_DIR = os.path.join(RESULTS_ROOT, "json")
PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
FIM_DIR = os.path.join(RESULTS_ROOT, "fim_maps")
INTERACTION_DIR = os.path.join(RESULTS_ROOT, "interaction_maps")
for d in [JSON_DIR, PLOT_DIR, LOG_DIR, FIM_DIR, INTERACTION_DIR]:
    os.makedirs(d, exist_ok=True)

# === Protocol parameters ===
DATASET = "wikitext-103-raw-v1"
N_TOKENS = 500_000
W_SINK = 4
GROUP_SIZE = 32
DEVICE = "cuda"
DTYPE = torch.bfloat16

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}
PRIMARY_MODEL = "qwen2.5-7b"

# Calibration parameters
CAL_SEQ_LEN = 1024
N_CAL_SEQS = 200  # reduced from 2000 to fit in memory with gradients


def save_json(data, filename, subdir=JSON_DIR):
    path = os.path.join(subdir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def load_json(filename, subdir=JSON_DIR):
    path = os.path.join(subdir, filename)
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


def quantize_tensor(tensor, quant_type, group_size=32):
    if quant_type == "int4":
        return quantize_int4_grouped(tensor, group_size)
    elif quant_type == "int8":
        return quantize_int8_grouped(tensor, group_size)
    return tensor


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


def quantize_cache_per_layer(past, layer_quant_map, group_size=32):
    """Quantize cache with per-layer K and V precision.

    layer_quant_map: dict of layer_idx -> (k_type, v_type)
    where types are None, 'int4', 'int8'
    """
    D = n_cache_layers(past)
    for li in range(D):
        k_type, v_type = layer_quant_map.get(li, (None, None))
        if k_type is None and v_type is None:
            continue
        k, v = _cache_get_kv(past, li)
        k_sink = k[:, :, :W_SINK, :]
        v_sink = v[:, :, :W_SINK, :]
        k_far = k[:, :, W_SINK:, :]
        v_far = v[:, :, W_SINK:, :]
        if k_type is not None:
            k_far = quantize_tensor(k_far, k_type, group_size)
        if v_type is not None:
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


def get_attn_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find model layers")


def get_attn_module(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attn"):
        return layer.attn
    raise RuntimeError("Cannot find attention module")


# ============================================================
# Experiments 1-2: Gradient Capture + Diagonal Fisher
# ============================================================
def experiment_1_2_gradient_capture(model, tokenizer, model_key, n_layers):
    """Capture KV gradients and compute diagonal Fisher map."""
    print("\n" + "=" * 60)
    print(f"Experiments 1-2: Gradient Capture + Diagonal Fisher ({model_key})")
    print("=" * 60)
    t0 = time.time()

    cached = load_json(f"{model_key}_diag_fisher.json", FIM_DIR)
    if cached is not None:
        print("  Loaded cached results")
        return cached

    tokens = load_wikitext_tokens(tokenizer)
    layers = get_attn_layers(model)
    rng = np.random.RandomState(0)

    # Accumulators for diagonal Fisher
    fim_k = torch.zeros(n_layers, dtype=torch.float64)
    fim_v = torch.zeros(n_layers, dtype=torch.float64)

    # Accumulators for cross-layer covariance (experiment 3)
    # Store flattened gradient norms per layer per sample for later
    # covariance computation
    k_grad_norms = []  # list of [n_layers] tensors
    v_grad_norms = []
    n_counted = 0

    print(f"  Calibrating on {N_CAL_SEQS} sequences (len={CAL_SEQ_LEN})...")

    for seq_idx in range(N_CAL_SEQS):
        if seq_idx % 50 == 0:
            print(f"    Sequence {seq_idx}/{N_CAL_SEQS}...")

        start = rng.randint(0, max(1, len(tokens) - CAL_SEQ_LEN - 1))
        input_ids = (
            torch.tensor(tokens[start : start + CAL_SEQ_LEN], dtype=torch.long)
            .unsqueeze(0)
            .to(DEVICE)
        )

        model.zero_grad()

        # Hook to capture K and V projection outputs with retain_grad
        kv_activations = {}

        def make_hook(layer_idx, kv_type):
            def hook_fn(module, input, output):
                output.retain_grad()
                kv_activations[(layer_idx, kv_type)] = output

            return hook_fn

        handles = []
        for li, layer in enumerate(layers):
            attn = get_attn_module(layer)
            handles.append(attn.k_proj.register_forward_hook(make_hook(li, "k")))
            handles.append(attn.v_proj.register_forward_hook(make_hook(li, "v")))

        try:
            with torch.enable_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                loss.backward()

                k_norms = torch.zeros(n_layers, dtype=torch.float64)
                v_norms = torch.zeros(n_layers, dtype=torch.float64)

                for li in range(n_layers):
                    k_act = kv_activations.get((li, "k"))
                    v_act = kv_activations.get((li, "v"))

                    if k_act is not None and k_act.grad is not None:
                        g = k_act.grad.float()
                        fim_k[li] += g.pow(2).mean().item()
                        k_norms[li] = g.pow(2).sum().item()
                    else:
                        # Fallback to parameter gradients
                        attn = get_attn_module(layers[li])
                        if attn.k_proj.weight.grad is not None:
                            g = attn.k_proj.weight.grad.float()
                            fim_k[li] += g.pow(2).mean().item()
                            k_norms[li] = g.pow(2).sum().item()

                    if v_act is not None and v_act.grad is not None:
                        g = v_act.grad.float()
                        fim_v[li] += g.pow(2).mean().item()
                        v_norms[li] = g.pow(2).sum().item()
                    else:
                        attn = get_attn_module(layers[li])
                        if attn.v_proj.weight.grad is not None:
                            g = attn.v_proj.weight.grad.float()
                            fim_v[li] += g.pow(2).mean().item()
                            v_norms[li] = g.pow(2).sum().item()

                k_grad_norms.append(k_norms)
                v_grad_norms.append(v_norms)
                n_counted += 1

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at seq {seq_idx}, stopping early")
            clear_gpu()
            break
        except Exception as e:
            print(f"    Error at seq {seq_idx}: {e}")
        finally:
            for h in handles:
                h.remove()
            kv_activations.clear()
            model.zero_grad(set_to_none=True)
            clear_gpu()

    if n_counted == 0:
        print("  WARNING: No successful calibration sequences")
        return None

    fim_k /= n_counted
    fim_v /= n_counted
    fim_total = fim_k + fim_v
    ranking = torch.argsort(fim_total, descending=True).tolist()

    results = {
        "model": model_key,
        "n_cal_seqs": n_counted,
        "cal_seq_len": CAL_SEQ_LEN,
        "fim_k": fim_k.tolist(),
        "fim_v": fim_v.tolist(),
        "fim_total": fim_total.tolist(),
        "ranking": ranking,
        "k_grad_norms": [kn.tolist() for kn in k_grad_norms],
        "v_grad_norms": [vn.tolist() for vn in v_grad_norms],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    save_json(results, f"{model_key}_diag_fisher.json", FIM_DIR)

    # Plot diagonal Fisher
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(n_layers))
    ax.bar(x, fim_total.tolist(), color="steelblue", alpha=0.7, label="Total")
    ax.bar(x, fim_k.tolist(), color="coral", alpha=0.5, label="K")
    ax.bar(x, fim_v.tolist(), color="forestgreen", alpha=0.5, label="V")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fisher Score")
    ax.set_title(f"Diagonal Fisher Trace ({model_key})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"layer_vs_fisher_trace_{model_key}.png"), dpi=300
    )
    plt.close()

    print(f"  FIM ranking (top 5): {ranking[:5]}")
    print(f"  Elapsed: {results['elapsed_sec']}s")
    return results


# ============================================================
# Experiment 3: Cross-Layer Fisher Covariance
# ============================================================
def experiment_3_cross_layer_covariance(fisher_data, model_key, n_layers):
    """Compute cross-layer gradient covariance matrix."""
    print("\n" + "=" * 60)
    print(f"Experiment 3: Cross-Layer Fisher Covariance ({model_key})")
    print("=" * 60)
    t0 = time.time()

    cached = load_json(f"{model_key}_gradient_covariance.json", INTERACTION_DIR)
    if cached is not None:
        print("  Loaded cached results")
        return cached

    k_norms = np.array(fisher_data["k_grad_norms"])  # [n_samples, n_layers]
    v_norms = np.array(fisher_data["v_grad_norms"])
    combined = k_norms + v_norms  # [n_samples, n_layers]

    n_samples = combined.shape[0]
    print(f"  Computing {n_layers}x{n_layers} covariance from {n_samples} samples...")

    # Compute covariance: C[i,j] = mean(g_i * g_j)
    # Using norm products as proxy for dot product of gradient vectors
    C = np.zeros((n_layers, n_layers))
    for s in range(n_samples):
        g = combined[s]  # [n_layers]
        C += np.outer(g, g)
    C /= n_samples

    # Normalized covariance
    diag = np.diag(C)
    C_norm = np.zeros_like(C)
    for i in range(n_layers):
        for j in range(n_layers):
            denom = np.sqrt(max(diag[i] * diag[j], 1e-30))
            C_norm[i, j] = C[i, j] / denom

    # K-K, V-V, K-V covariance matrices
    C_KK = np.zeros((n_layers, n_layers))
    C_VV = np.zeros((n_layers, n_layers))
    C_KV = np.zeros((n_layers, n_layers))
    for s in range(n_samples):
        gk = k_norms[s]
        gv = v_norms[s]
        C_KK += np.outer(gk, gk)
        C_VV += np.outer(gv, gv)
        C_KV += np.outer(gk, gv)
    C_KK /= n_samples
    C_VV /= n_samples
    C_KV /= n_samples

    results = {
        "model": model_key,
        "n_layers": n_layers,
        "n_samples": n_samples,
        "covariance": C.tolist(),
        "covariance_normalized": C_norm.tolist(),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    save_json(results, f"{model_key}_gradient_covariance.json", INTERACTION_DIR)
    save_json(
        {"model": model_key, "C_KK": C_KK.tolist()},
        f"{model_key}_KK_covariance.json",
        INTERACTION_DIR,
    )
    save_json(
        {"model": model_key, "C_VV": C_VV.tolist()},
        f"{model_key}_VV_covariance.json",
        INTERACTION_DIR,
    )
    save_json(
        {"model": model_key, "C_KV": C_KV.tolist()},
        f"{model_key}_KV_covariance.json",
        INTERACTION_DIR,
    )

    return results


# ============================================================
# Experiment 4: Interaction Visualization
# ============================================================
def experiment_4_visualization(cov_data, model_key, n_layers):
    """Generate interaction heatmaps."""
    print(f"\n  Generating interaction heatmaps for {model_key}...")

    C = np.array(cov_data["covariance"])
    C_norm = np.array(cov_data["covariance_normalized"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(C, aspect="auto", cmap="viridis")
    ax1.set_xlabel("Layer j")
    ax1.set_ylabel("Layer i")
    ax1.set_title(f"Gradient Covariance ({model_key})")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(C_norm, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xlabel("Layer j")
    ax2.set_ylabel("Layer i")
    ax2.set_title(f"Normalized Covariance ({model_key})")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"layer_interaction_heatmap_{model_key}.png"), dpi=300
    )
    plt.close()

    # K-K, V-V, K-V heatmaps
    for mat_name in ["KK", "VV", "KV"]:
        mat_data = load_json(f"{model_key}_{mat_name}_covariance.json", INTERACTION_DIR)
        if mat_data is None:
            continue
        M = np.array(mat_data[f"C_{mat_name}"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xlabel("Layer j")
        ax.set_ylabel("Layer i")
        ax.set_title(f"{mat_name} Covariance ({model_key})")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"interaction_{mat_name}_heatmap_{model_key}.png"),
            dpi=300,
        )
        plt.close()


# ============================================================
# Experiment 5: Interaction Score
# ============================================================
def experiment_5_interaction_score(cov_data, model_key, n_layers):
    """Compute per-layer interaction score."""
    print(f"\n  Computing interaction scores for {model_key}...")

    C = np.array(cov_data["covariance"])
    C_norm = np.array(cov_data["covariance_normalized"])

    # Interaction score: sum of absolute off-diagonal elements
    interaction_scores = []
    for l in range(n_layers):
        score = np.sum(np.abs(C_norm[l, :])) - np.abs(C_norm[l, l])
        interaction_scores.append(float(score))

    interaction_ranking = np.argsort(interaction_scores)[::-1].tolist()

    results = {
        "model": model_key,
        "interaction_scores": interaction_scores,
        "interaction_ranking": interaction_ranking,
        "mean_interaction": float(np.mean(interaction_scores)),
        "off_diagonal_energy": float(
            np.sum(np.abs(C_norm)) - np.sum(np.abs(np.diag(C_norm)))
        ),
    }

    save_json(results, f"{model_key}_interaction_scores.json", INTERACTION_DIR)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(n_layers))
    ax.bar(x, interaction_scores, color="darkorange")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Interaction Score")
    ax.set_title(f"Cross-Layer Interaction Score ({model_key})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"layer_vs_interaction_score_{model_key}.png"), dpi=300
    )
    plt.close()

    print(f"  Interaction ranking (top 5): {interaction_ranking[:5]}")
    return results


# ============================================================
# Experiment 6: Correlation with Quantization Sensitivity
# ============================================================
def experiment_6_quant_sensitivity_correlation(
    model, tokenizer, model_key, n_layers, fisher_data, interaction_data
):
    """Single-layer INT4 quantization and correlation with Fisher/interaction."""
    print("\n" + "=" * 60)
    print(f"Experiment 6: Quantization Sensitivity Correlation ({model_key})")
    print("=" * 60)
    t0 = time.time()

    cached = load_json(f"quant_sensitivity_{model_key}.json")
    if cached is not None:
        print("  Loaded cached results")
        return cached

    L = 2048
    seeds = [0, 1, 2]
    decode_tokens = 64

    # Baseline PPL
    print("  Computing baseline PPL...")
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
        _, _, V_dim = logits[:, :-1, :].shape
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V_dim).float(), targets)
        baseline_ppls.append(math.exp(min(loss.item(), 20)))
        del past, out, logits
        clear_gpu()
    baseline_ppl = np.mean(baseline_ppls)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Per-layer quantization: full KV, K-only, V-only
    ppl_deltas_kv = []
    ppl_deltas_k = []
    ppl_deltas_v = []

    for li in range(n_layers):
        if li % 7 == 0:
            print(f"  Layer {li}/{n_layers}...")

        for quant_mode, delta_list in [
            ("kv", ppl_deltas_kv),
            ("k_only", ppl_deltas_k),
            ("v_only", ppl_deltas_v),
        ]:
            layer_ppls = []
            for seed in seeds:
                try:
                    passage = load_passage(tokenizer, L, seed, extra=decode_tokens)
                    input_ids = passage[:, :L].to(DEVICE)
                    continuation = passage[:, L : L + decode_tokens].to(DEVICE)

                    with torch.no_grad():
                        out = model(input_ids, use_cache=True)
                        past = out.past_key_values

                    # Build quant map
                    qmap = {}
                    if quant_mode == "kv":
                        qmap[li] = ("int4", "int4")
                    elif quant_mode == "k_only":
                        qmap[li] = ("int4", None)
                    elif quant_mode == "v_only":
                        qmap[li] = (None, "int4")
                    past = quantize_cache_per_layer(past, qmap, GROUP_SIZE)

                    logits_q = [out.logits[:, -1:, :].cpu()]
                    for t in range(decode_tokens):
                        tok = continuation[:, t : t + 1]
                        with torch.no_grad():
                            out2 = model(tok, past_key_values=past, use_cache=True)
                        logits_q.append(out2.logits.cpu())
                        past = out2.past_key_values
                    logits_q = torch.cat(logits_q, dim=1)

                    targets = continuation.cpu().reshape(-1)
                    _, _, V_dim = logits_q[:, :-1, :].shape
                    loss = F.cross_entropy(
                        logits_q[:, :-1, :].reshape(-1, V_dim).float(), targets
                    )
                    layer_ppls.append(math.exp(min(loss.item(), 20)))
                    del past, out, logits_q
                    clear_gpu()
                except torch.cuda.OutOfMemoryError:
                    clear_gpu()
                    continue

            if layer_ppls:
                delta = (np.mean(layer_ppls) - baseline_ppl) / baseline_ppl * 100
            else:
                delta = float("nan")
            delta_list.append(delta)

    # Compute correlations
    from scipy.stats import spearmanr

    fim_total = fisher_data["fim_total"]
    fim_k = fisher_data["fim_k"]
    fim_v = fisher_data["fim_v"]
    interaction_scores = interaction_data["interaction_scores"]

    correlations = {}
    for signal_name, signal, delta_name, deltas in [
        ("diagonal_fisher", fim_total, "ppl_delta_kv", ppl_deltas_kv),
        ("interaction_score", interaction_scores, "ppl_delta_kv", ppl_deltas_kv),
        ("fisher_k", fim_k, "ppl_delta_k", ppl_deltas_k),
        ("fisher_v", fim_v, "ppl_delta_v", ppl_deltas_v),
    ]:
        valid = [(s, d) for s, d in zip(signal, deltas) if not math.isnan(d) and d != 0]
        if len(valid) > 3:
            ss, dd = zip(*valid)
            rho, pval = spearmanr(ss, dd)
            correlations[f"{signal_name}_vs_{delta_name}"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(pval, 6),
                "n": len(valid),
            }
            print(f"  {signal_name} vs {delta_name}: rho={rho:.4f}, p={pval:.6f}")

    results = {
        "model": model_key,
        "baseline_ppl": round(baseline_ppl, 4),
        "ppl_deltas_kv": [
            round(d, 4) if not math.isnan(d) else "nan" for d in ppl_deltas_kv
        ],
        "ppl_deltas_k": [
            round(d, 4) if not math.isnan(d) else "nan" for d in ppl_deltas_k
        ],
        "ppl_deltas_v": [
            round(d, 4) if not math.isnan(d) else "nan" for d in ppl_deltas_v
        ],
        "correlations": correlations,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    save_json(results, f"quant_sensitivity_{model_key}.json")

    # Plot: interaction vs fisher scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    valid_kv = [d for d in ppl_deltas_kv if not math.isnan(d)]
    if len(valid_kv) == n_layers:
        axes[0].scatter(fim_total, valid_kv, alpha=0.7, color="steelblue")
        axes[0].set_xlabel("Diagonal Fisher Score")
        axes[0].set_ylabel("PPL Delta (%)")
        axes[0].set_title("Fisher vs Sensitivity")

        axes[1].scatter(interaction_scores, valid_kv, alpha=0.7, color="darkorange")
        axes[1].set_xlabel("Interaction Score")
        axes[1].set_ylabel("PPL Delta (%)")
        axes[1].set_title("Interaction vs Sensitivity")

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"interaction_vs_fisher_scatter_{model_key}.png"),
        dpi=300,
    )
    plt.close()

    # K vs V sensitivity plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_layers)
    width = 0.35
    valid_k = [d if not math.isnan(d) else 0 for d in ppl_deltas_k]
    valid_v = [d if not math.isnan(d) else 0 for d in ppl_deltas_v]
    ax.bar(x - width / 2, valid_k, width, label="K-only INT4", color="coral")
    ax.bar(x + width / 2, valid_v, width, label="V-only INT4", color="forestgreen")
    ax.set_xlabel("Layer")
    ax.set_ylabel("PPL Delta (%)")
    ax.set_title(f"K vs V Quantization Sensitivity ({model_key})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"layer_vs_K_V_sensitivity_{model_key}.png"), dpi=300
    )
    plt.close()

    return results


# ============================================================
# Experiment 7: Interaction-Guided Precision Allocation
# ============================================================
def experiment_7_interaction_strategies(
    model, tokenizer, model_key, n_layers, interaction_data, cov_data
):
    """Test interaction-guided precision allocation strategies."""
    print("\n" + "=" * 60)
    print(f"Experiment 7: Interaction-Guided Strategies ({model_key})")
    print("=" * 60)
    t0 = time.time()

    ranking = interaction_data["interaction_ranking"]
    n_top20 = max(1, n_layers // 5)
    n_mid40 = max(1, int(n_layers * 0.4))

    strategies = {}

    # Strategy A: Interaction Top Protection
    qmap_a = {}
    for li in range(n_layers):
        if li in ranking[:n_top20]:
            qmap_a[li] = (None, None)  # FP16
        else:
            qmap_a[li] = ("int4", "int4")
    strategies["InterTop"] = qmap_a

    # Strategy B: Interaction Multi-Tier
    qmap_b = {}
    for li in range(n_layers):
        if li in ranking[:n_top20]:
            qmap_b[li] = (None, None)
        elif li in ranking[n_top20 : n_top20 + n_mid40]:
            qmap_b[li] = ("int8", "int8")
        else:
            qmap_b[li] = ("int4", "int4")
    strategies["InterMulti"] = qmap_b

    # Strategy C: Spectral Clustering
    try:
        from sklearn.cluster import SpectralClustering

        C_norm = np.array(cov_data["covariance_normalized"])
        # Make affinity matrix (similarity)
        affinity = np.abs(C_norm)
        np.fill_diagonal(affinity, 1.0)
        sc = SpectralClustering(n_clusters=3, affinity="precomputed", random_state=42)
        labels = sc.fit_predict(affinity)

        # Assign clusters: highest mean interaction -> FP16, lowest -> INT4
        cluster_means = {}
        interaction_scores = interaction_data["interaction_scores"]
        for c in range(3):
            members = [i for i in range(n_layers) if labels[i] == c]
            cluster_means[c] = np.mean([interaction_scores[i] for i in members])

        sorted_clusters = sorted(
            cluster_means.keys(), key=lambda c: cluster_means[c], reverse=True
        )
        cluster_precision = {
            sorted_clusters[0]: (None, None),  # FP16
            sorted_clusters[1]: ("int8", "int8"),
            sorted_clusters[2]: ("int4", "int4"),
        }

        qmap_c = {}
        for li in range(n_layers):
            qmap_c[li] = cluster_precision[labels[li]]
        strategies["InterCluster"] = qmap_c

        # Save cluster visualization
        fig, ax = plt.subplots(figsize=(12, 3))
        colors_map = {
            sorted_clusters[0]: "red",
            sorted_clusters[1]: "orange",
            sorted_clusters[2]: "green",
        }
        bar_colors = [colors_map[labels[i]] for i in range(n_layers)]
        ax.bar(range(n_layers), interaction_scores, color=bar_colors)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Interaction Score")
        ax.set_title(
            f"Spectral Clustering ({model_key}): Red=FP16, Orange=INT8, Green=INT4"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"interaction_cluster_map_{model_key}.png"), dpi=300
        )
        plt.close()

    except ImportError:
        print("  sklearn not available, skipping spectral clustering")
    except Exception as e:
        print(f"  Spectral clustering failed: {e}")

    # Also add v47-style baselines for comparison
    # FP16 baseline
    strategies["FP16"] = {li: (None, None) for li in range(n_layers)}
    # Uniform INT8
    strategies["INT8"] = {li: ("int8", "int8") for li in range(n_layers)}
    # Uniform INT4
    strategies["INT4"] = {li: ("int4", "int4") for li in range(n_layers)}
    # v47 G_fim_multi (using Fisher ranking from exp 1-2)
    fisher_ranking = interaction_data.get("fisher_ranking", list(range(n_layers)))

    qmap_fim = {}
    for li in range(n_layers):
        if li in fisher_ranking[:n_top20]:
            qmap_fim[li] = (None, None)
        elif li in fisher_ranking[n_top20 : n_top20 + n_mid40]:
            qmap_fim[li] = ("int8", "int8")
        else:
            qmap_fim[li] = ("int4", "int4")
    strategies["FIM_Multi"] = qmap_fim

    # Evaluate each strategy
    L = 2048
    seeds = [0, 1, 2]
    decode_tokens = 64
    results = {"model": model_key, "strategies": {}}

    for sname, qmap in strategies.items():
        print(f"\n  Evaluating strategy: {sname}")
        ppls_fp = []
        ppls_q = []
        agrees = []

        for seed in seeds:
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
                past_q = quantize_cache_per_layer(past_q, qmap, GROUP_SIZE)
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

                pred_fp = logits_fp[:, :-1, :].argmax(dim=-1)
                pred_q = logits_q[:, :-1, :].argmax(dim=-1)
                agrees.append((pred_fp == pred_q).float().mean().item())

                del logits_fp, logits_q
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at seed {seed}")
                clear_gpu()
                continue

        if ppls_fp:
            results["strategies"][sname] = {
                "ppl_fp16": round(np.mean(ppls_fp), 4),
                "ppl_strategy": round(np.mean(ppls_q), 4),
                "delta_pct": round(
                    (np.mean(ppls_q) - np.mean(ppls_fp)) / np.mean(ppls_fp) * 100, 2
                ),
                "token_agreement": round(np.mean(agrees), 4),
            }
            print(
                f"    PPL delta={results['strategies'][sname]['delta_pct']:.2f}% "
                f"agree={results['strategies'][sname]['token_agreement']:.4f}"
            )

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"strategy_eval_{model_key}.json")
    return results


# ============================================================
# Experiment 10: K vs V Precision Sweep
# ============================================================
def experiment_10_kv_asymmetry(model, tokenizer, model_key, n_layers):
    """Test asymmetric K/V precision configurations."""
    print("\n" + "=" * 60)
    print(f"Experiment 10: K vs V Precision Sweep ({model_key})")
    print("=" * 60)
    t0 = time.time()

    cached = load_json(f"kv_asymmetry_{model_key}.json")
    if cached is not None:
        print("  Loaded cached results")
        return cached

    configs = {
        "baseline": (None, None),
        "KV-1_KFP16_VINT4": (None, "int4"),
        "KV-2_KINT8_VINT4": ("int8", "int4"),
        "KV-3_KINT4_VINT8": ("int4", "int8"),
        "KV-4_KINT4_VINT4": ("int4", "int4"),
        "KV-5_KINT8_VINT8": ("int8", "int8"),
    }

    L = 2048
    seeds = [0, 1, 2]
    decode_tokens = 64
    results = {"model": model_key, "configs": {}}

    for cname, (k_type, v_type) in configs.items():
        print(f"\n  Config: {cname}")
        ppls_fp = []
        ppls_q = []
        agrees = []
        logit_errs = []

        for seed in seeds:
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
                qmap = {li: (k_type, v_type) for li in range(n_layers)}
                with torch.no_grad():
                    out_q = model(input_ids, use_cache=True)
                    past_q = out_q.past_key_values
                past_q = quantize_cache_per_layer(past_q, qmap, GROUP_SIZE)
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

                pred_fp = logits_fp[:, :-1, :].argmax(dim=-1)
                pred_q = logits_q[:, :-1, :].argmax(dim=-1)
                agrees.append((pred_fp == pred_q).float().mean().item())

                diff = (logits_fp[:, :-1, :] - logits_q[:, :-1, :]).abs()
                logit_errs.append(diff.max(dim=-1).values.float().mean().item())

                del logits_fp, logits_q
                clear_gpu()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at seed {seed}")
                clear_gpu()
                continue

        if ppls_fp:
            results["configs"][cname] = {
                "k_type": k_type,
                "v_type": v_type,
                "ppl_fp16": round(np.mean(ppls_fp), 4),
                "ppl_strategy": round(np.mean(ppls_q), 4),
                "delta_pct": round(
                    (np.mean(ppls_q) - np.mean(ppls_fp)) / np.mean(ppls_fp) * 100, 2
                ),
                "token_agreement": round(np.mean(agrees), 4),
                "mean_logit_error": round(np.mean(logit_errs), 4),
            }
            print(
                f"    delta={results['configs'][cname]['delta_pct']:.2f}% "
                f"agree={results['configs'][cname]['token_agreement']:.4f}"
            )

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, f"kv_asymmetry_{model_key}.json")
    return results


# ============================================================
# Experiment 11: K vs V Fisher Sensitivity
# ============================================================
def experiment_11_kv_fisher(fisher_data, model_key, n_layers):
    """Compute K/V Fisher ratio per layer."""
    print(f"\n  Computing K/V Fisher sensitivity for {model_key}...")

    fim_k = np.array(fisher_data["fim_k"])
    fim_v = np.array(fisher_data["fim_v"])

    kv_ratio = np.where(fim_v > 1e-30, fim_k / fim_v, 0.0)

    results = {
        "model": model_key,
        "fim_k": fim_k.tolist(),
        "fim_v": fim_v.tolist(),
        "kv_ratio": kv_ratio.tolist(),
        "mean_kv_ratio": float(np.mean(kv_ratio)),
    }

    save_json(results, f"{model_key}_key_value_sensitivity.json", FIM_DIR)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = list(range(n_layers))

    axes[0].bar(x, fim_k.tolist(), color="coral")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Fisher K")
    axes[0].set_title(f"Key Sensitivity ({model_key})")

    axes[1].bar(x, fim_v.tolist(), color="forestgreen")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Fisher V")
    axes[1].set_title(f"Value Sensitivity ({model_key})")

    axes[2].bar(x, kv_ratio.tolist(), color="purple")
    axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("K/V Ratio")
    axes[2].set_title(f"K/V Sensitivity Ratio ({model_key})")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"layer_vs_KV_ratio_{model_key}.png"), dpi=300)
    plt.close()

    return results


# ============================================================
# Experiment 14: KV Bandwidth Reduction Analysis
# ============================================================
def experiment_14_bandwidth_analysis(model_key, n_layers, kv_asym_results):
    """Compute bandwidth savings for each KV configuration."""
    print(f"\n  Computing KV bandwidth savings for {model_key}...")

    dtype_bytes = {"fp16": 2.0, "int8": 1.0, "int4": 0.5, None: 2.0}

    configs = {
        "baseline": (None, None),
        "KV-1_KFP16_VINT4": (None, "int4"),
        "KV-2_KINT8_VINT4": ("int8", "int4"),
        "KV-3_KINT4_VINT8": ("int4", "int8"),
        "KV-4_KINT4_VINT4": ("int4", "int4"),
        "KV-5_KINT8_VINT8": ("int8", "int8"),
    }

    baseline_bytes = 2.0 + 2.0  # K FP16 + V FP16 = 4 bytes per element
    results = {"model": model_key, "configs": {}}

    for cname, (k_type, v_type) in configs.items():
        k_bytes = dtype_bytes.get(k_type, 2.0)
        v_bytes = dtype_bytes.get(v_type, 2.0)
        total_bytes = k_bytes + v_bytes
        reduction = 1.0 - total_bytes / baseline_bytes

        quality = kv_asym_results.get("configs", {}).get(cname, {})

        results["configs"][cname] = {
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "total_bytes": total_bytes,
            "bandwidth_reduction": round(reduction, 4),
            "ppl_delta_pct": quality.get("delta_pct", "N/A"),
            "token_agreement": quality.get("token_agreement", "N/A"),
        }

    save_json(results, "kv_bandwidth_savings.json")

    # Plot: quality vs bandwidth tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    for cname, data in results["configs"].items():
        if cname == "baseline":
            continue
        bw_red = data["bandwidth_reduction"] * 100
        ppl_d = data.get("ppl_delta_pct", 0)
        if ppl_d == "N/A":
            continue
        ax.scatter(bw_red, ppl_d, s=100, zorder=5)
        ax.annotate(cname, (bw_red, ppl_d), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Bandwidth Reduction (%)")
    ax.set_ylabel("PPL Delta (%)")
    ax.set_title(f"Quality vs Bandwidth Tradeoff ({model_key})")
    ax.axhline(y=3, color="red", linestyle="--", alpha=0.5, label="3% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"quality_vs_bandwidth_tradeoff_{model_key}.png"),
        dpi=300,
    )
    plt.close()

    return results


# ============================================================
# Experiment 9/15: Architecture Comparison
# ============================================================
def experiment_9_15_architecture_comparison(all_model_data):
    """Compare interaction patterns across architectures."""
    print("\n" + "=" * 60)
    print("Experiments 9/15: Architecture Comparison")
    print("=" * 60)

    results = {"models": {}}

    for mk, data in all_model_data.items():
        interaction = data.get("interaction_scores", {})
        fisher = data.get("fisher_data", {})
        kv_fisher = data.get("kv_fisher", {})

        results["models"][mk] = {
            "mean_interaction": interaction.get("mean_interaction", "N/A"),
            "off_diagonal_energy": interaction.get("off_diagonal_energy", "N/A"),
            "mean_K_sensitivity": (
                float(np.mean(fisher.get("fim_k", [0]))) if fisher else "N/A"
            ),
            "mean_V_sensitivity": (
                float(np.mean(fisher.get("fim_v", [0]))) if fisher else "N/A"
            ),
            "mean_KV_ratio": kv_fisher.get("mean_kv_ratio", "N/A"),
        }

    save_json(results, "interaction_architecture_comparison.json")
    save_json(results, "kv_architecture_comparison.json")

    # Comparison plot
    models = list(results["models"].keys())
    if len(models) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Interaction strength comparison
        for mk in models:
            data = all_model_data[mk]
            scores = data.get("interaction_scores", {}).get("interaction_scores", [])
            if scores:
                axes[0].plot(scores, label=mk, alpha=0.8)
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Interaction Score")
        axes[0].set_title("Cross-Architecture Interaction")
        axes[0].legend()

        # K/V ratio comparison
        for mk in models:
            kv = data.get("kv_fisher", {})
            ratios = kv.get("kv_ratio", [])
            if ratios:
                axes[1].plot(ratios, label=mk, alpha=0.8)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("K/V Ratio")
        axes[1].set_title("K/V Sensitivity Ratio")
        axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[1].legend()

        # Off-diagonal energy bar
        energies = [
            results["models"][mk].get("off_diagonal_energy", 0) for mk in models
        ]
        axes[2].bar(models, energies, color=["steelblue", "coral"])
        axes[2].set_ylabel("Off-Diagonal Energy")
        axes[2].set_title("Interaction Strength")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, "kv_sensitivity_architecture_comparison.png"),
            dpi=300,
        )
        plt.close()

    return results


# ============================================================
# Final Report
# ============================================================
def generate_report(all_model_data):
    print("\n" + "=" * 60)
    print("Generating Final Report")
    print("=" * 60)

    report = []
    report.append("# BPA v48 — Cross-Layer Fisher Interaction Analysis\n")
    report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("**GPU**: AMD Radeon Pro W7900 (48GB)")
    report.append("**Dtype**: BF16\n")

    # Key findings
    report.append("## Key Findings\n")

    for mk, data in all_model_data.items():
        corr = data.get("sensitivity", {}).get("correlations", {})
        report.append(f"### {mk}\n")

        # Diagonal Fisher vs Interaction
        diag = corr.get("diagonal_fisher_vs_ppl_delta_kv", {})
        inter = corr.get("interaction_score_vs_ppl_delta_kv", {})
        report.append(
            f"- Diagonal Fisher vs PPL delta: rho={diag.get('spearman_rho', 'N/A')}, p={diag.get('p_value', 'N/A')}"
        )
        report.append(
            f"- Interaction score vs PPL delta: rho={inter.get('spearman_rho', 'N/A')}, p={inter.get('p_value', 'N/A')}"
        )

        # K vs V
        fk = corr.get("fisher_k_vs_ppl_delta_k", {})
        fv = corr.get("fisher_v_vs_ppl_delta_v", {})
        report.append(
            f"- Fisher K vs K-only PPL delta: rho={fk.get('spearman_rho', 'N/A')}"
        )
        report.append(
            f"- Fisher V vs V-only PPL delta: rho={fv.get('spearman_rho', 'N/A')}"
        )
        report.append("")

    # Strategy comparison
    report.append("## Strategy Comparison\n")
    for mk, data in all_model_data.items():
        strats = data.get("strategies", {}).get("strategies", {})
        if strats:
            report.append(f"### {mk}\n")
            report.append(
                "| Strategy | PPL FP16 | PPL Strategy | Delta (%) | Token Agree |"
            )
            report.append(
                "|----------|----------|-------------|-----------|-------------|"
            )
            for sname, sdata in strats.items():
                report.append(
                    f"| {sname} | {sdata.get('ppl_fp16', 'N/A')} | "
                    f"{sdata.get('ppl_strategy', 'N/A')} | "
                    f"{sdata.get('delta_pct', 'N/A')} | "
                    f"{sdata.get('token_agreement', 'N/A')} |"
                )
            report.append("")

    # KV Asymmetry
    report.append("## Key vs Value Precision Asymmetry\n")
    for mk, data in all_model_data.items():
        kv = data.get("kv_asymmetry", {}).get("configs", {})
        if kv:
            report.append(f"### {mk}\n")
            report.append("| Config | PPL Delta (%) | Token Agree | Logit Err |")
            report.append("|--------|--------------|-------------|-----------|")
            for cname, cdata in kv.items():
                report.append(
                    f"| {cname} | {cdata.get('delta_pct', 'N/A')} | "
                    f"{cdata.get('token_agreement', 'N/A')} | "
                    f"{cdata.get('mean_logit_error', 'N/A')} |"
                )
            report.append("")

    # Bandwidth analysis
    report.append("## KV Bandwidth Savings\n")
    bw = load_json("kv_bandwidth_savings.json")
    if bw:
        report.append("| Config | BW Reduction | PPL Delta (%) | Token Agree |")
        report.append("|--------|-------------|--------------|-------------|")
        for cname, cdata in bw.get("configs", {}).items():
            report.append(
                f"| {cname} | {round(cdata.get('bandwidth_reduction', 0) * 100, 1)}% | "
                f"{cdata.get('ppl_delta_pct', 'N/A')} | "
                f"{cdata.get('token_agreement', 'N/A')} |"
            )
        report.append("")

    # Analysis
    report.append("## Analysis\n")
    report.append(
        "### Does cross-layer Fisher interaction predict quantization sensitivity?\n"
    )
    report.append(
        "Compare Spearman correlations above. If interaction_score has higher |rho| than diagonal_fisher, the hypothesis is supported.\n"
    )
    report.append("### Why does Qwen collapse under INT4 while Mistral does not?\n")
    report.append(
        "Compare off-diagonal energy and mean interaction scores. Stronger cross-layer coupling means quantization noise in one layer propagates to others.\n"
    )

    # Plots
    report.append("## Plots\n")
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.endswith(".png"):
            report.append(f"![{f}](plots/{f})")
    report.append("")

    report_path = os.path.join(RESULTS_ROOT, "bpa48_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"  Report saved: {report_path}")


# ============================================================
# Main
# ============================================================
def run_model(model_key):
    print(f"\n{'#' * 60}")
    print(f"# Model: {model_key}")
    print(f"{'#' * 60}")

    model, tokenizer, n_layers = load_model_and_tokenizer(model_key)
    data = {"n_layers": n_layers}

    # Exp 1-2: Gradient capture + diagonal Fisher
    fisher_data = experiment_1_2_gradient_capture(model, tokenizer, model_key, n_layers)
    data["fisher_data"] = fisher_data

    # Exp 3: Cross-layer covariance
    cov_data = experiment_3_cross_layer_covariance(fisher_data, model_key, n_layers)

    # Exp 4: Visualization
    experiment_4_visualization(cov_data, model_key, n_layers)

    # Exp 5: Interaction scores
    interaction_data = experiment_5_interaction_score(cov_data, model_key, n_layers)
    # Add Fisher ranking for strategy comparison
    interaction_data["fisher_ranking"] = fisher_data["ranking"]
    data["interaction_scores"] = interaction_data

    # Exp 6: Correlation with quantization sensitivity
    sensitivity = experiment_6_quant_sensitivity_correlation(
        model, tokenizer, model_key, n_layers, fisher_data, interaction_data
    )
    data["sensitivity"] = sensitivity

    # Exp 7: Interaction-guided strategies
    strategies = experiment_7_interaction_strategies(
        model, tokenizer, model_key, n_layers, interaction_data, cov_data
    )
    data["strategies"] = strategies

    # Exp 10: KV asymmetry sweep
    kv_asym = experiment_10_kv_asymmetry(model, tokenizer, model_key, n_layers)
    data["kv_asymmetry"] = kv_asym

    # Exp 11: K/V Fisher sensitivity
    kv_fisher = experiment_11_kv_fisher(fisher_data, model_key, n_layers)
    data["kv_fisher"] = kv_fisher

    # Exp 14: Bandwidth analysis
    bw = experiment_14_bandwidth_analysis(model_key, n_layers, kv_asym)
    data["bandwidth"] = bw

    # Unload model
    del model
    clear_gpu()

    return data


def main():
    print("=" * 60)
    print("BPA v48: Cross-Layer Fisher Interaction Analysis")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    print(f"GPU: {gpu_info()}")

    t_total = time.time()
    all_model_data = {}

    # Primary model
    all_model_data[PRIMARY_MODEL] = run_model(PRIMARY_MODEL)

    # Secondary model
    try:
        all_model_data["mistral-7b"] = run_model("mistral-7b")
    except Exception as e:
        print(f"\nERROR with mistral-7b: {e}")
        import traceback

        traceback.print_exc()

    # Exp 9/15: Architecture comparison
    experiment_9_15_architecture_comparison(all_model_data)

    # Final report
    generate_report(all_model_data)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"BPA v48 COMPLETE")
    print(f"Total runtime: {total_time:.0f}s ({total_time / 3600:.1f} hours)")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
