#!/usr/bin/env python3
"""BPA v27: Confirmatory H100 pass.

Re-runs headline configs from v26, verifies results, and optionally
adds a third architecture (Llama-2-7b-hf if accessible).
"""

import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================

RESULTS_ROOT = os.environ.get(
    "RESULTS_ROOT", "/mnt/tmpfs/knlp/results/v27/h100_confirmatory"
)
os.makedirs(RESULTS_ROOT, exist_ok=True)

MODEL_REGISTRY = {
    "qwen7b": {
        "hf_name": "Qwen/Qwen2.5-7B",
        "D": 28,
        "n_kv_heads": 4,
        "head_dim": 128,
    },
    "mistral7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "D": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
    },
    "llama2_7b": {
        "hf_name": "NousResearch/Llama-2-7b-hf",
        "D": 32,
        "n_kv_heads": 32,
        "head_dim": 128,
    },
}

SEEDS = [0, 1, 2]
L_SET = [8192, 32768]
W_SINK = 4
W_MIN = 1024
GROUP_SIZE = 32
DECODE_TOKENS = 64
DATASET = "wikitext-2-raw-v1"
EPS_3 = 3.0
EPS_1 = 1.0


# ============================================================
# Utilities
# ============================================================


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def get_gpu_info():
    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "torch_version": torch.__version__,
        "backend": f"cuda={torch.version.cuda}",
    }


def load_wikitext_passages(tokenizer, L, seed, num_passages=1):
    from datasets import load_dataset

    ds = load_dataset("wikitext", DATASET, split="test")
    text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    rng = np.random.RandomState(seed)
    max_start = len(tokens) - (L + DECODE_TOKENS) - 1
    passages = []
    for _ in range(num_passages):
        start = rng.randint(0, max(1, max_start))
        passage = tokens[start : start + L + DECODE_TOKENS].unsqueeze(0)
        passages.append(passage)
    return passages


# ============================================================
# Quantization backends (identical to v26)
# ============================================================


def quantize_int4_grouped(tensor, group_size=32):
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    n_groups = (head_dim + group_size - 1) // group_size
    padded_dim = n_groups * group_size

    if padded_dim > head_dim:
        pad = torch.zeros(
            *orig_shape[:-1], padded_dim - head_dim,
            device=tensor.device, dtype=tensor.dtype
        )
        tensor = torch.cat([tensor, pad], dim=-1)

    reshaped = tensor.reshape(*orig_shape[:-1], n_groups, group_size)
    amax = reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    quantized = (reshaped / scale).round().clamp(-8, 7)
    dequantized = quantized * scale
    result = dequantized.reshape(*orig_shape[:-1], padded_dim)
    return result[..., :head_dim]


def quantize_int8(tensor):
    amax = tensor.abs().amax().clamp(min=1e-8)
    scale = amax / 127.0
    quantized = (tensor / scale).round().clamp(-128, 127)
    return quantized * scale


def compute_kv_ratio(D, n_kv_heads, head_dim, k, group_size=32):
    dense_layer = 2 * n_kv_heads * head_dim * 2
    n_groups = (head_dim + group_size - 1) // group_size
    int4_layer = int(2 * n_kv_heads * head_dim * 0.5 + 2 * n_kv_heads * n_groups * 2)
    int8_layer = 2 * n_kv_heads * head_dim * 1 + 2 * n_kv_heads * 2
    total = k * int8_layer + (D - k) * int4_layer
    return total / (D * dense_layer)


# ============================================================
# Cache helpers (transformers 5.x compatible)
# ============================================================


def _cache_get_kv(past, layer_idx):
    if hasattr(past, "key_cache"):
        return past.key_cache[layer_idx], past.value_cache[layer_idx]
    return past[layer_idx]


def _cache_set_kv(past, layer_idx, k, v):
    if hasattr(past, "key_cache"):
        past.key_cache[layer_idx] = k
        past.value_cache[layer_idx] = v
    else:
        past[layer_idx] = (k, v)


def cache_length(past):
    if hasattr(past, "key_cache"):
        return past.key_cache[0].shape[2]
    return past[0][0].shape[2]


def n_layers_in_cache(past):
    if hasattr(past, "key_cache"):
        return len(past.key_cache)
    return len(past)


# ============================================================
# Evaluation core
# ============================================================


def run_eval(model, tokenizer, passage, L, layer_bits=None):
    """Run one evaluation. Returns (ppl, p50_ms)."""
    device = next(model.parameters()).device
    input_ids = passage[:, :L].to(device)
    continuation = passage[:, L : L + DECODE_TOKENS].to(device)

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    # Apply quantization to far region if layer_bits specified
    if layer_bits is not None:
        clen = cache_length(past)
        far_end = clen - W_MIN
        if far_end > W_SINK:
            n_lay = n_layers_in_cache(past)
            for li in range(n_lay):
                k, v = _cache_get_kv(past, li)
                k_sink = k[:, :, :W_SINK, :]
                v_sink = v[:, :, :W_SINK, :]
                k_far = k[:, :, W_SINK:far_end, :]
                v_far = v[:, :, W_SINK:far_end, :]
                k_near = k[:, :, far_end:, :]
                v_near = v[:, :, far_end:, :]

                if layer_bits[li] == 8:
                    k_q = quantize_int8(k_far)
                    v_q = quantize_int8(v_far)
                else:
                    k_q = quantize_int4_grouped(k_far, GROUP_SIZE)
                    v_q = quantize_int4_grouped(v_far, GROUP_SIZE)

                k_new = torch.cat([k_sink, k_q, k_near], dim=2)
                v_new = torch.cat([v_sink, v_q, v_near], dim=2)
                _cache_set_kv(past, li, k_new, v_new)

    # Decode
    losses = []
    latencies = []
    pos = cache_length(past)

    for t in range(DECODE_TOKENS):
        tok = continuation[:, t : t + 1]
        position_ids = torch.tensor([[pos]], device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model(tok, past_key_values=past, use_cache=True, position_ids=position_ids)

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = out.logits[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, tok.squeeze(1))
        losses.append(loss.item())
        past = out.past_key_values
        pos += 1

    ppl = float(np.exp(np.mean(losses)))
    p50_ms = float(np.median(latencies))
    return ppl, p50_ms


# ============================================================
# Phase 0: Dense baselines
# ============================================================


def phase0_baselines(model_key, model_cfg):
    print(f"\n{'='*60}")
    print(f"Phase 0: Dense baselines — {model_key}")
    print(f"{'='*60}")

    hf_name = model_cfg["hf_name"]
    D = model_cfg["D"]

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = {
        "version": "v27",
        "model": model_key,
        "n_layers": D,
        "dense_ppls": {},
        "int8_all": {"evals": {}},
        "gpu_info": get_gpu_info(),
        "timestamp": datetime.now().isoformat(),
    }

    for L in L_SET:
        for seed in SEEDS:
            key = f"L{L}_s{seed}"
            print(f"  Dense {key}...")
            passages = load_wikitext_passages(tokenizer, L, seed)
            ppl, p50 = run_eval(model, tokenizer, passages[0], L)
            results["dense_ppls"][key] = float(ppl)
            print(f"    PPL={ppl:.4f}, p50={p50:.2f}ms")

            # INT8-all
            print(f"  INT8-all {key}...")
            layer_bits = [8] * D
            ppl_q, p50_q = run_eval(model, tokenizer, passages[0], L, layer_bits)
            delta = (ppl_q - ppl) / ppl * 100
            results["int8_all"]["evals"][key] = {
                "ppl": float(ppl_q),
                "delta_pct": round(delta, 2),
                "pass_1pct": abs(delta) <= EPS_1,
                "pass_3pct": abs(delta) <= EPS_3,
                "p50_ms": round(p50_q, 2),
            }
            print(f"    PPL={ppl_q:.4f}, delta={delta:+.2f}%")

            torch.cuda.empty_cache()

    # Summarize INT8-all
    all_deltas = [abs(v["delta_pct"]) for v in results["int8_all"]["evals"].values()]
    results["int8_all"]["max_delta"] = round(max(all_deltas), 2)
    results["int8_all"]["pass_1pct"] = all(
        v["pass_1pct"] for v in results["int8_all"]["evals"].values()
    )
    results["int8_all"]["pass_3pct"] = all(
        v["pass_3pct"] for v in results["int8_all"]["evals"].values()
    )

    path = os.path.join(RESULTS_ROOT, f"phase0_{model_key}.json")
    save_json(results, path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Phase 1: Oracle sensitivity ranking
# ============================================================


def phase1_oracle(model_key, model_cfg, phase0_results):
    print(f"\n{'='*60}")
    print(f"Phase 1: Oracle sensitivity — {model_key}")
    print(f"{'='*60}")

    hf_name = model_cfg["hf_name"]
    D = model_cfg["D"]

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Oracle at L=8192 only (faster)
    L = 8192
    layer_scores = []

    for layer_i in range(D):
        layer_bits = [8] * D
        layer_bits[layer_i] = 4

        deltas = []
        for seed in SEEDS:
            key = f"L{L}_s{seed}"
            dense_ppl = phase0_results["dense_ppls"][key]
            passages = load_wikitext_passages(tokenizer, L, seed)
            ppl_q, _ = run_eval(model, tokenizer, passages[0], L, layer_bits)
            delta = abs((ppl_q - dense_ppl) / dense_ppl * 100)
            deltas.append(delta)
            torch.cuda.empty_cache()

        max_d = max(deltas)
        mean_d = sum(deltas) / len(deltas)
        layer_scores.append(
            {"layer": layer_i, "max_delta": round(max_d, 4), "mean_delta": round(mean_d, 4)}
        )
        print(f"  Layer {layer_i:2d}: max_delta={max_d:.4f}%")

    # Sort by max_delta descending
    layer_scores.sort(key=lambda x: x["max_delta"], reverse=True)
    oracle_ranking = [s["layer"] for s in layer_scores]

    results = {
        "version": "v27",
        "model": model_key,
        "n_layers": D,
        "oracle_L": L,
        "seeds": SEEDS,
        "oracle_ranking": oracle_ranking,
        "oracle_scores": layer_scores,
        "gpu_info": get_gpu_info(),
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(RESULTS_ROOT, f"oracle_sensitivity_{model_key}.json")
    save_json(results, path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Phase 2: k-sweep
# ============================================================


def phase2_ksweep(model_key, model_cfg, phase0_results, oracle_results):
    print(f"\n{'='*60}")
    print(f"Phase 2: k-sweep — {model_key}")
    print(f"{'='*60}")

    hf_name = model_cfg["hf_name"]
    D = model_cfg["D"]
    n_kv_heads = model_cfg["n_kv_heads"]
    head_dim = model_cfg["head_dim"]
    oracle_ranking = oracle_results["oracle_ranking"]

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    k_values = [0, 1, 2, 3, 4, 6, 8, 12]
    k_results = {}

    for k in k_values:
        if k > D:
            continue

        protected = oracle_ranking[:k]
        layer_bits = [4] * D
        for li in protected:
            layer_bits[li] = 8

        kv_ratio = compute_kv_ratio(D, n_kv_heads, head_dim, k, GROUP_SIZE)
        config_key = f"g32_k{k}"

        print(f"\n  --- k={k}, protected={protected[:4]}... kv_ratio={kv_ratio:.4f} ---")

        evals = {}
        for L in L_SET:
            for seed in SEEDS:
                key = f"L{L}_s{seed}"
                dense_ppl = phase0_results["dense_ppls"][key]
                passages = load_wikitext_passages(tokenizer, L, seed)

                torch.cuda.empty_cache()
                ppl_q, p50 = run_eval(model, tokenizer, passages[0], L, layer_bits)
                delta = (ppl_q - dense_ppl) / dense_ppl * 100

                evals[key] = {
                    "ppl": round(float(ppl_q), 4),
                    "delta_pct": round(delta, 2),
                    "pass_1pct": abs(delta) <= EPS_1,
                    "pass_3pct": abs(delta) <= EPS_3,
                    "p50_ms": round(p50, 2),
                }
                print(f"    {key}: PPL={ppl_q:.4f} delta={delta:+.2f}%")

        all_deltas = [abs(v["delta_pct"]) for v in evals.values()]
        k_results[config_key] = {
            "k": k,
            "protected_layers": protected,
            "evals": evals,
            "max_delta": round(max(all_deltas), 2),
            "pass_1pct": all(v["pass_1pct"] for v in evals.values()),
            "pass_3pct": all(v["pass_3pct"] for v in evals.values()),
            "kv_ratio": round(kv_ratio, 6),
        }

    # Determine k*
    k_star_3 = D
    k_star_1 = D
    for k in sorted(k_values):
        key = f"g32_k{k}"
        if key in k_results and k_results[key]["pass_3pct"] and k < k_star_3:
            k_star_3 = k
        if key in k_results and k_results[key]["pass_1pct"] and k < k_star_1:
            k_star_1 = k

    if k_star_3 == D:
        k_star_3 = None
    if k_star_1 == D:
        k_star_1 = None

    results = {
        "version": "v27",
        "model": model_key,
        "n_layers": D,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "k_star_1pct": k_star_1,
        "k_star_3pct": k_star_3,
        "k_over_D_3pct": round(k_star_3 / D, 4) if k_star_3 is not None else None,
        "k_results": k_results,
        "oracle_ranking_used": oracle_ranking[:12],
        "gpu_info": get_gpu_info(),
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(RESULTS_ROOT, f"k_star_{model_key}.json")
    save_json(results, path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Phase 3: Verification against v26
# ============================================================


def phase3_verify(v27_results, v26_path):
    """Compare v27 results against v26 to confirm reproducibility."""
    print(f"\n{'='*60}")
    print("Phase 3: Verification against v26")
    print(f"{'='*60}")

    verification = {}
    for model_key in ["qwen7b", "mistral7b"]:
        v26_file = os.path.join(v26_path, f"k_star_{model_key}.json")
        if not os.path.exists(v26_file):
            print(f"  {model_key}: v26 file not found, skipping verification")
            continue

        with open(v26_file) as f:
            v26 = json.load(f)

        v27 = v27_results.get(model_key)
        if v27 is None:
            continue

        v26_kstar = v26.get("k_star_3pct")
        v27_kstar = v27.get("k_star_3pct")

        match = v26_kstar == v27_kstar
        verification[model_key] = {
            "v26_kstar_3pct": v26_kstar,
            "v27_kstar_3pct": v27_kstar,
            "match": match,
            "v26_kv_ratio": v26["k_results"].get(f"g32_k{v26_kstar}", {}).get("kv_ratio"),
            "v27_kv_ratio": v27["k_results"].get(f"g32_k{v27_kstar}", {}).get("kv_ratio"),
        }
        status = "MATCH" if match else "MISMATCH"
        print(f"  {model_key}: v26 k*={v26_kstar}, v27 k*={v27_kstar} -> {status}")

    return verification


# ============================================================
# Main
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen7b", "mistral7b"],
        help="Models to run",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip phase 0 (use existing baselines)",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip oracle (use existing rankings)",
    )
    parser.add_argument(
        "--v26-path",
        default="/mnt/tmpfs/knlp/results/v26/artifacts/v26",
        help="Path to v26 artifacts for verification",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BPA v27: H100 Confirmatory Pass")
    print(f"Models: {args.models}")
    print(f"GPU: {get_gpu_info()['device_name']}")
    print(f"Results: {RESULTS_ROOT}")
    print("=" * 60)

    all_kstar_results = {}

    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model: {model_key}, skipping")
            continue

        model_cfg = MODEL_REGISTRY[model_key]
        print(f"\n{'#'*60}")
        print(f"# Model: {model_key} ({model_cfg['hf_name']})")
        print(f"{'#'*60}")

        # Phase 0
        p0_path = os.path.join(RESULTS_ROOT, f"phase0_{model_key}.json")
        if args.skip_phase0 and os.path.exists(p0_path):
            print(f"  Loading existing Phase 0 from {p0_path}")
            with open(p0_path) as f:
                p0 = json.load(f)
        else:
            p0 = phase0_baselines(model_key, model_cfg)

        # Phase 1
        oracle_path = os.path.join(RESULTS_ROOT, f"oracle_sensitivity_{model_key}.json")
        if args.skip_oracle and os.path.exists(oracle_path):
            print(f"  Loading existing oracle from {oracle_path}")
            with open(oracle_path) as f:
                oracle = json.load(f)
        else:
            oracle = phase1_oracle(model_key, model_cfg, p0)

        # Phase 2
        kstar = phase2_ksweep(model_key, model_cfg, p0, oracle)
        all_kstar_results[model_key] = kstar

        print(f"\n  RESULT: {model_key} k*(3%)={kstar['k_star_3pct']}, "
              f"k*(1%)={kstar['k_star_1pct']}, "
              f"kv_ratio={kstar['k_results'].get(f'g32_k{kstar[\"k_star_3pct\"]}', {}).get('kv_ratio', 'N/A')}")

    # Phase 3: Verify against v26
    verification = phase3_verify(all_kstar_results, args.v26_path)
    save_json(verification, os.path.join(RESULTS_ROOT, "verification_vs_v26.json"))

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mk, res in all_kstar_results.items():
        k3 = res["k_star_3pct"]
        k1 = res["k_star_1pct"]
        kv = res["k_results"].get(f"g32_k{k3}", {}).get("kv_ratio", "N/A") if k3 is not None else "N/A"
        print(f"  {mk}: k*(3%)={k3}, k*(1%)={k1}, kv_ratio={kv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
