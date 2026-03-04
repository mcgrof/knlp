#!/usr/bin/env python3
"""BPA v28 Phase 1: Depth extension with Qwen2.5-14B (D=48).

Runs the canonical protocol on a D>32 model to extend the O(1)
k* scaling evidence. Procedure:
1. Quick 1-seed oracle sweep at L=8K (all 48 layers)
2. Refine top-8 layers with 3 seeds
3. k-sweep at L={8K, 32K} with 3 seeds
4. Determine k*(3%) and kv_ratio
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/mnt/tmpfs/knlp/results/v28")
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Frozen protocol
SEEDS = [0, 1, 2]
L_SET = [8192, 32768]
W_SINK = 4
W_MIN = 1024
GROUP_SIZE = 32
DECODE_TOKENS = 64
DATASET = "wikitext-103-raw-v1"
N_TOKENS = 500000
EPSILON = 0.03
VERSION = "v28"

# Target model
MODEL_KEY = "qwen25_14b"
HF_NAME = "Qwen/Qwen2.5-14B"
ARCH = "Qwen2"
D = 48
N_KV_HEADS = 8
HEAD_DIM = 128


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def get_gpu_info():
    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


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


def load_passage(tokenizer, L, seed):
    token_data = load_wikitext_tokens(tokenizer)
    seq_len = L + DECODE_TOKENS
    rng = np.random.RandomState(seed)
    start = rng.randint(0, max(1, len(token_data) - seq_len))
    batch = token_data[start : start + seq_len]
    return torch.from_numpy(batch).unsqueeze(0)


def quantize_int4_grouped(tensor, group_size=32):
    shape = tensor.shape
    hd = shape[-1]
    ng = (hd + group_size - 1) // group_size
    pd = ng * group_size
    if pd > hd:
        pad = torch.zeros(
            *shape[:-1],
            pd - hd,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=-1)
    r = tensor.reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = amax / 7.0
    q = (r / s).round().clamp(-8, 7)
    return (q * s).reshape(*shape[:-1], pd)[..., :hd]


def quantize_int8(tensor):
    amax = tensor.abs().amax().clamp(min=1e-8)
    s = amax / 127.0
    return ((tensor / s).round().clamp(-128, 127)) * s


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


def run_eval(model, tokenizer, passage, L, layer_bits=None):
    device = next(model.parameters()).device
    input_ids = passage[:, :L].to(device)
    continuation = passage[:, L : L + DECODE_TOKENS].to(device)

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    if layer_bits is not None:
        clen = cache_length(past)
        far_end = clen - W_MIN
        if far_end > W_SINK:
            for li in range(n_cache_layers(past)):
                k, v = _cache_get_kv(past, li)
                k_s = k[:, :, :W_SINK, :]
                v_s = v[:, :, :W_SINK, :]
                k_f = k[:, :, W_SINK:far_end, :]
                v_f = v[:, :, W_SINK:far_end, :]
                k_n = k[:, :, far_end:, :]
                v_n = v[:, :, far_end:, :]
                if layer_bits[li] == 8:
                    k_q = quantize_int8(k_f)
                    v_q = quantize_int8(v_f)
                else:
                    k_q = quantize_int4_grouped(k_f, GROUP_SIZE)
                    v_q = quantize_int4_grouped(v_f, GROUP_SIZE)
                _cache_set_kv(
                    past,
                    li,
                    torch.cat([k_s, k_q, k_n], dim=2),
                    torch.cat([v_s, v_q, v_n], dim=2),
                )

    all_logits = [out.logits[:, -1:, :]]
    for t in range(DECODE_TOKENS):
        tok = continuation[:, t : t + 1]
        with torch.no_grad():
            out = model(tok, past_key_values=past, use_cache=True)
        all_logits.append(out.logits)
        past = out.past_key_values

    logits = torch.cat(all_logits, dim=1)
    B, T, V = logits[:, :-1, :].shape
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, V).float(),
        continuation.reshape(-1),
        reduction="mean",
    )
    ppl = math.exp(min(loss.item(), 20))
    return ppl


def compute_kv_ratio(num_layers, n_kv_heads, head_dim, k, g=32):
    dense = 2 * n_kv_heads * head_dim * 2
    ng = (head_dim + g - 1) // g
    i4 = int(2 * n_kv_heads * head_dim * 0.5 + 2 * n_kv_heads * ng * 2)
    i8 = 2 * n_kv_heads * head_dim + 2 * n_kv_heads * 2
    return (k * i8 + (num_layers - k) * i4) / (num_layers * dense)


def main():
    t_start = time.time()
    gpu = get_gpu_info()
    print("=" * 60)
    print(f"BPA {VERSION} Phase 1: Depth Extension")
    print(f"Model: {HF_NAME} (D={D})")
    print(f"GPU: {gpu['device_name']}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("cuda").eval()
    print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Check context window
    cfg = AutoConfig.from_pretrained(HF_NAME)
    max_ctx = getattr(cfg, "max_position_embeddings", 131072)
    print(f"  max_position_embeddings={max_ctx}")

    # === Phase 1a: Quick oracle sweep (1 seed, L=8K) ===
    print(f"\n--- Oracle Sweep (1 seed, L=8192) ---")
    oracle_L = 8192
    oracle_seed = 0
    passage = load_passage(tokenizer, oracle_L, oracle_seed)
    dense_ppl_oracle = run_eval(model, tokenizer, passage, oracle_L)
    print(f"  Dense PPL at L={oracle_L} s={oracle_seed}: {dense_ppl_oracle:.4f}")

    layer_deltas_coarse = []
    for li in range(D):
        lb = [8] * D
        lb[li] = 4
        ppl_q = run_eval(model, tokenizer, passage, oracle_L, lb)
        delta = abs((ppl_q - dense_ppl_oracle) / dense_ppl_oracle * 100)
        layer_deltas_coarse.append((li, delta))
        if li % 8 == 0 or delta > 1.0:
            print(f"    Layer {li:2d}: delta={delta:.4f}%")
        torch.cuda.empty_cache()

    layer_deltas_coarse.sort(key=lambda x: x[1], reverse=True)
    coarse_ranking = [ld[0] for ld in layer_deltas_coarse]
    print(f"\n  Top-8 (coarse): {coarse_ranking[:8]}")
    print(f"  Top-8 deltas: " f"{[round(ld[1], 4) for ld in layer_deltas_coarse[:8]]}")

    # === Phase 1b: Refine top-8 with 3 seeds ===
    print(f"\n--- Refining top-8 with 3 seeds ---")
    top8 = coarse_ranking[:8]
    refined_deltas = {}
    for seed in SEEDS:
        passage = load_passage(tokenizer, oracle_L, seed)
        dense_ppl = run_eval(model, tokenizer, passage, oracle_L)
        for li in top8:
            lb = [8] * D
            lb[li] = 4
            ppl_q = run_eval(model, tokenizer, passage, oracle_L, lb)
            delta = abs((ppl_q - dense_ppl) / dense_ppl * 100)
            if li not in refined_deltas:
                refined_deltas[li] = []
            refined_deltas[li].append(delta)
            torch.cuda.empty_cache()

    # Rank by max delta across seeds
    refined_scores = [(li, max(refined_deltas[li])) for li in top8]
    refined_scores.sort(key=lambda x: x[1], reverse=True)
    refined_top8 = [rs[0] for rs in refined_scores]

    # Build final ranking: refined top-8 + rest in coarse order
    final_ranking = refined_top8 + [
        li for li in coarse_ranking if li not in refined_top8
    ]
    print(f"  Refined top-8: {refined_top8}")
    print(f"  Refined max deltas: " f"{[round(rs[1], 4) for rs in refined_scores]}")

    # === Phase 1c: k-sweep (k=0..4) across L={8K,32K}, 3 seeds ===
    print(f"\n--- k-sweep (L={L_SET}, 3 seeds) ---")

    # Dense baselines
    dense_ppls = {}
    for L in L_SET:
        for seed in SEEDS:
            key = f"L{L}_s{seed}"
            passage = load_passage(tokenizer, L, seed)
            ppl = run_eval(model, tokenizer, passage, L)
            dense_ppls[key] = ppl
            print(f"  Dense {key}: PPL={ppl:.4f}")
            torch.cuda.empty_cache()

    k_results = {}
    for k in range(5):
        protected = final_ranking[:k]
        lb = [4] * D
        for li in protected:
            lb[li] = 8
        kv_ratio = compute_kv_ratio(D, N_KV_HEADS, HEAD_DIM, k)

        evals = {}
        for L in L_SET:
            for seed in SEEDS:
                key = f"L{L}_s{seed}"
                passage = load_passage(tokenizer, L, seed)
                torch.cuda.empty_cache()
                ppl_q = run_eval(model, tokenizer, passage, L, lb)
                delta = (ppl_q - dense_ppls[key]) / dense_ppls[key] * 100
                evals[key] = {
                    "dense_ppl": round(dense_ppls[key], 4),
                    "quant_ppl": round(ppl_q, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                }

        all_d = [abs(v["delta_pct"]) for v in evals.values()]
        p3 = all(v["pass_3pct"] for v in evals.values())
        k_results[k] = {
            "k": k,
            "protected": protected,
            "kv_ratio": round(kv_ratio, 6),
            "max_delta": round(max(all_d), 2),
            "pass_3pct": p3,
            "evals": evals,
        }
        tag = "Y" if p3 else "N"
        print(
            f"  k={k}: max_delta={max(all_d):.2f}% "
            f"PASS_3%={tag} kv_ratio={kv_ratio:.4f}"
        )

    # Find k*
    kstar3 = None
    for k in range(5):
        if k_results[k]["pass_3pct"]:
            kstar3 = k
            break

    elapsed = time.time() - t_start

    result = {
        "version": VERSION,
        "phase": "depth_extension",
        "model": MODEL_KEY,
        "hf_name": HF_NAME,
        "arch": ARCH,
        "D": D,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_ctx": max_ctx,
        "L_set": L_SET,
        "oracle_ranking_coarse": coarse_ranking,
        "oracle_scores_coarse": [
            {"layer": li, "max_delta": round(d, 4)} for li, d in layer_deltas_coarse
        ],
        "oracle_ranking_refined": final_ranking,
        "refined_scores": [
            {"layer": li, "max_delta": round(d, 4)} for li, d in refined_scores
        ],
        "k_star_3pct": kstar3,
        "kv_ratio": (k_results[kstar3]["kv_ratio"] if kstar3 is not None else None),
        "max_delta": (k_results[kstar3]["max_delta"] if kstar3 is not None else None),
        "k_results": {str(k): v for k, v in k_results.items()},
        "gpu_info": gpu,
        "elapsed_hours": round(elapsed / 3600, 2),
        "protocol": {
            "dataset": DATASET,
            "n_tokens": N_TOKENS,
            "decode_tokens": DECODE_TOKENS,
            "seeds": SEEDS,
            "W_sink": W_SINK,
            "W_min": W_MIN,
            "group_size": GROUP_SIZE,
            "epsilon": EPSILON,
        },
        "timestamp": datetime.now().isoformat(),
    }

    save_json(result, os.path.join(RESULTS_ROOT, "depth_extension_results.json"))

    print(f"\n{'='*60}")
    print(f"Depth Extension Complete ({elapsed/3600:.1f}h)")
    print(f"  Model: {HF_NAME} (D={D})")
    print(f"  k*(3%): {kstar3}")
    if kstar3 is not None:
        print(f"  k*/D: {kstar3/D:.4f}")
        print(f"  kv_ratio: {k_results[kstar3]['kv_ratio']}")
        print(f"  max_delta: {k_results[kstar3]['max_delta']}%")
    print("=" * 60)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
