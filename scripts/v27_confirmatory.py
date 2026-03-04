#!/usr/bin/env python3
"""BPA v27: Lightweight confirmatory pass on H100.

Re-runs ONLY the headline k* configurations from v26 to verify
reproducibility. Does not repeat full oracle or baselines.
Then attempts Llama-2-7b-hf as a third architecture.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_ROOT = os.environ.get(
    "RESULTS_ROOT", "/mnt/tmpfs/knlp/results/v27/h100_confirmatory"
)
V26_ROOT = "/mnt/tmpfs/knlp/results/v26/artifacts/v26"
os.makedirs(RESULTS_ROOT, exist_ok=True)

SEEDS = [0, 1, 2]
L_SET = [8192, 32768]
W_SINK = 4
W_MIN = 1024
GROUP_SIZE = 32
DECODE_TOKENS = 64
DATASET = "wikitext-2-raw-v1"


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
        "backend": f"cuda={torch.version.cuda}",
    }


_TOKEN_CACHE = {}


def load_wikitext_tokens(tokenizer):
    """Load wikitext tokens (cached)."""
    key = id(tokenizer)
    if key not in _TOKEN_CACHE:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer.encode(text)
        arr = np.array(tokens[:500000], dtype=np.int64)
        print(f"  Loaded wikitext: {len(arr)} tokens")
        _TOKEN_CACHE[key] = arr
    return _TOKEN_CACHE[key]


def load_wikitext_passages(tokenizer, L, seed):
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
            *shape[:-1], pd - hd, device=tensor.device, dtype=tensor.dtype
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


def n_layers(past):
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
            for li in range(n_layers(past)):
                k, v = _cache_get_kv(past, li)
                k_s, v_s = k[:, :, :W_SINK, :], v[:, :, :W_SINK, :]
                k_f, v_f = k[:, :, W_SINK:far_end, :], v[:, :, W_SINK:far_end, :]
                k_n, v_n = k[:, :, far_end:, :], v[:, :, far_end:, :]
                if layer_bits[li] == 8:
                    k_q, v_q = quantize_int8(k_f), quantize_int8(v_f)
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
    latencies = []
    for t in range(DECODE_TOKENS):
        tok = continuation[:, t : t + 1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        all_logits.append(out.logits)
        past = out.past_key_values

    logits = torch.cat(all_logits, dim=1)  # [B, DECODE_TOKENS+1, V]
    # logits[:, :-1, :] predicts continuation tokens
    B, T, V = logits[:, :-1, :].shape
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, V).float(),
        continuation.reshape(-1),
        reduction="mean",
    )
    ppl = math.exp(min(loss.item(), 20))
    return ppl, float(np.median(latencies))


def compute_kv_ratio(D, n_kv_heads, head_dim, k, g=32):
    dense = 2 * n_kv_heads * head_dim * 2
    ng = (head_dim + g - 1) // g
    i4 = int(2 * n_kv_heads * head_dim * 0.5 + 2 * n_kv_heads * ng * 2)
    i8 = 2 * n_kv_heads * head_dim + 2 * n_kv_heads * 2
    return (k * i8 + (D - k) * i4) / (D * dense)


def verify_headline(
    model_key, hf_name, D, n_kv_heads, head_dim, oracle_ranking, k_star
):
    """Verify one headline result by re-running k=k* across all (L, seed)."""
    print(f"\n{'='*60}")
    print(f"Verifying {model_key}: k*={k_star}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=torch.float16, trust_remote_code=True
    )
    model = model.to("cuda").eval()

    protected = oracle_ranking[:k_star]
    layer_bits = [4] * D
    for li in protected:
        layer_bits[li] = 8
    kv_ratio = compute_kv_ratio(D, n_kv_heads, head_dim, k_star)

    results = {
        "model": model_key,
        "k": k_star,
        "protected": protected,
        "kv_ratio": round(kv_ratio, 6),
        "evals": {},
    }

    # First get dense baselines
    dense_ppls = {}
    for L in L_SET:
        for seed in SEEDS:
            key = f"L{L}_s{seed}"
            print(f"  Dense {key}...")
            passage = load_wikitext_passages(tokenizer, L, seed)
            ppl, p50 = run_eval(model, tokenizer, passage, L)
            dense_ppls[key] = ppl
            print(f"    PPL={ppl:.4f}")
            torch.cuda.empty_cache()

    # Now quantized at k*
    for L in L_SET:
        for seed in SEEDS:
            key = f"L{L}_s{seed}"
            print(f"  k={k_star} {key}...")
            passage = load_wikitext_passages(tokenizer, L, seed)
            torch.cuda.empty_cache()
            ppl_q, p50 = run_eval(model, tokenizer, passage, L, layer_bits)
            delta = (ppl_q - dense_ppls[key]) / dense_ppls[key] * 100
            results["evals"][key] = {
                "dense_ppl": round(dense_ppls[key], 4),
                "quant_ppl": round(ppl_q, 4),
                "delta_pct": round(delta, 2),
                "pass_3pct": abs(delta) <= 3.0,
                "pass_1pct": abs(delta) <= 1.0,
                "p50_ms": round(p50, 2),
            }
            print(f"    PPL={ppl_q:.4f} delta={delta:+.2f}%")
            torch.cuda.empty_cache()

    all_deltas = [abs(v["delta_pct"]) for v in results["evals"].values()]
    results["max_delta"] = round(max(all_deltas), 2)
    results["pass_3pct"] = all(v["pass_3pct"] for v in results["evals"].values())
    results["pass_1pct"] = all(v["pass_1pct"] for v in results["evals"].values())

    status = "CONFIRMED" if results["pass_3pct"] else "FAILED"
    print(f"\n  {model_key} k*={k_star}: {status} (max_delta={results['max_delta']}%)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_new_model_quick(model_key, hf_name, D, n_kv_heads, head_dim):
    """Quick oracle + k-sweep for a new model. Oracle at L=8K, 1 seed only."""
    from transformers import AutoConfig

    print(f"\n{'='*60}")
    print(f"New model quick screen: {model_key} ({hf_name})")
    print(f"{'='*60}")

    config = AutoConfig.from_pretrained(hf_name)
    max_ctx = getattr(config, "max_position_embeddings", 131072)
    max_L = max_ctx - DECODE_TOKENS
    print(f"  max_position_embeddings={max_ctx}, max usable L={max_L}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=torch.float16, trust_remote_code=True
    )
    model = model.to("cuda").eval()

    # Determine L values that fit in context window
    model_L_set = [L for L in L_SET if L <= max_L]
    if not model_L_set:
        # Use half and full of max_L
        model_L_set = [max_L // 2, max_L]
    print(f"  Using L_SET={model_L_set} (model max_ctx={max_ctx})")

    # Dense baseline at oracle L, seed=0
    L = model_L_set[0]
    seed = 0
    print(f"  Dense baseline L={L} s={seed}...")
    passage = load_wikitext_passages(tokenizer, L, seed)
    dense_ppl, _ = run_eval(model, tokenizer, passage, L)
    print(f"    Dense PPL={dense_ppl:.4f}")

    # Quick oracle: 1 seed, L=8K
    print(f"\n  Oracle screening (1 seed, L=8K)...")
    layer_deltas = []
    for li in range(D):
        lb = [8] * D
        lb[li] = 4
        ppl_q, _ = run_eval(model, tokenizer, passage, L, lb)
        delta = abs((ppl_q - dense_ppl) / dense_ppl * 100)
        layer_deltas.append((li, delta))
        if li % 8 == 0 or delta > 1.0:
            print(f"    Layer {li:2d}: delta={delta:.4f}%")
        torch.cuda.empty_cache()

    layer_deltas.sort(key=lambda x: x[1], reverse=True)
    ranking = [ld[0] for ld in layer_deltas]
    print(f"\n  Top-8 ranking: {ranking[:8]}")
    print(f"  Top-8 deltas: {[round(ld[1], 4) for ld in layer_deltas[:8]]}")

    # Quick k-sweep: k in {0, 1, 2, 3, 4}
    print(f"\n  k-sweep (L={model_L_set}, 3 seeds)...")

    # Get all baselines first
    dense_ppls = {}
    for L_eval in model_L_set:
        for s in SEEDS:
            key = f"L{L_eval}_s{s}"
            p = load_wikitext_passages(tokenizer, L_eval, s)
            ppl, _ = run_eval(model, tokenizer, p, L_eval)
            dense_ppls[key] = ppl
            print(f"    Dense {key}: PPL={ppl:.4f}")
            torch.cuda.empty_cache()

    k_results = {}
    for k in [0, 1, 2, 3, 4]:
        protected = ranking[:k]
        lb = [4] * D
        for li in protected:
            lb[li] = 8
        kv_ratio = compute_kv_ratio(D, n_kv_heads, head_dim, k)

        evals = {}
        for L_eval in model_L_set:
            for s in SEEDS:
                key = f"L{L_eval}_s{s}"
                p = load_wikitext_passages(tokenizer, L_eval, s)
                torch.cuda.empty_cache()
                ppl_q, p50 = run_eval(model, tokenizer, p, L_eval, lb)
                delta = (ppl_q - dense_ppls[key]) / dense_ppls[key] * 100
                evals[key] = {
                    "ppl": round(ppl_q, 4),
                    "delta_pct": round(delta, 2),
                    "pass_3pct": abs(delta) <= 3.0,
                    "pass_1pct": abs(delta) <= 1.0,
                    "p50_ms": round(p50, 2),
                }

        all_d = [abs(v["delta_pct"]) for v in evals.values()]
        p3 = all(v["pass_3pct"] for v in evals.values())
        p1 = all(v["pass_1pct"] for v in evals.values())
        k_results[f"g32_k{k}"] = {
            "k": k,
            "protected": protected,
            "kv_ratio": round(kv_ratio, 6),
            "max_delta": round(max(all_d), 2),
            "pass_3pct": p3,
            "pass_1pct": p1,
            "evals": evals,
        }
        print(f"    k={k}: max_delta={max(all_d):.2f}% pass_3%={'Y' if p3 else 'N'}")

    # Find k*
    kstar3 = None
    kstar1 = None
    for k in [0, 1, 2, 3, 4]:
        cfg = k_results[f"g32_k{k}"]
        if cfg["pass_3pct"] and kstar3 is None:
            kstar3 = k
        if cfg["pass_1pct"] and kstar1 is None:
            kstar1 = k

    results = {
        "version": "v27",
        "model": model_key,
        "hf_name": hf_name,
        "D": D,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "max_ctx": max_ctx,
        "L_set": model_L_set,
        "oracle_ranking": ranking,
        "oracle_scores": [
            {"layer": li, "max_delta": round(d, 4)} for li, d in layer_deltas
        ],
        "k_star_3pct": kstar3,
        "k_star_1pct": kstar1,
        "k_over_D_3pct": round(kstar3 / D, 4) if kstar3 is not None else None,
        "k_results": k_results,
        "gpu_info": get_gpu_info(),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  RESULT: k*(3%)={kstar3}, k*(1%)={kstar1}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("BPA v27: H100 Confirmatory Pass")
    print(f"GPU: {get_gpu_info()['device_name']}")
    print("=" * 60)

    all_results = {}
    t_start = time.time()

    # Step 1: Verify Qwen7B headline
    v26_qwen = json.load(open(os.path.join(V26_ROOT, "k_star_qwen7b.json")))
    v26_qwen_oracle = json.load(
        open(os.path.join(V26_ROOT, "oracle_sensitivity_qwen7b.json"))
    )
    qwen_result = verify_headline(
        "qwen7b",
        "Qwen/Qwen2.5-7B",
        28,
        4,
        128,
        v26_qwen_oracle["oracle_ranking"],
        v26_qwen["k_star_3pct"],
    )
    all_results["qwen7b_verify"] = qwen_result
    save_json(qwen_result, os.path.join(RESULTS_ROOT, "verify_qwen7b.json"))

    # Step 2: Verify Mistral7B headline
    v26_mistral = json.load(open(os.path.join(V26_ROOT, "k_star_mistral7b.json")))
    v26_mistral_oracle = json.load(
        open(os.path.join(V26_ROOT, "oracle_sensitivity_mistral7b.json"))
    )
    mistral_result = verify_headline(
        "mistral7b",
        "mistralai/Mistral-7B-v0.1",
        32,
        8,
        128,
        v26_mistral_oracle["oracle_ranking"],
        v26_mistral["k_star_3pct"],
    )
    all_results["mistral7b_verify"] = mistral_result
    save_json(mistral_result, os.path.join(RESULTS_ROOT, "verify_mistral7b.json"))

    # Step 3: Try Llama-2-7b-hf as third architecture
    print("\n\nAttempting third architecture: Llama-2-7b-hf...")
    try:
        llama_result = run_new_model_quick(
            "llama2_7b", "NousResearch/Llama-2-7b-hf", 32, 32, 128
        )
        all_results["llama2_7b"] = llama_result
        save_json(llama_result, os.path.join(RESULTS_ROOT, "k_star_llama2_7b.json"))
    except Exception as e:
        print(f"  Llama-2-7b-hf FAILED: {e}")
        all_results["llama2_7b"] = {"error": str(e)}
        save_json(
            {"error": str(e), "timestamp": datetime.now().isoformat()},
            os.path.join(RESULTS_ROOT, "k_star_llama2_7b.json"),
        )

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V27 Confirmatory Pass Complete ({elapsed/3600:.1f}h)")
    print(f"{'='*60}")

    summary = {
        "version": "v27",
        "gpu_info": get_gpu_info(),
        "elapsed_hours": round(elapsed / 3600, 2),
        "results": {},
    }

    for key, res in all_results.items():
        if "error" in res:
            summary["results"][key] = {"status": "FAILED", "error": res["error"]}
        elif "pass_3pct" in res:
            summary["results"][key] = {
                "status": "CONFIRMED" if res["pass_3pct"] else "FAILED",
                "k": res.get("k"),
                "max_delta": res.get("max_delta"),
            }
        elif "k_star_3pct" in res:
            summary["results"][key] = {
                "status": "COMPLETED",
                "k_star_3pct": res["k_star_3pct"],
                "k_star_1pct": res["k_star_1pct"],
            }

    summary["timestamp"] = datetime.now().isoformat()
    save_json(summary, os.path.join(RESULTS_ROOT, "v27_summary.json"))

    for key, info in summary["results"].items():
        print(f"  {key}: {info}")


if __name__ == "__main__":
    main()
