#!/usr/bin/env python3
"""BPA v28: Canonical evaluation under frozen protocol.

Re-runs all headline models under the single frozen protocol
defined in artifacts/v28/canonical_protocol.json. Produces the
canonical results table for the paper.

All discrepancies from v26/v27 are resolved:
- DATASET constant matches actual load (wikitext-103-raw-v1)
- DECODE_TOKENS fixed at 64
- Text sampling: single contiguous passage per (L, seed)
- PPL: shifted logits, capped at exp(20)
"""

import csv
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
V26_ROOT = "/mnt/tmpfs/knlp/results/v26/artifacts/v26"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# === Frozen protocol parameters (from canonical_protocol.json) ===
SEEDS = [0, 1, 2]
L_SET = [8192, 32768]
W_SINK = 4
W_MIN = 1024
GROUP_SIZE = 32
DECODE_TOKENS = 64
DATASET = "wikitext-103-raw-v1"  # Fixed: v27 had stale "wikitext-2"
N_TOKENS = 500000
EPSILON = 0.03
VERSION = "v28"


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


def compute_kv_ratio(D, n_kv_heads, head_dim, k, g=32):
    dense = 2 * n_kv_heads * head_dim * 2
    ng = (head_dim + g - 1) // g
    i4 = int(2 * n_kv_heads * head_dim * 0.5 + 2 * n_kv_heads * ng * 2)
    i8 = 2 * n_kv_heads * head_dim + 2 * n_kv_heads * 2
    return (k * i8 + (D - k) * i4) / (D * dense)


def run_oracle_sweep(model, tokenizer, D, L, seed):
    """Oracle sensitivity: set each layer to INT4, rest INT8."""
    print(f"  Oracle sweep: L={L}, seed={seed}, D={D}")
    passage = load_passage(tokenizer, L, seed)
    dense_ppl = run_eval(model, tokenizer, passage, L)
    print(f"    Dense PPL={dense_ppl:.4f}")

    layer_deltas = []
    for li in range(D):
        lb = [8] * D
        lb[li] = 4
        ppl_q = run_eval(model, tokenizer, passage, L, lb)
        delta = abs((ppl_q - dense_ppl) / dense_ppl * 100)
        layer_deltas.append((li, delta))
        if li % 8 == 0 or delta > 1.0:
            print(f"    Layer {li:2d}: delta={delta:.4f}%")
        torch.cuda.empty_cache()

    layer_deltas.sort(key=lambda x: x[1], reverse=True)
    ranking = [ld[0] for ld in layer_deltas]
    return ranking, layer_deltas, dense_ppl


def run_k_sweep(model, tokenizer, D, n_kv_heads, head_dim, ranking, L_set, max_k=4):
    """k-sweep across all (L, seed) pairs."""
    dense_ppls = {}
    for L in L_set:
        for seed in SEEDS:
            key = f"L{L}_s{seed}"
            passage = load_passage(tokenizer, L, seed)
            ppl = run_eval(model, tokenizer, passage, L)
            dense_ppls[key] = ppl
            print(f"    Dense {key}: PPL={ppl:.4f}")
            torch.cuda.empty_cache()

    k_results = {}
    for k in range(max_k + 1):
        protected = ranking[:k]
        lb = [4] * D
        for li in protected:
            lb[li] = 8
        kv_ratio = compute_kv_ratio(D, n_kv_heads, head_dim, k)

        evals = {}
        for L in L_set:
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
            f"    k={k}: max_delta={max(all_d):.2f}% "
            f"PASS_3%={tag} kv_ratio={kv_ratio:.4f}"
        )

    return k_results, dense_ppls


# ===== Model definitions =====

MODELS = [
    {
        "key": "qwen25_7b",
        "hf_name": "Qwen/Qwen2.5-7B",
        "arch": "Qwen2",
        "D": 28,
        "n_kv_heads": 4,
        "head_dim": 128,
        "use_v26_oracle": True,
    },
    {
        "key": "mistral_7b",
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "arch": "Mistral",
        "D": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "use_v26_oracle": True,
    },
    {
        "key": "llama2_7b",
        "hf_name": "NousResearch/Llama-2-7b-hf",
        "arch": "Llama",
        "D": 32,
        "n_kv_heads": 32,
        "head_dim": 128,
        "use_v26_oracle": False,
    },
]


def load_v26_oracle(model_key):
    """Load oracle ranking from v26 artifacts."""
    name_map = {
        "qwen25_7b": "qwen7b",
        "mistral_7b": "mistral7b",
    }
    v26_key = name_map.get(model_key)
    if v26_key is None:
        return None
    path = os.path.join(V26_ROOT, f"oracle_sensitivity_{v26_key}.json")
    if not os.path.exists(path):
        print(f"  WARNING: v26 oracle not found: {path}")
        return None
    data = json.load(open(path))
    return data["oracle_ranking"]


def evaluate_model(mdef):
    key = mdef["key"]
    hf_name = mdef["hf_name"]
    D = mdef["D"]
    n_kv = mdef["n_kv_heads"]
    hd = mdef["head_dim"]

    print(f"\n{'='*60}")
    print(f"Model: {key} ({hf_name})")
    print(f"  D={D}, n_kv_heads={n_kv}, head_dim={hd}")
    print(f"{'='*60}")

    cfg = AutoConfig.from_pretrained(hf_name)
    max_ctx = getattr(cfg, "max_position_embeddings", 131072)
    max_L = max_ctx - DECODE_TOKENS
    model_L_set = [L for L in L_SET if L <= max_L]
    if not model_L_set:
        half_L = max_L // 2 - 32
        full_L = max_L - 64
        model_L_set = [half_L, full_L]
    print(f"  max_ctx={max_ctx}, L_set={model_L_set}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("cuda").eval()

    # Oracle ranking
    if mdef.get("use_v26_oracle"):
        ranking = load_v26_oracle(key)
        if ranking is None:
            print("  Falling back to fresh oracle...")
            ranking, _, _ = run_oracle_sweep(model, tokenizer, D, model_L_set[0], 0)
    else:
        ranking, layer_deltas, _ = run_oracle_sweep(
            model, tokenizer, D, model_L_set[0], 0
        )

    print(f"  Oracle ranking (top 8): {ranking[:8]}")

    # k-sweep
    k_results, dense_ppls = run_k_sweep(
        model,
        tokenizer,
        D,
        n_kv,
        hd,
        ranking,
        model_L_set,
    )

    # Find k*
    kstar3 = None
    for k in sorted(k_results.keys()):
        if k_results[k]["pass_3pct"]:
            kstar3 = k
            break

    kv_ratio = k_results[kstar3]["kv_ratio"] if kstar3 is not None else None
    max_delta = k_results[kstar3]["max_delta"] if kstar3 is not None else None

    result = {
        "version": VERSION,
        "model": key,
        "hf_name": hf_name,
        "arch": mdef["arch"],
        "D": D,
        "n_kv_heads": n_kv,
        "head_dim": hd,
        "max_ctx": max_ctx,
        "L_set": model_L_set,
        "oracle_ranking": ranking,
        "k_star_3pct": kstar3,
        "kv_ratio": kv_ratio,
        "max_delta": max_delta,
        "k_results": {str(k): v for k, v in k_results.items()},
        "gpu_info": get_gpu_info(),
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

    print(f"\n  RESULT: k*(3%)={kstar3}, kv_ratio={kv_ratio}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_canonical_csv(all_results, path):
    """Write canonical results table as CSV."""
    rows = []
    for res in all_results:
        rows.append(
            {
                "model": res["hf_name"],
                "architecture": res["arch"],
                "D": res["D"],
                "n_kv_heads": res["n_kv_heads"],
                "head_dim": res["head_dim"],
                "GPU": res["gpu_info"]["device_name"],
                "quant_scheme": f"g{GROUP_SIZE}_INT4_INT8_symmetric",
                "ranking_method": "oracle_per_layer_ablation",
                "epsilon": f"{EPSILON*100:.0f}%",
                "k_star": res["k_star_3pct"],
                "k_star_over_D": (
                    round(res["k_star_3pct"] / res["D"], 4)
                    if res["k_star_3pct"] is not None
                    else None
                ),
                "kv_ratio": res["kv_ratio"],
                "max_delta_pct": res["max_delta"],
                "PASS_3pct": ("Y" if res["k_star_3pct"] is not None else "N"),
                "L_set": str(res["L_set"]),
            }
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Canonical CSV: {path}")


def main():
    t_start = time.time()
    gpu = get_gpu_info()
    print("=" * 60)
    print(f"BPA {VERSION}: Canonical Evaluation")
    print(f"GPU: {gpu['device_name']}")
    print(f"PyTorch: {gpu['torch_version']}")
    print(f"CUDA: {gpu['cuda_version']}")
    print(f"Protocol: {DATASET}, seeds={SEEDS}, L={L_SET}")
    print("=" * 60)

    # Select models to run based on command line
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        models = [m for m in MODELS if m["key"] in keys]
    else:
        models = MODELS

    all_results = []
    for mdef in models:
        try:
            result = evaluate_model(mdef)
            all_results.append(result)
            save_json(
                result,
                os.path.join(
                    RESULTS_ROOT,
                    f"canonical_{mdef['key']}.json",
                ),
            )
        except Exception as e:
            print(f"\n  FAILED: {mdef['key']}: {e}")
            import traceback

            traceback.print_exc()

    # Write canonical table
    if all_results:
        csv_path = os.path.join(RESULTS_ROOT, "canonical_results_table.csv")
        write_canonical_csv(all_results, csv_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Complete ({elapsed/3600:.1f}h)")
    for r in all_results:
        print(
            f"  {r['model']}: k*={r['k_star_3pct']} "
            f"kv_ratio={r['kv_ratio']} "
            f"max_delta={r['max_delta']}%"
        )
    print("=" * 60)

    # Save summary
    summary = {
        "version": VERSION,
        "gpu_info": gpu,
        "elapsed_hours": round(elapsed / 3600, 2),
        "results": [
            {
                "model": r["model"],
                "arch": r["arch"],
                "D": r["D"],
                "k_star_3pct": r["k_star_3pct"],
                "kv_ratio": r["kv_ratio"],
                "max_delta": r["max_delta"],
            }
            for r in all_results
        ],
        "timestamp": datetime.now().isoformat(),
    }
    save_json(
        summary,
        os.path.join(RESULTS_ROOT, "canonical_summary.json"),
    )


if __name__ == "__main__":
    main()
