#!/usr/bin/env python
"""
BPA v12: Frontier stress benchmark.

Extends v11 with:
- Long context sweep (L up to 16384/32768)
- Attention-grounded retrieval predictor
- Scaling exponent measurement
- Layer-adaptive W
- 32K stress test

Reuses v11 infrastructure for dense/BPA decode, KV eviction, etc.

Commands:
    phase0  -- GPU preflight + model loading
    phase1  -- Bandwidth sensitivity sweep
    phase2  -- Retrieval predictor (attention supervision)
    phase3  -- Scaling exponent
    phase4  -- Layer-adaptive W
    phase5  -- 32K stress
    alltune -- Run matched-quality tuning for all L values
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Import v11 infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.bpa_v11_bench import (
    CHUNK_SIZE,
    DEFAULT_MODEL,
    DTYPE,
    AdaptiveController,
    DecodeResult,
    build_keep_mask,
    build_run_meta,
    compute_ppl,
    context_regime,
    evict_kv_cache,
    get_cpu_rss_mb,
    get_gpu_mem,
    get_text_batch,
    gpu_preflight,
    gpu_sync,
    kv_bytes_per_token,
    kv_cache_len,
    load_hf_model,
    load_validation_tokens,
    print_summary,
    reset_gpu_mem,
    run_bpa_decode,
    run_dense_decode,
    save_all_results,
    select_far_chunks_random,
)
from transformers.cache_utils import DynamicCache

OUTPUT_DIR = "bpa_v12_results"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Phase 0: Preflight + model loading
# ============================================================


def cmd_phase0(args):
    """GPU preflight and model loading."""
    device_str = args.device
    info = gpu_preflight(device_str)

    if not info:
        print("FATAL: GPU required")
        sys.exit(1)

    hip = info.get("hip_version")
    if hip is None:
        print("FATAL: torch.version.hip is None — ROCm required")
        sys.exit(1)

    print(f"  torch: {info['torch_version']}")
    print(f"  ROCm/HIP: {hip}")
    print(f"  Device: {info['device_name']}")
    print(f"  Memory: {info['total_memory_gb']}GB")

    # Load primary model
    print("\n--- Primary model ---")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)

    # Quick smoke test
    rng = np.random.RandomState(1)
    tokens = load_validation_tokens(tokenizer)
    test_ids = get_text_batch(tokens, 1, 128, rng).to(device_str)
    with torch.no_grad():
        out = model(test_ids, use_cache=True)
    print(f"  Smoke test OK: logits shape {out.logits.shape}")

    del model, out
    torch.cuda.empty_cache()

    # Attempt larger model
    larger_models = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
    larger_result = None
    for lm in larger_models:
        print(f"\n--- Attempting {lm} ---")
        try:
            lmodel, ltok, lmax, lcfg = load_hf_model(lm, device_str)
            # Smoke test at L=8192
            test_ids = get_text_batch(tokens, 1, 256, rng).to(device_str)
            with torch.no_grad():
                out = lmodel(test_ids, use_cache=True)
            print(f"  Smoke test OK: {lm}")
            larger_result = {
                "model": lm,
                "max_ctx": lmax,
                "config": lcfg,
                "status": "OK",
            }
            del lmodel, out
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  FAILED: {e}")
            torch.cuda.empty_cache()
            continue

    result = {
        "primary_model": args.model,
        "primary_max_ctx": max_ctx,
        "primary_config": model_config,
        "gpu_info": info,
        "larger_model": larger_result,
    }

    ensure_dir(args.output_dir)
    path = os.path.join(args.output_dir, "phase0_preflight.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {path}")


# ============================================================
# Phase 1: Bandwidth sensitivity sweep
# ============================================================


def cmd_phase1(args):
    """Bandwidth sensitivity sweep across L and batch sizes."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)
    run_meta = build_run_meta(device_str, args.model, max_ctx, model_config)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    results = []
    for bs in batch_sizes:
        for L in seq_lens:
            if L > max_ctx:
                print(f"  SKIP L={L} > max_ctx={max_ctx}")
                continue
            if L + args.steps > len(token_data) // bs:
                print(f"  SKIP L={L} bs={bs}: not enough tokens")
                continue
            for seed in seeds:
                print(
                    f"  dense L={L:6d} bs={bs} seed={seed}...",
                    end="",
                    flush=True,
                )
                try:
                    r = run_dense_decode(
                        model,
                        token_data,
                        L,
                        args.steps,
                        seed,
                        device_str,
                        model_config,
                        max_ctx,
                        batch_size=bs,
                    )
                    r.run_meta = run_meta
                    results.append(r)
                    print(
                        f" PPL={r.ppl:.1f} p50={r.decode_per_token_ms:.2f}ms"
                        f" gpu={r.peak_gpu_alloc_mb:.0f}MB"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")
                    torch.cuda.empty_cache()

    # Compute bandwidth scaling
    print("\n=== Bandwidth Scaling Analysis ===")
    by_bs = {}
    for r in results:
        key = r.batch_size
        if key not in by_bs:
            by_bs[key] = {}
        L = r.seq_len
        if L not in by_bs[key]:
            by_bs[key][L] = []
        by_bs[key][L].append(r.decode_per_token_ms)

    scaling = []
    for bs in sorted(by_bs.keys()):
        Ls = sorted(by_bs[bs].keys())
        print(f"\n  Batch={bs}:")
        for L in Ls:
            med = float(np.median(by_bs[bs][L]))
            print(f"    L={L:6d}: p50={med:.2f}ms")

        # Check scaling ratios between consecutive L
        for i in range(1, len(Ls)):
            L_prev, L_curr = Ls[i - 1], Ls[i]
            p50_prev = float(np.median(by_bs[bs][L_prev]))
            p50_curr = float(np.median(by_bs[bs][L_curr]))
            ratio = p50_curr / p50_prev if p50_prev > 0 else 0
            L_ratio = L_curr / L_prev
            entry = {
                "batch_size": bs,
                "L_from": L_prev,
                "L_to": L_curr,
                "L_ratio": L_ratio,
                "latency_ratio": round(ratio, 3),
                "p50_from_ms": round(p50_prev, 3),
                "p50_to_ms": round(p50_curr, 3),
                "bandwidth_bound": ratio >= 1.6,
            }
            scaling.append(entry)
            marker = " <-- BANDWIDTH BOUND" if ratio >= 1.6 else ""
            print(
                f"    {L_prev}->{L_curr} ({L_ratio}x L):"
                f" {ratio:.2f}x latency{marker}"
            )

    ensure_dir(args.output_dir)
    save_all_results(results, args.output_dir)

    scaling_path = os.path.join(args.output_dir, "bandwidth_scaling.json")
    with open(scaling_path, "w") as f:
        json.dump(scaling, f, indent=2)
    print(f"\nSaved scaling: {scaling_path}")

    any_bw = any(s["bandwidth_bound"] for s in scaling)
    print(f"\nBandwidth-bound regime found: {'YES' if any_bw else 'NO'}")


# ============================================================
# Phase 2: Attention-grounded retrieval predictor
# ============================================================


@torch.no_grad()
def record_attention_traces(
    model,
    token_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    model_config,
    max_ctx,
    W_min=64,
):
    """Run dense decode while recording attention maps.

    Returns per-token features and retrieval labels.
    """
    rng = np.random.RandomState(seed)
    idx = get_text_batch(token_data, 1, seq_len + decode_steps, rng).to(device_str)
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    # Prefill with attention outputs
    out = model(prefix, use_cache=True, output_attentions=True)
    past = out.past_key_values

    features = []  # per decode step
    labels = []  # 1 if far_attention_mass > threshold

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]
        out = model(
            next_token,
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
        )
        past = out.past_key_values

        # Extract attention maps: list of [B, n_heads, 1, seq_len+step+1]
        attn_weights = out.attentions  # tuple of per-layer tensors

        # Compute far_attention_mass across all layers
        cache_len = kv_cache_len(past)
        local_start = max(0, cache_len - W_min)

        far_mass_per_layer = []
        attn_entropy_per_layer = []
        for layer_attn in attn_weights:
            # layer_attn: [B, n_heads, 1, cache_len]
            w = layer_attn[0, :, 0, :]  # [n_heads, cache_len]
            # Far tokens are everything before local_start
            if local_start > 0:
                far_w = w[:, :local_start]
                far_mass = far_w.sum(dim=-1).mean().item()
            else:
                far_mass = 0.0
            far_mass_per_layer.append(far_mass)

            # Per-layer attention entropy
            ent = -(w * torch.log(w + 1e-10)).sum(dim=-1).mean().item()
            attn_entropy_per_layer.append(ent)

        avg_far_mass = float(np.mean(far_mass_per_layer))
        max_far_mass = float(np.max(far_mass_per_layer))
        avg_attn_entropy = float(np.mean(attn_entropy_per_layer))

        # Token-level features
        logits_t = out.logits[0, 0, :]  # [vocab]
        probs = F.softmax(logits_t.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        resid_norm = logits_t.float().norm().item()

        # Logit margin (top1 - top2)
        top2 = torch.topk(logits_t.float(), 2).values
        margin = (top2[0] - top2[1]).item()

        # Additional cheap features
        top5_vals = torch.topk(probs, 5).values
        top5_mass = top5_vals.sum().item()
        step_pos = step / max(decode_steps - 1, 1)

        # Attention-derived features (need attn maps)
        far_mass_std = float(np.std(far_mass_per_layer))
        attn_entropy_std = float(np.std(attn_entropy_per_layer))

        feat = {
            "entropy": entropy,
            "resid_norm": resid_norm,
            "logit_margin": margin,
            "top5_mass": top5_mass,
            "step_pos": step_pos,
            "avg_attn_entropy": avg_attn_entropy,
            "attn_entropy_std": attn_entropy_std,
            "avg_far_mass": avg_far_mass,
            "max_far_mass": max_far_mass,
            "far_mass_std": far_mass_std,
        }
        features.append(feat)

        # Store raw mass; labeling done after collecting all
        labels.append(avg_far_mass)

    return features, labels


def _fit_logreg(X, y, n_epochs=300, lr=0.1):
    """Fit logistic regression and return AUC + confusion metrics."""
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    n_features = X_norm.shape[1]
    w = np.zeros(n_features, dtype=np.float32)
    b = 0.0

    for _epoch in range(n_epochs):
        logits = X_norm @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad_w = X_norm.T @ (probs - y) / len(y)
        grad_b = (probs - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b

    logits = X_norm @ w + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    # ROC-AUC via trapezoidal
    order = np.argsort(-probs)
    y_sorted = y[order]
    tp = fp = 0
    auc = 0.0
    tp_prev = fp_prev = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp + tp_prev) / 2.0
        tp_prev = tp
        fp_prev = fp
    auc = auc / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5

    # Confusion at threshold=0.5
    preds = (probs > 0.5).astype(int)
    tp_cm = int(((preds == 1) & (y == 1)).sum())
    fp_cm = int(((preds == 1) & (y == 0)).sum())
    fn_cm = int(((preds == 0) & (y == 1)).sum())
    tn_cm = int(((preds == 0) & (y == 0)).sum())
    precision = tp_cm / (tp_cm + fp_cm) if (tp_cm + fp_cm) > 0 else 0
    recall = tp_cm / (tp_cm + fn_cm) if (tp_cm + fn_cm) > 0 else 0

    return {
        "roc_auc": round(float(auc), 4),
        "confusion": {"tp": tp_cm, "fp": fp_cm, "fn": fn_cm, "tn": tn_cm},
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "weights": w.tolist(),
        "bias": float(b),
        "feature_mean": X_mean.tolist(),
        "feature_std": X_std.tolist(),
    }


def train_retrieval_predictor(features_list, labels_list):
    """Train retrieval predictors with multiple feature sets.

    Tests cheap-only features (no attention maps needed at runtime)
    and all features (including attention-derived) to diagnose whether
    the limitation is in features or the prediction task itself.

    Returns dict with results for each feature set.
    """
    # Flatten all data
    all_feats = []
    y_rows = []
    for feats, labs in zip(features_list, labels_list):
        for f, l in zip(feats, labs):
            all_feats.append(f)
            y_rows.append(l)
    y = np.array(y_rows, dtype=np.float32)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(
        f"  Training data: {len(y)} samples, "
        f"{n_pos} positive ({100*n_pos/len(y):.1f}%), "
        f"{n_neg} negative"
    )

    if n_pos == 0 or n_neg == 0:
        print("  WARNING: Only one class present — cannot train predictor")
        return {
            "status": "FAIL",
            "reason": "single_class",
            "n_pos": n_pos,
            "n_neg": n_neg,
            "roc_auc": 0.5,
        }

    # Feature sets to test
    feature_sets = {
        "cheap": ["entropy", "resid_norm", "logit_margin", "top5_mass", "step_pos"],
        "attn": ["avg_attn_entropy", "attn_entropy_std"],
        "all": [
            "entropy",
            "resid_norm",
            "logit_margin",
            "top5_mass",
            "step_pos",
            "avg_attn_entropy",
            "attn_entropy_std",
            "far_mass_std",
        ],
    }

    results = {"n_samples": len(y), "n_pos": n_pos, "n_neg": n_neg}

    for set_name, feat_keys in feature_sets.items():
        X = np.array([[f[k] for k in feat_keys] for f in all_feats], dtype=np.float32)
        res = _fit_logreg(X, y)
        res["feature_names"] = feat_keys
        results[set_name] = res
        print(
            f"  {set_name:6s} features ({len(feat_keys)}): "
            f"AUC={res['roc_auc']:.4f} "
            f"P={res['precision']:.3f} R={res['recall']:.3f}"
        )

    # Best AUC for summary
    best_set = max(feature_sets.keys(), key=lambda k: results[k]["roc_auc"])
    results["best_set"] = best_set
    results["roc_auc"] = results[best_set]["roc_auc"]
    results["status"] = "OK"
    return results


def cmd_phase2(args):
    """Phase 2: Retrieval-grounded predictor."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    # Phase 2 needs eager attention to capture attention weights
    print(f"Loading model {args.model} (eager attention)...")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model)
    max_ctx = getattr(config, "max_position_embeddings", None) or getattr(
        config, "n_positions", 1024
    )
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads if hidden and n_heads else None
    model_config = {
        "n_layers": n_layers,
        "hidden_size": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", 50257),
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPE, attn_implementation="eager"
    )
    model = model.to(device_str)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(
        f"  max_ctx={max_ctx} layers={n_layers} heads={n_heads} kv_heads={n_kv_heads}"
    )

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    all_features = []
    all_labels = []
    trace_results = []

    for L in seq_lens:
        for seed in seeds:
            W_min = max(64, L // 4)
            print(
                f"  Recording attention L={L} seed={seed} W_min={W_min}...",
                end="",
                flush=True,
            )
            feats, labs = record_attention_traces(
                model,
                token_data,
                L,
                args.steps,
                seed,
                device_str,
                model_config,
                max_ctx,
                W_min=W_min,
            )
            all_features.append(feats)
            all_labels.append(labs)  # raw far_mass values

            avg_far = np.mean(labs)
            print(f" avg_far_mass={avg_far:.3f}")
            trace_results.append(
                {
                    "L": L,
                    "seed": seed,
                    "n_steps": len(labs),
                    "avg_far_mass": round(avg_far, 4),
                }
            )

    # Apply threshold: use p75 of far_mass as threshold for "retrieval required"
    all_masses = []
    for labs in all_labels:
        all_masses.extend(labs)
    all_masses = np.array(all_masses)
    threshold = float(np.percentile(all_masses, 75))
    print(f"\n  Far mass threshold (p75): {threshold:.4f}")
    print(
        f"  Distribution: min={all_masses.min():.3f}"
        f" p25={np.percentile(all_masses,25):.3f}"
        f" p50={np.percentile(all_masses,50):.3f}"
        f" p75={np.percentile(all_masses,75):.3f}"
        f" max={all_masses.max():.3f}"
    )

    # Convert raw far_mass to binary labels
    binary_labels = []
    for labs in all_labels:
        binary_labels.append([1 if m > threshold else 0 for m in labs])

    n_pos = sum(sum(bl) for bl in binary_labels)
    n_total = sum(len(bl) for bl in binary_labels)
    print(f"  Positive: {n_pos}/{n_total}" f" ({100*n_pos/n_total:.1f}%)")

    for tr, labs in zip(trace_results, binary_labels):
        tr["n_retrieval"] = sum(labs)
        tr["pct_retrieval"] = round(100 * sum(labs) / len(labs), 1)

    # Train predictor
    print("\n=== Training retrieval predictor ===")
    predictor = train_retrieval_predictor(all_features, binary_labels)
    predictor["far_mass_threshold"] = round(threshold, 4)

    def json_safe(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            val = json_safe(obj)
            if val is not obj:
                return val
            return super().default(obj)

    ensure_dir(args.output_dir)
    pred_path = os.path.join(args.output_dir, "retrieval_predictor.json")
    with open(pred_path, "w") as f:
        json.dump(predictor, f, indent=2, cls=NumpyEncoder)
    print(f"Saved predictor: {pred_path}")

    traces_path = os.path.join(args.output_dir, "attention_traces.json")
    with open(traces_path, "w") as f:
        json.dump(trace_results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved traces: {traces_path}")

    # Also dump raw features for analysis
    raw_path = os.path.join(args.output_dir, "attention_features_raw.json")
    raw = []
    for feats, labs in zip(all_features, binary_labels):
        for f, l in zip(feats, labs):
            row = dict(f)
            row["retrieval_required"] = int(l)
            raw.append(row)
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, cls=NumpyEncoder)
    print(f"Saved raw features: {raw_path}")


# ============================================================
# Phase 3: Scaling exponent
# ============================================================


def cmd_phase3(args):
    """Phase 3: Compute scaling exponent beta."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    tol = args.tol

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    # Load tuned configs if available, else run quick tuning
    config_dir = os.path.join(args.config_dir, "selected_config_v11")
    tuned = {}
    for L in seq_lens:
        cfg_path = os.path.join(config_dir, f"L{L}_tol{tol}.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                tuned[L] = json.load(f)
            print(f"  Loaded tuned config for L={L}")

    # Get dense baselines
    print("\n=== Dense baselines ===")
    dense_ppls = {}
    for L in seq_lens:
        if L > max_ctx:
            continue
        ppls = []
        for seed in seeds:
            r = run_dense_decode(
                model,
                token_data,
                L,
                args.steps,
                seed,
                device_str,
                model_config,
                max_ctx,
            )
            ppls.append(r.ppl)
            print(f"  Dense L={L} seed={seed}: PPL={r.ppl:.1f}")
        dense_ppls[L] = float(np.mean(ppls))

    # Run BPA with tuned configs and measure kept tokens
    print("\n=== BPA kept tokens ===")
    kept_data = {}
    for L in seq_lens:
        if L > max_ctx:
            continue
        if L in tuned and tuned[L].get("status") == "PASS":
            p = tuned[L]["params"]
            bpa_kw = dict(
                W_min=p["W_min"],
                W_max=p["W_max"],
                W_pressure_thresh=p["W_pressure_thresh"],
                W_decay=p["W_decay"],
                gate_every_k=p["gate_every_k"],
                B_far_target=p["B_far_target"],
                B_far_max=p["B_far_max"],
            )
        else:
            # Use conservative defaults
            bpa_kw = dict(
                W_min=max(64, int(L * 0.8)),
                W_max=max(128, int(L * 0.95)),
                B_far_target=4.0,
                B_far_max=8,
            )
            print(f"  No tuned config for L={L}, using conservative defaults")

        kepts = []
        for seed in seeds:
            print(
                f"  BPA L={L} seed={seed}...",
                end="",
                flush=True,
            )
            r = run_bpa_decode(
                model,
                token_data,
                L,
                args.steps,
                seed,
                device_str,
                model_config,
                max_ctx,
                **bpa_kw,
            )
            kepts.append(r.kv_kept_mean)
            print(
                f" PPL={r.ppl:.1f} kept={r.kv_kept_mean:.0f}"
                f" ({100*r.kv_kept_mean/L:.0f}%)"
            )
        kept_data[L] = {
            "mean_kept": float(np.mean(kepts)),
            "std_kept": float(np.std(kepts)),
            "dense_ppl": dense_ppls.get(L, 0),
        }

    # Fit scaling exponent: kept ~ L^beta
    # log(kept) = beta * log(L) + c
    print("\n=== Scaling Exponent ===")
    Ls = sorted(kept_data.keys())
    if len(Ls) >= 2:
        log_L = np.array([math.log(L) for L in Ls])
        log_kept = np.array([math.log(kept_data[L]["mean_kept"]) for L in Ls])

        # Least squares fit
        A = np.vstack([log_L, np.ones(len(log_L))]).T
        result = np.linalg.lstsq(A, log_kept, rcond=None)
        beta = float(result[0][0])
        intercept = float(result[0][1])

        # R-squared
        y_pred = beta * log_L + intercept
        ss_res = ((log_kept - y_pred) ** 2).sum()
        ss_tot = ((log_kept - log_kept.mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  beta = {beta:.4f}")
        print(f"  R^2  = {r_squared:.4f}")
        print(f"  Interpretation:")
        if beta < 0.85:
            print("    STRONG sublinear evidence")
        elif beta < 0.95:
            print("    MARGINAL sublinear")
        else:
            print("    LINEAR — local window dominates")
    else:
        beta = 1.0
        r_squared = 0.0
        intercept = 0.0
        print("  Not enough L values to fit exponent")

    # Print table
    print(f"\n  {'L':>6s}  {'kept':>8s}  {'% of L':>7s}  {'dense PPL':>10s}")
    for L in Ls:
        kept = kept_data[L]["mean_kept"]
        pct = 100 * kept / L
        dppl = kept_data[L]["dense_ppl"]
        print(f"  {L:6d}  {kept:8.0f}  {pct:6.1f}%  {dppl:10.1f}")

    scaling_result = {
        "beta": round(beta, 4),
        "r_squared": round(r_squared, 4),
        "intercept": round(intercept, 4),
        "tol_pct": tol,
        "per_L": {str(L): kept_data[L] for L in Ls},
    }

    ensure_dir(args.output_dir)
    path = os.path.join(args.output_dir, "scaling_exponent.json")
    with open(path, "w") as f:
        json.dump(scaling_result, f, indent=2)
    print(f"\nSaved: {path}")


# ============================================================
# Phase 4: Layer-adaptive W
# ============================================================


@torch.no_grad()
def run_layer_adaptive_bpa_decode(
    model,
    token_data,
    seq_len,
    decode_steps,
    seed,
    device_str,
    model_config,
    max_ctx,
    W_min_per_layer,
    W_max_per_layer,
    gate_every_k=4,
    B_far_target=4.0,
    B_far_max=8,
    batch_size=1,
):
    """BPA decode with per-layer W_min/W_max.

    Each layer gets its own local window size.
    """
    n_layers = model_config["n_layers"]
    rng = np.random.RandomState(seed)
    idx = get_text_batch(token_data, batch_size, seq_len + decode_steps, rng)
    idx = idx.to(device_str)
    prefix = idx[:, :seq_len]
    continuation = idx[:, seq_len : seq_len + decode_steps]

    rss_before = get_cpu_rss_mb()

    # Warmup
    if device_str != "cpu":
        with torch.no_grad():
            _ = model(prefix[:, :16])
        gpu_sync(device_str)

    reset_gpu_mem(device_str)

    # Prefill
    t0 = time.perf_counter()
    out = model(prefix, use_cache=True)
    past = out.past_key_values
    gpu_sync(device_str)
    prefill_ms = (time.perf_counter() - t0) * 1000

    decode_latencies = []
    all_logits = [out.logits[:, -1:, :]]
    kept_tokens_log = []
    sel_rng = np.random.RandomState(seed + 1000)
    actual_pos = seq_len
    has_evicted = False

    for step in range(decode_steps):
        next_token = continuation[:, step : step + 1]

        # Gate update with per-layer eviction
        if step % gate_every_k == 0:
            cache_len = kv_cache_len(past)

            # Compute per-layer keep masks and evict
            # Use the most aggressive layer's W to determine
            # global eviction (since HF DynamicCache has same
            # seq_len across layers, we use uniform cache len)
            # We evict using the MINIMUM W across layers
            min_W = min(W_max_per_layer)
            if cache_len > min_W:
                # Per-layer eviction: each layer keeps its own W
                new_cache = DynamicCache()
                for layer_idx in range(n_layers):
                    k, v = past[layer_idx]
                    W_layer = W_max_per_layer[layer_idx]
                    if cache_len > W_layer:
                        # Keep last W_layer tokens + far chunks
                        n_chunks = (cache_len + CHUNK_SIZE - 1) // CHUNK_SIZE
                        far_end = max(0, (cache_len - W_layer) // CHUNK_SIZE)
                        far_chunks = select_far_chunks_random(
                            n_chunks,
                            far_end,
                            int(B_far_target),
                            sel_rng,
                        )
                        mask = build_keep_mask(
                            cache_len, W_layer, far_chunks, CHUNK_SIZE
                        )
                        indices = mask.to(device_str).nonzero(as_tuple=True)[0]
                        k_new = k[:, :, indices, :]
                        v_new = v[:, :, indices, :]
                    else:
                        k_new = k
                        v_new = v
                    new_cache.update(k_new, v_new, layer_idx)
                past = new_cache
                has_evicted = True

        kept_tokens_log.append(kv_cache_len(past))

        pos_ids = None
        if has_evicted:
            pos_ids = torch.tensor([[actual_pos]], device=device_str, dtype=torch.long)

        gpu_sync(device_str)
        t0 = time.perf_counter()
        out = model(
            next_token,
            past_key_values=past,
            use_cache=True,
            position_ids=pos_ids,
        )
        gpu_sync(device_str)
        dt = (time.perf_counter() - t0) * 1000
        decode_latencies.append(dt)
        past = out.past_key_values
        all_logits.append(out.logits)
        actual_pos += 1

    all_logits_cat = torch.cat(all_logits, dim=1)
    ppl = compute_ppl(all_logits_cat[:, :-1, :], continuation)

    kv_kept_mean = float(np.mean(kept_tokens_log)) if kept_tokens_log else 0
    kv_bpt = kv_bytes_per_token(model_config)
    kv_mb = kv_kept_mean * kv_bpt / 1e6
    dense_kept = seq_len + decode_steps / 2
    kv_ratio = kv_kept_mean / dense_kept if dense_kept > 0 else 1.0

    rss_after = get_cpu_rss_mb()
    peak_cpu_rss = max(rss_after, rss_before)
    gpu_alloc, gpu_reserved = get_gpu_mem(device_str)

    decode_arr = np.array(decode_latencies)
    total_decode_ms = sum(decode_latencies)

    del past, out
    torch.cuda.empty_cache()

    return DecodeResult(
        method="layer_adaptive_bpa",
        seq_len=seq_len,
        decode_steps=decode_steps,
        seed=seed,
        context_regime=context_regime(seq_len, max_ctx),
        prefill_ms=prefill_ms,
        decode_per_token_ms=float(np.median(decode_arr)),
        decode_p95_ms=float(np.percentile(decode_arr, 95)),
        gate_pct_of_total=0.0,
        throughput_toks_per_sec=(
            decode_steps / (total_decode_ms / 1000) if total_decode_ms > 0 else 0
        ),
        ppl=ppl,
        kv_kept_mean=kv_kept_mean,
        kv_mb_per_tok=kv_mb,
        kv_ratio=kv_ratio,
        peak_cpu_rss_mb=peak_cpu_rss,
        peak_gpu_alloc_mb=gpu_alloc,
        peak_gpu_reserved_mb=gpu_reserved,
        batch_size=batch_size,
    )


def make_layer_schedule(n_layers, L, profile="gradient"):
    """Generate per-layer W_min/W_max schedules.

    Profiles:
    - 'gradient': early=small, middle=medium, late=large
    - 'uniform': same W for all layers (baseline)
    """
    if profile == "uniform":
        W_max = max(128, int(L * 0.95))
        return [W_max] * n_layers

    # 'gradient': linear from 0.5*L to 0.95*L
    W_vals = []
    for i in range(n_layers):
        frac = i / max(n_layers - 1, 1)
        # Early layers: 0.5*L, Late layers: 0.95*L
        W_frac = 0.5 + 0.45 * frac
        W = max(64, int(L * W_frac))
        W_vals.append(W)
    return W_vals


def cmd_phase4(args):
    """Phase 4: Layer-adaptive W experiment."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)
    n_layers = model_config["n_layers"]

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    results = []
    for L in seq_lens:
        if L > max_ctx:
            continue

        for profile in ["uniform", "gradient"]:
            W_schedule = make_layer_schedule(n_layers, L, profile)
            avg_W = np.mean(W_schedule)
            print(
                f"\n  L={L} profile={profile}"
                f" avg_W={avg_W:.0f}"
                f" range=[{min(W_schedule)},{max(W_schedule)}]"
            )

            for seed in seeds:
                print(
                    f"    seed={seed}...",
                    end="",
                    flush=True,
                )
                r = run_layer_adaptive_bpa_decode(
                    model,
                    token_data,
                    L,
                    args.steps,
                    seed,
                    device_str,
                    model_config,
                    max_ctx,
                    W_min_per_layer=[max(64, w // 2) for w in W_schedule],
                    W_max_per_layer=W_schedule,
                    gate_every_k=4,
                    B_far_target=4.0,
                    B_far_max=8,
                )
                r.method = f"layer_bpa_{profile}"
                r.bpa_params = {
                    "profile": profile,
                    "W_schedule": W_schedule,
                    "avg_W": round(avg_W, 1),
                }
                results.append(r)
                print(
                    f" PPL={r.ppl:.1f} kept={r.kv_kept_mean:.0f}"
                    f" p50={r.decode_per_token_ms:.2f}ms"
                )

        # Also run dense for comparison
        for seed in seeds:
            print(f"    dense seed={seed}...", end="", flush=True)
            r = run_dense_decode(
                model,
                token_data,
                L,
                args.steps,
                seed,
                device_str,
                model_config,
                max_ctx,
            )
            results.append(r)
            print(f" PPL={r.ppl:.1f}")

    # Summary
    print("\n=== Layer-Adaptive Comparison ===")
    by_L = {}
    for r in results:
        L = r.seq_len
        if L not in by_L:
            by_L[L] = {}
        method = r.method
        if method not in by_L[L]:
            by_L[L][method] = []
        by_L[L][method].append(r)

    comparison = []
    for L in sorted(by_L.keys()):
        print(f"\n  L={L}:")
        for method in sorted(by_L[L].keys()):
            rs = by_L[L][method]
            ppls = [r.ppl for r in rs]
            kepts = [r.kv_kept_mean for r in rs]
            p50s = [r.decode_per_token_ms for r in rs]
            entry = {
                "L": L,
                "method": method,
                "ppl_mean": round(float(np.mean(ppls)), 2),
                "kept_mean": round(float(np.mean(kepts)), 1),
                "p50_mean_ms": round(float(np.mean(p50s)), 3),
            }
            comparison.append(entry)
            print(
                f"    {method:25s}: PPL={entry['ppl_mean']:8.1f}"
                f" kept={entry['kept_mean']:8.0f}"
                f" p50={entry['p50_mean_ms']:.2f}ms"
            )

    ensure_dir(args.output_dir)
    save_all_results(results, args.output_dir + "_phase4")

    comp_path = os.path.join(args.output_dir, "layer_adaptive_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {comp_path}")


# ============================================================
# Phase 5: 32K stress
# ============================================================


def cmd_phase5(args):
    """Phase 5: 32K context stress test."""
    device_str = args.device
    gpu_preflight(device_str)

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)

    if max_ctx < 32768:
        print(f"SKIP: model max_ctx={max_ctx} < 32768")
        return

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    L = 32768
    decode_steps = min(args.steps, 64)  # keep short for 32K
    seeds = [1]

    results = []

    # Dense baseline at 32K
    print(f"\n=== Dense L={L} ===")
    for seed in seeds:
        print(f"  dense seed={seed}...", end="", flush=True)
        try:
            r = run_dense_decode(
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device_str,
                model_config,
                max_ctx,
            )
            results.append(r)
            print(
                f" PPL={r.ppl:.1f} p50={r.decode_per_token_ms:.2f}ms"
                f" gpu={r.peak_gpu_alloc_mb:.0f}MB"
            )
        except Exception as e:
            print(f" OOM/ERROR: {e}")
            torch.cuda.empty_cache()
            results.append(
                DecodeResult(
                    method="dense",
                    seq_len=L,
                    decode_steps=decode_steps,
                    seed=seed,
                    context_regime="in_range",
                    prefill_ms=0,
                    decode_per_token_ms=0,
                    decode_p95_ms=0,
                    gate_pct_of_total=0,
                    throughput_toks_per_sec=0,
                    ppl=0,
                    kv_kept_mean=0,
                    kv_mb_per_tok=0,
                    kv_ratio=0,
                    peak_cpu_rss_mb=0,
                    peak_gpu_alloc_mb=0,
                    peak_gpu_reserved_mb=0,
                    quality_failed=True,
                )
            )

    # BPA at 32K with conservative config
    print(f"\n=== BPA L={L} ===")
    W_min = int(L * 0.9)
    W_max = int(L * 0.95)
    for seed in seeds:
        print(
            f"  bpa seed={seed} W=[{W_min},{W_max}]...",
            end="",
            flush=True,
        )
        try:
            r = run_bpa_decode(
                model,
                token_data,
                L,
                decode_steps,
                seed,
                device_str,
                model_config,
                max_ctx,
                W_min=W_min,
                W_max=W_max,
                B_far_target=4.0,
                B_far_max=8,
            )
            results.append(r)
            print(
                f" PPL={r.ppl:.1f} p50={r.decode_per_token_ms:.2f}ms"
                f" kept={r.kv_kept_mean:.0f}"
                f" gpu={r.peak_gpu_alloc_mb:.0f}MB"
            )
        except Exception as e:
            print(f" OOM/ERROR: {e}")
            torch.cuda.empty_cache()

    ensure_dir(args.output_dir)
    save_all_results(results, args.output_dir + "_phase5")

    summary = []
    for r in results:
        summary.append(
            {
                "method": r.method,
                "L": r.seq_len,
                "ppl": round(r.ppl, 2),
                "p50_ms": round(r.decode_per_token_ms, 3),
                "kept": round(r.kv_kept_mean, 0),
                "gpu_mb": round(r.peak_gpu_alloc_mb, 0),
                "failed": r.quality_failed,
            }
        )
    path = os.path.join(args.output_dir, "phase5_32k_stress.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {path}")


# ============================================================
# Matched-quality tuning (extended from v11)
# ============================================================


def cmd_alltune(args):
    """Run matched-quality tuning for extended L set."""
    device_str = args.device
    gpu_preflight(device_str)

    seq_lens = [int(x) for x in args.L.split(",")]
    tols = [float(x) for x in args.tol.split(",")]

    print(f"Loading model {args.model}...")
    model, tokenizer, max_ctx, model_config = load_hf_model(args.model, device_str)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)

    # Import tuning functions from v11
    from scripts.bpa_v11_bench import run_tuning, tune_static_sparse

    # Dense baselines
    dense_ppls = {}
    for L in seq_lens:
        if L > max_ctx:
            continue
        ppls = []
        for s in [1, 2]:
            r = run_dense_decode(
                model,
                token_data,
                L,
                args.steps,
                s,
                device_str,
                model_config,
                max_ctx,
            )
            ppls.append(r.ppl)
            print(f"  Dense L={L} seed={s}: PPL={r.ppl:.1f}")
        dense_ppls[L] = float(np.mean(ppls))
        print(f"  Dense L={L} avg PPL={dense_ppls[L]:.1f}")

    # Tune BPA
    all_tuning = []
    for L in seq_lens:
        if L > max_ctx or L not in dense_ppls:
            continue
        for tol in tols:
            print(f"\n=== Tuning BPA L={L} tol={tol}% ===")
            sel = run_tuning(
                model,
                token_data,
                L,
                args.steps,
                tol,
                device_str,
                model_config,
                max_ctx,
                dense_ppls[L],
                args.output_dir,
            )
            all_tuning.append(sel)

            print(f"  Tuning static_sparse L={L} tol={tol}%...")
            k_far_mean = 2.0
            if sel.get("status") == "PASS" and "params" in sel:
                k_far_mean = sel["params"].get("B_far_target", 2.0)
            ss = tune_static_sparse(
                model,
                token_data,
                L,
                args.steps,
                tol,
                device_str,
                model_config,
                max_ctx,
                dense_ppls[L],
                k_far_mean,
                args.output_dir,
            )
            all_tuning.append(ss)

    ensure_dir(args.output_dir)
    summary_path = os.path.join(args.output_dir, "tuning_summary_v12.json")
    with open(summary_path, "w") as f:
        json.dump(all_tuning, f, indent=2)
    print(f"\nSaved tuning summary: {summary_path}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v12 Frontier Stress")
    sub = parser.add_subparsers(dest="command")

    # Phase 0
    p0 = sub.add_parser("phase0")
    p0.add_argument("--model", default=DEFAULT_MODEL)
    p0.add_argument("--device", default="cuda")
    p0.add_argument("--output-dir", default=OUTPUT_DIR)

    # Phase 1
    p1 = sub.add_parser("phase1")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--L", default="1024,2048,4096,8192,16384")
    p1.add_argument("--steps", type=int, default=256)
    p1.add_argument("--seeds", default="1,2,3")
    p1.add_argument("--batch-sizes", default="1,4,8")
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--output-dir", default=OUTPUT_DIR)

    # Phase 2
    p2 = sub.add_parser("phase2")
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--L", default="1024,2048,4096")
    p2.add_argument("--steps", type=int, default=128)
    p2.add_argument("--seeds", default="1,2,3")
    p2.add_argument("--device", default="cuda")
    p2.add_argument("--output-dir", default=OUTPUT_DIR)

    # Phase 3
    p3 = sub.add_parser("phase3")
    p3.add_argument("--model", default=DEFAULT_MODEL)
    p3.add_argument("--L", default="1024,2048,4096,8192,16384")
    p3.add_argument("--steps", type=int, default=256)
    p3.add_argument("--seeds", default="1,2")
    p3.add_argument("--tol", type=float, default=1.0)
    p3.add_argument("--config-dir", default="bpa_v11_results")
    p3.add_argument("--device", default="cuda")
    p3.add_argument("--output-dir", default=OUTPUT_DIR)

    # Phase 4
    p4 = sub.add_parser("phase4")
    p4.add_argument("--model", default=DEFAULT_MODEL)
    p4.add_argument("--L", default="1024,2048,4096")
    p4.add_argument("--steps", type=int, default=128)
    p4.add_argument("--seeds", default="1,2")
    p4.add_argument("--device", default="cuda")
    p4.add_argument("--output-dir", default=OUTPUT_DIR)

    # Phase 5
    p5 = sub.add_parser("phase5")
    p5.add_argument("--model", default=DEFAULT_MODEL)
    p5.add_argument("--steps", type=int, default=64)
    p5.add_argument("--device", default="cuda")
    p5.add_argument("--output-dir", default=OUTPUT_DIR)

    # alltune
    pt = sub.add_parser("alltune")
    pt.add_argument("--model", default=DEFAULT_MODEL)
    pt.add_argument("--L", default="1024,2048,4096,8192,16384")
    pt.add_argument("--tol", default="1,3")
    pt.add_argument("--steps", type=int, default=256)
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--output-dir", default=OUTPUT_DIR)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "phase0": cmd_phase0,
        "phase1": cmd_phase1,
        "phase2": cmd_phase2,
        "phase3": cmd_phase3,
        "phase4": cmd_phase4,
        "phase5": cmd_phase5,
        "alltune": cmd_alltune,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
