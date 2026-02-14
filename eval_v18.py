#!/usr/bin/env python
"""
BPA v18 Evaluation: Bitter-Lesson Bit Allocation via Adam v-hat.

Phases:
  0: Baseline lock (dense, INT8, S2 manual)
  1: Adam v-hat sensitivity extraction
  2: Greedy bit allocation policy
  3: Correlation analysis (empirical vs adam)
  4: 7B/8B replication
  5: Bandwidth-bound regime

Usage:
    python eval_v18.py --phase 0
    python eval_v18.py --phase 1
    python eval_v18.py --phase 2
    python eval_v18.py --phase 3
    python eval_v18.py --phase 4
    python eval_v18.py --phase 5
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from scripts.bpa_v11_bench import (
    DTYPE,
    compute_ppl,
    get_cpu_rss_mb,
    get_gpu_mem,
    get_text_batch,
    gpu_sync,
    load_validation_tokens,
    reset_gpu_mem,
)
from backends.base import DenseBackend, V14StepStats
from eval_v15 import (
    V15Result,
    apply_quality_gating,
    build_scoreboard,
    gpu_preflight,
    load_model,
    run_single_eval,
)
from eval_v16 import (
    MixedPrecisionBackend,
    build_schedules,
    run_backend_sweep,
    run_dense_baselines,
    save_results,
)


# ============================================================
# Adam v-hat sensitivity extraction
# ============================================================


def extract_adam_vhat(model, token_data, device_str, n_steps=200, lr=1e-5):
    """Run short fine-tune to build Adam optimizer state, then
    extract v-hat (second moment estimate) per layer for KV params.

    Returns dict mapping layer_idx -> raw_vhat_score.
    """
    from transformers import get_linear_schedule_with_warmup

    model.train()
    # Only train KV projection parameters
    kv_params = []
    kv_param_names = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name:
            p.requires_grad = True
            kv_params.append(p)
            kv_param_names.append(name)
        else:
            p.requires_grad = False

    optimizer = torch.optim.Adam(kv_params, lr=lr, betas=(0.9, 0.999))

    print(f"  Fine-tuning {len(kv_params)} KV params for {n_steps} steps...")
    rng = np.random.RandomState(42)
    seq_len = 512
    total_len = seq_len + 1

    losses = []
    for step in range(n_steps):
        idx = get_text_batch(token_data, 1, total_len, rng).to(device_str)
        input_ids = idx[:, :seq_len]
        labels = idx[:, 1 : seq_len + 1]

        outputs = model(input_ids)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"    step {step + 1}/{n_steps}: loss={avg_loss:.4f}")

    # Extract v-hat from Adam optimizer state
    print("  Extracting v-hat from optimizer state...")
    layer_vhat = {}
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            state = optimizer.state[p]
            if "exp_avg_sq" not in state:
                continue
            vhat = state["exp_avg_sq"]
            # Find which layer this param belongs to
            idx_in_list = kv_params.index(p)
            pname = kv_param_names[idx_in_list]
            # Parse layer index from name like model.layers.0.self_attn.k_proj.weight
            parts = pname.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                continue

            mean_vhat = vhat.mean().item()
            if layer_idx not in layer_vhat:
                layer_vhat[layer_idx] = []
            layer_vhat[layer_idx].append(mean_vhat)

    # Average across K and V projections per layer
    raw_scores = {}
    for layer_idx in sorted(layer_vhat.keys()):
        raw_scores[layer_idx] = np.mean(layer_vhat[layer_idx])

    model.eval()
    # Reset requires_grad
    for p in model.parameters():
        p.requires_grad = False

    return raw_scores


def apply_root4_transform(raw_scores):
    """Apply 4th-root stabilization to raw v-hat scores."""
    root4_scores = {}
    for layer_idx, score in raw_scores.items():
        root4_scores[layer_idx] = float(score**0.25)

    # Normalize by max
    max_score = max(root4_scores.values()) if root4_scores else 1.0
    normalized = {}
    for layer_idx, score in root4_scores.items():
        normalized[layer_idx] = score / max_score

    return root4_scores, normalized


# ============================================================
# Greedy bit allocation
# ============================================================


def greedy_allocate(scores, n_layers):
    """Given per-layer scores, return layer indices sorted by
    descending sensitivity (most sensitive first)."""
    layer_order = sorted(scores.keys(), key=lambda l: scores[l], reverse=True)
    return layer_order


def build_schedule_from_scores(scores, n_layers, n_int8):
    """Build INT4/INT8 schedule: top n_int8 scored layers get INT8."""
    order = greedy_allocate(scores, n_layers)
    schedule = [4] * n_layers
    for i in range(min(n_int8, len(order))):
        schedule[order[i]] = 8
    return schedule


def build_random_schedule(n_layers, n_int8, seed=0):
    """Build random INT4/INT8 schedule with exactly n_int8 INT8 layers."""
    rng = np.random.RandomState(seed)
    int8_layers = set(rng.choice(n_layers, n_int8, replace=False))
    schedule = [8 if i in int8_layers else 4 for i in range(n_layers)]
    return schedule


# ============================================================
# Phase runners
# ============================================================


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Baseline lock."""
    outdir = os.path.join(args.outdir, "phase0")

    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    # INT8 everywhere
    from backends.quant import QuantBackend

    int8_be = QuantBackend()

    # S2 manual schedule from v16
    schedules = build_schedules(args.sensitivity_path)
    s2_sched = schedules["S2"]
    s2_be = MixedPrecisionBackend(layer_bits=s2_sched)
    s2_be._name = "S2_manual"

    backends = [
        ("INT8", int8_be),
        ("S2_manual", s2_be),
    ]
    comp_results = run_backend_sweep(
        backends,
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
    )

    all_results = dense_results + comp_results
    save_results(all_results, dense_ppls, outdir, 0, {"gpu_info": gpu_info})


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Adam v-hat sensitivity extraction."""
    outdir = os.path.join(args.outdir, "phase1")
    os.makedirs(outdir, exist_ok=True)
    sens_dir = os.path.join(args.outdir, "artifacts", "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 1: Adam v-hat sensitivity extraction")
    print("=" * 60)

    raw_scores = extract_adam_vhat(
        model, token_data, args.device, n_steps=args.calib_steps
    )
    root4_scores, normalized_scores = apply_root4_transform(raw_scores)

    # Print results
    print("\n  Layer sensitivity ranking (Adam v-hat, 4th-root, normalized):")
    for layer_idx in sorted(
        normalized_scores.keys(), key=lambda l: normalized_scores[l], reverse=True
    ):
        raw = raw_scores[layer_idx]
        r4 = root4_scores[layer_idx]
        norm = normalized_scores[layer_idx]
        print(f"    layer {layer_idx:2d}: raw={raw:.6f} root4={r4:.6f} norm={norm:.4f}")

    # Save artifacts
    raw_artifact = {
        "model": args.model,
        "n_steps": args.calib_steps,
        "lr": 1e-5,
        "scores": {str(k): v for k, v in raw_scores.items()},
    }
    with open(os.path.join(sens_dir, "adam_vhat_raw.json"), "w") as f:
        json.dump(raw_artifact, f, indent=2)

    root4_artifact = {
        "model": args.model,
        "scores_root4": {str(k): v for k, v in root4_scores.items()},
        "scores_normalized": {str(k): v for k, v in normalized_scores.items()},
        "ranking_sensitive_to_tolerant": sorted(
            normalized_scores.keys(),
            key=lambda l: normalized_scores[l],
            reverse=True,
        ),
    }
    with open(os.path.join(sens_dir, "adam_vhat_root4.json"), "w") as f:
        json.dump(root4_artifact, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "version": "v18",
        "gpu_info": gpu_info,
        "n_steps": args.calib_steps,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved sensitivity artifacts to {sens_dir}/")


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Greedy bit allocation policy."""
    outdir = os.path.join(args.outdir, "phase2")
    sched_dir = os.path.join(args.outdir, "artifacts", "bit_schedules")
    os.makedirs(sched_dir, exist_ok=True)

    # Load adam v-hat scores
    sens_dir = os.path.join(args.outdir, "artifacts", "sensitivity")
    root4_path = os.path.join(sens_dir, "adam_vhat_root4.json")
    with open(root4_path) as f:
        root4_data = json.load(f)
    adam_scores = {int(k): v for k, v in root4_data["scores_normalized"].items()}

    # Load empirical sensitivity (v16 oracle)
    with open(args.sensitivity_path) as f:
        emp_data = json.load(f)
    emp_ranked = emp_data["int4_ranked_tolerant_to_sensitive"]
    emp_scores = {}
    for entry in emp_ranked:
        # Higher delta = more sensitive = higher score
        emp_scores[entry["layer"]] = entry["ppl_delta_pct"]

    n_layers = model_config["n_layers"]

    print(f"\n{'=' * 60}")
    print("Phase 2: Greedy bit allocation")
    print("=" * 60)

    # Step 1: Search for minimal k (INT8 layers) using adam-root4
    print("\nSearching for minimal k (adam-root4)...")
    dense_results_quick, dense_ppls_quick = run_dense_baselines(
        model,
        token_data,
        [8192],
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    best_k_adam = None
    for k in range(0, n_layers + 1):
        sched = build_schedule_from_scores(adam_scores, n_layers, k)
        be = MixedPrecisionBackend(layer_bits=sched)
        be._name = f"adam_k{k}"

        results = run_backend_sweep(
            [(f"adam_k{k}", be)],
            model,
            token_data,
            [8192],
            args.decode_steps,
            args.seeds,
            args.device,
            max_ctx,
            model_config,
            dense_ppls_quick,
        )
        apply_quality_gating(results, dense_ppls_quick)
        n_pass = sum(1 for r in results if r.passed_3pct)
        n_total = len(results)
        n4 = sum(1 for b in sched if b == 4)
        print(f"    k={k}: {n_pass}/{n_total} @3%, {n4}xINT4 + {k}xINT8")
        if n_pass == n_total:
            best_k_adam = k
            break

    if best_k_adam is None:
        best_k_adam = n_layers
        print(f"  WARNING: No k found that passes, using all INT8")

    print(f"\n  Minimal k (adam-root4): {best_k_adam}")

    # Generate all schedules to compare
    adam_sched = build_schedule_from_scores(adam_scores, n_layers, best_k_adam)
    emp_sched = build_schedule_from_scores(emp_scores, n_layers, best_k_adam)
    random_sched = build_random_schedule(n_layers, best_k_adam, seed=42)

    # S2 manual from v16
    v16_schedules = build_schedules(args.sensitivity_path)
    s2_sched = v16_schedules["S2"]

    # Save schedules
    schedule_info = {
        "adam_root4": {
            "schedule": adam_sched,
            "k": best_k_adam,
            "int8_layers": [i for i, b in enumerate(adam_sched) if b == 8],
        },
        "empirical": {
            "schedule": emp_sched,
            "k": best_k_adam,
            "int8_layers": [i for i, b in enumerate(emp_sched) if b == 8],
        },
        "random": {
            "schedule": random_sched,
            "k": best_k_adam,
            "int8_layers": [i for i, b in enumerate(random_sched) if b == 8],
        },
        "S2_manual": {
            "schedule": s2_sched,
            "k": sum(1 for b in s2_sched if b == 8),
            "int8_layers": [i for i, b in enumerate(s2_sched) if b == 8],
        },
    }
    with open(os.path.join(sched_dir, "all_schedules.json"), "w") as f:
        json.dump(schedule_info, f, indent=2)

    # Step 2: Full evaluation of all schedules
    print(f"\n{'=' * 60}")
    print("Full evaluation of all schedules")
    print("=" * 60)

    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    backends = []
    for sname, sinfo in [
        ("adam_root4", {"sched": adam_sched}),
        ("empirical", {"sched": emp_sched}),
        ("random", {"sched": random_sched}),
        ("S2_manual", {"sched": s2_sched}),
    ]:
        be = MixedPrecisionBackend(layer_bits=sinfo["sched"])
        be._name = sname
        backends.append((sname, be))

    # Also test INT8 everywhere
    from backends.quant import QuantBackend

    int8_be = QuantBackend()
    backends.append(("INT8", int8_be))

    comp_results = run_backend_sweep(
        backends,
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
    )

    all_results = dense_results + comp_results
    save_results(
        all_results,
        dense_ppls,
        outdir,
        2,
        {
            "gpu_info": gpu_info,
            "best_k_adam": best_k_adam,
            "schedules": schedule_info,
        },
    )


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Correlation analysis."""
    from scipy.stats import spearmanr

    outdir = os.path.join(args.outdir, "phase3")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 3: Correlation analysis")
    print("=" * 60)

    # Load empirical sensitivity
    with open(args.sensitivity_path) as f:
        emp_data = json.load(f)
    emp_ranked = emp_data["int4_ranked_tolerant_to_sensitive"]
    emp_scores = {}
    for entry in emp_ranked:
        emp_scores[entry["layer"]] = entry["ppl_delta_pct"]

    # Load adam v-hat scores
    sens_dir = os.path.join(args.outdir, "artifacts", "sensitivity")
    raw_path = os.path.join(sens_dir, "adam_vhat_raw.json")
    root4_path = os.path.join(sens_dir, "adam_vhat_root4.json")

    with open(raw_path) as f:
        raw_data = json.load(f)
    raw_scores = {int(k): v for k, v in raw_data["scores"].items()}

    with open(root4_path) as f:
        root4_data = json.load(f)
    root4_scores = {int(k): v for k, v in root4_data["scores_root4"].items()}
    norm_scores = {int(k): v for k, v in root4_data["scores_normalized"].items()}

    # Align layers
    layers = sorted(set(emp_scores.keys()) & set(raw_scores.keys()))
    emp_vec = [emp_scores[l] for l in layers]
    raw_vec = [raw_scores[l] for l in layers]
    root4_vec = [root4_scores[l] for l in layers]
    norm_vec = [norm_scores[l] for l in layers]

    # Compute correlations
    corr_raw, p_raw = spearmanr(emp_vec, raw_vec)
    corr_root4, p_root4 = spearmanr(emp_vec, root4_vec)
    corr_norm, p_norm = spearmanr(emp_vec, norm_vec)

    print(f"\n  Spearman correlations (empirical INT4 delta vs Adam v-hat):")
    print(f"    raw v-hat:        rho={corr_raw:.4f}  p={p_raw:.4f}")
    print(f"    root4 v-hat:      rho={corr_root4:.4f}  p={p_root4:.4f}")
    print(f"    root4 normalized: rho={corr_norm:.4f}  p={p_norm:.4f}")

    # Per-layer comparison
    print(f"\n  Per-layer comparison (sorted by empirical sensitivity):")
    emp_order = sorted(layers, key=lambda l: emp_scores[l], reverse=True)
    adam_order = sorted(layers, key=lambda l: norm_scores[l], reverse=True)
    print(
        f"    {'Layer':>5s} {'Emp delta':>10s} {'Adam norm':>10s} {'Emp rank':>9s} {'Adam rank':>10s}"
    )
    for l in emp_order:
        emp_rank = emp_order.index(l) + 1
        adam_rank = adam_order.index(l) + 1
        print(
            f"    {l:5d} {emp_scores[l]:+10.3f}% {norm_scores[l]:10.4f} {emp_rank:9d} {adam_rank:10d}"
        )

    # Save
    correlations = {
        "method": "spearman",
        "n_layers": len(layers),
        "raw_vhat_vs_empirical": {
            "rho": round(corr_raw, 4),
            "p_value": round(p_raw, 4),
        },
        "root4_vhat_vs_empirical": {
            "rho": round(corr_root4, 4),
            "p_value": round(p_root4, 4),
        },
        "empirical_ranking": emp_order,
        "adam_root4_ranking": adam_order,
        "per_layer": {
            str(l): {
                "empirical_delta": emp_scores[l],
                "adam_norm": norm_scores[l],
                "adam_raw": raw_scores[l],
            }
            for l in layers
        },
    }
    with open(os.path.join(outdir, "correlations.json"), "w") as f:
        json.dump(correlations, f, indent=2)

    # Also save to artifacts
    sens_dir = os.path.join(args.outdir, "artifacts", "sensitivity")
    with open(os.path.join(sens_dir, "correlations.json"), "w") as f:
        json.dump(correlations, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "version": "v18",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved correlations to {outdir}/")


def run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 4: 7B/8B replication."""
    outdir = os.path.join(args.outdir, "phase4")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 4: 7B/8B replication")
    print("=" * 60)

    # Check available GPU memory
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU memory: {total_mem:.1f}GB")

    # Try loading Llama-3-8B or Qwen2.5-7B
    large_models = [
        ("Qwen/Qwen2.5-7B", "qwen7b"),
        ("meta-llama/Meta-Llama-3-8B", "llama3_8b"),
    ]

    model_8b = None
    model_name = None
    model_key = None
    for mpath, mkey in large_models:
        try:
            print(f"  Trying {mpath}...")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
            m = AutoModelForCausalLM.from_pretrained(
                mpath,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model_8b = m
            model_name = mpath
            model_key = mkey
            print(f"  Loaded {mpath} successfully")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if model_8b is None:
        print("  No 7B/8B model available. Flagging H100 requirement.")
        result = {
            "status": "SKIP",
            "reason": "No 7B/8B model loadable on W7900",
            "recommendation": "H100 required for A2 replication",
        }
        with open(os.path.join(outdir, "phase4_result.json"), "w") as f:
            json.dump(result, f, indent=2)

        meta = {
            "timestamp": datetime.now().isoformat(),
            "phase": 4,
            "version": "v18",
            "gpu_info": gpu_info,
            "status": "SKIP",
        }
        with open(os.path.join(outdir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nSaved Phase 4 skip result to {outdir}/")
        return

    # Get model config for 8B
    config_8b = {
        "n_layers": model_8b.config.num_hidden_layers,
        "n_heads": model_8b.config.num_attention_heads,
        "n_kv_heads": getattr(
            model_8b.config, "num_key_value_heads", model_8b.config.num_attention_heads
        ),
        "head_dim": model_8b.config.hidden_size // model_8b.config.num_attention_heads,
        "hidden_size": model_8b.config.hidden_size,
    }
    max_ctx_8b = getattr(model_8b.config, "max_position_embeddings", 32768)
    print(f"  Config: {config_8b}")
    print(f"  Max context: {max_ctx_8b}")

    # Load token data for this tokenizer
    print("  Loading validation data...")
    token_data_8b = load_validation_tokens(tok)

    # Run calibration to get Adam v-hat
    print("  Running Adam v-hat calibration on 8B model...")
    raw_scores_8b = extract_adam_vhat(
        model_8b,
        token_data_8b,
        args.device,
        n_steps=min(args.calib_steps, 100),
    )
    root4_8b, norm_8b = apply_root4_transform(raw_scores_8b)

    n_layers_8b = config_8b["n_layers"]

    # Search for minimal k at L=8192
    valid_L_8b = [L for L in [8192, 16384] if L <= max_ctx_8b]
    print(f"  Testing at L={valid_L_8b}")

    dense_results_8b, dense_ppls_8b = run_dense_baselines(
        model_8b,
        token_data_8b,
        [8192],
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx_8b,
        config_8b,
    )

    best_k_8b = None
    for k in range(0, n_layers_8b + 1, max(1, n_layers_8b // 8)):
        sched = build_schedule_from_scores(norm_8b, n_layers_8b, k)
        be = MixedPrecisionBackend(layer_bits=sched)
        be._name = f"adam_k{k}"

        try:
            results = run_backend_sweep(
                [(f"adam_k{k}", be)],
                model_8b,
                token_data_8b,
                [8192],
                args.decode_steps,
                args.seeds,
                args.device,
                max_ctx_8b,
                config_8b,
                dense_ppls_8b,
            )
            apply_quality_gating(results, dense_ppls_8b)
            n_pass = sum(1 for r in results if r.passed_3pct)
            n_total = len(results)
            print(f"    k={k}: {n_pass}/{n_total} @3%")
            if n_pass == n_total:
                best_k_8b = k
                break
        except Exception as e:
            print(f"    k={k}: OOM or error: {e}")
            break

    if best_k_8b is None:
        best_k_8b = n_layers_8b // 4
        print(f"  Using fallback k={best_k_8b}")

    # Full eval with best schedule
    adam_sched_8b = build_schedule_from_scores(norm_8b, n_layers_8b, best_k_8b)
    random_sched_8b = build_random_schedule(n_layers_8b, best_k_8b, seed=42)

    dense_results_full, dense_ppls_full = run_dense_baselines(
        model_8b,
        token_data_8b,
        valid_L_8b,
        args.decode_steps,
        [0, 1],  # fewer seeds for 8B to save time
        args.device,
        max_ctx_8b,
        config_8b,
    )

    adam_be = MixedPrecisionBackend(layer_bits=adam_sched_8b)
    adam_be._name = "adam_root4_8b"
    rand_be = MixedPrecisionBackend(layer_bits=random_sched_8b)
    rand_be._name = "random_8b"

    from backends.quant import QuantBackend

    int8_be = QuantBackend()

    backends_8b = [
        ("INT8_8b", int8_be),
        ("adam_root4_8b", adam_be),
        ("random_8b", rand_be),
    ]

    try:
        comp_results = run_backend_sweep(
            backends_8b,
            model_8b,
            token_data_8b,
            valid_L_8b,
            args.decode_steps,
            [0, 1],
            args.device,
            max_ctx_8b,
            config_8b,
            dense_ppls_full,
        )
        all_results_8b = dense_results_full + comp_results
        save_results(
            all_results_8b,
            dense_ppls_full,
            outdir,
            4,
            {
                "gpu_info": gpu_info,
                "model": model_name,
                "config": config_8b,
                "best_k": best_k_8b,
                "adam_schedule": adam_sched_8b,
            },
        )
    except Exception as e:
        print(f"  8B eval failed: {e}")
        result = {
            "status": "PARTIAL",
            "model": model_name,
            "best_k": best_k_8b,
            "error": str(e),
        }
        with open(os.path.join(outdir, "phase4_result.json"), "w") as f:
            json.dump(result, f, indent=2)

    # Cleanup
    del model_8b
    torch.cuda.empty_cache()


def run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 5: Bandwidth-bound regime testing."""
    outdir = os.path.join(args.outdir, "phase5")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Phase 5: Bandwidth-bound regime testing")
    print("=" * 60)

    # Load adam schedule
    sched_dir = os.path.join(args.outdir, "artifacts", "bit_schedules")
    sched_path = os.path.join(sched_dir, "all_schedules.json")
    if os.path.exists(sched_path):
        with open(sched_path) as f:
            sched_data = json.load(f)
        adam_sched = sched_data["adam_root4"]["schedule"]
    else:
        # Fall back to S2
        v16_scheds = build_schedules(args.sensitivity_path)
        adam_sched = v16_scheds["S2"]

    s2_scheds = build_schedules(args.sensitivity_path)
    s2_sched = s2_scheds["S2"]

    # Test multiple microbatch sizes
    microbatches = [1, 4, 8]
    L_test = 8192
    decode_steps = 256

    results = {}
    for mb in microbatches:
        print(f"\n  Microbatch={mb}, L={L_test}")

        rng = np.random.RandomState(0)
        total_len = L_test + decode_steps
        idx = get_text_batch(token_data, mb, total_len, rng).to(args.device)
        prefix = idx[:, :L_test]
        continuation = idx[:, L_test : L_test + decode_steps]

        dense_be = DenseBackend()
        dense_be.configure(L_test, model_config)

        from backends.quant import QuantBackend

        int8_be = QuantBackend()
        int8_be.configure(L_test, model_config)
        int8_be.calibrate(model, token_data, L_test, args.device, model_config)

        adam_be = MixedPrecisionBackend(layer_bits=adam_sched)
        adam_be._name = "adam_root4"
        adam_be.configure(L_test, model_config)
        adam_be.calibrate(model, token_data, L_test, args.device, model_config)

        s2_be = MixedPrecisionBackend(layer_bits=s2_sched)
        s2_be._name = "S2_manual"
        s2_be.configure(L_test, model_config)
        s2_be.calibrate(model, token_data, L_test, args.device, model_config)

        mb_results = {}
        for bname, be in [
            ("dense", dense_be),
            ("INT8", int8_be),
            ("adam_root4", adam_be),
            ("S2_manual", s2_be),
        ]:
            times = []
            for trial in range(3):
                torch.cuda.empty_cache()
                gpu_sync(args.device)
                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        logits, stats = be.run_decode(
                            model, prefix, continuation, args.device, max_ctx
                        )
                    gpu_sync(args.device)
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    del logits
                except Exception as e:
                    print(f"    {bname} mb={mb}: OOM ({e})")
                    break

            if times:
                avg_ms = np.mean(times) * 1000
                p50_per_tok = avg_ms / decode_steps
                mb_results[bname] = {
                    "total_ms": round(avg_ms, 1),
                    "p50_per_tok_ms": round(p50_per_tok, 2),
                    "trials": [round(t * 1000, 1) for t in times],
                }
                print(f"    {bname}: {avg_ms:.1f}ms total, {p50_per_tok:.2f}ms/tok")
            else:
                mb_results[bname] = {"status": "OOM"}
                print(f"    {bname}: OOM")

        results[f"mb{mb}"] = mb_results

    # Also test at L=16384 with mb=1 and mb=4
    for L_test2 in [16384, 32768]:
        if L_test2 > max_ctx:
            continue
        for mb in [1, 4]:
            key = f"L{L_test2}_mb{mb}"
            print(f"\n  L={L_test2}, mb={mb}")

            rng = np.random.RandomState(0)
            total_len = L_test2 + decode_steps
            try:
                idx = get_text_batch(token_data, mb, total_len, rng).to(args.device)
            except Exception:
                print(f"    Insufficient token data")
                continue

            prefix = idx[:, :L_test2]
            continuation = idx[:, L_test2 : L_test2 + decode_steps]

            dense_be = DenseBackend()
            dense_be.configure(L_test2, model_config)

            s2_be = MixedPrecisionBackend(layer_bits=s2_sched)
            s2_be._name = "S2_manual"
            s2_be.configure(L_test2, model_config)
            s2_be.calibrate(model, token_data, L_test2, args.device, model_config)

            mb_results = {}
            for bname, be in [("dense", dense_be), ("S2_manual", s2_be)]:
                times = []
                for trial in range(3):
                    torch.cuda.empty_cache()
                    gpu_sync(args.device)
                    t0 = time.perf_counter()
                    try:
                        with torch.no_grad():
                            logits, stats = be.run_decode(
                                model, prefix, continuation, args.device, max_ctx
                            )
                        gpu_sync(args.device)
                        elapsed = time.perf_counter() - t0
                        times.append(elapsed)
                        del logits
                    except Exception as e:
                        print(f"    {bname}: OOM ({e})")
                        break

                if times:
                    avg_ms = np.mean(times) * 1000
                    p50_per_tok = avg_ms / decode_steps
                    mb_results[bname] = {
                        "total_ms": round(avg_ms, 1),
                        "p50_per_tok_ms": round(p50_per_tok, 2),
                    }
                    print(f"    {bname}: {avg_ms:.1f}ms total, {p50_per_tok:.2f}ms/tok")
                else:
                    mb_results[bname] = {"status": "OOM"}
                    print(f"    {bname}: OOM")

            results[key] = mb_results

    with open(os.path.join(outdir, "bandwidth_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 5,
        "version": "v18",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved bandwidth results to {outdir}/")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v18")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--model", default="qwen05b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--L",
        nargs="+",
        type=int,
        default=[8192, 16384, 32768],
    )
    parser.add_argument("--outdir", default="results/v18")
    parser.add_argument(
        "--sensitivity_path",
        default="results/v15/phase4/layer_sensitivity.json",
    )
    parser.add_argument("--calib_steps", type=int, default=200)
    args = parser.parse_args()

    gpu_info = gpu_preflight(args.device)
    model, tokenizer, max_ctx, model_config = load_model(args.model, args.device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)
    valid_L = [L for L in args.L if L <= max_ctx]

    if args.phase == 0:
        run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 1:
        run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 2:
        run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 3:
        run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 4:
        run_phase4(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 5:
        run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
