#!/usr/bin/env python
"""
BPA v4 Phase 2: Online Evaluation with Budget-Calibrated Gate.

Evaluates the v4 gate at multiple enabled_rate budgets to find the
Pareto frontier of PPL vs KV savings at L=1024.

Variants:
  V0: Dense baseline (full causal attention)
  V1: BPA local-only (far-context disabled always)
  V2_30: V4 gate at 30% budget
  V2_40: V4 gate at 40% budget
  V2_50: V4 gate at 50% budget
  V2_60: V4 gate at 60% budget
  V2_70: V4 gate at 70% budget
  V3: BPA v3 gate (reference, fixed threshold=0.5)
  VR: Random gate matched to V2_50 enabled_rate

Usage:
    python scripts/bpa_v4_experiment.py \
        --checkpoint <path> \
        --gate-dir bpa_v4_gate_results \
        --seq-lens 512,1024 \
        --seeds 1,2,3 \
        --n-eval 50 \
        --output-dir bpa_v4_results
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA
from scripts.bpa_v3_experiment import (
    RunMetrics,
    build_local_mask,
    compute_gate_features,
    compute_ppl,
    evaluate_variant,
    get_text_batch,
    load_text_data,
    manual_forward,
    manual_forward_with_per_position_mask,
)
from scripts.bpa_v4_gate import V4Gate, find_budget_threshold
from utils.kv_accounting import compute_kv_accounting


def load_v4_gate(gate_dir: str, seed: int = 1):
    """Load a trained v4 gate and normalization stats."""
    # Try v4 best gate first
    ckpt_path = os.path.join(gate_dir, f"v4_best_gate_seed{seed}.pt")
    stats_path = os.path.join(gate_dir, f"v4_norm_stats_seed{seed}.npz")

    if not os.path.exists(ckpt_path):
        print(f"  WARNING: No v4 gate at {ckpt_path}")
        return None, None, None

    # Load gate results to determine architecture
    results_path = os.path.join(gate_dir, f"v4_gate_seed{seed}_detail.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            detail = json.load(f)
        n_features = len(detail["normalization"]["mean"])
    else:
        n_features = 7  # default

    gate = V4Gate(n_features, hidden=256, n_layers=3)
    gate.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    gate.eval()

    stats = np.load(stats_path)
    return gate, stats["mean"], stats["std"]


def load_v3_gate(data_dir: str, seed: int = 1):
    """Load/train v3-style gate for reference comparison."""
    from scripts.bpa_v3_experiment import train_quick_gate

    return train_quick_gate(data_dir, seed)


def evaluate_variant_budget(
    model: GPT2_BPA,
    text_data: np.ndarray,
    variant: str,
    seq_len: int,
    seed: int,
    n_eval: int,
    batch_size: int,
    local_window: int,
    gate=None,
    feat_mean=None,
    feat_std=None,
    target_enabled_rate: float = 0.5,
    is_random: bool = False,
) -> RunMetrics:
    """Evaluate with budget-calibrated thresholding.

    Instead of fixed threshold=0.5, uses quantile thresholding
    to hit the target_enabled_rate.
    """
    cfg = model.config
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    rng = np.random.RandomState(seed)

    ppls = []
    tokens_per_query_list = []
    enabled_rates = []
    wall_times = []
    tokens_seen = 0

    # Warmup
    warmup_idx = get_text_batch(text_data, batch_size, seq_len, rng)
    _ = manual_forward(model, warmup_idx)

    for i in range(n_eval):
        idx = get_text_batch(text_data, batch_size, seq_len, rng)
        B, T = idx.shape
        n_valid = max(T - local_window, 0)

        t0 = time.perf_counter()

        if is_random:
            # Random gate at target rate
            gate_decisions = rng.random(n_valid) < target_enabled_rate
        elif gate is not None and n_valid > 0:
            # Extract features and apply budget-calibrated threshold
            feats = compute_gate_features(model, idx, local_window)
            feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
            gate_logits = (
                gate(torch.tensor(feats_norm, dtype=torch.float32))
                .detach()
                .numpy()
                .flatten()
            )
            gate_probs = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -500, 500)))

            # Budget-calibrated: set threshold to hit target_enabled_rate
            threshold = find_budget_threshold(gate_probs, target_enabled_rate)
            gate_decisions = gate_probs >= threshold
        else:
            gate_decisions = np.zeros(n_valid, dtype=bool)

        enabled_rate_batch = float(gate_decisions.mean()) if n_valid > 0 else 0.0

        logits = manual_forward_with_per_position_mask(
            model, idx, gate_decisions, local_window
        )

        t1 = time.perf_counter()

        # Compute kept tokens
        if T > 0:
            total_attended = 0
            for t in range(T):
                if t < local_window:
                    total_attended += min(t + 1, T)
                else:
                    ti = t - local_window
                    if gate_decisions[ti]:
                        total_attended += min(t + 1, T)
                    else:
                        total_attended += local_window
            kept = total_attended / T
        else:
            kept = 0.0

        ppl = compute_ppl(logits, idx)
        ppls.append(ppl)
        tokens_per_query_list.append(kept)
        enabled_rates.append(enabled_rate_batch)
        wall_times.append((t1 - t0) * 1000)
        tokens_seen += B * T

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    [{i+1}/{n_eval}] "
                f"PPL={ppl:.1f} "
                f"kept={kept:.0f} "
                f"rate={enabled_rate_batch:.3f} "
                f"ms={wall_times[-1]:.0f}"
            )

    # Aggregate
    tpq = np.array(tokens_per_query_list)
    wall = np.array(wall_times)
    mean_enabled_rate = float(np.mean(enabled_rates))
    mean_kept = float(np.mean(tpq))

    # KV accounting
    kv = compute_kv_accounting(
        seq_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        local_window=local_window,
        enabled_rate=mean_enabled_rate,
        bytes_per_elem=2,
    )

    dense_flops = seq_len * d_model * n_layers
    total_tokens_per_batch = batch_size * seq_len
    ms_per_token = float(np.median(wall)) / total_tokens_per_batch

    return RunMetrics(
        variant=variant,
        seq_len=seq_len,
        seed=seed,
        ppl_mean=float(np.mean(ppls)),
        ppl_std=float(np.std(ppls)),
        enabled_rate=mean_enabled_rate,
        tokens_per_query_mean=mean_kept,
        tokens_per_query_p50=float(np.percentile(tpq, 50)),
        tokens_per_query_p95=float(np.percentile(tpq, 95)),
        tokens_per_query_p99=float(np.percentile(tpq, 99)),
        kv_bytes_written_per_token=kv.kv_bytes_written_per_token,
        kv_bytes_read_per_token=kv.kv_bytes_read_per_token,
        kv_bytes_total_per_token=kv.kv_bytes_total_per_token,
        peak_kv_bytes=kv.peak_kv_bytes,
        effective_kept_tokens=mean_kept,
        flops_proxy=kv.flops_proxy,
        flops_relative=kv.flops_proxy / dense_flops if dense_flops > 0 else 1.0,
        wall_ms_per_token=ms_per_token,
        tokens_per_sec=1000.0 / ms_per_token if ms_per_token > 0 else 0.0,
        tokens_seen=tokens_seen,
        n_eval_batches=n_eval,
    )


def run_grid(
    checkpoint: str,
    seq_lens: List[int],
    seeds: List[int],
    n_eval: int,
    batch_size: int,
    local_window: int,
    chunk_size: int,
    top_b: int,
    v4_gate_dir: str,
    v3_data_dir: str,
    text_data_path: str,
    output_dir: str,
    budgets: List[float] = None,
) -> Dict:
    """Run the full v4 experiment grid."""
    if budgets is None:
        budgets = [0.30, 0.40, 0.50, 0.60, 0.70]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "per_run_metrics"), exist_ok=True)

    text_data = load_text_data(text_data_path)
    print(f"Text data: {len(text_data)} tokens from {text_data_path}")

    all_results = []

    for seq_len in seq_lens:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"SEQ_LEN={seq_len}, SEED={seed}")
            print(f"{'='*60}")

            # Load model
            block_size = seq_len
            if checkpoint:
                ckpt_probe = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                wpe_size = ckpt_probe["model"]["transformer.wpe.weight"].shape[0]
                block_size = max(seq_len, wpe_size)
                del ckpt_probe

            cfg = BPAConfig(
                block_size=block_size,
                vocab_size=50304,
                n_layer=12,
                n_head=12,
                n_embd=768,
                local_window=local_window,
                chunk_size=chunk_size,
                top_b=top_b,
            )

            torch.manual_seed(seed)
            model = GPT2_BPA(cfg)

            if checkpoint:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model"])
                print(f"  Loaded checkpoint (iter={ckpt.get('iter_num', '?')})")
                del ckpt

            model.eval()

            # Load v4 gate
            v4_gate, v4_mean, v4_std = load_v4_gate(v4_gate_dir, seed=1)

            # Load v3 gate for reference
            v3_gate, v3_mean, v3_std = None, None, None
            if os.path.exists(v3_data_dir):
                v3_gate, v3_mean, v3_std = load_v3_gate(v3_data_dir, seed=1)

            seed_results = {}

            # V0: Dense baseline
            print(f"\n  --- V0_dense ---")
            m = evaluate_variant(
                model=model,
                text_data=text_data,
                variant="V0_dense",
                seq_len=seq_len,
                seed=seed,
                n_eval=n_eval,
                batch_size=batch_size,
                local_window=local_window,
            )
            seed_results["V0_dense"] = m
            all_results.append(m)

            # V1: Local-only
            print(f"\n  --- V1_local_only ---")
            m = evaluate_variant(
                model=model,
                text_data=text_data,
                variant="V1_local_only",
                seq_len=seq_len,
                seed=seed,
                n_eval=n_eval,
                batch_size=batch_size,
                local_window=local_window,
            )
            seed_results["V1_local_only"] = m
            all_results.append(m)

            # V2 at each budget level
            for budget in budgets:
                vname = f"V2_budget_{int(budget*100)}"
                print(f"\n  --- {vname} ---")
                m = evaluate_variant_budget(
                    model=model,
                    text_data=text_data,
                    variant=vname,
                    seq_len=seq_len,
                    seed=seed,
                    n_eval=n_eval,
                    batch_size=batch_size,
                    local_window=local_window,
                    gate=v4_gate,
                    feat_mean=v4_mean,
                    feat_std=v4_std,
                    target_enabled_rate=budget,
                )
                seed_results[vname] = m
                all_results.append(m)

            # V3 reference (v3 gate, fixed threshold=0.5)
            if v3_gate is not None:
                print(f"\n  --- V3_v3gate ---")
                m = evaluate_variant(
                    model=model,
                    text_data=text_data,
                    variant="V2_learned_gate",
                    seq_len=seq_len,
                    seed=seed,
                    n_eval=n_eval,
                    batch_size=batch_size,
                    local_window=local_window,
                    gate=v3_gate,
                    feat_mean=v3_mean,
                    feat_std=v3_std,
                    gate_threshold=0.5,
                )
                # Rename variant
                m = RunMetrics(**{**asdict(m), "variant": "V3_v3gate"})
                seed_results["V3_v3gate"] = m
                all_results.append(m)

            # VR: Random matched to V2_50
            v2_50_key = "V2_budget_50"
            if v2_50_key in seed_results:
                target_rate = seed_results[v2_50_key].enabled_rate
                print(f"\n  --- VR_random (matched rate={target_rate:.3f}) ---")
                m = evaluate_variant_budget(
                    model=model,
                    text_data=text_data,
                    variant="VR_random",
                    seq_len=seq_len,
                    seed=seed,
                    n_eval=n_eval,
                    batch_size=batch_size,
                    local_window=local_window,
                    target_enabled_rate=target_rate,
                    is_random=True,
                )
                seed_results["VR_random"] = m
                all_results.append(m)

            # Save per-run metrics
            for vname, metrics in seed_results.items():
                run_key = f"{vname}_L{seq_len}_S{seed}"
                run_path = os.path.join(
                    output_dir, "per_run_metrics", f"{run_key}.json"
                )
                with open(run_path, "w") as f:
                    json.dump(asdict(metrics), f, indent=2)

            # Print comparison
            dense_ppl = seed_results["V0_dense"].ppl_mean
            dense_flops = seed_results["V0_dense"].flops_proxy
            print(
                f"\n  {'Variant':<20} {'PPL':>8} {'vs Dense':>8} "
                f"{'Rate':>6} {'Kept':>6} {'KV_sav%':>8} {'ms/tok':>7}"
            )
            print(f"  {'-'*70}")
            for vname in sorted(seed_results.keys()):
                m = seed_results[vname]
                ppl_vs = (m.ppl_mean / dense_ppl - 1) * 100
                kv_sav = (
                    1
                    - m.kv_bytes_read_per_token
                    / seed_results["V0_dense"].kv_bytes_read_per_token
                ) * 100
                print(
                    f"  {vname:<20} {m.ppl_mean:>8.1f} {ppl_vs:>+7.1f}% "
                    f"{m.enabled_rate:>6.3f} {m.effective_kept_tokens:>6.0f} "
                    f"{kv_sav:>7.1f}% {m.wall_ms_per_token:>7.3f}"
                )

    # Save all results
    raw_path = os.path.join(output_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nAll results saved to {raw_path}")

    return {"results": all_results, "raw_path": raw_path}


def main():
    parser = argparse.ArgumentParser(
        description="BPA v4 Phase 2: Budget-Calibrated Gate Evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument(
        "--gate-dir",
        type=str,
        default="bpa_v4_gate_results",
        help="V4 gate results directory",
    )
    parser.add_argument(
        "--v3-data-dir",
        type=str,
        default="bpa_v2_trained_dataset",
        help="V3 gate training data",
    )
    parser.add_argument("--seq-lens", type=str, default="1024", help="Sequence lengths")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds")
    parser.add_argument("--n-eval", type=int, default=50, help="Eval batches per run")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--local-window", type=int, default=256, help="Local window")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size")
    parser.add_argument("--top-b", type=int, default=8, help="Top-B")
    parser.add_argument(
        "--text-data",
        type=str,
        default="gpt2/data/finewebedu/val.bin",
        help="Eval text",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v4_results",
        help="Output directory",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.30,0.40,0.50,0.60,0.70",
        help="Budget levels to evaluate",
    )
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [float(b) for b in args.budgets.split(",")]

    print("=" * 70)
    print("BPA v4 Phase 2: Budget-Calibrated Gate Evaluation")
    print("=" * 70)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Gate dir:     {args.gate_dir}")
    print(f"  Seq lens:     {seq_lens}")
    print(f"  Seeds:        {seeds}")
    print(f"  Budgets:      {budgets}")
    print(f"  n_eval:       {args.n_eval}")

    run_grid(
        checkpoint=args.checkpoint,
        seq_lens=seq_lens,
        seeds=seeds,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        local_window=args.local_window,
        chunk_size=args.chunk_size,
        top_b=args.top_b,
        v4_gate_dir=args.gate_dir,
        v3_data_dir=args.v3_data_dir,
        text_data_path=args.text_data,
        output_dir=args.output_dir,
        budgets=budgets,
    )


if __name__ == "__main__":
    main()
