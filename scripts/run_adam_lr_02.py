#!/usr/bin/env python3
"""Experiment harness for adam-lr-02.

Phase A: 6 configs x 3 seeds = 18 runs at 250M tokens
  R0: Baseline AdamW (no layer LR scaling)
  R1: Fisher-LR power=1, warmup=200, update_every=50, clamp=4
  R2: Fisher-LR power=2, warmup=200, update_every=50, clamp=2
  C1: Random-shuffle (Fisher multipliers shuffled across layers)
  C2: Depth-ramp heuristic (linear from low to high by depth)
  C3: Fisher-LR power=1 frozen after 200 steps

Phase B: 3 configs x 5 seeds = 15 runs at 1B tokens
  R0: Baseline AdamW
  R1: Fisher-LR power=1 (best Fisher from Phase A)
  Best control from Phase A (determined by decision gate)

All runs share identical: dataset (finewebedu), model (gpt2),
optimizer (adamw), lr=6e-4, wd=0.1, warmup=200, min_lr=6e-5,
eval cadence (every 50 iters).

Usage:
    python scripts/run_adam_lr_02.py [--phase A|B] [--dry-run]
"""

import argparse
import subprocess
import sys
import os
import json
import time
import hashlib
import re

PYTHON = os.path.expanduser("~/envs/w7900-ml/bin/python")
OUTDIR_A = "runs/adam-lr-02/phaseA"
OUTDIR_B = "runs/adam-lr-02/phaseB"
MAX_ITERS = 50000  # High ceiling; token budget is the real limit

# Token budgets
TOKEN_BUDGET_A = 250_000_000  # Phase A: 250M tokens
TOKEN_BUDGET_B = 1_000_000_000  # Phase B: 1B tokens

SEEDS_A = [0, 1, 2]
SEEDS_B = [0, 1, 2, 3, 4]

COMMON_ARGS = [
    PYTHON,
    "gpt2/train.py",
    "--architecture",
    "vanilla",
    "--model-name",
    "gpt2",
    "--dataset",
    "finewebedu",
    "--batch-size",
    "8",
    "--gradient-accumulation",
    "8",
    "--max-iters",
    str(MAX_ITERS),
    "--learning-rate",
    "6e-4",
    "--weight-decay",
    "0.1",
    "--warmup-steps",
    "200",
    "--min-lr",
    "6e-5",
    "--optimizer",
    "adamw",
    "--log-interval",
    "10",
    "--eval-interval",
    "50",
    "--eval-samples",
    "100",
    "--no-save-checkpoint",
    "--tracker",
    "none",
    "--device",
    "cuda",
]

# Layer LR common settings for adam-lr-02
# (warmup=200, update_every=50, per the protocol)
LAYER_LR_COMMON = [
    "--layer-lr-fim",
    "--layer-lr-fim-warmup-steps",
    "200",
    "--layer-lr-fim-update-every",
    "50",
]

CONFIGS_A = {
    "R0_baseline": [],
    "R1_fisher_p1": LAYER_LR_COMMON
    + [
        "--layer-lr-fim-power",
        "1",
        "--layer-lr-fim-clamp",
        "4",
        "--layer-lr-mode",
        "fisher",
    ],
    "R2_fisher_p2": LAYER_LR_COMMON
    + [
        "--layer-lr-fim-power",
        "2",
        "--layer-lr-fim-clamp",
        "2",
        "--layer-lr-mode",
        "fisher",
    ],
    "C1_random_shuffle": LAYER_LR_COMMON
    + [
        "--layer-lr-fim-power",
        "1",
        "--layer-lr-fim-clamp",
        "4",
        "--layer-lr-mode",
        "random_shuffle",
    ],
    "C2_depth_ramp": LAYER_LR_COMMON
    + [
        "--layer-lr-fim-power",
        "1",
        "--layer-lr-fim-clamp",
        "4",
        "--layer-lr-mode",
        "depth_ramp",
    ],
    "C3_frozen_200": LAYER_LR_COMMON
    + [
        "--layer-lr-fim-power",
        "1",
        "--layer-lr-fim-clamp",
        "4",
        "--layer-lr-mode",
        "fisher",
        "--layer-lr-freeze-after-steps",
        "200",
    ],
}


def write_manifest(run_dir, config_name, seed, token_budget, cmd):
    """Write a repro manifest for this run."""
    import torch

    manifest = {
        "config": config_name,
        "seed": seed,
        "token_budget": token_budget,
        "command": " ".join(cmd),
        "cwd": os.getcwd(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "pytorch_version": torch.__version__,
        "rocm_version": getattr(torch.version, "hip", "N/A"),
        "python": sys.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def parse_stdout(text):
    """Extract metrics from run stdout."""
    final_train_ppl = None
    best_val_ppl = None
    toks_per_sec = None
    final_iter = 0
    effective_batch = None
    val_trajectory = []

    for line in text.split("\n"):
        line = line.strip()

        if "Effective batch size" in line:
            try:
                effective_batch = int(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass

        # Training log lines
        m = re.match(
            r"Iter\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+ppl\s+([\d.]+)",
            line,
        )
        if m:
            it = int(m.group(1))
            final_iter = max(final_iter, it)
            final_train_ppl = float(m.group(3))

        # Eval lines
        em = re.match(
            r"Eval @ iter (\d+): train ([\d.]+), val ([\d.]+), ppl ([\d.]+)",
            line,
        )
        if em:
            val_trajectory.append(
                {
                    "iter": int(em.group(1)),
                    "train_loss": float(em.group(2)),
                    "val_loss": float(em.group(3)),
                    "val_ppl": float(em.group(4)),
                }
            )

        if "Best validation perplexity" in line:
            try:
                best_val_ppl = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass

    # Extract tokens/sec from last log line with ms/iter
    tokens_per_iter = (effective_batch or 240) * 1024
    for line in reversed(text.split("\n")):
        if "ms/iter" in line:
            try:
                for p in line.split("|"):
                    if "ms/iter" in p:
                        ms = float(p.strip().split("ms")[0])
                        toks_per_sec = tokens_per_iter / (ms / 1000)
                        break
            except (ValueError, IndexError):
                pass
            break

    return {
        "final_iter": final_iter,
        "effective_batch": effective_batch,
        "final_train_ppl": final_train_ppl,
        "best_val_ppl": best_val_ppl,
        "toks_per_sec": round(toks_per_sec, 0) if toks_per_sec else None,
        "val_trajectory": val_trajectory,
    }


def run_experiment(config_name, seed, extra_args, outdir, token_budget):
    """Run a single experiment."""
    run_name = f"{config_name}_s{seed}"
    run_dir = os.path.join(outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_jsonl = os.path.join(run_dir, "layer_lr.jsonl")
    json_output = os.path.join(run_dir, "metrics.json")
    log_file = os.path.join(run_dir, "stdout.log")

    cmd = (
        COMMON_ARGS
        + ["--max-tokens", str(token_budget)]
        + extra_args
        + [
            "--json-output",
            json_output,
            "--output-dir",
            run_dir,
        ]
    )

    if "--layer-lr-fim" in extra_args:
        cmd += ["--layer-lr-fim-log-jsonl", log_jsonl]

    # Write manifest before run
    write_manifest(run_dir, config_name, seed, token_budget, cmd)

    print(f"\n{'='*70}")
    print(f"Running: {run_name} (seed={seed}, tokens={token_budget:,})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.time()

    with open(log_file, "w") as f:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/data/knlp",
        )
        f.write(proc.stdout)

    elapsed = time.time() - t0

    metrics = parse_stdout(proc.stdout)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    tps_str = f"{metrics['toks_per_sec']:.0f}" if metrics["toks_per_sec"] else "N/A"

    print(
        f"[{run_name}] {status} | {elapsed:.0f}s | "
        f"iters={metrics['final_iter']} | "
        f"train_ppl={metrics['final_train_ppl']} | "
        f"best_val_ppl={metrics['best_val_ppl']} | "
        f"toks/s={tps_str}"
    )

    result = {
        "run_name": run_name,
        "config": config_name,
        "seed": seed,
        "token_budget": token_budget,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log_file": log_file,
        "json_output": json_output,
    }
    result.update(metrics)

    # Save per-run result
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_summary(results, phase):
    """Print summary table of results."""
    print(f"\n{'='*100}")
    print(f"PHASE {phase} RESULTS")
    print(f"{'='*100}")
    print(
        f"{'Config':<20s} {'Seed':>4s} {'Iters':>6s} {'TrainPPL':>10s} "
        f"{'BestValPPL':>10s} {'Toks/s':>8s} {'Time':>7s} {'Status':>6s}"
    )
    print("-" * 80)
    for r in results:
        tppl = f"{r['final_train_ppl']:.2f}" if r["final_train_ppl"] else "N/A"
        vppl = f"{r['best_val_ppl']:.2f}" if r["best_val_ppl"] else "N/A"
        tps = f"{r['toks_per_sec']:.0f}" if r["toks_per_sec"] else "N/A"
        status = "PASS" if r["exit_code"] == 0 else "FAIL"
        print(
            f"{r['config']:<20s} {r['seed']:>4d} {r['final_iter']:>6d} "
            f"{tppl:>10s} {vppl:>10s} {tps:>8s} "
            f"{r['elapsed_s']:>6.0f}s {status:>6s}"
        )

    # Per-config mean/std
    configs = {}
    for r in results:
        if r["best_val_ppl"] is not None:
            configs.setdefault(r["config"], []).append(r["best_val_ppl"])

    print(f"\n{'Config':<20s} {'Mean ValPPL':>12s} {'StdDev':>8s} {'N':>3s}")
    print("-" * 50)
    import numpy as np

    baseline_mean = None
    for name, vals in sorted(configs.items()):
        mean = np.mean(vals)
        std = np.std(vals)
        if "baseline" in name:
            baseline_mean = mean
        improvement = ""
        if baseline_mean and "baseline" not in name:
            pct = (mean - baseline_mean) / baseline_mean * 100
            improvement = f"  ({pct:+.1f}%)"
        print(f"{name:<20s} {mean:>12.2f} {std:>8.2f} {len(vals):>3d}" f"{improvement}")


def select_phase_b_configs():
    """Select Phase B configs from Phase A results.

    Per protocol: baseline + best Fisher + strongest control.
    Reads Phase A experiment_results.json to determine the
    best control automatically.
    """
    import numpy as np

    results_path = os.path.join(OUTDIR_A, "experiment_results.json")
    if not os.path.exists(results_path):
        # Fallback: load individual result.json files
        results = []
        for entry in sorted(os.listdir(OUTDIR_A)):
            rp = os.path.join(OUTDIR_A, entry, "result.json")
            if os.path.isfile(rp):
                with open(rp) as f:
                    results.append(json.load(f))
    else:
        with open(results_path) as f:
            results = json.load(f)

    # Find best control
    control_configs = ["C1_random_shuffle", "C2_depth_ramp", "C3_frozen_200"]
    control_means = {}
    for cc in control_configs:
        ppls = [
            r["best_val_ppl"]
            for r in results
            if r["config"] == cc and r.get("best_val_ppl") is not None
        ]
        if ppls:
            control_means[cc] = np.mean(ppls)

    if not control_means:
        print("WARNING: No control results found in Phase A.")
        print("Using C3_frozen_200 as default control for Phase B.")
        best_control = "C3_frozen_200"
    else:
        best_control = min(control_means, key=control_means.get)
        print(f"Phase B control selection from Phase A results:")
        for cc, mean in sorted(control_means.items(), key=lambda x: x[1]):
            marker = " <-- SELECTED" if cc == best_control else ""
            print(f"  {cc}: mean PPL = {mean:.2f}{marker}")

    configs = {
        "R0_baseline": CONFIGS_A["R0_baseline"],
        "R1_fisher_p1": CONFIGS_A["R1_fisher_p1"],
        best_control: CONFIGS_A[best_control],
    }
    return configs


def main():
    parser = argparse.ArgumentParser(description="adam-lr-02 experiment harness")
    parser.add_argument(
        "--phase",
        type=str,
        default="A",
        choices=["A", "B"],
        help="Which phase to run (A=250M, B=1B tokens)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run only this config (e.g., R0_baseline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only this seed",
    )
    args = parser.parse_args()

    if args.phase == "A":
        outdir = OUTDIR_A
        token_budget = TOKEN_BUDGET_A
        seeds = SEEDS_A
        configs = CONFIGS_A
    else:
        outdir = OUTDIR_B
        token_budget = TOKEN_BUDGET_B
        seeds = SEEDS_B
        configs = select_phase_b_configs()

    os.makedirs(outdir, exist_ok=True)

    # Filter by config/seed if requested
    if args.config:
        configs = {k: v for k, v in configs.items() if k == args.config}
        if not configs:
            print(f"Unknown config: {args.config}")
            sys.exit(1)
    if args.seed is not None:
        seeds = [args.seed]

    # Print plan
    total_runs = len(configs) * len(seeds)
    print(f"Phase {args.phase}: {total_runs} runs")
    print(f"  Token budget: {token_budget:,}")
    print(f"  Seeds: {seeds}")
    print(f"  Configs: {list(configs.keys())}")
    print(f"  Output: {outdir}")
    print()

    if args.dry_run:
        for name, extra_args in configs.items():
            for seed in seeds:
                cmd = COMMON_ARGS + ["--max-tokens", str(token_budget)] + extra_args
                print(f"[DRY-RUN] {name}_s{seed}:")
                print(f"  {' '.join(cmd)}\n")
        return

    results = []
    for name, extra_args in configs.items():
        for seed in seeds:
            result = run_experiment(name, seed, extra_args, outdir, token_budget)
            results.append(result)

    print_summary(results, args.phase)

    # Save all results
    summary_path = os.path.join(outdir, "experiment_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
