#!/usr/bin/env python3
"""Main experiment matrix for per-layer LR scaling (adam-lr-01 Step 3).

Runs R0 (baseline), R1 (power=1), R2 (power=2) with 2 seeds each.
Total: 6 runs. Time-budgeted: 1800s per run to ensure enough iterations
for per-layer LR scaling to activate and show meaningful effects.
Config.py auto-detects batch=24, grad_acc=10 on W7900, giving ~5.9s/iter
so 1800s yields ~300 iters.
"""

import subprocess
import sys
import os
import json
import time

PYTHON = os.path.expanduser("~/envs/w7900-ml/bin/python")
OUTDIR = "runs/adam-lr-01/main_v2"
MAX_ITERS = 5000  # High ceiling; actual budget is time-limited
MAX_TIME = 1800  # 30 minutes per run (wall-clock)
SEEDS = [0, 1]

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
    "--max-time",
    str(MAX_TIME),
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

CONFIGS = {
    "R0_baseline": [],
    "R1_power1": [
        "--layer-lr-fim",
        "--layer-lr-fim-power",
        "1",
        "--layer-lr-fim-warmup-steps",
        "20",
        "--layer-lr-fim-update-every",
        "10",
        "--layer-lr-fim-clamp",
        "4",
    ],
    "R2_power2": [
        "--layer-lr-fim",
        "--layer-lr-fim-power",
        "2",
        "--layer-lr-fim-warmup-steps",
        "20",
        "--layer-lr-fim-update-every",
        "10",
        "--layer-lr-fim-clamp",
        "4",
    ],
}


def run_experiment(name, seed, extra_args):
    """Run a single experiment."""
    run_name = f"{name}_s{seed}"
    run_dir = os.path.join(OUTDIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_jsonl = os.path.join(run_dir, "layer_lr.jsonl")
    json_output = os.path.join(run_dir, "metrics.json")
    log_file = os.path.join(run_dir, "stdout.log")

    cmd = (
        COMMON_ARGS
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

    print(f"\n{'='*70}")
    print(f"Running: {run_name} (seed={seed})")
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

    # Extract final metrics from output
    final_train_ppl = None
    final_val_ppl = None
    best_val_ppl = None
    toks_per_sec = None
    final_iter = 0
    effective_batch = None

    for line in proc.stdout.split("\n"):
        if "Best validation perplexity" in line:
            try:
                best_val_ppl = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Effective batch size" in line:
            try:
                effective_batch = int(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if line.strip().startswith("Iter") and "ppl" in line:
            try:
                parts = line.split("|")
                iter_part = parts[0].strip()
                iter_num = int(iter_part.split()[-1])
                final_iter = max(final_iter, iter_num)
                for p in parts:
                    p = p.strip()
                    if p.startswith("ppl"):
                        final_train_ppl = float(p.split()[-1])
            except (ValueError, IndexError):
                pass

    # Extract tokens/sec from log (use effective batch * 1024 for tokens/iter)
    tokens_per_iter = (effective_batch or 64) * 1024
    for line in reversed(proc.stdout.split("\n")):
        if "ms/iter" in line:
            try:
                parts = line.split("|")
                for p in parts:
                    if "ms/iter" in p:
                        ms_per_iter = float(p.strip().split("ms")[0])
                        toks_per_sec = tokens_per_iter / (ms_per_iter / 1000)
                        break
            except (ValueError, IndexError):
                pass
            break

    status = "PASS" if proc.returncode == 0 else "FAIL"
    tps_str = f"{toks_per_sec:.0f}" if toks_per_sec else "N/A"
    print(
        f"[{run_name}] {status} | {elapsed:.0f}s | iters={final_iter} | "
        f"train_ppl={final_train_ppl} | best_val_ppl={best_val_ppl} | "
        f"toks/s={tps_str}"
    )

    return {
        "run_name": run_name,
        "config": name,
        "seed": seed,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "final_iter": final_iter,
        "effective_batch": effective_batch,
        "final_train_ppl": final_train_ppl,
        "best_val_ppl": best_val_ppl,
        "toks_per_sec": round(toks_per_sec, 0) if toks_per_sec else None,
        "log_file": log_file,
        "json_output": json_output,
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results = []

    for name, extra_args in CONFIGS.items():
        for seed in SEEDS:
            result = run_experiment(name, seed, extra_args)
            results.append(result)

    # Summary table
    print(f"\n{'='*90}")
    print("MAIN EXPERIMENT RESULTS")
    print(f"{'='*90}")
    print(
        f"{'Run':<25s} {'Seed':>4s} {'Iters':>6s} {'TrainPPL':>10s} {'BestValPPL':>10s} "
        f"{'Toks/s':>8s} {'Time':>6s} {'Status':>6s}"
    )
    print("-" * 100)
    for r in results:
        tppl = f"{r['final_train_ppl']:.2f}" if r["final_train_ppl"] else "N/A"
        vppl = f"{r['best_val_ppl']:.2f}" if r["best_val_ppl"] else "N/A"
        tps = f"{r['toks_per_sec']:.0f}" if r["toks_per_sec"] else "N/A"
        status = "PASS" if r["exit_code"] == 0 else "FAIL"
        print(
            f"{r['config']:<25s} {r['seed']:>4d} {r['final_iter']:>6d} {tppl:>10s} {vppl:>10s} "
            f"{tps:>8s} {r['elapsed_s']:>5.0f}s {status:>6s}"
        )

    # Save results
    summary_path = os.path.join(OUTDIR, "experiment_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
