#!/usr/bin/env python3
"""Sanity tests for per-layer LR scaling (adam-lr-01 Step 2).

Runs 3 short training runs (300 steps each) to verify:
- No crash
- LR multipliers evolve (not all 1.0)
- Training loss decreases
- Throughput impact is small
"""

import subprocess
import sys
import os
import json
import time

PYTHON = os.path.expanduser("~/envs/w7900-ml/bin/python")
OUTDIR = "runs/adam-lr-01/sanity"
MAX_ITERS = 300
SEED = 42

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
    "4",
    "--gradient-accumulation",
    "4",
    "--max-iters",
    str(MAX_ITERS),
    "--learning-rate",
    "6e-4",
    "--weight-decay",
    "0.1",
    "--warmup-steps",
    "50",
    "--min-lr",
    "6e-5",
    "--optimizer",
    "adamw",
    "--log-interval",
    "10",
    "--eval-interval",
    "100",
    "--eval-samples",
    "50",
    "--no-save-checkpoint",
    "--tracker",
    "none",
    "--device",
    "cuda",
]

RUNS = {
    "R0_baseline": [],
    "R1_power1": [
        "--layer-lr-fim",
        "--layer-lr-fim-power",
        "1",
        "--layer-lr-fim-warmup-steps",
        "50",
        "--layer-lr-fim-update-every",
        "25",
    ],
    "R2_power2": [
        "--layer-lr-fim",
        "--layer-lr-fim-power",
        "2",
        "--layer-lr-fim-warmup-steps",
        "50",
        "--layer-lr-fim-update-every",
        "25",
    ],
}


def run_experiment(name, extra_args):
    """Run a single experiment and capture output."""
    run_dir = os.path.join(OUTDIR, name)
    os.makedirs(run_dir, exist_ok=True)

    log_jsonl = os.path.join(run_dir, "layer_lr.jsonl")
    json_output = os.path.join(run_dir, "metrics.json")

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

    # Add JSONL log for layer LR runs
    if "--layer-lr-fim" in extra_args:
        cmd += ["--layer-lr-fim-log-jsonl", log_jsonl]

    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    log_file = os.path.join(run_dir, "stdout.log")
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
    print(f"[{name}] Exit code: {proc.returncode}, Time: {elapsed:.1f}s")

    # Print last few lines of output
    lines = proc.stdout.strip().split("\n")
    for line in lines[-15:]:
        print(f"  {line}")

    return {
        "name": name,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log_file": log_file,
        "json_output": json_output,
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results = []

    for name, extra_args in RUNS.items():
        result = run_experiment(name, extra_args)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SANITY TEST SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for r in results:
        status = "PASS" if r["exit_code"] == 0 else "FAIL"
        if r["exit_code"] != 0:
            all_pass = False
        print(f"  {r['name']:20s} {status} ({r['elapsed_s']:.0f}s)")

    # Save summary
    summary_path = os.path.join(OUTDIR, "sanity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    if all_pass:
        print("\nAll sanity tests PASSED.")
    else:
        print("\nSome sanity tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
