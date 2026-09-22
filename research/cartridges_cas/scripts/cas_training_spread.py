#!/usr/bin/env python3
"""Reproduce the multi-GPU CAS isolated-training seed spread.

Each seed gets the faithful isolated recipe on one GPU. Repeated seeds are
determinism controls. Every completed cartridge is then evaluated in the same
software session with the Table-15 protocol, and a machine-readable summary is
written beside the raw logs and evaluator output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from typing import TextIO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--eval-runs", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=120)
    parser.add_argument("--kv-divisor", type=int, default=20)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", default="0.1")
    return parser.parse_args()


def faithful_recipe(args: argparse.Namespace) -> None:
    expected = {
        "global_batch": 128,
        "epochs": 80,
        "steps": 5000,
        "lr": "0.1",
        "kv_divisor": 20,
    }
    actual = {
        "global_batch": args.global_batch,
        "epochs": args.epochs,
        "steps": args.steps,
        "lr": args.lr,
        "kv_divisor": args.kv_divisor,
    }
    if actual != expected:
        raise SystemExit(f"training-spread recipe changed: {actual} != {expected}")


def launch(
    command: list[str], env: dict[str, str], log_path: Path
) -> tuple[subprocess.Popen, TextIO]:
    stream = log_path.open("w", encoding="utf-8")
    stream.write("COMMAND=" + " ".join(command) + "\n")
    stream.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
    stream.flush()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def run_waves(jobs: list[dict], gpus: list[str], make_process) -> None:
    for start in range(0, len(jobs), len(gpus)):
        active = []
        for offset, job in enumerate(jobs[start : start + len(gpus)]):
            gpu = gpus[offset]
            process, stream = make_process(job, gpu)
            active.append((job, gpu, process, stream))
            print(f"launched {job['label']} on GPU {gpu} as PID {process.pid}")
        failures = []
        try:
            for job, gpu, process, stream in active:
                status = process.wait()
                stream.close()
                print(f"finished {job['label']} on GPU {gpu}: status {status}")
                if status:
                    failures.append((job["label"], status))
        except BaseException:
            for _, _, process, stream in active:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                stream.close()
            raise
        if failures:
            raise SystemExit(f"jobs failed: {failures}")


def main() -> None:
    args = parse_args()
    faithful_recipe(args)
    if not args.gpus:
        raise SystemExit("at least one GPU is required")
    record = args.records / f"{args.patient}.txt"
    for path in (args.data, record):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing input: {path}")

    here = Path(__file__).resolve().parent
    trainer = here / "cas_train_isolated.py"
    evaluator = here / "cas_eval_table15.py"
    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / "logs"
    eval_dir = args.out / "eval"
    logs.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    jobs = []
    for index, seed in enumerate(args.seeds, start=1):
        label = f"s{seed}_r{index}"
        jobs.append(
            {
                "label": label,
                "seed": seed,
                "run_dir": args.out / "runs" / label,
            }
        )

    manifest = {
        "schema_version": 1,
        "knlp_commit": capture(
            ["git", "-C", str(here.parents[2]), "rev-parse", "HEAD"]
        ),
        "gpu_inventory": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ).splitlines(),
        "model": args.model,
        "patient": args.patient,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "inputs": {
            "parquet_sha256": sha256(args.data),
            "record_sha256": sha256(record),
        },
        "recipe": {
            "kv_tokens": "auto",
            "kv_divisor": args.kv_divisor,
            "global_batch": args.global_batch,
            "epochs": args.epochs,
            "steps": args.steps,
            "schedule_steps": args.steps,
            "schedule": "linear",
            "warmup_steps": 200,
            "warmup_min_lr": "2e-3",
            "alpha_f": "0.02",
            "lr": args.lr,
        },
        "evaluation": {
            "runs": args.eval_runs,
            "questions": 20,
            "max_completion": 2048,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    base_env = os.environ.copy()
    base_env.update(
        {
            "HF_HUB_OFFLINE": base_env.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": base_env.get("TRANSFORMERS_OFFLINE", "1"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    def train_process(job: dict, gpu: str):
        run_dir = job["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        env = base_env.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "inductor"),
                "PATIENT": args.patient,
                "DATA_PARQUET": str(args.data),
                "RECORDS_DIR": str(args.records),
                "KV_TOKENS": "auto",
                "KV_DIVISOR": str(args.kv_divisor),
                "GLOBAL_BS": str(args.global_batch),
                "EPOCHS": str(args.epochs),
                "STEPS": str(args.steps),
                "SCHED_STEPS": str(args.steps),
                "SCHEDULE": "linear",
                "WARMUP_STEPS": "200",
                "WARMUP_MIN_LR": "2e-3",
                "ALPHA_F": "0.02",
                "LR": args.lr,
                "SEED": str(job["seed"]),
                "SAVE_EVERY": str(args.save_every),
                "KEEP_LAST_N": "1",
                "NAME": f"cas_spread_{job['label']}",
                "OUT_DIR": str(run_dir),
            }
        )
        return launch(
            [sys.executable, str(trainer)], env, logs / f"train_{job['label']}.log"
        )

    run_waves(jobs, args.gpus, train_process)

    hashes_by_seed: dict[int, list[str]] = defaultdict(list)
    for job in jobs:
        cart = job["run_dir"] / "carts" / f"{args.patient}.pt"
        if not cart.is_file() or cart.stat().st_size == 0:
            raise SystemExit(f"training produced no cartridge: {cart}")
        job["cart"] = cart
        job["sha256"] = sha256(cart)
        hashes_by_seed[job["seed"]].append(job["sha256"])
    for seed, hashes in hashes_by_seed.items():
        if len(hashes) > 1 and len(set(hashes)) != 1:
            raise SystemExit(f"repeated seed {seed} produced different cartridges")

    def eval_process(job: dict, gpu: str):
        env = base_env.copy()
        output = eval_dir / f"table15_{job['label']}.json"
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "TORCHINDUCTOR_CACHE_DIR": str(eval_dir / f"inductor_{job['label']}"),
                "MODE": "cart",
                "CART_DIR": str(job["cart"].parent),
                "PATIENTS": args.patient,
                "MAX_Q": "20",
                "RUNS": str(args.eval_runs),
                "SAVE_RAW": "1",
                "MAX_COMPLETION": "2048",
                "TOP_K": "20",
                "TOP_P": "0.95",
                "DEVICE": "cuda:0",
                "MODEL": args.model,
                "OUT_JSON": str(output),
            }
        )
        job["eval"] = output
        return launch(
            [sys.executable, str(evaluator)], env, logs / f"eval_{job['label']}.log"
        )

    run_waves(jobs, args.gpus, eval_process)

    result_rows = []
    for job in jobs:
        result = json.loads(job["eval"].read_text(encoding="utf-8"))
        if result["top_k"] != 20 or result["top_p"] != 0.95:
            raise SystemExit(f"sampler mismatch in {job['eval']}")
        summary = result["summary"]
        if summary["runs"] != args.eval_runs:
            raise SystemExit(f"run-count mismatch in {job['eval']}")
        result_rows.append(
            {
                "label": job["label"],
                "seed": job["seed"],
                "cartridge_sha256": job["sha256"],
                "accuracy": summary["acc_mean"],
                "per_run_accuracy": summary["per_run_acc"],
            }
        )

    first_per_seed = {}
    for row in result_rows:
        first_per_seed.setdefault(row["seed"], row)
    distinct = [row["accuracy"] for row in first_per_seed.values()]
    report = {
        **manifest,
        "runs": result_rows,
        "repeated_seed_bit_identical": all(
            len(set(hashes)) == 1 for hashes in hashes_by_seed.values()
        ),
        "distinct_seed_count": len(distinct),
        "distinct_seed_accuracy_mean": statistics.mean(distinct),
        "distinct_seed_accuracy_sample_stddev": (
            statistics.stdev(distinct) if len(distinct) > 1 else 0.0
        ),
    }
    (args.out / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
