#!/usr/bin/env python3.12
"""Calibration head-to-head on Llama-3.1-8B.

Five configs through lm-eval: fp16, fp8_uncalib, fp8_calib (runtime),
asym_uncalib, asym_calib. Evals: WikiText PPL (lm-eval 'wikitext' task),
GSM8K n=500, MMLU n=500, NIAH multikey-3.

vLLM 0.19 does not take a custom JSON of calibration scales. The two
supported calibration paths are: (1) runtime calibration at engine
startup via calculate_kv_scales=True, which is what this script uses;
(2) pre-baked FP8 checkpoints produced by llm-compressor, deferred
to Phase 1b.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = Path("/workspace/results/calib_h2h")

BASE = (
    f"pretrained={MODEL},"
    "dtype=float16,"
    "gpu_memory_utilization=0.88,"
    "enforce_eager=True,"
    "attention_backend=FLASHINFER,"
    "max_model_len=32768"
)


def model_args_for(config: str) -> str:
    if config == "fp16":
        return BASE + ",kv_cache_dtype=auto"
    if config == "fp8_uncalib":
        return BASE + ",kv_cache_dtype=fp8_e4m3"
    if config == "fp8_calib":
        return BASE + ",kv_cache_dtype=fp8_e4m3,calculate_kv_scales=True"
    if config == "asym_uncalib":
        return BASE + ",kv_cache_dtype=auto;fp8_e4m3"
    if config == "asym_calib":
        return BASE + ",kv_cache_dtype=auto;fp8_e4m3,calculate_kv_scales=True"
    raise ValueError(f"unknown config: {config}")


def lm_eval(model_args, task, out_dir, limit=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3.12", "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", task,
        "--output_path", str(out_dir),
        "--seed", "42",
        "--batch_size", "auto",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(r.stdout)
    (out_dir / "stderr.log").write_text(r.stderr)
    if r.returncode != 0:
        print(f"FAIL {task}", flush=True)
        return None
    for p in out_dir.rglob("results_*.json"):
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["fp16", "fp8_uncalib", "fp8_calib",
                             "asym_uncalib", "asym_calib"])
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    tasks = [
        ("wikitext",      "wikitext",        None),
        ("gsm8k_n500",    "gsm8k",           500),
        ("mmlu_n500",     "mmlu",            500),
        ("niah_mk3",      "niah_multikey_3", None),
    ]

    summary = {"model": MODEL, "configs": {}}
    for cfg in args.configs:
        summary["configs"][cfg] = {}
        margs = model_args_for(cfg)
        for tname, task, limit in tasks:
            cfg_task_dir = OUT_DIR / cfg / tname
            done = cfg_task_dir / ".done"
            if args.skip_existing and done.exists():
                print(f"skip {cfg}/{tname}")
                continue
            p = lm_eval(margs, task, cfg_task_dir, limit=limit)
            if p:
                done.parent.mkdir(parents=True, exist_ok=True)
                done.touch()
                summary["configs"][cfg][tname] = str(p)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
