#!/usr/bin/env python3.12
"""Calibration head-to-head on Llama-3.1-8B.

Runs the six configs through lm-eval-harness with four evals each:
  - WikiText-2 PPL at T in {2048, 16384, 32768}
  - GSM8K 8-shot n=500
  - MMLU 5-shot n=500
  - NIAH multikey-3 at 32K

Each config is a separate lm-eval invocation because the KV cache
dtype has to be set at LLM init time.  We write a small shell helper
that shells out to lm-eval with the right model_args per config.

Output: /workspace/results/calib_h2h/<config>__<task>/... JSON files
and a summary /workspace/results/calib_h2h/summary.json.

Invariants:
  - All six configs use vLLM 0.19 + FlashInfer backend.
  - Calibrated configs load scales from
    /workspace/results/kv_scales/<model_slug>.pertensor.json or
    .perchannel.json (produced earlier by saturation_sweep.py
    bootstrap, which in turn reads collect_kv_scales.py output).
  - Eval sample sizes match what the paper already reports elsewhere
    so results slot into the existing tables.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES_DIR = Path("/workspace/results/kv_scales")
OUT_DIR = Path("/workspace/results/calib_h2h")

MODEL_ARGS_BASE = (
    f"pretrained={MODEL},"
    "dtype=float16,"
    "gpu_memory_utilization=0.90,"
    "enforce_eager=True,"
    "attention_backend=FLASHINFER,"
    "max_model_len=32768"
)


def model_args_for(config: str, model_slug: str) -> str:
    """Build the lm-eval --model_args string for a given config."""
    pt_path = SCALES_DIR / f"{model_slug}.pertensor.json"
    pc_path = SCALES_DIR / f"{model_slug}.perchannel.json"

    if config == "fp16":
        return MODEL_ARGS_BASE + ",kv_cache_dtype=auto"
    if config == "fp8_uncalib":
        return MODEL_ARGS_BASE + ",kv_cache_dtype=fp8_e4m3"
    if config == "fp8_calib_pt":
        return MODEL_ARGS_BASE + f",kv_cache_dtype=fp8_e4m3,quantization_param_path={pt_path}"
    if config == "fp8_calib_pc":
        return MODEL_ARGS_BASE + f",kv_cache_dtype=fp8_e4m3,quantization_param_path={pc_path}"
    if config == "asym_uncalib":
        return MODEL_ARGS_BASE + ",kv_cache_dtype=auto;fp8_e4m3"
    if config == "asym_calib":
        return MODEL_ARGS_BASE + f",kv_cache_dtype=auto;fp8_e4m3,quantization_param_path={pt_path}"
    raise ValueError(f"unknown config: {config}")


def lm_eval(model_args: str, task: str, out_dir: Path, limit: int = None):
    """Run a single lm-eval invocation.  Returns path to results JSON."""
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
        print(f"FAIL {task}: see {out_dir}/stderr.log", flush=True)
        return None
    # Find the results JSON
    for p in out_dir.rglob("results_*.json"):
        return p
    return None


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs", nargs="+",
        default=["fp16", "fp8_uncalib", "fp8_calib_pt", "fp8_calib_pc",
                 "asym_uncalib", "asym_calib"],
    )
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    model_slug = slug(MODEL)

    # Eval battery
    tasks = [
        ("wikitext_2k",       "wikitext",       None,
            ",max_seq_length=2048"),
        ("wikitext_16k",      "wikitext",       None,
            ",max_seq_length=16384"),
        ("wikitext_32k",      "wikitext",       None,
            ",max_seq_length=32768"),
        ("gsm8k_n500",        "gsm8k",          500, ""),
        ("mmlu_n500",         "mmlu",           500, ""),
        ("niah_mk3_32k",      "niah_multikey_3", None, ""),
    ]

    summary = {"model": MODEL, "configs": {}}
    for cfg in args.configs:
        cfg_dir = OUT_DIR / cfg
        summary["configs"][cfg] = {"tasks": {}}
        base_args = model_args_for(cfg, model_slug)
        for tname, task, limit, suffix in tasks:
            done_marker = cfg_dir / tname / ".done"
            if args.skip_existing and done_marker.exists():
                print(f"skip {cfg}/{tname}")
                continue
            margs = base_args + suffix
            out_path = lm_eval(margs, task, cfg_dir / tname, limit=limit)
            if out_path:
                done_marker.parent.mkdir(parents=True, exist_ok=True)
                done_marker.touch()
                summary["configs"][cfg]["tasks"][tname] = str(out_path)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
