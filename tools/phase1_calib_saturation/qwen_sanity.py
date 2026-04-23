#!/usr/bin/env python3.12
"""Qwen2.5-7B: confirm calibrated symmetric FP8 still collapses.

Five-minute sanity check.  Expected result: with vLLM runtime
calibration (calculate_kv_scales=True), Qwen2.5-7B still collapses
because the precision floor is a bit-budget problem, not a scale-
finding problem.  If it unexpectedly survives, that is a paper-
changing finding and we expand to the full eval battery.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("/workspace/results/qwen_sanity")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    model_args = (
        f"pretrained={MODEL},"
        "dtype=float16,"
        "gpu_memory_utilization=0.88,"
        "enforce_eager=True,"
        "attention_backend=FLASHINFER,"
        "max_model_len=2048,"
        "kv_cache_dtype=fp8_e4m3,"
        "calculate_kv_scales=True"
    )

    cmd = [
        "python3.12", "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", "wikitext",
        "--output_path", str(OUT_DIR),
        "--seed", "42",
        "--batch_size", "auto",
    ]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT_DIR / "stdout.log").write_text(r.stdout)
    (OUT_DIR / "stderr.log").write_text(r.stderr)

    for p in OUT_DIR.rglob("results_*.json"):
        data = json.load(open(p))
        wt = data.get("results", {}).get("wikitext", {})
        ppl = wt.get("word_perplexity,none") or wt.get("bits_per_byte,none")
        print(f"\n=== Qwen2.5-7B, calibrated FP8 (calculate_kv_scales), T=2048 ===")
        print(f"WikiText PPL: {ppl}")
        print(f"Full results: {p}")
        return 0
    print("no results found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
