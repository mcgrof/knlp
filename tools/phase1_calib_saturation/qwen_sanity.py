#!/usr/bin/env python3.12
"""Qwen2.5-7B calibrated FP8 sanity check.

Expect: calibrated per-tensor FP8 still collapses Qwen (because the
precision floor is a bit-budget problem, not a scale-finding problem).

Runs exactly one invocation: WikiText-2 PPL at T=2048 with calibrated
per-tensor FP8 scales.  Five minutes total.  Writes result JSON and
prints the ppl number.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCALES_DIR = Path("/workspace/results/kv_scales")
OUT_DIR = Path("/workspace/results/qwen_sanity")


def slug(m: str) -> str:
    return m.replace("/", "__")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    pt_path = SCALES_DIR / f"{slug(MODEL)}.pertensor.json"
    if not pt_path.exists():
        print(f"ERR: no scales for {MODEL}.  Run collect_kv_scales.py first.",
              file=sys.stderr)
        sys.exit(1)

    model_args = (
        f"pretrained={MODEL},"
        "dtype=float16,"
        "gpu_memory_utilization=0.90,"
        "enforce_eager=True,"
        "attention_backend=FLASHINFER,"
        "max_model_len=2048,"
        "kv_cache_dtype=fp8_e4m3,"
        f"quantization_param_path={pt_path}"
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

    # Parse the results JSON
    for p in OUT_DIR.rglob("results_*.json"):
        data = json.load(open(p))
        wt = data.get("results", {}).get("wikitext", {})
        ppl = wt.get("word_perplexity,none") or wt.get("bits_per_byte,none")
        print(f"\n=== Qwen2.5-7B, calibrated FP8 per-tensor, T=2048 ===")
        print(f"WikiText PPL: {ppl}")
        print(f"Full results: {p}")
        return 0
    print("no results found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
