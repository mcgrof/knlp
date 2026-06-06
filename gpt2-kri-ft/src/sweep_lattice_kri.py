"""Phase 5 — Lattice-KRI hyperparameter sweep driver.

Runs eval_lattice_kri.py across the Lattice knobs, writing one JSONL per
config so plot_lattice_kri_curves.py can compare them. Per the task: do NOT
sweep the full cross-product first. This driver varies one axis at a time
around the defaults (the cheap search), which is enough to find the slope of
each knob; lock the top configs and only then pay for SmolLM2 4K/8K.

Defaults (the rel_orth main config): lambda_orth=0.25, lambda_redun=0.25,
block_size=16, summary_mode="k", include_sink_recent_in_basis=false.

Grid (one-axis-at-a-time, ~14 configs):
  lambda_orth   in {0.05, 0.10, 0.50, 1.0}     (rel_orth)
  lambda_redun  in {0.05, 0.10, 0.50, 1.0}     (mmr)
  block_size    in {8, 32}
  summary_mode  in {"v", "kv_cat"}
  include_basis in {true}

Use --full for the dense cross-product only once a region looks promising.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

DEFAULTS = dict(
    lambda_orth=0.25,
    lambda_redun=0.25,
    block_size=16,
    summary_mode="k",
    include_sink_recent_in_basis="false",
)

ONE_AXIS = {
    "lambda_orth": [0.05, 0.10, 0.50, 1.0],
    "lambda_redun": [0.05, 0.10, 0.50, 1.0],
    "block_size": [8, 32],
    "summary_mode": ["v", "kv_cat"],
    "include_sink_recent_in_basis": ["true"],
}

# Just the lattice variants + the baseline they must beat — keep the sweep cheap.
SWEEP_ROUTERS = (
    "kri_q,kri_q_novelty,lattice_kri_rel_only,lattice_kri_orth_only,"
    "lattice_kri_rel_orth,lattice_kri_mmr,lattice_kri_residual_rel"
)


def _configs(full):
    if full:
        axes = {k: [DEFAULTS[k]] + v for k, v in ONE_AXIS.items()}
        keys = list(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):
            yield dict(zip(keys, combo))
        return
    yield dict(DEFAULTS)  # baseline
    for axis, vals in ONE_AXIS.items():
        for val in vals:
            cfg = dict(DEFAULTS)
            cfg[axis] = val
            yield cfg


def _tag(cfg):
    return "_".join(f"{k}{cfg[k]}" for k in sorted(cfg)).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--dataset_name", default="roneneldan/TinyStories")
    ap.add_argument("--tokenizer_name", default="openai-community/gpt2")
    ap.add_argument("--topks", default="1,2,4,8,16")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--local_window_tokens", type=int, default=128)
    ap.add_argument("--sink_blocks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--full",
        action="store_true",
        help="dense cross-product (expensive) instead of one-axis",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = list(_configs(args.full))
    print(f"[sweep] {len(configs)} configs -> {out_dir}", flush=True)
    for i, cfg in enumerate(configs):
        tag = _tag(cfg)
        out = out_dir / f"sweep_{tag}.jsonl"
        if out.exists():
            print(f"[{i + 1}/{len(configs)}] skip (exists) {tag}", flush=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            "src.eval_lattice_kri",
            "--models",
            args.models,
            "--routers",
            SWEEP_ROUTERS,
            "--seq_len",
            str(args.seq_len),
            "--block_size",
            str(cfg["block_size"]),
            "--local_window_tokens",
            str(args.local_window_tokens),
            "--sink_blocks",
            str(args.sink_blocks),
            "--topks",
            args.topks,
            "--n_batches",
            str(args.n_batches),
            "--batch_size",
            str(args.batch_size),
            "--dataset_name",
            args.dataset_name,
            "--tokenizer_name",
            args.tokenizer_name,
            "--lambda_orth",
            str(cfg["lambda_orth"]),
            "--lambda_redun",
            str(cfg["lambda_redun"]),
            "--summary_mode",
            cfg["summary_mode"],
            "--include_sink_recent_in_basis",
            str(cfg["include_sink_recent_in_basis"]),
            "--seed",
            str(args.seed),
            "--output",
            str(out),
        ]
        print(f"[{i + 1}/{len(configs)}] {tag}", flush=True)
        subprocess.run(cmd, check=True)
    print("[sweep] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
