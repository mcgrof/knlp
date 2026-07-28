# SPDX-License-Identifier: MIT
"""Generate independent TensorFlow SavedModel outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--feeds", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["b1_s32", "b4_s32", "b1_s128"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serve = tf.saved_model.load(str(args.model)).signatures["serve"]
    manifest = {"tensorflow": tf.__version__, "cases": {}}

    for case_name in args.cases:
        batch, sequence = (int(part[1:]) for part in case_name.split("_"))
        token_ids = np.fromfile(
            args.feeds / f"token_ids_{case_name}.bin",
            dtype=np.float32,
        ).reshape(batch, sequence)
        padding_mask = np.fromfile(
            args.feeds / f"padding_mask_{case_name}.bin",
            dtype=np.float32,
        ).reshape(batch, sequence)

        for _ in range(args.warmup):
            serve(
                token_ids=tf.constant(token_ids),
                padding_mask=tf.constant(padding_mask),
            )["output_0"].numpy()

        durations = []
        output = None
        for _ in range(args.steps):
            start = time.perf_counter()
            output = serve(
                token_ids=tf.constant(token_ids),
                padding_mask=tf.constant(padding_mask),
            )["output_0"].numpy()
            durations.append(time.perf_counter() - start)

        output_path = args.output_dir / f"{case_name}.bin"
        output.tofile(output_path)
        manifest["cases"][case_name] = {
            "shape": list(output.shape),
            "durations_seconds": durations,
            "median_seconds": float(np.median(durations)),
            "output": str(output_path),
        }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
