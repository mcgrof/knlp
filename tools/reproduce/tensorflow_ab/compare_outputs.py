# SPDX-License-Identifier: MIT
"""Chunked comparison of SavedModel raw output tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(
    left_path: Path,
    right_path: Path,
    vocab_size: int,
    chunk_elements: int,
) -> dict[str, object]:
    left = np.memmap(left_path, mode="r", dtype=np.float32)
    right = np.memmap(right_path, mode="r", dtype=np.float32)
    if left.size != right.size:
        raise ValueError(
            f"element count differs: {left_path}={left.size}, "
            f"{right_path}={right.size}"
        )
    if left.size % vocab_size:
        raise ValueError(
            f"{left_path} has {left.size} elements, which is not "
            f"divisible by vocab size {vocab_size}"
        )

    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    sum_squared = 0.0
    exact = 0
    close_1e4 = True
    close_1e3 = True
    for start in range(0, left.size, chunk_elements):
        left_chunk = np.asarray(
            left[start : start + chunk_elements],
            dtype=np.float64,
        )
        right_chunk = np.asarray(
            right[start : start + chunk_elements],
            dtype=np.float64,
        )
        difference = np.abs(left_chunk - right_chunk)
        max_abs = max(max_abs, float(difference.max()))
        max_rel = max(
            max_rel,
            float((difference / np.maximum(np.abs(left_chunk), 1e-6)).max()),
        )
        sum_abs += float(difference.sum())
        sum_squared += float(np.square(difference).sum())
        exact += int(np.count_nonzero(left_chunk == right_chunk))
        close_1e4 &= bool(np.allclose(left_chunk, right_chunk, rtol=1e-4, atol=1e-5))
        close_1e3 &= bool(np.allclose(left_chunk, right_chunk, rtol=1e-3, atol=1e-4))

    rows = left.size // vocab_size
    left_top1 = np.asarray(left).reshape(rows, vocab_size).argmax(axis=1)
    right_top1 = np.asarray(right).reshape(rows, vocab_size).argmax(axis=1)
    top1_matches = int(np.count_nonzero(left_top1 == right_top1))
    return {
        "elements": int(left.size),
        "max_abs": max_abs,
        "mean_abs": sum_abs / left.size,
        "rmse": math.sqrt(sum_squared / left.size),
        "max_rel_floor_1e-6": max_rel,
        "exact_fraction": exact / left.size,
        "allclose_rtol_1e-4_atol_1e-5": close_1e4,
        "allclose_rtol_1e-3_atol_1e-4": close_1e3,
        "top1_matches": top1_matches,
        "top1_total": int(rows),
        "top1_fraction": top1_matches / rows,
    }


def compare_roots(
    left_root: Path,
    right_root: Path,
    cases: list[str],
    vocab_size: int = 256000,
    chunk_elements: int = 4_000_000,
) -> dict[str, object]:
    summary = {"cases": {}}
    for case_name in cases:
        left = left_root / case_name / "output_0.bin"
        right = right_root / case_name / "output_0.bin"
        summary["cases"][case_name] = {
            "sha256": {
                "left": sha256_file(left),
                "right": sha256_file(right),
            },
            "metrics": compare(left, right, vocab_size, chunk_elements),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-label", default="reference")
    parser.add_argument("--right-label", default="candidate")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["b1_s32", "b4_s32", "b1_s128"],
    )
    parser.add_argument("--vocab-size", type=int, default=256000)
    parser.add_argument("--chunk-elements", type=int, default=4_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = compare_roots(
        args.left_dir,
        args.right_dir,
        args.cases,
        args.vocab_size,
        args.chunk_elements,
    )
    summary["left_label"] = args.left_label
    summary["right_label"] = args.right_label
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
