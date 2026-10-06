#!/usr/bin/env python3
"""Compare KV-Lingo mapper and optimizer states for topology checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .kv_lingo_data import file_sha256


def tensor_leaves(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from tensor_leaves(value[key], f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{prefix}/{index}")


def compare_tensors(reference, candidate):
    left = dict(tensor_leaves(reference))
    right = dict(tensor_leaves(candidate))
    if left.keys() != right.keys():
        return {
            "status": "STRUCTURE_MISMATCH",
            "reference_only": sorted(left.keys() - right.keys()),
            "candidate_only": sorted(right.keys() - left.keys()),
        }
    rows = []
    difference_sq = 0.0
    reference_sq = 0.0
    for name in left:
        ref = left[name]
        actual = right[name]
        if ref.shape != actual.shape or ref.dtype != actual.dtype:
            rows.append(
                {
                    "name": name,
                    "status": "TYPE_OR_SHAPE_MISMATCH",
                    "reference_shape": list(ref.shape),
                    "candidate_shape": list(actual.shape),
                    "reference_dtype": str(ref.dtype),
                    "candidate_dtype": str(actual.dtype),
                }
            )
            continue
        ref_float = ref.detach().to(torch.float32)
        actual_float = actual.detach().to(torch.float32)
        difference = actual_float - ref_float
        diff_sq = float(torch.sum(difference * difference, dtype=torch.float64))
        ref_sq = float(torch.sum(ref_float * ref_float, dtype=torch.float64))
        difference_sq += diff_sq
        reference_sq += ref_sq
        rows.append(
            {
                "name": name,
                "status": "PASS",
                "shape": list(ref.shape),
                "dtype": str(ref.dtype),
                "exact": bool(torch.equal(ref, actual)),
                "finite": bool(torch.isfinite(actual_float).all()),
                "max_abs_difference": (
                    float(difference.abs().max()) if difference.numel() else 0.0
                ),
                "l2_difference": math.sqrt(diff_sq),
                "reference_l2": math.sqrt(ref_sq),
            }
        )
    passing = [row for row in rows if row["status"] == "PASS"]
    return {
        "status": "PASS" if len(passing) == len(rows) else "MISMATCH",
        "tensor_count": len(rows),
        "exact_tensor_count": sum(row["exact"] for row in passing),
        "all_finite": all(row["finite"] for row in passing),
        "max_abs_difference": max(
            (row["max_abs_difference"] for row in passing), default=0.0
        ),
        "relative_l2_difference": math.sqrt(difference_sq)
        / max(math.sqrt(reference_sq), torch.finfo(torch.float64).tiny),
        "tensors": rows,
    }


def load(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != "kv_lingo_stage2_checkpoint_v1":
        raise ValueError(f"not a KV-Lingo Stage-2 checkpoint: {path}")
    return value


def summarize_metric(value):
    metric = value["metrics"][-1]
    return {
        key: metric.get(key)
        for key in (
            "step",
            "sample_cursor",
            "mean_kl",
            "sample_kls",
            "learning_rate",
            "gradient_norm_before_clip",
            "world_size",
            "rank_assignment",
            "rank_compute_seconds",
            "rank_reduction_seconds",
            "global_step_seconds",
        )
        if key in metric
    }


def summarize_distributed(value):
    state = value.get("distributed")
    if state is None:
        return None
    return {
        key: state.get(key) for key in ("schema", "world_size", "backend", "assignment")
    } | {"rank_rng_count": len(state.get("rank_rng", []))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    reference = load(args.reference)
    candidate = load(args.candidate)
    identity_fields = (
        "direction",
        "global_step",
        "sample_cursor",
        "total_schedule_steps",
        "warmup_steps",
        "effective_batch",
        "learning_rate",
    )
    identity = {
        field: {
            "reference": reference.get(field),
            "candidate": candidate.get(field),
            "equal": reference.get(field) == candidate.get(field),
        }
        for field in identity_fields
    }
    report = {
        "schema": "kv_lingo_checkpoint_comparison_v1",
        "label": args.label,
        "reference": {
            "path": str(args.reference),
            "sha256": file_sha256(args.reference),
            "distributed": summarize_distributed(reference),
            "last_metric": summarize_metric(reference),
        },
        "candidate": {
            "path": str(args.candidate),
            "sha256": file_sha256(args.candidate),
            "distributed": summarize_distributed(candidate),
            "last_metric": summarize_metric(candidate),
        },
        "identity": identity,
        "identity_pass": all(value["equal"] for value in identity.values()),
        "translator": compare_tensors(reference["translator"], candidate["translator"]),
        "optimizer": compare_tensors(reference["optimizer"], candidate["optimizer"]),
        "Generated-by": "OpenAI Codex",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
