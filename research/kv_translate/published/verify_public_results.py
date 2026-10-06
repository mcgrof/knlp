#!/usr/bin/env python3
"""Verify and summarize the released KV-Lingo aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PAIRS = {
    (1000, 1000),
    (1000, 5000),
    (5000, 1000),
    (5000, 5000),
}


def summarize(value: dict) -> dict:
    if value.get("schema") != "knlp_kv_lingo_closeout_aggregate_v1":
        raise ValueError("unexpected public-results schema")
    if value.get("acceptance") != {
        "overall_deficit_f1_points_max": 3.0,
        "each_domain_deficit_f1_points_max": 5.0,
        "pooled_turn6_10_deficit_f1_points_max": 5.0,
        "unhealthy_rate_excess_percentage_points_max": 2.0,
        "requirement": (
            "All gates for both starts and both references; decisions use point "
            "estimates, not bootstrap bounds."
        ),
    }:
        raise ValueError("acceptance rule differs from the frozen rule")

    pairs = value.get("pairs", [])
    identities = {
        (
            pair["forward_4b_to_8b_training_steps"],
            pair["reverse_8b_to_4b_training_steps"],
        )
        for pair in pairs
    }
    if identities != EXPECTED_PAIRS:
        raise ValueError("checkpoint-pair grid is incomplete")

    comparisons = [item for pair in pairs for item in pair["comparisons"]]
    if len(comparisons) != 16:
        raise ValueError("expected 16 start/reference comparisons")
    if any(len(pair["comparisons"]) != 4 for pair in pairs):
        raise ValueError("each checkpoint pair must contain four comparisons")

    for comparison in comparisons:
        gate_pass = all(comparison["gates"].values())
        if comparison["passed"] != gate_pass:
            raise ValueError("stored pass flag differs from its gates")
        failed = sorted(
            name for name, passed in comparison["gates"].items() if not passed
        )
        if sorted(comparison["failed_gates"]) != failed:
            raise ValueError("stored failed-gate list differs from its gates")

    starting_4b = [item for item in comparisons if item["starting_model"] == "Qwen3-4B"]
    starting_8b = [item for item in comparisons if item["starting_model"] == "Qwen3-8B"]
    summary = {
        "checkpoint_pairs": len(pairs),
        "comparisons": len(comparisons),
        "passing_comparisons": sum(item["passed"] for item in comparisons),
        "passing_full_pairs": sum(
            pair["passes_all_start_reference_comparisons"] for pair in pairs
        ),
        "failed_4b_start_comparisons": sum(not item["passed"] for item in starting_4b),
        "passing_8b_start_comparisons": sum(item["passed"] for item in starting_8b),
        "all_late_turn_gates_pass": all(
            item["gates"]["turn6_10_deficit_at_most_5pp"] for item in comparisons
        ),
        "all_unhealthy_excess_gates_pass": all(
            item["gates"]["unhealthy_excess_at_most_2pp"] for item in comparisons
        ),
        "absolute_unhealthy_rows": sum(
            count
            for pair in pairs
            for count in pair["absolute_unhealthy_rows_all_starts"].values()
        ),
    }
    expected = {
        "checkpoint_pairs": 4,
        "comparisons": 16,
        "passing_comparisons": 7,
        "passing_full_pairs": 0,
        "failed_4b_start_comparisons": 8,
        "passing_8b_start_comparisons": 7,
        "all_late_turn_gates_pass": True,
        "all_unhealthy_excess_gates_pass": True,
        "absolute_unhealthy_rows": 0,
    }
    if summary != expected:
        raise ValueError(f"aggregate closeout counts changed: {summary!r}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--expect-sha256")
    args = parser.parse_args(argv)
    payload = args.results.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if args.expect_sha256 and digest != args.expect_sha256:
        parser.error(f"SHA-256 mismatch: {digest}")
    result = summarize(json.loads(payload))
    print(json.dumps({"sha256": digest, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
