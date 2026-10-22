# SPDX-License-Identifier: MIT
"""One frozen primary load, all controls and ten paired serving blocks."""

import argparse
import json
from pathlib import Path

from .goodput import compare_blocks
from .production import read_inputs, sha
from .production_plan import orders
from .production_serving_plan import audit_entry, checked_manifest
from .production_timing_analysis import read


def inference(cells, arms, primary_rate):
    if primary_rate is None:
        return {
            "tested": False,
            "reason": "Bounded native-only pilot found no informative stress cell",
            "practical_improvement_established": False,
        }
    selected = [
        r for r in cells if r["input_tokens"] == 8192 and r["rate"] == primary_rate
    ]
    if len(selected) != len(arms) * 10 or {
        (r["arm"], r["block"]) for r in selected
    } != {(a, b) for a in arms for b in range(10)}:
        raise ValueError("Primary inference requires every planned arm and block once")
    grouped = {
        arm: [
            next(
                r["summary"]["goodput_requests_per_second"]
                for r in selected
                if r["arm"] == arm and r["block"] == block
            )
            for block in range(10)
        ]
        for arm in arms
    }
    result = compare_blocks(
        grouped["AF"], {a: values for a, values in grouped.items() if a != "AF"}
    )
    valid_harness = all(r["harness_screen_passed"] for r in selected)
    return {
        "tested": True,
        "rate": primary_rate,
        "input_tokens": 8192,
        "all_primary_harness_screens_passed": valid_harness,
        "paired_inference": result,
        "practical_improvement_established": valid_harness
        and result["improvement_vs_all_controls"],
        "scope": "Inference unit is a paired invocation block, never an individual request or token. Harness-screen failure prevents a serving improvement claim; every failed or late request already contributes zero goodput.",
    }


def analyze(folder, root, manifest, quality, quality_digest, protocol, digest, data):
    arms = ["native_fi", "AF", "native_fa3"] + (
        ["SD"] if "SD" in quality["serving_eligible_candidates"] else []
    )
    expected_orders = orders(arms)
    entries = manifest["invocations"]
    expected = [
        (block, arm)
        for block, arm_order in enumerate(expected_orders)
        for arm in arm_order
    ]
    if (
        manifest["arm_orders"] != expected_orders
        or [(r["block"], r["arm"]) for r in entries] != expected
    ):
        raise ValueError("Confirmation invocation order or comparison family differs")
    if manifest["inference"]["controls"] != [a for a in arms if a != "AF"]:
        raise ValueError("Confirmation control family differs")
    cells, invocations, failures, traces = [], [], [], {}
    previous_end = None
    for entry in entries:
        value = read(folder / entry["plan_file"])
        if value["purpose"] != "confirmation" or value["arm"] != entry["arm"]:
            raise ValueError("Confirmation plan identity differs")
        block = entry["block"]
        if block in traces and value["cells"] != traces[block]:
            raise ValueError("Arms did not replay identical per-block traces and SLOs")
        traces[block] = value["cells"]
        try:
            _, audit = audit_entry(
                folder, root, entry, quality, quality_digest, protocol, digest, data
            )
            lifecycle = read(root / entry["name"] / "server-exit.json")
            if previous_end is not None and lifecycle["started_ns"] < previous_end:
                raise ValueError(
                    "Fresh server invocations overlap or violate frozen order"
                )
            previous_end = lifecycle["ended_ns"]
            invocations.append(
                {
                    "name": entry["name"],
                    "worker_identity_sha256": audit["worker_identity_sha256"],
                    "resources": audit["resources"],
                    "started_ns": lifecycle["started_ns"],
                    "ended_ns": previous_end,
                }
            )
            for spec, row in zip(value["cells"], audit["cells"]):
                if spec["slo"] != manifest["slo_selection"]["selected"]["slo"]:
                    raise ValueError("Frozen baseline-relative SLO changed")
                # Keep all detailed per-request evidence in the independently
                # replayable capture; the final block report need not repeat it.
                summary = {
                    k: v
                    for k, v in row["summary"].items()
                    if k not in ("requests", "tokens_per_event")
                }
                cells.append(
                    {
                        "block": block,
                        "arm": entry["arm"],
                        "label": row["label"],
                        "input_tokens": spec["trace"]["input_tokens"],
                        "rate": spec["trace"]["nominal_rate"],
                        "harness_screen_passed": row["harness_screen_passed"],
                        "summary": summary,
                        "client_resources": row["client_resources"],
                        "worker": row["worker"],
                        "storage": row["storage"],
                        "server_metrics": row["server_metrics"],
                        "boundary_counts": {
                            k: v
                            for k, v in row["boundaries"].items()
                            if k != "requests"
                        },
                        "client_sha256": row["client_sha256"],
                        "trace_sha256": row["trace_sha256"],
                    }
                )
        except (ValueError, KeyError, TypeError, AssertionError, OSError) as error:
            failures.append(
                {"invocation": entry["name"], "error": repr(error), "replaced": False}
            )
    decision = (
        inference(cells, arms, manifest["primary_stress_load"]["rate"])
        if not failures
        else {
            "tested": False,
            "practical_improvement_established": False,
            "reason": "Incomplete or invalid frozen invocation evidence; no replacement blocks or pooled alternatives",
        }
    )
    return {
        "protocol_sha256": digest,
        "source_sha": protocol["source_sha"],
        "quality_analysis_sha256": quality_digest,
        "quality_objective": "aggregate_math",
        "complete": not failures,
        "failures": failures,
        "invocations": invocations,
        "primary": decision,
        "cells": cells,
        "symmetric_support_gap": quality["symmetric_support_gap"],
        "scope": "Dense Qwen3 FP16, exact fixed-AF and per-token/head-SD native-query workers, common 40-GiB persistent budget and 76-GiB sampled driver envelope, forced 128-token outputs. Nonprimary loads and 1K are descriptive; no model-family or natural-length serving claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    qdigest = sha(args.quality_analysis)
    manifest = checked_manifest(
        args.plans,
        args.manifest_sha256,
        "confirmation",
        protocol,
        args.protocol_sha256,
        qdigest,
    )
    result = analyze(
        args.plans,
        args.root,
        manifest,
        read(args.quality_analysis),
        qdigest,
        protocol,
        args.protocol_sha256,
        data,
    )
    result["serving_manifest_sha256"] = args.manifest_sha256
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
