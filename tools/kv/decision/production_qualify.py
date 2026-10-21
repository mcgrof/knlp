# SPDX-License-Identifier: MIT
"""Run excluded, prospectively frozen actual-worker numerical cases."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from tools.kv.native.serving_events import request_metrics
from tools.kv.native.serving_pilot import rpc
from .production import read_inputs, sha
from .serving import replay


CASES = [[1024, n] for n in [1, 8, 32, 33, 44, 45, 59, 60, 96]] + [
    [8192, 1],
    [8192, 32],
]
EXTRA_LONG_CASES = {
    "native_fi": [[8192, 35]],
    "native_fa3": [[8192, 35]],
    "AF": [[8192, 44], [8192, 45], [8192, 46]],
    "SF": [[8192, 59], [8192, 60], [8192, 70]],
}


def trace(data, length, count, label):
    samples = [
        s
        for s in data["samples"]["development"]
        if s["kind"] == "retrieval" and s["length"] == length
    ]
    if len(samples) != 32:
        raise ValueError("Fixed excluded development prompts differ")
    return {
        "mode": "open",
        "model": "study",
        "seed": 20261020,
        "arrival_seconds": 1,
        "drain_seconds": 3600,
        "request_timeout_seconds": 3600,
        "requests": [
            {
                "id": f"{label}-{i:03d}",
                "offset_ns": 0,
                "prompt_token_ids": samples[i % len(samples)]["token_ids"],
                "max_tokens": 32,
            }
            for i in range(count)
        ],
    }


def check_requests(capture):
    expected = {r["id"] for r in capture["plan"]["requests"]}
    rows = capture["requests"]
    if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
        raise ValueError("Numerical request inventory differs")
    for row in rows:
        metric = request_metrics(row)
        if (
            not metric["valid_completion"]
            or metric["output_tokens"] != 32
            or metric["finish_reasons"] != ["length"]
        ):
            raise ValueError("Numerical request failed or returned unexpected outputs")


def coverage(rows, arm):
    if arm not in EXTRA_LONG_CASES:
        raise ValueError("Unknown actual-worker arm")
    if not rows or any(r.get("passed") is False for r in rows):
        raise ValueError("Numerical checks failed or are absent")
    for row in rows:
        if row["kind"] == "writer" and row.get("exact") is not True:
            raise ValueError("Writer byte-equivalence evidence is missing")
        if row["kind"] in ("graph", "attention") and row.get("passed") is not True:
            raise ValueError("Numerical comparison verdict is missing")
    phases = {}
    for kind in ("writer", "attention"):
        phases[kind] = {
            phase: sorted(
                {r["layer"] for r in rows if r["kind"] == kind and r["phase"] == phase}
            )
            for phase in ("prefill", "mixed", "decode")
        }
        if any(len(names) != 36 for names in phases[kind].values()):
            raise ValueError(
                "Every layer must be checked during fresh, mixed and decode work"
            )
        if len({name for names in phases[kind].values() for name in names}) != 36:
            raise ValueError("Layer identities change between execution phases")
    graphs = [r for r in rows if r["kind"] == "graph"]
    shapes = sorted({r["padded_tokens"] for r in graphs})
    if shapes != [1, 8, 32, 44, 45, 59, 60, 96] or any(
        not r["cache_codes_exact"] for r in graphs
    ):
        raise ValueError("Complete graph shape and byte-exact cache coverage required")
    routes = [r for r in rows if r["kind"] == "routing"]
    if (
        max(r["requests"] for r in routes) < 96
        or len({tuple(r["slots"]) for r in routes}) < 2
    ):
        raise ValueError("Missing full-batch or changing slot-mapping evidence")
    if any(r.get("query_dtype", "torch.float16") != "torch.float16" for r in rows):
        raise ValueError("Non-native query operand found")
    long_requests = max(
        (
            r["requests"]
            for r in routes
            if r.get("phase") == "decode"
            and r.get("positions")
            and min(r["positions"]) >= 8191
        ),
        default=0,
    )
    if long_requests < max(n for _, n in EXTRA_LONG_CASES[arm]):
        raise ValueError("Missing long-context decode near this arm's cache capacity")
    return {
        "layers_by_phase": phases,
        "graph_shapes": shapes,
        "graph_checks": len(graphs),
        "maximum_actual_requests": max(r["requests"] for r in routes),
        "routing_checks": len(routes),
        "total_checks": len(rows),
        "maximum_long_context_requests": long_requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    if (
        protocol["numerical_cases"] != CASES
        or protocol["numerical_extra_long_cases"] != EXTRA_LONG_CASES
        or protocol["numerical_output_tokens"] != 32
    ):
        raise ValueError(
            "Numerical case family differs from the prospectively fixed recipe"
        )
    worker_path = args.worker_output / "worker-identity.json"
    worker = json.loads(worker_path.read_text())
    if worker["allocation"] != "capacity":
        raise ValueError("Numerical qualification requires the full-budget worker")
    if worker["protocol_sha256"] != args.protocol_sha256:
        raise ValueError("Worker protocol differs")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, receipts = [], []
    for i, (length, count) in enumerate(CASES + EXTRA_LONG_CASES[worker["arm"]]):
        label = f"numerical-{i:02d}-{length}-{count}"
        plan = trace(data, length, count, label)
        (args.output / (label + "-trace.json")).write_text(json.dumps(plan) + "\n")
        rpc("study_begin", label, "qualification")
        try:
            capture = asyncio.run(
                replay(
                    plan,
                    "http://127.0.0.1:8000",
                    args.output / (label + "-attempts.jsonl"),
                )
            )
            (args.output / (label + "-client.json")).write_text(
                json.dumps(capture, allow_nan=False) + "\n"
            )
            check_requests(capture)
        finally:
            receipt = rpc("study_end")
            (args.output / (label + "-worker.json")).write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        if len(receipt["results"]) != 1:
            raise ValueError("Expected one actual worker")
        value = receipt["results"][0]
        if value["preemption_events"] or not value["scheduler_events"]:
            raise ValueError(
                "Numerical workload preempted or scheduler evidence is absent"
            )
        path = args.worker_output / label / "numerics/checks.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
        receipts.append(
            {
                "label": label,
                "numerical_checks_sha256": sha(path),
                "worker_receipt_sha256": sha(
                    args.worker_output / label / "receipt.json"
                ),
                "storage_sha256": sha(args.worker_output / label / "storage.json"),
            }
        )
    build = rpc("study_build_identity", "numerical")
    result = {
        "passed": True,
        "arm": worker["arm"],
        "at": time.time(),
        "protocol_sha256": args.protocol_sha256,
        "worker_identity_sha256": sha(worker_path),
        "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "coverage": coverage(rows, worker["arm"]),
        "cases": receipts,
        "build_identity": build,
        "scope": "Excluded actual-worker numerical and routing qualification; no task-quality or serving-performance certificate",
    }
    (args.output / "QUALIFIED.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
