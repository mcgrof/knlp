# SPDX-License-Identifier: MIT
"""Collect one frozen serving invocation after actual-worker quality gates.

The caller launches a fresh worker, retains its resource windows and provides
an immutable native-only pilot or confirmation plan. This collector chooses
no SLO, load, arm order, retry, replacement block or quality objective.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import re

from tools.kv.native.serving_pilot import rpc
from tools.kv.native.serving_telemetry import MetricSampler
from .goodput import evaluate
from .production import read_inputs, sha
from .production_analysis import build_identity
from .production_boundaries import boundaries, client_resources
from .production_plan import harness_gate, stable
from .production_quiescence import wait as wait_quiescent
from .serving import replay, validate


def save(path, value):
    # Long token-ID arrays repeat across the all-offered trace and client
    # records. Compact JSON retains every value without whitespace expansion.
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def validate_plan(plan, quality, protocol, digest, data):
    if (
        plan["protocol_sha256"] != digest
        or quality["protocol_sha256"] != digest
        or plan["source_sha"] != protocol["source_sha"]
        or quality["source_sha"] != protocol["source_sha"]
        or quality["preselected_serving_route"] != "aggregate_math"
        or "AF" not in quality["serving_eligible_candidates"]
    ):
        raise ValueError("Serving requires the exact qualified aggregate-math recipe")
    purpose, arm = plan["purpose"], plan["arm"]
    if purpose not in ("soak", "slo_calibration", "load_selection", "confirmation"):
        raise ValueError("Unknown frozen serving phase")
    if purpose != "confirmation" and arm != "native_fa3":
        raise ValueError(
            "Candidate timing cannot influence native-only pilot selection"
        )
    if arm not in ("native_fi", "native_fa3", *quality["serving_eligible_candidates"]):
        raise ValueError("Unqualified candidate cannot enter serving confirmation")
    if not plan["cells"] or len({c["label"] for c in plan["cells"]}) != len(
        plan["cells"]
    ):
        raise ValueError("Missing or repeated frozen serving cells")
    prompts = {s["id"]: s for s in data["samples"]["development"] if s["kind"] == "nll"}
    ids = set()
    for cell in plan["cells"]:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", cell["label"])
            or cell["label"] == "warmup"
        ):
            raise ValueError("Invalid or reserved cell label")
        trace = cell["trace"]
        validate(trace)
        for request in trace["requests"]:
            sample = prompts[request["source_sample_id"]]
            if (
                request["id"] in ids
                or request["max_tokens"] != 128
                or len(request["prompt_token_ids"]) != sample["length"]
                or request["prompt_token_ids"]
                != sample["token_ids"][: sample["length"]]
                or (purpose != "confirmation" and sample["length"] != 8192)
            ):
                raise ValueError(
                    "Serving workload differs from its fixed development prefixes"
                )
            ids.add(request["id"])
        if purpose in ("load_selection", "confirmation") and cell["slo"] is None:
            raise ValueError("Load selection/confirmation requires already frozen SLOs")
        if purpose in ("soak", "slo_calibration") and cell["slo"] is not None:
            raise ValueError("SLOs must be derived only after the calibration pilot")


def warmup(data):
    prompts = [
        s
        for s in data["samples"]["development"]
        if s["kind"] == "nll" and s["length"] == 8192
    ]
    if len(prompts) != 16:
        raise ValueError("Warmup needs the fixed development prompts")
    return {
        "mode": "open",
        "model": "study",
        "seed": 20261020,
        "arrival_seconds": 1,
        "drain_seconds": 120,
        "request_timeout_seconds": 240,
        "requests": [
            {
                "id": f"warmup-{i:03d}",
                "offset_ns": 0,
                "prompt_token_ids": prompts[i % 16]["token_ids"][:8192],
                "max_tokens": 128,
            }
            for i in range(32)
        ],
    }


def collect(trace, label, output, server, slo):
    save(output / (label + "-trace.json"), trace)
    rpc("study_begin", label, "timing")
    quiescence = []
    try:
        with MetricSampler("http://127.0.0.1:8000") as metrics:
            capture = asyncio.run(
                replay(
                    trace, "http://127.0.0.1:8000", output / (label + "-attempts.jsonl")
                )
            )
        save(output / (label + "-client.json"), capture)
        save(output / (label + "-server-metrics.json"), metrics.rows)
        wait_quiescent(quiescence)
    finally:
        save(output / (label + "-quiescence.json"), quiescence)
        save(output / (label + "-worker.json"), rpc("study_end"))
    steps = json.loads((server / label / "steps.json").read_text())
    completed = json.loads((server / label / "scheduler-completions.json").read_text())
    asgi = [
        json.loads(line)
        for line in (server / "asgi-events.jsonl").read_text().splitlines()
    ]
    boundary, client = boundaries(capture, asgi, steps, completed), client_resources(
        capture
    )
    result = {
        "label": label,
        "slo_defined": slo is not None,
        "summary": evaluate(capture, slo) if slo is not None else None,
        "boundaries": boundary,
        "client_resources": client,
        "harness_screen_passed": harness_gate(client, boundary),
        "native_calibration_stability": (
            stable(capture, boundary, client) if slo is None else None
        ),
        "client_sha256": sha(output / (label + "-client.json")),
        "trace_sha256": sha(output / (label + "-trace.json")),
    }
    save(output / (label + "-audit.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Frozen serving plan changed")
    plan = json.loads(args.plan.read_text())
    if sha(args.quality_analysis) != plan["quality_analysis_sha256"]:
        raise ValueError("Actual-worker certificate identity differs")
    quality = json.loads(args.quality_analysis.read_text())
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    validate_plan(plan, quality, protocol, args.protocol_sha256, data)
    worker_path = args.worker_output / "worker-identity.json"
    worker = json.loads(worker_path.read_text())
    if (
        worker["protocol_sha256"] != args.protocol_sha256
        or worker["arm"] != plan["arm"]
        or worker["allocation"] != "capacity"
    ):
        raise ValueError("Fresh worker differs from the qualified serving recipe")
    args.output.mkdir(parents=True, exist_ok=False)
    start = {
        "purpose": plan["purpose"],
        "arm": plan["arm"],
        "protocol_sha256": args.protocol_sha256,
        "plan_sha256": args.plan_sha256,
        "quality_analysis_sha256": sha(args.quality_analysis),
        "worker_identity_sha256": sha(worker_path),
        "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    save(args.output / "START.json", start)
    warm = collect(warmup(data), "warmup", args.output, args.worker_output, None)
    if warm["boundaries"]["worker_admitted"] != 32 or not all(
        r["valid_completion"]
        and r["client_output_tokens"] == 128
        and r["finish_reasons"] == ["length"]
        for r in warm["boundaries"]["requests"]
    ):
        raise ValueError("Excluded warmup did not complete every request once")
    audits = []
    for cell in plan["cells"]:
        audits.append(
            collect(
                cell["trace"],
                cell["label"],
                args.output,
                args.worker_output,
                cell["slo"],
            )
        )
    receipt = rpc("study_build_identity", "timing")
    build = build_identity(args.worker_output, receipt, "timing")
    qualified = quality["qualified_builds"][plan["arm"]]
    if any(
        qualified["loaded_shared_objects_sha256"].get(name) != digest
        for name, digest in build["loaded_shared_objects_sha256"].items()
    ):
        raise ValueError(
            "Serving loaded a compiled module absent from its quality qualification"
        )
    save(
        args.output / "DONE.json",
        {
            **start,
            "build_identity": receipt,
            "labels": [r["label"] for r in audits],
            "scope": "Complete fixed collection; no performance verdict before independent block analysis",
        },
    )


if __name__ == "__main__":
    main()
