# SPDX-License-Identifier: MIT
"""Recompute all-offered serving evidence from complete fresh-worker captures."""

import argparse
import json
from pathlib import Path

from tools.kv.native.serving_confirmation import server_gauges
from .goodput import evaluate
from .production import read_inputs, sha
from .paged_memory import audit_record
from .production_analysis import build_identity, resources
from .production_boundaries import boundaries, client_resources
from .production_plan import harness_gate, stable
from .production_quiescence import verified as quiescent
from .production_timing import validate_plan, warmup


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cell(root, label, trace, slo, protocol):
    """Replay raw scoring while allowing genuine overload/preemption/failures."""
    server, output = root / "server", root / "timing"
    capture = read(output / (label + "-client.json"))
    if read(output / (label + "-trace.json")) != trace or capture["plan"] != trace:
        raise ValueError("Retained arrivals or workload differ from the frozen plan")
    attempts = lines(output / (label + "-attempts.jsonl"))
    if len(attempts) != len(capture["requests"]) or sorted(
        attempts, key=lambda row: row["id"]
    ) != sorted(capture["requests"], key=lambda row: row["id"]):
        raise ValueError("Per-attempt sink differs from the complete client capture")
    if (
        capture["application_retries"] != 0
        or capture["connector_limit"] != 0
        or capture["keepalive_timeout_seconds"] != 300
    ):
        raise ValueError("Client connection policy differs")
    cpu = read(root / "cpu-contract.json")
    if capture["client_cpu_affinity"] != cpu["placement"]["client"]:
        raise ValueError("Measured client affinity differs")
    quiet = read(output / (label + "-quiescence.json"))
    if not quiescent(quiet) or quiet[0]["results"][0]["at_ns"] < capture["end_ns"]:
        raise ValueError("Worker was not quiescent after the client interval")
    steps = read(server / label / "steps.json")
    completed = read(server / label / "scheduler-completions.json")
    worker = read(server / label / "receipt.json")
    if read(output / (label + "-worker.json"))["results"] != [worker]:
        raise ValueError("RPC and retained timing receipt differ")
    if (
        worker["label"] != label
        or worker["mode"] != "timing"
        or worker["oracle_checks"] != 0
        or worker["decision_numerical_checks"] != 0
        or worker["steps"] != len(steps)
        or worker["scheduler_events"] != len(completed)
        or worker["preemption_events"]
        != sum(len(r["preempted_request_ids"]) for r in steps)
        or any(r["label"] != label for r in steps)
    ):
        raise ValueError("Timing worker instrumentation or interval differs")
    memory = worker["memory"]
    if (
        memory["peak_reserved_since_reset"]
        > protocol["torch_allocator_reserved_cap_bytes"]
        or memory["cuda_device_used_bytes"]
        > protocol["resource_policy"]["driver_envelope_bytes"]
    ):
        raise ValueError("Timing exceeded the common memory envelope")
    boundary = boundaries(
        capture, lines(server / "asgi-events.jsonl"), steps, completed
    )
    client = client_resources(capture)
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
    if result != read(output / (label + "-audit.json")):
        raise ValueError("Stored serving audit differs from its raw evidence")
    metrics = read(output / (label + "-server-metrics.json"))
    if not metrics or any(r["end_ns"] < r["start_ns"] for r in metrics):
        raise ValueError("Missing or invalid server metric sampling")
    storage = read(server / label / "storage.json")
    verified_storage = audit_record(
        storage,
        storage["arm"],
        protocol["geometry"],
        protocol["persistent_budget_bytes"],
    )
    initial = read(server / "initial-storage.json")
    if any(
        storage[k] != initial[k]
        for k in (
            "arm",
            "layers",
            "KV",
            "scales_including_native_Q_constant",
            "persistent_combined",
        )
    ):
        raise ValueError("Persistent storage changed during serving")
    return {
        **result,
        "worker": worker,
        "storage": verified_storage,
        "server_metrics": server_gauges(metrics),
        "server_metric_scrapes": len(metrics),
        "server_metric_errors": sum(r.get("http_status") != 200 for r in metrics),
        "quiescence": quiet,
    }


def invocation(
    root, plan, plan_digest, quality, quality_digest, protocol, digest, data
):
    validate_plan(plan, quality, protocol, digest, data)
    if plan["quality_analysis_sha256"] != quality_digest:
        raise ValueError("Plan names a different worker quality analysis")
    arm = plan["arm"]
    resource = resources(root, protocol, digest, arm, phase="timing")
    identity_path = root / "server/worker-identity.json"
    identity = read(identity_path)
    start, done = (read(root / "timing" / name) for name in ("START.json", "DONE.json"))
    expected = {
        "purpose": plan["purpose"],
        "arm": arm,
        "protocol_sha256": digest,
        "plan_sha256": plan_digest,
        "quality_analysis_sha256": quality_digest,
        "worker_identity_sha256": sha(identity_path),
        "client_cpu_affinity": resource["cpu_placement"]["client"],
    }
    if start != expected or any(done.get(k) != v for k, v in expected.items()):
        raise ValueError("Timing collector identity differs")
    if done["labels"] != [c["label"] for c in plan["cells"]]:
        raise ValueError("Missing, repeated or reordered timing cell")
    qualified_resource = quality["resource_records"][arm]
    if (
        qualified_resource["cpu_placement"] != resource["cpu_placement"]
        or qualified_resource["startup"]["uuid"] != resource["startup"]["uuid"]
    ):
        raise ValueError("Timing device or CPU placement differs from qualification")
    build = build_identity(root / "server", done["build_identity"], "timing")
    qualified = quality["qualified_builds"][arm]
    if (
        any(
            qualified["loaded_shared_objects_sha256"].get(name) != value
            for name, value in build["loaded_shared_objects_sha256"].items()
        )
        or build["package_versions"] != qualified["package_versions"]
    ):
        raise ValueError("Timing compiled modules or packages differ from quality")
    warm = cell(root, "warmup", warmup(data), None, protocol)
    if warm["boundaries"]["worker_admitted"] != 32 or not all(
        r["valid_completion"]
        and r["client_output_tokens"] == 128
        and r["finish_reasons"] == ["length"]
        for r in warm["boundaries"]["requests"]
    ):
        raise ValueError("Excluded warmup was incomplete")
    cells = [
        cell(root, c["label"], c["trace"], c["slo"], protocol) for c in plan["cells"]
    ]
    return {
        "protocol_sha256": digest,
        "source_sha": protocol["source_sha"],
        "plan_sha256": plan_digest,
        "quality_analysis_sha256": quality_digest,
        "purpose": plan["purpose"],
        "arm": arm,
        "worker_identity": identity,
        "worker_identity_sha256": sha(identity_path),
        "resources": resource,
        "build_identity": build,
        "warmup": warm,
        "cells": cells,
        "scope": "Independent single-invocation audit; no paired-block performance verdict",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Invocation plan changed")
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    result = invocation(
        args.root,
        read(args.plan),
        args.plan_sha256,
        read(args.quality_analysis),
        sha(args.quality_analysis),
        protocol,
        args.protocol_sha256,
        data,
    )
    args.output.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
