# SPDX-License-Identifier: MIT
"""Read matched Qwen3 HTTP profiles outside the fixed performance family."""

from pathlib import Path
import argparse, collections, gzip, hashlib, json

from .serving_confirmation import cell, dense_weight_contract
from .serving_storage_audit import storage
from .profile_ranges import join_cuda_events, _union

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def category(name):
    if name == "_gather":
        return "compact_value_gather"
    if name == "_mixed_writer" or "reshape_and_cache" in name:
        return "cache_writer"
    if "flashinfer::" in name or "flash::" in name:
        return "attention_and_merge"
    if name.startswith("nvjet_") or "gemm" in name.lower():
        return "matrix_multiply"
    return "other"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root, freeze, arm):
    plan = read(freeze / f"plan-{arm}.json")
    context = "8k" if plan["length"] == 8192 else "1k"
    base = root / f"fp16-{context}-profile-diagnostic-{arm}"
    spec = read(freeze / "PROFILE-PLAN.json")
    assert read(base / "plan.json") == plan
    assert read(base / "DONE.json") == {
        "complete": True,
        "stage": "descriptive",
        "block": "profile-diagnostic",
        "arm": arm,
    }
    assert plan["diagnostic"]["excluded_from_confirmation"] and plan["diagnostic"][
        "profile_plan_sha256"
    ] == sha(freeze / "PROFILE-PLAN.json")
    assert (
        plan["diagnostic"]["wrapper_sha256"]
        == sha(freeze / "run.py")
        == spec["wrapper_sha256"]
    )
    assert (
        plan["source_sha"] == spec["source_sha"]
        and plan["diagnostic"]["profile_request_count"] == 64
    )
    inputs = base / "server/inputs"
    p = read(inputs / "protocol.json")
    ident = read(base / "server/worker-identity.json")
    assert (
        p["source_sha"] == plan["source_sha"]
        and p["backend_sources"] == plan["backend_sources"] == ident["sources"]
    )
    assert (
        ident["torch_threads"] == 22
        and ident["calibration_before_worker_warmup_and_capture"]
    )
    qual = read(base / "qualification.json")
    assert (
        qual["passed"]
        and sha(base / "qualification.json") == plan["qualification"]["sha256"]
    )
    assert all(qual["checks"].values())
    for name, h in qual["implementation_sha256"].items():
        assert p["runner_files_sha256"][name] == h
    for name, h in p["inputs_sha256"].items():
        assert sha(inputs / name) == h
    assert (
        p["engine"]["dtype"] == "float16"
        and p["engine"]["kv_cache_memory_bytes"] == plan["cache_bytes"]
    )
    assert p["engine"]["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    installed_path = base / "server/before-capture/installed.json"
    weights = dense_weight_contract(p, read(installed_path))
    cells = {}
    for item in plan["cells"]:
        path = freeze / item["trace"]["file"]
        assert sha(path) == item["trace"]["sha256"]
        cells[item["label"]] = cell(
            base,
            item["label"],
            read(path),
            expected_mode="profile" if item["label"] == "open-near" else "timing",
        )
    assert cells["open-near"]["offered"] == 64
    assert cells["closed-32"]["offered"] == len(
        read(freeze / plan["cells"][0]["trace"]["file"])["requests"]
    )
    physical = []
    for label in ["before", "after"]:
        r = read(base / f"{label}-timing-storage.json")["results"]
        assert len(r) == 1
        physical.append(storage(r[0], arm == "candidate", p["geometry"]))
    assert (
        physical[0]["unique_KV_backing_bytes"] == physical[1]["unique_KV_backing_bytes"]
    )
    folder = base / "server/open-near"
    tracepath = folder / "trace.json.gz"
    trace = json.loads(gzip.decompress(tracepath.read_bytes()))
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert kernels and {e["args"]["device"] for e in kernels} == {0}
    rows = read(folder / "steps.json")
    schedule = {r["batch_id"]: r for r in rows}
    assert len(schedule) == len(rows)
    joins = join_cuda_events(trace)
    assert joins["matched_cuda_kernels"] + joins["unmatched_cuda_kernels"] == len(
        kernels
    )
    sums = collections.defaultdict(lambda: {"count": 0, "device_sum_us": 0.0})
    phases = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"count": 0, "device_sum_us": 0.0})
    )
    host = collections.defaultdict(lambda: {"calls": 0, "sum_us": 0.0})
    named_kernels = collections.defaultdict(lambda: {"count": 0, "device_sum_us": 0.0})
    host_batches = collections.defaultdict(
        lambda: {"ranges": 0, "host_duration_sum_us": 0.0}
    )
    for e in events:
        if e.get("ph") == "X" and e.get("name", "").startswith("knlp|batch|"):
            bid = int(e["name"].split("|")[2])
            assert bid in schedule
            phase = (
                "prefill_or_mixed"
                if any(n > 1 for n in schedule[bid]["scheduled_tokens"].values())
                else "decode"
            )
            host_batches[phase]["ranges"] += 1
            host_batches[phase]["host_duration_sum_us"] += e["dur"]
    for e in kernels:
        a = sums[category(e["name"])]
        a["count"] += 1
        a["device_sum_us"] += e["dur"]
        named = named_kernels[e["name"]]
        named["count"] += 1
        named["device_sum_us"] += e["dur"]
        named["category"] = category(e["name"])
    for group in joins["groups"]:
        bid = int(group["tag"].split("|")[2])
        assert bid in schedule
        phase = (
            "prefill_or_mixed"
            if any(n > 1 for n in schedule[bid]["scheduled_tokens"].values())
            else "decode"
        )
        for e in group["kernels"]:
            a = phases[phase][category(e["name"])]
            a["count"] += 1
            a["device_sum_us"] += e["device_duration_us"]
        for e in group["launches"]:
            a = host[e["name"]]
            a["calls"] += 1
            a["sum_us"] += e["host_duration_us"]
    operations = read(folder / "operations.json")
    route_counts = collections.Counter()
    gather_pages = []
    for operation in operations:
        kind = operation["tag"].split("|")[1]
        if kind == "gather":
            gather_pages.append(operation["pages"])
            route_counts["compact_gather_calls"] += 1
        elif kind == "prefill":
            dtypes = operation["cache_dtypes"]
            key = (
                (
                    "observed_fp16_staged_prefill"
                    if arm == "candidate"
                    else "observed_native_fp16_prefill"
                )
                if dtypes == ["torch.float16", "torch.float16"]
                else "observed_mixed_cache_prefill_fallback"
            )
            route_counts[key] += 1
        elif kind == "decode":
            route_counts["observed_python_decode_calls"] += 1
    return {
        "dense_weight_contract": weights,
        "installed_weight_receipt_sha256": sha(installed_path),
        "observed_python_route_counts": dict(route_counts),
        "compact_gather_page_occurrences": {
            "calls": len(gather_pages),
            "max": max(gather_pages) if gather_pages else None,
            "sum": sum(gather_pages),
        },
        "route_count_scope": "Python paths reached while profiling. Captured CUDA graphs can execute without re-entering per-layer Python wrappers; these counts are not total decode steps. Native FA3 is not a FlashInfer wrapper, so absent operation rows do not mean absent attention.",
        "cells": cells,
        "source_sha": p["source_sha"],
        "backend_sources": p["backend_sources"],
        "protocol_sha256": sha(inputs / "protocol.json"),
        "plan_sha256": sha(freeze / f"plan-{arm}.json"),
        "physical_storage": physical,
        "trace_sha256": sha(tracepath),
        "trace_event_count": len(events),
        "kernel_events": len(kernels),
        "matched_kernel_events": joins["matched_cuda_kernels"],
        "unmatched_kernel_events": joins["unmatched_cuda_kernels"],
        "all_gpu_kernel_duration_sum_us": sum(e["dur"] for e in kernels),
        "all_gpu_kernel_duration_union_us": _union(
            [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
        ),
        "all_gpu_kernels_by_category": dict(sums),
        "all_gpu_kernel_names": dict(named_kernels),
        "matched_gpu_kernels_by_phase": {k: dict(v) for k, v in phases.items()},
        "matched_host_cuda_calls": dict(host),
        "host_execute_ranges_by_phase": dict(host_batches),
        "host_execute_range_scope": "Host execute_model wall ranges can include waiting and asynchronous launches; they are not pure CPU compute or GPU duration. CUDA graph API calls remain explicit in matched_host_cuda_calls; their absence is not by itself evidence of a graph fallback.",
        "scheduler_steps": len(rows),
        "profiled_original_first64": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confirmation", type=Path, required=True)
    a = p.parse_args()
    confirmation = read(a.confirmation)
    spec = read(a.freeze / "PROFILE-PLAN.json")
    assert confirmation["complete"] and not confirmation["pending"]
    assert sha(a.confirmation) == spec["parent_result_sha256"]
    assert confirmation["design_sha256"] == spec["parent_design_sha256"]
    x = {
        "arms": {
            arm: audit(a.root, a.freeze, arm)
            for arm in ["candidate", "native_fi", "native_fa3"]
        },
        "excluded_from_confirmation": True,
        "one_diagnostic_run_per_arm": True,
        "scope": "The identical first64 original arrivals after a full closed32 prelude. Profiles change runtime overhead; component sums are not wall-clock causal decompositions, and client/server clocks are never subtracted. Prefill/decode use actual scheduler metadata; gather includes its fused value cast. Graphs can hide per-layer CPU ranges; unmatched kernels stay visible. This is a bounded bottleneck diagnostic, not new superiority evidence.",
        "reader_sha256": sha(Path(__file__)),
    }
    parent = next(b for b in confirmation["blocks"] if b["block"] == 0)
    for arm, value in x["arms"].items():
        assert value["protocol_sha256"] == parent["arms"][arm]["protocol_sha256"]
        assert value["source_sha"] == confirmation["source_sha"]
    x["parent_confirmation_sha256"] = sha(a.confirmation)
    x["matched_parent_protocols"] = True
    a.output.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                arm: {
                    k: v
                    for k, v in row.items()
                    if k
                    in [
                        "kernel_events",
                        "matched_kernel_events",
                        "unmatched_kernel_events",
                        "all_gpu_kernels_by_category",
                        "matched_gpu_kernels_by_phase",
                    ]
                }
                for arm, row in x["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
