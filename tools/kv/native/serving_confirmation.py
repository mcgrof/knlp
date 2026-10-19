# SPDX-License-Identifier: MIT
"""Replay dense FP16 HTTP confirmation from client and worker records.

This CPU reader checks the pinned H100 study's execution policy and all
offered requests. It retains failures and corrects the complete declared
comparison family. Profiling requires an explicit diagnostic cell call;
the full confirmation path accepts timing captures only.
"""

from pathlib import Path
import argparse
import collections
import hashlib
import json
import math

from .serving_verify import verify
from .serving_statistics import paired_effect, holm
from .serving_storage_audit import storage

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dense_weight_contract(protocol, installed):
    """Validate the structured expectation emitted by both measured harnesses."""
    expected = {
        "format": "fp16",
        "attention_layers": 36,
        "linear_modules": 144,
        "embedding_modules": 2,
    }
    assert protocol["packed_execution"] == expected
    assert protocol["geometry"] == {
        "layers": 36,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
    }
    assert len(installed["layers"]) == expected["attention_layers"]
    assert not installed["packed"]
    weights = installed["weights"]
    assert weights["format"] == "fp16"
    assert len(weights["linear_modules"]) == expected["linear_modules"]
    assert len(weights["embedding_modules"]) == expected["embedding_modules"]
    assert weights["parameters"] and all(
        p["dtype"] == "torch.float16" for p in weights["parameters"]
    )
    assert weights["dispatch_marker"] == "aten::mm"
    return {
        **expected,
        "parameter_count": len(weights["parameters"]),
        "unique_parameter_bytes": weights["unique_parameter_bytes"],
        "parameter_dtype": "torch.float16",
        "scope": "Per-invocation installed weight inventory; actual operator/kernel dispatch is separately attested by the source-matched production qualification.",
    }


def server_gauges(scrapes):
    names = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
    ]
    series = {n: [] for n in names}
    for scrape in scrapes:
        if scrape.get("http_status") != 200:
            continue
        for name in names:
            values = [
                float(line.rsplit(" ", 1)[1])
                for line in scrape["prometheus_text"].splitlines()
                if line.startswith(name + "{")
            ]
            assert len(values) == 1 and math.isfinite(values[0]) and values[0] >= 0
            series[name].append(values[0])
    assert all(series.values())
    return {
        "sampled": {
            n: {
                "samples": len(v),
                "min": min(v),
                "max": max(v),
                "first": v[0],
                "last": v[-1],
            }
            for n, v in series.items()
        },
        "scope": "One-second server metric scrapes; extrema are sampled, counters include earlier cells, and the last scrape may precede final completion. Worker receipts supply exact per-cell preemption counts. No server/client clock subtraction.",
    }


def valid_completion(row):
    """Match the independently checked completion inventory, including invalids."""
    if row["status"] != "completed":
        return False
    done, tokens, finishes = 0, 0, 0
    for event in row["events"]:
        if event["data"] == "[DONE]":
            done += 1
            continue
        for choice in json.loads(event["data"]).get("choices", []):
            tokens += len(choice.get("token_ids", []))
            finishes += choice.get("finish_reason") is not None
    return done == 1 and tokens > 0 and finishes > 0


def stream_gaps(requests):
    # These are client-visible token-bearing event gaps. A multi-token event
    # cannot disclose its internal token times; TCP coalescing can yield zeros.
    gaps = []
    counts = collections.Counter()
    sizes = []
    per_request = []
    for row in requests:
        if not valid_completion(row):
            continue
        times = []
        counts["completed_requests"] += 1
        for event in row["events"]:
            counts["sse_events"] += 1
            if event["data"] == "[DONE]":
                counts["end_of_stream_events"] += 1
                continue
            value = json.loads(event["data"])
            ids = []
            visible = False
            for choice in value.get("choices", []):
                assert choice.get("index", 0) == 0
                ids.extend(choice.get("token_ids", []))
                visible |= bool(choice.get("text"))
            if ids:
                times.append(event["at_ns"])
                sizes.append(len(ids))
                counts["token_bearing_events"] += 1
                counts["multi_token_events"] += len(ids) > 1
                counts["token_events_without_visible_text"] += not visible
            else:
                counts["events_without_tokens"] += 1
        local = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
        assert all(x >= 0 for x in local)
        gaps.extend(local)
        per_request.append(len(local))

    def quantile(q):
        if not gaps:
            return None
        values = sorted(gaps)
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] if lo == hi else values[lo] * (hi - i) + values[hi] * (i - lo)

    return {
        "counts": dict(counts),
        "event_token_count_histogram": dict(collections.Counter(sizes)),
        "token_bearing_event_gaps_ms": {
            "n": len(gaps),
            "p50": quantile(0.5),
            "p95": quantile(0.95),
            "max": max(gaps) if gaps else None,
            "zero_gaps": sum(x == 0 for x in gaps),
        },
        "per_request_gap_counts": per_request,
        "scope": "Conditional on completed requests; pooled descriptive client-visible event gaps. Multi-token interiors are unobserved; these are not token-level tail latencies or independent experimental replicates.",
    }


def backlog(result):
    rows = result["requests"]
    end = result["end_ns"]
    arrival_end = result["start_ns"] + result["plan"]["arrival_duration_ns"]
    events = []
    for r in rows:
        events.append((r["intended_ns"], 1))
        events.append((r.get("end_ns") or end, -1))
    events.sort()
    level = peak = 0
    area = 0.0
    previous = result["start_ns"]
    for stamp, delta in events:
        assert stamp >= previous and stamp <= end
        area += level * (stamp - previous) / 1e9
        level += delta
        assert level >= 0
        peak = max(peak, level)
        previous = stamp
    assert level == 0

    def at(t):
        return sum(r["intended_ns"] <= t < (r.get("end_ns") or end) for r in rows)

    span = arrival_end - result["start_ns"]
    return {
        "offered_minus_terminal_peak": peak,
        "request_seconds_of_outstanding_work": area,
        "outstanding_at_arrival_end": at(arrival_end),
        "outstanding_at_arrival_quarters": [
            at(result["start_ns"] + round(span * q / 4)) for q in range(1, 5)
        ],
        "drain_seconds_after_last_intended_arrival": max(
            0, (max(r.get("end_ns") or end for r in rows) - arrival_end) / 1e9
        ),
        "finite_observation_seconds": (end - result["start_ns"]) / 1e9,
        "arrival_interval_seconds": span / 1e9,
        "steady_state_or_sustainable_rate_established": False,
    }


def cell(root, label, expected, *, expected_mode="timing"):
    raw = root / f"{label}-client.json"
    x = read(raw)
    assert x["plan"] == expected
    v = verify(x)
    s = x["summary"]
    assert sum(valid_completion(r) for r in x["requests"]) == v["completed"]
    w = read(root / f"server/{label}/receipt.json")
    rows = read(root / f"server/{label}/steps.json")
    assert expected_mode in ("timing", "profile")
    assert (
        w["mode"] == expected_mode
        and w["oracle_checks"] == 0
        and w["steps"] == len(rows)
    )
    assert w["preemption_events"] == sum(len(r["preempted_request_ids"]) for r in rows)
    assert all(r["label"] == label for r in rows) and len(
        {r["batch_id"] for r in rows}
    ) == len(rows)
    from .serving_scheduler import admissions

    admission_rows = list(admissions(rows))
    new = [rid for row in admission_rows for rid in row["fresh"]]
    assert len(new) == len(set(new))
    traceids = [r["id"] for r in expected["requests"]]
    matched = {}
    for rid in new:
        candidates = [i for i in traceids if rid.startswith("cmpl-" + i + "-0-")]
        assert len(candidates) == 1 and candidates[0] not in matched
        matched[candidates[0]] = rid
    valid = s["counts"]["completed"] == s["offered"] and all(
        count == 0 for kind, count in s["counts"].items() if kind != "completed"
    )
    if valid:
        assert (
            set(matched) == set(traceids) and v["output_tokens"] == len(traceids) * 128
        )
    memory = w["memory"]
    p = read(root / "server/inputs/protocol.json")
    memory_ok = (
        memory["peak_reserved_since_reset"] <= p["torch_allocator_reserved_cap_bytes"]
        and memory["cuda_device_used_bytes"] <= p["total_driver_envelope_bytes"]
    )
    metrics = read(root / f"{label}-server-metrics.json")
    assert metrics and all(r["end_ns"] >= r["start_ns"] for r in metrics)
    return {
        "all_offered_completed_validly": valid,
        "memory_envelope_passed": memory_ok,
        "offered": s["offered"],
        "counts": s["counts"],
        "worker_admitted_requests": len(matched),
        "noncompletion_records": [
            {
                "request_id": r["id"],
                "status": "invalid" if r["status"] == "completed" else r["status"],
                "error": r.get("error"),
                "worker_admitted": r["id"] in matched,
                "received_headers": "headers_ns" in r,
                "received_events": len(r["events"]),
            }
            for r in x["requests"]
            if not valid_completion(r)
        ],
        "request_throughput_per_s": s["request_throughput_per_s"],
        "output_token_throughput_per_s": v["output_tokens"]
        / ((x["end_ns"] - x["start_ns"]) / 1e9),
        "latency": s["metrics_conditional_on_valid_completion"],
        "stream_events": stream_gaps(x["requests"]),
        "backlog": backlog(x),
        "preemption_events": w["preemption_events"],
        "actual_fresh_admission_cohorts": dict(
            collections.Counter(len(r["fresh"]) for r in admission_rows if r["fresh"])
        ),
        "actual_scheduled_request_counts": dict(
            collections.Counter(len(r["scheduled_tokens"]) for r in rows)
        ),
        "actual_worker_steps": len(rows),
        "peak_reserved_bytes": memory["peak_reserved_since_reset"],
        "sampled_driver_bytes": memory["cuda_device_used_bytes"],
        "client_sha256": sha(raw),
        "worker_steps_sha256": sha(root / f"server/{label}/steps.json"),
        "server_metrics": server_gauges(metrics),
        "server_metric_scrapes": len(metrics),
        "server_metric_errors": sum(r.get("http_status") != 200 for r in metrics),
    }


def memory_observations(base, inventories, cells, protocol):
    startup = [
        json.loads(line)
        for line in (base / "server/allocation.jsonl").read_text().splitlines()
    ]
    assert startup and startup[0]["phase"] == "device_initialized_before_model_load"
    observations = [
        {"phase": row["phase"], "driver_used_bytes": row["driver_device_used_bytes"]}
        for row in startup
    ]
    observations.extend(
        {
            "phase": label + "_timing",
            "driver_used_bytes": row["sampled_driver_used_bytes"],
        }
        for label, row in zip(("before", "after"), inventories)
    )
    observations.extend(
        {"phase": label + "_end", "driver_used_bytes": row["sampled_driver_bytes"]}
        for label, row in cells.items()
    )
    peak = max(row["driver_used_bytes"] for row in observations)
    torch_peak = max(
        [row["allocator_peak_bytes"] for row in inventories]
        + [row["peak_reserved_bytes"] for row in cells.values()]
    )
    return {
        "driver_observations": observations,
        "maximum_observed_driver_used_bytes": peak,
        "driver_envelope_passed_at_observation_points": peak
        <= protocol["total_driver_envelope_bytes"],
        "maximum_recorded_torch_reserved_peak_bytes": torch_peak,
        "torch_envelope_passed_at_observation_points": torch_peak
        <= protocol["torch_allocator_reserved_cap_bytes"],
        "scope": "Driver maximum over startup allocation and timing boundary snapshots; not continuous peak monitoring. Torch values are reported peaks since their worker reset, not a guaranteed uninterrupted startup peak.",
    }


def run(root, planpath, prefix="fp16-8k"):
    design = read(planpath / "DESIGN.json")
    assert design["format"] == "fp16" and design["length"] in (1024, 8192)
    assert design["blocks"] == len(design["orders"]) == 10
    assert design["output_tokens"] == 128
    rows = []
    descriptive = {}
    failed_launches, pending = [], []
    allvalid = True
    for block in list(range(design["blocks"])) + ["descriptive"]:
        order = design["arms"] if block == "descriptive" else design["orders"][block]
        arms = {}
        for arm in order:
            base = root / f"{prefix}-{block}-{arm}"
            if not (base / "DONE.json").exists():
                records = list(
                    (root / "jobs").glob(f"*-{prefix}-{block}-{arm}-result.json")
                )
                assert len(records) <= 1
                if records and read(records[0])["passed"] is False:
                    failed_launches.append(
                        {"block": block, "arm": arm, "receipt": read(records[0])}
                    )
                else:
                    pending.append({"block": block, "arm": arm})
                allvalid = False
                continue
            assert read(base / "DONE.json")["complete"]
            expected = read(planpath / f"plan-{block}-{arm}.json")
            assert read(base / "plan.json") == expected and expected[
                "frozen_design_sha256"
            ] == sha(planpath / "DESIGN.json")
            p = read(base / "server/inputs/protocol.json")
            identity = read(base / "server/worker-identity.json")
            assert (
                p["source_sha"] == design["source_sha"]
                and p["backend_sources"]
                == expected["backend_sources"]
                == identity["sources"]
                and identity["torch_threads"] == 22
                and identity["calibration_before_worker_warmup_and_capture"]
            )
            done = read(base / "DONE.json")
            assert done == {
                "complete": True,
                "stage": expected["stage"],
                "block": expected["block"],
                "arm": arm,
            }
            assert read(base / "command.json")["plan_sha256"] == sha(
                planpath / f"plan-{block}-{arm}.json"
            )
            qual = read(base / "qualification.json")
            assert (
                sha(base / "qualification.json")
                == expected["qualification"]["sha256"]
                == sha(planpath / expected["qualification"]["file"])
            )
            assert qual["passed"] and all(
                qual["checks"][k] is True
                for k in [
                    "worker_identity",
                    "calibration_before_capture",
                    "physical_storage",
                    "weight_dispatch",
                    "changing_graphs",
                    "inprocess_parity",
                    "mixed_and_idle_routes",
                    "nll",
                    "retrieval",
                    "gsm8k",
                ]
            )
            for key in ["format", "arm", "length", "cache_bytes", "backend_sources"]:
                assert qual[key] == expected[key]
            for name, digest in qual["implementation_sha256"].items():
                assert p["runner_files_sha256"][name] == digest
            for name, digest in p["inputs_sha256"].items():
                assert sha(base / "server/inputs" / name) == digest
            engine = p["engine"]
            assert (
                engine["dtype"] == "float16"
                and engine["tensor_parallel_size"] == 1
                and engine["kv_cache_memory_bytes"] == expected["cache_bytes"]
            )
            assert (
                engine["max_num_seqs"] == 96
                and engine["max_num_batched_tokens"] == 65536
                and engine["block_size"] == 16
                and engine["stream_interval"] == 1
            )
            assert (
                not engine["enable_prefix_caching"]
                and not engine["enable_chunked_prefill"]
                and not engine["calculate_kv_scales"]
                and not engine["enforce_eager"]
            )
            assert (
                engine["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
                and p["torch"] == "2.11.0+cu129"
            )
            weight_path = base / "server/before-capture/installed.json"
            weights = dense_weight_contract(p, read(weight_path))
            inventories = []
            for label in ["before", "after"]:
                raw = read(base / f"{label}-timing-storage.json")["results"]
                assert len(raw) == 1
                st = storage(raw[0], arm == "candidate", p["geometry"])
                assert st["unique_KV_backing_bytes"] <= expected["cache_bytes"]
                inventories.append(st)
            assert all(
                inventories[0][key] == inventories[1][key]
                for key in [
                    "K_payload_bytes",
                    "V_payload_bytes",
                    "unique_KV_backing_bytes",
                    "logical_KV_tokens_per_layer",
                ]
            )
            cells = {}
            for c in expected["cells"]:
                file = planpath / c["trace"]["file"]
                assert sha(file) == c["trace"]["sha256"]
                cells[c["label"]] = cell(base, c["label"], read(file))
                allvalid &= (
                    cells[c["label"]]["all_offered_completed_validly"]
                    and cells[c["label"]]["memory_envelope_passed"]
                )
            memory = memory_observations(base, inventories, cells, p)
            allvalid &= (
                memory["driver_envelope_passed_at_observation_points"]
                and memory["torch_envelope_passed_at_observation_points"]
            )
            arms[arm] = {
                "cells": cells,
                "dense_weight_contract": weights,
                "installed_weight_receipt_sha256": sha(weight_path),
                "physical_storage_before_and_after": inventories,
                "memory_observations": memory,
                "protocol_sha256": sha(base / "server/inputs/protocol.json"),
            }
        if block == "descriptive":
            descriptive = arms
        else:
            rows.append({"block": block, "order": order, "arms": arms})
    family = []
    complete_primary = all(len(b["arms"]) == len(design["arms"]) for b in rows)
    for h in design["primary_family"] if complete_primary else []:
        label = h["cell"]
        metric = h["metric"]
        baseline = h["baseline"]

        def value(a, b):
            r = b["arms"][a]["cells"][label]
            return (
                r["request_throughput_per_s"]
                if metric == "request_throughput_per_s"
                else r["latency"]["latency_arrival_ms"]["p95"]
            )

        effect = paired_effect(
            [value("candidate", b) for b in rows],
            [value(baseline, b) for b in rows],
            higher_is_better=h["direction"] == "higher",
            screen_percent=100 * h["median_relative_improvement_screen"],
        )
        effect.update(
            id=baseline + "-" + label,
            baseline=baseline,
            cell=label,
            metric=metric,
            all_blocks_valid_and_complete=allvalid,
        )
        family.append(effect)
    return {
        "complete": not pending,
        "failed_launches": failed_launches,
        "pending": pending,
        "design_sha256": sha(planpath / "DESIGN.json"),
        "source_sha": design["source_sha"],
        "blocks": rows,
        "descriptive": descriptive,
        "primary_family": holm(family) if family else [],
        "all_blocks_valid_and_complete": allvalid,
        "pilot_pooled": False,
        "no_operational_SLO_or_goodput_claim": True,
        "reader_sha256": sha(Path(__file__)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--prefix", default="fp16-8k")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    x = run(a.root, a.plan, a.prefix)
    a.output.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    print(json.dumps(x["primary_family"], indent=2))


if __name__ == "__main__":
    main()
