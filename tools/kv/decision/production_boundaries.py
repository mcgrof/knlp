# SPDX-License-Identifier: MIT
"""Join every offered HTTP request to ASGI and actual worker admissions.

All timestamps are observed on the same host's monotonic clock. ASGI entry
is an application boundary, not a measurement of TCP accept. Failed requests
stay present even when neither the application nor worker ever saw them.
"""

import collections
import math
import re

from tools.kv.native.serving_events import quantile, request_metrics
from tools.kv.native.serving_scheduler import admissions
from .serving import validate


def request_id(internal, expected):
    match = re.fullmatch(r"cmpl-(.+)-0-[0-9a-f]{8}", internal)
    if match is None or match[1] not in expected:
        raise ValueError("Worker request ID is not one of the offered requests")
    return match[1]


def boundaries(capture, asgi, steps, completed):
    validate(capture["plan"])
    expected = {r["id"] for r in capture["plan"]["requests"]}
    client = capture["requests"]
    if len(client) != len(expected) or {r["id"] for r in client} != expected:
        raise ValueError("Every offered request must remain in the client evidence")
    selected_asgi = collections.defaultdict(list)
    for row in asgi:
        if row["id"] in expected:
            selected_asgi[row["id"]].append(row)
        elif capture["start_ns"] <= row["at_ns"] <= capture["end_ns"]:
            raise ValueError(
                "An unrelated HTTP request overlapped this measured interval"
            )
    starts, resumed_tokens = {}, 0
    observed_steps = list(admissions(steps))
    for step, admitted in zip(steps, observed_steps):
        for internal in admitted["fresh"]:
            external = request_id(internal, expected)
            if external in starts:
                raise ValueError("One offered ID produced multiple worker requests")
            starts[external] = {
                "internal_id": internal,
                "worker_start_ns": step["worker_monotonic_ns"],
            }
        resumed_tokens += sum(
            step["scheduled_tokens"][rid]
            for rid in admitted["resumed_via_new"] + admitted["resumed_via_cached"]
        )
    completion_ids = {r["batch_id"] for r in completed}
    if len(completion_ids) != len(completed):
        raise ValueError("Repeated scheduler-completion record")
    linked = {r["scheduler_batch_id"] for r in steps}
    if None in linked or linked != completion_ids:
        raise ValueError("Worker steps and scheduler completions do not match")
    tokens = collections.Counter()
    for row in completed:
        for internal, count in row["completed_tokens"].items():
            if type(count) is not int or count < 0:
                raise ValueError("Invalid completed token count")
            tokens[request_id(internal, expected)] += count
    result = []
    reuse, created = 0, 0
    for row in client:
        events = row["connection_events"]
        counts = collections.Counter(r["event"] for r in events)
        if counts["request_start"] > 1:
            raise ValueError("An offered request was retried")
        reuse += counts["connection_reuseconn"]
        created += counts["connection_create_end"]
        accepted = [
            r["at_ns"]
            for r in selected_asgi[row["id"]]
            if r["event"] == "asgi_accepted"
        ]
        if len(accepted) > 1:
            raise ValueError("An offered ID entered the ASGI application repeatedly")
        if selected_asgi[row["id"]] and not accepted:
            raise ValueError("ASGI response evidence lacks its acceptance boundary")
        worker = starts.get(row["id"])
        if worker is not None and (
            not accepted or accepted[0] > worker["worker_start_ns"]
        ):
            raise ValueError("Worker execution lacks a preceding ASGI acceptance")
        try:
            metric = request_metrics(row)
        except (KeyError, ValueError, TypeError):
            metric = {"valid_completion": False}
        if metric["valid_completion"]:
            if worker is None or metric["first_token_ns"] < worker["worker_start_ns"]:
                raise ValueError(
                    "Successful response lacks a preceding worker admission"
                )
            if tokens[row["id"]] != metric["output_tokens"]:
                raise ValueError("Successful client and scheduler token counts differ")
        boundary = (
            "worker_started"
            if worker is not None
            else "asgi_accepted" if accepted else "client_only"
        )
        result.append(
            {
                "id": row["id"],
                "status": row["status"],
                "error": row.get("error"),
                "furthest_observed_boundary": boundary,
                "intended_ns": row["intended_ns"],
                "dispatch_ns": row.get("dispatch_ns"),
                "send_ns": row.get("send_ns"),
                "headers_ns": row.get("headers_ns"),
                "asgi_accepted_ns": accepted[0] if accepted else None,
                "worker": worker,
                "first_token_ns": metric.get("first_token_ns"),
                "last_token_ns": metric.get("last_token_ns"),
                "eos_ns": metric.get("eos_ns"),
                "scheduler_output_tokens": tokens[row["id"]],
                "client_output_tokens": metric.get("output_tokens", 0),
                "finish_reasons": metric.get("finish_reasons", []),
                "valid_completion": metric["valid_completion"],
                "connection_events": events,
                "asgi_events": selected_asgi[row["id"]],
            }
        )
    return {
        "offered": len(client),
        "asgi_accepted": sum(bool(selected_asgi[i]) for i in expected),
        "worker_admitted": len(starts),
        "connection_creations": created,
        "connection_reuses": reuse,
        "requests": result,
        "preemption_events": sum(len(r["preempted_request_ids"]) for r in steps),
        "tokens_scheduled_on_resumption": resumed_tokens,
        "maximum_actual_scheduled_requests": max(
            (r["scheduled_sequences"] for r in completed), default=0
        ),
        "maximum_actual_decode_requests": max(
            (r["active_decode_sequences"] for r in completed), default=0
        ),
        "maximum_running_requests": max((r["running"] for r in completed), default=0),
        "maximum_waiting_requests": max((r["waiting"] for r in completed), default=0),
        "maximum_requests_with_computed_tokens_before_schedule": max(
            (
                sum(value[0] > 0 for value in r["computed_before"].values())
                for r in completed
            ),
            default=0,
        ),
        "scope": "Same-host observed boundaries. Running/waiting counts are scheduler state; requests with computed tokens are measured before scheduling. Resumption counts describe scheduled recomputation work, not kernel-active time. Failed or disconnected requests retain their furthest observed boundary.",
    }


def client_resources(capture):
    rows = capture["client_resources"]
    if len(rows) < 2:
        raise ValueError("Client resource telemetry is incomplete")
    cpu_rates, byte_rates = [], []
    for before, after in zip(rows, rows[1:]):
        seconds = (after["at_ns"] - before["at_ns"]) / 1e9
        cpu = sum(
            after[k] - before[k] for k in ("cpu_user_seconds", "cpu_system_seconds")
        )
        byte_delta = sum(
            after["network"][k] - before["network"][k]
            for k in ("bytes_sent", "bytes_recv")
        )
        if seconds <= 0 or min(cpu, byte_delta) < 0:
            raise ValueError("Client resource counters or clocks moved backwards")
        cpu_rates.append(cpu / seconds)
        byte_rates.append(byte_delta / seconds)
    lag = [
        (r["dispatch_ns"] - r["intended_ns"]) / 1e6
        for r in capture["requests"]
        if "dispatch_ns" in r
    ]
    if any(not math.isfinite(x) or x < 0 for x in lag):
        raise ValueError("Invalid client dispatch timing")
    return {
        "samples": len(rows),
        "cpu_affinity": capture["client_cpu_affinity"],
        "fd_limit": capture["client_fd_limit"],
        "maximum_observed_open_fds": max(r["open_fds"] for r in rows),
        "peak_observed_cpu_core_equivalents": max(cpu_rates),
        "peak_observed_whole_host_network_bytes_per_second": max(byte_rates),
        "dispatch_lag_ms": {
            "n": len(lag),
            "p50": quantile(lag, 0.5),
            "p95": quantile(lag, 0.95),
            "p99": quantile(lag, 0.99),
            "max": max(lag) if lag else None,
        },
        "scope": "CPU rates are process CPU seconds divided by elapsed seconds; one busy core is 1.0. Network counters are whole-host, not client-attributed, and loopback has no inferred physical link limit. Half-second samples can miss short saturation.",
    }
