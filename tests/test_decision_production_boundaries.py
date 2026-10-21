# SPDX-License-Identifier: MIT
"""Connection failures remain visible beside admitted, completed requests."""

import copy
import json

import pytest
from decision_storage_fixture import memory_fixture

from tools.kv.decision.production_boundaries import boundaries, client_resources
from tools.kv.decision.production import sha
from tools.kv.decision.goodput import evaluate
from tools.kv.decision.production_plan import harness_gate
from tools.kv.decision.production_timing_analysis import cell


def fixture():
    start = 1_000_000_000
    offered = [
        {
            "id": name,
            "offset_ns": i * 100_000_000,
            "prompt_token_ids": [1, 2],
            "max_tokens": 2,
        }
        for i, name in enumerate(("good", "disconnected"))
    ]
    requests = []
    for row in offered:
        arrival = start + row["offset_ns"]
        requests.append(
            {
                **row,
                "intended_ns": arrival,
                "dispatch_ns": arrival,
                "send_ns": arrival,
                "end_ns": arrival + 60_000_000,
                "status": "error",
                "error": "ServerDisconnectedError()",
                "connection_events": [{"event": "request_start", "at_ns": arrival}],
                "events": [],
            }
        )
    good = requests[0]
    good.update(status="completed", error=None)
    good["events"] = [
        {
            "at_ns": start + at,
            "data": json.dumps(
                {"choices": [{"token_ids": [token], "finish_reason": reason}]}
            ),
        }
        for at, token, reason in [(30_000_000, 10, None), (50_000_000, 11, "length")]
    ]
    good["events"].append({"at_ns": start + 60_000_000, "data": "[DONE]"})
    capture = {
        "plan": {
            "mode": "open",
            "model": "study",
            "seed": 20261020,
            "arrival_seconds": 1,
            "drain_seconds": 1,
            "request_timeout_seconds": 2,
            "requests": offered,
        },
        "start_ns": start,
        "end_ns": start + 1_000_000_000,
        "requests": requests,
    }
    asgi = [{"id": "good", "event": "asgi_accepted", "at_ns": start + 10_000_000}]
    internal = "cmpl-good-0-0123abcd"
    steps, completed = [], []
    for i in range(2):
        steps.append(
            {
                "batch_id": i,
                "scheduler_batch_id": 10 + i,
                "worker_monotonic_ns": start + (20 + i * 20) * 1_000_000,
                "new_request_ids": [internal] if i == 0 else [],
                "resumed_request_ids": [],
                "preempted_request_ids": [],
                "scheduled_tokens": {internal: 2 if i == 0 else 1},
            }
        )
        completed.append(
            {
                "batch_id": 10 + i,
                "completed_tokens": {internal: 1},
                "scheduled_sequences": 1,
                "active_decode_sequences": i,
                "running": 1,
                "waiting": 0,
                "computed_before": {internal: [2 * i, 2]},
            }
        )
    return capture, asgi, steps, completed


def test_pre_worker_disconnect_stays_in_complete_offered_inventory():
    result = boundaries(*fixture())
    assert result["offered"] == 2
    assert result["asgi_accepted"] == result["worker_admitted"] == 1
    failed = result["requests"][1]
    assert failed["furthest_observed_boundary"] == "client_only"
    assert failed["error"] == "ServerDisconnectedError()"
    assert not failed["valid_completion"]
    assert result["requests"][0]["scheduler_output_tokens"] == 2
    assert result["maximum_actual_decode_requests"] == 1


def test_all_failed_requests_can_have_no_worker_steps():
    capture, _, _, _ = fixture()
    for row in capture["requests"]:
        row.update(status="error", events=[])
    result = boundaries(capture, [], [], [])
    assert result["offered"] == 2 and result["worker_admitted"] == 0
    assert all(
        r["furthest_observed_boundary"] == "client_only" for r in result["requests"]
    )


@pytest.mark.parametrize(
    "defect", ["missing", "asgi", "early", "id", "tokens", "retry", "scheduler"]
)
def test_inconsistent_boundaries_do_not_qualify_the_client(defect):
    capture, asgi, steps, completed = fixture()
    if defect == "missing":
        capture["requests"].pop()
    elif defect == "asgi":
        asgi = []
    elif defect == "early":
        asgi[0]["at_ns"] += 100_000_000
    elif defect == "id":
        steps[0]["new_request_ids"] = ["cmpl-unoffered-0-0123abcd"]
        steps[0]["scheduled_tokens"] = {"cmpl-unoffered-0-0123abcd": 2}
    elif defect == "tokens":
        completed[0]["completed_tokens"] = {}
    elif defect == "retry":
        capture["requests"][0]["connection_events"] *= 2
    else:
        steps[0]["scheduler_batch_id"] = None
    with pytest.raises(ValueError):
        boundaries(capture, asgi, steps, completed)


def test_client_cpu_uses_core_seconds_without_diluting_single_thread_saturation():
    capture, _, _, _ = fixture()
    capture.update(client_cpu_affinity=[8, 9], client_fd_limit=[8192, 8192])
    capture["client_resources"] = [
        {
            "at_ns": i * 500_000_000,
            "cpu_user_seconds": i * 0.4,
            "cpu_system_seconds": i * 0.1,
            "open_fds": 20,
            "network": {"bytes_sent": i * 1000, "bytes_recv": i * 1000},
        }
        for i in range(3)
    ]
    result = client_resources(capture)
    assert result["peak_observed_cpu_core_equivalents"] == 1
    assert result["peak_observed_whole_host_network_bytes_per_second"] == 4000
    assert result["dispatch_lag_ms"]["max"] == 0


@pytest.mark.parametrize(
    "defect", [None, "attempt", "summary", "oracle", "memory", "pending"]
)
def test_independent_timing_reader_keeps_disconnect_and_rejects_changed_evidence(
    tmp_path, defect
):
    capture, asgi, steps, completed = fixture()
    capture.update(
        arrival_end_ns=2_000_000_000,
        drain_deadline_ns=3_000_000_000,
        application_retries=0,
        connector_limit=0,
        keepalive_timeout_seconds=300,
        client_cpu_affinity=[8, 9],
        client_fd_limit=[8192, 8192],
    )
    capture["client_resources"] = [
        {
            "at_ns": i * 500_000_000,
            "cpu_user_seconds": 0,
            "cpu_system_seconds": 0,
            "open_fds": 20,
            "network": {"bytes_sent": 0, "bytes_recv": 0},
        }
        for i in range(3)
    ]
    # Retain a legitimate preemption: overload timing must not inherit the
    # quality collector's zero-preemption rule.
    interrupted, finished = copy.deepcopy(steps[-1]), copy.deepcopy(completed[-1])
    interrupted.update(
        preempted_request_ids=["cmpl-good-0-0123abcd"], scheduled_tokens={}
    )
    finished.update(
        completed_tokens={}, scheduled_sequences=0, active_decode_sequences=0
    )
    steps[-1].update(
        batch_id=2, scheduler_batch_id=12, new_request_ids=["cmpl-good-0-0123abcd"]
    )
    completed[-1]["batch_id"] = 12
    steps.insert(1, interrupted)
    completed.insert(1, finished)
    for row in steps:
        row["label"] = "cell"
    worker = {
        "label": "cell",
        "mode": "timing",
        "steps": 3,
        "scheduler_events": 3,
        "preemption_events": 1,
        "oracle_checks": 0,
        "decision_numerical_checks": 0,
        "memory": {"peak_reserved_since_reset": 60, "cuda_device_used_bytes": 70},
    }
    quiet = [
        {
            "results": [
                {
                    "at_ns": t,
                    "scheduler_seen": True,
                    "running": 0,
                    "waiting": 0,
                    "requests": 0,
                }
            ]
        }
        for t in (2_100_000_000, 2_350_000_000)
    ]
    if defect == "oracle":
        worker["decision_numerical_checks"] = 1
    elif defect == "memory":
        worker["memory"]["peak_reserved_since_reset"] = 69
    elif defect == "pending":
        quiet[-1]["results"][0]["requests"] = 1

    def save(name, value, jsonl=False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(x) + "\n" for x in value) if jsonl else json.dumps(value)
        )

    save("cpu-contract.json", {"placement": {"client": [8, 9]}})
    save("timing/cell-trace.json", capture["plan"])
    save("timing/cell-client.json", capture)
    attempts = capture["requests"][:-1] if defect == "attempt" else capture["requests"]
    save("timing/cell-attempts.jsonl", attempts, True)
    save("timing/cell-quiescence.json", quiet)
    save("timing/cell-worker.json", {"results": [worker]})
    save("server/cell/steps.json", steps)
    save("server/cell/scheduler-completions.json", completed)
    save("server/cell/receipt.json", worker)
    geometry, storage = memory_fixture()
    save("server/cell/storage.json", storage)
    save("server/initial-storage.json", storage)
    save("server/asgi-events.jsonl", asgi, True)
    metrics = "\n".join(
        f'vllm:{name}{{model_name="study"}} 0'
        for name in (
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
            "num_preemptions_total",
        )
    )
    save(
        "timing/cell-server-metrics.json",
        [{"start_ns": 1, "end_ns": 2, "http_status": 200, "prometheus_text": metrics}],
    )
    slo = {"ttft_ms": 100, "tpot_ms": 100, "eos_ms": 100}
    boundary, client = boundaries(capture, asgi, steps, completed), client_resources(
        capture
    )
    audit = {
        "label": "cell",
        "slo_defined": True,
        "summary": evaluate(capture, slo),
        "boundaries": boundary,
        "client_resources": client,
        "harness_screen_passed": harness_gate(client, boundary),
        "native_calibration_stability": None,
        "client_sha256": sha(tmp_path / "timing/cell-client.json"),
        "trace_sha256": sha(tmp_path / "timing/cell-trace.json"),
    }
    if defect == "summary":
        audit["summary"]["slo_completed"] = 2
    save("timing/cell-audit.json", audit)
    protocol = {
        "torch_allocator_reserved_cap_bytes": 68,
        "resource_policy": {"driver_envelope_bytes": 76},
        "geometry": geometry,
        "persistent_budget_bytes": 4096,
    }
    if defect is None:
        result = cell(tmp_path, "cell", capture["plan"], slo, protocol)
        assert result["summary"]["offered"] == 2
        assert result["summary"]["slo_fraction_of_all_offered"] == 0.5
        assert result["boundaries"]["preemption_events"] == 1
    else:
        with pytest.raises(ValueError):
            cell(tmp_path, "cell", capture["plan"], slo, protocol)
