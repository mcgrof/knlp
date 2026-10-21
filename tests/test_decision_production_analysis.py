# SPDX-License-Identifier: MIT
"""Keep worker aggregate-math and answer-preservation objectives distinct."""

import copy
import json

import pytest

from tools.kv.decision.production_analysis import cohort_receipt, decisions, feasibility
from tools.kv.decision.statistics import paired_binary


def endpoints():
    native = {i: i < 1200 for i in range(1319)}
    candidate = dict(native)
    for i in range(25):
        candidate[i] = False
    for i in range(1200, 1240):
        candidate[i] = True
    perfect = {i: True for i in range(200)}
    result = {"gsm8k": paired_binary(native, candidate, 0.025 / 4)}
    for length in (1024, 8192):
        result[f"retrieval-{length}"] = paired_binary(perfect, perfect, 0.025 / 4)
        result[f"nll-{length}"] = {"relative_one_sided_bounds": [0.0, 0.0]}
    return result


def test_preservation_can_fail_while_aggregate_math_qualifies():
    result = decisions(endpoints())
    assert result == {"preservation": False, "aggregate_math": True}


@pytest.mark.parametrize(
    "endpoint", ["gsm8k", "retrieval-1024", "retrieval-8192", "nll-1024", "nll-8192"]
)
def test_aggregate_math_still_requires_every_other_endpoint(endpoint):
    result = copy.deepcopy(endpoints())
    if endpoint == "gsm8k":
        result[endpoint]["native_correct"] = 600
    elif endpoint.startswith("retrieval"):
        result[endpoint]["gross_loss_one_sided_bounds"][1] = 0.051
    else:
        result[endpoint]["relative_one_sided_bounds"][1] = 0.011
    assert decisions(result) == {"preservation": False, "aggregate_math": False}


def test_worker_family_zero_loss_feasibility_is_checked():
    result = feasibility(0.025 / 4)
    assert result["gsm8k_zero_loss_upper"] < 0.02
    assert result["retrieval_zero_loss_upper"] < 0.05
    with pytest.raises(ValueError):
        feasibility(1e-9)


@pytest.mark.parametrize(
    "defect", [None, "rpc", "preempted", "oracle", "torch", "driver", "steps"]
)
def test_every_quality_cohort_requires_actual_worker_resource_receipts(
    tmp_path, defect
):
    label = "quality-gsm8k-0000"
    server, quality = tmp_path / "server", tmp_path / "quality"
    (server / label).mkdir(parents=True)
    quality.mkdir()
    worker = {
        "label": label,
        "mode": "timing",
        "oracle_checks": 0,
        "decision_numerical_checks": 0,
        "preemption_events": 0,
        "steps": 1,
        "scheduler_events": 1,
        "memory": {"peak_reserved_since_reset": 60, "cuda_device_used_bytes": 70},
    }
    steps = [{"label": label, "preempted_request_ids": []}]
    if defect == "preempted":
        steps[0]["preempted_request_ids"] = ["request"]
    elif defect == "oracle":
        worker["decision_numerical_checks"] = 1
    elif defect == "torch":
        worker["memory"]["peak_reserved_since_reset"] = 69
    elif defect == "driver":
        worker["memory"]["cuda_device_used_bytes"] = 77
    elif defect == "steps":
        steps = []
    (server / label / "receipt.json").write_text(json.dumps(worker))
    (server / label / "steps.json").write_text(json.dumps(steps))
    rpc = {"results": [dict(worker)]}
    if defect == "rpc":
        rpc["results"][0]["steps"] = 2
    (quality / (label + "-worker.json")).write_text(json.dumps(rpc))
    protocol = {
        "torch_allocator_reserved_cap_bytes": 68,
        "resource_policy": {"driver_envelope_bytes": 76},
    }
    if defect is None:
        assert cohort_receipt(server, quality, label, protocol) == worker
    else:
        with pytest.raises(ValueError):
            cohort_receipt(server, quality, label, protocol)
