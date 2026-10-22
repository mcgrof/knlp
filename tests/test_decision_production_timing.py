# SPDX-License-Identifier: MIT
"""Serving selection cannot bypass quality, use candidate pilots or alter inputs."""

import pytest

from tools.kv.decision.production_timing import validate_plan


def inputs():
    data = {
        "samples": {
            "development": [
                {"id": "dev", "kind": "nll", "length": 8192, "token_ids": [1] * 8321}
            ]
        }
    }
    quality = {
        "protocol_sha256": "protocol",
        "source_sha": "source",
        "preselected_serving_route": "aggregate_math",
        "serving_eligible_candidates": ["AF"],
    }
    protocol = {"source_sha": "source"}
    trace = {
        "mode": "open",
        "model": "study",
        "seed": 20261020,
        "arrival_seconds": 120,
        "drain_seconds": 120,
        "request_timeout_seconds": 240,
        "requests": [
            {
                "id": "r-0",
                "offset_ns": 1,
                "source_sample_id": "dev",
                "prompt_token_ids": [1] * 8192,
                "max_tokens": 128,
            }
        ],
    }
    plan = {
        "protocol_sha256": "protocol",
        "source_sha": "source",
        "purpose": "slo_calibration",
        "arm": "native_fa3",
        "cells": [{"label": "calibration", "trace": trace, "slo": None}],
    }
    return plan, quality, protocol, data


def test_native_only_calibration_accepts_exact_qualified_worker_and_inputs():
    plan, quality, protocol, data = inputs()
    validate_plan(plan, quality, protocol, "protocol", data)


@pytest.mark.parametrize(
    "defect",
    [
        "quality",
        "route",
        "source",
        "candidate_pilot",
        "unqualified_symmetric",
        "changed_input",
        "missing_slo",
        "warmup",
        "path",
    ],
)
def test_timing_gates_reject_changed_or_unqualified_work(defect):
    plan, quality, protocol, data = inputs()
    if defect == "quality":
        quality["serving_eligible_candidates"] = []
    elif defect == "route":
        quality["preselected_serving_route"] = "preservation"
    elif defect == "source":
        plan["source_sha"] = "different"
    elif defect == "candidate_pilot":
        plan["arm"] = "AF"
    elif defect == "unqualified_symmetric":
        plan.update(arm="SD", purpose="confirmation")
    elif defect == "changed_input":
        plan["cells"][0]["trace"]["requests"][0]["prompt_token_ids"][1] = 2
    elif defect == "missing_slo":
        plan["purpose"] = "load_selection"
    elif defect == "warmup":
        plan["cells"][0]["label"] = "warmup"
    else:
        plan["cells"][0]["label"] = ".."
    with pytest.raises(ValueError):
        validate_plan(plan, quality, protocol, "protocol", data)
