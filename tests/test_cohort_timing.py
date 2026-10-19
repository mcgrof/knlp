# SPDX-License-Identifier: MIT
"""Reject incomplete, shifted or preempted host-clock measurement records."""

import copy

import pytest

from tools.kv.native.cohort_audit import verify


def record():
    return {
        "start_monotonic_ns": 1_000_000_000,
        "end_monotonic_ns": 3_000_000_000,
        "outputs": [
            {
                "internal_request_id": str(i),
                "prompt_token_ids": [i],
                "output_token_ids": [9] * 128,
                "finish_reason": "length",
                "num_cached_tokens": 0,
            }
            for i in range(2)
        ],
        "completions": [
            {
                "completed_monotonic": 1.01 + index * 0.01,
                "batch_id": index,
                "preempted": [],
                "scheduled_tokens": {"0": 1, "1": 1},
                "decode_request_ids": ["0", "1"] if index else [],
                "completed_tokens": {"0": 1, "1": 1},
            }
            for index in range(128)
        ],
    }


def test_known_complete_cohort():
    result = verify(record(), [[0], [1]])
    assert result["seconds"] == 2
    assert result["complete_request_tokens_per_second"] == 128
    assert result["full_active_completed_decode_steps"] == 127


@pytest.mark.parametrize("fault", ["clock", "missing", "duplicate", "preemption"])
def test_corrupted_or_incomplete_cohort(fault):
    row = copy.deepcopy(record())
    if fault == "clock":
        row["end_monotonic_ns"] = 2_000_000_000
    elif fault == "missing":
        row["completions"].pop()
    elif fault == "duplicate":
        row["completions"][-1]["batch_id"] = 0
    else:
        row["completions"][10]["preempted"] = ["0"]
    with pytest.raises(ValueError):
        verify(row, [[0], [1]])
