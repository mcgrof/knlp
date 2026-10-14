# SPDX-License-Identifier: MIT
"""Keep a delayed prefill and duplicate worker events out of decode evidence."""

import json

import pytest

from tools.kv.native.verify_timing import verify


def fixture(tmp_path):
    def event(batch, stamp, scheduled, decoded, counts):
        return dict(
            batch_id=batch,
            completed_monotonic=stamp,
            scheduled_tokens=scheduled,
            decode_request_ids=decoded,
            completed_tokens=counts,
            preempted=[],
        )

    path = tmp_path / "cell.json"
    row = dict(
        batch=2,
        output_tokens=256,
        output_tokens_per_request=128,
        seconds=100,
        end_to_end_output_tokens_per_second=2.56,
        request_metrics=[{"internal_request_id": rid} for rid in ["a", "b"]],
        completion_linked_decode=dict(steps=1, seconds=1, tokens_per_second=2),
        active_concurrency_qualified=False,
    )
    completions = [
        event(0, 40, {"a": 8192}, [], {"a": 1}),
        event(1, 70, {"a": 1, "b": 8192}, ["a"], {"a": 1, "b": 1}),
        event(2, 71, {"a": 1, "b": 1}, ["a", "b"], {"a": 1, "b": 1}),
        event(3, 72, {"a": 1, "b": 1}, ["a", "b"], {"a": 1, "b": 1}),
        event(4, 100, {"b": 1}, ["b"], {"b": 1}),
    ]
    device = [
        dict(
            events=[
                dict(
                    batch_id=3,
                    model_execute_cuda_ms=7,
                    scheduled_tokens={"a": 1, "b": 1},
                )
            ]
        )
    ]
    for target, data in [
        (path, row),
        (path.with_suffix(".completions.json"), completions),
        (path.with_suffix(".device.json"), device),
    ]:
        target.write_text(json.dumps(data))
    return path


def test_reader_excludes_delayed_prefill_and_draining_tail(tmp_path):
    result = verify(fixture(tmp_path))
    assert result["completed_host_seconds"] == 1
    assert result["worker_execute_cuda_ms"] == 7


def test_reader_rejects_duplicate_worker_join(tmp_path):
    path = fixture(tmp_path)
    device_path = path.with_suffix(".device.json")
    data = json.loads(device_path.read_text())
    data[0]["events"] *= 2
    device_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="one-to-one"):
        verify(path)
