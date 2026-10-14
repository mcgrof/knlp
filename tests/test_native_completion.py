# SPDX-License-Identifier: MIT
from tools.kv.native.completion import decode_intervals
from tools.kv.native import schedule
import pytest


def event(batch, time, scheduled, decoded, counts, preempted=()):
    return dict(
        batch_id=batch,
        completed_monotonic=time,
        scheduled_tokens=scheduled,
        decode_request_ids=decoded,
        completed_tokens=counts,
        preempted=list(preempted),
    )


def test_delayed_prefill_ahead_scheduling_and_draining_tail():
    # Both decode batches were scheduled before delayed prefill completed.
    rows = [
        event(0, 50, {"a": 8192}, [], {"a": 1}),
        event(1, 70, {"a": 1, "b": 8192}, ["a"], {"a": 1, "b": 1}),
        event(2, 71, {"a": 1, "b": 1}, ["a", "b"], {"a": 1, "b": 1}),
        event(3, 72, {"a": 1, "b": 1}, ["a", "b"], {"a": 1, "b": 1}),
        event(4, 100, {"b": 1}, ["b"], {"b": 1}),
    ]
    result = decode_intervals(rows, ["a", "b"])
    assert result["seconds"] == 1
    assert result["steps"] == 1
    assert result["tokens_per_second"] == 2


def test_preemption_or_new_admission_breaks_interval():
    rows = [event(0, 0, {"a": 5}, [], {"a": 1})]
    rows += [
        event(i, i, {"a": 1}, ["a"], {"a": 1}, ["a"] if i == 3 else [])
        for i in range(1, 7)
    ]
    result = decode_intervals(rows, ["a"])
    assert [r["batch_id"] for r in result["intervals"]] == [2, 5, 6]


def test_scheduled_without_returned_tokens_is_not_a_completion():
    rows = [event(i, i, {"a": 1}, ["a"], {"a": n}) for i, n in enumerate([1, 0, 1, 1])]
    assert decode_intervals(rows, ["a"])["steps"] == 1


def test_explicit_frontend_to_worker_ids(monkeypatch):
    monkeypatch.setattr(schedule, "_request_ids", {"has-dashes": "opaque-uuid"})
    assert schedule.internal_ids(["has-dashes"]) == ["opaque-uuid"]
    with pytest.raises(KeyError):
        schedule.internal_ids(["unknown"])
    monkeypatch.setattr(schedule, "_request_ids", {"a": "same", "b": "same"})
    with pytest.raises(ValueError):
        schedule.internal_ids(["a", "b"])
