# SPDX-License-Identifier: MIT
"""Guard fresh-admission attribution and immutable scheduler provenance."""
import json

import pytest

from tools.kv.native.serving_admission_audit import sha, summarize


def step(index, tokens, new=(), resumed=(), preempted=()):
    return {
        "batch_id": index,
        "scheduled_tokens": tokens,
        "new_request_ids": list(new),
        "resumed_request_ids": list(resumed),
        "preempted_request_ids": list(preempted),
    }


def test_recomputed_context_is_not_a_new_admission(tmp_path):
    path = tmp_path / "steps.json"
    path.write_text(
        json.dumps(
            [
                step(0, {"old": 1024}, new=["old"]),
                step(1, {"fresh": 1024, "old": 1}, new=["fresh"]),
                step(2, {"fresh": 1}, preempted=["old"]),
                step(3, {"old": 1040, "fresh": 1}, resumed=["old"]),
                step(4, {}),
            ]
        )
    )
    out = summarize(path, sha(path), 1024)
    assert out["steps"] == 5
    assert out["steps_by_kind"]["mixed_prefill_or_recompute_decode"] == 2
    assert out["new_request_admission_steps"] == 2
    assert out["new_request_admission_mixed_with_decode_steps"] == 1
    assert out["new_request_admission_count_histogram"] == {1: 2}
    assert out["resumed_request_events"] == out["preemption_events"] == 1
    assert out["steps_with_longer_recomputed_context"] == 1


def test_rejects_changed_capture_even_with_valid_json(tmp_path):
    path = tmp_path / "steps.json"
    path.write_text(json.dumps([step(1, {"new": 8192}, new=["new"])]))
    digest = sha(path)
    path.write_text(json.dumps([step(1, {"old": 1})]))
    with pytest.raises(ValueError, match="differ"):
        summarize(path, digest, 8192)


def test_rejects_chunked_prefill_in_frozen_whole_prompt_study(tmp_path):
    path = tmp_path / "steps.json"
    path.write_text(json.dumps([step(1, {"new": 256}, new=["new"])]))
    with pytest.raises(AssertionError):
        summarize(path, sha(path), 1024)


def test_rejects_duplicate_step_identity(tmp_path):
    path = tmp_path / "steps.json"
    path.write_text(json.dumps([step(1, {"old": 1}), step(1, {"old": 1})]))
    with pytest.raises(AssertionError):
        summarize(path, sha(path), 1024)
