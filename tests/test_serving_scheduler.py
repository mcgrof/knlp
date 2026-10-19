# SPDX-License-Identifier: MIT
import pytest
from tools.kv.native.serving_scheduler import admissions


def step(index, tokens, new=(), resumed=(), preempted=()):
    return {
        "batch_id": index,
        "scheduled_tokens": tokens,
        "new_request_ids": list(new),
        "resumed_request_ids": list(resumed),
        "preempted_request_ids": list(preempted),
    }


@pytest.mark.parametrize("via_new", [False, True])
def test_both_resumption_encodings_preserve_one_first_admission(via_new):
    rows = [
        step(0, {"r": 8192}, new=["r"]),
        step(1, {}, preempted=["r"]),
        step(
            2,
            {"r": 8217},
            new=["r"] if via_new else [],
            resumed=[] if via_new else ["r"],
        ),
    ]
    result = list(admissions(rows, 8192))
    assert [r["fresh"] for r in result] == [["r"], [], []]
    assert result[2]["resumed_via_new" if via_new else "resumed_via_cached"] == ["r"]


def test_repeated_new_id_without_preemption_is_not_silently_deduplicated():
    with pytest.raises(AssertionError, match="without preemption"):
        list(
            admissions(
                [step(0, {"r": 8192}, new=["r"]), step(1, {"r": 8192}, new=["r"])], 8192
            )
        )


def test_cached_resume_without_recorded_preemption_is_rejected():
    with pytest.raises(AssertionError):
        list(
            admissions(
                [step(0, {"r": 8192}, new=["r"]), step(1, {"r": 8200}, resumed=["r"])],
                8192,
            )
        )
