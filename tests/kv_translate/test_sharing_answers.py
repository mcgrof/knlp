# SPDX-License-Identifier: GPL-2.0
"""Short answers: a right one scores right, and each broken one is named."""

from __future__ import annotations

import pytest

from research.kv_translate.sharing import answers

# ---- short answers --------------------------------------------------------


@pytest.mark.parametrize(
    "output,accepted,correct",
    [
        ("Miquette Giraudy", ["Miquette Giraudy"], True),
        ("miquette giraudy.", ["Miquette Giraudy"], True),
        ("The Beatles", ["Beatles"], True),
        ("Answer: 1961", ["1961"], True),
        ("The answer is: New York City", ["NYC", "New York City"], True),
        ('"Paris"\nIt is the capital.', ["Paris"], True),
        ("<think>maybe Rome</think>Paris", ["Paris"], True),
        ("Paris, France", ["Paris"], False),
        ("Steve Hillage", ["Miquette Giraudy"], False),
        ("", ["Paris"], False),
        ("   \n  ", ["Paris"], False),
        ("<think>Paris", ["Paris"], False),
        ("the", ["the"], False),
    ],
)
def test_a_short_answer_is_right_only_when_it_equals_an_accepted_one(
    output, accepted, correct
):
    got = answers.score_document(
        output, accepted, stopped=True, new_tokens=5, max_new_tokens=48
    )
    assert got["correct"] is correct


def test_overlap_and_containment_are_reported_and_never_decide():
    got = answers.score_document(
        "Paris, France", ["Paris"], stopped=True, new_tokens=4, max_new_tokens=48
    )
    assert got["correct"] is False
    assert got["contains_accepted"] is True
    assert 0 < got["token_f1"] < 1


def test_a_wrong_answer_that_is_well_formed_is_healthy():
    got = answers.score_document(
        "Rome", ["Paris"], stopped=True, new_tokens=1, max_new_tokens=48
    )
    assert got["correct"] is False and got["health_events"] == []


@pytest.mark.parametrize(
    "output,stopped,new,events",
    [
        ("", True, 0, ["empty"]),
        ("Paris and then more and more", False, 48, ["unterminated"]),
        ("<think>still thinking", False, 48, ["unterminated_thinking", "unterminated"]),
        ("spam\n" * 9, True, 20, ["repetition"]),
        ("abcdefghijklmnopqrstuvwx" * 6, True, 40, ["repetition"]),
    ],
)
def test_each_way_of_being_broken_is_named(output, stopped, new, events):
    got = answers.health(
        output,
        stopped=stopped,
        new_tokens=new,
        max_new_tokens=48,
        parsed=bool(output.strip()) and "<think>" not in output,
    )
    for e in events:
        assert e in got


def test_stopping_below_the_cap_is_not_a_failure_to_stop():
    got = answers.health(
        "Paris", stopped=False, new_tokens=3, max_new_tokens=48, parsed=True
    )
    assert "unterminated" not in got
