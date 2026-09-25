# SPDX-License-Identifier: GPL-2.0
"""The frozen question set, exercised against the defects it exists to prevent.

Each test names a way the freeze could fail open. A synthetic benchmark is used
so the suite runs anywhere; where the real pinned file is staged, the last test
also checks the real patient against it.
"""

from __future__ import annotations

import json
import os

import pytest

from research.cartridges_cas.scripts import c2_questions
from tests.cartridges_cas.entry_guard import assert_entry_point_is_sound

PINNED = os.environ.get("C2_BENCHMARK_JSON", "/data/cart_work/benchmark_v5.json")
PINNED_SHA = "82d34d9da47ab279d7aa89a6bdf298c0ac79f1e506e1dd0a3ea69a1ad5e2cb45"


def make_question(n, gold_slot=0, options=None):
    opts = options or [f"option {n}{c}" for c in "abcde"]
    return {
        "No": n,
        "question": f"question number {n}?",
        **{f"answer_{c}": opts[i] for i, c in enumerate("abcde")},
        "correct": opts[gold_slot],
    }


def write_benchmark(tmp_path, questions, patient="patient_02"):
    data = {
        patient: {
            "texts": {"text_0": "a clinical note"},
            "name": "Someone",
            "birthday": "1970-01-01",
            "diagnosis": "something",
            "questions": questions,
        }
    }
    p = tmp_path / "benchmark.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def twenty(tmp_path):
    qs = [make_question(n, gold_slot=n % 5) for n in range(20)]
    return write_benchmark(tmp_path, qs)


def test_ids_are_contiguous_and_in_numeric_order(twenty):
    f = c2_questions.freeze(twenty, "patient_02", per_patient=20)
    ids = [q["question_id"] for q in f["questions"]]
    assert ids == [f"patient_02_{n}" for n in range(20)]
    assert f["selection_is_exhaustive"] is True


def test_every_gold_resolves_to_exactly_one_option(twenty):
    f = c2_questions.freeze(twenty, "patient_02", per_patient=20)
    for q in f["questions"]:
        assert q["options"][q["gold_index"]] == q["gold_text"]
        matches = [
            o
            for o in q["options_normalised"]
            if o == q["options_normalised"][q["gold_index"]]
        ]
        assert len(matches) == 1


def test_gold_matching_two_options_is_a_hard_failure(tmp_path):
    # The same text in two slots: no scorer can say which one was meant.
    dup = ["same", "same", "c", "d", "e"]
    qs = [make_question(n) for n in range(19)]
    qs.append(make_question(19, gold_slot=0, options=dup))
    path = write_benchmark(tmp_path, qs)
    with pytest.raises(SystemExit):
        c2_questions.freeze(path, "patient_02", per_patient=20)


def test_gold_matching_no_option_is_a_hard_failure(tmp_path):
    qs = [make_question(n) for n in range(20)]
    qs[7]["correct"] = "an answer that is not among the options"
    path = write_benchmark(tmp_path, qs)
    with pytest.raises(SystemExit):
        c2_questions.freeze(path, "patient_02", per_patient=20)


def test_a_wrong_benchmark_hash_refuses(twenty):
    with pytest.raises(SystemExit):
        c2_questions.freeze(twenty, "patient_02", want_sha256="0" * 64)


def test_the_right_benchmark_hash_is_accepted(twenty):
    got = c2_questions.sha256_file(twenty)
    f = c2_questions.freeze(twenty, "patient_02", want_sha256=got)
    assert f["benchmark_sha256"] == got


def test_a_missing_patient_refuses(twenty):
    with pytest.raises(SystemExit):
        c2_questions.freeze(twenty, "patient_99", per_patient=20)


def test_question_set_hash_is_stable_across_invocations(twenty):
    a = c2_questions.freeze(twenty, "patient_02", per_patient=20)
    b = c2_questions.freeze(twenty, "patient_02", per_patient=20)
    assert a["question_set_sha256"] == b["question_set_sha256"]


def test_question_set_hash_moves_when_one_option_character_changes(tmp_path):
    qs = [make_question(n) for n in range(20)]
    before = c2_questions.freeze(
        write_benchmark(tmp_path, qs), "patient_02", per_patient=20
    )["question_set_sha256"]
    qs[3]["answer_c"] = qs[3]["answer_c"] + " revised"
    after = c2_questions.freeze(
        write_benchmark(tmp_path, qs), "patient_02", per_patient=20
    )["question_set_sha256"]
    assert before != after


def test_asking_for_more_questions_than_exist_refuses(tmp_path):
    qs = [make_question(n) for n in range(5)]
    path = write_benchmark(tmp_path, qs)
    with pytest.raises(SystemExit):
        c2_questions.freeze(path, "patient_02", per_patient=20)


def test_a_non_exhaustive_selection_is_labelled_as_such(tmp_path):
    qs = [make_question(n) for n in range(20)]
    path = write_benchmark(tmp_path, qs)
    f = c2_questions.freeze(path, "patient_02", per_patient=4)
    assert f["selection_is_exhaustive"] is False
    assert f["questions_available"] == 20 and f["questions_selected"] == 4


def test_the_entry_point_can_actually_be_executed():
    """A __main__ guard that calls a name defined below it raises NameError.

    That happened once in this program on a rented card. The check is cheap and
    the failure it catches is only visible at the most expensive moment.
    """
    assert_entry_point_is_sound(c2_questions)


@pytest.mark.skipif(not os.path.exists(PINNED), reason="pinned benchmark not staged")
def test_the_real_patient_freezes_to_twenty_exhaustive_questions():
    f = c2_questions.freeze(
        PINNED, "patient_02", want_sha256=PINNED_SHA, per_patient=20
    )
    assert f["questions_selected"] == 20
    assert f["selection_is_exhaustive"] is True
    assert [q["question_id"] for q in f["questions"]] == [
        f"patient_02_{n}" for n in range(20)
    ]
