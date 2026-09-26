# SPDX-License-Identifier: GPL-2.0
"""The frozen target, and the ordering it is supposed to enforce.

A threshold chosen after the candidate's outcome is visible is not a threshold,
so most of these tests are about refusal: the cases where freezing must fail
because something has already happened that would let the bar be chosen to fit.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from research.cartridges_cas.scripts import c2_target
from tests.cartridges_cas.entry_guard import assert_entry_point_is_sound

N_QUESTIONS = 20


def qid(i):
    return f"patient_02_{i}"


def rows(n_correct, n=N_QUESTIONS, outcome="exact", cap_hit=False):
    return [
        {
            "question_id": qid(i),
            "correct": 1 if i < n_correct else 0,
            "outcome": outcome,
            "cap_hit": cap_hit,
        }
        for i in range(n)
    ]


def write_rows(path, r):
    with open(path, "w", encoding="utf-8") as f:
        for row in r:
            f.write(json.dumps(row) + "\n")


def write_questions(path, n=N_QUESTIONS, sha="deadbeef"):
    doc = {
        "question_set_sha256": sha,
        "questions": [{"question_id": qid(i)} for i in range(n)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    write_rows(d / c2_target.FLOOR_ROWS, rows(2))
    write_rows(d / c2_target.NATIVE200_ROWS, rows(11))
    write_questions(tmp_path / "QUESTIONS.json")
    return d


@pytest.fixture
def questions(tmp_path, run_dir):
    return str(tmp_path / "QUESTIONS.json")


# ---- the arithmetic -------------------------------------------------------


@pytest.mark.parametrize(
    "floor,native,expected",
    [
        (2, 11, 11),
        (0, 10, 9),  # 0.90*10 is exactly 9.0; the bar is nine, not ten
        (2, 20, 19),
        (5, 4, None),  # the native reference did not beat the floor
        (7, 7, None),  # a tie leaves no headroom either
        (0, 1, 1),
        (19, 20, 20),
    ],
)
def test_threshold_table(floor, native, expected):
    assert c2_target.threshold_from(floor, native) == expected


def test_threshold_matches_the_specified_formula_everywhere():
    """The integer form must equal ceil(F + 0.90*(N-F)) for every pair in range.

    Note what this does and does not show. It shows the two agree on counts out
    of twenty -- all 441 pairs, no exceptions. It does not show the float form
    is unsafe in general, and an earlier version of this file wrongly claimed
    that it rounds an exact nine up to ten. It does not: 0 + 0.90*10 is exactly
    9.0. Integers are used because exactness by construction does not depend on
    the input range, not because a counterexample was found in it.
    """
    for floor in range(N_QUESTIONS + 1):
        for native in range(floor + 1, N_QUESTIONS + 1):
            assert c2_target.threshold_from(floor, native) == math.ceil(
                floor + 0.90 * (native - floor)
            ), (floor, native)


def test_the_specifications_own_integer_expression_is_what_is_computed():
    """The queue writes the bar as (F + 9N + 9) // 10; compute exactly that."""
    for floor in range(N_QUESTIONS + 1):
        for native in range(floor + 1, N_QUESTIONS + 1):
            assert (
                c2_target.threshold_from(floor, native)
                == (floor + 9 * native + 9) // 10
            ), (floor, native)


def test_the_float_form_is_not_actually_wrong_on_these_inputs():
    """Pin the corrected fact so the false justification cannot come back."""
    assert 0 + 0.90 * (10 - 0) == 9.0
    assert math.ceil(0 + 0.90 * (10 - 0)) == 9
    assert c2_target.threshold_from(0, 10) == 9


def test_the_no_headroom_case_yields_the_named_verdict(tmp_path, questions):
    d = tmp_path / "flat"
    d.mkdir()
    write_rows(d / c2_target.FLOOR_ROWS, rows(6))
    write_rows(d / c2_target.NATIVE200_ROWS, rows(6))
    t = c2_target.do_freeze(str(d), questions)
    assert t["threshold"] is None
    assert t["verdict_if_no_headroom"] == c2_target.NO_HEADROOM


# ---- the ordering ---------------------------------------------------------


def test_a_clean_freeze_records_both_references_and_the_bar(run_dir, questions):
    t = c2_target.do_freeze(str(run_dir), questions)
    assert (t["floor_correct"], t["native200_correct"], t["threshold"]) == (2, 11, 11)
    assert set(t["inputs"]) == {c2_target.FLOOR_ROWS, c2_target.NATIVE200_ROWS}
    assert t["question_set_sha256"] == "deadbeef"
    assert os.path.exists(run_dir / c2_target.TARGET_FILENAME)


def test_freezing_refuses_once_a_mapped_arm_has_rows(run_dir, questions):
    write_rows(run_dir / "rows.mapped_learned_source.step25.jsonl", rows(9))
    with pytest.raises(SystemExit, match="REFUSING"):
        c2_target.do_freeze(str(run_dir), questions)


def test_freezing_refuses_an_undeclared_mapped_cache(run_dir, questions):
    """Preparation is allowed, but only when it is declared, not assumed."""
    carts = run_dir / "carts"
    carts.mkdir()
    (carts / "mapped_learned_source.pt").write_bytes(b"\0")
    with pytest.raises(SystemExit, match="undeclared"):
        c2_target.do_freeze(str(run_dir), questions)


def test_freezing_allows_a_declared_mapped_cache(run_dir, questions):
    """Building a map before the bar is fixed is legitimate; scoring is not."""
    carts = run_dir / "carts"
    carts.mkdir()
    blob = carts / "mapped_learned_source.pt"
    blob.write_bytes(b"a fitted map")
    manifest = {
        "why": "the map has to exist before the arm can run at all",
        "artifacts": [
            {
                "path": "carts/mapped_learned_source.pt",
                "sha256": c2_target.sha256_file(str(blob)),
            }
        ],
    }
    (run_dir / c2_target.PREPARATION_MANIFEST).write_text(json.dumps(manifest))
    t = c2_target.do_freeze(str(run_dir), questions)
    assert t["threshold"] == 11
    assert "carts/mapped_learned_source.pt" in t["declared_unscored_preparation"]


def test_a_declaration_that_does_not_match_the_disk_is_refused(run_dir, questions):
    carts = run_dir / "carts"
    carts.mkdir()
    blob = carts / "mapped_learned_source.pt"
    blob.write_bytes(b"a fitted map")
    manifest = {
        "artifacts": [{"path": "carts/mapped_learned_source.pt", "sha256": "0" * 64}]
    }
    (run_dir / c2_target.PREPARATION_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="hashes to"):
        c2_target.do_freeze(str(run_dir), questions)


def test_a_declared_score_is_still_refused(run_dir, questions):
    """Declaring a row file as preparation must not launder it into one."""
    rowfile = run_dir / "rows.mapped_learned_source.step25.jsonl"
    write_rows(rowfile, rows(9))
    manifest = {
        "artifacts": [
            {
                "path": "rows.mapped_learned_source.step25.jsonl",
                "sha256": c2_target.sha256_file(str(rowfile)),
            }
        ]
    }
    (run_dir / c2_target.PREPARATION_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="REFUSING"):
        c2_target.do_freeze(str(run_dir), questions)


def test_a_native_arm_artifact_does_not_block_the_freeze(run_dir, questions):
    """Only mapped arms are disqualifying; the references are supposed to exist."""
    write_rows(run_dir / "rows.native_first_p.step50.jsonl", rows(8))
    t = c2_target.do_freeze(str(run_dir), questions)
    assert t["threshold"] == 11


def test_freezing_twice_refuses(run_dir, questions):
    c2_target.do_freeze(str(run_dir), questions)
    with pytest.raises(SystemExit, match="already exists"):
        c2_target.do_freeze(str(run_dir), questions)


def test_a_missing_reference_blocks_rather_than_guessing(tmp_path, questions):
    d = tmp_path / "partial"
    d.mkdir()
    write_rows(d / c2_target.FLOOR_ROWS, rows(2))
    with pytest.raises(SystemExit, match="BLOCKED"):
        c2_target.do_freeze(str(d), questions)


def test_rows_that_do_not_cover_the_question_set_are_rejected(tmp_path, questions):
    d = tmp_path / "short"
    d.mkdir()
    write_rows(d / c2_target.FLOOR_ROWS, rows(2, n=19))
    write_rows(d / c2_target.NATIVE200_ROWS, rows(11))
    with pytest.raises(SystemExit, match="INVALID"):
        c2_target.do_freeze(str(d), questions)


def test_rows_with_the_wrong_ids_are_rejected(tmp_path, questions):
    d = tmp_path / "wrongids"
    d.mkdir()
    bad = rows(2)
    bad[0]["question_id"] = "patient_03_0"
    write_rows(d / c2_target.FLOOR_ROWS, bad)
    write_rows(d / c2_target.NATIVE200_ROWS, rows(11))
    with pytest.raises(SystemExit, match="do not cover"):
        c2_target.do_freeze(str(d), questions)


def test_an_unexpected_question_set_hash_refuses(run_dir, questions):
    with pytest.raises(SystemExit, match="INVALID"):
        c2_target.do_freeze(str(run_dir), questions, "a_different_hash")


# ---- checking after the fact ---------------------------------------------


def test_check_passes_on_an_untouched_run(run_dir, questions):
    c2_target.do_freeze(str(run_dir), questions)
    t = c2_target.do_check(str(run_dir), questions)
    assert t["threshold"] == 11


def test_check_fails_when_a_reference_moves_by_one_byte(run_dir, questions):
    c2_target.do_freeze(str(run_dir), questions)
    p = run_dir / c2_target.FLOOR_ROWS
    p.write_text(p.read_text().replace('"correct": 0', '"correct": 1', 1))
    with pytest.raises(SystemExit, match="changed after the freeze"):
        c2_target.do_check(str(run_dir), questions)


def test_check_fails_when_a_reference_is_removed(run_dir, questions):
    c2_target.do_freeze(str(run_dir), questions)
    os.remove(run_dir / c2_target.NATIVE200_ROWS)
    with pytest.raises(SystemExit, match="is gone"):
        c2_target.do_check(str(run_dir), questions)


def test_check_fails_against_a_different_question_set(run_dir, tmp_path, questions):
    c2_target.do_freeze(str(run_dir), questions)
    other = write_questions(tmp_path / "OTHER.json", sha="something_else")
    with pytest.raises(SystemExit, match="not the one frozen against"):
        c2_target.do_check(str(run_dir), other)


# ---- the guard the runner calls ------------------------------------------


def test_the_guard_is_inert_when_no_arm_is_mapped(run_dir, questions):
    assert (
        c2_target.require_target_for_mapped(
            str(run_dir), questions, ["receiver_no_cartridge", "native_first_p"]
        )
        is None
    )


def test_the_guard_blocks_a_mapped_arm_with_no_target(run_dir, questions):
    with pytest.raises(SystemExit, match="BLOCKED"):
        c2_target.require_target_for_mapped(
            str(run_dir), questions, ["mapped_learned_source"]
        )


def test_the_guard_admits_a_mapped_arm_once_a_target_is_frozen(run_dir, questions):
    c2_target.do_freeze(str(run_dir), questions)
    t = c2_target.require_target_for_mapped(
        str(run_dir), questions, ["mapped_learned_source"]
    )
    assert t["threshold"] == 11


def test_the_guard_blocks_a_mapped_arm_when_there_was_no_headroom(tmp_path, questions):
    d = tmp_path / "flat2"
    d.mkdir()
    write_rows(d / c2_target.FLOOR_ROWS, rows(6))
    write_rows(d / c2_target.NATIVE200_ROWS, rows(6))
    c2_target.do_freeze(str(d), questions)
    with pytest.raises(SystemExit, match=c2_target.NO_HEADROOM):
        c2_target.require_target_for_mapped(
            str(d), questions, ["mapped_fresh_first_p_source"]
        )


# ---- the comparator ------------------------------------------------------


def test_bad_counts_unresolved_and_capped_rows_once_each():
    r = rows(0, n=5)
    r[0]["outcome"] = "abstain"
    r[1]["outcome"] = "no_answer"
    r[2]["outcome"] = "bare_letter"
    r[3]["cap_hit"] = True
    r[4]["outcome"] = "abstain"
    r[4]["cap_hit"] = True  # both, still one bad row
    assert c2_target.count_bad(r) == 5
    assert c2_target.count_bad(rows(3, n=5)) == 0


def test_the_comparator_choice_is_written_into_the_frozen_file(run_dir, questions):
    t = c2_target.do_freeze(str(run_dir), questions)
    assert "same checkpoint" in t["bad_rule"]
    assert t["floor_bad"] == 0 and t["native200_bad"] == 0


def test_the_entry_point_can_actually_be_executed():
    assert_entry_point_is_sound(c2_target)


# ---- arm identity is declared, not spelled -------------------------------


def test_an_undeclared_arm_name_stops_the_run(run_dir, questions):
    """Not silently classified as a reference, which would bypass the guard."""
    with pytest.raises(SystemExit, match="is not a declared arm"):
        c2_target.require_target_for_mapped(
            str(run_dir), questions, ["native_first_p", "mapped_somethign_else"]
        )


def test_a_typo_in_a_mapped_arm_name_stops_the_run(run_dir, questions):
    """The failure mode the filename-prefix guard had.

    A guard that asks whether a name starts with "mapped_" lets a misspelled
    reference arm through as a reference and a misspelled mapped arm through
    as mapped-but-unrecognised. Declaring the set makes either one an error.
    """
    with pytest.raises(SystemExit, match="is not a declared arm"):
        c2_target.classify_arm("mapped_learned_sauce")
    with pytest.raises(SystemExit, match="is not a declared arm"):
        c2_target.classify_arm("native_first_q")


def test_renaming_a_mapped_arm_does_not_get_past_the_guard(run_dir, questions):
    """A name outside the registry is refused rather than treated as safe."""
    with pytest.raises(SystemExit, match="is not a declared arm"):
        c2_target.require_target_for_mapped(
            str(run_dir), questions, ["totally_innocent_arm"]
        )


def test_every_declared_arm_classifies(run_dir, questions):
    for a in c2_target.REFERENCE_ARMS:
        assert c2_target.classify_arm(a) == "reference"
    for a in c2_target.MAPPED_ARMS:
        assert c2_target.classify_arm(a) == "mapped"


def test_the_registry_matches_the_queues_arm_names():
    """The three arms the execution queue names, plus the floor."""
    assert set(c2_target.MAPPED_ARMS) == {
        "mapped_learned_source",
        "mapped_fresh_first_p_source",
    }
    assert "native_first_p" in c2_target.REFERENCE_ARMS
    assert "receiver_no_cartridge" in c2_target.REFERENCE_ARMS
