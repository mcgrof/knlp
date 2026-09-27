# SPDX-License-Identifier: GPL-2.0
"""One fixture per way the pilot actually broke.

These are not hypothetical faults. Each named case below happened on a rented
card, cost money, and in three instances produced a number that was reported
before anyone noticed it was wrong. The test is the record of what went wrong
and the proof that the same input is now refused rather than absorbed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from research.cartridges_cas.scripts import c2_runplan as rp

IDS = [f"patient_02_{i}" for i in range(20)]


def row(qid, gen="an answer", correct=1, outcome="exact", cap=False):
    return {
        "question_id": qid,
        "generated": gen,
        "correct": correct,
        "outcome": outcome,
        "cap_hit": cap,
    }


def full_rows(n_correct=20):
    return [row(q, correct=1 if i < n_correct else 0) for i, q in enumerate(IDS)]


def receipt(identity="stack-A", completed=True):
    return {"completed": completed, "execution_identity": identity}


# ---- the matrix is declared, so a shortfall is visible -------------------


def test_the_matrix_is_sixteen_cells_and_three_hundred_and_twenty_answers():
    """The pilot planned 320, delivered 314, and nothing noticed.

    Nothing in that driver knew how many answers it owed, so the shortfall was
    invisible until a reviewer counted them by hand afterwards.
    """
    t = rp.expected_totals()
    assert t == {"cells": 16, "questions": 320}
    ids = [c["id"] for c in rp.cells()]
    assert len(ids) == len(set(ids)) == 16
    assert rp.FLOOR_ARM in ids
    for arm in rp.INITIALIZER_ARMS:
        for c in rp.CHECKPOINTS:
            assert f"{arm}.step{c}" in ids


def test_the_floor_has_no_checkpoint_axis():
    floor = [c for c in rp.cells() if c["arm"] == rp.FLOOR_ARM]
    assert len(floor) == 1 and floor[0]["checkpoint"] is None


# ---- fault: missing step0 ------------------------------------------------


def test_zero_updates_is_never_waited_on_from_a_trainer():
    """FAULT: missing step0.

    The pilot asked the trainer for a zero-update checkpoint, the trainer
    silently declined, and the driver read the absence as 'that state does not
    exist'. All three arms lost the row. The state is the initializer file that
    already exists; it needs scoring, not producing.
    """
    zero = [c for c in rp.cells() if c["checkpoint"] == 0]
    assert len(zero) == 3
    for c in zero:
        assert rp.initializer_cell_is_not_a_trainer_output(c)
    for c in rp.cells():
        if c["checkpoint"] not in (0, None):
            assert not rp.initializer_cell_is_not_a_trainer_output(c)


# ---- fault: 14-of-20 evaluation -----------------------------------------


def test_a_fourteen_of_twenty_cell_is_partial_and_names_what_is_missing():
    """FAULT: 14-of-20 evaluation.

    The instance was reclaimed mid-cell. Fourteen rows landed. Counting lines
    cannot tell that apart from a finished cell, and the missing six were the
    difference between a resolved and an unresolved conclusion.
    """
    rows = [row(q) for q in IDS[:14]]
    s = rp.cell_status(rows, receipt(), IDS)
    assert s["status"] == "partial"
    assert s["missing"] == IDS[14:]
    assert len(s["missing"]) == 6


def test_a_partial_cell_is_never_skipped():
    rows = [row(q) for q in IDS[:14]]
    s = rp.cell_status(rows, receipt(), IDS)
    ok, why = rp.may_skip(s, receipt(), "stack-A")
    assert not ok and "not complete" in why


def test_rows_present_but_no_receipt_is_still_partial():
    """The writer can die between the last row and the receipt."""
    s = rp.cell_status(full_rows(), None, IDS)
    assert s["status"] == "partial" and "receipt" in s["reason"]


def test_duplicate_identities_are_invalid_not_merely_surplus():
    rows = full_rows()[:19] + [row(IDS[0])]
    s = rp.cell_status(rows, receipt(), IDS)
    assert s["status"] == "invalid" and "duplicate" in s["reason"]


def test_a_complete_cell_under_this_identity_may_be_skipped():
    s = rp.cell_status(full_rows(), receipt(), IDS)
    assert s["status"] == "complete"
    ok, _ = rp.may_skip(s, receipt(), "stack-A")
    assert ok


def test_a_cell_finished_by_another_stack_is_not_skipped():
    """Resuming across stacks mixes two software versions in one measurement."""
    s = rp.cell_status(full_rows(), receipt("stack-B"), IDS)
    ok, why = rp.may_skip(s, receipt("stack-B"), "stack-A")
    assert not ok and "different execution identity" in why


# ---- fault: only step25 checkpoint --------------------------------------


def test_an_arm_with_only_one_checkpoint_reports_the_rest_absent():
    """FAULT: only step25 checkpoint."""
    have = {25}
    for c in rp.CHECKPOINTS:
        rows = full_rows() if c in have else None
        s = rp.cell_status(rows, receipt() if c in have else None, IDS)
        assert s["status"] == ("complete" if c in have else "absent")


# ---- fault: missing optimizer state -------------------------------------


def test_a_bare_cache_cannot_resume_training():
    """FAULT: missing optimizer state.

    A key-value checkpoint is a state you can evaluate. Continuing from it
    resets the optimizer moments, the schedule position and the data order, so
    the resumed run is a different experiment under the same name.
    """
    ok, missing = rp.training_resume_is_possible({})
    assert not ok and "only a cache" in missing[0]
    ok, missing = rp.training_resume_is_possible(
        {"optimizer": {}, "scheduler": {}, "rng": {}}
    )
    assert not ok and missing == ["data_position"]
    ok, missing = rp.training_resume_is_possible({k: {} for k in rp.RESUME_STATE_KEYS})
    assert ok and missing == []


def test_a_none_valued_state_field_does_not_count_as_present():
    ok, missing = rp.training_resume_is_possible(
        {"optimizer": {}, "scheduler": {}, "rng": None, "data_position": 0}
    )
    assert not ok and missing == ["rng"]


# ---- fault: writer killed midrow ----------------------------------------


def test_a_truncated_final_row_is_dropped_and_reported(tmp_path):
    """FAULT: writer killed midrow."""
    p = tmp_path / "rows.jsonl"
    with open(p, "w") as f:
        for q in IDS[:3]:
            f.write(json.dumps(row(q)) + "\n")
        f.write('{"question_id": "patient_02_3", "gener')  # killed here
    rows, info = rp.read_rows_tolerating_a_killed_writer(str(p))
    assert len(rows) == 3
    assert info["truncated_lines"] == 1


def test_an_atomic_write_leaves_no_half_file(tmp_path):
    p = tmp_path / "receipt.json"
    rp.atomic_write_json(str(p), {"completed": True})
    assert json.load(open(p))["completed"] is True
    with pytest.raises(TypeError):
        rp.atomic_write_json(str(p), {"bad": {1, 2}})  # sets are not JSON
    assert json.load(open(p))["completed"] is True  # old content intact
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


# ---- fault: changed model or target -------------------------------------


def test_a_moved_target_stops_the_run():
    """FAULT: changed model or target."""
    frozen = {
        "question_set_sha256": "abc",
        "threshold": 6,
        "floor_correct": 2,
        "native200_correct": 6,
    }
    assert rp.require_unchanged_target(frozen, dict(frozen))
    with pytest.raises(rp.Refused, match="threshold"):
        rp.require_unchanged_target(frozen, {**frozen, "threshold": 5})
    with pytest.raises(rp.Refused, match="question_set_sha256"):
        rp.require_unchanged_target(frozen, {**frozen, "question_set_sha256": "zzz"})


def test_a_changed_model_identity_stops_the_run():
    frozen = {
        "model": "Qwen/Qwen3-8b",
        "revision": "b968826d",
        "tokenizer_sha256": "aeb13307",
    }
    assert rp.require_unchanged_model(frozen, dict(frozen))
    with pytest.raises(rp.Refused, match="revision"):
        rp.require_unchanged_model(frozen, {**frozen, "revision": "deadbeef"})


# ---- fault: failed mapping subprocess -----------------------------------


def test_a_failed_stage_stops_the_run_instead_of_leaving_a_hole():
    """FAULT: failed mapping subprocess.

    This is the one that wasted a whole pipeline pass. The map step failed, the
    shell carried on, both candidate arms then failed against a missing file,
    and the run printed ALL DONE having produced nothing.
    """
    assert rp.check_subprocess(0, "map fit")
    with pytest.raises(rp.Refused, match="map fit exited 1"):
        rp.check_subprocess(1, "map fit")
    done = subprocess.run([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(rp.Refused, match="exited 3"):
        rp.check_subprocess(done, "map fit")


# ---- fault: worker lost before release ----------------------------------


def test_an_unreachable_worker_is_not_a_released_allocation():
    """FAULT: worker lost before release.

    The pilot lost ssh and treated the allocation as gone. It happened to be
    gone. The inference was invalid anyway, and on a live instance it bills
    until somebody notices.
    """
    s = rp.release_state(
        provider_says_present=True, worker_reachable=False, terminate_acknowledged=False
    )
    assert s["released"] is False
    assert "losing contact is not releasing" in s["action"]


def test_only_the_provider_can_confirm_release():
    s = rp.release_state(
        provider_says_present=False,
        worker_reachable=False,
        terminate_acknowledged=False,
    )
    assert s["released"] is True and "provider" in s["evidence"]
    s = rp.release_state(
        provider_says_present=True, worker_reachable=True, terminate_acknowledged=True
    )
    assert s["released"] is False and "poll until" in s["action"]


def test_a_stage_that_would_eat_the_export_reserve_is_refused():
    ok, _ = rp.deadline_allows(0, 1000, 300, 600)
    assert ok
    ok, why = rp.deadline_allows(0, 1000, 300, 800)
    assert not ok and "export reserve" in why
    ok, why = rp.deadline_allows(900, 1000, 300, 10)
    assert not ok and "reserved for export" in why


# ---- reporting: unknowns stay unknown -----------------------------------


def test_an_incomplete_earlier_checkpoint_makes_the_earliest_pass_unknown():
    """The pilot's fresh arm, exactly: step0 partial, step25 passing.

    Reporting 25 as the earliest pass states a cost comparison that was never
    measured, because zero updates might already have cleared the bar.
    """
    scores = {0: {"correct": 2}, 25: {"correct": 8}, 50: {"correct": 7}}
    r = rp.earliest_passing_checkpoint(scores, 6, complete_checkpoints={25, 50})
    assert r["earliest_complete_passing"] == 25
    assert r["earliest_passing"] == rp.UNKNOWN
    assert r["incomplete_checkpoints_before_it"] == [0]


def test_a_complete_arm_resolves_its_earliest_pass():
    scores = {0: {"correct": 3}, 25: {"correct": 7}}
    r = rp.earliest_passing_checkpoint(scores, 6, complete_checkpoints={0, 25})
    assert r["earliest_passing"] == 25 and r["why"] == "resolved"


def test_update_counts_are_not_reported_as_costs():
    """The pilot called four times the updates four times the cost."""
    c = rp.cost_to_quality(100, 25)
    assert c["update_ratio"] == 4.0
    assert c["cost_ratio"] == rp.UNKNOWN
    assert "never timed per recipe" in c["why"]


def test_an_unknown_earliest_pass_propagates_into_the_cost_reading():
    c = rp.cost_to_quality(rp.UNKNOWN, 25)
    assert c["update_ratio"] == rp.UNKNOWN and c["cost_ratio"] == rp.UNKNOWN


def test_the_module_has_no_entry_point_to_misplace():
    """It decides; it does not run. So there is nothing to execute wrongly."""
    import ast

    src = open(rp.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    assert not any(
        isinstance(n, ast.If)
        and getattr(getattr(n.test, "left", None), "id", None) == "__name__"
        for n in tree.body
    ), "this module should stay decision-only"
