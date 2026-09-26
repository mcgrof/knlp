# SPDX-License-Identifier: GPL-2.0
"""The guard as the real driver reaches it, not as a helper in isolation.

A guard function with passing unit tests enforces nothing if the program that
scores a candidate never calls it, or calls it after the expensive work has
already started. So these drive the actual entry point as a subprocess and
assert it refuses before it would have loaded anything.

That ordering is why these tests can run on a machine with no torch at all: if
any of them ever fails with ModuleNotFoundError instead of the expected
refusal, the guard has drifted below the model load and the test has caught
exactly the regression it exists for.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.cartridges_cas.scripts import c2_receiver_init, c2_target
from tests.cartridges_cas.entry_guard import assert_entry_point_is_sound

ROOT = Path(__file__).resolve().parents[2]
DRIVER = c2_receiver_init.__file__


def write_rows(path, n_correct, n=20):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "question_id": f"patient_02_{i}",
                        "correct": 1 if i < n_correct else 0,
                        "outcome": "exact",
                        "cap_hit": False,
                    }
                )
                + "\n"
            )


def write_questions(path, sha="frozenhash"):
    doc = {
        "patient": "patient_02",
        "question_set_sha256": sha,
        "benchmark_sha256": "b" * 64,
        "questions": [
            {
                "question_id": f"patient_02_{i}",
                "question": f"q{i}?",
                "options": [f"o{i}{c}" for c in "abcde"],
                "options_normalised": [f"o{i}{c}" for c in "abcde"],
                "gold_index": 0,
                "gold_text": f"o{i}a",
            }
            for i in range(20)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return str(path)


def run_driver(run_dir, questions, cartridge_list, target_file=None):
    cmd = [
        sys.executable,
        DRIVER,
        "--questions",
        str(questions),
        "--benchmark",
        "/nonexistent/benchmark.json",
        "--model",
        "Qwen/Qwen3-8b",
        "--cartridge-list",
        cartridge_list,
        "--run-dir",
        str(run_dir),
    ]
    if target_file:
        cmd += ["--target-file", str(target_file)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def questions(tmp_path):
    return write_questions(tmp_path / "QUESTIONS.json")


def assert_refused_before_loading_anything(r, needle):
    """The refusal must come from the guard, not from a missing dependency."""
    blob = r.stdout + r.stderr
    assert r.returncode != 0, blob
    assert "ModuleNotFoundError" not in blob, (
        "the driver got as far as importing a dependency before refusing, which "
        "means the guard has drifted below the expensive work:\n" + blob
    )
    assert needle in blob, blob


def test_the_driver_refuses_a_mapped_arm_with_no_target(run_dir, questions):
    r = run_driver(run_dir, questions, "mapped_learned_source.step0=/tmp/x.pt")
    assert_refused_before_loading_anything(r, "BLOCKED")


def test_the_driver_refuses_an_undeclared_arm(run_dir, questions):
    r = run_driver(run_dir, questions, "native_first_q.step0=/tmp/x.pt")
    assert_refused_before_loading_anything(r, "is not a declared arm")


def test_the_driver_refuses_a_renamed_mapped_arm(run_dir, questions):
    """Renaming past the guard was the old defect; it must not work now."""
    r = run_driver(run_dir, questions, "innocuous_arm.step0=/tmp/x.pt")
    assert_refused_before_loading_anything(r, "is not a declared arm")


def test_the_driver_refuses_a_stale_target(run_dir, questions):
    write_rows(run_dir / c2_target.FLOOR_ROWS, 2)
    write_rows(run_dir / c2_target.NATIVE200_ROWS, 11)
    c2_target.do_freeze(str(run_dir), str(questions))
    # A reference moves after the freeze.
    write_rows(run_dir / c2_target.FLOOR_ROWS, 5)
    r = run_driver(
        run_dir,
        questions,
        "mapped_learned_source.step0=/tmp/x.pt",
        target_file=run_dir / c2_target.TARGET_FILENAME,
    )
    assert_refused_before_loading_anything(r, "changed after the freeze")


def test_the_driver_refuses_a_target_frozen_against_other_questions(
    run_dir, questions, tmp_path
):
    write_rows(run_dir / c2_target.FLOOR_ROWS, 2)
    write_rows(run_dir / c2_target.NATIVE200_ROWS, 11)
    c2_target.do_freeze(str(run_dir), str(questions))
    other = write_questions(tmp_path / "OTHER.json", sha="a_different_set")
    r = run_driver(
        run_dir,
        other,
        "mapped_learned_source.step0=/tmp/x.pt",
        target_file=run_dir / c2_target.TARGET_FILENAME,
    )
    assert_refused_before_loading_anything(r, "not the one frozen against")


def test_the_driver_refuses_a_mapped_arm_when_the_references_left_no_headroom(
    run_dir, questions
):
    write_rows(run_dir / c2_target.FLOOR_ROWS, 6)
    write_rows(run_dir / c2_target.NATIVE200_ROWS, 6)
    c2_target.do_freeze(str(run_dir), str(questions))
    r = run_driver(
        run_dir,
        questions,
        "mapped_learned_source.step0=/tmp/x.pt",
        target_file=run_dir / c2_target.TARGET_FILENAME,
    )
    assert_refused_before_loading_anything(r, c2_target.NO_HEADROOM)


def test_a_reference_arm_passes_the_guard_and_proceeds(run_dir, questions):
    """The floor must NOT be blocked; it is what the target is built from.

    Here the driver is expected to get past the guard and then fail on a real
    dependency, which is the positive control for every test above.
    """
    r = run_driver(run_dir, questions, "receiver_no_cartridge=")
    blob = r.stdout + r.stderr
    assert r.returncode != 0
    assert "BLOCKED" not in blob and "is not a declared arm" not in blob
    assert "ModuleNotFoundError" in blob or "No module named" in blob, blob


def test_checkpoint_suffixes_do_not_change_arm_identity():
    for ckpt in ("step0", "step25", "step200"):
        arm, got, path = c2_receiver_init.parse_arm_spec(
            f"mapped_learned_source.{ckpt}=/tmp/x.pt"
        )
        assert arm == "mapped_learned_source"
        assert got == ckpt
        assert c2_target.classify_arm(arm) == "mapped"


def test_a_spec_without_an_equals_is_refused():
    with pytest.raises(SystemExit, match="has no '='"):
        c2_receiver_init.parse_arm_spec("native_first_p.step0")


def test_the_entry_point_can_actually_be_executed():
    assert_entry_point_is_sound(c2_receiver_init)
