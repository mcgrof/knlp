# SPDX-License-Identifier: GPL-2.0
"""The gate's stated policy, checked at its boundaries.

Two of these exist because the numbers were wrong in ways that read as
correct. The unhealthy allowance was written as two examples out of
sixty-four, which is 3.125 percentage points rather than the two the policy
states, and an allowance expressed as a count over a development set silently
rescales with that set. And the recovery interval resampled its numerator
against a fixed observed denominator, which holds constant the very term whose
uncertainty a ratio inherits.

The third exists because the fix for the second was written and never called.
A helper that nothing invokes passes every test of the helper.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "research" / "kv_translate" / "eos_gate.py"


def _rows(tmp_path, arm_unhealthy, native_unhealthy=0, n=64, arm="cand"):
    """A minimal scored set: perfect tasks, controlled free-generation health."""
    lines = []
    for cond, unhealthy in (
        (arm, arm_unhealthy),
        ("target_native", native_unhealthy),
        ("anchor77", 0),
    ):
        for i in range(n):
            bad = i < unhealthy
            lines.append(
                {
                    "kind": "free",
                    "condition": cond,
                    "doc": f"d{i}",
                    "primary": 0.0 if bad else 1.0,
                    "healthy": 0.0 if bad else 1.0,
                    "empty": 0.0,
                    "too_short": 0.0,
                    "degenerate": 1.0 if bad else 0.0,
                    "stopped": 0.0,
                    "cap_hit": 0.0,
                    "repeat_4gram_frac": 0.0,
                }
            )
    for kind in ("retrieval", "cloze"):
        for cond, v in (
            (arm, 1.0),
            ("target_native", 1.0),
            ("empty_cache", 0.0),
            ("anchor77", 1.0),
            (f"{arm}|wrong", 0.0),
            ("anchor77|wrong", 0.0),
        ):
            for i in range(n):
                lines.append(
                    {"kind": kind, "condition": cond, "doc": f"d{i}", "primary": v}
                )
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return p


def _run(tmp_path, rows, arm="cand"):
    out = tmp_path / "gate.json"
    r = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--rows",
            str(rows),
            "--arms",
            arm,
            "--anchor",
            "anchor77",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text())["arms"][arm]


def test_one_excess_unhealthy_output_of_sixty_four_passes(tmp_path):
    d = _run(tmp_path, _rows(tmp_path, arm_unhealthy=1))
    assert d["health"]["excess_unhealthy_vs_native"] == pytest.approx(1 / 64)
    assert d["health"]["within_two_points"] is True, d["health"]


def test_two_excess_unhealthy_outputs_of_sixty_four_fail(tmp_path):
    """3.125 points, against an allowance of two. The old count-based bar
    admitted this."""
    d = _run(tmp_path, _rows(tmp_path, arm_unhealthy=2))
    assert d["health"]["excess_unhealthy_vs_native"] == pytest.approx(2 / 64)
    assert d["health"]["within_two_points"] is False, d["health"]


def test_the_allowance_is_two_points_not_two_examples():
    from research.kv_translate.eos_gate import UNHEALTHY_POINTS

    assert UNHEALTHY_POINTS == 0.02
    assert UNHEALTHY_POINTS != 2.0 / 64.0


def test_recovery_resamples_the_denominator_too():
    """A ratio's interval must move when its denominator's inputs move.

    With the numerator fixed and only native/empty varying, an interval built
    from a fixed observed denominator cannot widen at all.
    """
    from research.kv_translate.eos_gate import recovery_interval

    rng = np.random.default_rng(0)
    n = 64
    by = {}
    # Native and empty vary across documents; the candidate is constant.
    nat = rng.uniform(0.6, 1.0, n)
    emp = rng.uniform(0.0, 0.3, n)
    for cond, vals in (
        ("cand", np.full(n, 0.5)),
        ("target_native", nat),
        ("empty_cache", emp),
    ):
        by[("retrieval", cond)] = {
            f"d{i}": {"primary": float(v)} for i, v in enumerate(vals)
        }
    iv = recovery_interval(by, "retrieval", "cand")
    assert iv["hi"] > iv["lo"], iv
    assert iv["denominator"] == pytest.approx(nat.mean() - emp.mean())
    assert iv["draws_kept"] == iv["draws"]


def test_recovery_is_undefined_rather_than_rescued_when_native_does_not_lead():
    """No incremental gain means no recovery of one, and no epsilon."""
    from research.kv_translate.eos_gate import recovery_interval

    n = 16
    by = {
        ("retrieval", c): {f"d{i}": {"primary": v} for i in range(n)}
        for c, v in (("cand", 0.5), ("target_native", 0.2), ("empty_cache", 0.2))
    }
    iv = recovery_interval(by, "retrieval", "cand")
    assert iv["undefined"] is True
    assert iv["point"] is None
    assert "no incremental gain" in iv["why"]


def test_the_gate_defines_no_helper_that_nothing_calls():
    """A fix that is written and never wired in passes every test of itself.

    That happened here: the joint recovery bootstrap was added, the call site
    replacement silently matched nothing, and the gate kept using the estimator
    the fix replaced while the new function sat unused and correct.
    """
    tree = ast.parse(GATE.read_text())
    defined = {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    }
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    dead = sorted(defined - called - {"main"})
    assert not dead, f"defined but never called in {GATE.name}: {dead}"
