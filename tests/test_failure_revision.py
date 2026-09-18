"""CPU tests for the failure-conditioned revision harness.

They cover the four checks required before any GPU run: no gold data in
model inputs, every model role charged, deadline and truncation
semantics, and resume safety (no reuse of a completed run, stable
seeds).
"""

import asyncio
import json
import os
from fractions import Fraction

import pytest

from failure_revision import prompts as P
from failure_revision.arms import ARMS, Budget, Ledger, Runner
from failure_revision.backend import FakeBackend
from failure_revision.grader import extract_answer, grade, parse_gold

ITEM = {
    "id": "q1",
    "question": "Tom has 3 apples and buys 4 more. How many?",
    "gold": "7/1",
}
REF = "Tom starts with 3 apples and then buys 4 more apples in total.\n3+4=7"
REFS = {"q1": REF}
FAILS = [f"attempt {k}: 3+4=8 so \\boxed{{8}}" for k in range(64)]


def responder_factory(solve_every=0):
    def respond(prompt, seed):
        if "Describe their shared approach" in prompt:
            return "summary note: recount the apples"
        if "write a strategy" in prompt:
            return "plan note: add the two quantities"
        if "Identify the step most likely" in prompt:
            return "critique: the addition is wrong"
        if solve_every and seed % solve_every == 0:
            return "3+4=7 \\boxed{7}"
        return "3+4=8 \\boxed{8}"

    return respond


def run_arm(tmp_path, arm, budget, responder, sec_per_token=0.0, tag="t"):
    b = FakeBackend(responder, sec_per_token=sec_per_token)
    led = Ledger(str(tmp_path))
    r = Runner(b, led, REFS, tag)
    rec = asyncio.run(r.run(ITEM, FAILS, arm, 0, budget, 1.2))
    led.close()
    rows = [json.loads(line) for line in open(os.path.join(tmp_path, "calls.jsonl"))]
    return rec, rows, b


# ------------------------------------------------------------------ grader
@pytest.mark.parametrize(
    "text,gold,ok",
    [
        ("so the answer is \\boxed{18}", "18", True),
        ("\\boxed{18.00}", "18", True),
        ("\\boxed{\\$1,250}", "1250", True),
        ("\\boxed{18 \\text{ dollars}}", "18", True),
        ("\\boxed{\\frac{3}{4}}", "0.75", True),
        ("\\boxed{50\\%}", "50", True),
        ("\\boxed{3, 4}", "3", False),
        ("The total is 1,000 dollars.", "1000", True),
        ("no numbers here", "1", False),
        ("\\boxed{17}", "18", False),
    ],
)
def test_grader_cases(text, gold, ok):
    assert grade(text, Fraction(gold))["correct"] is ok


def test_grader_multiple_is_malformed():
    ex = extract_answer("\\boxed{3, 4}")
    assert ex["value"] is None and ex["status"] == "malformed_multiple"


def test_parse_gold():
    assert parse_gold("blah\n#### 1,234") == 1234
    assert parse_gold("x #### -5") == -5


# ------------------------------------------------- 1. no gold in any input
@pytest.mark.parametrize("arm", ARMS)
def test_no_gold_in_model_inputs(tmp_path, arm):
    _, rows, b = run_arm(
        tmp_path, arm, Budget("tokens", tokens=3000), responder_factory()
    )
    assert b.calls, "no model calls made"
    for c in b.calls:
        for line in REF.splitlines():
            if len(line) > 40:
                assert line not in c["prompt"]
        assert "7/1" not in c["prompt"]


def test_leak_is_refused():
    with pytest.raises(AssertionError):
        P.check_no_gold("prompt with " + REF.splitlines()[0], "q1", REFS)


# --------------------------------------------- 2. every model role charged
@pytest.mark.parametrize(
    "arm,roles",
    [
        ("B", {"summary", "revised"}),
        ("D", {"plan", "revised"}),
        ("R", {"solve", "critique", "correction"}),
    ],
)
def test_all_roles_charged(tmp_path, arm, roles):
    rec, rows, _ = run_arm(
        tmp_path, arm, Budget("tokens", tokens=2000), responder_factory()
    )
    assert roles <= {r["role"] for r in rows}
    assert rec["gen_tokens"] == sum(r["gen_tokens"] for r in rows)
    assert rec["gen_tokens"] <= 2000
    aux = [r for r in rows if r["role"] in ("summary", "plan", "critique")]
    assert aux and all(not r["graded"] for r in aux)
    assert rec["n_aux"] == len(aux)


def test_summary_sees_latest_screen_failures(tmp_path):
    _, rows, b = run_arm(
        tmp_path, "B", Budget("tokens", tokens=400), responder_factory()
    )
    first = b.calls[0]["prompt"]
    assert (
        "attempt 63:" in first and "attempt 60:" in first and "attempt 59:" not in first
    )


def test_plan_arm_sees_no_failure_contents(tmp_path):
    _, rows, b = run_arm(
        tmp_path, "D", Budget("tokens", tokens=400), responder_factory()
    )
    assert "attempt 63:" not in b.calls[0]["prompt"]
    assert "64 earlier attempts" in b.calls[0]["prompt"]


# ------------------------------------- 3. deadline and truncation semantics
def test_success_after_deadline_not_counted(tmp_path):
    # Each solve takes 0.3 s; budget 0.2 s: the call is cut at the deadline.
    rec, rows, _ = run_arm(
        tmp_path,
        "A",
        Budget("time", seconds=0.2),
        responder_factory(solve_every=1),
        sec_per_token=0.3 / 15,
    )
    assert not rec["solved"]
    assert rows[-1]["finish"] == "deadline" and not rows[-1]["graded"]


def test_success_within_deadline_counted(tmp_path):
    rec, rows, _ = run_arm(
        tmp_path,
        "A",
        Budget("time", seconds=5.0),
        responder_factory(solve_every=1),
        sec_per_token=0.001,
    )
    assert rec["solved"] and rec["t_success"] <= 5.0
    assert rows[-1]["counted_success"]


def test_stops_at_first_success(tmp_path):
    rec, rows, _ = run_arm(
        tmp_path, "A", Budget("tokens", tokens=100000), responder_factory(solve_every=3)
    )
    assert rec["solved"]
    assert sum(r["counted_success"] for r in rows) == 1
    assert rows[-1]["counted_success"]


def test_truncation_flagged(tmp_path):
    def long(prompt, seed):
        return "x" * 5000 + " \\boxed{7}"

    rec, rows, _ = run_arm(tmp_path, "A", Budget("tokens", tokens=9000), long)
    assert rows[0]["finish"] == "length" and rec["n_truncated_out"] >= 1
    assert not rec["solved"]


def test_token_budget_respected(tmp_path):
    rec, rows, _ = run_arm(
        tmp_path, "C", Budget("tokens", tokens=500), responder_factory()
    )
    assert rec["gen_tokens"] <= 500
    assert all(r["temperature"] == 1.2 for r in rows)


# ------------------------------------------------------ 4. resume safety
def test_seeds_stable_and_distinct(tmp_path):
    _, rows1, _ = run_arm(
        tmp_path / "a", "A", Budget("tokens", tokens=300), responder_factory()
    )
    _, rows2, _ = run_arm(
        tmp_path / "b", "A", Budget("tokens", tokens=300), responder_factory()
    )
    s1 = [r["seed"] for r in rows1]
    assert s1 == [r["seed"] for r in rows2]
    assert len(set(s1)) == len(s1)
    _, rows3, _ = run_arm(
        tmp_path / "c", "C", Budget("tokens", tokens=300), responder_factory()
    )
    assert not set(s1) & {r["seed"] for r in rows3}


def test_completed_runs_are_recorded_for_skip(tmp_path):
    rec, _, _ = run_arm(
        tmp_path, "A", Budget("tokens", tokens=300), responder_factory()
    )
    led = Ledger(str(tmp_path))
    assert rec["run_key"] in led.done_runs()
    led.close()


def test_bound_trace():
    from failure_revision.backend import CharTokenizer

    text = "a" * 2000
    b, cut = P.bound_trace(CharTokenizer(), text)
    assert cut and len(b) == P.TRACE_HEAD + P.TRACE_TAIL + len(P.TRUNC_MARK)
