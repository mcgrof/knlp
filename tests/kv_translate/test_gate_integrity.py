# SPDX-License-Identifier: GPL-2.0
"""Fixtures for the defects a decision-bearing gate actually suffered.

Each case here is a real failure that reached a recorded result, not a
hypothetical one. A gate that reports a verdict from incomplete or
self-inconsistent evidence is worse than one that refuses, because the refusal
is visible and the verdict is not, so every case asserts that the harness
fails closed rather than producing a plausible number.

The defects, in the order they were found:

Substring scoring accepted an answer with the gold code embedded in a longer
token. Three responses were scored correct that way, one of them the target
model's own, which silently moved the ceiling the arms were measured against.

A control was renamed to be per-arm, and the gate kept looking up the old
name. It found nothing, and recorded every arm as failing a criterion that had
never been evaluated -- including arms scoring 0.672 against a control scoring
zero.

The remaining cases are the ones the same class of bug would produce next: a
missing or duplicated row, a NaN reaching an aggregate, and a scored artifact
that is not the artifact whose identity was recorded.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from research.kv_translate.rescore_gold import (  # noqa: E402
    score_cloze,
    score_retrieval,
)

GOLD = "AWJG3EPU"


# ---- the scorer -------------------------------------------------------


def test_substring_scoring_accepts_a_longer_token_and_exact_does_not():
    """The defect verbatim: the run's own target model emitted this string."""
    s = score_retrieval("AWJG3EPU3EPU", GOLD)
    assert s["substring"] == 1.0, "the original metric accepted it, by construction"
    assert s["exact_any_run"] == 0.0
    assert s["exact_first_run"] == 0.0
    assert s["first_run"] == "AWJG3EPU3EPU"


@pytest.mark.parametrize(
    "produced",
    [
        " AWJG3EPU",
        "The archive registration code is AWJG3EPU.",
        "AWJG3EPU\n",
        "Answer: AWJG3EPU!",
        "(AWJG3EPU)",
    ],
)
def test_surrounding_whitespace_and_punctuation_are_not_part_of_the_answer(produced):
    assert score_retrieval(produced, GOLD)["exact_any_run"] == 1.0


@pytest.mark.parametrize(
    "produced",
    ["XAWJG3EPU", "AWJG3EPUX", "AWJG3EP", "AWJG-3EPU", "AWJG3EPU2"],
)
def test_extra_or_missing_alphanumerics_are_rejected(produced):
    assert score_retrieval(produced, GOLD)["exact_any_run"] == 0.0


def test_listing_several_candidates_is_not_answering():
    s = score_retrieval("Either ZZZZZZZZ or AWJG3EPU", GOLD)
    assert s["exact_any_run"] == 1.0, "the code is present as a whole run"
    assert s["exact_first_run"] == 0.0, "but it is not what was answered first"
    assert s["n_code_like_runs"] == 2, (
        "ZZZZZZZZ and AWJG3EPU are eight characters from the code alphabet; "
        "EITHER is neither the right length nor spelled from it"
    )


def test_ordinary_words_are_not_candidate_answers():
    """Without this the first-answer statistic measures the prose around it.

    Every word here but the code avoids I and O, so an alphabet test alone
    admits PASSAGE and STATES; the length test is what excludes them.
    """
    s = score_retrieval("Either the passage states AWJG3EPU", GOLD)
    assert s["exact_first_run"] == 1.0
    assert s["n_code_like_runs"] == 1


def test_a_refusal_is_a_miss_not_a_parse_error():
    s = score_retrieval("I do not know.", GOLD)
    assert s["exact_any_run"] == 0.0
    assert s["first_run"] == ""


def test_cloze_keeps_its_old_metric_and_gains_an_exact_one():
    gold = "Gambia 's national team only"
    assert score_cloze(gold, gold)["overlap"] == 1.0
    assert score_cloze(gold, gold)["exact_span"] == 1.0
    scrambled = "only team national 's Gambia"
    assert score_cloze(scrambled, gold)["overlap"] == 1.0, "bag of words, by design"
    assert score_cloze(scrambled, gold)["exact_span"] == 0.0, "order now visible"


# ---- the gate's inputs ------------------------------------------------


def _rows(conds, docs=("d1", "d2"), kinds=("retrieval",)):
    out = []
    for c in conds:
        for d in docs:
            for k in kinds:
                out.append(
                    {
                        "doc": d,
                        "kind": k,
                        "condition": c,
                        "produced": GOLD,
                        "primary": 1.0,
                        "correct": 1.0,
                        "n_tokens": 4,
                    }
                )
    return out


def _by_doc(rows, kind, cond):
    d = {}
    for r in rows:
        if r["kind"] == kind and r["condition"] == cond:
            d.setdefault(r["doc"], []).append(r["primary"])
    return d


def test_a_control_that_was_never_run_is_absent_not_zero():
    """The gate defect: looking up a control name the run does not produce.

    The lookup must come back empty so a caller can refuse, rather than
    returning a neutral value that reads as a real measurement of failure.
    """
    rows = _rows(["armA", "armA|wrong"])
    assert _by_doc(rows, "retrieval", "armA|wrong"), "the per-arm control exists"
    assert _by_doc(rows, "retrieval", "wrong_prompt") == {}, (
        "the old shared name must resolve to nothing, so a gate keyed on it "
        "cannot mistake absence for a measured failure"
    )


def test_every_arm_has_its_own_control():
    rows = _rows(["armA", "armA|wrong", "armB"])
    arms = {c for c in {r["condition"] for r in rows} if not c.endswith("|wrong")}
    missing = [a for a in arms if not _by_doc(rows, "retrieval", a + "|wrong")]
    assert missing == ["armB"], "armB is gated against a control that does not exist"


def test_a_duplicated_document_is_detected():
    rows = _rows(["armA"], docs=("d1", "d1"))
    counts = {}
    for r in rows:
        counts[(r["kind"], r["condition"], r["doc"])] = (
            counts.get((r["kind"], r["condition"], r["doc"]), 0) + 1
        )
        assert counts[(r["kind"], r["condition"], r["doc"])] <= 2
    assert max(counts.values()) == 2, "the fixture really does duplicate"
    dupes = [k for k, v in counts.items() if v > 1]
    assert dupes, "a row-per-cell check must see it"


def test_unequal_document_sets_between_a_pair_are_detected():
    rows = _rows(["armA"], docs=("d1", "d2")) + _rows(["armA|wrong"], docs=("d1",))
    a = set(_by_doc(rows, "retrieval", "armA"))
    b = set(_by_doc(rows, "retrieval", "armA|wrong"))
    assert a != b
    assert sorted(a & b) == ["d1"], "a paired comparison may only use the overlap"


def test_a_nan_does_not_pass_through_an_aggregate():
    rows = _rows(["armA"])
    rows[0]["primary"] = float("nan")
    vals = [v for vs in _by_doc(rows, "retrieval", "armA").values() for v in vs]
    assert any(math.isnan(v) for v in vals)
    assert math.isnan(sum(vals) / len(vals)), (
        "the mean is NaN rather than silently dropping the row, which is what "
        "makes a downstream refusal possible"
    )


def test_reordered_rows_do_not_change_a_paired_comparison():
    rows = _rows(["armA", "armA|wrong"], docs=("d1", "d2", "d3"))
    a = _by_doc(rows, "retrieval", "armA")
    b = _by_doc(list(reversed(rows)), "retrieval", "armA")
    assert a == b, "pairing is by document identity, never by row order"


# ---- artifact identity ------------------------------------------------


def test_a_mismatched_artifact_identity_is_refused():
    """A scored artifact must be the one whose identity was recorded."""
    import torch

    from research.kv_translate import freeze

    t = torch.randn(4, 3)
    h = freeze.tensor_hash(t)
    assert freeze.tensor_hash(t.clone()) == h, "identity is over content"
    t2 = t.clone()
    t2[0, 0] += 1e-3
    assert freeze.tensor_hash(t2) != h, "a changed weight is a different artifact"


def test_bf16_and_fp32_copies_of_one_map_are_not_the_same_byte_count():
    """The byte defect: a count assumed a precision nothing was stored at."""
    import torch

    n = 1000
    assert torch.zeros(n, dtype=torch.float32).numel() * 4 == 2 * (
        torch.zeros(n, dtype=torch.bfloat16).numel() * 2
    ), "a parameter-count estimate must name the precision it assumes"


# ---- the gold manifest is a precondition, not an optional extra -------


def test_rescoring_refuses_a_document_with_no_gold_entry(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps(
            {
                "doc": "unknown",
                "kind": "retrieval",
                "condition": "armA",
                "produced": GOLD,
                "primary": 1.0,
                "correct": 1.0,
            }
        )
        + "\n"
    )
    man = tmp_path / "gold.json"
    man.write_text(json.dumps({"documents": [{"doc": "d1", "retrieval_gold": GOLD}]}))
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "research/kv_translate/rescore_gold.py"),
            "--rows",
            str(rows),
            "--manifest",
            str(man),
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "INVALID" in (r.stdout + r.stderr)


# ---- a required check that fails must fail the run -------------------


def _recorder():
    """The sentinel's required-check bookkeeping, isolated for testing.

    Mirrors run_sentinel.require and its final verdict. The defect this
    guards against is a run reporting an unqualified pass while carrying a
    numerical check it did not honour: the committed sentinel recorded a
    tolerance of 1e-4 beside a measured 7.8e-4 and still said passed, because
    the boolean came from a different quantity than the field name implied.
    """
    checks = {"required_checks": {}}
    failures = []

    def require(name, measured, limit, why=""):
        ok = bool(measured <= limit)
        checks["required_checks"][name] = {
            "measured": float(measured),
            "limit": float(limit),
            "passed": ok,
            "why": why,
        }
        if not ok:
            failures.append(f"{name}: {measured:.3e} exceeds {limit:.3e}")
        return ok

    def verdict():
        req = checks["required_checks"]
        return bool(req) and all(c["passed"] for c in req.values()) and not failures

    return checks, failures, require, verdict


def test_a_failed_required_check_forces_a_failed_run():
    checks, failures, require, verdict = _recorder()
    require("fold_exact_in_float32", 1.4e-5, 1e-4)
    assert verdict(), "a run whose checks all hold must pass"
    require("fold_preserves_argmax", 0.03, 0.0)
    assert not verdict(), "one failed required check must fail the whole run"
    assert failures and "fold_preserves_argmax" in failures[0]


def test_a_recorded_limit_is_the_one_that_is_enforced():
    """The exact shape of the defect: measured above limit, still 'passed'."""
    checks, _, require, verdict = _recorder()
    require("fold_kl", 7.842e-4, 1e-4)
    rec = checks["required_checks"]["fold_kl"]
    assert rec["measured"] > rec["limit"]
    assert rec["passed"] is False
    assert verdict() is False, (
        "a limit that appears in the record must gate the verdict; if it is "
        "not meant to gate, it must not be recorded as a limit"
    )


def test_a_run_with_no_required_checks_does_not_pass_by_default():
    _, _, _, verdict = _recorder()
    assert verdict() is False, "an empty check set is not evidence of anything"


def test_near_miss_analysis_counts_deletions():
    """The diagnostic defect that closed a branch it should not have.

    Scoring requires a run of the code's exact length, which is right. Reusing
    that filter to ask how close a wrong answer came discards every dropped
    character, and dropped characters were fourteen of the sixteen near misses.
    """
    from research.kv_translate.rescore_gold import edit_distance, runs

    gold = "XYFJ3KLN"
    dropped = "XYJ3KLN"
    assert len(dropped) != len(gold)
    assert edit_distance(dropped, gold) == 1
    same_length_only = [x for x in runs(dropped) if len(x) == len(gold)]
    assert same_length_only == [], "the scoring filter hides this answer"
    unfiltered = [x for x in runs(dropped)]
    assert min(edit_distance(x, gold) for x in unfiltered) == 1


def _example(prompt="the code is ABCD1234 here", answer="ABCD1234"):
    class E:
        doc_id = "d0"
        kind = "code"
        query = "What is the code?"
        meta = {"code": answer}

    e = E()
    e.answer = answer
    e.prompt_ids = list(range(8))
    return e


class _Tok:
    """Whitespace tokeniser: one id per character, so lengths are checkable."""

    eos_token_id = 999

    def __call__(self, s, return_tensors=None, add_special_tokens=None):
        import torch

        class R:
            input_ids = torch.tensor([[ord(c) for c in s]])

        return R()

    def decode(self, ids, skip_special_tokens=True):
        return "the code is ABCD1234 here"


def test_eos_switch_changes_only_the_end_label():
    """The termination ablation must differ by exactly one label.

    If the off arm differs by anything else -- a dropped answer token, a
    different example, a shifted offset -- it is not an ablation of
    termination and whatever it measures is not attributable to it.
    """
    from research.kv_translate.objective_examples import supervised_targets

    tok, e = _Tok(), _example()
    q_on, lab_on = supervised_targets(tok, e, tok.eos_token_id, supervise_eos=True)
    q_off, lab_off = supervised_targets(tok, e, tok.eos_token_id, supervise_eos=False)
    assert q_on.tolist() == q_off.tolist()
    assert lab_on == lab_off + [tok.eos_token_id]
    assert len(lab_on) - len(lab_off) == 1


def test_off_arm_is_not_validated_against_the_on_arm_invariant():
    """Each arm is checked against its own end-of-sequence invariant.

    Validating the off arm against "labels end at EOS" would abort every run;
    reporting that check as passed would be worse.
    """
    from research.kv_translate.objective_examples import verify_examples

    tok, e = _Tok(), _example()
    assert verify_examples(tok, [e], tok.eos_token_id, supervise_eos=True) == []
    assert verify_examples(tok, [e], tok.eos_token_id, supervise_eos=False) == []


def test_acceptance_limit_is_declared_not_derived():
    """The limit must not move to meet a measurement.

    This check was briefly rewired to accept whatever a square-root-of-width
    heuristic predicted, on the argument that 1e-6 was unreachable for a
    4096-wide float32 dot product. The A100 receipts refute that argument:
    the same artifact agrees to 2.8e-7 there. The heuristic is kept as a
    reported diagnostic, and the acceptance limit stays where it was declared,
    so that a device or implementation change fails the check instead of
    being absorbed by it.
    """
    from research.kv_translate.qualify_operator import (
        LIMITS,
        heuristic_error_scale,
    )

    assert LIMITS["operator_matches_reference"] == 1e-6
    assert LIMITS["operator_is_deterministic"] == 0.0
    assert LIMITS["serving_cast_within_one_step"] == 1.001
    # The heuristic survives as a number to report, and is not the limit.
    assert heuristic_error_scale(4096, "float32") > LIMITS["operator_matches_reference"]
    assert heuristic_error_scale(4096, "float64") < heuristic_error_scale(
        4096, "float32"
    )
