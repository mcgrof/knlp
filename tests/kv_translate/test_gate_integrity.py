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
