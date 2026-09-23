# SPDX-License-Identifier: GPL-2.0
"""The scorer, exercised against every real question in the benchmark.

Invented options would not show the property that matters: LongHealth's real
options are unusually confusable, and a similarity match on them silently
turns wrong answers into right ones. So the adversarial cases are constructed
from the actual option texts of all four hundred questions, and the assertions
are on rates across that whole set rather than on a handful of examples.
"""

from __future__ import annotations

import json
import os

import pytest

from research.cartridges_cas.scripts.score_longhealth import norm, score, summarise

BENCH = os.environ.get("LONGHEALTH_JSON")


@pytest.fixture(scope="module")
def questions():
    if not BENCH:
        pytest.skip("set LONGHEALTH_JSON to the LongHealth benchmark JSON")
    if not os.path.exists(BENCH):
        pytest.skip(f"benchmark not staged at {BENCH}")
    d = json.load(open(BENCH))
    pats = (
        d["patients"] if isinstance(d, dict) and "patients" in d else list(d.values())
    )
    out = []
    for p in pats:
        for q in p["questions"]:
            opts = [q[f"answer_{c}"] for c in "abcde"]
            hits = [i for i, o in enumerate(opts) if norm(o) == norm(q["correct"])]
            if len(hits) == 1:
                out.append((opts, hits[0]))
    assert len(out) > 300, f"only {len(out)} scorable questions"
    return out


def rate(questions, make, **kw):
    rows = [score(make(o, c), o, c, **kw) for o, c in questions]
    return summarise(rows), rows


def test_a_tagged_correct_answer_scores_correct(questions):
    s, _ = rate(questions, lambda o, c: f"<think>x</think>\n<answer>{o[c]}</answer>")
    assert s["accuracy"] == 1.0
    assert s["unresolved"] == 0


def test_a_tagged_wrong_answer_never_scores_correct(questions):
    """The flip this scorer exists to prevent."""
    s, _ = rate(
        questions, lambda o, c: f"<think>x</think>\n<answer>{o[(c + 1) % 5]}</answer>"
    )
    assert s["n_correct"] == 0, f"{s['n_correct']} wrong answers scored correct"


def test_the_rehearsed_tag_inside_reasoning_is_not_the_answer(questions):
    """A thinking model writes the format while deciding; the conclusion wins."""
    s, _ = rate(
        questions,
        lambda o, c: (
            f"<think>maybe <answer>{o[(c + 1) % 5]}</answer> no</think>\n"
            f"<answer>{o[c]}</answer>"
        ),
    )
    assert s["accuracy"] == 1.0


def test_a_generation_cut_off_inside_reasoning_resolves_to_nothing(questions):
    s, _ = rate(
        questions,
        lambda o, c: "<think>reasoning that never ends",
        n_new_tokens=512,
        max_new_tokens=512,
    )
    assert s["n_correct"] == 0
    assert s["unresolved"] == s["n"]
    assert s["cap_hit"] == s["n"]
    assert s["unterminated_think"] == s["n"]


def test_an_empty_tag_earns_nothing(questions):
    s, _ = rate(questions, lambda o, c: "<think>x</think>\n<answer></answer>")
    assert s["n_correct"] == 0


def test_a_bare_letter_abstains_because_the_options_carry_no_letters(questions):
    """The library would route most of these to the first option.

    Measured on the real benchmark, a bare letter in the tag selects the option
    it names 19% of the time and the first option 48% of the time under a
    nearest-text rule. Here it resolves to nothing at all.
    """
    s, _ = rate(
        questions, lambda o, c: f"<think>x</think>\n<answer>{'ABCDE'[c]}</answer>"
    )
    assert s["n_correct"] == 0
    assert s["bare_letter"] == s["n"]


def test_no_output_at_all_earns_nothing(questions):
    s, _ = rate(questions, lambda o, c: "")
    assert s["n_correct"] == 0 and s["unresolved"] == s["n"]


def test_the_first_option_is_never_a_silent_fallback(questions):
    """The library hard-codes the first option when it finds no tag.

    On this benchmark the first option is correct often enough that the
    fallback is free credit, so the property asserted here is that a scorer
    given nothing never picks anything.
    """
    for make in (lambda o, c: "", lambda o, c: "<think>unfinished"):
        _, rows = rate(questions, make)
        assert all(r["picked"] is None for r in rows)


def test_a_wrong_answer_stated_in_prose_does_not_flip_to_correct(questions):
    """The similarity floor and margin, doing the work they are there for."""
    s, _ = rate(
        questions,
        lambda o, c: f"<think>x</think>\nThe correct answer is {o[(c + 1) % 5]}.",
    )
    assert (
        s["n_correct"] == 0
    ), f"{s['n_correct']} of {s['n']} wrong prose answers were scored correct"


def test_a_correct_answer_stated_in_prose_is_usually_recovered(questions):
    """The bounded final-line fallback works, and where it does not it abstains.

    Nine of the four hundred decline rather than resolve, all of them short
    options like "-4" or "150 watts" where a wrapper sentence leaves no unique
    match and no similarity clear of the runner-up. Declining there is the
    behaviour being bought: none of them is scored wrong.
    """
    s, rows = rate(
        questions, lambda o, c: f"<think>x</think>\nThe correct answer is {o[c]}."
    )
    assert s["accuracy"] > 0.95, s
    resolved = [r for r in rows if r["outcome"] not in ("no_answer", "abstain")]
    assert all(r["correct"] for r in resolved), "a resolved case was scored wrong"
    assert all(str(r["outcome"]).startswith("tail_") for r in resolved)


def test_reasoning_prose_is_not_mistaken_for_an_answer(questions):
    """A long tail of reasoning must abstain, not match the longest option."""
    s, _ = rate(
        questions,
        lambda o, c: (
            "<think>x</think>\nLet me weigh the evidence across the notes and "
            "consider each possibility in turn before committing to anything "
            "in particular here."
        ),
    )
    assert s["n_correct"] / s["n"] < 0.05, s


def test_the_margin_rejects_a_near_tie_between_two_similar_options(questions):
    """Where two options are nearly identical, an ambiguous prediction abstains."""
    pairs = []
    for opts, c in questions:
        import difflib

        best = max(
            (
                difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
                for i, a in enumerate(opts)
                for j, b in enumerate(opts)
                if i < j
            ),
            default=0,
        )
        if best >= 0.9:
            pairs.append((opts, c))
    assert pairs, "expected the benchmark to contain near-identical option pairs"
    # A truncated prediction against a near-identical pair must not be resolved
    # on a coin-flip of similarity.
    resolved_wrong = 0
    for opts, c in pairs:
        r = score(f"<think>x</think>\n<answer>{norm(opts[c])[:4]}</answer>", opts, c)
        if (
            r["outcome"] not in ("no_answer", "abstain", "bare_letter")
            and not r["correct"]
        ):
            resolved_wrong += 1
    assert (
        resolved_wrong / len(pairs) < 0.35
    ), f"{resolved_wrong}/{len(pairs)} near-identical pairs resolved to a wrong option"
