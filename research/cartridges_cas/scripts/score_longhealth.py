#!/usr/bin/env python3
"""Score a LongHealth multiple-choice answer, or say there was not one.

Every scorer this program has used for this benchmark gets it wrong, and they
get it wrong in different directions, so this states the rule as a ladder that
ends in an abstention rather than in a guess.

What the existing ones do. The library's own scorer searches the first
``<answer>`` tag in the raw generation without removing the reasoning block,
so a model that rehearses the format while thinking is scored on the
rehearsal; and when it finds no tag at all it does not guess randomly, it
hard-codes the first option. The paper-faithful evaluator strips the reasoning
and takes the last tag, which is right, but when there is no tag it feeds the
final three hundred characters of reasoning to a similarity match, and a long
prediction scores highest against the longest option roughly half the time.
The letter-protocol re-scorer is sound for its own prompt and wrong here,
because under this protocol the model emits option text.

The load-bearing part is the last step. LongHealth's options are unusually
confusable -- across the benchmark a majority of questions have two options
more than sixty percent similar and a third have two above eighty -- so a
nearest-option rule quietly turns wrong answers into right ones: Valsartan
lands on Candesartan, Once lands on Thrice, May 2020 lands on February 2020.
A similarity match is therefore allowed only when it clears an absolute floor
*and* beats the runner-up by a margin.

There is no letter step. It belongs in the evaluator whose prompt labels the
options A through E; the prompt this scores presents five unlabelled lines, so
a bare letter names nothing and abstains rather than selecting a slot by
position.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BARE_LETTER_RE = re.compile(r"^[\s*_#\-]*\(?([A-Ea-e])[\).:\-]?[\s*_]*$")

FLOOR = 0.55
MARGIN = 0.08
TAIL_MAX_CHARS = 400

UNRESOLVED = ("no_answer", "abstain", "bare_letter")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return " ".join(s.split()).strip(" \t.;:,*_\"'")


def _sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def score(
    text,
    options,
    correct_index,
    *,
    n_new_tokens=None,
    max_new_tokens=None,
    floor=FLOOR,
    margin=MARGIN,
):
    """One question's outcome, with the reason it came out that way."""
    out = {
        "correct": 0,
        "outcome": None,
        "pred": None,
        "picked": None,
        "answer_tag": False,
        "think_closed": None,
        "cap_hit": None,
        "top_ratio": None,
        "margin": None,
    }
    if n_new_tokens is not None and max_new_tokens is not None:
        out["cap_hit"] = bool(n_new_tokens >= max_new_tokens)
    has_open, has_close = "<think>" in text, "</think>" in text
    out["think_closed"] = has_close if (has_open or has_close) else None

    # An opened reasoning block that never closed means the generation was cut
    # off inside it; there is no answer after it to read.
    body = text.split("</think>")[-1] if has_close else ("" if has_open else text)
    tags = ANSWER_RE.findall(body)
    out["answer_tag"] = bool(tags)
    pred = norm(tags[-1]) if tags else ""
    if not pred:
        lines = [ln for ln in ANSWER_RE.sub(" ", body).splitlines() if ln.strip()]
        tail = norm(lines[-1]) if lines else ""
        pred = tail if 0 < len(tail) <= TAIL_MAX_CHARS else ""
        if pred:
            out["outcome"] = "tail_line"
    out["pred"] = pred
    if not pred:
        out["outcome"] = "no_answer"
        return out

    opts = [norm(o) for o in options]
    tagged = out["outcome"] == "tail_line"

    exact = [i for i, o in enumerate(opts) if o == pred]
    if len(exact) == 1:
        out.update(
            outcome="tail_exact" if tagged else "exact",
            picked=exact[0],
            correct=int(exact[0] == correct_index),
        )
        return out
    if len(exact) > 1:
        out.update(
            outcome="option_text_duplicated",
            picked=exact[0],
            correct=int(correct_index in exact),
        )
        return out

    if BARE_LETTER_RE.match(pred):
        # The options carry no letters in this prompt, so a letter is not a
        # choice. Selecting the slot it would name in a differently formatted
        # prompt would be inventing an answer the model did not give.
        out["outcome"] = "bare_letter"
        return out

    contained = [i for i, o in enumerate(opts) if o and (o in pred or pred in o)]
    if len(contained) == 1:
        out.update(
            outcome="tail_contain" if tagged else "contain",
            picked=contained[0],
            correct=int(contained[0] == correct_index),
        )
        return out

    ranked = sorted(((_sim(pred, o), i) for i, o in enumerate(opts)), reverse=True)
    out["top_ratio"] = round(ranked[0][0], 4)
    out["margin"] = round(ranked[0][0] - ranked[1][0], 4)
    if ranked[0][0] >= floor and (ranked[0][0] - ranked[1][0]) >= margin:
        out.update(
            outcome="tail_fuzzy" if tagged else "fuzzy",
            picked=ranked[0][1],
            correct=int(ranked[0][1] == correct_index),
        )
        return out
    out["outcome"] = "abstain"
    return out


def summarise(rows):
    n = len(rows)
    g = lambda p: sum(1 for r in rows if p(r))  # noqa: E731
    resolved = g(lambda r: r["outcome"] not in UNRESOLVED)
    correct = sum(r["correct"] for r in rows)
    return {
        "n": n,
        "n_correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "unresolved": g(lambda r: r["outcome"] in UNRESOLVED),
        "no_answer": g(lambda r: r["outcome"] == "no_answer"),
        "abstain": g(lambda r: r["outcome"] == "abstain"),
        "bare_letter": g(lambda r: r["outcome"] == "bare_letter"),
        "from_tail_line": g(lambda r: str(r["outcome"]).startswith("tail_")),
        "cap_hit": g(lambda r: r["cap_hit"]),
        "unterminated_think": g(lambda r: r["think_closed"] is False),
        "answer_tag_missing": g(lambda r: not r["answer_tag"]),
        "resolved_n": resolved,
        "accuracy_over_resolved": (round(correct / resolved, 4) if resolved else None),
    }
