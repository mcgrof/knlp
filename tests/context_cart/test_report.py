# SPDX-License-Identifier: MIT
"""Gates refuse incomplete or mismatched question populations."""

from __future__ import annotations

import pytest

from research.context_cart import dataset as D
from research.context_cart import evaluate as E
from research.context_cart import report as REP


def _rows(arm: str, worlds: int, correct: float, families=D.FAMILIES) -> list[dict]:
    rows = []
    for w in range(worlds):
        for f in families:
            rows.append(
                {
                    "arm": arm,
                    "world_id": f"w{w}",
                    "qid": f"{f}:{w}",
                    "family": f,
                    "attribute": "team",
                    "stratum": "ordinary",
                    "gold": "alpha",
                    "pred": "alpha" if (w * 7 + len(f)) % 10 < correct * 10 else "beta",
                    "em": (w * 7 + len(f)) % 10 < correct * 10,
                }
            )
    return rows


def test_reader_gate_needs_every_family():
    complete = _rows("full", 8, 1.0)
    assert REP.reader_gate(complete)["pass"]
    only_direct = [r for r in complete if r["family"] == "direct"]
    with pytest.raises(ValueError, match="families absent"):
        REP.reader_gate(only_direct)


def test_pilot_gate_refuses_partial_or_duplicated_arms():
    arms = {
        "full": _rows("full", 8, 1.0),
        "empty": _rows("empty", 8, 0.0),
        "untrained": _rows("untrained", 8, 0.0),
        "learned": _rows("learned", 8, 0.9),
        "wrong": _rows("wrong", 8, 0.0),
        "summary": _rows("summary", 8, 0.2),
    }
    gate = REP.pilot_gate(arms, "learned", "wrong")
    assert gate["checks"]["beats_wrong"]
    # one answered question cannot pass against complete controls
    partial = dict(arms, learned=[r for r in arms["learned"] if r["qid"] == "direct:0"])
    with pytest.raises(ValueError):
        REP.pilot_gate(partial, "learned", "wrong")
    # a control missing worlds is not silently intersected away
    short = dict(arms, summary=arms["summary"][:-4])
    with pytest.raises(ValueError, match="expected 32|populations differ"):
        REP.pilot_gate(short, "learned", "wrong")
    # duplicated rows are an error, not a larger sample
    doubled = dict(arms, learned=arms["learned"] + arms["learned"][:1])
    with pytest.raises(ValueError, match="duplicate"):
        REP.pilot_gate(doubled, "learned", "wrong")
    # gold mismatch between arms is an error
    altered = dict(arms, wrong=[dict(r, gold="omega") for r in arms["wrong"]])
    with pytest.raises(ValueError, match="gold answers differ"):
        REP.pilot_gate(altered, "learned", "wrong")


def test_paired_bootstrap_requires_matched_worlds():
    a, b = _rows("a", 6, 0.5), _rows("b", 6, 0.3)
    out = E.paired_bootstrap(a, b, resamples=50)
    assert out["worlds"] == 6
    with pytest.raises(ValueError):
        E.paired_bootstrap(a, b[:-4], resamples=50)
