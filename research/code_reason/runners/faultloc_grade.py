#!/usr/bin/env python3
"""Paper-faithful fault-localization grading (hunk-overlap Top-N, All/Any).

The Agentic Code Reasoning paper scores fault localization by matching a
model's ranked predicted regions against the gold buggy-file hunks: Top-N
"Any" succeeds if at least one gold hunk is covered by a top-N prediction,
Top-N "All" succeeds only if every gold hunk is covered. We reproduce that
here.

Two things are deliberate and load-bearing for validity:
- The score reads ONLY the model's final ranked-predictions block, capped at a
  fixed N, IDENTICALLY for standard and semi-formal. It never reads a
  certificate's cited evidence -- otherwise semi-formal would win simply by
  citing more lines (a format advantage, not better localization).
- "Covers" means same file AND overlapping line ranges; adjacency is not
  coverage.
"""

from __future__ import annotations

import os

MAX_PRED = 5


def _same_file(a, b):
    return os.path.basename(str(a)) == os.path.basename(str(b))


def _overlap(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def parse_predictions(answer, cap=MAX_PRED):
    """Extract the ranked (file,line_start,line_end) list from a final answer.

    Condition-neutral: the same extractor for both modes, capped at `cap`.
    Accepts the canonical key `ranked_locations` (or a couple of aliases) and
    ignores everything else (including any certificate).
    """
    answer = answer or {}
    for key in ("ranked_locations", "locations", "predictions"):
        v = answer.get(key)
        if isinstance(v, list):
            out = []
            for item in v[:cap]:
                if not isinstance(item, dict):
                    continue
                f = item.get("file")
                ls = item.get("line_start")
                le = item.get("line_end", ls)
                if f is None or ls is None:
                    continue
                out.append(
                    {
                        "file": f,
                        "line_start": int(ls),
                        "line_end": int(le if le is not None else ls),
                    }
                )
            return out
    return []


def _covers(pred, hunk):
    return _same_file(pred["file"], hunk["file"]) and _overlap(
        pred["line_start"], pred["line_end"], hunk["line_start"], hunk["line_end"]
    )


def grade_bug(predictions, gold_hunks, ns=(1, 3, 5)):
    """Return {top_N_all, top_N_any: bool} for one bug."""
    out = {}
    for n in ns:
        topn = predictions[:n]
        covered = [any(_covers(p, h) for p in topn) for h in gold_hunks]
        out[f"top_{n}_any"] = bool(gold_hunks) and any(covered)
        out[f"top_{n}_all"] = bool(gold_hunks) and all(covered)
    return out


def aggregate(per_bug, ns=(1, 3, 5)):
    """Mean over bugs of each All/Any Top-N flag."""
    n = len(per_bug) or 1
    agg = {}
    for k in [f"top_{x}_{m}" for x in ns for m in ("all", "any")]:
        agg[k] = round(sum(1 for b in per_bug if b.get(k)) / n, 4)
    agg["n_bugs"] = len(per_bug)
    return agg


def _self_test():
    # Single gold hunk, prediction covers it at rank 1.
    g1 = [{"file": "A.java", "line_start": 100, "line_end": 110}]
    p1 = [{"file": "A.java", "line_start": 105, "line_end": 105}]
    r = grade_bug(p1, g1)
    assert r["top_1_any"] and r["top_1_all"] and r["top_5_all"], r

    # Two gold hunks: rank-1 covers hunk A, rank-3 covers hunk B.
    g2 = [
        {"file": "A.java", "line_start": 100, "line_end": 110},
        {"file": "A.java", "line_start": 300, "line_end": 305},
    ]
    p2 = [
        {"file": "A.java", "line_start": 105, "line_end": 105},  # rank1 -> A
        {"file": "A.java", "line_start": 200, "line_end": 200},  # rank2 miss
        {"file": "A.java", "line_start": 302, "line_end": 302},  # rank3 -> B
    ]
    r2 = grade_bug(p2, g2)
    assert r2["top_1_any"] and not r2["top_1_all"], r2  # only A at rank1
    assert r2["top_3_all"] and r2["top_3_any"], r2  # both by rank3

    # Wrong file never covers.
    assert not grade_bug([{"file": "B.java", "line_start": 105, "line_end": 105}], g1)[
        "top_5_any"
    ]
    # Adjacent but non-overlapping is not coverage.
    assert not grade_bug([{"file": "A.java", "line_start": 111, "line_end": 120}], g1)[
        "top_5_any"
    ]

    # Condition-neutral parse: cap at 5, ignore a certificate's extra citations.
    ans = {
        "ranked_locations": [
            {"file": f"F{i}.java", "line_start": i, "line_end": i} for i in range(1, 9)
        ],
        "certificate": {"premises": [{"evidence": [{"file": "X"}]}]},
    }
    preds = parse_predictions(ans)
    assert len(preds) == 5 and preds[0]["file"] == "F1.java", preds

    # Aggregate.
    agg = aggregate([grade_bug(p1, g1), grade_bug(p2, g2)])
    assert agg["top_1_any"] == 1.0 and agg["top_1_all"] == 0.5, agg
    assert agg["n_bugs"] == 2
    print("[faultloc_grade] self-test PASS: All/Any Top-N + neutral parse + agg")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
