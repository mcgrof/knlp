# SPDX-License-Identifier: GPL-2.0
"""Rescore a finished task run against recovered gold answers.

The original scorer asked whether the gold code occurred anywhere in what the
model said. That accepts an answer with the code embedded in a longer token --
a model that emits ``AWJG3EPU3EPU`` when the code is ``AWJG3EPU`` did not
return the code, it returned something containing it -- and three such
responses were scored correct, one of them the target model's own.

This recomputes both metrics side by side rather than replacing one with the
other. The original substring score stays, because it is what the recorded
verdict was computed from and rewriting it would make the history unreadable;
the exact score is added beside it. Neither is allowed to be the only number
reported.

The exact rule: split the response into maximal alphanumeric runs and require
one of them to equal the gold code. Surrounding whitespace and punctuation are
accepted because they are not part of the answer; additional alphanumeric
characters are not, because they change the string the answer is. Whether the
*first* such run is the gold is recorded separately, since a model that lists
several candidates has not answered either.

The target model's own output is never treated as gold. Where its answer and
an arm's answer are compared, that is a descriptive similarity and is labelled
as one.

Env: none; reads JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.tasks import CODE_ALPHABET  # noqa: E402

RUN = re.compile(r"[A-Za-z0-9]+")
# The planted codes are drawn from an alphabet that omits the characters a
# reader confuses: no 0, 1, I or O. Requiring a candidate to be spelled from
# that set is what separates a wrong code from an ordinary word -- without it
# "EITHER" is a six-character run and counts as an answer the model offered,
# which makes the first-answer statistic a measure of the surrounding prose.
CODE_CHARS = set(CODE_ALPHABET)


def runs(text: str) -> list:
    return RUN.findall(text.upper())


def code_like(x: str, gold: str) -> bool:
    """A run that could be an answer: the code's length, the code's alphabet.

    Length is what does the work. Restricting to the alphabet alone is not
    enough, because ordinary words avoid I and O often enough to qualify --
    "PASSAGE" and "STATES" both do -- and every one of them would then count
    as an answer the model offered.
    """
    return len(x) == len(gold) and set(x) <= CODE_CHARS


def score_retrieval(produced: str, gold: str) -> dict:
    r = runs(produced)
    g = gold.upper()
    cands = [x for x in r if code_like(x, g)]
    long_runs = [x for x in r if len(x) >= 6]
    return {
        "substring": float(g in produced.upper()),
        "exact_any_run": float(g in r),
        "exact_first_run": float(bool(cands) and cands[0] == g),
        # Descriptive, and deliberately wider than the candidate rule: an
        # answer that ran the code together with something else has no
        # candidate at all, and that is the case worth being able to see.
        "first_run": long_runs[0] if long_runs else "",
        "n_code_like_runs": len(cands),
    }


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_cloze(produced: str, gold: str) -> dict:
    want = normalise(gold).split()
    got = normalise(produced).split()
    if not want:
        return {"overlap": 0.0, "exact_prefix": 0.0, "exact_span": 0.0}
    window = got[: max(len(want), 1) * 3]
    pool, hit = list(window), 0
    for w in want:
        if w in pool:
            pool.remove(w)
            hit += 1
    return {
        "overlap": hit / len(want),
        "exact_prefix": float(normalise(produced).startswith(normalise(gold))),
        "exact_span": float(" ".join(want) in " ".join(got)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rows)]
    man = json.load(open(args.manifest))
    gold = {d["doc"]: d for d in man["documents"]}

    missing = sorted({r["doc"] for r in rows} - set(gold))
    if missing:
        raise SystemExit(
            f"INVALID: {len(missing)} documents in the rows have no gold entry, "
            f"first: {missing[:3]}"
        )

    out_rows = []
    for r in rows:
        g = gold[r["doc"]]
        rec = dict(r)
        if r["kind"] == "retrieval":
            s = score_retrieval(r.get("produced", ""), g["retrieval_gold"])
            rec.update({"gold": g["retrieval_gold"], **s})
            rec["recorded_correct"] = r.get("correct")
            rec["disagrees_with_recorded"] = (
                r.get("correct") is not None
                and float(r["correct"]) != s["exact_any_run"]
            )
            if s["first_run"]:
                rec["edit_distance_to_gold"] = edit_distance(
                    s["first_run"], g["retrieval_gold"].upper()
                )
        elif r["kind"] == "cloze":
            s = score_cloze(r.get("produced", ""), g["cloze_gold"])
            rec.update({"gold": g["cloze_gold"], **s})
            rec["recorded_overlap"] = r.get("overlap")
        out_rows.append(rec)

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # Per-condition comparison of the two retrieval metrics.
    conds = sorted({r["condition"] for r in out_rows if r["kind"] == "retrieval"})
    print(
        "%-22s %7s %7s %7s  %s"
        % ("condition", "subst", "exact", "first", "changed rows")
    )
    summary = {}
    for c in conds:
        sel = [r for r in out_rows if r["kind"] == "retrieval" and r["condition"] == c]
        n = len(sel)
        sub = sum(r["substring"] for r in sel)
        ex = sum(r["exact_any_run"] for r in sel)
        fi = sum(r["exact_first_run"] for r in sel)
        chg = [r["doc"] for r in sel if r["substring"] != r["exact_any_run"]]
        summary[c] = {
            "n": n,
            "substring": sub / n,
            "exact_any_run": ex / n,
            "exact_first_run": fi / n,
            "substring_count": int(sub),
            "exact_count": int(ex),
            "changed_docs": chg,
        }
        print(
            "%-22s %3d/%-3d %3d/%-3d %3d/%-3d  %s"
            % (c, sub, n, ex, n, fi, n, ", ".join(chg) if chg else "-")
        )

    cl = {}
    for c in sorted({r["condition"] for r in out_rows if r["kind"] == "cloze"}):
        sel = [r for r in out_rows if r["kind"] == "cloze" and r["condition"] == c]
        cl[c] = {
            "n": len(sel),
            "overlap": sum(r["overlap"] for r in sel) / len(sel),
            "exact_span": sum(r["exact_span"] for r in sel) / len(sel),
            "matches_recorded_overlap": all(
                r["recorded_overlap"] is None
                or abs(r["overlap"] - r["recorded_overlap"]) < 1e-9
                for r in sel
            ),
        }

    meta = {
        "rows": os.path.abspath(args.rows),
        "manifest": os.path.abspath(args.manifest),
        "n_rows": len(out_rows),
        "retrieval": summary,
        "cloze": cl,
        "rule": (
            "exact_any_run requires the gold code to equal a complete maximal "
            "alphanumeric run in the response; substring is the original "
            "metric and is retained unchanged for comparison"
        ),
    }
    with open(args.out + ".summary.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print()
    print(
        "cloze overlap reproduces the recorded value: "
        + str(all(v["matches_recorded_overlap"] for v in cl.values()))
    )
    print(f"wrote {args.out} and {args.out}.summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
