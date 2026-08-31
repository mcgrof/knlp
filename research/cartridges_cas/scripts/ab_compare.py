#!/usr/bin/env python3
"""Compare two training arms question by question, and say what a null
result rules out.

Two cartridges trained from the same starting point on the same schedule
differ only in how their target rows were written.  Comparing their
curves by eye asks whether the means look different; that is the wrong
question at twenty questions per checkpoint, because the honest answer
is almost always "cannot tell".  Pairing by question removes the
question-difficulty variance, which is most of the variance, and turns
the comparison into one about the difference itself.

The paired quantity is each target question's margin standardised
within its own checkpoint, pooled over target and control questions --
the same standardisation the curve view uses.  Standardising matters:
margins are log odds and move with how extreme a checkpoint's letter
logits are, so a raw margin difference between arms partly measures
which arm ended up more confident, which is not what is being asked.

A null result here is only worth reporting with the interval attached.
"No difference" and "no difference larger than 0.05 would have been
missed" are different claims, and only the second one is evidence, so
the half-width of the 95% interval is reported as the smallest effect
this comparison could have resolved.

Usage: ab_compare.py --a JSON --b JSON [--name-a X] [--name-b Y] [--out MD]
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def zscore(vals):
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    if sd == 0:
        return [0.0 for _ in vals]
    return [(v - m) / sd for v in vals]


def target_z(entry):
    """Per-target-question margin, standardised within the checkpoint."""
    t = [r["margin"] for r in entry["target_rows"]]
    c = [r["margin"] for r in entry["control_rows"]]
    z = zscore(t + c)[: len(t)]
    return {r["qid"]: zi for r, zi in zip(entry["target_rows"], z)}


def target_hits(entry):
    return {r["qid"]: bool(r["hit"]) for r in entry["target_rows"]}


def paired_stats(da):
    """Mean paired difference with a 95% interval, Student t."""
    n = len(da)
    if n < 2:
        return None
    m = statistics.fmean(da)
    sd = statistics.stdev(da)
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    # 1.96 is close enough at n=20 to be honest about, and the interval is
    # being used as a resolution bound rather than a decision rule
    half = 1.96 * se
    return dict(
        n=n,
        mean=m,
        sd=sd,
        se=se,
        lo=m - half,
        hi=m + half,
        half_width=half,
        t=(m / se) if se > 0 else 0.0,
    )


def sign_test(da):
    pos = sum(1 for d in da if d > 0)
    neg = sum(1 for d in da if d < 0)
    return pos, neg, len(da) - pos - neg


def step_of(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ra = json.loads(Path(args.a).read_text())
    rb = json.loads(Path(args.b).read_text())
    shared = sorted(
        set(ra["conditions"]) & set(rb["conditions"]), key=lambda n: step_of(n)
    )
    assert shared, "the two runs share no checkpoint names"

    L = [
        f"# {args.name_a} against {args.name_b}, paired by question",
        "",
        f"Model {ra['model']}, target {ra['patient']}. Each target question's "
        "margin is standardised within its own checkpoint, then paired across "
        "the two arms. A positive difference favours "
        f"**{args.name_a}**. The interval is the range this comparison can "
        "resolve; a mean inside it is a result the experiment could not have "
        "distinguished from no difference.",
        "",
        "| checkpoint | mean diff | 95% interval | resolves | sign (+/-/0) | acc diff |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for name in shared:
        za, zb = target_z(ra["conditions"][name]), target_z(rb["conditions"][name])
        qids = sorted(set(za) & set(zb))
        if not qids:
            continue
        da = [za[q] - zb[q] for q in qids]
        st = paired_stats(da)
        if st is None:
            continue
        ha, hb = target_hits(ra["conditions"][name]), target_hits(
            rb["conditions"][name]
        )
        acc_d = statistics.fmean([float(ha[q]) - float(hb[q]) for q in qids])
        pos, neg, tie = sign_test(da)
        rows.append((name, st, acc_d))
        L.append(
            f"| {name} | {st['mean']:+.3f} | [{st['lo']:+.3f}, {st['hi']:+.3f}] | "
            f"±{st['half_width']:.3f} | {pos}/{neg}/{tie} | {acc_d:+.3f} |"
        )

    trained = [r for r in rows if step_of(r[0]) > 0]
    if trained:
        means = [r[1]["mean"] for r in trained]
        halves = [r[1]["half_width"] for r in trained]
        overall = statistics.fmean(means)
        best = min(halves)
        agree = sum(1 for m in means if m > 0)
        L += [
            "",
            "## Reading",
            "",
            f"- Over the {len(trained)} trained checkpoints the mean paired "
            f"difference is **{overall:+.3f}**, and {agree} of {len(trained)} "
            f"favour {args.name_a}.",
            f"- The tightest checkpoint resolves a difference of **±{best:.3f}**; "
            f"anything smaller than that is below this instrument's resolution "
            f"and must not be reported as a difference.",
            "- A difference that changes sign across the curve is not a "
            "difference, however large any single checkpoint looks.",
            "- Twenty questions on one document and one seed. A null here "
            "bounds the effect on this measurement; it does not bound it "
            "on free generation or on another document.",
        ]

    md = "\n".join(L)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        print(f"\nAB_COMPARE_DONE {args.out}")


if __name__ == "__main__":
    main()
