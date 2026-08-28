#!/usr/bin/env python3
"""Paired analysis of per-question forced-choice margins.

Two comparisons the accuracy-only view could not make.

The first is document specificity: the gap between a cartridge's margin
on its own patient's questions and its margin on other patients'
questions, which its record cannot answer. That gap is the cartridge's
document advantage. It has real dynamic range, unlike accuracy on
twenty items, and it distinguishes encoding a document from getting
better at multiple-choice questions in general.

The second is arm against arm on the same questions. Paired
differences over shared items remove the question-difficulty variance
that makes unpaired accuracy comparisons hopeless at this sample size,
so a difference that accuracy cannot resolve may still be visible
here -- or provably absent, which is equally useful.

Reports a paired t statistic and a sign test. With twenty questions
neither is proof; they are stated to keep the size of a claim
proportional to the evidence behind it.

Usage: control_aware_choice_stats.py --choice JSON [--out MD]
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def paired(a_rows, b_rows, key="margin"):
    """Pair by question id and return per-question differences."""
    a = {r["qid"]: r[key] for r in a_rows}
    b = {r["qid"]: r[key] for r in b_rows}
    shared = sorted(set(a) & set(b))
    return [a[q] - b[q] for q in shared]


def t_stat(d):
    n = len(d)
    if n < 2:
        return None, None
    m = statistics.fmean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return math.inf if m else 0.0, n - 1
    return m / (sd / math.sqrt(n)), n - 1


def sign_test(d):
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    return pos, neg, len(d) - pos - neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = json.loads(Path(args.choice).read_text())
    cond = rep["conditions"]
    names = list(cond)

    L = [
        "# Forced-choice margins, per question",
        "",
        f"Model {rep['model']}, target patient {rep['patient']}, control "
        f"patients {', '.join(rep['control_patients'])}. The control "
        "questions belong to other patients and cannot be answered from "
        "the target record, so the gap between the two columns is the "
        "cartridge's document advantage.",
        "",
        "## Document advantage",
        "",
        "| condition | target acc | target margin | control margin | advantage |",
        "|---|---|---|---|---|",
    ]
    adv = {}
    for n in names:
        t, c = cond[n]["target"], cond[n]["control"]
        adv[n] = t["margin_mean"] - c["margin_mean"]
        L.append(
            f"| {n} | {t['acc']:.3f} | {t['margin_mean']:+.3f} | "
            f"{c['margin_mean']:+.3f} | **{adv[n]:+.3f}** |"
        )

    base = "start" if "start" in cond else names[0]
    L += [
        "",
        f"## Each arm against the shared starting cartridge (`{base}`)",
        "",
        "Paired over the target patient's questions.",
        "",
        "| arm | mean margin change | t | better / worse / tied |",
        "|---|---|---|---|",
    ]
    for n in names:
        if n == base:
            continue
        d = paired(cond[n]["target_rows"], cond[base]["target_rows"])
        t, _ = t_stat(d)
        p, q, z = sign_test(d)
        L.append(
            f"| {n} | {statistics.fmean(d):+.3f} | "
            f"{'inf' if t == math.inf else f'{t:+.2f}'} | {p} / {q} / {z} |"
        )

    pairs = [
        (a, b)
        for a in names
        for b in names
        if a < b
        and a.rsplit("_s", 1)[-1] == b.rsplit("_s", 1)[-1]
        and a != base
        and b != base
    ]
    if pairs:
        L += [
            "",
            "## Arm against arm, same seed, same questions",
            "",
            "| pair | mean margin difference | t | first better / worse / tied |",
            "|---|---|---|---|",
        ]
        for a, b in pairs:
            d = paired(cond[a]["target_rows"], cond[b]["target_rows"])
            t, _ = t_stat(d)
            p, q, z = sign_test(d)
            L.append(
                f"| {a} vs {b} | {statistics.fmean(d):+.3f} | "
                f"{'inf' if t == math.inf else f'{t:+.2f}'} | {p} / {q} / {z} |"
            )

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"CHOICE_STATS {args.out}")
    print(text)


if __name__ == "__main__":
    main()
