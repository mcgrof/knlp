#!/usr/bin/env python3
"""Re-score the saved Cartridge rows, and pair the arms the way they must be paired.

A review reproduced the screen's three headline scores from its archived rows
and upheld them, and corrected two readings of those same rows. Both
corrections are arithmetic on evidence that is already on disk, so both are
checked here rather than argued.

The first is a conditioning error. The screen reported accuracy among the
answers each arm managed to produce and called that column the one isolating
correctness from termination. It does not: each arm's denominator is a
different subset, chosen by the model's own decision to stop, and on this
evidence only two of twenty questions resolve in both the cartridge arm and
the bare one. A ratio over subsets selected by the thing being measured
isolates nothing, so the unconditional score is the result and the conditional
one is a description.

The second is a unit error. Seven more correct answers out of twenty is
thirty-five percentage points, not seven, and the screen compared a
seven-point figure against a spread measured elsewhere as though the two were
the same quantity.

What replaces them is a paired count -- which questions each arm gains, loses,
newly resolves and stops resolving -- and an interval from resampling patients
rather than questions, because two questions from one patient are not two
independent observations. That interval is descriptive. It is not a gate, and
it is not a measure of how much a retrained cartridge would move.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from research.cartridges_cas.scripts.score_longhealth import (  # noqa: E402
    UNRESOLVED,
    score,
    summarise,
)

ARMS = ("nocart", "source", "fulldoc")


def load(rows_path):
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[r["question_id"]] = r
    return rows, by_arm


def rescore(rows):
    """Run the archived responses back through the scorer, and require a match.

    The point is not to produce new numbers. It is to establish that the
    scorer in the tree is the one that produced the archived labels, which a
    review could not check because the scorer was not archived beside them.
    """
    out, disagreements = [], []
    for r in rows:
        opts = r.get("options")
        if opts is None:
            # The rows record the gold and the index rather than the option
            # list; rebuild from the manifest when it is supplied.
            continue
        s = score(
            r["response"],
            opts,
            r["gold_index"],
            n_new_tokens=r["generated_tokens"],
            max_new_tokens=512,
        )
        out.append(s)
        if s["correct"] != r["correct"] or s["outcome"] != r["outcome"]:
            disagreements.append(
                {
                    "question_id": r["question_id"],
                    "arm": r["arm"],
                    "archived": {"correct": r["correct"], "outcome": r["outcome"]},
                    "rescored": {"correct": s["correct"], "outcome": s["outcome"]},
                }
            )
    return out, disagreements


def pair(by_arm, a, b):
    """What ``a`` gains and loses against ``b``, question by question.

    A net figure hides a swap. Reporting gains and losses separately is what
    distinguishes an arm that fixed nine and broke one from an arm that fixed
    eight and broke nothing, and those are different claims.
    """
    qids = sorted(set(by_arm[a]) & set(by_arm[b]))
    gains = [q for q in qids if by_arm[a][q]["correct"] > by_arm[b][q]["correct"]]
    losses = [q for q in qids if by_arm[a][q]["correct"] < by_arm[b][q]["correct"]]
    res = lambda r: r["outcome"] not in UNRESOLVED  # noqa: E731
    newly = [q for q in qids if res(by_arm[a][q]) and not res(by_arm[b][q])]
    lost = [q for q in qids if not res(by_arm[a][q]) and res(by_arm[b][q])]
    both = [q for q in qids if res(by_arm[a][q]) and res(by_arm[b][q])]
    return {
        "n_questions": len(qids),
        "gains": len(gains),
        "gain_ids": gains,
        "losses": len(losses),
        "loss_ids": losses,
        "newly_resolved": len(newly),
        "lost_resolution": len(lost),
        "net_resolution": len(newly) - len(lost),
        "resolved_in_both": len(both),
        "resolved_in_both_ids": both,
        "correct_in_both_subset": {
            a: sum(by_arm[a][q]["correct"] for q in both),
            b: sum(by_arm[b][q]["correct"] for q in both),
        },
        "conditional_accuracy_warning": (
            "each arm's resolved set is chosen by its own decision to stop, so "
            "a ratio computed over those sets conditions on the outcome; with "
            f"{len(both)} of {len(qids)} questions resolved in both arms the "
            "conditional comparison rests on almost nothing"
        ),
    }


def cluster_bootstrap(by_arm, a, b, n_boot=10000, seed=0):
    """A difference in accuracy, resampling patients rather than questions.

    Two questions drawn from one patient's record are not two independent
    observations, so an interval that resamples questions is narrower than the
    evidence supports. This resamples the patients. It is reported to say how
    little twenty questions pin down, and it is descriptive: it was computed
    after the outcome, it is not a threshold, and it does not describe how much
    a differently trained cartridge would move.
    """
    import numpy as np

    qids = sorted(set(by_arm[a]) & set(by_arm[b]))
    patients = sorted({by_arm[a][q]["patient"] for q in qids})
    per = {
        p: [
            (by_arm[a][q]["correct"], by_arm[b][q]["correct"])
            for q in qids
            if by_arm[a][q]["patient"] == p
        ]
        for p in patients
    }
    obs = (
        sum(x for p in patients for x, _ in per[p])
        - sum(y for p in patients for _, y in per[p])
    ) / len(qids)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(patients), size=len(patients))
        num = den = 0.0
        for i in pick:
            for x, y in per[patients[i]]:
                num += x - y
                den += 1
        draws.append(num / max(den, 1))
    return {
        "difference": round(float(obs), 4),
        "difference_in_percentage_points": round(float(obs) * 100, 1),
        "ci95": [
            round(float(np.percentile(draws, 2.5)), 4),
            round(float(np.percentile(draws, 97.5)), 4),
        ],
        "clusters": len(patients),
        "questions": len(qids),
        "draws": n_boot,
        "unit_note": (
            "a difference of seven correct answers in twenty questions is "
            "thirty-five percentage points; the screen reported it as seven"
        ),
        "scope": (
            "descriptive, computed after the outcome, resampling patients. Not "
            "an acceptance threshold and not a measure of retraining spread, "
            "which is a different source of uncertainty and cannot be "
            "substituted for this one"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--manifest", default="", help="supplies the option lists")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, by_arm = load(args.rows)
    present = [a for a in ARMS if a in by_arm]
    report = {
        "contract": "cartridge_c1_rescore_v1",
        "rows": os.path.abspath(args.rows),
        "arms": present,
        "questions_per_arm": {a: len(by_arm[a]) for a in present},
    }

    if args.manifest:
        man = json.load(open(args.manifest))
        opts = {
            d["doc"]: [d.get(f"answer_{c}") for c in "abcde"]
            for d in man.get("documents", [])
        }
        for r in rows:
            if r["doc"] in opts and opts[r["doc"]][0] is not None:
                r["options"] = opts[r["doc"]]
        scored, disagree = rescore(rows)
        report["rescore"] = {
            "rows_rescored": len(scored),
            "disagreements": len(disagree),
            "disagreement_detail": disagree[:10],
            "note": (
                "the archived labels and a fresh run of the tree's scorer "
                "over the archived responses must agree; a disagreement "
                "means the scorer beside the result is not the one that "
                "produced it"
            ),
        }
    else:
        report["rescore"] = {
            "skipped": (
                "no manifest supplied, so the option lists the scorer "
                "needs were not available; the archived rows record "
                "the gold and its index but not the five options"
            ),
        }

    report["unconditional"] = {
        a: {
            "correct": sum(r["correct"] for r in by_arm[a].values()),
            "n": len(by_arm[a]),
            "accuracy": round(
                sum(r["correct"] for r in by_arm[a].values()) / len(by_arm[a]), 4
            ),
        }
        for a in present
    }
    report["unconditional_note"] = (
        "the result. Every question is counted, including the ones an arm "
        "failed to answer, because failing to answer is a way of getting it "
        "wrong under a fixed cap"
    )
    report["paired"] = {}
    report["intervals"] = {}
    for a, b in (("source", "nocart"), ("fulldoc", "nocart"), ("source", "fulldoc")):
        if a in by_arm and b in by_arm:
            report["paired"][f"{a}_vs_{b}"] = pair(by_arm, a, b)
            report["intervals"][f"{a}_vs_{b}"] = cluster_bootstrap(by_arm, a, b)

    report["corrections_applied"] = [
        "conditional accuracy demoted to a description: the denominators are "
        "subsets the model selected by stopping, and only the questions "
        "resolved in both arms support any comparison at all",
        "differences are stated in percentage points, not in answers",
        "gains and losses are reported separately, so a net figure cannot hide "
        "a swap",
        "intervals resample patients, not questions",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    print(f"{len(present)} arms, {report['questions_per_arm']}")
    for a in present:
        u = report["unconditional"][a]
        print(f"  {a:9s} {u['correct']:2d}/{u['n']} = {u['accuracy']:.3f}")
    print()
    for k, p in report["paired"].items():
        print(
            f"  {k:20s} gains {p['gains']}, losses {p['losses']}, "
            f"newly resolved {p['newly_resolved']}, lost {p['lost_resolution']}, "
            f"resolved in both {p['resolved_in_both']}"
        )
    print()
    for k, i in report["intervals"].items():
        print(
            f"  {k:20s} {i['difference_in_percentage_points']:+.1f} points, "
            f"95% [{i['ci95'][0]*100:+.1f}, {i['ci95'][1]*100:+.1f}]"
        )
    r = report.get("rescore", {})
    if "disagreements" in r:
        print(
            f"\n  rescore: {r['rows_rescored']} rows, {r['disagreements']} disagreements"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
