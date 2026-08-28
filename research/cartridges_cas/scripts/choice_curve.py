#!/usr/bin/env python3
"""Read a training curve without letting confidence scale fake a trend.

Document advantage -- own-patient margin minus other patients' margins
-- is measured in log odds, so it grows when a model's letter logits
are extreme and shrinks when they are compressed, independently of how
much the document actually helps.  That is harmless when comparing
checkpoints at similar confidence, which is what the arm comparison
did, and misleading across a training curve, where an untrained
cartridge produces margins spread over ten log units and a trained one
produces margins spread over two.  Taken raw, the curve would show a
collapse that is mostly the model becoming calibrated.

So report three views and let them disagree in public:

  raw advantage        target margin minus control margin, log odds
  standardised         the same difference after z-scoring every
                       question's margin within the checkpoint, which
                       removes the global scale and asks only whether
                       the target questions sit further up this
                       checkpoint's own distribution than the controls
  accuracy gap         target accuracy minus control accuracy, immune
                       to scale and blunt for it

A real change in document knowledge should move the standardised
measure and the accuracy gap together.  A change only in the raw
number is a confidence effect wearing a knowledge costume.

Usage: choice_curve.py --choice JSON [--reference JSON] [--out MD]
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def zscore(vals):
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    if sd == 0:
        return [0.0 for _ in vals]
    return [(v - m) / sd for v in vals]


def views(entry):
    t = [r["margin"] for r in entry["target_rows"]]
    c = [r["margin"] for r in entry["control_rows"]]
    z = zscore(t + c)
    zt, zc = z[: len(t)], z[len(t) :]
    return dict(
        raw=statistics.fmean(t) - statistics.fmean(c),
        standardised=statistics.fmean(zt) - statistics.fmean(zc),
        acc_gap=entry["target"]["acc"] - entry["control"]["acc"],
        target_acc=entry["target"]["acc"],
        margin_spread=statistics.pstdev(t + c),
    )


def step_of(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice", required=True)
    ap.add_argument("--reference", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rep = json.loads(Path(args.choice).read_text())
    conds = rep["conditions"]
    names = sorted(conds, key=lambda n: step_of(n))

    L = [
        "# Cartridge training curve, three views",
        "",
        f"Model {rep['model']}, target {rep['patient']}, controls "
        f"{', '.join(rep['control_patients'])}. The raw column is in log "
        "odds and moves with the model's overall confidence; the "
        "standardised column removes that scale; the accuracy gap is "
        "immune to it and blunt. Trust a trend the last two share.",
        "",
        "| checkpoint | target acc | raw advantage | standardised | accuracy gap | margin spread |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        v = views(conds[n])
        L.append(
            f"| {n} | {v['target_acc']:.3f} | {v['raw']:+.3f} | "
            f"{v['standardised']:+.3f} | {v['acc_gap']:+.3f} | "
            f"{v['margin_spread']:.2f} |"
        )

    if args.reference and Path(args.reference).is_file():
        ref = json.loads(Path(args.reference).read_text())
        L += [
            "",
            "## References, measured the same way",
            "",
            "| condition | target acc | raw advantage | standardised | accuracy gap |",
            "|---|---|---|---|---|",
        ]
        for n, e in ref["conditions"].items():
            v = views(e)
            L.append(
                f"| {n} | {v['target_acc']:.3f} | {v['raw']:+.3f} | "
                f"{v['standardised']:+.3f} | {v['acc_gap']:+.3f} |"
            )

    best_std = max(names, key=lambda n: views(conds[n])["standardised"])
    best_acc = max(names, key=lambda n: views(conds[n])["acc_gap"])
    L += [
        "",
        "## Reading",
        "",
        f"- Standardised advantage peaks at `{best_std}`; the accuracy "
        f"gap peaks at `{best_acc}`.",
        "- A peak that both agree on, followed by decline, means the "
        "objective keeps improving after the cartridge has stopped "
        "gaining document knowledge, and training past that point is "
        "spending compute to make the artifact worse.",
        "- Twenty target questions is a blunt instrument for the "
        "accuracy gap; treat a one-checkpoint difference as noise and "
        "look for a sustained direction.",
    ]
    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"CURVE_REPORT {args.out}")
    print(text)


if __name__ == "__main__":
    main()
