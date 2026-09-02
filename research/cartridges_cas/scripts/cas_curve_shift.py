#!/usr/bin/env python3
"""Compare held-out loss curves of cartridge training runs: how many
optimizer steps earlier (or later) an arm reaches the loss levels the
baseline reaches, paired by seed, against the floor that same-seed
replicates set.

Curves come from the ``VAL_CSV,<step>,<loss>`` lines that
``cas_train_isolated.py`` prints when VAL_PARQUET is set (step 0 is the
untrained start; the last line is the final state), or from any text
file with ``<step>,<loss>`` lines. Each curve is named ``arm:seed``.

Targets are loss levels: fractions (default 50, 75 and 90 percent) of
the drop the baseline's seed-mean curve makes from step 0 to the last
step every curve shares. Only targets every baseline seed reaches are
scored. steps_to_target is the first crossing, linearly interpolated
on the evaluation grid, so its resolution is a fraction of the grid
spacing.

For every arm and seed with a baseline run at the same seed the shift
is d = steps_to_target(baseline, seed) - steps_to_target(arm, seed):
positive means the arm got there first. It is reported raw and with
the step-0 offset removed (the arm's curve is shifted vertically so it
starts where the baseline started, which separates a head start that
persists as a constant offset from a genuine acceleration). The sign
is reported explicitly per seed (how many seeds were faster, slower,
or never reached the target), and the magnitude is judged against the
floor by its absolute value, so a consistently slower arm reads as
such rather than as noise.

Extra curves with the same name and seed are replicates; their shifts
against the primary curve set the nondeterminism floor, the largest
absolute replicate shift per target. A replicate may carry a tag
(``base:42#cross=path``) and the floor is then kept per tag, e.g. a
``within`` floor from a same-machine rerun and a ``cross`` floor from
a rerun elsewhere; untagged replicates form the ``replicate`` floor.
``--floor-for arm=tag`` picks the floor an arm is judged against
(default: the largest floor of any tag), and the tag applied is
recorded with every verdict. Floors come from the baseline's
replicates only, unless ``--floor-any-arm``. Vertical differences at
fixed steps are reported as well, because a horizontal shift is
ill-conditioned wherever the curve is flat.

Usage:
    cas_curve_shift.py --baseline base \\
        --curve base:42=logs/train_base_s42.log \\
        --curve base:42#within=logs/train_base_s42_rep.log \\
        --curve base:42#cross=logs/train_base_s42_otherpod.log \\
        --curve meta:42=logs/train_meta_s42.log ... \\
        --floor-for meta=within --out shift.json
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r"(?:^|VAL_CSV,)\s*(\d+|None)\s*,\s*([-+0-9.eE]+)\s*$")


def read_curve(path):
    pts = {}
    last_step = -1
    for line in open(path):
        # trainer logs carry tqdm cursor noise before VAL_CSV
        line = line.rstrip()
        m = LINE.search(
            line[line.find("VAL_CSV") :] if "VAL_CSV" in line else line.strip()
        )
        if not m:
            continue
        step, loss = m.group(1), float(m.group(2))
        if step == "None":
            continue
        step = int(step)
        pts[step] = loss  # a repeated step keeps the last value
        last_step = max(last_step, step)
    assert pts, f"no curve lines in {path}"
    return sorted(pts.items())


def interp(curve, step):
    """Loss at ``step`` by linear interpolation on the grid."""
    xs = [s for s, _ in curve]
    if step <= xs[0]:
        return curve[0][1]
    if step >= xs[-1]:
        return curve[-1][1]
    for (s0, l0), (s1, l1) in zip(curve, curve[1:]):
        if s0 <= step <= s1:
            return l0 + (l1 - l0) * (step - s0) / (s1 - s0)


def steps_to(curve, level):
    """First step at which the curve reaches ``level`` (interpolated), or
    None if it never does."""
    prev = None
    for s, l in curve:
        if l <= level:
            if prev is None:
                return float(s)
            s0, l0 = prev
            if l0 == l:
                return float(s)
            return s0 + (s - s0) * (l0 - level) / (l0 - l)
        prev = (s, l)
    return None


def mean_curve(curves):
    steps = sorted(set.intersection(*[set(s for s, _ in c) for c in curves]))
    return [(s, sum(dict(c)[s] for c in curves) / len(curves)) for s in steps]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--baseline", required=True, help="arm name of the baseline")
    ap.add_argument("--curve", action="append", required=True, help="arm:seed=path")
    ap.add_argument("--targets", default="0.5,0.75,0.9")
    ap.add_argument(
        "--end", type=int, default=0, help="last step scored (default: common last)"
    )
    ap.add_argument("--fixed-steps", default="0,30,60,120,180,240,300,420")
    ap.add_argument(
        "--floor-for",
        action="append",
        default=[],
        help="arm=tag: judge this arm against the floor of that replicate tag",
    )
    ap.add_argument(
        "--floor-any-arm",
        action="store_true",
        help="take floor pairs from every arm's replicates, not only the baseline's",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = defaultdict(list)  # (arm, seed) -> [(tag, curve), ...]
    for spec in a.curve:
        key, path = spec.split("=", 1)
        key, _, tag = key.partition("#")
        arm, seed = key.split(":")
        runs[(arm, int(seed))].append((tag or "replicate", read_curve(path)))
    arms = sorted({k[0] for k in runs})
    assert a.baseline in arms, f"baseline {a.baseline} not among {arms}"
    end = a.end or min(c[-1][0] for cs in runs.values() for _, c in cs)
    # the first curve listed for an arm and seed is its primary; the rest
    # are replicates that only feed the floor
    curves, reps = {}, defaultdict(list)
    for k, cs in runs.items():
        for i, (tag, c) in enumerate(cs):
            c = [(s, l) for s, l in c if s <= end]
            if i == 0:
                curves[k] = [c]
            else:
                reps[k].append((tag, c))
    floor_for = dict(x.split("=", 1) for x in a.floor_for)

    base_seeds = sorted(s for (arm, s) in curves if arm == a.baseline)
    base_mean = mean_curve([curves[(a.baseline, s)][0] for s in base_seeds])
    l0, lend = base_mean[0][1], base_mean[-1][1]
    fracs = [float(x) for x in a.targets.split(",")]
    levels = {f: l0 - f * (l0 - lend) for f in fracs}
    # a target counts only when every baseline seed reaches it
    scored = {
        f: lv
        for f, lv in levels.items()
        if all(steps_to(curves[(a.baseline, s)][0], lv) is not None for s in base_seeds)
    }
    fixed = [int(x) for x in a.fixed_steps.split(",") if int(x) <= end]

    report = {
        "baseline": a.baseline,
        "end_step": end,
        "baseline_seeds": base_seeds,
        "levels": {str(f): lv for f, lv in levels.items()},
        "scored_targets": sorted(scored),
        "baseline_start_loss": l0,
        "baseline_end_loss": lend,
        "floor": {},
        "arms": {},
        "fixed_steps": {},
    }

    # replicate floors: each replicate against its primary, kept per tag
    floors = defaultdict(lambda: defaultdict(float))  # tag -> target -> max
    floor_pairs = defaultdict(int)
    for (arm, seed), rs in reps.items():
        if arm != a.baseline and not a.floor_any_arm:
            continue
        primary = curves[(arm, seed)][0]
        for tag, rc in rs:
            floor_pairs[tag] += 1
            for f, lv in scored.items():
                ti, tj = steps_to(primary, lv), steps_to(rc, lv)
                if ti is None or tj is None:
                    floors[tag][str(f)] = float("inf")
                else:
                    floors[tag][str(f)] = max(floors[tag][str(f)], abs(ti - tj))
    report["floor"] = {
        tag: {"pairs": floor_pairs[tag], "max_abs_shift": dict(floors[tag])}
        for tag in floors
    }

    def floor_of(arm, f):
        """The floor an arm is judged against at target f, and its tag."""
        tag = floor_for.get(arm)
        if tag is not None:
            assert tag in floors, f"no replicate tagged {tag} for --floor-for {arm}"
            return floors[tag].get(str(f)), tag
        cands = [(floors[t].get(str(f)), t) for t in floors if str(f) in floors[t]]
        if not cands:
            return None, None
        return max(cands)

    # per arm, paired by seed against the baseline
    for arm in arms:
        if arm == a.baseline:
            continue
        per_seed = {}
        for (arm_, seed), cs in curves.items():
            if arm_ != arm or (a.baseline, seed) not in curves:
                continue
            base = curves[(a.baseline, seed)][0]
            armc = cs[0]
            off = armc[0][1] - base[0][1]
            armc_o = [(s, l - off) for s, l in armc]
            row = {"step0_offset": off, "targets": {}}
            for f, lv in scored.items():
                tb, ta, tao = (
                    steps_to(base, lv),
                    steps_to(armc, lv),
                    steps_to(armc_o, lv),
                )
                row["targets"][str(f)] = {
                    "base_steps": tb,
                    "arm_steps": ta,
                    "shift": (tb - ta) if (ta is not None) else None,
                    "shift_offset_removed": (tb - tao) if (tao is not None) else None,
                }
            per_seed[str(seed)] = row
        summary = {}
        for f in scored:
            ds = [r["targets"][str(f)]["shift"] for r in per_seed.values()]
            dso = [
                r["targets"][str(f)]["shift_offset_removed"] for r in per_seed.values()
            ]
            fin = [d for d in ds if d is not None]
            fino = [d for d in dso if d is not None]
            n = len(fin)
            mean = sum(fin) / n if n else None
            sd = (
                math.sqrt(sum((d - mean) ** 2 for d in fin) / (n - 1))
                if n > 1
                else None
            )
            n_pos, n_neg = sum(d > 0 for d in fin), sum(d < 0 for d in fin)
            n_unreached = len(ds) - n
            if n_unreached:
                sign = "unreached"
            elif n_pos == n:
                sign = "all_positive"
            elif n_neg == n:
                sign = "all_negative"
            else:
                sign = "mixed"
            fl, fl_tag = floor_of(arm, f)
            summary[str(f)] = {
                "n_seeds": len(ds),
                "n_reached": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_unreached": n_unreached,
                "mean_shift": mean,
                "sd_shift": sd,
                "sign": sign,
                "mean_shift_offset_removed": (sum(fino) / len(fino)) if fino else None,
                "floor": fl,
                "floor_tag": fl_tag,
                # magnitude against the floor; the sign is reported apart
                "beats_floor": mean is not None
                and fl is not None
                and abs(mean) > 2 * fl,
            }
        report["arms"][arm] = {"per_seed": per_seed, "summary": summary}

    # vertical differences at fixed steps, seed-mean per arm
    for arm in arms:
        seeds = sorted(s for (arm_, s) in curves if arm_ == arm)
        mc = mean_curve([curves[(arm, s)][0] for s in seeds])
        report["fixed_steps"][arm] = {
            str(st): interp(mc, st) for st in fixed if st <= mc[-1][0]
        }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)

    print(
        f"baseline {a.baseline}: seeds {base_seeds}, start {l0:.5f} end({end}) {lend:.5f}; "
        f"targets " + ", ".join(f"{f}: {lv:.5f}" for f, lv in scored.items())
    )
    for tag in floors:
        print(
            f"replicate floor '{tag}' ({floor_pairs[tag]} pairs): "
            + ", ".join(f"{f}: {v:.1f} steps" for f, v in floors[tag].items())
        )
    print(
        "SHIFT,arm,target,reached/seeds,pos/neg/unreached,mean,sd,"
        "mean_offset_removed,sign,verdict,floor_tag"
    )
    for arm, r in report["arms"].items():
        for f, s in r["summary"].items():
            m = s["mean_shift"]
            print(
                f"SHIFT,{arm},{f},{s['n_reached']}/{s['n_seeds']},"
                f"{s['n_pos']}/{s['n_neg']}/{s['n_unreached']},"
                f"{'' if m is None else f'{m:.1f}'},{'' if s['sd_shift'] is None else f'{s['sd_shift']:.1f}'},"
                f"{'' if s['mean_shift_offset_removed'] is None else f'{s['mean_shift_offset_removed']:.1f}'},"
                f"{s['sign']},"
                f"{'beats_floor' if s['beats_floor'] else 'within_floor' if s['floor'] is not None else 'no_floor'},"
                f"{s['floor_tag'] or ''}"
            )
    hdr = "step," + ",".join(arms)
    print(hdr)
    for st in fixed:
        vals = [report["fixed_steps"][arm].get(str(st)) for arm in arms]
        print(f"{st}," + ",".join("" if v is None else f"{v:.5f}" for v in vals))
    print(f"SHIFT_DONE {a.out}")


if __name__ == "__main__":
    main()
