# SPDX-License-Identifier: MIT
"""Paired analysis of intervention cells with predeclared screens.

Each run directory must be complete, record a clean workload and cover
exactly the seeds its recipe names, once each; every per-model quantity is
keyed by model identity across all packs, never by position. For a
candidate and its parent the report gives, at each student milestone:

  transfer    aux-only gain difference (pp), paired bootstrap interval
  utility     teacher and all-output (shared-init) accuracy differences
  speed       per-seed epochs and seconds for the all-output student and
              the teacher to reach a fixed accuracy; the geometric-mean
              time ratio with an upper bootstrap bound; censored pairs
              follow the rules below
  stability   largest mHC row/column error over every forward

Screens (frozen before the runs they judge):
  transfer:  mean <= -5 pp and interval upper bound < 0 at the last epoch
  utility:   interval lower bound >= -2 pp for teacher and all-output
  speed:     upper bound of mean log(candidate/parent time) <= log(1.10)
             for the all-output student and the teacher; a pair where the
             candidate never reaches the threshold but the parent does
             fails; a pair where the parent never reaches it is unresolved
  stability: every tracked mHC error <= 1e-4

Usage: python3 -m research.subliminal_learning.analyze PLAN.json OUT.json
PLAN.json: {"root": dir, "pairs": [[parent, candidate], ...],
            "level": 0.95, "threshold": 95.0, "resamples": 100000}
"""

import csv
import json
import math
import os
import sys

import numpy as np

STAGES = ("aux_shared", "full_shared", "aux_other", "full_other")


class RunError(Exception):
    pass


def load(run_dir):
    """Per-model tables of one complete run, validated."""
    st = json.load(open(os.path.join(run_dir, "status.json")))
    if not st.get("complete"):
        raise RunError(f"{run_dir}: incomplete")
    rec = json.load(open(os.path.join(run_dir, "recipe.json")))
    git = rec.get("git", {})
    if git.get("workload_dirty") is not False:
        raise RunError(f"{run_dir}: workload not recorded clean ({git})")
    recipe = rec["recipe"]
    expected = list(recipe["model_keys"])
    if len(set(expected)) != len(expected):
        raise RunError(f"{run_dir}: duplicate model keys in recipe")
    results = json.load(open(os.path.join(run_dir, "results.json")))
    key_of = {}  # (pack tag, index) -> model key
    per_model = {}
    for i, pack in sorted(results.items(), key=lambda kv: int(kv[0])):
        tag = f"pack{int(i):02d}"
        for j, k in enumerate(pack["keys"]):
            if k in per_model:
                raise RunError(f"{run_dir}: model {k} in two packs")
            key_of[(tag, j)] = k
            per_model[k] = {"pack": pack["results"], "index": j}
    if sorted(per_model) != sorted(expected):
        raise RunError(f"{run_dir}: packs cover {sorted(per_model)} not {expected}")

    def table(name):
        rows = list(csv.DictReader(open(os.path.join(run_dir, name))))
        keys = [r["model_key"] for r in rows]
        if sorted(keys) != sorted(expected) or len(set(keys)) != len(keys):
            raise RunError(f"{run_dir}/{name}: seeds {keys} != {expected}")
        return {r["model_key"]: r for r in rows}

    milestones = {}
    for e in recipe.get("student_eval_epochs") or []:
        milestones[int(e)] = table(f"seeds_e{e}.csv")
    final = table("seeds.csv")
    last = max(milestones) if milestones else recipe["student_epochs"]
    milestones.setdefault(last, final)
    curves = {k: {s: {} for s in ("teacher",) + STAGES} for k in expected}
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        r = json.loads(line)
        if "test_acc" not in r or r.get("epoch", 0) < 1:
            continue
        for j, acc in enumerate(r["test_acc"]):
            k = key_of[(r["pack"], j)]
            curves[k][r["stage"]][r["epoch"]] = (
                100 * acc,
                r.get("epoch_seconds"),
            )
    stability = {}
    for k, pm in per_model.items():
        vals = [
            v
            for kk, v in pm["pack"].items()
            if kk.startswith("sinkhorn_track_") and v is not None
        ]
        if vals:
            stability[k] = max(max(v) for v in vals)
    return {
        "dir": run_dir,
        "recipe": recipe,
        "git": git,
        "keys": expected,
        "milestones": milestones,
        "curves": curves,
        "stability": stability,
    }


def time_to(curve, threshold):
    """(epochs, seconds) to first reach threshold, or None if censored."""
    elapsed = 0.0
    for e in sorted(curve):
        acc, sec = curve[e]
        elapsed += sec or 0.0
        if acc >= threshold:
            return e, (elapsed if sec is not None else None)
    return None


def boot(values, level, resamples, seed=20260928):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (resamples, len(v)))].mean(1)
    lo, hi = np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(v.mean()), float(lo), float(hi)]


def speed(parent, cand, stage, threshold, level, resamples):
    rows, logs, status = [], [], "ok"
    for k in parent["keys"]:
        tp = time_to(parent["curves"][k][stage], threshold)
        tc = time_to(cand["curves"][k][stage], threshold)
        rows.append({"model": k, "parent": tp, "candidate": tc})
        if tp is None:
            status = "unresolved"
        elif tc is None:
            status = "fail" if status != "unresolved" else status
        else:
            logs.append(math.log(tc[0] / tp[0]))
    out = {"per_seed": rows, "status": status}
    if status == "ok" and logs:
        m, lo, hi = boot(logs, level, resamples)
        out["mean_log_ratio"] = [m, lo, hi]
        out["geometric_ratio"] = math.exp(m)
        out["pass"] = hi <= math.log(1.10)
    else:
        out["pass"] = False
    return out


def compare(parent, cand, level, threshold, resamples):
    if parent["keys"] != cand["keys"]:
        raise RunError(f"seed sets differ: {parent['dir']} vs {cand['dir']}")
    out = {"parent": parent["dir"], "candidate": cand["dir"], "milestones": {}}
    for e in sorted(set(parent["milestones"]) & set(cand["milestones"])):
        b, c = parent["milestones"][e], cand["milestones"][e]
        row = {}
        for key in (
            "gain_aux_shared",
            "acc_teacher",
            "acc_full_shared",
            "acc_full_other",
            "gain_aux_other",
        ):
            d = [100 * (float(c[k][key]) - float(b[k][key])) for k in parent["keys"]]
            row[key] = boot(d, level, resamples)
        out["milestones"][str(e)] = row
    last = str(max(int(e) for e in out["milestones"]))
    fin = out["milestones"][last]
    out["speed"] = {
        "full_shared": speed(parent, cand, "full_shared", threshold, level, resamples),
        "teacher": speed(parent, cand, "teacher", threshold, level, resamples),
    }
    worst = max(
        [*parent["stability"].values(), *cand["stability"].values()], default=None
    )
    out["screens"] = {
        "transfer": fin["gain_aux_shared"][0] <= -5 and fin["gain_aux_shared"][2] < 0,
        "utility_teacher": fin["acc_teacher"][1] >= -2,
        "utility_all_output": fin["acc_full_shared"][1] >= -2,
        "speed_all_output": out["speed"]["full_shared"]["pass"],
        "speed_teacher": out["speed"]["teacher"]["pass"],
        "stability": worst is None or worst <= 1e-4,
    }
    out["max_sinkhorn_error"] = worst
    out["promote"] = all(out["screens"].values())
    return out


def main():
    plan = json.load(open(sys.argv[1]))
    root = plan["root"]
    level = plan.get("level", 0.95)
    threshold = plan.get("threshold", 95.0)
    resamples = plan.get("resamples", 100000)
    runs, report = {}, {"plan": plan, "comparisons": [], "errors": []}
    for parent, cand in plan["pairs"]:
        try:
            for name in (parent, cand):
                if name not in runs:
                    runs[name] = load(os.path.join(root, name))
            report["comparisons"].append(
                compare(runs[parent], runs[cand], level, threshold, resamples)
            )
        except (RunError, FileNotFoundError, KeyError) as exc:
            report["errors"].append(f"{parent} vs {cand}: {exc}")
    json.dump(report, open(sys.argv[2], "w"), indent=1, default=str)
    for c in report["comparisons"]:
        fin = c["milestones"][max(c["milestones"], key=int)]
        print(
            f"{os.path.basename(c['candidate'])} vs {os.path.basename(c['parent'])}: "
            f"transfer {fin['gain_aux_shared']}, teacher {fin['acc_teacher']}, "
            f"all-output {fin['acc_full_shared']}, speed "
            f"{c['speed']['full_shared'].get('geometric_ratio')} "
            f"({c['speed']['full_shared']['status']}), promote={c['promote']} {c['screens']}"
        )
    for e in report["errors"]:
        print("ERROR", e)


if __name__ == "__main__":
    main()
