# SPDX-License-Identifier: MIT
"""Seed-level statistics and figures from saved subliminal-learning runs.

Everything is read from run receipts (seeds.csv, status.json,
metrics.jsonl); no number is entered by hand. Uncertainty is a paired
bootstrap over independent model keys (seeds), never over test images.

  summary RUN_DIR...            per-run condition means, gains, CIs,
                                assay qualification, six-bar figure
  compare BASELINE CANDIDATE... paired architecture contrasts on the
                                aux_shared transfer gain plus
                                non-inferiority of teacher and ordinary
                                (full-output) distillation accuracy

An incomplete run is reported as incomplete from its metrics log and is
never folded into a comparison.
"""

import argparse
import csv
import json
import os

import numpy as np

CONDITIONS = [
    ("acc_init", "Reference (W0)"),
    ("acc_teacher", "Teacher"),
    ("acc_aux_shared", "Student aux / shared init"),
    ("acc_full_shared", "Student all / shared init"),
    ("acc_aux_other", "Student aux / other init"),
    ("acc_full_other", "Student all / other init"),
]
QUALIFY = {"teacher": 0.95, "full_shared": 0.80, "aux_margin": 0.10}


def load_rows(run_dir):
    with open(os.path.join(run_dir, "seeds.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k.startswith(("acc_", "gain_")):
                r[k] = float(v)
    return rows


def status(run_dir):
    p = os.path.join(run_dir, "status.json")
    return json.load(open(p)) if os.path.exists(p) else {"complete": False}


def recipe(run_dir):
    return json.load(open(os.path.join(run_dir, "recipe.json")))["recipe"]


def bootstrap_ci(values, level=0.95, n_boot=20000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def fmt(x):
    return f"{100 * x:6.2f}"


def summarize(run_dir, plot=True):
    st = status(run_dir)
    rc = recipe(run_dir)
    out = {
        "run": run_dir,
        "arch": rc["arch"],
        "protocol": rc["protocol"],
        "smoke": rc["smoke"],
        "complete": st.get("complete", False),
    }
    if not out["complete"]:
        last = {}
        mpath = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(mpath):
            for line in open(mpath):
                row = json.loads(line)
                last[row["stage"]] = row
        out["incomplete_last_epochs"] = {
            k: {"epoch": v["epoch"], "mean_test_acc": float(np.mean(v["test_acc"]))}
            for k, v in last.items()
        }
        out["stopped_at"] = st.get("stopped_at")
        return out
    rows = load_rows(run_dir)
    out["n_models"] = len(rows)
    out["stage_seconds"] = st.get("stage_seconds")
    out["elapsed_s"] = st.get("elapsed_s")
    for key, _ in CONDITIONS:
        vals = [r[key] for r in rows]
        out[key] = {"mean": float(np.mean(vals)), "ci95": bootstrap_ci(vals)}
    for key in ("gain_aux_shared", "gain_aux_other", "gain_full_shared"):
        vals = [r[key] for r in rows]
        out[key] = {"mean": float(np.mean(vals)), "ci95": bootstrap_ci(vals)}
    margin = [r["gain_aux_shared"] - r["gain_aux_other"] for r in rows]
    out["aux_shared_minus_other_gain"] = {
        "mean": float(np.mean(margin)),
        "ci95": bootstrap_ci(margin),
    }
    checks = {
        "teacher_mean_ge_95": out["acc_teacher"]["mean"] >= QUALIFY["teacher"],
        "full_shared_mean_ge_80": out["acc_full_shared"]["mean"]
        >= QUALIFY["full_shared"],
        "aux_margin_ge_10pp": out["aux_shared_minus_other_gain"]["mean"]
        >= QUALIFY["aux_margin"],
        "aux_margin_ci_positive": out["aux_shared_minus_other_gain"]["ci95"][0] > 0,
        "unused_rows_unchanged": all(
            r[c] == "True"
            for r in rows
            for c in (
                "teacher_aux_rows_unchanged",
                "aux_shared_digit_rows_unchanged",
                "aux_other_digit_rows_unchanged",
            )
        ),
        "alternate_inits_distinct": all(
            r["w0_digest"] != r["w0_other_digest"] for r in rows
        ),
    }
    out["qualification"] = checks
    out["qualified"] = all(checks.values()) and not rc["smoke"]
    if plot:
        six_bar(
            rows,
            os.path.join(run_dir, "six_conditions.png"),
            f"{rc['arch']} / {rc['protocol']} / {len(rows)} models",
        )
    return out


def six_bar(rows, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means, errs, labels = [], [[], []], []
    for key, label in CONDITIONS:
        vals = [r[key] for r in rows]
        m = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        means.append(m)
        errs[0].append(m - lo if lo == lo else 0)
        errs[1].append(hi - m if hi == hi else 0)
        labels.append(label)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#888888", "#3b6ea5", "#c0504d", "#c0504d", "#c0504d", "#c0504d"]
    bars = ax.bar(range(len(means)), means, yerr=errs, capsize=4, color=colors)
    for b in bars[-2:]:
        b.set_alpha(0.45)
    ax.axhline(means[0], ls=":", c="black", lw=1)
    ax.set_xticks(range(len(means)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MNIST test accuracy (outputs 0-9)")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def compare(baseline_dir, candidate_dirs, n_contrasts=None):
    base = {r["model_key"]: r for r in load_rows(baseline_dir)}
    n_contrasts = n_contrasts or len(candidate_dirs)
    level = 1 - 0.05 / n_contrasts
    out = {
        "baseline": baseline_dir,
        "baseline_arch": recipe(baseline_dir)["arch"],
        "ci_level": level,
        "candidates": [],
    }
    for cdir in candidate_dirs:
        if not status(cdir).get("complete") or not status(baseline_dir).get("complete"):
            out["candidates"].append({"run": cdir, "error": "incomplete run"})
            continue
        if recipe(cdir)["protocol"] != recipe(baseline_dir)["protocol"]:
            out["candidates"].append({"run": cdir, "error": "protocol mismatch"})
            continue
        cand = {r["model_key"]: r for r in load_rows(cdir)}
        keys = sorted(set(base) & set(cand))
        missing = sorted(set(base) ^ set(cand))

        def diffs(field):
            return [cand[k][field] - base[k][field] for k in keys]

        res = {
            "run": cdir,
            "arch": recipe(cdir)["arch"],
            "paired_models": len(keys),
            "unpaired_models": missing,
        }
        for field, name in (
            ("gain_aux_shared", "transfer_contrast"),
            ("acc_teacher", "teacher_diff"),
            ("acc_full_shared", "full_distill_diff"),
            ("gain_aux_other", "other_init_gain_diff"),
        ):
            d = diffs(field)
            res[name] = {"mean": float(np.mean(d)), "ci": bootstrap_ci(d, level)}
        tc = res["transfer_contrast"]
        res["screen"] = {
            "reduction_ge_5pp": tc["mean"] <= -0.05,
            "ci_excludes_zero": tc["ci"][1] < 0 or tc["ci"][0] > 0,
            "teacher_noninferior_2pp": res["teacher_diff"]["ci"][0] >= -0.02,
            "full_distill_noninferior_2pp": res["full_distill_diff"]["ci"][0] >= -0.02,
        }
        out["candidates"].append(res)
    return out


def print_summary(s):
    head = f"== {s['arch']} / {s['protocol']}{' (SMOKE)' if s['smoke'] else ''}"
    print(head, s["run"])
    if not s["complete"]:
        print(
            "  INCOMPLETE",
            json.dumps(s.get("stopped_at")),
            json.dumps(s.get("incomplete_last_epochs")),
        )
        return
    for key, label in CONDITIONS:
        v = s[key]
        print(
            f"  {label:28s} {fmt(v['mean'])}%  [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]"
        )
    for key in ("gain_aux_shared", "gain_aux_other", "aux_shared_minus_other_gain"):
        v = s[key]
        print(
            f"  {key:28s} {fmt(v['mean'])}pp [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]"
        )
    print(f"  qualified={s['qualified']} {json.dumps(s['qualification'])}")
    print(f"  models={s['n_models']} elapsed={s['elapsed_s']:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    summary_p = sub.add_parser("summary")
    summary_p.add_argument("runs", nargs="+")
    summary_p.add_argument("--json", help="write the summaries here")
    compare_p = sub.add_parser("compare")
    compare_p.add_argument("baseline")
    compare_p.add_argument("candidates", nargs="+")
    compare_p.add_argument(
        "--contrasts",
        type=int,
        default=None,
        help="multiplicity for the CI level (default: #candidates)",
    )
    compare_p.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "summary":
        out = [summarize(r) for r in a.runs]
        for s in out:
            print_summary(s)
    else:
        out = compare(a.baseline, a.candidates, a.contrasts)
        print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
