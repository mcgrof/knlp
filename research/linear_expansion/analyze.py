"""Screen decision from tuning records, and paired confirmation tables.

The screen rule is fixed before any confirmation run: F2 is promising when
its tuned curve CE at update 512 beats tuned D by at least 0.02 nat/token,
or when it reaches a common curve-loss threshold with at least 10% less
training-work time. The threshold is tuned D's curve CE at update 384,
rounded up to the next 0.01, and a crossing needs two consecutive
scheduled evaluations at or below it.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def target_from(ce384):
    return math.ceil(ce384 * 100) / 100


def crossing(curve, times, target):
    """Training seconds at the second of two consecutive evals <= target."""
    updates = sorted(curve)
    for a, b in zip(updates, updates[1:]):
        if curve[a] <= target and curve[b] <= target:
            return times[b]
    return None


def tuned_record(tuning_dir, trial):
    # Continuations to 512 share the directory of their 128-update trial.
    trial = trial.removesuffix("-to512")
    record = json.loads((tuning_dir / trial / "RECORD.json").read_text())
    curve = {c["update"]: c["ce"] for c in record["curve"]}
    times = {c["update"]: c["training_seconds"] for c in record["curve"]}
    return curve, times


def screen(out):
    tuning = json.loads((out / "TUNING.json").read_text())
    sel = tuning["selected"]
    rows = {}
    for arm, s in sel.items():
        curve, times = tuned_record(out / "tuning", s["trial"])
        rows[arm] = {"setting": s, "curve": curve, "times": times}
    target = target_from(rows["D"]["curve"][384])
    result = {"target_ce": target, "arms": {}}
    for arm, r in rows.items():
        result["arms"][arm] = {
            "lr": r["setting"]["lr"],
            "multiplier": r["setting"]["multiplier"],
            "boundary_limited": r["setting"]["boundary_limited"],
            "ce512": r["curve"][512],
            "score512": r["setting"]["score512"],
            "train_seconds_512": r["times"][512],
            "seconds_to_target": crossing(r["curve"], r["times"], target),
        }
    d, f2 = result["arms"]["D"], result["arms"]["F2"]
    gain = d["ce512"] - f2["ce512"]
    faster = None
    if d["seconds_to_target"] and f2["seconds_to_target"]:
        faster = 1 - f2["seconds_to_target"] / d["seconds_to_target"]
    result["F2_vs_D_ce512_gain"] = gain
    result["F2_vs_D_time_saving"] = faster
    result["promising"] = gain >= 0.02 or (faster is not None and faster >= 0.10)
    later = [u for u in rows["D"]["curve"] if u >= 384]
    result["F2_worse_at_every_late_checkpoint"] = all(
        rows["F2"]["curve"][u] > rows["D"]["curve"][u] for u in later
    )
    return result


def paired(out):
    conf = json.loads((out / "CONFIRMATION.json").read_text())
    runs = conf["runs"]
    seeds = sorted({r["seed"] for r in runs.values()})
    table = {}
    for endpoint in ("equal_tokens", "equal_time"):
        for metric in ("final_ce_exported", "curve_ce"):
            for control in ("D", "F1", "A4"):
                diffs = []
                for seed in seeds:
                    c = runs.get(f"confirm-{control}-{seed}")
                    t = runs.get(f"confirm-F2-{seed}")
                    if not c or not t:
                        continue
                    ce, te = c["endpoints"].get(endpoint), t["endpoints"].get(endpoint)
                    if ce and te:
                        diffs.append(ce[metric] - te[metric])
                if diffs:
                    table[f"{endpoint}:{metric}:{control}-F2"] = {
                        "per_seed": diffs,
                        "mean": statistics.mean(diffs),
                        "median": statistics.median(diffs),
                        "stdev": statistics.stdev(diffs) if len(diffs) > 1 else None,
                        "all_positive": all(x > 0 for x in diffs),
                    }
    return table


def freeze(out, source, data_sha):
    """Write FROZEN.json before any confirmation run is opened."""
    import random

    tuning = json.loads((out / "TUNING.json").read_text())
    profile = json.loads((out / "PROFILE.json").read_text())
    s = screen(out)
    order = {}
    for seed in (75211, 75212, 75213):
        arms = ["D", "F1", "F2", "A4"]
        random.Random(f"ab-expansion-order:{seed}").shuffle(arms)
        order[str(seed)] = arms
    frozen = {
        "source_commit": source,
        "data_manifest_sha256": data_sha,
        "selected": tuning["selected"],
        "targets": "q_proj, k_proj, v_proj, o_proj in all 4 attention blocks",
        "precision": "FP32 master/state/CE, BF16 autocast matrix ops, TF32 off",
        "optimizer": "AdamW betas (0.9, 0.95) eps 1e-8 wd 0, foreach, unfused",
        "warmup_updates": 32,
        "seeds": [75211, 75212, 75213],
        "order": order,
        "token_endpoint": 2048,
        "time_endpoint_seconds": profile["time_endpoint_seconds"],
        "time_overshoot_limit_fraction": 0.01,
        "max_updates": 4096,
        "time_to_target_ce": s["target_ce"],
        "export_gates": {
            "fp32_logits_rel_l2": 1e-5,
            "fp32_ce_abs": 1e-5,
            "bf16_logits_rel_l2": 5e-3,
            "bf16_ce_abs": 5e-3,
        },
        "decision_rules": "PLAN.md section 10, unchanged",
        "screen": s,
    }
    (out / "FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["screen", "paired", "freeze"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--source")
    p.add_argument("--data-sha")
    a = p.parse_args()
    if a.what == "freeze":
        result = freeze(a.out, a.source, a.data_sha)
    else:
        result = screen(a.out) if a.what == "screen" else paired(a.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
