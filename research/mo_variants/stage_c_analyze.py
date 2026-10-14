"""Paired eight-seed confirmation decisions without outcome selection."""

import argparse
import csv
import math
import statistics
from pathlib import Path

from .stage_c_common import SEEDS, read, sha, write


def summary(values):
    values = list(values)
    mean = statistics.mean(values)
    radius = (
        2.364624251 * statistics.stdev(values) / math.sqrt(8)
        if len(values) == 8
        else None
    )
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "range": [min(values), max(values)],
        "paired_seed_95_interval_descriptive": (
            [mean - radius, mean + radius] if radius is not None else None
        ),
    }


def sign_test(differences):
    nonzero = [x for x in differences if x != 0]
    n = len(nonzero)
    wins = sum(x < 0 for x in nonzero)
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def holm(claims):
    prior = 0
    for rank, claim in enumerate(sorted(claims, key=lambda c: c["one_sided_sign_p"])):
        prior = max(prior, min(1, (len(claims) - rank) * claim["one_sided_sign_p"]))
        claim["holm_adjusted_p"] = prior


def decisions(candidate, baseline, kind, baseline_name):
    differences = {
        name: [candidate[s][name] - baseline[s][name] for s in SEEDS]
        for name in (
            "primary_ce",
            "gpu_final_ce",
            "cpu_final_ce",
            "primary_int4_ce",
            "int4_damage",
        )
    }
    if kind == "training":
        d = differences["primary_ce"]
        checks = {
            "eight_primary_wins": all(x < 0 for x in d),
            "mean_gain_at_least_point005": -statistics.mean(d) >= 0.005,
            "lower_final_mean": statistics.mean(differences["gpu_final_ce"]) < 0,
            "no_final_regression_over_point01": max(differences["gpu_final_ce"])
            <= 0.01,
        }
    elif kind == "quantization":
        d = differences["primary_int4_ce"]
        damage = differences["int4_damage"]
        checks = {
            "eight_absolute_int4_wins": all(x < 0 for x in d),
            "damage_margin_met": -statistics.mean(damage)
            >= ({"V00": 0.01, "V01": 0.005}[baseline_name]),
            "seven_damage_wins": sum(x < 0 for x in damage) >= 7,
            "no_clean_regression_over_point01": max(differences["cpu_final_ce"])
            <= 0.01,
        }
    else:
        d = differences["primary_ce"]
        checks = {
            "eight_primary_wins": all(x < 0 for x in d),
            "mean_gain_at_least_point005": -statistics.mean(d) >= 0.005,
        }
    return {
        "checks": checks,
        "meets_performance_rule": all(checks.values()),
        "paired_differences": differences,
        "paired_summaries": {k: summary(v) for k, v in differences.items()},
        "one_sided_sign_p": sign_test(d),
    }


def analyze(root):
    root = Path(root)
    r = root / "records"
    plan = read(r / "STAGE-C-PLAN.json")
    data = {}
    endpoints = []
    curves = []
    checkpoints = []
    scope = {x["variant"]: x for x in read(r / "STAGE-C-NUMERICAL-SCOPE.json")["rows"]}
    for job in plan["jobs"]:
        jid = job["id"]
        result = read(r / "runs" / jid / "RESULT.json")
        q = read(r / "quantization" / (jid + ".json"))
        n = read(r / "numerics" / (jid + ".json"))
        assert result["status"] == q["status"] == n["status"] == "complete"
        assert (
            result["completed_step"] == 4000
            and q["device"] == "cpu"
            and q["gpu_device_handles"] == []
        )
        assert result["metadata"]["job"] == job
        assert q["array_sha256"] == plan["data"]["confirmation_files"]["final.npy"]
        bystep = {x["step"]: x for x in result["validation"]}
        assert len(bystep) == 40
        cases = {x["name"]: x for x in q["cases"]}
        assert len(cases) == 10
        values = {
            "primary_ce": statistics.mean(
                bystep[s]["ce"] for s in plan["primary_steps"]
            ),
            "early_window_ce": statistics.mean(
                bystep[s]["ce"] for s in plan["early_window_steps"]
            ),
            "gpu_final_ce": result["final_gpu_clean_ce"],
            "cpu_final_ce": q["clean_ce"],
            "primary_int4_ce": cases["rowwise4"]["ce"],
            "int4_damage": cases["rowwise4"]["delta_ce"],
            "numerical_samples_pass": n["finite_reference_pass"],
            "historical_samples_pass": scope[job["variant"]]["historical_and_B_pass"],
            "training_seconds": result["training_seconds"],
            "elapsed_seconds": result["end_to_end_seconds"],
            "cases": cases,
            "validation": result["validation"],
            "finite_misses": n["finite_misses"],
        }
        data.setdefault(job["row"], {})[job["seed"]] = values
        endpoints.append(
            {
                "row": job["row"],
                "seed": job["seed"],
                **{k: v for k, v in values.items() if k not in ("cases", "validation")},
                "quantization_cases": q["cases"],
                "cpu_minus_gpu_clean_ce": q["cpu_minus_gpu_clean_ce"],
                "phase_and_perturbation_errors": {
                    x["name"]: x["phase_and_perturbation_errors"]
                    for x in q["cases"]
                    if x["phase_and_perturbation_errors"]
                },
            }
        )
        for x in result["validation"]:
            curves.append({"row": job["row"], "seed": job["seed"], **x})
        for cp in result["checkpoints"]:
            p = root / "checkpoints" / jid / Path(cp["path"]).name
            assert sha(p) == cp["sha256"]
            checkpoints.append({"job_id": jid, "retained_local_path": str(p), **cp})
    assert all(sorted(d) == list(SEEDS) for d in data.values())
    # Every named shared weight and original routing initialization is paired.
    for seed in SEEDS:
        identities = [
            read(r / "runs" / f"C-{row}-{seed}" / "RESULT.json")["identities"]
            for row in data
        ]
        assert all(
            x["common_weights"] == identities[0]["common_weights"]
            and x["routing_initialization"] == identities[0]["routing_initialization"]
            for x in identities
        )
    rankings = []
    for row, values in data.items():
        low = next(
            (
                b
                for b in (2, 3, 4, 8)
                if all(
                    values[s]["cases"]["rowwise" + str(b)]["within_point1_nats"]
                    for s in SEEDS
                )
            ),
            None,
        )
        rankings.append(
            {
                "row": row,
                "means": {
                    metric: summary(values[s][metric] for s in SEEDS)
                    for metric in (
                        "primary_ce",
                        "early_window_ce",
                        "gpu_final_ce",
                        "cpu_final_ce",
                        "primary_int4_ce",
                        "int4_damage",
                    )
                },
                "lowest_passing_tested_precision": "INT" + str(low) if low else "none",
                "confirmation_numerical_samples_pass": all(
                    values[s]["numerical_samples_pass"] for s in SEEDS
                ),
                "historical_and_confirmation_samples_pass": all(
                    values[s]["numerical_samples_pass"]
                    and values[s]["historical_samples_pass"]
                    for s in SEEDS
                ),
            }
        )
    verdicts = []
    for claim in plan["claim_family"]:
        kind = (
            "quantization"
            if "-quantization-" in claim["name"]
            else "mechanism" if "-ordered-carry-" in claim["name"] else "training"
        )
        d = decisions(
            data[claim["candidate"]], data[claim["baseline"]], kind, claim["baseline"]
        )
        numerical = all(
            x["numerical_samples_pass"] and x["historical_samples_pass"]
            for arm in (claim["candidate"], claim["baseline"])
            for x in data[arm].values()
        )
        verdicts.append(
            {
                **claim,
                **d,
                "kind": kind,
                "numerical_qualification_pass": numerical,
                "qualified_confirmation": d["meets_performance_rule"] and numerical,
            }
        )
    holm(verdicts)
    paired = []
    for row in data:
        for baseline in ("V00", "V01", "V30-S", "V30-C"):
            for seed in SEEDS:
                paired.append(
                    {
                        "row": row,
                        "baseline": baseline,
                        "seed": seed,
                        **{
                            metric: data[row][seed][metric]
                            - data[baseline][seed][metric]
                            for metric in (
                                "primary_ce",
                                "gpu_final_ce",
                                "cpu_final_ce",
                                "primary_int4_ce",
                                "int4_damage",
                            )
                        },
                    }
                )
    targets = []
    matched = []
    for row, values in data.items():
        for seed, value in values.items():
            observations = value["validation"]
            for threshold in (3.0, 4.0):
                hits = [
                    b
                    for a, b in zip(observations, observations[1:])
                    if a["ce"] <= threshold
                    and b["ce"] <= threshold
                    and b["step"] == a["step"] + 100
                ]
                first = hits[0] if hits else None
                targets.append(
                    {
                        "row": row,
                        "seed": seed,
                        "threshold": threshold,
                        "censored": first is None,
                        "confirmation_step": first["step"] if first else None,
                        "tokens": first["tokens"] if first else None,
                        "elapsed_seconds": (
                            first["end_to_end_seconds"] if first else None
                        ),
                    }
                )
            for budget in plan["matched_time_seconds"]:
                eligible = [
                    x for x in observations if x["end_to_end_seconds"] <= budget
                ]
                last = eligible[-1] if eligible else None
                matched.append(
                    {
                        "row": row,
                        "seed": seed,
                        "budget_seconds": budget,
                        "step": last["step"] if last else None,
                        "ce": last["ce"] if last else None,
                        "censored": last is None,
                    }
                )
    costs = {
        "training_seconds": sum(
            v["training_seconds"] for d in data.values() for v in d.values()
        ),
        "training_process_seconds": sum(
            v["elapsed_seconds"] for d in data.values() for v in d.values()
        ),
        "CPU_endpoint_seconds": sum(
            read(r / "quantization" / (j["id"] + ".json"))["total_cpu_seconds"]
            for j in plan["jobs"]
        ),
        "CPU_numerical_seconds": sum(
            read(r / "numerics" / (j["id"] + ".json"))["cpu_seconds"]
            for j in plan["jobs"]
        ),
        "checkpoint_bytes": sum(c["bytes"] for c in checkpoints),
        "billing": "recorded separately by owned-pod lifecycle; CPU command times are not additive GPU billable time",
    }
    record = {
        "status": "complete",
        "stage": "C",
        "D_enabled": False,
        "plan_sha256": sha(r / "STAGE-C-PLAN.json"),
        "rankings": rankings,
        "decisions": verdicts,
        "endpoints": endpoints,
        "targets": targets,
        "cost": costs,
        "limits": [
            "Eight independent seeds; no universal guarantee.",
            "Flagged numerical baselines remain flagged.",
            "Matched time is specific to this H100 configuration; no interpolation.",
            "Dequantized weight rounding is not packed-kernel speed or measured allocation.",
            "No depth scaling or subliminal-transfer experiment was run.",
        ],
        "depth_gate_candidates": sorted(
            {
                v["candidate"]
                for v in verdicts
                if v["kind"] in ("training", "quantization")
                and v["qualified_confirmation"]
            }
        ),
    }
    write(r / "STAGE-C-RESULTS.json", record)
    write(r / "STAGE-C-COST.json", costs)
    write(
        r / "STAGE-C-CHECKPOINTS.json",
        {"status": "complete", "outside_git": True, "checkpoints": checkpoints},
    )
    for name, rows in (
        ("STAGE-C-PAIRED.csv", paired),
        ("STAGE-C-CURVES.csv", curves),
        ("STAGE-C-MATCHED-TIME.csv", matched),
        ("STAGE-C-TARGETS.csv", targets),
    ):
        with (r / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Stage C confirmation results",
        "",
        "All 56 fixed trajectories reached step 4000 on the frozen H100 runtime. Each comparison uses eight fresh paired seeds and the reserved confirmation arrays. D remains disabled.",
        "",
        "Primary CE averages steps 2000 through 4000, inclusive. Lower is better. INT4 damage is measured against each checkpoint's own CPU clean final loss.",
        "",
        "| Row | Primary CE | Final GPU CE | INT4 CE | INT4 damage | Lowest all-seed precision | Historical and C samples pass |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rankings:
        m = row["means"]
        lines.append(
            f"| {row['row']} | {m['primary_ce']['mean']:.6f} | {m['gpu_final_ce']['mean']:.6f} | {m['primary_int4_ce']['mean']:.6f} | {m['int4_damage']['mean']:.6f} | {row['lowest_passing_tested_precision']} | {row['historical_and_confirmation_samples_pass']} |"
        )
    lines += ["", "Each predeclared claim is reported separately:", ""]
    for v in verdicts:
        lines.append(
            f"- {v['name']}: performance rule {v['meets_performance_rule']}; numerical qualification {v['numerical_qualification_pass']}; qualified confirmation {v['qualified_confirmation']}. Paired values and each criterion are retained in the JSON. Holm-adjusted one-sided sign p={v['holm_adjusted_p']:.6g}."
        )
    lines += [
        "",
        "Time-to-target, matched-time loss, scalar storage sensitivity, secondary quantizers, every seed, and all numerical misses are retained in the JSON and CSV files. Small scalar storage savings do not establish useful model compression. All checkpoints include model, optimizer and RNG state.",
        "",
        "Depth-gate candidates: "
        + (", ".join(record["depth_gate_candidates"]) or "none")
        + ". This prepares a later resource decision; it does not execute D.",
        "",
        "",
    ]
    (r / "STAGE-C-RESULTS.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = r / "stage-c-plots"
    plots.mkdir(exist_ok=True)
    for x, name in (
        ("tokens", "loss-vs-tokens"),
        ("end_to_end_seconds", "loss-vs-time"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for row, values in data.items():
            observations = [values[s]["validation"] for s in SEEDS]
            ax.plot(
                [statistics.mean(t[k][x] for t in observations) for k in range(40)],
                [statistics.mean(t[k]["ce"] for t in observations) for k in range(40)],
                label=row,
            )
        ax.set(xlabel=x, ylabel="confirmation curve CE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / (name + ".png"), dpi=160)
        plt.close(fig)
    for metric, name in (
        ("primary_int4_ce", "absolute-int4-loss"),
        ("int4_damage", "int4-damage"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        names = list(data)
        ax.boxplot(
            [[data[n][s][metric] for s in SEEDS] for n in names], tick_labels=names
        )
        ax.set(ylabel=metric)
        fig.tight_layout()
        fig.savefig(plots / (name + ".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    analyze(a.root)
