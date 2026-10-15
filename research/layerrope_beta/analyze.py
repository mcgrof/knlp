# SPDX-License-Identifier: MIT
"""Report every beta, seed and quantization scope without selecting a final-data winner."""

import argparse
from pathlib import Path

import numpy as np

from research.layerrope_beta.plan import language_rows, LM_SEEDS
from research.mo_variants.stage_c_common import read, write
from research.subliminal_learning import analyze as mnist_analysis


def analyze(root):
    root = Path(root)
    plan, selected = read(root / "PLAN.json"), read(root / "SELECTED.json")
    language = {}
    for row in language_rows():
        results, quantized, statistics = [], [], []
        for seed in LM_SEEDS:
            folder = root / "language/runs" / f"final-{row['id']}-seed{seed}"
            result = read(folder / "RESULT.json")
            if result["status"] != "complete" or result["completed_steps"] != 2000:
                raise ValueError("Incomplete language endpoint: " + str(folder))
            results.append(result)
            quantized.append(read(folder / "QUANTIZATION.json"))
            statistics.append(read(folder / "statistics-step2000.json"))
        policies = {}
        names = [x["name"] for x in quantized[0]["weight_activation_residual"]]
        for name in names:
            cases = [
                next(x for x in q["weight_activation_residual"] if x["name"] == name)
                for q in quantized
            ]
            policies[name] = dict(
                mean_ce=float(np.mean([x["ce"] for x in cases])),
                mean_damage=float(np.mean([x["delta_ce"] for x in cases])),
                seed_damage={str(s): x["delta_ce"] for s, x in zip(LM_SEEDS, cases)},
                passes=sum(x["delta_ce"] <= 0.1 for x in cases),
                n=3,
                ideal_weight_bytes=cases[0].get("ideal_weight_bytes"),
                weight_parameter_fraction=cases[0]["weight_parameter_fraction"],
            )
        cache = {}
        for name in [x["name"] for x in quantized[0]["cache"]]:
            cases = [
                next(x for x in q["cache"] if x["name"] == name) for q in quantized
            ]
            cache[name] = dict(
                mean_ce=float(np.mean([x["ce"] for x in cases])),
                mean_damage=float(np.mean([x["delta_ce"] for x in cases])),
                seed_ce={str(s): x["ce"] for s, x in zip(LM_SEEDS, cases)},
                seed_damage={str(s): x["delta_ce"] for s, x in zip(LM_SEEDS, cases)},
                passes=sum(x["delta_ce"] <= 0.1 for x in cases),
                n=3,
                ideal_bytes_per_token=cases[0]["storage"]["ideal_total_bytes"],
            )
        increment = {
            str(seed): q["seed_ce"][str(seed)]
            - cache["symmetric-K3V3"]["seed_ce"][str(seed)]
            for seed in LM_SEEDS
            for q in [cache["symmetric-K3V2"]]
        }
        residuals = []
        for seed, stats in zip(LM_SEEDS, statistics):
            last = next(
                x
                for x in stats["residuals"]
                if x["layer"] == 31 and x["site"] == "post_mlp"
            )
            residuals.append(
                dict(
                    seed=seed,
                    per_stream=last["per_stream"],
                    stream_mean=last.get("stream_mean"),
                )
            )
        language[row["id"]] = dict(
            arm=row["arm"],
            beta=row["beta"],
            selected_lr=selected["lrs"][row["id"]],
            mean_primary_ce=float(
                np.mean(
                    [
                        np.mean([p["ce"] for p in r["curve"] if p["step"] >= 1000])
                        for r in results
                    ]
                )
            ),
            mean_clean_ce=float(np.mean([q["clean"]["ce"] for q in quantized])),
            clean_ce_by_seed={
                str(s): q["clean"]["ce"] for s, q in zip(LM_SEEDS, quantized)
            },
            incremental_k3v3_to_k3v2=dict(
                mean=float(np.mean(list(increment.values()))), by_seed=increment
            ),
            policies=policies,
            cache=cache,
            last_layer_residuals=residuals,
            training_seconds_by_seed={
                str(s): sum(x["seconds"] for x in r["steps"])
                for s, r in zip(LM_SEEDS, results)
            },
        )
    for name, row in language.items():
        if row["arm"] not in ("L", "ML"):
            continue
        baseline = language["R" if row["arm"] == "L" else "M"]
        pairs = [
            row["incremental_k3v3_to_k3v2"]["by_seed"][str(s)]
            - baseline["incremental_k3v3_to_k3v2"]["by_seed"][str(s)]
            for s in LM_SEEDS
        ]
        row["incremental_damage_minus_same_routing_baseline"] = dict(
            mean=float(np.mean(pairs)),
            pairs=pairs,
            lower_damage_pairs=sum(x < 0 for x in pairs),
            n=3,
        )
    for name, row in language.items():
        if row["arm"] == "M":
            baseline_name = "R"
        elif row["arm"] == "ML":
            baseline_name = name.replace("ML-", "L-", 1)
        else:
            continue
        baseline = language[baseline_name]
        pairs = [
            row["incremental_k3v3_to_k3v2"]["by_seed"][str(s)]
            - baseline["incremental_k3v3_to_k3v2"]["by_seed"][str(s)]
            for s in LM_SEEDS
        ]
        row["mhc_incremental_damage_contrast"] = dict(
            baseline=baseline_name,
            mean=float(np.mean(pairs)),
            pairs=pairs,
            lower_damage_pairs=sum(x < 0 for x in pairs),
            n=3,
        )
    runs = {
        name: mnist_analysis.load(str(root / "mnist/runs" / name))
        for name in plan["mnist"]["recipes"]
    }
    for name, run in runs.items():
        if len(run["keys"]) != 10 or not all(
            len(run["curves"][k][stage]) == (5 if stage == "teacher" else 80)
            for k in run["keys"]
            for stage in ("teacher",) + mnist_analysis.STAGES
        ):
            raise ValueError("Incomplete MNIST curves: " + name)
        packets = read(root / "mnist/runs" / name / "results.json")
        for packet in packets.values():
            values = packet["results"]
            if not values["teacher_frozen"] or not all(
                all(values[field])
                for field in (
                    "teacher_aux_rows_unchanged",
                    "aux_shared_digit_rows_unchanged",
                    "aux_other_digit_rows_unchanged",
                )
            ):
                raise ValueError("MNIST immutable-state invariant failed: " + name)
        baseline_name = name.split("-full-beta-", 1)[0].split("-input-only", 1)[0]
        baseline_packets = read(root / "mnist/runs" / baseline_name / "results.json")
        for key, packet in packets.items():
            baseline = baseline_packets[key]
            if packet["keys"] != baseline["keys"] or any(
                packet["results"][field] != baseline["results"][field]
                for field in ("common_digest", "common_digest_other", "noise_digest")
            ):
                raise ValueError(
                    "MNIST common weights, controls or noise differ: " + name
                )
    comparisons = {}
    for name, run in runs.items():
        if "-full-beta-" in name:
            baseline = name.split("-full-beta-", 1)[0]
            comparisons[name] = mnist_analysis.compare(
                runs[baseline],
                run,
                plan["mnist"]["confidence_level"],
                95.0,
                plan["mnist"]["bootstrap_resamples"],
            )
    summaries = {}
    digit_quantization = {}
    for name, run in runs.items():
        table = run["milestones"][80]
        fields = (
            "acc_init",
            "acc_teacher",
            "acc_aux_shared",
            "acc_full_shared",
            "gain_aux_shared",
            "gain_aux_other",
        )
        summaries[name] = {
            field: float(np.mean([100 * float(table[k][field]) for k in run["keys"]]))
            for field in fields
        }
        quantized = read(root / "mnist/runs" / name / "QUANTIZATION.json")
        if quantized["status"] != "complete" or quantized["seed_order"] != [
            f"seed-{s}" for s in plan["mnist"]["recipes"][name]["seeds"]
        ]:
            raise ValueError("Incomplete or mismatched MNIST quantization: " + name)
        roles = {}
        for role in quantized["roles"]:
            if not role["weights_unchanged"] or len(role["cases"]) != 12:
                raise ValueError("Incomplete MNIST quantization role: " + name)
            roles[role["role"]] = dict(
                mean_clean_accuracy_pp=float(100 * np.mean(role["clean"]["accuracy"])),
                mean_clean_ce=float(np.mean(role["clean"]["ce"])),
                policies={
                    case["name"]: dict(
                        mean_accuracy_pp=float(100 * np.mean(case["accuracy"])),
                        mean_damage_pp=float(np.mean(case["accuracy_damage_pp"])),
                        seed_damage_pp=dict(
                            zip(quantized["seed_order"], case["accuracy_damage_pp"])
                        ),
                        mean_ce=float(np.mean(case["ce"])),
                        mean_delta_ce=float(np.mean(case["delta_ce"])),
                        weight_parameter_fraction=case["weight_parameter_fraction"],
                        ideal_weight_bytes_per_model=case.get(
                            "ideal_weight_bytes_per_model"
                        ),
                    )
                    for case in role["cases"]
                },
            )
        if set(roles) != {"teacher", "aux_shared", "full_shared"}:
            raise ValueError("Missing MNIST quantization roles: " + name)
        digit_quantization[name] = roles
    result = dict(
        status="complete",
        language=language,
        mnist=dict(
            summaries=summaries,
            primary_comparisons=comparisons,
            quantization=digit_quantization,
        ),
        limits=plan["limits"],
    )
    write(root / "REPORT.json", result)
    lines = [
        "# Full input/output gain and beta study",
        "",
        "All beta initializations are reported. Story-model comparisons use three paired seeds and remain descriptive.",
        "",
        "| Row | Initial beta | Selected LR | Clean CE | K3/V3 to K3/V2 increment | Block FP4 damage | All-matrix FP4 damage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in language.items():
        lines.append(
            f"| {name} | {r['beta']} | {r['selected_lr']} | {r['mean_clean_ce']:.6f} | {r['incremental_k3v3_to_k3v2']['mean']:.6f} | {r['policies']['blocks-FP4-weights']['mean_damage']:.6f} | {r['policies']['all-FP4-weights']['mean_damage']:.6f} |"
        )
    lines += [
        "",
        "MNIST gains are percentage points above each arm's own initialization. The nine primary contrasts use the frozen family-adjusted confidence level.",
        "",
        "| Row | Initial accuracy | Teacher accuracy | Auxiliary accuracy | Transfer gain | All-output accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in summaries.items():
        lines.append(
            f"| {name} | {r['acc_init']:.3f} | {r['acc_teacher']:.3f} | {r['acc_aux_shared']:.3f} | {r['gain_aux_shared']:.3f} | {r['acc_full_shared']:.3f} |"
        )
    lines += [
        "",
        "Quantization is numerical simulation with dequantization. Read the report's per-seed results, coverage and byte counts before calling a lower loss or variance a compression improvement.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    analyze(parser.parse_args().root)
