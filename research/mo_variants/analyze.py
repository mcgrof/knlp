"""Complete Stage A tables. Discovery ranks do not establish confirmed winners."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

from .coordinator import atomic, read, sha


def describe(values):
    if not values:
        return None
    n = len(values)
    mean = st.mean(values)
    margin = (
        {2: 12.706, 3: 4.303}.get(n, 0) * st.stdev(values) / math.sqrt(n)
        if n > 1
        else None
    )
    return {
        "n": n,
        "mean": mean,
        "median": st.median(values),
        "min": min(values),
        "max": max(values),
        "t_95_interval_descriptive": (
            [mean - margin, mean + margin] if margin is not None else None
        ),
    }


def target(result, ce):
    curve = result["validation"]
    for a, b in zip(curve, curve[1:]):
        if a["ce"] <= ce and b["ce"] <= ce and b["step"] - a["step"] == 100:
            return {
                "censored": False,
                "step": b["step"],
                "first_crossing_step": a["step"],
                "tokens": b["tokens"],
                "training_seconds": b["training_seconds"],
                "elapsed_seconds": b["end_to_end_seconds"],
                "estimated_flops": b["step"] * result["optimizer_step_flops_estimate"],
            }
    return {
        "censored": True,
        "step": result["completed_step"],
        "tokens": result["completed_step"] * 2048,
        "training_seconds": result["training_seconds"],
        "elapsed_seconds": result["end_to_end_seconds"],
        "estimated_flops": result["completed_step"]
        * result["optimizer_step_flops_estimate"],
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    records = Path(root) / "records"
    selected = read(records / "SELECTED-LRS.json")
    jobs = read(records / "JOBS.json")["jobs"]
    results = {
        j["id"]: read(records / "runs" / j["id"] / "RESULT.json")
        for j in jobs
        if j["stage"].startswith("A-")
        and (records / "runs" / j["id"] / "RESULT.json").exists()
    }
    endpoints, curves, pilots, paired, rankings = [], [], [], [], []
    for j in jobs:
        if j["stage"] == "A-pilot":
            r = results.get(j["id"], {})
            values = {c["step"]: c["ce"] for c in r.get("validation", [])}
            pilots.append(
                {
                    "variant": j["variant"],
                    "lr": j["lr"],
                    "status": r.get("status", "missing"),
                    "ce300": values.get(300),
                    "ce400": values.get(400),
                    "ce500": values.get(500),
                    "score": (
                        st.mean([values[s] for s in (300, 400, 500)])
                        if all(s in values for s in (300, 400, 500))
                        else None
                    ),
                    "training_seconds": r.get("training_seconds"),
                    "error": r.get("error"),
                }
            )
        if j["stage"] != "A-discovery":
            continue
        r = results.get(j["id"])
        if not r or r["status"] != "complete":
            endpoints.append(
                {
                    "variant": j["variant"],
                    "seed": j["seed"],
                    "status": r["status"] if r else "missing",
                    "reason": (
                        r.get("error") if r else selected[j["variant"]]["status"]
                    ),
                }
            )
            continue
        q = read(records / "quantization" / (j["id"] + ".json"))
        d = read(records / "numerics" / (j["id"] + ".json"))
        assert q["status"] == "complete" and d["status"] == "complete"
        ce = [c["ce"] for c in r["validation"] if 500 <= c["step"] <= 1000]
        assert len(ce) == 6
        ep = {
            "variant": j["variant"],
            "seed": j["seed"],
            "status": "complete",
            "lr": r["metadata"]["selected_lr"],
            "primary_ce500_1000": st.mean(ce),
            "gpu_clean_endpoint_ce": r["validation"][-1]["ce"],
            "cpu_clean_endpoint_ce": q["clean_ce"],
            "cpu_minus_gpu_clean_ce": q["cpu_minus_gpu_clean_ce"],
            "training_seconds": r["training_seconds"],
            "end_to_end_seconds": r["end_to_end_seconds"],
            "finite_reference_pass": d["finite_reference_pass"],
            "sampled_finite_misses": len(d["finite_misses"]),
            "checkpoint_sha256": r["checkpoint"]["sha256"],
            "checkpoint_path": r["checkpoint"]["path"],
            "quantized_fraction": q["coverage"],
            "ce3_target": target(r, 3),
            "ce4_target": target(r, 4),
        }
        for bit in q["bits"]:
            for key in (
                "ce",
                "delta_ce",
                "perplexity_ratio",
                "within_point1_nats",
                "ideal_packed_total_bytes",
                "scale_bytes",
                "unquantized_parameter_bytes",
            ):
                ep[f"int{bit['bits']}_{key}"] = bit[key]
        endpoints.append(ep)
        for c in r["validation"]:
            curves.append({"variant": j["variant"], "seed": j["seed"], **c})
    completed = {
        (e["variant"], e["seed"]): e for e in endpoints if e["status"] == "complete"
    }
    for e in completed.values():
        for base in ("V00", "V01"):
            b = completed.get((base, e["seed"]))
            if b:
                paired.append(
                    {
                        "variant": e["variant"],
                        "baseline": base,
                        "seed": e["seed"],
                        **{
                            key: e[key] - b[key]
                            for key in (
                                "primary_ce500_1000",
                                "gpu_clean_endpoint_ce",
                                "int4_ce",
                                "int4_delta_ce",
                                "int3_ce",
                                "int3_delta_ce",
                            )
                        },
                    }
                )
    for variant in sorted(selected):
        eps = [e for e in completed.values() if e["variant"] == variant]
        row = {
            "variant": variant,
            "lr": selected[variant]["lr"],
            "completed_seeds": len(eps),
            "boundary_best_on_grid": selected[variant]["boundary_best_on_grid"],
            "qualified_finite_samples": len(eps) == 3
            and all(e["finite_reference_pass"] for e in eps)
            and all(
                read(records / "numerics" / (j["id"] + ".json"))[
                    "finite_reference_pass"
                ]
                for j in jobs
                if j["variant"] == variant
                and j["stage"] in ("A-qualification", "A-pilot")
                and results.get(j["id"], {}).get("status") == "complete"
            ),
            "int4_pass_all3": len(eps) == 3
            and all(e["int4_within_point1_nats"] for e in eps),
            "int3_pass_all3": len(eps) == 3
            and all(e["int3_within_point1_nats"] for e in eps),
            "maximum_tested_quantization": (
                "INT3"
                if len(eps) == 3 and all(e["int3_within_point1_nats"] for e in eps)
                else (
                    "INT4"
                    if len(eps) == 3 and all(e["int4_within_point1_nats"] for e in eps)
                    else "neither passed all3"
                )
            ),
        }
        for key in (
            "primary_ce500_1000",
            "gpu_clean_endpoint_ce",
            "int4_ce",
            "int4_delta_ce",
            "int3_ce",
            "int3_delta_ce",
            "training_seconds",
            "end_to_end_seconds",
        ):
            row[key] = describe([e[key] for e in eps])
        row["paired_vs_baselines"] = {
            b: {
                key: describe(
                    [
                        p[key]
                        for p in paired
                        if p["variant"] == variant and p["baseline"] == b
                    ]
                )
                for key in (
                    "primary_ce500_1000",
                    "gpu_clean_endpoint_ce",
                    "int4_ce",
                    "int4_delta_ce",
                    "int3_ce",
                    "int3_delta_ce",
                )
            }
            for b in ("V00", "V01")
        }
        row["targets"] = {
            str(c): [{"seed": e["seed"], **e[f"ce{c}_target"]} for e in eps]
            for c in (3, 4)
        }
        rankings.append(row)
    eligible = sorted(
        [r for r in rankings if r["completed_seeds"] == 3],
        key=lambda r: (r["primary_ce500_1000"]["mean"], r["variant"]),
    )
    for i, r in enumerate(eligible):
        r["clean_rank"] = i + 1
    events = [
        json.loads(l)
        for l in (records / "STAGE-A-EVENTS.jsonl").read_text().splitlines()
    ]
    terminals = [e for e in events if e["event"] == "terminal"]
    assert len({e["job_id"] for e in terminals}) == len(terminals)
    cost = {
        "stage_a_reservation_seconds": sum(e["reservation_seconds"] for e in terminals),
        "stage_a_training_seconds": sum(e["training_seconds"] for e in terminals),
        "stage_a_scheduled_eval_seconds": sum(
            e["scheduled_eval_seconds"] for e in terminals
        ),
        "cpu_diagnostics_seconds": sum(
            e["seconds"] for e in events if e["event"] == "cpu_diagnostics"
        ),
        "cpu_quantization_seconds": sum(
            e["seconds"] for e in events if e["event"] == "cpu_quantization"
        ),
        "by_stage": {
            s: {
                "jobs": sum(e["stage"] == s for e in terminals),
                "reservation_seconds": sum(
                    e["reservation_seconds"] for e in terminals if e["stage"] == s
                ),
            }
            for s in ("A-qualification", "A-pilot", "A-discovery")
        },
        "preparation_reservation_seconds_prior_to_this_run": 184,
        "parent_ledger_seconds_unchanged": 23413.955,
        "energy": "unavailable AMD power sensor; no energy claim",
    }
    cost["followup_reserved_seconds_including_preparation"] = (
        cost["stage_a_reservation_seconds"] + 184
    )
    cost["additive_seconds_with_parent_different_timing_basis"] = (
        cost["followup_reserved_seconds_including_preparation"] + 23413.955
    )
    record = {
        "status": "complete",
        "scope": "Stage A32-row discovery screen only, fixed discovery recipe",
        "source_sha256": sha(records / "SOURCE.json"),
        "rankings": rankings,
        "endpoints": endpoints,
        "paired_differences": paired,
        "lr_surface": pilots,
        "cost": cost,
        "limits": [
            "three discovery seeds after32-row tuning search; no independent eight-seed confirmation or causal winner verdict",
            "no new confirmation-array evaluation",
            "no D16/D64 replication",
            "only primary rowwise INT4/INT3 quantizer; no INT2/groupwise/control-rounding test",
            "dequantized CPU evaluation does not establish packed-kernel memory or speed",
            "unresolved actual-routing finite misses remain visible",
        ],
    }
    atomic(records / "STAGE-A-RESULTS.json", record)
    atomic(records / "STAGE-A-COST.json", cost)
    write_csv(records / "LR-SURFACE.csv", pilots)
    write_csv(records / "DISCOVERY-CURVES.csv", curves)
    matched = []
    for budget in (180, 300, 450, 600):
        for e in completed.values():
            cs = [
                c
                for c in curves
                if c["variant"] == e["variant"] and c["seed"] == e["seed"]
            ]
            prior = [c for c in cs if c["end_to_end_seconds"] <= budget]
            supported = bool(prior and cs[-1]["end_to_end_seconds"] >= budget)
            last = prior[-1] if supported else None
            matched.append(
                {
                    "variant": e["variant"],
                    "seed": e["seed"],
                    "elapsed_budget_seconds": budget,
                    "ce_last_scheduled_before_budget": (last["ce"] if last else None),
                    "observed_at_seconds": (
                        last["end_to_end_seconds"] if last else None
                    ),
                    "observed_step": last["step"] if last else None,
                    "supported_by_trajectory": supported,
                    "method": "last scheduled observation; no interpolation or endpoint extrapolation",
                }
            )
    write_csv(records / "MATCHED-TIME-BUDGETS.csv", matched)
    write_csv(records / "PAIRED-DIFFERENCES.csv", paired)
    write_csv(
        records / "DISCOVERY-ENDPOINTS.csv",
        [
            {k: json.dumps(v) if isinstance(v, dict) else v for k, v in e.items()}
            for e in endpoints
        ],
    )
    flat = [
        {
            "variant": r["variant"],
            "rank": r.get("clean_rank"),
            "lr": r["lr"],
            "complete_seeds": r["completed_seeds"],
            "primary_mean": (
                r["primary_ce500_1000"]["mean"] if r["primary_ce500_1000"] else None
            ),
            "int4_mean": r["int4_ce"]["mean"] if r["int4_ce"] else None,
            "int4_penalty": (
                r["int4_delta_ce"]["mean"] if r["int4_delta_ce"] else None
            ),
            "int3_mean": r["int3_ce"]["mean"] if r["int3_ce"] else None,
            "maximum_tested_quantization": r["maximum_tested_quantization"],
            "finite_reference_pass": r["qualified_finite_samples"],
            "boundary_best_on_grid": r["boundary_best_on_grid"],
        }
        for r in rankings
    ]
    write_csv(records / "ALL-ROWS-RANKING.csv", flat)
    lines = [
        "# Stage A results",
        "",
        "The broad32-row TinyStories screen is complete. These are discovery results, not independent confirmation.",
        "",
        "All rows, seeds, LR outcomes, failures and sampled finite-reference misses are retained. Lower cross-entropy (CE) means better prediction.",
        "",
        "| Row | Clean rank | LR | Seeds | Primary mean CE | INT4 absolute CE | INT4 damage | Lowest passing tested precision | Finite samples pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    def fmt(v):
        return f"{v:.6f}" if v is not None else "missing"

    for r in sorted(flat, key=lambda r: (r["rank"] or 99, r["variant"])):
        lines.append(
            f"| {r['variant']} | {r['rank'] or 'missing'} | {r['lr']} | {r['complete_seeds']} | {fmt(r['primary_mean'])} | {fmt(r['int4_mean'])} | {fmt(r['int4_penalty'])} | {r['maximum_tested_quantization']} | {r['finite_reference_pass']} |"
        )
    lines.extend(
        [
            "",
            f"Stage A reserved {cost['stage_a_reservation_seconds']/3600:.3f} GPU-hours. CPU routing checks used {cost['cpu_diagnostics_seconds']/3600:.3f} hours; CPU quantization used {cost['cpu_quantization_seconds']/3600:.3f} hours. Tuning, qualification and selected-model costs are separate in STAGE-A-COST.json.",
            "",
            "Use PAIRED-DIFFERENCES.csv for every fresh-seed comparison against M (V00) and original M-O (V01). Negative differences favor the variant. Use the same-LR pilot surface as a one-seed tuning diagnostic. It does not isolate architecture across fresh seeds.",
            "",
            "Convergence requires CE<=3 at two consecutive100-step evaluations. CE<=4 is diagnostic. STAGE-A-RESULTS.json reports censored targets in steps, tokens, estimated FLOPs and elapsed time. Lower endpoint CE alone is not faster convergence.",
            "",
            "Ordered-depth controls here are fixed rows. Each selected operator's own shuffled/constant controls, eight-seed confirmation, other depths, another quantizer and control rounding have not run. Stage A cannot establish those mechanisms or guarantees.",
            "",
            "Every endpoint and optimizer/RNG state remains outside Git at its recorded checkpoint path. Ideal packed bytes include scales, unquantized parameters and persistent buffers; CPU dequantization is not packed inference.",
            "",
            "The authorized run stops here. No B/C/D jobs are submitted.",
        ]
    )

    for baseline in ("V00", "V01"):
        favorable = [
            r["variant"]
            for r in rankings
            if r["variant"] != baseline
            and r["paired_vs_baselines"][baseline]["primary_ce500_1000"]
            and r["paired_vs_baselines"][baseline]["primary_ce500_1000"]["n"] == 3
            and r["paired_vs_baselines"][baseline]["primary_ce500_1000"]["max"] < 0
        ]
        lines.extend(
            [
                "",
                f"Rows with lower primary CE in all three discovery pairs versus {baseline}: {', '.join(favorable) or 'none'}. This is an observed discovery direction; independent confirmation and the practical winner rule have not been tested.",
            ]
        )
    lines.extend(
        [
            "",
            "MATCHED-TIME-BUDGETS.csv compares the last scheduled observation within common elapsed budgets. It leaves missing support blank, with no interpolation or endpoint extrapolation.",
            "",
            "The finite-reference flag includes each row's qualification and LR pilot samples as well as discovery samples. Unresolved misses prevent a qualified winner claim even when the measured quality improves.",
        ]
    )
    (records / "RESULTS.md").write_text("\n".join(lines) + "\n")
    plots(records, rankings, endpoints, curves, paired)


def plots(records, rankings, endpoints, curves, paired):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = records / "plots"
    out.mkdir(exist_ok=True)
    colors = plt.get_cmap("turbo")
    for axis_key, name, xlabel in (
        ("tokens", "loss-vs-tokens", "Predicted training tokens"),
        ("end_to_end_seconds", "loss-vs-time", "Elapsed trajectory seconds"),
    ):
        fig, ax = plt.subplots(figsize=(11, 7))
        for i, r in enumerate(rankings):
            for seed in (76311, 76312, 76313):
                cs = [
                    c
                    for c in curves
                    if c["variant"] == r["variant"] and c["seed"] == seed
                ]
                ax.plot(
                    [c[axis_key] for c in cs],
                    [c["ce"] for c in cs],
                    color=colors(i / 31),
                    alpha=0.55,
                    linewidth=0.7,
                    label=r["variant"] if seed == 76311 else None,
                )
        ax.set(xlabel=xlabel, ylabel="Discovery curve CE (nats)")
        ax.legend(ncol=4, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for i, r in enumerate(rankings):
        eps = [
            e
            for e in endpoints
            if e["variant"] == r["variant"] and e["status"] == "complete"
        ]
        for e in eps:
            axs[0].scatter(i, e["int4_delta_ce"], s=10, color=colors(i / 31))
            axs[1].scatter(i, e["int4_ce"], s=10, color=colors(i / 31))
    axs[0].axhline(0.1, color="black", linestyle="--", linewidth=0.8)
    for ax in axs:
        ax.set_xticks(
            range(32), [r["variant"] for r in rankings], rotation=90, fontsize=7
        )
    axs[0].set_ylabel("INT4 damage (nats)")
    axs[1].set_ylabel("Absolute INT4 CE (nats)")
    fig.tight_layout()
    fig.savefig(out / "int4-damage-and-quality.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(rankings):
        values = [
            p["primary_ce500_1000"]
            for p in paired
            if p["variant"] == r["variant"] and p["baseline"] == "V00"
        ]
        ax.scatter([i] * len(values), values, s=12, color=colors(i / 31))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(32), [r["variant"] for r in rankings], rotation=90, fontsize=7)
    ax.set_ylabel("Paired primary CE minus M (nats)")
    fig.tight_layout()
    fig.savefig(out / "paired-discovery-vs-M.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, r in enumerate(rankings):
        eps = [
            e
            for e in endpoints
            if e["variant"] == r["variant"] and e["status"] == "complete"
        ]
        for bit in (4, 3):
            if eps:
                ax.scatter(
                    st.mean(e[f"int{bit}_ideal_packed_total_bytes"] for e in eps)
                    / 2**20,
                    st.mean(e[f"int{bit}_ce"] for e in eps),
                    marker="o" if bit == 4 else "x",
                    color=colors(i / 31),
                    label=r["variant"] if bit == 4 else None,
                )
    ax.set(
        xlabel="Ideal packed artifact MiB (includes scales and FP32 exclusions)",
        ylabel="Absolute endpoint CE (nats)",
    )
    ax.legend(ncol=4, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "bytes-vs-quality.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    analyze(args.root)
