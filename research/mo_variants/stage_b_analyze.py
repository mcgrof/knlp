"""Report all fixed continuations and prepare, without executing, C nominees."""

import argparse
import csv
from pathlib import Path
import statistics as st

from .analyze import describe, target
from .coordinator import atomic, read, sha
from .stage_b_prepare import ROWS, SEEDS


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    r = Path(root) / "records"
    selection = read(r / "STAGE-B-SELECTION.json")
    scope = {x["variant"]: x for x in read(r / "NUMERICAL-SCOPE.json")["rows"]}
    old = read(r / "STAGE-A-RESULTS.json")
    old_ep = {(x["variant"], x["seed"]): x for x in old["endpoints"]}
    endpoints, curves, checkpoints = [], [], []
    for job in selection["jobs"]:
        result = read(r / "runs" / job["id"] / "RESULT.json")
        assert result["status"] == "complete" and result["completed_step"] == 2000
        quant = read(r / "quantization" / (job["id"] + ".json"))
        numerical = read(r / "numerics" / (job["id"] + ".json"))
        assert quant["status"] == numerical["status"] == "complete"
        prior = read(r / "runs" / job["resume_from"] / "RESULT.json")
        joined = list(prior["validation"])
        for point in result["validation"]:
            joined.append(
                {
                    **point,
                    "training_seconds": prior["training_seconds"]
                    + point["training_seconds"],
                    "end_to_end_seconds": prior["end_to_end_seconds"]
                    + point["end_to_end_seconds"],
                }
            )
        combined = {
            **result,
            "validation": joined,
            "training_seconds": prior["training_seconds"] + result["training_seconds"],
            "end_to_end_seconds": prior["end_to_end_seconds"]
            + result["end_to_end_seconds"],
        }
        values = [p["ce"] for p in joined if 1000 <= p["step"] <= 2000]
        assert len(values) == 11
        endpoint = {
            "variant": job["variant"],
            "seed": job["seed"],
            "job_id": job["id"],
            "primary_ce1000_2000": st.mean(values),
            "gpu_clean_endpoint_ce": joined[-1]["ce"],
            "cpu_clean_endpoint_ce": quant["clean_ce"],
            "cpu_minus_gpu_clean_ce": quant["cpu_minus_gpu_clean_ce"],
            "stage_b_training_seconds": result["training_seconds"],
            "stage_b_elapsed_seconds": result["end_to_end_seconds"],
            "combined_training_seconds": combined["training_seconds"],
            "combined_elapsed_seconds": combined["end_to_end_seconds"],
            "finite_reference_pass_at2000": numerical["finite_reference_pass"],
            "endpoint_finite_misses": len(numerical["finite_misses"]),
            "historical_row_finite_pass": scope[job["variant"]]["historical_row_flag"],
            "checkpoint": result["checkpoint"],
            "targets": {str(c): target(combined, c) for c in (3, 4)},
            "quantized_fraction": quant["coverage"],
        }
        for bit in quant["bits"]:
            for key in (
                "ce",
                "delta_ce",
                "within_point1_nats",
                "ideal_packed_total_bytes",
                "scale_bytes",
                "unquantized_parameter_bytes",
            ):
                endpoint[f"int{bit['bits']}_{key}"] = bit[key]
        endpoints.append(endpoint)
        curves.extend(
            {
                "variant": job["variant"],
                "seed": job["seed"],
                "stage": "A" if p["step"] <= 1000 else "B",
                **p,
            }
            for p in joined
        )
        checkpoints.append(
            {
                "job_id": job["id"],
                "step1000_retained": prior["checkpoint"],
                "step2000": result["checkpoint"],
            }
        )
    assert len(endpoints) == 24
    eps = {(e["variant"], e["seed"]): e for e in endpoints}
    metrics = (
        "primary_ce1000_2000",
        "gpu_clean_endpoint_ce",
        "int4_ce",
        "int4_delta_ce",
        "int3_ce",
        "int3_delta_ce",
    )
    paired = []
    for e in endpoints:
        for base in ("V00", "V01"):
            b = eps[base, e["seed"]]
            paired.append(
                {
                    "variant": e["variant"],
                    "baseline": base,
                    "seed": e["seed"],
                    **{k: e[k] - b[k] for k in metrics},
                    "target3_steps_difference": e["targets"]["3"]["step"]
                    - b["targets"]["3"]["step"],
                    "target3_elapsed_difference": e["targets"]["3"]["elapsed_seconds"]
                    - b["targets"]["3"]["elapsed_seconds"],
                }
            )
    order = [
        {
            "comparison": "V01-minus-" + base,
            "seed": seed,
            **{k: eps["V01", seed][k] - eps[base, seed][k] for k in metrics},
        }
        for base in ("V02", "V03")
        for seed in SEEDS
    ]
    rankings = []
    for variant in ROWS:
        ee = [eps[variant, s] for s in SEEDS]
        row = {
            "variant": variant,
            "completed_seeds": 3,
            "lr": 0.003,
            **{k: describe([e[k] for e in ee]) for k in metrics},
            "qualified_finite_samples": scope[variant]["historical_row_flag"]
            and all(e["finite_reference_pass_at2000"] for e in ee),
            "selected_recipe_finite_samples": all(
                e["finite_reference_pass_at2000"] for e in ee
            ),
            "historical_row_flag": scope[variant]["historical_row_flag"],
            "lowest_passing_tested_precision": next(
                (
                    f"INT{bits}"
                    for bits in (3, 4)
                    if all(e[f"int{bits}_within_point1_nats"] for e in ee)
                ),
                "neither",
            ),
            "paired": {
                b: {
                    k: describe([e[k] - eps[b, e["seed"]][k] for e in ee])
                    for k in metrics
                }
                for b in ("V00", "V01")
            },
            "targets": {
                str(c): [{"seed": e["seed"], **e["targets"][str(c)]} for e in ee]
                for c in (3, 4)
            },
        }
        deltas = [
            e["primary_ce1000_2000"] - eps["V00", e["seed"]]["primary_ce1000_2000"]
            for e in ee
        ]
        row["raw_training_eligible"] = (
            variant != "V00"
            and sum(x < 0 for x in deltas) >= 2
            and row["paired"]["V00"]["gpu_clean_endpoint_ce"]["mean"] < 0
        )
        row["raw_quantization_eligible"] = (
            variant != "V00"
            and row["paired"]["V00"]["int4_delta_ce"]["mean"] < 0
            and row["paired"]["V00"]["gpu_clean_endpoint_ce"]["mean"] <= 0.01
        )
        row["a_primary_delta_vs_M"] = describe(
            [
                old_ep[variant, s]["primary_ce500_1000"]
                - old_ep["V00", s]["primary_ce500_1000"]
                for s in SEEDS
            ]
        )
        row["a_int4_delta_vs_M"] = describe(
            [old_ep[variant, s]["int4_ce"] - old_ep["V00", s]["int4_ce"] for s in SEEDS]
        )
        rankings.append(row)

    def nominate(track, qualified):
        eligible = [
            x
            for x in rankings
            if x["raw_" + track + "_eligible"]
            and (not qualified or x["qualified_finite_samples"])
        ]
        metric = "primary_ce1000_2000" if track == "training" else "int4_ce"
        return (
            min(
                eligible,
                key=lambda x: (x[metric]["mean"], x[metric]["max"], x["variant"]),
            )["variant"]
            if eligible
            else None
        )

    nominations = {
        kind: {
            track: nominate(track, kind == "qualified")
            for track in ("training", "quantization")
        }
        for kind in ("raw", "qualified")
    }
    atomic(
        r / "STAGE-C-CANDIDATES.json",
        {
            "status": "prepared_only",
            "execution_authorized": False,
            "nominations": nominations,
            "source_stage_b_selection_sha256": sha(r / "STAGE-B-SELECTION.json"),
            "candidate_eligibility": rankings,
            "policy": "raw rankings and numerically qualified rankings shown separately; no threshold changes; unresolved historical misses excluded from qualified nomination",
            "fresh_seeds_reserved": list(range(76411, 76419)),
            "confirmation_data_evaluated": False,
            "independent_confirmation_claims_tested": False,
            "execution_gate": "separate post-B review and resource decision required",
        },
    )
    events = [
        read_event
        for line in (r / "STAGE-B-EVENTS.jsonl").read_text().splitlines()
        if (read_event := __import__("json").loads(line))
    ]
    terminal = [e for e in events if e["event"] == "terminal"]
    assert len(terminal) == 24 and len({e["job_id"] for e in terminal}) == 24
    cost = {
        "stage_b_reservation_seconds": sum(e["reservation_seconds"] for e in terminal),
        "stage_b_training_seconds": sum(e["training_seconds"] for e in terminal),
        "stage_b_scheduled_eval_seconds": sum(
            e["scheduled_eval_seconds"] for e in terminal
        ),
        "cpu_diagnostics_seconds": sum(
            e["seconds"] for e in events if e["event"] == "cpu_diagnostics"
        ),
        "cpu_quantization_seconds": sum(
            e["seconds"] for e in events if e["event"] == "cpu_quantization"
        ),
        "extra_checks": read(r / "STAGE-B-CHECKS-COST.json"),
        "cpu_preparation_seconds": read(r / "STAGE-B-PLAN.json")[
            "cpu_preparation_seconds"
        ],
        "endpoint_checkpoint_bytes": sum(e["checkpoint"]["bytes"] for e in endpoints),
        "stage_a_ledger_unchanged": True,
        "energy": "unavailable AMD sensor; no energy claim",
    }
    atomic(r / "STAGE-B-COST.json", cost)
    atomic(
        r / "STAGE-B-CHECKPOINTS.json",
        {"status": "complete", "checkpoints": checkpoints, "outside_git": True},
    )
    atomic(
        r / "STAGE-B-RESULTS.json",
        {
            "status": "complete",
            "scope": "24 exact checkpoint continuations; discovery only",
            "rankings": rankings,
            "endpoints": endpoints,
            "paired_differences": paired,
            "direct_order_controls": order,
            "stage_c_nominations": nominations,
            "cost": cost,
            "source_sha256": sha(r / "STAGE-B-SOURCE.json"),
            "timing_basis": "stitched A and B runner intervals; excludes startup/load, queue waits and CPU checks; reservations separately recorded",
            "limits": [
                "same three discovery seeds selected after 32-row search; no independent confirmation",
                "historical numerical misses remain unresolved",
                "no confirmation-array evaluation or new depth/quantizer/control-rounding",
            ],
        },
    )
    csv_file(r / "STAGE-B-PAIRED.csv", paired)
    csv_file(r / "STAGE-B-ORDER-CONTROLS.csv", order)
    csv_file(r / "STAGE-B-CURVES.csv", curves)
    import json

    csv_file(
        r / "STAGE-B-ENDPOINTS.csv",
        [
            {k: json.dumps(v) if isinstance(v, dict) else v for k, v in e.items()}
            for e in endpoints
        ],
    )
    matched = []
    for budget in read(r / "STAGE-B-PLAN.json")["matched_elapsed_budgets_seconds"]:
        for e in endpoints:
            trajectory = [
                c
                for c in curves
                if c["variant"] == e["variant"] and c["seed"] == e["seed"]
            ]
            before = [c for c in trajectory if c["end_to_end_seconds"] <= budget]
            supported = bool(before and trajectory[-1]["end_to_end_seconds"] >= budget)
            point = before[-1] if supported else None
            matched.append(
                {
                    "variant": e["variant"],
                    "seed": e["seed"],
                    "elapsed_budget_seconds": budget,
                    "supported": supported,
                    "ce": point["ce"] if point else None,
                    "step": point["step"] if point else None,
                    "observed_seconds": point["end_to_end_seconds"] if point else None,
                }
            )
    csv_file(r / "STAGE-B-MATCHED-TIME.csv", matched)
    lines = [
        "# Stage B results",
        "",
        "All 24 fixed checkpoint continuations reached global step 2000. These are discovery results from the same three seeds. Stages C/D remain disabled.",
        "",
        "Primary CE averages steps 1000 through 2000 inclusive. Lower CE means better prediction. INT4 damage uses each checkpoint's own CPU clean baseline.",
        "",
        "| Variant | A primary delta vs M | B primary CE | B primary delta vs M | Endpoint CE | INT4 CE | INT4 damage | Lowest passing precision | Historical and B samples pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for x in rankings:
        lines.append(
            f"| {x['variant']} | {x['a_primary_delta_vs_M']['mean']:.6f} | {x['primary_ce1000_2000']['mean']:.6f} | {x['paired']['V00']['primary_ce1000_2000']['mean']:.6f} | {x['gpu_clean_endpoint_ce']['mean']:.6f} | {x['int4_ce']['mean']:.6f} | {x['int4_delta_ce']['mean']:.6f} | {x['lowest_passing_tested_precision']} | {x['qualified_finite_samples']} |"
        )
    lines.extend(
        [
            "",
            f"B reserved {cost['stage_b_reservation_seconds']/3600:.3f} GPU-hours, plus the separately recorded resume/replay checks. CPU endpoint screens used {cost['cpu_quantization_seconds']/3600:.3f} hours. All 1000-step states remain intact; 24 new model/optimizer/RNG checkpoints were saved and reload-hash verified.",
            "",
            "V19/V20 retain their historical flags from six misses at the unused .01 pilot rate. The selected .003 samples and new endpoint samples are reported separately. Exploratory continuation does not resolve numerical qualification.",
            "",
        ]
    )
    for x in rankings:
        if x["variant"] == "V00":
            continue
        vals = [
            p["primary_ce1000_2000"]
            for p in paired
            if p["variant"] == x["variant"] and p["baseline"] == "V00"
        ]
        direction = (
            "discovery improvement"
            if all(v < 0 for v in vals)
            else (
                "mixed-seed result"
                if any(v < 0 for v in vals)
                else "no discovery improvement"
            )
        )
        lines.append(
            f"{x['variant']}: {direction} versus M; primary paired deltas {', '.join(f'{v:+.6f}' for v in vals)}. Absolute INT4 delta versus M is {x['paired']['V00']['int4_ce']['mean']:+.6f}; damage delta is {x['paired']['V00']['int4_delta_ce']['mean']:+.6f}. "
            + (
                "Numerical qualification unresolved."
                if not x["qualified_finite_samples"]
                else "Recorded finite samples pass."
            )
            + ""
        )
        lines.append("")
        mo = x["paired"]["V01"]
        lines.append(
            f"Against original M-O, {x['variant']} has mean primary delta "
            f"{mo['primary_ce1000_2000']['mean']:+.6f}, absolute INT4 delta "
            f"{mo['int4_ce']['mean']:+.6f}, and damage delta "
            f"{mo['int4_delta_ce']['mean']:+.6f}."
        )
        earlier = x["a_primary_delta_vs_M"]["mean"]
        later = x["paired"]["V00"]["primary_ce1000_2000"]["mean"]
        trend = (
            "persists unchanged"
            if later == earlier and later < 0
            else (
                "persists and grows"
                if later < earlier < 0
                else (
                    "persists but shrinks"
                    if earlier < later < 0
                    else (
                        "reverses"
                        if earlier < 0 <= later
                        else (
                            "appears in the later window"
                            if later < 0 <= earlier
                            else "is absent in both window means"
                        )
                    )
                )
            )
        )
        lines.extend(
            [
                f"The mean training gain versus M {trend}; seed directions remain in the paired table.",
                "",
            ]
        )
    lines.extend(["Direct order controls:", ""])
    for base in ("V02", "V03"):
        rows = [x for x in order if x["comparison"] == "V01-minus-" + base]
        lines.append(
            f"V01 versus {base}: primary deltas {', '.join(f'{x['primary_ce1000_2000']:+.6f}' for x in rows)}; INT4 damage deltas {', '.join(f'{x['int4_delta_ce']:+.6f}' for x in rows)}. These three discovery pairs do not establish an independent ordered-depth effect."
        )
        lines.append("")
    target_steps = {e["targets"]["3"]["step"] for e in endpoints}
    same_target = all(
        e["targets"]["3"]["step"] == eps["V00", e["seed"]]["targets"]["3"]["step"]
        and not e["targets"]["3"]["censored"]
        for e in endpoints
    )
    lines.extend(
        [
            (
                "No target-speed gain in optimizer steps or tokens is observed at the 100-step evaluation resolution. Target wall times are reported separately."
                if same_target
                else "Target crossing steps differ or are censored; inspect all retained seed pairs before a speed claim."
            ),
            "",
            (
                "No eligible discovery nominee remains; close this bounded search without C/D."
                if not any(nominations["raw"].values())
                else "Eligible discovery nominees require the separate post-B review and resource decision before independent confirmation. Unresolved numerical qualification remains a blocker."
            ),
            "",
        ]
    )
    lines.extend(
        [
            f"CE<=3 target confirmation steps across the cohort: {sorted(target_steps)}. Target times and censored results are retained in STAGE-B-RESULTS.json. Later loss alone does not establish faster convergence.",
            "",
            f"Raw C nominations: {json.dumps(nominations['raw'])}. Numerically qualified nominations: {json.dumps(nominations['qualified'])}. These are eligible/ineligible discovery decisions, not confirmed winners. The candidate manifest is preparation only.",
            "",
            "STAGE-B-PAIRED.csv retains every pair against M and original M-O. STAGE-B-ORDER-CONTROLS.csv retains direct shuffled/constant pairs. STAGE-B-CURVES.csv holds complete A-to-B curves; STAGE-B-MATCHED-TIME.csv uses the last supported scheduled observation without interpolation or extrapolation. Stitched timing excludes process startup/load and queue/CPU waits; queue reservation cost is separate.",
            "",
            "All nonselected Stage A rows remain in the original report without rerunning them. No eight-seed confirmation, other depth, secondary quantizer, control rounding, packed inference benchmark or new subliminal-transfer experiment ran.",
            "",
        ]
    )
    (r / "STAGE-B-RESULTS.md").write_text("\n".join(lines) + "\n")
    plots(r, curves)


def plots(r, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = r / "stage-b-plots"
    out.mkdir(exist_ok=True)
    for name, xkey, label in (
        ("loss-vs-tokens", "tokens", "Predicted tokens"),
        ("loss-vs-time", "end_to_end_seconds", "Stitched runner seconds"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for variant in ROWS:
            points = [c for c in curves if c["variant"] == variant]
            groups = {
                s: [c for c in points if c["step"] == s]
                for s in sorted({c["step"] for c in points})
            }
            ax.plot(
                [st.mean(c[xkey] for c in g) for g in groups.values()],
                [st.mean(c["ce"] for c in g) for g in groups.values()],
                label=variant,
            )
        ax.set(
            xlabel=label,
            ylabel="Discovery cross-entropy",
            title="Three-seed means; exact A-to-B continuations",
        )
        ax.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    analyze(p.parse_args().root)
