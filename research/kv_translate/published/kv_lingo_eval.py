#!/usr/bin/env python3
"""CPU summaries and frozen quality gates for retained-span CoQA runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from . import coqa
from .kv_lingo_data import write_json


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unhealthy(row: dict, prefix: str) -> bool:
    return bool(row[f"{prefix}_health"])


def aggregate(rows: list[dict], treatment: str, reference: str) -> dict:
    by_domain = {}
    for domain in coqa.DOMAINS:
        selected = [row for row in rows if row["domain"] == domain]
        if not selected:
            raise ValueError(f"no rows for frozen domain {domain}")
        treatment_f1 = np.mean([row[f"{treatment}_f1"] for row in selected])
        reference_f1 = np.mean([row[f"{reference}_f1"] for row in selected])
        treatment_health = np.mean([unhealthy(row, treatment) for row in selected])
        reference_health = np.mean([unhealthy(row, reference) for row in selected])
        by_domain[domain] = {
            "rows": len(selected),
            "treatment_f1": float(treatment_f1),
            "reference_f1": float(reference_f1),
            "reference_minus_treatment_f1": float(reference_f1 - treatment_f1),
            "health_excess": float(treatment_health - reference_health),
        }
    late = [row for row in rows if row["turn"] >= 6]
    late_by_domain = {
        domain: [row for row in late if row["domain"] == domain]
        for domain in coqa.DOMAINS
    }
    overall_deficit = np.mean(
        [by_domain[domain]["reference_minus_treatment_f1"] for domain in coqa.DOMAINS]
    )
    late_deficit = np.mean(
        [row[f"{reference}_f1"] - row[f"{treatment}_f1"] for row in late]
    )
    equal_domain_late_deficit = np.mean(
        [
            np.mean(
                [
                    row[f"{reference}_f1"] - row[f"{treatment}_f1"]
                    for row in late_by_domain[domain]
                ]
            )
            for domain in coqa.DOMAINS
        ]
    )
    health_excess = np.mean(
        [unhealthy(row, treatment) - unhealthy(row, reference) for row in rows]
    )
    gates = {
        "overall_deficit_at_most_3pp": bool(overall_deficit <= 0.03),
        "turn6_10_deficit_at_most_5pp": bool(late_deficit <= 0.05),
        "each_domain_deficit_at_most_5pp": bool(
            all(
                value["reference_minus_treatment_f1"] <= 0.05
                for value in by_domain.values()
            )
        ),
        "unhealthy_excess_at_most_2pp": bool(health_excess <= 0.02),
    }
    return {
        "equal_domain_overall_deficit": float(overall_deficit),
        "turn6_10_deficit": float(late_deficit),
        "pooled_turn6_10_deficit": float(late_deficit),
        "equal_domain_turn6_10_deficit": float(equal_domain_late_deficit),
        "unhealthy_rate_excess": float(health_excess),
        "by_domain": by_domain,
        "gates": gates,
        "passed": all(gates.values()),
    }


def attach_native_reference(translated: list[dict], native: list[dict]) -> list[dict]:
    index = {
        (row["conversation_id"], row["starting_model"], row["turn"]): row
        for row in native
    }
    result = []
    for row in translated:
        key = (row["conversation_id"], row["starting_model"], row["turn"])
        reference = index[key]
        result.append(
            {
                **row,
                "native_trajectory_f1": reference["native_trajectory_f1"],
                "native_trajectory_health": reference["native_trajectory_health"],
            }
        )
    return result


def cluster_bootstrap(rows: list[dict], treatment: str, reference: str, draws=10_000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)
    ids = sorted(grouped)
    rng = np.random.default_rng(20261002)
    values = []
    for _ in range(draws):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        selected = [
            row for conversation_id in sampled for row in grouped[conversation_id]
        ]
        values.append(
            np.mean(
                [row[f"{reference}_f1"] - row[f"{treatment}_f1"] for row in selected]
            )
        )
    return {
        "draws": draws,
        "seed": 20261002,
        "cluster": "conversation",
        "estimand": "pooled conversation/turn mean deficit",
        "deficit_interval_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def equal_domain_cluster_bootstrap(
    rows: list[dict], treatment: str, reference: str, draws=10_000
):
    """Resample whole conversations within domain for the equal-domain mean."""

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["domain"]][row["conversation_id"]].append(row)
    rng = np.random.default_rng(20261002)
    values = []
    for _ in range(draws):
        domain_means = []
        for domain in coqa.DOMAINS:
            ids = sorted(grouped[domain])
            sampled = rng.choice(ids, size=len(ids), replace=True)
            selected = [
                row
                for conversation_id in sampled
                for row in grouped[domain][conversation_id]
            ]
            domain_means.append(
                np.mean(
                    [
                        row[f"{reference}_f1"] - row[f"{treatment}_f1"]
                        for row in selected
                    ]
                )
            )
        values.append(np.mean(domain_means))
    return {
        "draws": draws,
        "seed": 20261002,
        "cluster": "complete conversations resampled within domain",
        "estimand": "equal-domain mean deficit",
        "deficit_interval_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def diagnostic_slices(rows: list[dict], treatment: str, reference: str) -> dict:
    result = {}
    for field in ("receiver", "turn", "domain"):
        values = {}
        for value in sorted({row[field] for row in rows}, key=str):
            selected = [row for row in rows if row[field] == value]
            values[str(value)] = {
                "rows": len(selected),
                "reference_minus_treatment_f1": float(
                    np.mean(
                        [
                            row[f"{reference}_f1"] - row[f"{treatment}_f1"]
                            for row in selected
                        ]
                    )
                ),
            }
        result[f"by_{field}"] = values
    return result


def summarize(translated: list[dict], native: list[dict], *, bootstrap=True):
    joined = attach_native_reference(translated, native)
    starts = {}
    for start in ("4B", "8B"):
        rows = [row for row in joined if row["starting_model"] == start]
        same = aggregate(rows, "translated", "same_transcript_native")
        trajectory = aggregate(rows, "translated", "native_trajectory")
        if bootstrap:
            same["bootstrap"] = cluster_bootstrap(
                rows, "translated", "same_transcript_native"
            )
            same["equal_domain_bootstrap"] = equal_domain_cluster_bootstrap(
                rows, "translated", "same_transcript_native"
            )
            trajectory["bootstrap"] = cluster_bootstrap(
                rows, "translated", "native_trajectory"
            )
            trajectory["equal_domain_bootstrap"] = equal_domain_cluster_bootstrap(
                rows, "translated", "native_trajectory"
            )
        same["diagnostic_slices"] = diagnostic_slices(
            rows, "translated", "same_transcript_native"
        )
        trajectory["diagnostic_slices"] = diagnostic_slices(
            rows, "translated", "native_trajectory"
        )
        starts[start] = {
            "same_transcript_native": same,
            "alternating_native_trajectory": trajectory,
            "passed": same["passed"] and trajectory["passed"],
        }
    return {
        "schema": "kv_lingo_retained_coqa_summary_v1",
        "starting_models": starts,
        "passed": all(value["passed"] for value in starts.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(read_jsonl(args.raw), read_jsonl(args.native))
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
