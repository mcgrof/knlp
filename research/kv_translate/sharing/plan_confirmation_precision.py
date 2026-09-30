#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Rare-event-safe precision check for grouped 16K confirmation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(events, groups, alpha):
    events = np.asarray(events)
    return np.where(
        events == groups,
        1.0,
        beta.ppf(1 - alpha, events + 1, groups - events),
    )


def cohort(name, documents, code, alpha_domain):
    document_zero = float(clopper_pearson_upper(0, documents, alpha_domain))
    code_zero = float(clopper_pearson_upper(0, code, alpha_domain))
    return {
        "name": name,
        "groups": {"document": documents, "code": code},
        "zero_adverse_group_upper": {
            "document": document_zero,
            "code": code_zero,
            "equal_domain_weighted": 0.5 * (document_zero + code_zero),
        },
    }


def simulation(documents, code, alpha_domain, *, draws, seed):
    rng = np.random.default_rng(seed)
    scenarios = [
        (0.0, 0.0),
        (0.0, 0.01),
        (0.01, 0.01),
        (0.0, 0.03),
        (0.01, 0.03),
        (0.03, 0.03),
        (0.0, 0.0625),
        (0.05, 0.05),
        (0.10, 0.10),
    ]
    rows = []
    for document_rate, code_rate in scenarios:
        document_events = rng.binomial(documents, document_rate, draws)
        code_events = rng.binomial(code, code_rate, draws)
        upper = 0.5 * (
            clopper_pearson_upper(document_events, documents, alpha_domain)
            + clopper_pearson_upper(code_events, code, alpha_domain)
        )
        worst_case_mean = 0.5 * (document_rate + code_rate)
        rows.append(
            {
                "document_adverse_group_probability": document_rate,
                "code_adverse_group_probability": code_rate,
                "worst_case_weighted_mean_deficit": worst_case_mean,
                "coverage": float(np.mean(upper >= worst_case_mean)),
                "correctness_gate_pass_probability": float(np.mean(upper <= 0.03)),
                "health_gate_pass_probability": float(np.mean(upper <= 0.02)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    documents = inventory["inventory"]["document_groups_16k"]
    code = inventory["inventory"]["code_groups_16k"]
    alpha_family = 0.05
    alpha_domain = alpha_family / 2
    maximum = cohort("maximum_precision", documents, code, alpha_domain)
    equal = cohort("equal_count", code, code, alpha_domain)
    best = maximum["zero_adverse_group_upper"]["equal_domain_weighted"]
    receipt = {
        "schema": "kv_native_16k_precision_v1",
        "status": "DEVELOPMENT_ONLY_DATA_LIMITED",
        "estimands": {
            "correctness": "Within each source group, mean(BF16_correct - K16V8_correct) over the two fixed paired questions; mean within domain; document/code domain means weighted 0.5 each.",
            "health": "Within each source group, K16V8_health_event - BF16_health_event using the frozen any-query context health event; mean within domain; domains weighted 0.5 each.",
        },
        "method": {
            "name": "adverse-group domination with exact Clopper-Pearson upper bounds",
            "construction": "Each bounded paired group deficit/excess X is at most I[X>0]. Bound each domain's adverse-group probability at one-sided alpha=0.025, then average the two bounds. Bonferroni gives at least 95% simultaneous coverage for the equally weighted mean.",
            "rare_event_safe": True,
            "zero_events_has_zero_width": False,
            "within_group_query_correlation": "absorbed by treating the source group, not either question, as the Bernoulli unit",
            "limitation": "Conservative: negative differences and sub-unit harm magnitudes cannot tighten the bound. This method was fixed without confirmation outcomes.",
            "family_alpha": alpha_family,
            "per_domain_alpha": alpha_domain,
        },
        "gates": {
            "one_sided_95_correctness_upper_max": 0.03,
            "one_sided_95_health_excess_upper_max": 0.02,
            "observed_per_domain_deficit_max": 0.05,
        },
        "cohorts": [maximum, equal],
        "best_case_weighted_upper_with_zero_adverse_groups": best,
        "best_case_correctness_gate_possible": best <= 0.03,
        "best_case_health_gate_possible": best <= 0.02,
        "simulation": {
            "draws_per_scenario": args.draws,
            "seed": args.seed,
            "cohort": maximum["groups"],
            "assumption_for_coverage_stress": "Every adverse group has unit harm and no group has a gain, the worst case for the domination step.",
            "scenarios": simulation(
                documents,
                code,
                alpha_domain,
                draws=args.draws,
                seed=args.seed,
            ),
        },
        "decision": {
            "A3_READY": False,
            "reason": "Even all 385 unused document groups and all 33 unused 16K-eligible code repositories cannot meet either confidence margin when zero adverse groups are observed.",
            "outcome_accessed": False,
            "gpu_allocation_admitted": False,
            "next": "Retain the exactly-16K point result as development-only. A4 may address only the independently useful BF16 streaming adapter after A1/A2; K16/V8 confirmation is not admitted.",
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
