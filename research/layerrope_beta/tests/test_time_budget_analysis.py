# SPDX-License-Identifier: MIT
"""Keep paired inference and precision floors conditional on complete data."""

import copy
import pytest
from research.layerrope_beta.time_budget.analysis import analyze
from research.layerrope_beta.time_budget.protocol import freeze


def matrix():
    p = freeze()
    records = []
    for row in p["rows"]:
        for seed in p["final_seeds"]:
            clean = 2.0 if row in ("R", "L") else 2.03

            def case(c):
                # An explicit below-floor failure, and a floor passing all seeds.
                damage = (
                    0.11
                    if c["name"] == "symmetric-K2V2"
                    else 0.02 if c["name"] != "clean" else 0
                )
                return dict(**c, ce=clean + damage)

            primary = dict(
                cache=[case(c) for c in p["cache_cases"]],
                weight_activation_residual=[
                    dict(w, ce=clean + 0.01) for w in p["weight_policies"]
                ],
            )
            supporting = dict(
                cache=[
                    case(c)
                    for c in p["cache_cases"]
                    if c["name"] in p["supporting_cache_names"]
                ]
            )
            records.append(
                dict(
                    row=row,
                    seed=seed,
                    endpoints={"615.0": supporting, "787.0": primary},
                    routing_constraint_pass=True,
                )
            )
    return records


def test_complete_paired_analysis_keeps_failures_and_reports_menu_limited_floor():
    records = matrix()
    result = analyze(records)
    assert len(result["contrasts"]) == 10
    assert result["contrasts"][0]["statement"] == "practical_median_improvement"
    assert result["contrasts"][4]["statement"] == "median_noninferiority"  # ML-M ties
    assert all(f["observed_floor"] == 3072 for f in result["floors"])
    records[0]["routing_constraint_pass"] = False
    assert not analyze(records)[
        "routing_constraint_pass"
    ]  # retain CE, restrict mechanism claim
    with pytest.raises(ValueError, match="All 32"):
        analyze(records[:-1])
    with pytest.raises(ValueError, match="All 32"):
        analyze(records + [records[0]])
    changed = copy.deepcopy(records)
    changed[0]["endpoints"]["787.0"]["cache"].pop()
    with pytest.raises(ValueError, match="Incomplete cache"):
        analyze(changed)
    for r in records:
        for c in r["endpoints"]["787.0"]["cache"]:
            if c["name"] != "clean":
                c["ce"] += 0.2
    assert all(f["observed_floor"] is None for f in analyze(records)["floors"])


def test_analysis_rejects_duplicate_cases_and_nonfinite_weight_results():
    rows = matrix()
    rows[0]["endpoints"]["787.0"]["cache"].append(
        rows[0]["endpoints"]["787.0"]["cache"][0]
    )
    with pytest.raises(ValueError, match="Incomplete cache"):
        analyze(rows)
    rows = matrix()
    rows[0]["endpoints"]["787.0"]["weight_activation_residual"][0]["ce"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite weight"):
        analyze(rows)
