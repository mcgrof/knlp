# SPDX-License-Identifier: MIT
"""Pilot-only selection and one frozen all-control confirmation family."""

import copy

import pytest

from tools.kv.decision import production_serving_plan as planner
from tools.kv.decision.production_plan import PILOT_RULES
from tools.kv.decision.production_serving_analysis import inference


@pytest.mark.parametrize("symmetric", [False, True])
def test_confirmation_replays_exact_cells_across_every_qualified_arm(
    monkeypatch, symmetric
):
    written = {}

    def emit(output, name, value, *args):
        written[name] = copy.deepcopy(value)
        return {
            "name": name,
            "plan_file": name + ".json",
            "plan_sha256": name,
            "arm": value["arm"],
        }

    def trace(data, length, rate, seed, label, **kwargs):
        return {
            "input_tokens": length,
            "nominal_rate": rate,
            "trace_seed": seed,
            "label": label,
        }

    monkeypatch.setattr(planner, "emit", emit)
    monkeypatch.setattr(planner, "trace", trace)
    slo = {"ttft_ms": 100, "tpot_ms": 10, "eos_ms": 2000}
    previous = {"slo_selection": {"selected": {"slo": slo}}}
    native_plan = {
        "cells": [
            {"trace": {"nominal_rate": rate}, "slo": slo}
            for rate in PILOT_RULES["load_selection_rates"]
        ]
    }
    native_audit = {
        "cells": [
            {
                "summary": {
                    "slo_fraction_of_all_offered": 0.8 if rate == 3.2 else 0.99
                },
                "harness_screen_passed": True,
            }
            for rate in PILOT_RULES["load_selection_rates"]
        ]
    }
    quality = {"serving_eligible_candidates": ["AF", "SD"] if symmetric else ["AF"]}
    result = planner.confirmation(
        None,
        previous,
        native_plan,
        native_audit,
        {"source_sha": "source"},
        "protocol",
        quality,
        "quality",
        {},
    )
    assert result["primary_stress_load"]["rate"] == 3.2
    assert result["rates_8k"] == [1.0, 2.9, 3.2, 4.0]
    assert len(written) == (40 if symmetric else 30)
    assert ("SD" in result["inference"]["controls"]) == symmetric
    seeds = set()
    for block in range(10):
        values = [
            value
            for name, value in written.items()
            if name.startswith(f"block-{block:02d}-")
        ]
        assert all(v["cells"] == values[0]["cells"] for v in values)
        for cell in values[0]["cells"]:
            seed = cell["trace"]["trace_seed"]
            assert seed not in seeds
            seeds.add(seed)
            assert cell["slo"] == slo


def block_cells(arms):
    return [
        {
            "input_tokens": 8192,
            "rate": 2.9,
            "arm": arm,
            "block": block,
            "harness_screen_passed": True,
            "summary": {"goodput_requests_per_second": 1.1 if arm == "AF" else 1.0},
        }
        for block in range(10)
        for arm in arms
    ]


def test_practical_claim_requires_both_native_controls_and_symmetric_when_qualified():
    arms = ["AF", "native_fi", "native_fa3", "SD"]
    rows = block_cells(arms)
    assert inference(rows, arms, 2.9)["practical_improvement_established"]
    for row in rows:
        if row["arm"] == "SD":
            row["summary"]["goodput_requests_per_second"] = 1.2
    result = inference(rows, arms, 2.9)
    assert result["paired_inference"]["family_size"] == 3
    assert not result["practical_improvement_established"]


def test_harness_saturation_prevents_claim_without_erasing_scores():
    arms = ["AF", "native_fi", "native_fa3"]
    rows = block_cells(arms)
    rows[0]["harness_screen_passed"] = False
    result = inference(rows, arms, 2.9)
    assert result["paired_inference"]["improvement_vs_all_controls"]
    assert not result["practical_improvement_established"]


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_incomplete_or_repeated_primary_block_cannot_enter_inference(defect):
    arms = ["AF", "native_fi", "native_fa3"]
    rows = block_cells(arms)
    if defect == "missing":
        rows.pop()
    else:
        rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError):
        inference(rows, arms, 2.9)


def test_no_informative_pilot_does_not_select_a_better_looking_confirmation_cell():
    arms = ["AF", "native_fi", "native_fa3"]
    result = inference(block_cells(arms), arms, None)
    assert not result["tested"] and not result["practical_improvement_established"]
