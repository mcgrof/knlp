# SPDX-License-Identifier: MIT
"""Candidate observations cannot tune the primary serving comparison."""

import collections

import pytest

from tools.kv.decision.production_plan import PILOT_RULES, orders, stress_load, trace
from tools.kv.decision.serving import validate


@pytest.mark.parametrize(
    "arms",
    [
        ["native_fi", "AF", "native_fa3"],
        ["native_fi", "AF", "native_fa3", "SF"],
    ],
)
def test_ten_arm_orders_balance_positions_and_include_every_control(arms):
    result = orders(arms)
    assert len(result) == 10 and result == orders(arms)
    assert all(len(row) == len(arms) and set(row) == set(arms) for row in result)
    for position in range(len(arms)):
        counts = collections.Counter(row[position] for row in result)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_traces_preserve_prompt_ids_and_are_independent_between_blocks():
    data = {
        "samples": {
            "development": [
                {
                    "id": f"document-{i}",
                    "kind": "nll",
                    "length": 1024,
                    "token_ids": [i] * 1153,
                }
                for i in range(16)
            ]
        }
    }
    first = trace(data, 1024, 2.9, 10, "block-0")
    assert first == trace(data, 1024, 2.9, 10, "block-0")
    assert first != trace(data, 1024, 2.9, 11, "block-1")
    validate(first)
    offsets = [r["offset_ns"] for r in first["requests"]]
    assert offsets == sorted(offsets) and max(offsets) < 120e9
    assert len(set(r["id"] for r in first["requests"])) == len(offsets)
    assert all(len(r["prompt_token_ids"]) == 1024 for r in first["requests"])


def pilots():
    return [
        {
            "arm": "native_fa3",
            "rate": rate,
            "harness_passed": True,
            "resource_passed": True,
            "slo_fraction": 0.99,
        }
        for rate in PILOT_RULES["load_selection_rates"]
    ]


def test_primary_load_uses_only_fixed_native_cells_and_tie_rule():
    rows = pilots()
    for row in rows:
        if row["rate"] == 2:
            row["slo_fraction"] = 0.75
        if row["rate"] == 3.5:
            row["slo_fraction"] = 0.85
    assert stress_load(rows)["rate"] == 2
    next(r for r in rows if r["rate"] == 2)["harness_passed"] = False
    assert stress_load(rows)["rate"] == 3.5


@pytest.mark.parametrize("defect", ["missing", "candidate", "duplicate"])
def test_partial_or_candidate_tuned_load_selection_is_rejected(defect):
    rows = pilots()
    if defect == "missing":
        rows.pop()
    elif defect == "candidate":
        rows[0]["arm"] = "AF"
    else:
        rows.append(dict(rows[0]))
    with pytest.raises(ValueError):
        stress_load(rows)


def test_uninformative_bounded_pilot_reports_a_limit_without_expansion():
    assert stress_load(pilots())["rate"] is None
