from copy import deepcopy

import pytest

from research.kv_translate.published.select_kv_lingo_checkpoint_pair import select


def comparison(*, domain_deficit=0.01, overall_deficit=0.01):
    gates = {
        "overall_deficit_at_most_3pp": overall_deficit <= 0.03,
        "turn6_10_deficit_at_most_5pp": True,
        "each_domain_deficit_at_most_5pp": domain_deficit <= 0.05,
        "unhealthy_excess_at_most_2pp": True,
    }
    return {
        "equal_domain_overall_deficit": overall_deficit,
        "turn6_10_deficit": 0.01,
        "equal_domain_turn6_10_deficit": 0.01,
        "unhealthy_rate_excess": 0.0,
        "by_domain": {
            "only": {
                "treatment_f1": 0.5,
                "reference_f1": 0.5 + domain_deficit,
                "reference_minus_treatment_f1": domain_deficit,
                "health_excess": 0.0,
                "rows": 10,
            }
        },
        "diagnostic_slices": {"by_turn": {"1": {"rows": 1}}},
        "equal_domain_bootstrap": {"deficit_interval_95": [-0.01, 0.02]},
        "gates": gates,
        "passed": all(gates.values()),
    }


def cell(*, domain_deficit=0.01, overall_deficit=0.01):
    starts = {}
    for start in ("4B", "8B"):
        starts[start] = {
            reference: comparison(
                domain_deficit=domain_deficit, overall_deficit=overall_deficit
            )
            for reference in (
                "same_transcript_native",
                "alternating_native_trajectory",
            )
        }
        starts[start]["passed"] = all(
            starts[start][reference]["passed"]
            for reference in (
                "same_transcript_native",
                "alternating_native_trajectory",
            )
        )
    return {
        "starting_models": starts,
        "passed": all(value["passed"] for value in starts.values()),
    }


def test_closes_when_neither_mixed_cell_passes():
    result = select(
        {
            "forward_1000_reverse_1000": cell(domain_deficit=0.06),
            "forward_5000_reverse_5000": cell(domain_deficit=0.06),
            "forward_5000_reverse_1000": cell(domain_deficit=0.06),
            "forward_1000_reverse_5000": cell(overall_deficit=0.04),
        }
    )
    assert result["selected_checkpoint_pair"] is None
    assert result["verdict"] == "CLOSE_CHECKPOINT_EXPLORATION"


def test_selects_eligible_cell_then_applies_frozen_tie_breaks():
    names = (
        "forward_1000_reverse_1000",
        "forward_5000_reverse_5000",
        "forward_5000_reverse_1000",
        "forward_1000_reverse_5000",
    )
    summaries = {name: cell() for name in names}
    summaries["forward_1000_reverse_5000"] = cell(domain_deficit=0.02)
    result = select(summaries)
    assert result["selected_checkpoint_pair"] == "forward_5000_reverse_1000"

    tied = {name: cell() for name in names}
    assert select(tied)["selected_checkpoint_pair"] == "forward_5000_reverse_1000"


def test_rejects_inconsistent_stored_gate_decision():
    names = (
        "forward_1000_reverse_1000",
        "forward_5000_reverse_5000",
        "forward_5000_reverse_1000",
        "forward_1000_reverse_5000",
    )
    summaries = {name: cell() for name in names}
    broken = deepcopy(summaries["forward_5000_reverse_1000"])
    broken["starting_models"]["4B"]["same_transcript_native"]["passed"] = False
    summaries["forward_5000_reverse_1000"] = broken
    with pytest.raises(ValueError, match="stored gate decision"):
        select(summaries)
