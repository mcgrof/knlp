#!/usr/bin/env python3
"""Apply the frozen KV-Lingo saved-checkpoint pair selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kv_lingo_data import write_json

CHECKPOINT_PAIRS = (
    "forward_1000_reverse_1000",
    "forward_5000_reverse_5000",
    "forward_5000_reverse_1000",
    "forward_1000_reverse_5000",
)
MIXED_CHECKPOINT_PAIRS = (
    "forward_5000_reverse_1000",
    "forward_1000_reverse_5000",
)
STARTING_MODELS = ("4B", "8B")
REFERENCES = ("same_transcript_native", "alternating_native_trajectory")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def comparison_passed(comparison: dict) -> bool:
    return bool(
        comparison["equal_domain_overall_deficit"] <= 0.03
        and comparison["turn6_10_deficit"] <= 0.05
        and all(
            domain["reference_minus_treatment_f1"] <= 0.05
            for domain in comparison["by_domain"].values()
        )
        and comparison["unhealthy_rate_excess"] <= 0.02
    )


def summarize_cell(summary: dict) -> dict:
    comparisons = {}
    for start in STARTING_MODELS:
        for reference in REFERENCES:
            comparison = summary["starting_models"][start][reference]
            passed = comparison_passed(comparison)
            if passed != comparison["passed"] or passed != all(
                comparison["gates"].values()
            ):
                raise ValueError(
                    f"stored gate decision disagrees with recomputation: {start}/{reference}"
                )
            equal_domain_treatment = sum(
                value["treatment_f1"] for value in comparison["by_domain"].values()
            ) / len(comparison["by_domain"])
            equal_domain_reference = sum(
                value["reference_f1"] for value in comparison["by_domain"].values()
            ) / len(comparison["by_domain"])
            comparisons[f"{start}/{reference}"] = {
                "passed": passed,
                "equal_domain_treatment_f1": equal_domain_treatment,
                "equal_domain_reference_f1": equal_domain_reference,
                "equal_domain_overall_deficit": comparison[
                    "equal_domain_overall_deficit"
                ],
                "pooled_turn6_10_deficit": comparison["turn6_10_deficit"],
                "equal_domain_turn6_10_deficit_diagnostic": comparison[
                    "equal_domain_turn6_10_deficit"
                ],
                "unhealthy_rate_excess": comparison["unhealthy_rate_excess"],
                "by_domain": comparison["by_domain"],
                "by_turn": comparison["diagnostic_slices"]["by_turn"],
                "paired_conversation_interval_95": comparison["equal_domain_bootstrap"][
                    "deficit_interval_95"
                ],
            }
    eligible = all(value["passed"] for value in comparisons.values())
    stored_passed = bool(summary["passed"])
    if eligible != stored_passed:
        raise ValueError("stored cell decision disagrees with recomputation")
    return {
        "eligible": eligible,
        "worst_domain_deficit": max(
            domain["reference_minus_treatment_f1"]
            for comparison in comparisons.values()
            for domain in comparison["by_domain"].values()
        ),
        "worst_equal_domain_overall_deficit": max(
            comparison["equal_domain_overall_deficit"]
            for comparison in comparisons.values()
        ),
        "comparisons": comparisons,
    }


def select(summaries: dict[str, dict]) -> dict:
    if set(summaries) != set(CHECKPOINT_PAIRS):
        raise ValueError(
            f"expected exactly {CHECKPOINT_PAIRS}, got {tuple(sorted(summaries))}"
        )
    pairs = {name: summarize_cell(summaries[name]) for name in CHECKPOINT_PAIRS}
    eligible = [name for name in MIXED_CHECKPOINT_PAIRS if pairs[name]["eligible"]]
    if not eligible:
        selected = None
        verdict = "CLOSE_CHECKPOINT_EXPLORATION"
    elif len(eligible) == 1:
        selected = eligible[0]
        verdict = "FREEZE_SELECTED_PAIR_BEFORE_CONFIRMATION"
    else:
        selected = min(
            eligible,
            key=lambda name: (
                pairs[name]["worst_domain_deficit"],
                pairs[name]["worst_equal_domain_overall_deficit"],
                0 if name == "forward_5000_reverse_1000" else 1,
            ),
        )
        verdict = "FREEZE_SELECTED_PAIR_BEFORE_CONFIRMATION"
    return {
        "schema": "kv_lingo_saved_checkpoint_pair_selection_v1",
        "checkpoint_pairs": pairs,
        "eligible_mixed_checkpoint_pairs": eligible,
        "selected_checkpoint_pair": selected,
        "verdict": verdict,
        "selection_rule": {
            "eligibility": "all gates in all four starting-owner/reference comparisons",
            "primary_tie_break": "smallest worst domain deficit",
            "secondary_tie_break": "smallest worst equal-domain overall deficit",
            "final_tie_break": "forward_5000_reverse_1000",
        },
        "Generated-by": "OpenAI Codex",
    }


def parse_cell(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected CHECKPOINT_PAIR=SUMMARY.json"
        ) from error
    if name not in CHECKPOINT_PAIRS:
        raise argparse.ArgumentTypeError(f"unknown checkpoint pair {name!r}")
    return name, Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", type=parse_cell, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = dict(args.cell)
    if len(paths) != len(args.cell):
        parser.error("duplicate --cell name")
    result = select({name: read_json(path) for name, path in paths.items()})
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
