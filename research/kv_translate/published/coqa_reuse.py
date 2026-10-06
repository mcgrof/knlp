#!/usr/bin/env python3
"""CPU contracts and summaries for reusable CoQA passage caches."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re

import numpy as np

from . import coqa


def normalize_passage(text: str) -> str:
    """Normalize only whitespace and case for replication exclusions."""
    return " ".join(text.casefold().split())


def passage_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_replication_conversations(
    dataset: dict,
    excluded_ids: set[str],
    *,
    per_domain: int = 20,
) -> dict:
    """Select the next passage-independent cohort in frozen hash order."""
    excluded_passages = {
        normalize_passage(story["story"])
        for story in dataset["data"]
        if story["id"] in excluded_ids
    }
    selected_passages = set(excluded_passages)
    by_domain = defaultdict(list)
    for story in dataset["data"]:
        by_domain[story["source"]].append(story)

    selected = []
    exclusions = []
    eligible_inventory = {}
    for domain in coqa.DOMAINS:
        ordered = sorted(
            by_domain[domain],
            key=lambda row: (coqa.selection_digest(row["id"]), row["id"]),
        )
        domain_selected = []
        eligible_unique = 0
        inventory_seen = set(excluded_passages)
        for story in ordered:
            normalized = normalize_passage(story["story"])
            if story["id"] in excluded_ids:
                reason = "previous_conversation_id"
            elif normalized in excluded_passages:
                reason = "passage_matches_previous_cohort"
            elif normalized in inventory_seen:
                reason = "duplicate_candidate_passage"
            else:
                reason = None
                inventory_seen.add(normalized)
                eligible_unique += 1

            if reason is not None:
                exclusions.append(
                    {
                        "conversation_id": story["id"],
                        "domain": domain,
                        "reason": reason,
                        "raw_passage_sha256": passage_sha256(story["story"]),
                        "normalized_passage_sha256": passage_sha256(normalized),
                    }
                )
                continue
            if normalized in selected_passages:
                exclusions.append(
                    {
                        "conversation_id": story["id"],
                        "domain": domain,
                        "reason": "passage_selected_in_another_domain",
                        "raw_passage_sha256": passage_sha256(story["story"]),
                        "normalized_passage_sha256": passage_sha256(normalized),
                    }
                )
                continue
            if len(domain_selected) < per_domain:
                domain_selected.append(story)
                selected_passages.add(normalized)
        eligible_inventory[domain] = eligible_unique
        selected.extend(domain_selected)

    selected_by_domain = {
        domain: [row for row in selected if row["source"] == domain]
        for domain in coqa.DOMAINS
    }
    sufficient = all(len(selected_by_domain[d]) == per_domain for d in coqa.DOMAINS)
    return {
        "selected": selected,
        "selected_by_domain": selected_by_domain,
        "exclusions": exclusions,
        "eligible_unique_passages_by_domain": eligible_inventory,
        "sufficient": sufficient,
        "requested_per_domain": per_domain,
    }


def passage_cut(prompt: str, offsets: list[tuple[int, int]]) -> dict:
    """Return the safe token cut immediately before the first question."""
    matches = list(re.finditer(r"(?m)^Question:", prompt))
    if not matches:
        raise ValueError("prompt has no Question marker")
    boundary = matches[0].start()
    cut = 0
    crossing = None
    for index, (start, end) in enumerate(offsets):
        if start < boundary < end:
            crossing = index
        if end <= boundary and end > start:
            cut = index + 1
    if cut < 1 or cut >= len(offsets):
        raise ValueError(f"invalid passage cut {cut} for {len(offsets)} tokens")
    if any(start < boundary for start, _end in offsets[cut + 1 :]):
        raise ValueError("token offsets are not monotonic around passage boundary")
    return {
        "early_cut": cut,
        "first_question_character_offset": boundary,
        "token_crossing_boundary": crossing,
        "crossing_token_left_in_suffix": crossing is None or crossing >= cut,
    }


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _validate_rows(rows: list[dict], arms: tuple[str, ...]):
    if not rows:
        raise ValueError("no passage-reuse rows")
    if {row["domain"] for row in rows} != set(coqa.DOMAINS):
        raise ValueError("passage-reuse rows must cover every frozen CoQA domain")
    for row in rows:
        for arm in arms:
            if f"{arm}_f1" not in row or f"{arm}_health" not in row:
                raise ValueError(f"row {row.get('row_id')} lacks {arm}")


def _arm_summary(rows: list[dict], arm: str) -> dict:
    def cell(cell_rows):
        return {
            "questions": len(cell_rows),
            "f1": _mean(row[f"{arm}_f1"] for row in cell_rows),
            "unhealthy": sum(bool(row[f"{arm}_health"]) for row in cell_rows),
        }

    return {
        "pooled": cell(rows),
        "by_domain": {
            domain: cell([row for row in rows if row["domain"] == domain])
            for domain in coqa.DOMAINS
        },
        "by_turn": {
            str(turn): cell([row for row in rows if row["turn"] == turn])
            for turn in sorted({row["turn"] for row in rows})
        },
    }


def _contrast(rows: list[dict], reference: str, treatment: str) -> dict:
    def cell(cell_rows):
        return {
            "questions": len(cell_rows),
            "reference_minus_treatment_f1": _mean(
                row[f"{reference}_f1"] - row[f"{treatment}_f1"] for row in cell_rows
            ),
            "treatment_unhealthy_excess": _mean(
                bool(row[f"{treatment}_health"]) - bool(row[f"{reference}_health"])
                for row in cell_rows
            ),
        }

    by_domain = {
        domain: cell([row for row in rows if row["domain"] == domain])
        for domain in coqa.DOMAINS
    }
    return {
        "pooled": cell(rows),
        "equal_domain_reference_minus_treatment_f1": _mean(
            by_domain[domain]["reference_minus_treatment_f1"] for domain in coqa.DOMAINS
        ),
        "by_domain": by_domain,
        "by_turn": {
            str(turn): cell([row for row in rows if row["turn"] == turn])
            for turn in sorted({row["turn"] for row in rows})
        },
    }


def _bootstrap(rows: list[dict], *, replicates: int) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["domain"]][row["conversation_id"]].append(row)
    rng = np.random.default_rng(coqa.BOOTSTRAP_SEED)
    values = {"early_loss": [], "late_loss": [], "loss_difference": []}
    has_late = all("native_late_f1" in row for row in rows)
    for _ in range(replicates):
        domain_early = []
        domain_late = []
        for domain in coqa.DOMAINS:
            ids = sorted(grouped[domain])
            picked = rng.choice(ids, size=len(ids), replace=True)
            sampled = [row for cid in picked for row in grouped[domain][str(cid)]]
            domain_early.append(
                _mean(
                    row["native_early_f1"] - row["translated_early_f1"]
                    for row in sampled
                )
            )
            if has_late:
                domain_late.append(
                    _mean(
                        row["native_late_f1"] - row["translated_late_f1"]
                        for row in sampled
                    )
                )
        early = _mean(domain_early)
        values["early_loss"].append(early)
        if has_late:
            late = _mean(domain_late)
            values["late_loss"].append(late)
            values["loss_difference"].append(early - late)

    intervals = {}
    for name, samples in values.items():
        if samples:
            intervals[f"{name}_95_interval"] = [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ]
    return {
        "seed": coqa.BOOTSTRAP_SEED,
        "replicates": replicates,
        "unit": "complete conversations resampled within domain",
        "intervals": intervals,
        "interpretation": "descriptive only; not population noninferiority",
    }


def _transfer_screen(contrast: dict, *, require_turn_one: bool) -> dict:
    requirements = {
        "equal_domain_deficit_at_most_0.03": contrast[
            "equal_domain_reference_minus_treatment_f1"
        ]
        <= 0.03,
        "each_domain_deficit_at_most_0.05": all(
            contrast["by_domain"][domain]["reference_minus_treatment_f1"] <= 0.05
            for domain in coqa.DOMAINS
        ),
        "pooled_health_excess_at_most_0.02": contrast["pooled"][
            "treatment_unhealthy_excess"
        ]
        <= 0.02,
    }
    if require_turn_one:
        requirements["turn_1_deficit_at_most_0.03"] = (
            contrast["by_turn"]["1"]["reference_minus_treatment_f1"] <= 0.03
        )
    return {"requirements": requirements, "passed": all(requirements.values())}


def summarize_r1(rows: list[dict], *, replicates: int = coqa.BOOTSTRAP_REPLICATES):
    arms = (
        "native_early",
        "translated_early",
        "native_late",
        "translated_late",
    )
    _validate_rows(rows, arms)
    arm_summaries = {arm: _arm_summary(rows, arm) for arm in arms}
    native_control = _contrast(rows, "native_early", "native_late")
    early = _contrast(rows, "native_early", "translated_early")
    late = _contrast(rows, "native_late", "translated_late")
    control_requirements = {
        "absolute_equal_domain_f1_difference_at_most_0.01": abs(
            native_control["equal_domain_reference_minus_treatment_f1"]
        )
        <= 0.01,
        "each_domain_absolute_f1_difference_at_most_0.02": all(
            abs(native_control["by_domain"][domain]["reference_minus_treatment_f1"])
            <= 0.02
            for domain in coqa.DOMAINS
        ),
        "absolute_pooled_health_difference_at_most_0.01": abs(
            native_control["pooled"]["treatment_unhealthy_excess"]
        )
        <= 0.01,
    }
    screens = {
        "native_cut_stability": {
            "requirements": control_requirements,
            "passed": all(control_requirements.values()),
        },
        "late_positive_control": _transfer_screen(late, require_turn_one=False),
        "early_reuse": _transfer_screen(early, require_turn_one=True),
    }
    if not screens["native_cut_stability"]["passed"]:
        state = "CONTROL_UNSTABLE"
    elif not screens["late_positive_control"]["passed"]:
        state = "POSITIVE_CONTROL_NOT_REPRODUCED"
    elif not screens["early_reuse"]["passed"]:
        state = "CLOSE_EARLY_REUSE_PROMOTION"
    else:
        state = "R2"
    early_gap = early["equal_domain_reference_minus_treatment_f1"]
    late_gap = late["equal_domain_reference_minus_treatment_f1"]
    return {
        "schema": "coqa_passage_reuse_r1_result_v1",
        "rows": len(rows),
        "arms": arm_summaries,
        "contrasts": {
            "native_cut_stability": native_control,
            "early_transfer_loss": early,
            "late_transfer_loss": late,
            "equal_domain_early_minus_late_transfer_loss": early_gap - late_gap,
        },
        "bootstrap": _bootstrap(rows, replicates=replicates),
        "screens": screens,
        "next_state": state,
        "meaning": "prospective development point screens only",
    }


def summarize_r2(rows: list[dict], *, replicates: int = coqa.BOOTSTRAP_REPLICATES):
    arms = ("native_early", "translated_early")
    _validate_rows(rows, arms)
    early = _contrast(rows, "native_early", "translated_early")
    screen = _transfer_screen(early, require_turn_one=True)
    return {
        "schema": "coqa_passage_reuse_r2_result_v1",
        "rows": len(rows),
        "arms": {arm: _arm_summary(rows, arm) for arm in arms},
        "early_transfer_loss": early,
        "bootstrap": _bootstrap(rows, replicates=replicates),
        "screen": screen,
        "next_state": (
            "INDEPENDENT_REPLICATION_POINT_SCREEN_PASS"
            if screen["passed"]
            else "DEVELOPMENT_ONLY_PROMOTION_CLOSED"
        ),
        "meaning": "independent-cohort point screen; uncertainty is descriptive",
    }
