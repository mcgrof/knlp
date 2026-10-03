# SPDX-License-Identifier: MIT
"""Bounded prompt-layout planning with an explicit host-residency budget.

This module plans only among layouts approved by the caller.  It does not
rewrite prompt text, infer semantic equivalence, or claim that a requested
cache action was enforced.  The score is an auditable structural estimate:
prefix tokens that can be reused if the selected prefixes remain resident.
Physical cache events remain the serving backend's responsibility.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from .workflow import canonical_sha256

PREFIX_PLAN_SCHEMA = "knlp.agent_baselines.prefix_plan.v2"
PREFIX_IDENTITY_SCOPE = "global-within-plan"


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a nonempty identifier without whitespace")
    return value


def _normalize_families(families: Iterable[dict]) -> tuple[list[dict], list[str]]:
    normalized = []
    layout_names: set[str] | None = None
    family_ids = set()
    prefix_tokens_by_identity: dict[str, int] = {}
    for family in families:
        if not isinstance(family, dict):
            raise TypeError("Each family must be an object")
        family_id = _identifier(family.get("family_id"), "family_id")
        if family_id in family_ids:
            raise ValueError("family_id values must be unique")
        family_ids.add(family_id)
        candidates = family.get("candidates")
        if not isinstance(candidates, dict) or len(candidates) < 2:
            raise ValueError("Each family needs at least two layout candidates")
        current_names = set(candidates)
        if layout_names is None:
            layout_names = current_names
        elif current_names != layout_names:
            raise ValueError("Every family must expose the same layout names")
        clean_candidates = {}
        request_count = None
        for layout_name, candidate in candidates.items():
            _identifier(layout_name, "layout name")
            if not isinstance(candidate, dict):
                raise TypeError("Each layout candidate must be an object")
            prefixes = candidate.get("prefixes")
            if not isinstance(prefixes, list) or not prefixes:
                raise ValueError("candidate prefixes must be a nonempty list")
            clean_prefixes = [
                _identifier(prefix, "prefix identity") for prefix in prefixes
            ]
            if request_count is None:
                request_count = len(clean_prefixes)
            elif len(clean_prefixes) != request_count:
                raise ValueError("Layout candidates must cover the same requests")
            prefix_tokens = _positive_int(
                candidate.get("prefix_tokens"), "prefix_tokens"
            )
            for prefix in clean_prefixes:
                previous = prefix_tokens_by_identity.setdefault(prefix, prefix_tokens)
                if previous != prefix_tokens:
                    raise ValueError(
                        "A global prefix identity must have one prefix_tokens value"
                    )
            clean_candidates[layout_name] = {
                "prefix_tokens": prefix_tokens,
                "prefixes": clean_prefixes,
            }
        normalized.append(
            {
                "family_id": family_id,
                "requests": request_count,
                "candidates": clean_candidates,
            }
        )
    if not normalized:
        raise ValueError("At least one workload family is required")
    return normalized, sorted(layout_names or ())


def _score_assignment(
    families: list[dict],
    assignment: dict[str, str],
    *,
    host_capacity_bytes: int,
    kv_bytes_per_token: int,
) -> dict:
    selected_candidates = []
    global_counts: dict[str, int] = {}
    prefix_tokens: dict[str, int] = {}
    for family in families:
        family_id = family["family_id"]
        layout = assignment[family_id]
        candidate = family["candidates"][layout]
        selected_candidates.append((family, layout, candidate))
        for prefix in candidate["prefixes"]:
            global_counts[prefix] = global_counts.get(prefix, 0) + 1
            prefix_tokens[prefix] = candidate["prefix_tokens"]

    globally_reusable = {prefix for prefix, count in global_counts.items() if count > 1}
    resident_prefix_tokens = sum(prefix_tokens[prefix] for prefix in globally_reusable)
    reuse_tokens = sum(
        (global_counts[prefix] - 1) * prefix_tokens[prefix]
        for prefix in globally_reusable
    )

    seen: dict[str, int] = {}
    family_scores = []
    for family, layout, candidate in selected_candidates:
        family_id = family["family_id"]
        contributed_reuse = 0
        contributed_prefixes = set()
        for prefix in candidate["prefixes"]:
            if seen.get(prefix, 0):
                contributed_reuse += candidate["prefix_tokens"]
                contributed_prefixes.add(prefix)
            seen[prefix] = seen.get(prefix, 0) + 1
        family_scores.append(
            {
                "family_id": family_id,
                "layout": layout,
                "requests": family["requests"],
                "reusable_prefixes": len(contributed_prefixes),
                "estimated_reuse_tokens": contributed_reuse,
            }
        )
    resident_bytes = resident_prefix_tokens * kv_bytes_per_token
    feasible = resident_bytes <= host_capacity_bytes
    return {
        "assignment": dict(sorted(assignment.items())),
        "assignment_sha256": canonical_sha256(assignment),
        "feasible": feasible,
        "estimated_reuse_tokens": reuse_tokens if feasible else None,
        "estimated_recomputed_tokens_avoided": reuse_tokens if feasible else None,
        "resident_prefix_tokens": resident_prefix_tokens,
        "resident_bytes": resident_bytes,
        "host_capacity_bytes": host_capacity_bytes,
        "global_reusable_prefixes": len(globally_reusable),
        "family_scores": family_scores,
    }


def _best(scores: list[dict], *, required: bool = True) -> dict | None:
    feasible = [score for score in scores if score["feasible"]]
    if not feasible:
        if required:
            raise ValueError("No candidate assignment fits the host-residency budget")
        return None
    return deepcopy(
        min(
            feasible,
            key=lambda score: (
                -score["estimated_reuse_tokens"],
                score["resident_bytes"],
                tuple(score["assignment"].items()),
            ),
        )
    )


def compile_prefix_plan(
    families: Iterable[dict],
    *,
    host_capacity_bytes: int,
    kv_bytes_per_token: int,
    max_assignments: int = 256,
) -> dict:
    """Enumerate fixed and per-family approved layouts under one tier budget."""
    host_capacity_bytes = _positive_int(host_capacity_bytes, "host_capacity_bytes")
    kv_bytes_per_token = _positive_int(kv_bytes_per_token, "kv_bytes_per_token")
    max_assignments = _positive_int(max_assignments, "max_assignments")
    clean, layouts = _normalize_families(families)
    assignment_count = len(layouts) ** len(clean)
    if assignment_count > max_assignments:
        raise ValueError(
            f"Bounded search would examine {assignment_count} assignments, "
            f"above max_assignments={max_assignments}"
        )
    fixed_scores = [
        _score_assignment(
            clean,
            {family["family_id"]: layout for family in clean},
            host_capacity_bytes=host_capacity_bytes,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        for layout in layouts
    ]
    joint_scores = [
        _score_assignment(
            clean,
            {family["family_id"]: layout for family, layout in zip(clean, assignment)},
            host_capacity_bytes=host_capacity_bytes,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        for assignment in itertools.product(layouts, repeat=len(clean))
    ]
    strongest_fixed = _best(fixed_scores, required=False)
    selected = _best(joint_scores)
    incremental_reuse = None
    fixed_unavailable_reason = None
    if strongest_fixed is None:
        fixed_unavailable_reason = "No uniform layout fits the host-residency budget"
    else:
        incremental_reuse = (
            selected["estimated_reuse_tokens"]
            - strongest_fixed["estimated_reuse_tokens"]
        )
    return {
        "schema": PREFIX_PLAN_SCHEMA,
        "objective": "maximize estimated reusable prefix tokens",
        "claim_boundary": (
            "Structural estimate only; engine telemetry is required to establish "
            "physical restoration or avoided recomputation."
        ),
        "host_capacity_bytes": host_capacity_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "max_assignments": max_assignments,
        "prefix_identity_scope": PREFIX_IDENTITY_SCOPE,
        "families": clean,
        "families_sha256": canonical_sha256(clean),
        "layouts": layouts,
        "assignments_enumerated": assignment_count,
        "fixed_candidates": fixed_scores,
        "joint_candidates": joint_scores,
        "strongest_fixed": strongest_fixed,
        "strongest_fixed_unavailable_reason": fixed_unavailable_reason,
        "selected": selected,
        "estimated_incremental_reuse_tokens": incremental_reuse,
    }


def validate_prefix_plan(value: Any) -> dict:
    if not isinstance(value, dict) or value.get("schema") != PREFIX_PLAN_SCHEMA:
        raise ValueError(f"Expected prefix-plan schema {PREFIX_PLAN_SCHEMA}")
    if value.get("prefix_identity_scope") != PREFIX_IDENTITY_SCOPE:
        raise ValueError("Unexpected prefix identity scope")
    families = value.get("families")
    if not isinstance(families, list):
        raise ValueError("families must contain the normalized planner inputs")
    expected = compile_prefix_plan(
        families,
        host_capacity_bytes=value.get("host_capacity_bytes"),
        kv_bytes_per_token=value.get("kv_bytes_per_token"),
        max_assignments=value.get("max_assignments"),
    )
    if value != expected:
        raise ValueError("Prefix plan does not match its normalized inputs")
    return deepcopy(value)
