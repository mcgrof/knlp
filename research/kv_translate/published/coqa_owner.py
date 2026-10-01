#!/usr/bin/env python3
"""CPU contracts for CoQA conversations owned by a native 14B cache."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from . import coqa

STREAMS = ("translated", "same_transcript_native", "native_32b_only")


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _validate(rows: list[dict], stage: str) -> None:
    if stage not in ("s1", "s2"):
        raise ValueError(f"quality stage must be s1 or s2, got {stage!r}")
    if not rows:
        raise ValueError("no native-source-owner rows")
    if {row["domain"] for row in rows} != set(coqa.DOMAINS):
        raise ValueError("rows must cover every frozen CoQA domain")
    expected_turns = set(range(1, 11))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)
        for stream in STREAMS:
            if not math.isfinite(float(row[f"{stream}_f1"])):
                raise ValueError(f"row {row.get('row_id')} lacks finite {stream}_f1")
            if not isinstance(row[f"{stream}_health"], list):
                raise ValueError(f"row {row.get('row_id')} lacks {stream}_health")
    for conversation_id, conversation_rows in grouped.items():
        turns = {row["turn"] for row in conversation_rows}
        if turns != expected_turns:
            raise ValueError(
                f"conversation {conversation_id} has turns {sorted(turns)}, "
                f"expected {sorted(expected_turns)}"
            )


def _arm(rows: list[dict], stream: str) -> dict:
    def cell(cell_rows):
        return {
            "answers": len(cell_rows),
            "f1": _mean(row[f"{stream}_f1"] for row in cell_rows),
            "unhealthy": sum(bool(row[f"{stream}_health"]) for row in cell_rows),
        }

    return {
        "pooled": cell(rows),
        "by_domain": {
            domain: cell([row for row in rows if row["domain"] == domain])
            for domain in coqa.DOMAINS
        },
        "by_turn": {
            str(turn): cell([row for row in rows if row["turn"] == turn])
            for turn in range(1, 11)
        },
    }


def _contrast(rows: list[dict], reference: str) -> dict:
    def cell(cell_rows):
        return {
            "answers": len(cell_rows),
            "reference_minus_translated_f1": _mean(
                row[f"{reference}_f1"] - row["translated_f1"] for row in cell_rows
            ),
            "translated_unhealthy_excess": _mean(
                bool(row["translated_health"]) - bool(row[f"{reference}_health"])
                for row in cell_rows
            ),
        }

    by_domain = {
        domain: cell([row for row in rows if row["domain"] == domain])
        for domain in coqa.DOMAINS
    }
    late = [row for row in rows if row["turn"] >= 6]
    return {
        "pooled": cell(rows),
        "equal_domain_reference_minus_translated_f1": _mean(
            by_domain[domain]["reference_minus_translated_f1"]
            for domain in coqa.DOMAINS
        ),
        "turns_6_to_10_equal_domain_reference_minus_translated_f1": _mean(
            _mean(
                row[f"{reference}_f1"] - row["translated_f1"]
                for row in late
                if row["domain"] == domain
            )
            for domain in coqa.DOMAINS
        ),
        "by_domain": by_domain,
        "by_turn": {
            str(turn): cell([row for row in rows if row["turn"] == turn])
            for turn in range(1, 11)
        },
    }


def _screen(contrast: dict, *, require_domains: bool) -> dict:
    requirements = {
        "equal_domain_deficit_at_most_0.03": contrast[
            "equal_domain_reference_minus_translated_f1"
        ]
        <= 0.03,
        "pooled_health_excess_at_most_0.02": contrast["pooled"][
            "translated_unhealthy_excess"
        ]
        <= 0.02,
        "turns_6_to_10_equal_domain_deficit_at_most_0.05": contrast[
            "turns_6_to_10_equal_domain_reference_minus_translated_f1"
        ]
        <= 0.05,
    }
    if require_domains:
        requirements["each_domain_deficit_at_most_0.05"] = all(
            contrast["by_domain"][domain]["reference_minus_translated_f1"] <= 0.05
            for domain in coqa.DOMAINS
        )
    return {"requirements": requirements, "passed": all(requirements.values())}


def _bootstrap(rows: list[dict], *, replicates: int) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["domain"]][row["conversation_id"]].append(row)
    rng = np.random.default_rng(coqa.BOOTSTRAP_SEED)
    samples = {reference: [] for reference in STREAMS[1:]}
    for _ in range(replicates):
        losses = {reference: [] for reference in STREAMS[1:]}
        for domain in coqa.DOMAINS:
            ids = sorted(grouped[domain])
            picked = rng.choice(ids, size=len(ids), replace=True)
            sample = [row for cid in picked for row in grouped[domain][str(cid)]]
            for reference in STREAMS[1:]:
                losses[reference].append(
                    _mean(
                        row[f"{reference}_f1"] - row["translated_f1"] for row in sample
                    )
                )
        for reference in STREAMS[1:]:
            samples[reference].append(_mean(losses[reference]))
    return {
        "seed": coqa.BOOTSTRAP_SEED,
        "replicates": replicates,
        "unit": "complete conversations resampled within domain",
        "intervals": {
            f"{reference}_loss_95_interval": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for reference, values in samples.items()
        },
        "interpretation": "descriptive only; not population noninferiority",
    }


def summarize(
    rows: list[dict],
    stage: str,
    *,
    replicates: int = coqa.BOOTSTRAP_REPLICATES,
) -> dict:
    """Apply the prospective S1/S2 screens against both native references."""
    _validate(rows, stage)
    contrasts = {reference: _contrast(rows, reference) for reference in STREAMS[1:]}
    screens = {
        reference: _screen(contrast, require_domains=stage == "s2")
        for reference, contrast in contrasts.items()
    }
    passed = all(screen["passed"] for screen in screens.values())
    if stage == "s1":
        next_state = "S2" if passed else "CLOSE_CONFIGURATION"
    else:
        next_state = "SCOPED_REPLICATION_PASS" if passed else "DEVELOPMENT_ONLY"
    return {
        "schema": f"coqa_native_source_owner_{stage}_result_v1",
        "stage": stage,
        "rows": len(rows),
        "conversations": len({row["conversation_id"] for row in rows}),
        "answers": len(rows) * len(STREAMS),
        "arms": {stream: _arm(rows, stream) for stream in STREAMS},
        "contrasts": contrasts,
        "screens": screens,
        "all_required_gates_pass": passed,
        "next_state": next_state,
        "bootstrap": _bootstrap(rows, replicates=replicates),
        "meaning": "prospective point screens; intervals are descriptive",
    }
