# SPDX-License-Identifier: GPL-2.0
"""Contracts for the bounded repair screen decision."""

import pytest

from research.kv_translate.sharing.repair_screen_gate import assess


def rows(counts, unhealthy=()):
    out = []
    for route, correct in counts.items():
        for query in range(32):
            context = query // 2
            out.append(
                {
                    "identity": "run",
                    "query_id": f"q{query}",
                    "context_id": f"c{context}",
                    "source_group_id": f"g{context}",
                    "domain": "document" if query < 16 else "code",
                    "length_bucket": 16384,
                    "route": route,
                    "correct": query < correct,
                    "health_events": ["empty"] if (route, query) in unhealthy else [],
                }
            )
    return out


def test_half_the_positive_gap_and_two_answers_advance():
    report = assess(
        rows(
            {
                "native_restore": 20,
                "native_recompute": 20,
                "mapped_unrepaired": 10,
                "mapped_uniform": 15,
            }
        )
    )
    assert report["positive_native_gap"] == 10
    assert report["candidates"]["mapped_uniform"]["advances"]
    assert report["verdict"] == "SCREEN_SIGNAL"


def test_a_new_health_failure_blocks_an_otherwise_passing_arm():
    report = assess(
        rows(
            {
                "native_restore": 20,
                "mapped_unrepaired": 10,
                "mapped_uniform": 16,
            },
            unhealthy={("mapped_uniform", 31)},
        )
    )
    arm = report["candidates"]["mapped_uniform"]
    assert arm["new_unhealthy_generations"] == 1
    assert not arm["advances"]


def test_no_positive_native_gap_is_uninformative():
    report = assess(
        rows(
            {
                "native_restore": 10,
                "mapped_unrepaired": 10,
                "mapped_uniform": 20,
            }
        )
    )
    assert report["verdict"] == "PILOT_UNINFORMATIVE"
    assert not report["candidates"]["mapped_uniform"]["advances"]


def test_adaptive_selector_must_beat_its_matched_control():
    report = assess(
        rows(
            {
                "native_restore": 20,
                "mapped_unrepaired": 8,
                "mapped_uniform_blocks": 14,
                "mapped_kri_blocks": 14,
            }
        )
    )
    value = report["adaptive_selector_value"]["mapped_kri_blocks"]
    assert value["net_correct_over_control"] == 0
    assert not value["observed_selector_value"]


def test_a_smoke_subset_cannot_be_mistaken_for_the_full_screen():
    subset = [row for row in rows({"native_restore": 2, "mapped_unrepaired": 1})
              if int(row["query_id"][1:]) < 4]  # fmt: skip
    with pytest.raises(ValueError, match="requires 32 queries"):
        assess(subset)
