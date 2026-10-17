# SPDX-License-Identifier: MIT
"""Reconstruct speed eligibility from complete, independently read evidence."""

import math
import statistics


def decision(evidence, expected):
    """Missing/failed prerequisites cannot be rescued by timing observations.

    The evidence adapter must read raw samples, synchronized timers and source
    receipts, rather than copying the worker's declared qualification flags.
    """
    from ..prequantized.contract import paired_nll
    from .statistics import bounds

    result = {
        "unlocked_speed_claim": False,
        "measured_gain_over_accepted": False,
        "measured_gain_over_native": False,
        "prerequisites_passed": False,
    }
    try:
        assert evidence["identity"] == expected["identity"]
        assert evidence["complete"] is True and evidence["fatal"] is None
        assert evidence["infrastructure_errors"] == []
        for phase in ("production-check-0", "production-check-1"):
            calls = evidence["production"][phase]
            assert len(calls) == 180
            for cohort, (fresh, decode) in enumerate(
                zip([8, 7, 7, 7, 3], [0, 8, 15, 22, 29])
            ):
                group = calls[cohort * 36 : (cohort + 1) * 36]
                assert {x["layer"] for x in group} == set(expected["layers"])
                for row in group:
                    assert row["routing_equivalence_passed"] is True
                    assert row["forward_placement_verified"] is True
                    assert row["finite"] is True
                    assert row["lengths"] == [8192] * fresh
                    assert row["decode_tokens_checked"] == decode
                    assert row["output_sha256"] == row["trusted_sha256"]
                    assert row["lse_sha256"] == row["trusted_lse_sha256"]
        memory = evidence["clean_memory"]
        assert memory["oracle_buffers_present"] is False
        assert memory["pool_bytes"] == 48 * 2**30
        assert (
            memory["payload_bytes"] == 256 * 2**20
            and memory["payload_allocations"] == 1
        )
        assert memory["persistent_v_shadow_bytes"] == 0
        assert memory["allocator_peak_bytes"] <= 68 * 2**30
        assert memory["driver_observed_max_bytes"] <= 72 * 2**30
        assert memory["driver_samples"] > 0 and memory["driver_cadence_seconds"] > 0
        assert evidence["fixture_faults_detected"] == sorted(
            [
                "omit_source_offset",
                "decode_page_substitution",
                "shift_output_suffix",
                "stale_mapping",
            ]
        )
        quality = evidence["quality_samples"]
        checks = {}
        for role in ("native", "accepted"):
            checks[role] = {}
            for stage, count in [("nll", 32), ("retrieval", 100), ("gsm8k", 1319)]:
                a, b = quality["candidate"][stage], quality[role][stage]
                assert len(a) == count and set(a) == set(b) == set(
                    expected["sample_ids"][stage]
                )
                assert all(math.isfinite(v) for v in [*a.values(), *b.values()])
                if stage == "nll":
                    stat = paired_nll(a, b)
                    passed = stat["supported_margin"]
                else:
                    assert all(type(v) is bool for v in [*a.values(), *b.values()])
                    losses = sum(b[k] and not a[k] for k in b)
                    upper = bounds(losses, count)[1]
                    floor = 0.8 if stage == "retrieval" else 0.5
                    passed = (
                        upper <= (0.05 if stage == "retrieval" else 0.02)
                        and sum(b.values()) / count >= floor
                    )
                    stat = {
                        "losses": losses,
                        "loss_upper": upper,
                        "baseline_accuracy": sum(b.values()) / count,
                    }
                checks[role][stage] = {"passed": bool(passed), "statistics": stat}
                assert passed
        result["quality"] = checks
        pilots = evidence["pilot"]
        assert len(pilots) == 3
        assert [x["order"] for x in pilots] == expected["pilot_orders"]
        assert all(
            x["synchronized_complete_request_timer_verified"] is True for x in pilots
        )
        assert all(
            math.isfinite(x["ratios"][r]) and x["ratios"][r] > 0
            for x in pilots
            for r in ("accepted", "native")
        )
        assert statistics.median(x["ratios"]["accepted"] for x in pilots) >= 1.01
        result["prerequisites_passed"] = True
        blocks = evidence["confirmation"]
        assert len(blocks) == 10
        assert [x["order"] for x in blocks] == expected["confirmation_orders"]
        assert all(
            x["synchronized_complete_request_timer_verified"] is True for x in blocks
        )
        hypotheses = {}
        for role in ("accepted", "native"):
            ratios = [x["ratios"][role] for x in blocks]
            assert all(math.isfinite(x) and x > 0 for x in ratios)
            wins = sum(x > 1 for x in ratios)
            hypotheses[role] = {
                "median": statistics.median(ratios),
                "wins": wins,
                "p": sum(math.comb(10, k) for k in range(wins, 11)) / 1024,
            }
        running = 0
        for i, role in enumerate(sorted(hypotheses, key=lambda x: hypotheses[x]["p"])):
            running = max(running, (2 - i) * hypotheses[role]["p"])
            hypotheses[role]["holm_p"] = min(1, running)
        result["hypotheses"] = hypotheses
        result["measured_gain_over_accepted"] = (
            hypotheses["accepted"]["median"] >= 1.01
            and hypotheses["accepted"]["holm_p"] <= 0.05
        )
        result["measured_gain_over_native"] = (
            hypotheses["native"]["median"] >= 1.01
            and hypotheses["native"]["holm_p"] <= 0.05
        )
        result["unlocked_speed_claim"] = (
            result["measured_gain_over_accepted"]
            and result["measured_gain_over_native"]
        )
    except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        result["rejection"] = type(exc).__name__ + ": " + str(exc)
    return result
