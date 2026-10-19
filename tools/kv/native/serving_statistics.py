# SPDX-License-Identifier: MIT
"""Paired run-level effects with exact sign tests and Holm family correction.

Requests and tokens within a run are not independent statistical replicates.
The median interval uses order statistics and reports its actual coverage;
the practical median screen is separate from an interval-bound claim.
"""

import math
import statistics


def paired_effect(candidate, baseline, *, higher_is_better, screen_percent, alpha=0.05):
    if not candidate or len(candidate) != len(baseline):
        raise ValueError("paired block inventories must match")
    if any(not math.isfinite(x) or x <= 0 for x in [*candidate, *baseline]):
        raise ValueError("paired measurements must be positive and finite")
    if not 0 < alpha < 1 or screen_percent < 0:
        raise ValueError("invalid prospective decision rule")
    direction = 1 if higher_is_better else -1
    effects = [direction * (a / b - 1) * 100 for a, b in zip(candidate, baseline)]
    n = len(effects)
    wins = sum(value > 0 for value in effects)
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
    ordered, rank, coverage = sorted(effects), 0, 1.0
    for k in range(1, (n + 1) // 2 + 1):
        error = 2 * sum(math.comb(n, j) for j in range(k)) / 2**n
        if error <= alpha:
            rank, coverage = k, 1 - error
    median = statistics.median(effects)
    return {
        "blocks": n,
        "candidate_values": list(candidate),
        "baseline_values": list(baseline),
        "paired_effect_percent": effects,
        "positive_means_better": True,
        "median_effect_percent": median,
        "median_effect_interval_percent": (
            [ordered[rank - 1], ordered[n - rank]] if rank else [None, None]
        ),
        "median_interval_coverage_at_least": coverage,
        "median_interval_order_statistic_rank": rank,
        "median_interval_unbounded": rank == 0,
        "wins": wins,
        "ties": sum(value == 0 for value in effects),
        "one_sided_sign_p": p,
        "ties_counted_as_nonwins_conservatively": True,
        "screen_percent": screen_percent,
        "median_screen_passed": median >= screen_percent,
        "screen_is_not_an_interval_bound": True,
        "resampling_unit": "independent matched run/block",
    }


def holm(rows, *, alpha=0.05):
    """Correct the full predeclared primary family; never filter failed cells."""
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("primary family must have distinct comparisons")
    if any(not 0 <= r["one_sided_sign_p"] <= 1 for r in rows):
        raise ValueError("invalid family p value")
    ordered = sorted(rows, key=lambda r: r["one_sided_sign_p"])
    previous = 0.0
    result = {}
    for index, row in enumerate(ordered):
        adjusted = min(
            1.0, max(previous, (len(rows) - index) * row["one_sided_sign_p"])
        )
        previous = adjusted
        complete = row.get("all_blocks_valid_and_complete") is True
        result[row["id"]] = {
            **row,
            "primary_family_size": len(rows),
            "holm_adjusted_p": adjusted,
            "holm_passed": adjusted <= alpha,
            "superiority_supported": complete
            and adjusted <= alpha
            and row["median_screen_passed"],
            "reliability_required_for_superiority": True,
        }
    return [result[row["id"]] for row in rows]
