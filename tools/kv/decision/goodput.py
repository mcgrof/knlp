# SPDX-License-Identifier: MIT
"""All-offered SLO goodput and paired block inference for frozen HTTP traces."""

import math
import statistics

from tools.kv.native.serving_events import quantile, request_metrics


def evaluate(capture, slo):
    """Keep late/failed requests in the denominator and successful latency scope."""
    from .serving import validate

    limits = {
        "ttft_arrival_ms": "ttft_ms",
        "tpot_ms": "tpot_ms",
        "latency_arrival_ms": "eos_ms",
    }
    if set(slo) != set(limits.values()) or any(
        not math.isfinite(value) or value <= 0 for value in slo.values()
    ):
        raise ValueError("Require three positive, finite frozen SLO limits")
    plan = capture["plan"]
    validate(plan)
    expected = {row["id"]: row for row in plan["requests"]}
    rows = capture["requests"]
    if len(expected) != len(rows) or {r["id"] for r in rows} != set(expected):
        raise ValueError("All offered IDs must appear exactly once")
    seconds = (capture["arrival_end_ns"] - capture["start_ns"]) / 1e9
    if seconds != plan["arrival_seconds"] or seconds <= 0:
        raise ValueError("Arrival window differs from frozen plan")
    end = (capture["end_ns"] - capture["start_ns"]) / 1e9
    if end < seconds:
        raise ValueError("Observation ended before the arrival window")
    deadline = capture["arrival_end_ns"] + int(plan["drain_seconds"] * 1e9)
    if deadline != capture["drain_deadline_ns"]:
        raise ValueError("Drain deadline differs from frozen plan")
    scored, successful = [], []
    for row in rows:
        frozen = expected[row["id"]]
        if row["intended_ns"] != capture["start_ns"] + frozen["offset_ns"]:
            raise ValueError("Request arrival changed")
        if (
            row["max_tokens"] != frozen["max_tokens"]
            or row["prompt_token_ids"] != frozen["prompt_token_ids"]
        ):
            raise ValueError("Request workload changed")
        try:
            metric = request_metrics(row)
            valid = (
                metric["valid_completion"]
                and metric["output_tokens"] == frozen["max_tokens"]
                and metric["finish_reasons"] == ["length"]
                and all(
                    metric[name] is not None
                    and math.isfinite(metric[name])
                    and metric[name] >= 0
                    for name in limits
                )
                and metric["first_token_ns"]
                <= metric["last_token_ns"]
                <= metric["eos_ns"]
                <= row["end_ns"]
            )
        except (ValueError, TypeError, KeyError):
            metric, valid = {}, False
        within_drain = valid and metric["eos_ns"] <= deadline
        passes = within_drain and all(
            metric[name] <= slo[key] for name, key in limits.items()
        )
        scored.append(
            {
                "id": row["id"],
                "status": row["status"],
                "valid": bool(valid),
                "within_drain": bool(within_drain),
                "slo_pass": bool(passes),
                "metrics": metric,
            }
        )
        if valid:
            successful.append(metric)
    good = sum(row["slo_pass"] for row in scored)
    complete = sum(row["within_drain"] for row in scored)
    gaps = [gap for metric in successful for gap in metric["inter_token_event_gaps_ms"]]
    return {
        "offered": len(rows),
        "valid_successful": len(successful),
        "valid_within_drain": complete,
        "slo_completed": good,
        "success_fraction_of_all_offered": complete / len(rows),
        "slo_fraction_of_all_offered": good / len(rows),
        "goodput_requests_per_second": good / seconds,
        "completed_requests_per_second_including_drain": complete / end,
        "arrival_seconds": seconds,
        "observed_seconds_including_drain": end,
        "frozen_slo": slo,
        "latencies_conditional_on_valid_success": {
            name: {
                "n": len(successful),
                "p50": quantile([r[name] for r in successful], 0.5),
                "p95": quantile([r[name] for r in successful], 0.95),
            }
            for name in limits
        },
        "token_event_gap_ms_conditional_on_valid_success": {
            "n": len(gaps),
            "p50": quantile(gaps, 0.5),
            "p95": quantile(gaps, 0.95),
            "p99": quantile(gaps, 0.99),
        },
        "tokens_per_event": [
            count for metric in successful for count in metric["tokens_per_event"]
        ],
        "token_resolution": "Coalesced token timestamps are not independently observable; gaps are between token-bearing SSE events, and TPOT divides first-to-last time by total tokens minus one.",
        "requests": scored,
    }


def median_lower(values, alpha):
    """Distribution-free one-sided median bound with conservative tie handling."""
    if not values or not 0 < alpha < 0.5 or not all(math.isfinite(x) for x in values):
        raise ValueError("Invalid median interval inputs")
    ordered = sorted(values)
    n, rank, tail = len(values), 0, 0.0
    for k in range(1, n + 1):
        error = sum(math.comb(n, j) for j in range(k)) / 2**n
        if error <= alpha:
            rank, tail = k, error
        else:
            break
    return {
        "lower": ordered[rank - 1] if rank else None,
        "unbounded_below": rank == 0,
        "rank": rank,
        "coverage_at_least": 1 - tail,
        "alpha": alpha,
    }


def compare_blocks(candidate, controls, *, alpha=0.05, screen_percent=5):
    """Holm tests plus stronger simultaneous Bonferroni median lower bounds.

    Require all ten scheduled blocks. Requests within a block are correlated;
    neither they nor generated tokens increase the inference sample size.
    Zero-goodput controls permit absolute differences, not finite percentages.
    """
    if len(candidate) != 10 or not controls or not 0 < alpha < 0.5:
        raise ValueError("Require ten paired blocks and the full control family")
    if screen_percent < 0 or not math.isfinite(screen_percent):
        raise ValueError("Invalid practical improvement screen")
    values = [*candidate, *(x for rows in controls.values() for x in rows)]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Goodput must be finite and nonnegative")
    result = []
    for name, baseline in controls.items():
        if len(baseline) != len(candidate):
            raise ValueError("Incomplete paired control")
        absolute = [a - b for a, b in zip(candidate, baseline)]
        relative = (
            [(a / b - 1) * 100 for a, b in zip(candidate, baseline)]
            if all(b > 0 for b in baseline)
            else None
        )
        wins = sum(delta > 0 for delta in absolute)
        p = sum(math.comb(10, k) for k in range(wins, 11)) / 2**10
        lower = median_lower(absolute, alpha / len(controls))
        result.append(
            {
                "control": name,
                "candidate": list(candidate),
                "baseline": list(baseline),
                "paired_absolute_goodput_delta": absolute,
                "paired_relative_percent": relative,
                "median_relative_percent": (
                    statistics.median(relative) if relative else None
                ),
                "median_absolute_delta": statistics.median(absolute),
                "one_sided_sign_p": p,
                "ties_count_as_nonwins": True,
                "simultaneous_median_absolute_lower": lower,
                "simultaneous_median_relative_lower": (
                    median_lower(relative, alpha / len(controls)) if relative else None
                ),
                "interval_adjustment": "Bonferroni across every named control; conservative simultaneous bounds accompanying Holm tests",
            }
        )
    previous = 0.0
    for index, row in enumerate(sorted(result, key=lambda r: r["one_sided_sign_p"])):
        adjusted = min(
            1.0, max(previous, (len(controls) - index) * row["one_sided_sign_p"])
        )
        previous = adjusted
        row["holm_adjusted_p"] = adjusted
        row["practical_improvement_established"] = bool(
            adjusted <= alpha
            and row["simultaneous_median_absolute_lower"]["lower"] is not None
            and row["simultaneous_median_absolute_lower"]["lower"] > 0
            and row["median_relative_percent"] is not None
            and row["median_relative_percent"] >= screen_percent
        )
    return {
        "blocks": 10,
        "family_size": len(controls),
        "alpha": alpha,
        "screen_percent": screen_percent,
        "comparisons": result,
        "improvement_vs_all_controls": all(
            r["practical_improvement_established"] for r in result
        ),
    }
