# SPDX-License-Identifier: MIT
"""Prospective native-only pilot rules and matched open-loop trace construction."""

import math
import statistics

import numpy as np

from .goodput import evaluate


PILOT_RULES = {
    "arm": "native_fa3",
    "soak_rate": 2.9,
    "soak_arrival_seconds": 60,
    "secondary_1k_rate": 2.9,
    "slo_calibration_rates_in_order": [2.9, 1.5, 1.0, 0.5],
    "load_selection_rates": [0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 3.2, 3.5, 4.0],
    "initial_confirmation_rates": [1.0, 2.9, 4.0],
    "arrival_seconds": 120,
    "drain_seconds": 120,
    "request_timeout_seconds": 240,
    "post_client_quiescence_deadline_seconds": 30,
    "post_client_quiescence_scope": "Two empty scheduler observations at least 200 ms apart before closing a cell; this cleanup does not extend the fixed success/drain deadline",
    "output_tokens": 128,
    "slo_p95_multiplier": 1.1,
    "stability_maximum_drain_seconds": 30,
    "stability_second_half_ttft_factor": 1.25,
    "stability_second_half_ttft_allowance_ms": 50,
    "target_slo_fraction_range": [0.7, 0.9],
    "target_slo_fraction": 0.8,
    "maximum_client_dispatch_p99_ms": 10,
    "maximum_sampled_client_cpu_core_equivalents": 0.9,
    "blocks": 10,
    "seed": 20261020,
}


def trace(data, length, rate, seed, label, *, seconds=120):
    """Independent Poisson arrivals; one trace is replayed unchanged by all arms."""
    if length not in (1024, 8192) or rate <= 0 or not math.isfinite(rate):
        raise ValueError("Unsupported context or invalid open-loop arrival rate")
    prompts = [
        s
        for s in data["samples"]["development"]
        if s["kind"] == "nll" and s["length"] == length
    ]
    if len(prompts) != 16:
        raise ValueError("Require the fixed sixteen excluded development documents")
    rng = np.random.default_rng(seed)
    requests, at = [], 0.0
    while True:
        at += float(rng.exponential(1 / rate))
        offset = round(at * 1e9)
        if offset >= seconds * 1e9:
            break
        prompt = prompts[int(rng.integers(0, len(prompts)))]
        tokens = prompt["token_ids"][:length]
        if len(tokens) != length:
            raise ValueError("Prompt has fewer tokens than the declared geometry")
        requests.append(
            {
                "id": f"{label}-{len(requests):04d}",
                "offset_ns": offset,
                "prompt_token_ids": tokens,
                "max_tokens": 128,
                "source_sample_id": prompt["id"],
            }
        )
    if not requests:
        raise ValueError("Frozen arrival trace is empty")
    return {
        "mode": "open",
        "model": "study",
        "seed": 20261020,
        "arrival_seconds": seconds,
        "drain_seconds": 120,
        "request_timeout_seconds": 240,
        "requests": requests,
        "arrival_process": "Seeded Poisson; independent trace per block, identical across arms",
        "trace_seed": seed,
        "nominal_rate": rate,
        "input_tokens": length,
        "scope": "Fixed-length Wikipedia development prefixes with 128 forced output tokens; no natural-length customer-trace claim",
    }


def orders(arms):
    """Ten blocks with position-balanced arm orders, fixed before timing."""
    if len(arms) != len(set(arms)) or set(arms) not in (
        {"native_fi", "AF", "native_fa3"},
        {"native_fi", "AF", "native_fa3", "SF"},
    ):
        raise ValueError("Timing requires asymmetric and both native controls")
    rng = np.random.default_rng(20261020)
    names = list(rng.permutation(sorted(arms)))
    if len(names) == 4:
        indices = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
        result = []
        while len(result) < 10:
            for index in rng.permutation(4):
                result.append([str(names[i]) for i in indices[index]])
        return result[:10]
    result = []
    for group in range(4):
        base = names if group % 2 == 0 else list(reversed(names))
        for offset in rng.permutation(3):
            result.append([str(x) for x in base[offset:] + base[:offset]])
    return result[:10]


def harness_gate(client, boundary):
    return (
        client["dispatch_lag_ms"]["p99"] is not None
        and client["dispatch_lag_ms"]["p99"]
        <= PILOT_RULES["maximum_client_dispatch_p99_ms"]
        and client["peak_observed_cpu_core_equivalents"]
        <= PILOT_RULES["maximum_sampled_client_cpu_core_equivalents"]
        and boundary["connection_reuses"] > 0
    )


def stable(capture, boundary, client):
    # Use unbounded positive SLOs only to obtain valid completions and native
    # latency observations; the actual SLO numbers are frozen separately below.
    audit = evaluate(capture, {k: 1e12 for k in ("ttft_ms", "tpot_ms", "eos_ms")})
    rows = audit["requests"]
    midpoint = capture["start_ns"] + int(capture["plan"]["arrival_seconds"] * 1e9 / 2)
    intended = {r["id"]: r["intended_ns"] for r in capture["requests"]}
    halves = [
        [
            r["metrics"]["ttft_arrival_ms"]
            for r in rows
            if r["valid"] and (intended[r["id"]] >= midpoint) == later
        ]
        for later in (False, True)
    ]
    drift_passed = all(halves) and statistics.mean(halves[1]) <= (
        PILOT_RULES["stability_second_half_ttft_factor"] * statistics.mean(halves[0])
        + PILOT_RULES["stability_second_half_ttft_allowance_ms"]
    )
    last = max(
        (r["metrics"].get("eos_ns") or capture["end_ns"] for r in rows),
        default=capture["end_ns"],
    )
    drain = max(0, (last - capture["arrival_end_ns"]) / 1e9)
    return {
        "passed": bool(
            audit["valid_within_drain"] == audit["offered"]
            and not boundary["preemption_events"]
            and drift_passed
            and drain <= PILOT_RULES["stability_maximum_drain_seconds"]
            and harness_gate(client, boundary)
        ),
        "mean_ttft_by_arrival_half_ms": [
            statistics.mean(x) if x else None for x in halves
        ],
        "ttft_drift_passed": bool(drift_passed),
        "drain_seconds": drain,
        "scope": "Prospective finite-pilot stability screen, not proof of stationary throughput",
    }


def derive_slo(capture, boundary, client):
    check = stable(capture, boundary, client)
    if not check["passed"]:
        raise ValueError(
            "Native calibration pilot did not pass the fixed stability screen"
        )
    values = evaluate(capture, {k: 1e12 for k in ("ttft_ms", "tpot_ms", "eos_ms")})
    mapping = {
        "ttft_ms": "ttft_arrival_ms",
        "tpot_ms": "tpot_ms",
        "eos_ms": "latency_arrival_ms",
    }
    return {
        name: 1.1 * values["latencies_conditional_on_valid_success"][metric]["p95"]
        for name, metric in mapping.items()
    }


def stress_load(pilots):
    """Select only from complete native-only pilots, with no candidate fallback."""
    if len({r["rate"] for r in pilots}) != len(pilots):
        raise ValueError("Native pilot loads are repeated")
    if sorted(r["rate"] for r in pilots) != PILOT_RULES["load_selection_rates"]:
        raise ValueError("All bounded native-only load cells must be retained")
    if any(r["arm"] != "native_fa3" for r in pilots):
        raise ValueError("Candidate observations cannot select the primary load")
    low, high = PILOT_RULES["target_slo_fraction_range"]
    eligible = [
        r
        for r in pilots
        if r["harness_passed"]
        and r["resource_passed"]
        and low <= r["slo_fraction"] <= high
    ]
    if not eligible:
        return {
            "rate": None,
            "reason": "No informative 70–90% native SLO-attainment cell in the fixed bounded pilot",
        }
    best = min(
        eligible, key=lambda r: (round(abs(r["slo_fraction"] - 0.8), 12), r["rate"])
    )
    return {
        "rate": best["rate"],
        "native_pilot_slo_fraction": best["slo_fraction"],
        "selection": "Closest to 80% SLO attainment among eligible 70–90% cells; lower rate breaks ties",
    }
