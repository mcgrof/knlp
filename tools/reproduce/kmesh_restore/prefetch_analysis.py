# SPDX-License-Identifier: MIT
"""Verify fixed-arrival evidence and analyze paired engine-process blocks."""

from __future__ import annotations

from collections import defaultdict
import math
import random

MODES = ("staged", "direct")
FANOUTS = (4, 8)
LEADS = (0, 25, 100, 250)
ARMS = {"demand", "early", "declaration", "external_resident", "engine_resident"}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261020
CONFIRMATION_BLOCKS = 10
ENDPOINTS = ("a_deadline_first_token_ms", "deadline_group_completion_ms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(
        type(value) in (int, float) and math.isfinite(value),
        "nonfinite or invalid numeric evidence",
    )
    return value


def stamp(value):
    require(type(value) is int and value >= 0, "invalid monotonic timestamp")
    return value


def duration(record, key, expected):
    actual = number(record[key])
    require(
        actual >= 0 and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6),
        "stored duration disagrees with event timestamps: " + key,
    )


def _verify_clock(row):
    arm = row["arm"]
    require(isinstance(arm, str) and arm in ARMS, "unknown experiment arm")
    lead = number(row["lead_ms"])
    require(lead in LEADS and (arm == "early" or lead == 0), "unexpected lead")
    fanout = row["fanout"]
    require(type(fanout) is int and fanout in FANOUTS, "unexpected fanout")
    origin, arrival = stamp(row["trace_origin_ns"]), stamp(row["scheduled_arrival_ns"])
    require(
        arrival - origin == 250_000_000, "scheduled arrival moved from fixed origin"
    )
    group_submit = stamp(row["group_submit_ns"])
    group_end, drained = stamp(row["group_end_ns"]), stamp(row["drain_end_ns"])
    require(
        arrival <= group_submit <= group_end <= drained, "invalid group event order"
    )
    children = row["rows"]
    require(isinstance(children, list) and len(children) == fanout, "missing children")
    names, internal_ids, submissions = set(), set(), []
    first_times, eighth_times, delays, actual_waits = [], [], [], []
    for child in children:
        name, internal = child["request_id"], child["engine_request_id"]
        require(
            isinstance(name, str)
            and name
            and name not in names
            and isinstance(internal, str)
            and internal
            and internal not in internal_ids,
            "missing or duplicated child identity",
        )
        names.add(name)
        internal_ids.add(internal)
        tokens = child["token_ids"]
        require(
            isinstance(tokens, list)
            and len(tokens) == 8
            and all(type(token) is int and token >= 0 for token in tokens),
            "missing eight-token continuation",
        )
        submitted = stamp(child["submit_ns"])
        accepted = stamp(child["submit_return_ns"])
        first = stamp(child["first_token_observed_ns"])
        eighth = stamp(child["eighth_token_observed_ns"])
        terminal = stamp(child["terminal_observed_ns"])
        require(
            type(child["first_observed_token_count"]) is int
            and child["first_observed_token_count"] == 1
            and first < eighth,
            "first-token clock lacks an earlier single-token observation",
        )
        require(
            group_submit
            <= submitted
            <= accepted
            <= first
            <= eighth
            <= terminal
            <= drained,
            "invalid child event order",
        )
        duration(child, "submission_delay_ms", (submitted - arrival) / 1e6)
        duration(child, "deadline_first_token_ms", (first - arrival) / 1e6)
        duration(child, "deadline_completion_ms", (eighth - arrival) / 1e6)
        require(number(child["ttft_ms"]) > 0, "invalid engine TTFT diagnostic")
        submissions.append(submitted)
        first_times.append(first)
        eighth_times.append(eighth)
        delays.append((submitted - arrival) / 1e6)
        actual_waits.append((first - submitted) / 1e6)
    require(submissions == sorted(submissions), "request A is not first submitted")
    require(group_end == max(eighth_times), "group completion is not last eighth token")
    duration(row, "children_wall_ms", (group_end - group_submit) / 1e6)
    duration(row, "deadline_group_completion_ms", (group_end - arrival) / 1e6)
    duration(row, "workflow_completion_ms", (group_end - origin) / 1e6)
    warm_rows = row["warm_rows"]
    require(isinstance(warm_rows, list), "invalid preparation outputs")
    preparation_fields = (
        "prepare_start_ns",
        "external_ready_ns",
        "warm_submit_ns",
        "warm_done_ns",
    )
    skipped = row.get("warm_skipped_due_to_arrival", False)
    require(type(skipped) is bool, "invalid late-warm skip label")
    if arm == "early":
        start, external = (stamp(row[key]) for key in preparation_fields[:2])
        require(
            origin <= arrival - round(lead * 1e6) <= start <= external <= drained,
            "invalid external preparation clock",
        )
        if skipped:
            require(
                not warm_rows
                and row["warm_submit_ns"] is None
                and row["warm_done_ns"] is None
                and arrival <= external,
                "late-warm skip lacks a due arrival or completion evidence",
            )
            expected_ready = False
        else:
            require(len(warm_rows) == 1, "missing separate preparation continuation")
            warm_tokens = warm_rows[0]["token_ids"]
            require(
                isinstance(warm_rows[0]["request_id"], str)
                and warm_rows[0]["request_id"]
                and warm_rows[0]["request_id"] not in names
                and isinstance(warm_tokens, list)
                and len(warm_tokens) == 1
                and type(warm_tokens[0]) is int
                and warm_tokens[0] >= 0,
                "preparation output was counted as a child",
            )
            submitted, ready = (stamp(row[key]) for key in preparation_fields[2:])
            require(
                external <= submitted <= ready <= drained
                and submitted <= submissions[0],
                "invalid engine preparation clock",
            )
            expected_ready = ready <= arrival
    else:
        require(
            not skipped
            and not warm_rows
            and all(row[key] is None for key in preparation_fields),
            "reference arm contains speculative preparation",
        )
        expected_ready = False
    require(
        type(row["ready_by_scheduled_arrival"]) is bool
        and row["ready_by_scheduled_arrival"] == expected_ready,
        "readiness uses a different arrival clock",
    )
    return {
        "a_deadline_first_token_ms": (first_times[0] - arrival) / 1e6,
        "deadline_group_completion_ms": (group_end - arrival) / 1e6,
        "a_submission_first_token_ms": actual_waits[0],
        "actual_group_completion_ms": (group_end - min(submissions)) / 1e6,
        "workflow_completion_ms": (group_end - origin) / 1e6,
        "submission_delays_ms": delays,
        "all_request_deadline_first_token_ms": [
            (t - arrival) / 1e6 for t in first_times
        ],
        "all_request_submission_first_token_ms": actual_waits,
        "ready_by_scheduled_arrival": expected_ready,
        "warm_request_count": len(warm_rows),
        "warm_skipped_due_to_arrival": skipped,
    }


def verify_clock(row):
    """Derive client-observed waits; do not authenticate physical execution."""
    try:
        return _verify_clock(row)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("incomplete fixed-arrival clock evidence") from error


def _grid(rows, *, pilot, selected_lead_ms=None):
    require(isinstance(rows, list) and rows, "missing experiment rows")
    expected_cases = (
        {(arm, 0) for arm in ARMS - {"early"}} | {("early", lead) for lead in LEADS}
        if pilot
        else {("demand", 0), ("early", selected_lead_ms)}
    )
    strata = {
        (mode, fanout) for mode in MODES for fanout in ((4,) if pilot else FANOUTS)
    }
    expected_samples = set(range(-3, 5 if pilot else 20))
    grouped, block_strata = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
        {},
    )
    for row in rows:
        require(isinstance(row, dict), "invalid experiment row")
        try:
            mode, fanout, block, sample = (
                row[k] for k in ("mode", "fanout", "block_id", "sample")
            )
            require(
                isinstance(mode, str) and (mode, fanout) in strata,
                "unexpected experiment stratum",
            )
            require(isinstance(block, str) and block, "missing process block identity")
            require(
                type(sample) is int and sample in expected_samples,
                "missing or extra sample",
            )
            require(
                type(row["warmup"]) is bool and row["warmup"] == (sample < 0),
                "wrong warmup label",
            )
            metrics = verify_clock(row)
            case = (row["arm"], row["lead_ms"])
            require(
                case in expected_cases, "wrong lead or control in frozen experiment"
            )
            stratum = (mode, fanout)
            require(
                block not in block_strata or block_strata[block] == stratum,
                "process block reused across strata",
            )
            block_strata[block] = stratum
            samples = grouped[stratum][block][case]
            require(sample not in samples, "duplicated sample or pair")
            samples[sample] = metrics
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("incomplete experiment metadata") from error
    require(set(grouped) == strata, "missing experiment stratum")
    for blocks in grouped.values():
        require(
            len(blocks) == (2 if pilot else CONFIRMATION_BLOCKS),
            "missing or extra process block",
        )
        for cases in blocks.values():
            require(set(cases) == expected_cases, "missing control or paired arm")
            require(
                all(set(samples) == expected_samples for samples in cases.values()),
                "missing warmup, sample or pair",
            )
    return grouped


def choose_lead(pilot_rows):
    """Choose a common lead by readiness, never by favorable pilot latency."""
    grouped = _grid(pilot_rows, pilot=True)
    readiness = {}
    for mode in MODES:
        readiness[mode] = {}
        for lead in LEADS:
            samples = [
                metrics
                for cases in grouped[(mode, 4)].values()
                for sample, metrics in cases[("early", lead)].items()
                if sample >= 0
            ]
            readiness[mode][str(lead)] = {
                "ready": sum(m["ready_by_scheduled_arrival"] for m in samples),
                "groups": len(samples),
                "warm_requests": sum(m["warm_request_count"] for m in samples),
                "warm_skipped_due_to_arrival": sum(
                    m["warm_skipped_due_to_arrival"] for m in samples
                ),
            }
    qualifying = [
        lead
        for lead in LEADS
        if all(readiness[mode][str(lead)]["ready"] >= 9 for mode in MODES)
    ]
    return {
        "selected_lead_ms": qualifying[0] if qualifying else 250,
        "readiness_rule_satisfied": bool(qualifying),
        "fallback_to_tested_limit": not qualifying,
        "readiness_by_path_and_lead_ms": readiness,
        "process_blocks_per_path": 2,
        "scope": "Observed pilot readiness counts do not certify a readiness probability or SLO; fresh-process provenance requires separate receipts.",
    }


def percentile(values, q):
    ordered = sorted(values)
    require(bool(ordered), "missing statistical samples")
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "count": len(values),
        "mean_ms": mean(values),
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
    }


def mean(values):
    return math.fsum(values) / len(values)


def confirm(process_rows, *, selected_lead_ms):
    """Analyze the frozen lead using paired process-block means and bootstrap."""
    require(
        type(selected_lead_ms) is int and selected_lead_ms in LEADS,
        "invalid frozen confirmation lead",
    )
    grouped = _grid(process_rows, pilot=False, selected_lead_ms=selected_lead_ms)
    rng, cells = random.Random(BOOTSTRAP_SEED), {}
    for mode in MODES:
        for fanout in FANOUTS:
            blocks = grouped[(mode, fanout)]
            block_means, measured = [], {"demand": [], "early": []}
            for block in sorted(blocks):
                cases = blocks[block]
                pair = {}
                for arm, lead in (("demand", 0), ("early", selected_lead_ms)):
                    samples = [cases[(arm, lead)][j] for j in range(20)]
                    measured[arm].extend(samples)
                    pair[arm] = [mean([s[key] for s in samples]) for key in ENDPOINTS]
                block_means.append(pair)
            baseline = [mean([b["demand"][i] for b in block_means]) for i in range(2)]
            early = [mean([b["early"][i] for b in block_means]) for i in range(2)]
            require(
                all(value > 0 for value in baseline),
                "percentage baseline must be positive",
            )
            absolute, relative = [[], []], [[], []]
            for _ in range(BOOTSTRAP_DRAWS):
                draw = [
                    block_means[rng.randrange(len(block_means))] for _ in block_means
                ]
                for i in range(2):
                    demand_mean, early_mean = mean(
                        [b["demand"][i] for b in draw]
                    ), mean([b["early"][i] for b in draw])
                    require(
                        demand_mean > 0,
                        "percentage bootstrap baseline must be positive",
                    )
                    gain = demand_mean - early_mean
                    absolute[i].append(gain)
                    relative[i].append(100 * gain / demand_mean)
            endpoints = {}
            for i, key in enumerate(ENDPOINTS):
                endpoints[key] = {
                    "demand": distribution([m[key] for m in measured["demand"]]),
                    "early": distribution([m[key] for m in measured["early"]]),
                    "improvement_ms": {
                        "estimate": baseline[i] - early[i],
                        "ci95": [percentile(absolute[i], q) for q in (0.025, 0.975)],
                    },
                    "improvement_percent": {
                        "estimate": 100 * (baseline[i] - early[i]) / baseline[i],
                        "ci95": [percentile(relative[i], q) for q in (0.025, 0.975)],
                    },
                }
            useful = [
                key
                for i, key in enumerate(ENDPOINTS)
                if endpoints[key]["improvement_percent"]["estimate"] >= 5
                and endpoints[key]["improvement_ms"]["ci95"][0] > 0
                and endpoints[ENDPOINTS[1 - i]]["improvement_percent"]["ci95"][0] >= -5
            ]
            cells[f"{mode}-n{fanout}"] = {
                "process_blocks": len(block_means),
                "measured_groups_per_arm": len(measured["demand"]),
                "warmup_groups_per_arm": 3 * len(block_means),
                "early_ready_by_scheduled_arrival": sum(
                    m["ready_by_scheduled_arrival"] for m in measured["early"]
                ),
                "warm_request_counts": {
                    arm: sum(m["warm_request_count"] for m in values)
                    for arm, values in measured.items()
                },
                "early_warm_skipped_due_to_arrival": sum(
                    m["warm_skipped_due_to_arrival"] for m in measured["early"]
                ),
                "endpoints": endpoints,
                "all_request_deadline_first_token": {
                    arm: distribution(
                        [
                            value
                            for m in values
                            for value in m["all_request_deadline_first_token_ms"]
                        ]
                    )
                    for arm, values in measured.items()
                },
                "submission_delay": {
                    arm: distribution(
                        [value for m in values for value in m["submission_delays_ms"]]
                    )
                    for arm, values in measured.items()
                },
                "engineering_rule_satisfied": bool(useful),
                "qualifying_endpoints": useful,
            }
    return {
        "selected_lead_ms": selected_lead_ms,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": "paired engine-process block; both endpoints resampled jointly",
        "estimator": "difference of equally weighted process-block arm means; percentage is ratio of aggregate means",
        "percentile_convention": "linear interpolation at (n-1)*q",
        "confidence_scope": "Nominal per-endpoint 95% intervals, not simultaneous familywise intervals; ten process clusters limit tail evidence.",
        "execution_scope": "Clock/artifact consistency only; fresh-process identity and native numerical/route/drain qualification require separate receipts.",
        "cells": cells,
    }
