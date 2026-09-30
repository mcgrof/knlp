#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Apply the frozen bounded-repair quality heuristic to saved scores.

The heuristic is deliberately small-sample and descriptive.  It decides
whether another development allocation is warranted; it is not a confidence
interval, a deployment claim, or permission to discard any measured arm.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys

NATIVE = "native_restore"
UNREPAIRED = "mapped_unrepaired"
MIN_GAIN = 2
MIN_GAP_FRACTION = 0.5
EXPECTED_QUERIES = 32
EXPECTED_CONTEXTS = 16
EXPECTED_LENGTH = 16384


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def route_rows(rows):
    by = collections.defaultdict(dict)
    identities = set()
    for row in rows:
        if row["correct"] is None:
            raise ValueError(f"{row['query_id']} has no usable score")
        key = row["query_id"]
        route = row["route"]
        if key in by[route]:
            raise ValueError(f"duplicate score for {route}/{key}")
        by[route][key] = row
        identities.add(row["identity"])
    if len(identities) != 1:
        raise ValueError("scores contain more than one run identity")
    if NATIVE not in by or UNREPAIRED not in by:
        raise ValueError(f"scores require {NATIVE} and {UNREPAIRED}")
    expected = set(by[NATIVE])
    if len(expected) != EXPECTED_QUERIES:
        raise ValueError(
            f"screen requires {EXPECTED_QUERIES} queries, found {len(expected)}"
        )
    native_rows = list(by[NATIVE].values())
    per_context = collections.Counter(row["context_id"] for row in native_rows)
    if len(per_context) != EXPECTED_CONTEXTS or set(per_context.values()) != {2}:
        raise ValueError("screen requires 16 contexts with two queries each")
    domains = collections.Counter(row["domain"] for row in native_rows)
    if domains != {"document": 16, "code": 16}:
        raise ValueError(f"screen requires 16 questions per domain, found {domains}")
    if {int(row["length_bucket"]) for row in native_rows} != {EXPECTED_LENGTH}:
        raise ValueError(f"screen requires only {EXPECTED_LENGTH}-token prefixes")
    if len({row["source_group_id"] for row in native_rows}) != EXPECTED_CONTEXTS:
        raise ValueError("screen requires 16 independent source groups")
    for route, found in by.items():
        if set(found) != expected:
            raise ValueError(f"{route} does not cover the same query IDs")
    return by, identities.pop()


def quality(row_map):
    correct = {query for query, row in row_map.items() if row["correct"]}
    unhealthy = {
        query for query, row in row_map.items() if row.get("health_events", [])
    }
    events = collections.Counter(
        event for row in row_map.values() for event in row.get("health_events", [])
    )
    return {
        "queries": len(row_map),
        "correct": len(correct),
        "unhealthy": len(unhealthy),
        "health_events": dict(sorted(events.items())),
        "correct_query_ids": correct,
        "unhealthy_query_ids": unhealthy,
    }


def timing_summary(rows):
    by = collections.defaultdict(list)
    for row in rows:
        by[row["route"]].append(float(row["seconds_total"]))
    return {
        route: {
            "requests": len(values),
            "p50_seconds": statistics.median(values),
            "p95_seconds": sorted(values)[
                max(0, (95 * len(values) + 99) // 100 - 1)
            ],
        }
        for route, values in sorted(by.items())
    }


def assess(scores, timings=()):
    by, identity = route_rows(scores)
    summaries = {route: quality(rows) for route, rows in by.items()}
    native = summaries[NATIVE]
    unrepaired = summaries[UNREPAIRED]
    gap = native["correct"] - unrepaired["correct"]
    candidates = {}
    for route, summary in summaries.items():
        if not route.startswith("mapped_") or route == UNREPAIRED:
            continue
        gain = summary["correct"] - unrepaired["correct"]
        new_unhealthy = sorted(
            summary["unhealthy_query_ids"] - unrepaired["unhealthy_query_ids"]
        )
        fraction = gain / gap if gap > 0 else None
        checks = {
            "gain_at_least_two": gain >= MIN_GAIN,
            "half_positive_native_gap_closed": fraction is not None
            and fraction >= MIN_GAP_FRACTION,
            "no_new_unhealthy_generations": not new_unhealthy,
        }
        candidates[route] = {
            "correct": summary["correct"],
            "unhealthy": summary["unhealthy"],
            "net_correct_gain_over_unrepaired": gain,
            "positive_native_gap_fraction_closed": fraction,
            "new_unhealthy_generations": len(new_unhealthy),
            "new_unhealthy_query_ids": new_unhealthy,
            "checks": checks,
            "advances": all(checks.values()),
        }

    adaptive_value = {}
    for adaptive, control in (
        ("mapped_key_deviation", "mapped_uniform"),
        ("mapped_kri_blocks", "mapped_uniform_blocks"),
    ):
        if adaptive in summaries and control in summaries:
            adaptive_value[adaptive] = {
                "control": control,
                "net_correct_over_control": summaries[adaptive]["correct"]
                - summaries[control]["correct"],
                "observed_selector_value": summaries[adaptive]["correct"]
                > summaries[control]["correct"],
            }

    for summary in summaries.values():
        summary["correct_query_ids"] = sorted(summary["correct_query_ids"])
        summary["unhealthy_query_ids"] = sorted(summary["unhealthy_query_ids"])
    advances = sorted(route for route, result in candidates.items() if result["advances"])
    return {
        "schema": "bounded_repair_screen_gate_v1",
        "identity": identity,
        "thresholds": {
            "net_correct_gain_over_unrepaired_min": MIN_GAIN,
            "positive_native_gap_fraction_closed_min": MIN_GAP_FRACTION,
            "new_unhealthy_generations_max": 0,
            "expected_queries": EXPECTED_QUERIES,
            "expected_contexts": EXPECTED_CONTEXTS,
            "expected_length": EXPECTED_LENGTH,
        },
        "native_comparator": NATIVE,
        "unrepaired_comparator": UNREPAIRED,
        "positive_native_gap": gap,
        "native_controls_correct_difference": summaries[NATIVE]["correct"]
        - summaries.get("native_recompute", summaries[NATIVE])["correct"],
        "quality": summaries,
        "candidates": candidates,
        "adaptive_selector_value": adaptive_value,
        "advancing_routes": advances,
        "verdict": (
            "PILOT_UNINFORMATIVE"
            if gap <= 0
            else ("SCREEN_SIGNAL" if advances else "SCREEN_FAIL")
        ),
        "timing": timing_summary(timings) if timings else {},
        "note": (
            "Resource-allocation heuristic only; all arms remain development "
            "evidence and adaptive selectors still require matched-control value."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--timings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        result = assess(
            load_jsonl(args.scores),
            load_jsonl(args.timings) if args.timings else (),
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"invalid repair-screen evidence: {error}", file=sys.stderr)
        return 2
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(result["verdict"], ",".join(result["advancing_routes"]) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
