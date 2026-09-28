#!/usr/bin/env python3
"""Forecast the rest of a workload from the contexts that have been run.

Rates are taken per length and never carried from one length to another:
prefill, translation and storage do not scale alike, and a rate measured at
4,096 tokens says little about 32,768. A length with no measured context is
reported as unknown and is not filled in with a guess.

Three quantities are kept apart. Time to the first token is what a user
waits. Preparation is work done offline to build and store caches. Allocated
time is what the rental bills, which includes both, the model loads, the
start-up of the machine and the time it sits idle.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

CONTINGENCY = 0.30


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def measured(run):
    """Seconds per context, by domain and length, from what was run."""
    ledger = load_jsonl(os.path.join(run, "CACHE_BYTES.jsonl"))
    gens = load_jsonl(os.path.join(run, "generations.jsonl"))
    times = load_jsonl(os.path.join(run, "timings.jsonl"))
    by = collections.defaultdict(lambda: collections.defaultdict(float))
    seen = collections.defaultdict(set)
    length_of = {}
    for g in gens:
        length_of[g["context_id"]] = (g["domain"], int(g["length_bucket"]))
    for r in ledger:
        key = length_of.get(r["context_id"])
        if key is None:
            continue
        role = r["role"]
        by[key][f"{role}_prefill"] += r["seconds_prefill"]
        by[key][f"{role}_write"] += r["seconds_write"]
        by[key][f"{role}_bytes"] += r["logical_bytes"]
        seen[key].add(r["context_id"])
    for g in gens:
        key = (g["domain"], int(g["length_bucket"]))
        label = (
            "diagnostic"
            if (g.get("mapper") and g["mapper"] != "cache_bridge")
            else "quality"
        )
        by[key][label] += g["request"]["seconds_total"] + g["seconds_generating"]
        by[key][f"{label}_new_tokens"] += g["new_tokens"]
    for t in times:
        key = (t["domain"], int(t["length_bucket"]))
        by[key]["timing"] += t["seconds_total"]
        by[key]["timing_requests"] += 1
    out = {}
    for key, v in by.items():
        n = len(seen[key]) or 1
        out[key] = {k: val / n for k, val in v.items()}
        out[key]["contexts_measured"] = len(seen[key])
    return out


def wall_per_context(run):
    """Wall seconds between one context finishing and the next, in order."""
    import datetime as dt

    rows = []
    cdir = os.path.join(run, "contexts")
    for fn in os.listdir(cdir):
        with open(os.path.join(cdir, fn)) as f:
            r = json.load(f)
        rows.append((dt.datetime.strptime(r["finished_utc"], "%Y-%m-%dT%H:%M:%SZ"), r))
    rows.sort(key=lambda x: x[0])
    return rows


def forecast(run, records, rate, storage_rate, fixed):
    per = measured(run)
    done = {fn[:-5] for fn in os.listdir(os.path.join(run, "contexts"))}
    left = collections.Counter(
        (r["domain"], int(r["length_bucket"]))
        for r in records
        if r["context_id"] not in done
    )
    rows, unknown = [], []
    total = collections.Counter()
    for key, n in sorted(left.items()):
        m = per.get(key)
        if not m:
            unknown.append({"domain": key[0], "length": key[1], "contexts": n})
            continue
        prep = (
            m.get("source_prefill", 0)
            + m.get("source_write", 0)
            + m.get("target_prefill", 0)
            + m.get("target_write", 0)
        )
        row = {
            "domain": key[0],
            "length": key[1],
            "contexts_left": n,
            "contexts_measured": m["contexts_measured"],
            "seconds_per_context": {
                "preparation": round(prep, 2),
                "quality": round(m.get("quality", 0), 2),
                "timing": round(m.get("timing", 0), 2),
            },
            "stored_bytes_per_context": {
                "source": int(m.get("source_bytes", 0)),
                "target": int(m.get("target_bytes", 0)),
            },
        }
        for k, v in row["seconds_per_context"].items():
            total[k] += v * n
        rows.append(row)
    work = sum(total.values())
    seconds = work + sum(fixed.values())
    with_margin = seconds * (1 + CONTINGENCY)
    hourly = rate + storage_rate
    return {
        "strata": rows,
        "lengths_with_no_measurement": unknown,
        "seconds": {
            "preparation": round(total["preparation"], 1),
            "quality": round(total["quality"], 1),
            "timing": round(total["timing"], 1),
            "fixed": fixed,
            "total": round(seconds, 1),
            "total_with_contingency": round(with_margin, 1),
        },
        "contingency": CONTINGENCY,
        "hours_with_contingency": round(with_margin / 3600, 3),
        "rate_per_hour": {
            "card": rate,
            "storage": storage_rate,
            "total": round(hourly, 4),
        },
        "dollars_with_contingency": round(with_margin / 3600 * hourly, 2),
        "complete": not unknown,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument(
        "--rate", type=float, required=True, help="dollars per hour for the card"
    )
    ap.add_argument("--storage-rate", type=float, default=0.0)
    ap.add_argument("--fixed", default="{}", help="JSON of fixed seconds, by name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = load_jsonl(args.records)
    out = forecast(
        args.run, records, args.rate, args.storage_rate, json.loads(args.fixed)
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "hours_with_contingency",
                    "dollars_with_contingency",
                    "complete",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
