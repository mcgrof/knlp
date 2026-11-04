#!/usr/bin/env python3
"""Summarize a vLLM + LMCache MP run of run.sh as Markdown.

  summarize.py <run-dir>

Reports, for each I/O mode and phase, the time to first token (p50 and
p90 over requests, median over repeats, and each repeat's p50 so a
drifting repeat is visible), requests per second, NFS OPEN
and CLOSE calls per request from the client's counters, metadata-server
operations per request when they were recorded, and the change from
stock (held_fd_capacity 0) to held descriptors. It also checks that every
LMCache request found its whole prefix cached and compares each request's
first token with the recompute arm's for the same prompt and suffix.
"""

import collections
import glob
import json
import os
import re
import statistics
import sys

ARM = re.compile(r"^(?P<io>[a-z]+)-h(?P<held>\d+)-r(?P<rep>\d+)$")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None


def mds_ops(path):
    """Sum of metadata-server operation counters in a counters file."""
    total, seen = 0, False
    try:
        for line in open(path):
            if line.startswith("pnfs_mds_nfs_op{"):
                total += float(line.split()[-1])
                seen = True
    except OSError:
        return None
    return total if seen else None


def phase_stats(arm_dir, name):
    doc = load(os.path.join(arm_dir, name + ".json"))
    if not doc:
        return None
    ok = [r for r in doc["records"] if "error" not in r]
    lat = [r["latency_s"] * 1e3 for r in ok]
    n = max(len(ok), 1)
    rpc = doc.get("nfs_rpcs") or {}
    before = mds_ops(os.path.join(arm_dir, name + ".before"))
    after = mds_ops(os.path.join(arm_dir, name + ".after"))
    load_reads = 0
    for f in glob.glob(os.path.join(arm_dir, name + "-load*.json")) + glob.glob(
        os.path.join(arm_dir, name + "-bg-c*.json")
    ):
        d = load(f)
        if d and d.get("seconds"):
            load_reads += d["complete"] * d["batch"] / d["seconds"]
    return {
        "requests": len(ok),
        "errors": doc["errors"],
        "p50": pct(lat, 0.5),
        "p90": pct(lat, 0.9),
        "rps": len(ok) / doc["wall_s"] if doc["wall_s"] else 0,
        # the client sends OPEN_NOATTR when it needs no attributes back
        "open": sum(rpc.get(k, {}).get("calls", 0) for k in ("OPEN", "OPEN_NOATTR"))
        / n,
        "close": rpc.get("CLOSE", {}).get("calls", 0) / n,
        # in a busy phase the load's operations are counted too, so only
        # an idle phase gives operations per request
        "mds": (
            (after - before) / n
            if before is not None and after is not None and name.startswith("idle")
            else None
        ),
        "mds_s": (
            (after - before) / doc["wall_s"]
            if before is not None and after is not None and doc["wall_s"]
            else None
        ),
        "load_reads": load_reads or None,
        "cached": collections.Counter(r.get("cached_tokens") for r in ok),
        "prefix": doc["prefix_tokens"],
        "texts": {(r["prompt"], r["pass"]): r["text"] for r in ok},
    }


def fmt(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def change(a, b):
    return "-" if not a or b is None else f"{(b - a) / a * 100:+.1f}%"


def main():
    run = sys.argv[1]
    arms = {}
    for d in sorted(os.listdir(run)):
        m = ARM.match(d)
        if m:
            arms[d] = (m["io"], int(m["held"]), int(m["rep"]))
    phases = sorted(
        {
            os.path.basename(f)[:-5]
            for d in arms
            for f in glob.glob(os.path.join(run, d, "*-c*.json"))
            if re.match(r"^(idle|busy)-c\d+$", os.path.basename(f)[:-5])
        },
        key=lambda p: (p.split("-")[0] != "idle", int(p.split("-c")[1])),
    )
    rec = {p: phase_stats(os.path.join(run, "recompute"), p) for p in phases}
    out = ["# vLLM + LMCache MP: stock against held descriptors", ""]
    out.append(
        "Time to first token in ms (p50, p90 over requests; median over repeats)."
    )
    out.append("OPEN and CLOSE are NFS calls per request from the client's counters;")
    out.append("MDS/req is metadata-server operations per request (idle phases only);")
    out.append("MDS/s counts every client, including the load in busy phases.")
    out.append("")
    out.append(
        "| I/O | phase | held | p50 | p50 by repeat | p90 | req/s | OPEN | CLOSE | MDS/req | MDS/s | load reads/s | p50 change | req/s change |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    problems = []
    agree, compared = 0, 0
    differ = collections.Counter()
    for io in sorted({v[0] for v in arms.values()}):
        for p in phases:
            row = {}
            for held in sorted({v[1] for v in arms.values() if v[0] == io}):
                stats = [
                    phase_stats(os.path.join(run, d), p)
                    for d, v in arms.items()
                    if v[0] == io and v[1] == held
                ]
                stats = [s for s in stats if s]
                if not stats:
                    continue
                for s in stats:
                    if s["errors"]:
                        problems.append(
                            f"{io} held {held} {p}: {s['errors']} failed requests"
                        )
                    short = sum(
                        c
                        for k, c in s["cached"].items()
                        if k is None or k < s["prefix"]
                    )
                    if short:
                        problems.append(
                            f"{io} held {held} {p}: {short} requests found less than the whole prefix cached"
                        )
                    r = rec.get(p) or rec.get(p.replace("busy", "idle"))
                    if r:
                        for key, text in s["texts"].items():
                            if key in r["texts"]:
                                compared += 1
                                agree += text == r["texts"][key]
                                if text != r["texts"][key]:
                                    differ[key] += 1
                med = lambda k: (
                    statistics.median([s[k] for s in stats if s[k] is not None])
                    if any(s[k] is not None for s in stats)
                    else None
                )
                row[held] = {
                    k: med(k)
                    for k in (
                        "p50",
                        "p90",
                        "rps",
                        "open",
                        "close",
                        "mds",
                        "mds_s",
                        "load_reads",
                    )
                }
                row[held]["reps"] = "/".join(
                    fmt(s["p50"], ".0f")
                    for _, s in sorted(
                        zip(
                            [
                                v[2]
                                for d, v in arms.items()
                                if v[0] == io and v[1] == held
                            ],
                            [
                                phase_stats(os.path.join(run, d), p)
                                for d, v in arms.items()
                                if v[0] == io and v[1] == held
                            ],
                        ),
                        key=lambda t: t[0],
                    )
                    if s
                )
            base = row.get(0)
            for held, v in row.items():
                out.append(
                    f"| {io} | {p} | {held} | {fmt(v['p50'])} | {v['reps']} | {fmt(v['p90'])} | {fmt(v['rps'], '.2f')} | "
                    f"{fmt(v['open'], '.1f')} | {fmt(v['close'], '.1f')} | {fmt(v['mds'], '.1f')} | {fmt(v['mds_s'], ',.0f')} | "
                    f"{fmt(v['load_reads'], ',.0f')} | "
                    f"{'' if held == 0 or not base else change(base['p50'], v['p50'])} | "
                    f"{'' if held == 0 or not base else change(base['rps'], v['rps'])} |"
                )
    out.append("")
    out.append("Recompute (vLLM without LMCache):")
    out.append("")
    for p, r in rec.items():
        if r:
            out.append(
                f"- {p}: p50 {fmt(r['p50'])} ms, p90 {fmt(r['p90'])} ms, {fmt(r['rps'], '.2f')} req/s"
            )
    out.append("")
    if compared:
        out.append(
            f"First token equal to recompute in {agree} of {compared} matched requests."
        )
        for (prompt, pas), c in sorted(differ.items()):
            out.append(f"- prompt {prompt} pass {pas} differs in {c} phase(s)")
    out.append(
        "Problems: "
        + (
            "; ".join(problems)
            if problems
            else "none (every request succeeded with its whole prefix cached)"
        )
    )
    print("\n".join(out))


if __name__ == "__main__":
    main()
