#!/usr/bin/env python3
"""Summarize pNFS read-path runs.

  summarize.py screen <dir>       paired blocks: stock (held 0) against held
  summarize.py sweep <dir>        many-client cells
  summarize.py fingerprint <dir>  RPCs per phase and peer for each arm

The screen statistic is the equal-weighted mean over blocks of the paired
block-median difference (stock minus held), absolute and relative to the
stock median, with a 95% percentile block bootstrap (20000 resamples,
seed 20261027). A positive difference means held descriptors are faster.
Each summary is printed and written next to the runs as <mode>-summary.json.
"""

import glob
import json
import os
import random
import re
import statistics
import sys


def boot(vals, n=20000, seed=20261027):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(vals) for _ in vals) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def screen(d):
    cells = {}
    bad = []
    for f in sorted(glob.glob(os.path.join(d, "b*-held*-k*.json"))):
        r = json.load(open(f))
        shape = os.path.basename(f).split("-")[0]
        if r["failures"] or r["digest_mismatches"]:
            bad.append(os.path.basename(f))
        cells[(shape, r["held"], r["block"])] = r
    out = {"shapes": {}, "failed_cells": bad}
    for shape in sorted({k[0] for k in cells}):
        caps = sorted({k[1] for k in cells if k[0] == shape})
        held = max(caps)
        rows, diffs, rels = [], [], []
        for b in sorted({k[2] for k in cells if k[0] == shape}):
            o, h = cells.get((shape, 0, b)), cells.get((shape, held, b))
            if not (o and h):
                continue
            mo, mh = statistics.median(o["latencies_ms"]), statistics.median(
                h["latencies_ms"]
            )
            diffs.append(mo - mh)
            rels.append((mo - mh) / mo)
            rows.append(
                {
                    "block": b,
                    "stock_ms": mo,
                    "held_ms": mh,
                    "diff_ms": mo - mh,
                    "rel": (mo - mh) / mo,
                    "stock_p99_ms": o["p99_ms"],
                    "held_p99_ms": h["p99_ms"],
                    "io": o["io"],
                    "complete": (o["complete"], h["complete"]),
                    "verified": (o["verified_objects"], h["verified_objects"]),
                }
            )
        s = {"held_capacity": held, "blocks": rows}
        if diffs:
            s["mean_diff_ms"], s["mean_rel"] = statistics.fmean(
                diffs
            ), statistics.fmean(rels)
            if len(diffs) > 1:
                s["boot95_diff_ms"], s["boot95_rel"] = boot(diffs), boot(rels)
        out["shapes"][shape] = s
    for shape, s in out["shapes"].items():
        print(f"{shape} (held_fd_capacity {s['held_capacity']}):")
        for r in s["blocks"]:
            print(
                f"  block {r['block']}: stock {r['stock_ms']:.3f} ms, held {r['held_ms']:.3f} ms, "
                f"diff {r['diff_ms']:.3f} ms ({100 * r['rel']:.1f}%), p99 {r['stock_p99_ms']:.1f} / {r['held_p99_ms']:.1f} ms, {r['io']}"
            )
        if "boot95_diff_ms" in s:
            lo, hi = s["boot95_diff_ms"]
            rlo, rhi = s["boot95_rel"]
            print(
                f"  mean {s['mean_diff_ms']:.3f} ms ({100 * s['mean_rel']:.1f}%), 95% block bootstrap "
                f"[{lo:.3f}, {hi:.3f}] ms, [{100 * rlo:.1f}%, {100 * rhi:.1f}%]"
            )
    if bad:
        print("cells with failures or digest mismatches:", bad)
    return out, 1 if bad else 0


def sweep(d):
    rows = []
    for cell in sorted(
        glob.glob(os.path.join(d, "n*-held*")),
        key=lambda c: [int(x) for x in re.findall(r"\d+", os.path.basename(c))],
    ):
        n, held = map(int, re.search(r"n(\d+)-held(\d+)$", cell).groups())
        recs = [
            json.load(open(f)) for f in sorted(glob.glob(os.path.join(cell, "c*.json")))
        ]
        if len(recs) != n:
            rows.append(
                {"n": n, "held": held, "error": f"{len(recs)} of {n} client records"}
            )
            continue

        def ops(name):
            out = {}
            for line in open(os.path.join(cell, f"mds-ops.{name}")):
                k, v = line.rsplit(" ", 1)
                out[re.search(r'op="([^"]+)"', k).group(1)] = int(v)
            return out

        def cg(name):
            t = open(os.path.join(cell, f"cg.{name}")).read().split()
            return dict(zip(t[0::2], map(int, t[1::2])))

        ob, oa = ops("before"), ops("after")
        dops = {k: oa[k] - ob.get(k, 0) for k in oa if oa[k] - ob.get(k, 0)}
        cb, ca = cg("before"), cg("after")
        cpu = {k: (ca[k] - cb[k]) / 1e6 for k in ca}
        objs = sum(r["complete"] * r["batch"] for r in recs)
        per_k = (lambda x: 1000.0 * x / objs) if objs else (lambda x: None)
        rows.append(
            {
                "n": n,
                "held": held,
                "size": recs[0]["size"],
                "io": recs[0]["io"],
                "object_reads_per_s": objs / recs[0]["seconds"],
                "p50_ms_median_client": statistics.median(r["p50_ms"] for r in recs),
                "p99_ms_median_client": statistics.median(r["p99_ms"] for r in recs),
                "p99_ms_worst_client": max(r["p99_ms"] for r in recs),
                "failures": sum(r["failures"] for r in recs),
                "digest_mismatches": sum(r["digest_mismatches"] for r in recs),
                "mds_ops_per_1k_reads": {k: per_k(v) for k, v in dops.items()},
                "metadata_cpu_ms_per_1k_reads": per_k(
                    1000.0 * (cpu.get("mds", 0) + cpu.get("rondb", 0))
                ),
            }
        )
    print(
        f"{'N':>3} {'held':>4} {'reads/s':>9} {'p50 ms':>8} {'p99 ms':>8} {'MDS ops/1k':>11} {'MDS+DB ms/1k':>13} fail bad"
    )
    rc = 0
    for r in rows:
        if "error" in r:
            print(r)
            rc = 1
            continue
        rc |= 1 if r["failures"] or r["digest_mismatches"] else 0
        print(
            f"{r['n']:>3} {r['held']:>4} {r['object_reads_per_s']:>9.0f} {r['p50_ms_median_client']:>8.2f} "
            f"{r['p99_ms_median_client']:>8.1f} {sum(r['mds_ops_per_1k_reads'].values()):>11.0f} "
            f"{r['metadata_cpu_ms_per_1k_reads']:>13.1f} {r['failures']:>4} {r['digest_mismatches']:>3}"
        )
    return rows, rc


def fingerprint(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "held*.summary.json"))):
        held = int(re.search(r"held(\d+)", f).group(1))
        s = json.load(open(f))
        out[held] = {ph: v["requests_by_peer"] for ph, v in s["phases"].items()}
        out[held]["failed"] = s["failed_by_peer"]
        out[held]["fallback_events"] = {
            k: v for k, v in s["events"].items() if "fallback" in k
        }
    for held, v in sorted(out.items()):
        for ph in ("first", "repeat"):
            calls = {
                p.split("/")[0] + " " + k: c
                for p, r in v.get(ph, {}).items()
                for k, c in r.items()
            }
            print(f"held {held:>3} {ph:>6}: {calls}")
        if v["failed"] or v["fallback_events"]:
            print(
                f"held {held:>3} failed RPCs {v['failed']}, fallback {v['fallback_events']}"
            )
    return out, 0


def main():
    mode, d = sys.argv[1], sys.argv[2]
    result, rc = {"screen": screen, "sweep": sweep, "fingerprint": fingerprint}[mode](d)
    json.dump(
        result,
        open(os.path.join(d, f"{mode}-summary.json"), "w"),
        indent=1,
        default=str,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
