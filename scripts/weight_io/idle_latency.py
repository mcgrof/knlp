#!/usr/bin/env python3
"""Measure single-read latency as a function of the idle gap before it.

A consumer NVMe device with autonomous power-state transitions enabled
leaves its active state after a short idle period. The first command after
that pays the exit latency of the deepest state reached, which can be
milliseconds. A replay that settles for a second before every trial then
measures that exit latency, not the device's loaded read latency.

This probe issues one aligned O_DIRECT read of `--size` bytes at a random
page-aligned offset of `--file`, after sleeping `gap` seconds, and repeats
that `--reps` times per gap. It reports the wall latency per read. Run it
on an otherwise idle device; any background I/O keeps the device awake
and hides the effect.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", required=True)
    ap.add_argument("--gaps", default="0,0.01,0.05,0.2,0.5,1,2")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gaps = [float(g) for g in args.gaps.split(",")]
    size = os.stat(args.file).st_size
    fd = os.open(args.file, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, args.size)
    rng = random.Random(args.seed)
    pages = size // args.size - 1

    def one():
        off = rng.randrange(0, pages) * args.size
        t0 = time.monotonic_ns()
        n = os.preadv(fd, [buf], off)
        t1 = time.monotonic_ns()
        assert n == args.size, n
        return (t1 - t0) / 1e3

    # warm the path once so the first measured read is not a code fault
    one()
    results = []
    for gap in gaps:
        lat = []
        for _ in range(args.reps):
            if gap:
                time.sleep(gap)
            lat.append(one())
        results.append(
            {
                "gap_s": gap,
                "reps": args.reps,
                "lat_us": lat,
                "median_us": statistics.median(lat),
                "min_us": min(lat),
                "max_us": max(lat),
            }
        )
        print(
            f"gap {gap:5.2f} s: median {results[-1]['median_us']:9.1f} us "
            f"min {min(lat):9.1f} max {max(lat):9.1f}"
        )
    os.close(fd)
    doc = {
        "file": args.file,
        "size_bytes": args.size,
        "seed": args.seed,
        "o_direct": True,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
