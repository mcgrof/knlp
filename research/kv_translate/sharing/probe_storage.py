#!/usr/bin/env python3
"""What is the storage under a directory, and can it be read around the cache?

A storage timing is only as good as the knowledge of what was timed. This
writes a file of random bytes, reads it back through the page cache and
around it, and reports the rates together with how many bytes the kernel
says it fetched from the device for each read. It also records what the
system will say about the mount and the block devices, which on a rented
machine may be little.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from . import cache_store
from .cache_store import storage_facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gib", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    tokens = int(args.gib * 2**30) // (2 * 2 * 8 * 128 * 2)
    tokens -= tokens % 16
    pairs = [
        (
            torch.randn(1, 8, tokens, 128).to(torch.bfloat16),
            torch.randn(1, 8, tokens, 128).to(torch.bfloat16),
        )
        for _ in range(2)
    ]
    path = os.path.join(args.dir, "probe.kv")
    out = {"storage": storage_facts(args.dir)}
    t0 = time.perf_counter()
    rec = cache_store.write(path, pairs, {"model": "probe", "revision": "0"})
    out["write"] = {
        "bytes": rec["logical_bytes"],
        "allocated_bytes": rec["allocated_bytes"],
        "seconds": round(time.perf_counter() - t0, 3),
    }
    reader = cache_store.Reader(rec["payload_bytes"], "cpu")
    out["reads"] = []
    for label, direct in (
        ("page_cache", False),
        ("direct", True),
        ("direct", True),
        ("page_cache", False),
    ):
        t0 = time.perf_counter()
        try:
            head, io = reader.read(path, direct=direct)
        except OSError as e:
            out["reads"].append({"state": label, "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        out["reads"].append(
            {
                "state": label,
                "seconds": round(dt, 3),
                "gib_per_second": round(io["bytes"] / 2**30 / dt, 2),
                "from_storage": io["from_storage"],
                "bytes": io["bytes"],
                "intact": reader.verify(head),
            }
        )
    os.remove(path)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
    print("STORAGE " + json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
