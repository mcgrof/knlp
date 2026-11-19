#!/usr/bin/env python3
"""Find what a released pool loader leaves for the cycle collector (GPU).

A server that switches expert pools should free the released pool when
its last reference drops. The mixed-stream test showed that GPU memory
is freed that way, but that without a full collection the test's step
that copies each request's logits to the host grew by up to 2 s per
pass. This script asks what objects survive a switch until a
collection, and whether they slow a device-to-host copy.

It repeats a switch-like cycle --cycles times with automatic collection
disabled: load one pool's per-layer files through PoolLoader (the
fastsafetensors io_uring DMA-BUF copier), optionally assemble each
layer into a new buffer as a delta switch does, then close and drop the
loader as the stream does. After each cycle it times --copies copies of
a logits-sized float32 tensor from the GPU to the host. At the end it
collects with DEBUG_SAVEALL and counts the collected objects by type,
then times the copies again.

Usage:
    loader_cycles.py OUT.json POOL_DIR [POOL_DIR ...]
        [--cycles 8] [--copies 200] [--assemble] [--vocab 50304]
"""

import argparse
import collections
import gc
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, write_json  # noqa: E402
from pool_switch import PoolLoader, evict  # noqa: E402


def copy_times(x, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        y = x.cpu()
        out.append(time.perf_counter() - t)
        del y
    return out


def summary(v):
    v = sorted(v)
    return {
        "median_ms": statistics.median(v) * 1e3,
        "p90_ms": v[int(0.9 * len(v))] * 1e3,
        "max_ms": v[-1] * 1e3,
        "sum_ms": sum(v) * 1e3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("pool_dirs", nargs="+")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--copies", type=int, default=200)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument(
        "--assemble",
        action="store_true",
        help="copy each loaded layer into a new buffer and close its loads",
    )
    args = ap.parse_args()
    if os.environ.get("FASTSAFETENSORS_DMABUF_STRICT") != "1":
        sys.exit("set FASTSAFETENSORS_DMABUF_STRICT=1")

    res = {"runtime": runtime_identity(), "args": vars(args), "cycles": []}
    x = torch.randn(args.vocab, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    copy_times(x, 20)  # warm
    gc.collect()
    res["copies_before"] = summary(copy_times(x, args.copies))
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    count0 = len(gc.get_objects())
    resident = None
    for c in range(args.cycles):
        d = args.pool_dirs[c % len(args.pool_dirs)]
        files = sorted(
            (os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors")),
            key=lambda p: int(os.path.basename(p)[5:].split(".")[0]),
        )
        for p in files:
            evict(p)
        post = None
        if args.assemble:

            def post(li, ts):
                return {k: v.clone() for k, v in ts.items()}

        t0 = time.perf_counter()
        ld = PoolLoader(files, args.inflight, post)
        ld.start()
        ld.join()
        t_load = time.perf_counter() - t0
        if args.assemble:
            for x_ in ld.loads:
                x_.close()
            ld.loads = [None] * len(ld.loads)
        ld.post = post = None
        # The released pool: close it and drop the last name, as a switch does.
        if resident is not None:
            resident.close()
        resident = ld
        ld = None
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        res["cycles"].append(
            {
                "pool": d,
                "load_s": t_load,
                "python_objects": len(gc.get_objects()) - count0,
                "allocated": torch.cuda.memory_allocated(),
                "outside": total - free - torch.cuda.memory_reserved(),
                "copies": summary(copy_times(x, args.copies)),
            }
        )
        print(json.dumps(res["cycles"][-1]), flush=True)
    resident.close()
    resident = None
    res["copies_before_collect"] = summary(copy_times(x, args.copies))
    n = gc.collect()
    kinds = collections.Counter(
        f"{type(o).__module__}.{type(o).__qualname__}" for o in gc.garbage
    )
    res["collected"] = n
    res["collected_by_type"] = dict(kinds.most_common(40))
    gc.set_debug(0)
    gc.garbage.clear()
    gc.collect()
    gc.enable()
    res["copies_after_collect"] = summary(copy_times(x, args.copies))
    free, total = torch.cuda.mem_get_info()
    res["outside_after_collect"] = total - free - torch.cuda.memory_reserved()
    print(json.dumps({k: res[k] for k in res if k.startswith("cop")}, indent=1))
    print("collected", n, json.dumps(res["collected_by_type"], indent=1))
    write_json(args.out, res)


if __name__ == "__main__":
    main()
