#!/usr/bin/env python3
"""A reader that drives LMCache's native file connector directly.

The compiled ``lmcache_fs`` extension is loaded from the LMCache tree by
file, so a reader needs neither the torch Python package nor the rest of
LMCache. It does need the interpreter the extension was built for, with
torch's C++ libraries on LD_LIBRARY_PATH (``run.sh`` sets both).

Modes:
  init         write --count sealed objects of --size bytes (seeded)
               through the connector, plus a manifest of their digests
  read         loop requests of --batch distinct keys over the first --hot
               objects, one request at a time, for --requests requests or
               --seconds seconds (optionally from --start-at, epoch
               seconds); verify every --verify-every-th request
  fingerprint  read each hot object once (first touch), then --requests
               repeat requests; record CLOCK_MONOTONIC phase windows so a
               trace taken with trace.sh can be split by phase

``--held`` sets the connector's ``held_fd_capacity``: 0 is stock LMCache
(open, read and close per object), a positive value keeps that many read
descriptors open across loads. Every record names the mode, the
capacity, the extension and each request's latency.
"""

import argparse
import ctypes
import glob
import hashlib
import importlib.machinery
import importlib.util
import json
import mmap
import os
import random
import select
import sys
import time


def load_fs_client(src):
    so = glob.glob(os.path.join(src, "lmcache", "lmcache_fs*.so"))
    if not so:
        sys.exit(f"no lmcache_fs extension under {src}/lmcache")
    loader = importlib.machinery.ExtensionFileLoader("lmcache_fs", so[0])
    spec = importlib.util.spec_from_file_location("lmcache_fs", so[0], loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.LMCacheFSClient, so[0]


def aligned(size, align):
    raw = mmap.mmap(-1, size + align)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    off = (-addr) % align
    return raw, memoryview(raw)[off : off + size]


def wait(client, fid, timeout=120.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for c in client.drain_completions():
            if c[0] == fid:
                return c
        select.select([client.event_fd()], [], [], 0.05)
    raise TimeoutError(f"future {fid}")


def key_of(i):
    return f"rpl2@00000000@0@{hashlib.sha256(f'rpl2:{i}'.encode()).hexdigest()}"


def quantile(lat, f):
    return lat[min(len(lat) - 1, int(f * len(lat)))] if lat else None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["init", "read", "fingerprint"])
    p.add_argument("--src", required=True, help="LMCache tree with built lmcache_fs")
    p.add_argument("--base", required=True, help="fixture directory on the mount")
    p.add_argument("--size", type=int, default=8 << 20)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hot", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--requests", type=int, default=0, help="stop after this many requests"
    )
    p.add_argument("--seconds", type=float, default=0, help="or stop after this long")
    p.add_argument("--start-at", type=float, default=0)
    p.add_argument("--held", type=int, default=0, help="held_fd_capacity")
    p.add_argument("--read-workers", type=int, default=16)
    p.add_argument("--verify-every", type=int, default=8)
    p.add_argument("--client", default="cli")
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    FS, so = load_fs_client(a.src)
    statdir = a.base if os.path.isdir(a.base) else os.path.dirname(a.base)
    align = os.statvfs(statdir).f_bsize
    manifest_path = os.path.join(a.base, "RPL2-MANIFEST.json")
    if a.mode == "init":
        if os.path.exists(manifest_path):
            sys.exit(f"{manifest_path} exists; the fixture is sealed")
        os.makedirs(a.base, exist_ok=True)
        rng = random.Random(f"{a.seed}:{a.size}")
        keys, views, keep, digests = [], [], [], {}
        for i in range(a.count):
            raw, v = aligned(a.size, align)
            v[:] = rng.randbytes(a.size)
            keys.append(key_of(i))
            views.append(v)
            keep.append(raw)
            digests[key_of(i)] = hashlib.sha256(v).hexdigest()
        c = FS(a.base, 4, "", True, 0)
        done = wait(c, c.submit_batch_set(keys, views), 600)
        c.close()
        if not done[1]:
            sys.exit(f"store failed: {done[2]}")
        doc = {
            "size": a.size,
            "count": a.count,
            "seed": a.seed,
            "align": align,
            "digests": digests,
        }
        json.dump(doc, open(manifest_path, "w"), indent=1)
        print(f"sealed {a.count} objects of {a.size} bytes in {a.base}")
        return
    if not a.requests and not a.seconds:
        sys.exit("give --requests or --seconds")
    m = json.load(open(manifest_path))
    size = m["size"]
    if a.batch > a.hot:
        sys.exit("--batch must not exceed --hot: a request reads distinct objects")
    hot = [key_of(i) for i in range(a.hot)]
    direct = size % align == 0
    c = FS(a.base, 4, "", True, 0, a.read_workers, 0, a.held)
    ex = wait(c, c.submit_batch_exists(hot))
    if not ex[3] or not all(ex[3]):
        sys.exit("hot objects missing; run init first")
    bufs = [aligned(size, align) for _ in range(a.batch)]
    views = [b[1] for b in bufs]
    windows = []
    mark = lambda msg: windows.append(
        {"mark": msg, "t_ns": time.monotonic_ns()}
    )  # noqa: E731
    first_lat = []
    if a.mode == "fingerprint":
        mark("phase first begin")
        for k in hot:
            t0 = time.monotonic_ns()
            done = wait(c, c.submit_batch_get([k], views[:1]))
            first_lat.append((time.monotonic_ns() - t0) / 1e6)
            if not done[1] or hashlib.sha256(views[0]).hexdigest() != m["digests"][k]:
                sys.exit(f"first touch of {k} failed or mismatched")
        mark("phase first end")
        mark("phase repeat begin")
    if a.start_at:
        time.sleep(max(0.0, a.start_at - time.time()))
    lat, fails, verified, bad = [], 0, 0, 0
    t_end = time.monotonic() + a.seconds if a.seconds else None
    n = 0
    while (a.requests and n < a.requests) or (t_end and time.monotonic() < t_end):
        keys = [hot[(n * a.batch + j) % len(hot)] for j in range(a.batch)]
        t0 = time.monotonic_ns()
        done = wait(c, c.submit_batch_get(keys, views))
        t1 = time.monotonic_ns()
        if not (done[1] and done[3] is not None and all(done[3])):
            fails += 1
        else:
            lat.append((t1 - t0) / 1e6)
            if n % a.verify_every == 0:
                for k, v in zip(keys, views):
                    verified += 1
                    if hashlib.sha256(v).hexdigest() != m["digests"][k]:
                        bad += 1
        n += 1
    if a.mode == "fingerprint":
        mark("phase repeat end")
    stats = c.held_fd_stats() if hasattr(c, "held_fd_stats") else None
    c.close()
    ordered = sorted(lat)
    rec = {
        "mode": a.mode,
        "client": a.client,
        "block": a.block,
        "held": a.held,
        "batch": a.batch,
        "hot": a.hot,
        "size": size,
        "io": "direct" if direct else "buffered",
        "seconds": a.seconds,
        "requests": n,
        "complete": len(lat),
        "failures": fails,
        "verified_objects": verified,
        "digest_mismatches": bad,
        "p50_ms": quantile(ordered, 0.5),
        "p90_ms": quantile(ordered, 0.9),
        "p99_ms": quantile(ordered, 0.99),
        "max_ms": ordered[-1] if ordered else None,
        "mean_ms": sum(lat) / len(lat) if lat else None,
        "latencies_ms": lat,
        "first_touch_ms": first_lat,
        "windows": windows,
        "held_fd_stats": stats,
        "extension": so,
    }
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    keep = (
        "client",
        "held",
        "io",
        "requests",
        "failures",
        "digest_mismatches",
        "p50_ms",
        "p99_ms",
    )
    print(json.dumps({k: rec[k] for k in keep}))
    sys.exit(1 if bad or fails else 0)


if __name__ == "__main__":
    main()
