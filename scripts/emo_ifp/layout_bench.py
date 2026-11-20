#!/usr/bin/env python3
"""Time pool loads from one shared copy against per-pool copies (GPU).

Two disk layouts hold the same expert bytes:

    packed       one file per pool and layer, and one delta file per
                 ordered pool pair and layer (pack_pool_layer.py); an
                 expert kept by several pools is stored several times
    single-copy  one file per layer with every routed expert once
                 (pack_single_copy.py); a pool or a switch delta is a
                 list of contiguous runs into it

The packed files are read by the fastsafetensors DMA-BUF copier
(PoolLoader, two layers in flight), as in the serving tests. The
single-copy runs are read by a small range reader built on the same
io_uring DMA-BUF primitives: per layer it allocates one compact GPU
buffer (gate/up slots, then down slots), registers it, and issues one READ_FIXED per run straight from the file into
its slot, two layers in flight. Both of these arms include
allocation, registration, reads and unregistration; buffers are freed
after the clock stops. Arms ending in "registered" allocate and register
every layer's buffer before the clock and time only the reads, as for
destination slots a server keeps registered. The packed-ranges and
packed-registered arms read the packed files with the same range
reader, so layout, reader and registration effects can be separated:

    packed             copier, packed files
    packed-ranges      range reader, packed files, cold
    packed-registered  range reader, packed files, registered
    single             range reader, single-copy file, cold
    single-registered  range reader, single-copy file, registered

Arms, alternated per trial: each pool from each layout, and each ordered
pool pair's delta from each layout. After the timed trials, one range
load of every pool and of every delta is copied to the host and every
expert slice is checked against the SHA-256 that pack_single_copy.py
recorded from the packed full model, read once through each layout.

Usage:
    layout_bench.py SINGLE_DIR PACKED_DIR OUT.json [--trials 3] \
        [--pools arc health psychology history]
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, stacked_name, write_json  # noqa: E402
from pool_switch import PoolLoader  # noqa: E402

from fastsafetensors.io_uring_dmabuf import (  # noqa: E402
    IoUring,
    dedicated_malloc,
    device_memcpy,
    device_scope,
    register_destination,
    release_dedicated,
    release_regions,
)

ALIGN = 4096
IS_HIP = torch.version.hip is not None


class Buf:
    """One dedicated GPU allocation, page aligned, registered for a file."""

    def __init__(self, ring, nbytes, fd):
        self.ptr = dedicated_malloc(nbytes + ALIGN, IS_HIP)
        self.base = self.ptr + (-self.ptr) % ALIGN
        self.nbytes = nbytes
        self.regions = register_destination(ring, self.base, nbytes, fd, True, IS_HIP)

    def read(self, ring, fd, file_off, dest_off, length):
        """Queue reads of [file_off, +length) into [dest_off, +length)."""
        uds = []
        while length:
            addr = self.base + dest_off
            reg = next(r for r in self.regions if r.dev_start <= addr < r.dev_end)
            n = min(length, reg.dev_end - addr)
            ud = ring.read_fixed(
                fd, file_off, reg.slot, reg.buf_off + addr - reg.dev_start, n
            )
            uds.append((ud, n))
            file_off, dest_off, length = file_off + n, dest_off + n, length - n
        return uds

    def unregister(self, ring):
        err = release_regions(ring, self.regions)
        if err is not None:
            raise err

    def free(self):
        release_dedicated(self.ptr)


class RangeLayer:
    """One layer's runs read into one buffer: w13 slots, then w2 slots."""

    def __init__(self, ring, fd, lay, runs):
        self.lay, self.runs, self.fd = lay, runs, fd
        self.n = sum(c for _, c in runs)
        self.b13, self.b2 = lay["w13_bytes"], lay["w2_bytes"]
        self.w2_base = self.n * self.b13
        self.buf = Buf(ring, self.n * (self.b13 + self.b2), fd)
        self.experts = [e for s, c in runs for e in lay["order"][s : s + c]]
        self.uds = []

    def issue(self, ring):
        lay, b13, b2 = self.lay, self.b13, self.b2
        slot = 0
        for start, count in self.runs:
            self.uds += self.buf.read(
                ring, self.fd, lay["w13_offset"] + start * b13, slot * b13, count * b13
            )
            self.uds += self.buf.read(
                ring,
                self.fd,
                lay["w2_offset"] + start * b2,
                self.w2_base + slot * b2,
                count * b2,
            )
            slot += count

    def wait(self, ring):
        for ud, n in self.uds:
            res = ring.wait(ud)
            if res < 0:
                raise OSError(-res, os.strerror(-res))
            if res != n:
                raise IOError(f"short read: {res} of {n} bytes")

    def unregister(self, ring):
        self.buf.unregister(ring)

    def free(self):
        self.buf.free()


def single_lays(single_dir, layout, key, which):
    """Per-layer read plans for a pool or delta in the single-copy file."""
    out = []
    for li in range(len(layout["layers"])):
        lay = dict(layout["layers"][str(li)])
        lay["path"] = os.path.join(single_dir, lay["file"])
        lay["runs"] = lay[which][key]
        out.append(lay)
    return out


def packed_lays(files, layout):
    """Per-layer read plans for packed stacked files: one run per region."""
    out = []
    for li, path in enumerate(files):
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        with open(path + ".json") as f:
            ids = json.load(f)["routed_expert_ids"]
        start = 8 + hlen
        ref = layout["layers"][str(li)]
        w13 = header[stacked_name(li, "w13")]["data_offsets"]
        w2 = header[stacked_name(li, "w2")]["data_offsets"]
        assert w13[1] - w13[0] == len(ids) * ref["w13_bytes"]
        assert w2[1] - w2[0] == len(ids) * ref["w2_bytes"]
        out.append(
            {
                "path": path,
                "order": ids,
                "runs": [[0, len(ids)]],
                "w13_offset": start + w13[0],
                "w2_offset": start + w2[0],
                "w13_bytes": ref["w13_bytes"],
                "w2_bytes": ref["w2_bytes"],
                "sha256": ref["sha256"],
            }
        )
    return out


def range_load(ring, lays, inflight=2, persistent=False):
    """Load runs for every layer; return (seconds, bytes, reads, layers).

    Cold (default): allocation, registration, reads and unregistration
    are all on the clock, two layers in flight, like the packed loader.
    Persistent: every layer's buffer is allocated and registered before
    the clock starts, as for slots a server keeps registered; only the
    reads are timed, all layers queued at once.
    """
    fds = {}
    for lay in lays:
        if lay["path"] not in fds:
            fds[lay["path"]] = os.open(lay["path"], os.O_RDONLY | os.O_DIRECT)
    done, pending = [], []
    if persistent:
        layers = [RangeLayer(ring, fds[l["path"]], l, l["runs"]) for l in lays]
        t0 = time.perf_counter()
        for rl in layers:
            rl.issue(ring)
            ring.submit()
        for rl in layers:
            rl.wait(ring)
        dt = time.perf_counter() - t0
        for rl in layers:
            rl.unregister(ring)
        done = layers
    else:
        t0 = time.perf_counter()
        for lay in lays:
            rl = RangeLayer(ring, fds[lay["path"]], lay, lay["runs"])
            rl.issue(ring)
            ring.submit()
            pending.append(rl)
            if len(pending) >= inflight:
                old = pending.pop(0)
                old.wait(ring)
                old.unregister(ring)
                done.append(old)
        for old in pending:
            old.wait(ring)
            old.unregister(ring)
            done.append(old)
        dt = time.perf_counter() - t0
    for fd in fds.values():
        os.close(fd)
    nbytes = sum(rl.n * (rl.b13 + rl.b2) for rl in done)
    nreads = sum(len(rl.uds) for rl in done)
    return dt, nbytes, nreads, done


def packed_load(files):
    t0 = time.perf_counter()
    ld = PoolLoader(files, inflight=2)
    ld.start()
    ld.join()
    dt = time.perf_counter() - t0
    for info in ld.load_info:
        assert not info["fallback"] and info["copier_class"].lower().startswith("dmab")
    nbytes = sum(os.path.getsize(f) for f in files)
    ld.close()
    return dt, nbytes


def host_bytes(buf, off, n):
    t = torch.empty(n, dtype=torch.uint8, device="cuda")
    with device_scope(0, IS_HIP):
        device_memcpy(t.data_ptr(), buf.base + off, n, IS_HIP)
    torch.cuda.synchronize()
    return t.cpu().numpy().tobytes()


def verify(layers, layout):
    bad = checked = 0
    for rl in layers:
        lay = rl.lay
        for slot, e in enumerate(rl.experts):
            for which, off, nb in (
                ("w13", slot * rl.b13, rl.b13),
                ("w2", rl.w2_base + slot * rl.b2, rl.b2),
            ):
                h = hashlib.sha256(host_bytes(rl.buf, off, nb)).hexdigest()
                bad += h != lay["sha256"][f"{which}:{e}"]
                checked += 1
    return checked, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("single_dir")
    ap.add_argument("packed_dir")
    ap.add_argument("out")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pools", nargs="+", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.single_dir, "LAYOUT.json")) as f:
        layout = json.load(f)
    pools = args.pools or layout["pools"]
    nl = len(layout["layers"])
    torch.zeros(1, device="cuda")
    ring = IoUring(entries=256, buffer_slots=64)

    def pfiles(name):
        return [
            os.path.join(args.packed_dir, f"{name}-r32-a4k", f"layer{li}.safetensors")
            for li in range(nl)
        ]

    def dfiles(a, b):
        return [
            os.path.join(
                args.packed_dir, "delta", f"{a}-to-{b}", f"layer{li}.safetensors"
            )
            for li in range(nl)
        ]

    arms = [("pool", p, None) for p in pools]
    arms += [("delta", a, b) for a in pools for b in pools if a != b]
    rows = []
    for t in range(args.trials):
        for kind, a, b in arms:
            names = (
                "packed",
                "packed-ranges",
                "packed-registered",
                "single",
                "single-registered",
            )
            order = names if t % 2 == 0 else names[::-1]
            files = pfiles(a) if kind == "pool" else dfiles(a, b)
            key = a if kind == "pool" else f"{a}->{b}"
            which = "pool_runs" if kind == "pool" else "delta_runs"
            for layout_name in order:
                if layout_name == "packed":
                    dt, nbytes = packed_load(files)
                    nreads = None
                else:
                    if layout_name.startswith("packed"):
                        lays = packed_lays(files, layout)
                    else:
                        lays = single_lays(args.single_dir, layout, key, which)
                    dt, nbytes, nreads, done = range_load(
                        ring, lays, persistent=layout_name.endswith("registered")
                    )
                    for rl in done:
                        rl.free()
                torch.cuda.empty_cache()
                row = {
                    "trial": t,
                    "kind": kind,
                    "pool": a,
                    "to": b,
                    "layout": layout_name,
                    "seconds": dt,
                    "bytes": nbytes,
                    "reads": nreads,
                    "GBps": nbytes / dt / 1e9,
                }
                rows.append(row)
                print(
                    f"trial {t} {kind:5} {a}{'->' + b if b else ''} {layout_name:17} "
                    f"{dt:.3f} s {nbytes / 1e9:.2f} GB {row['GBps']:.2f} GB/s"
                    + (f" reads {nreads}" if nreads else ""),
                    flush=True,
                )
    checks = {}
    for kind, a, b in arms:
        key = a if kind == "pool" else f"{a}->{b}"
        which = "pool_runs" if kind == "pool" else "delta_runs"
        files = pfiles(a) if kind == "pool" else dfiles(a, b)
        for name, lays in (
            ("single", single_lays(args.single_dir, layout, key, which)),
            ("packed", packed_lays(files, layout)),
        ):
            _, _, _, done = range_load(ring, lays)
            checked, bad = verify(done, layout)
            for rl in done:
                rl.free()
            checks[f"{name}:{key}"] = {"checked": checked, "mismatches": bad}
            print(
                f"verify {name} {key}: {checked} slices, {bad} mismatches", flush=True
            )
    ring.close()
    write_json(
        args.out,
        {
            "runtime": runtime_identity(),
            "args": vars(args),
            "rows": rows,
            "checks": checks,
        },
    )


if __name__ == "__main__":
    main()
