#!/usr/bin/env python3
"""Replay selected byte ranges of a checkpoint file with controlled I/O arms.

The replay answers: how many file-read bytes (F) and how much wall time does
it take to bring a given set of tensor byte ranges into host memory, under a
named read path and a verified host-page-cache state? It records the file's
physical extents so a device-level NVMe trace can be attributed to the same
ranges by logical block address.

Arms:
  pread    buffered pread of each range (POSIX read path, page cache)
  mmap     map the file and touch one byte per 4 KiB page of each range
  direct   O_DIRECT pread with ranges rounded out to the alignment unit;
           padded bytes are counted in F separately from useful bytes

Cache states:
  cold     posix_fadvise(DONTNEED) on the file, then verify with mincore
           that the target pages are not resident; residual residency is
           recorded, never hidden
  warm     read the ranges once with buffered pread, then verify residency

Each trial records monotonic nanosecond start/end so a tracer using the
same clock can be joined by time window. The output is one JSON document.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import mmap
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAGE = 4096
libc = ctypes.CDLL(None, use_errno=True)


# ----------------------------------------------------------------------
# residency via mincore
# ----------------------------------------------------------------------


def resident_pages(
    fd: int, ranges: list[tuple[int, int]], file_size: int
) -> tuple[int, int]:
    """Return (resident_pages, total_pages) over the page-rounded ranges."""
    if not ranges:
        return 0, 0
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    map_len = file_size + (-file_size % PAGE)
    base = libc.mmap(None, map_len, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if base in (None, ctypes.c_void_p(-1).value):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    res = tot = 0
    try:
        for start, length in ranges:
            s = start - (start % PAGE)
            e = start + length
            e = e + (-e % PAGE)
            n = (e - s) // PAGE
            vec = (ctypes.c_ubyte * n)()
            rc = libc.mincore(ctypes.c_void_p(base + s), ctypes.c_size_t(e - s), vec)
            if rc != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
            res += sum(1 for v in vec if v & 1)
            tot += n
    finally:
        libc.munmap(base, map_len)
    return res, tot


# ----------------------------------------------------------------------
# FIEMAP extents
# ----------------------------------------------------------------------

FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 0x1


def fiemap(fd: int, file_size: int) -> list[dict]:
    """Return the file's extents as logical/physical byte ranges."""
    extents = []
    n = 512
    hdr = struct.Struct("=QQIIII")
    ext = struct.Struct("=QQQQQIII")
    start = 0
    while True:
        buf = bytearray(hdr.size + n * ext.size)
        hdr.pack_into(buf, 0, start, file_size - start, FIEMAP_FLAG_SYNC, 0, n, 0)
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
        _, _, _, mapped, _, _ = hdr.unpack_from(buf, 0)
        last = None
        for i in range(mapped):
            lo, po, ln, _, _, flags, _, _ = ext.unpack_from(
                buf, hdr.size + i * ext.size
            )
            e = {"logical": lo, "physical": po, "length": ln, "flags": flags}
            extents.append(e)
            last = e
        if mapped < n or last is None or last["flags"] & 1:
            break
        start = last["logical"] + last["length"]
    return extents


def device_lbas(
    extents: list[dict],
    ranges: list[tuple[int, int]],
    part_start_sectors: int,
    sector=512,
) -> list[tuple[int, int]]:
    """Map file byte ranges to absolute device sector ranges [slba, nsectors]."""
    out = []
    for start, length in ranges:
        a, b = start, start + length
        for e in extents:
            lo, ln = e["logical"], e["length"]
            if b <= lo or a >= lo + ln:
                continue
            s = max(a, lo)
            t = min(b, lo + ln)
            phys = e["physical"] + (s - lo)
            out.append(
                (part_start_sectors + phys // sector, (t - s + sector - 1) // sector)
            )
    return out


# ----------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------


def arm_pread(fd: int, ranges, qd: int):
    def one(r):
        start, length = r
        got = 0
        while got < length:
            b = os.pread(fd, min(length - got, 64 << 20), start + got)
            if not b:
                break
            got += len(b)
        return got

    if qd <= 1:
        return sum(one(r) for r in ranges)
    with ThreadPoolExecutor(max_workers=qd) as ex:
        return sum(ex.map(one, ranges))


def arm_mmap(fd: int, ranges, file_size: int, qd: int):
    mm = mmap.mmap(fd, file_size, prot=mmap.PROT_READ)
    touched = 0
    for start, length in ranges:
        s = start - (start % PAGE)
        e = start + length
        acc = 0
        for off in range(s, e, PAGE):
            acc += mm[off]
            touched += 1
    mm.close()
    return touched * PAGE


def arm_direct(fd: int, ranges, align: int, qd: int):
    def one(r):
        start, length = r
        s = start - (start % align)
        e = start + length
        e = e + (-e % align)
        n = e - s
        buf = mmap.mmap(-1, n)  # page-aligned anonymous buffer
        got = 0
        while got < n:
            k = os.preadv(
                fd, [memoryview(buf)[got : got + min(n - got, 64 << 20)]], s + got
            )
            if k <= 0:
                break
            got += k
        buf.close()
        return got

    if qd <= 1:
        return sum(one(r) for r in ranges)
    with ThreadPoolExecutor(max_workers=qd) as ex:
        return sum(ex.map(one, ranges))


# ----------------------------------------------------------------------


def load_ranges(args) -> list[tuple[int, int]]:
    if args.ranges:
        return [tuple(x) for x in json.loads(args.ranges)]
    rows = []
    with open(args.selection) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if (
                    r.get("repo") == args.repo
                    and r.get("unit") == args.unit
                    and Path(r["file"]).name == Path(args.file).name
                ):
                    rows.append((r["abs_start"], r["nbytes"]))
    if not rows:
        sys.exit("no ranges selected")
    return rows


def chunk_ranges(ranges, chunk: int):
    """Split ranges into contiguous chunks of at most `chunk` bytes."""
    if not chunk:
        return ranges
    out = []
    for s, n in ranges:
        off = 0
        while off < n:
            k = min(chunk, n - off)
            out.append((s + off, k))
            off += k
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", required=True)
    ap.add_argument("--selection")
    ap.add_argument("--repo")
    ap.add_argument("--unit")
    ap.add_argument("--ranges", help="JSON list of [start, length]")
    ap.add_argument("--arm", choices=["pread", "mmap", "direct"], required=True)
    ap.add_argument("--cache", choices=["cold", "warm"], required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="split ranges into chunks of this many bytes",
    )
    ap.add_argument(
        "--qd", type=int, default=1, help="concurrent range reads (threads)"
    )
    ap.add_argument("--align", type=int, default=4096)
    ap.add_argument("--part-start-sectors", type=int, required=True)
    ap.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="seconds to sleep before and after each trial",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ranges = sorted(load_ranges(args))
    ranges = chunk_ranges(ranges, args.chunk)
    useful = sum(n for _, n in ranges)
    st = os.stat(args.file)
    fd0 = os.open(args.file, os.O_RDONLY)
    extents = fiemap(fd0, st.st_size)
    lbas = device_lbas(extents, ranges, args.part_start_sectors)
    flags = os.O_RDONLY | (os.O_DIRECT if args.arm == "direct" else 0)
    fd = os.open(args.file, flags)

    trials = []
    for t in range(args.trials):
        if args.cache == "cold":
            resid = None
            for attempt in range(4):
                os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
                resid = resident_pages(fd0, ranges, st.st_size)
                if resid[0] == 0:
                    break
                time.sleep(0.2)
        else:
            arm_pread(fd0, ranges, 1)
            resid = resident_pages(fd0, ranges, st.st_size)
        time.sleep(args.settle)
        t0 = time.monotonic_ns()
        if args.arm == "pread":
            got = arm_pread(fd, ranges, args.qd)
        elif args.arm == "mmap":
            got = arm_mmap(fd, ranges, st.st_size, args.qd)
        else:
            got = arm_direct(fd, ranges, args.align, args.qd)
        t1 = time.monotonic_ns()
        time.sleep(args.settle)
        t2 = time.monotonic_ns()
        after = resident_pages(fd0, ranges, st.st_size)
        trials.append(
            {
                "trial": t,
                "t_start_ns": t0,
                "t_end_ns": t1,
                "t_drain_end_ns": t2,
                "elapsed_ns": t1 - t0,
                "F_bytes": got,
                "F_kind": (
                    "touched_pages"
                    if args.arm == "mmap"
                    else ("padded_direct" if args.arm == "direct" else "read_return")
                ),
                "resident_pages_before": resid[0],
                "resident_pages_after": after[0],
                "target_pages": resid[1],
            }
        )
        print(
            f"trial {t}: {args.arm}/{args.cache} F={got} useful={useful} {(t1 - t0) / 1e6:.3f} ms "
            f"resident_before={resid[0]}/{resid[1]} after={after[0]}",
            flush=True,
        )
    os.close(fd)
    os.close(fd0)
    doc = {
        "file": args.file,
        "file_size": st.st_size,
        "inode": st.st_ino,
        "label": args.label,
        "repo": args.repo,
        "unit": args.unit,
        "arm": args.arm,
        "cache": args.cache,
        "qd": args.qd,
        "chunk": args.chunk,
        "align": args.align,
        "n_ranges": len(ranges),
        "W_required_bytes": useful,
        "ranges": ranges,
        "extents": extents,
        "device_lbas": lbas,
        "part_start_sectors": args.part_start_sectors,
        "trials": trials,
    }
    Path(args.out).write_text(json.dumps(doc))


if __name__ == "__main__":
    main()
