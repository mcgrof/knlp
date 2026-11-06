#!/usr/bin/env python3
"""Time each step in the life of an io_uring DMA-BUF registration (GPU).

The fastsafetensors DMA-BUF copier reads a file straight into GPU memory:
it exports the destination as a DMA-BUF, registers that DMA-BUF as an
io_uring fixed buffer for the file, issues READ_FIXED requests, and when
the reads finish it unregisters the buffer and closes the DMA-BUF before it
returns the tensors. An earlier one-layer test found 13-22 ms of that
release on the critical path for a 384 MiB registration. This script asks
whether the cost matters for ordinary weight loading, and where it comes
from.

Primitive mode times every step for one dedicated GPU allocation per size:
allocate, export, register, a first read of the whole buffer (the kernel
maps the DMA-BUF lazily on the first read), a second read into the same
registration, then release in one of two orders (unregister then close, as
the copier does, or close then unregister), then free. A "no read" variant
releases a registration that never did I/O, so it has no mapping to tear
down. After the first read, the first and last MiB of the buffer are
compared with the file.

Copier mode loads whole safetensors files one after another with the
unchanged copier, as a serving engine loads a model, and times the release
inside wait_io for each file.

Usage:
    dmabuf_release_cost.py OUT.json --file BIG_FILE [--sizes 64 256 ...]
        [--copier-files F1 F2 ...] [--trials 5]
"""

import argparse
import ctypes
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, write_json  # noqa

MiB = 1 << 20


def hip_d2h(dst, src, n):
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    rc = lib.hipMemcpy(dst, src, n, 2)  # hipMemcpyDeviceToHost
    if rc != 0:
        raise RuntimeError(f"hipMemcpy D2H failed: {rc}")


def read_all(ring, fd, slot, buf_off, size, io_bytes, qd):
    """Read [0, size) of fd into the registration; return seconds."""
    t0 = time.perf_counter()
    uds, off = [], 0
    while off < size:
        n = min(io_bytes, size - off)
        while ring.inflight + ring.pending >= qd:
            ring.submit(wait_nr=1)
        uds.append((ring.read_fixed(fd, off, slot, buf_off + off, n), n))
        off += n
    ring.submit()
    for ud, n in uds:
        res = ring.wait(ud)
        if res != n:
            raise RuntimeError(f"short or failed read: {res} != {n}")
    return time.perf_counter() - t0


def check_bytes(ptr, fd, size):
    ok = True
    for off in (0, size - MiB):
        host = ctypes.create_string_buffer(MiB)
        hip_d2h(ctypes.addressof(host), ptr + off, MiB)
        ok = ok and host.raw == os.pread(fd, MiB, off)
    return ok


def one_trial(io, ring, fd, size, order, do_read, io_bytes, qd):
    t = {}
    c = time.perf_counter
    s = c()
    ptr = io.dedicated_malloc(size, True)
    t["alloc_s"] = c() - s
    s = c()
    dfd, exp_off, dsize = io.export_device_range(ptr, size, True)
    t["export_s"] = c() - s
    t["export_offset"], t["dmabuf_bytes"] = exp_off, dsize
    s = c()
    slot = ring.register_dmabuf(dfd, fd)
    t["register_s"] = c() - s
    if do_read:
        t["read1_s"] = read_all(ring, fd, slot, exp_off, size, io_bytes, qd)
        t["bytes_ok"] = check_bytes(ptr, fd, size)
        t["read2_s"] = read_all(ring, fd, slot, exp_off, size, io_bytes, qd)
    if order == "unregister-first":
        s = c()
        ring.unregister(slot)
        t["unregister_s"] = c() - s
        s = c()
        os.close(dfd)
        t["close_s"] = c() - s
    else:
        s = c()
        os.close(dfd)
        t["close_s"] = c() - s
        s = c()
        ring.unregister(slot)
        t["unregister_s"] = c() - s
    t["release_s"] = t["unregister_s"] + t["close_s"]
    s = c()
    io.release_dedicated(ptr)
    t["free_s"] = c() - s
    return t


def primitive_mode(args):
    from fastsafetensors import io_uring_dmabuf as io

    fd = os.open(args.file, os.O_RDONLY | os.O_DIRECT)
    fsize = os.fstat(fd).st_size
    ring = io.IoUring(entries=2 * args.qd, buffer_slots=8)
    cells = []
    variants = [
        ("unregister-first", True),
        ("close-first", True),
        ("unregister-first", False),
    ]
    try:
        # One untimed warm-up trial, so first-use costs land outside the data.
        one_trial(
            io, ring, fd, 64 * MiB, "unregister-first", True, args.io_bytes, args.qd
        )
        for mib in args.sizes:
            size = mib * MiB
            assert size <= fsize, f"{args.file} is smaller than {mib} MiB"
            for order, do_read in variants:
                trials = [
                    one_trial(
                        io, ring, fd, size, order, do_read, args.io_bytes, args.qd
                    )
                    for _ in range(args.trials)
                ]
                med = {
                    k: statistics.median(tr[k] for tr in trials)
                    for k in trials[0]
                    if k.endswith("_s")
                }
                cell = {
                    "size_mib": mib,
                    "order": order,
                    "read": do_read,
                    "median": med,
                    "trials": trials,
                }
                if do_read:
                    cell["bytes_ok"] = all(tr["bytes_ok"] for tr in trials)
                    cell["read1_gbps"] = size / med["read1_s"] / 1e9
                    cell["read2_gbps"] = size / med["read2_s"] / 1e9
                cells.append(cell)
                print(
                    f"{mib} MiB {order} read={do_read}: "
                    f"unregister {med['unregister_s']*1e3:.2f} ms, "
                    f"close {med['close_s']*1e3:.2f} ms"
                    + (
                        f", read1 {med['read1_s']*1e3:.1f} ms, "
                        f"read2 {med['read2_s']*1e3:.1f} ms"
                        if do_read
                        else ""
                    ),
                    flush=True,
                )
    finally:
        ring.close()
        os.close(fd)
    return cells


def copier_mode(args):
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
    from fastsafetensors.allocation import SharedDeviceAllocation

    runs = []
    for trial in range(args.copier_trials):
        files = []
        t_all = time.perf_counter()
        for path in args.copier_files:
            loader = SafeTensorsFileLoader(
                SingleGroup(), "cuda:0", nogds=False, use_dmabuf=True
            )
            loader.add_filenames({0: [path]})
            meta, _ = next(iter(loader.meta.values()))
            t0 = time.perf_counter()
            cp = loader.copier_constructor(meta, loader.device, loader.framework)
            t1 = time.perf_counter()
            rel = {"s": 0.0}
            release = cp._release_registrations

            def timed_release(release=release, rel=rel):
                s = time.perf_counter()
                try:
                    return release()
                finally:
                    rel["s"] += time.perf_counter() - s

            cp._release_registrations = timed_release
            gbuf = cp.submit_io(True, 16 << 30)
            t2 = time.perf_counter()
            pieces = len(getattr(cp, "pieces", []) or [])
            owner = SharedDeviceAllocation(gbuf, loader.framework, loader.device)
            try:
                ts = cp.wait_io(gbuf, owner=owner)
            finally:
                owner.release()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            fallback = getattr(cp, "_fallback", None) is not None
            del ts
            loader.close()
            torch.cuda.empty_cache()
            files.append(
                {
                    "file": os.path.basename(path),
                    "bytes": os.path.getsize(path),
                    "construct_s": t1 - t0,
                    "submit_s": t2 - t1,
                    "wait_s": t3 - t2,
                    "release_in_wait_s": rel["s"],
                    "pieces": pieces,
                    "fallback": fallback,
                    "copier": type(cp).__name__,
                }
            )
        total = time.perf_counter() - t_all
        rsum = sum(f["release_in_wait_s"] for f in files)
        runs.append(
            {
                "trial": trial,
                "total_s": total,
                "bytes": sum(f["bytes"] for f in files),
                "release_sum_s": rsum,
                "release_fraction": rsum / total,
                "files": files,
            }
        )
        print(
            f"copier trial {trial}: {total:.2f} s total, release {rsum*1e3:.1f} ms "
            f"({100*rsum/total:.1f}%)",
            flush=True,
        )
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--file", required=True)
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 256, 384, 1024, 2048, 4094]
    )
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--io-bytes", type=int, default=64 * MiB)
    ap.add_argument("--qd", type=int, default=128)
    ap.add_argument("--copier-files", nargs="*", default=[])
    ap.add_argument("--copier-trials", type=int, default=3)
    args = ap.parse_args()
    torch.cuda.init()
    torch.empty(1, device="cuda")
    import fastsafetensors

    out = {
        "runtime": runtime_identity(),
        "fastsafetensors": os.path.dirname(fastsafetensors.__file__),
        "file": os.path.abspath(args.file),
        "io_bytes": args.io_bytes,
        "queue_depth": args.qd,
        "trials": args.trials,
        "env": {
            k: v for k, v in os.environ.items() if k.startswith("FASTSAFETENSORS_")
        },
    }
    out["primitive"] = primitive_mode(args)
    write_json(args.out, out)
    if args.copier_files:
        out["copier"] = copier_mode(args)
        write_json(args.out, out)


if __name__ == "__main__":
    main()
