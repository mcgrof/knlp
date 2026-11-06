#!/usr/bin/env python3
"""Time model weight loading through io_uring DMA-BUF and its release.

The fastsafetensors DMA-BUF copier reads a safetensors file straight
from NVMe into GPU memory. It exports the destination GPU memory as a
DMA-BUF, registers that DMA-BUF as an io_uring fixed buffer for the
file, issues READ_FIXED requests, and when the reads finish it
unregisters the buffer and closes the DMA-BUF before it hands the
tensors back. On kernels where unregistering waits for an RCU grace
period, that release adds a fixed delay to every file a model loads.
This script measures the load and the release.

Model mode (--model or --files) loads every safetensors file one after
another with the unchanged copier, as a serving engine loads a model,
and reports per file and per trial:

  wait_s             time to complete the reads and hand back tensors
  release_in_wait_s  the part of wait_s spent unregistering and closing
  total_s            the whole trial, all files
  release_sum_s      release_in_wait_s summed over the files
  read_sum_s         wait_s summed over the files, minus release_sum_s

Registration mode (--registration-file) times each step in the life of
one registration for a range of buffer sizes: allocate GPU memory,
export it, register it, read the file into it twice, release it in two
orders (unregister then close, as the copier does, or close then
unregister), and free the GPU memory. A third variant releases a
registration that never did I/O. After the first read the first and
last MiB of the buffer are compared with the file. unregister_s and
free_s show where the release cost lands.

The copier is in the 2026-10-28-iouring-dma-buf branch of
https://github.com/mcgrof/fastsafetensors. Files must live on XFS,
ext4 or a raw NVMe namespace. The script sets
FASTSAFETENSORS_DMABUF_STRICT=1 unless it is already set, so a refused
export or registration is an error instead of a silent fallback to host
reads. Tested on ROCm; the CUDA path of the byte check is untested.

Examples:
  dmabuf_release_bench.py --model /models/my-model --out load.json
  dmabuf_release_bench.py --registration-file /models/big.safetensors \\
      --out reg.json
"""

import argparse
import ctypes
import glob
import json
import os
import platform
import statistics
import sys
import time

os.environ.setdefault("FASTSAFETENSORS_DMABUF_STRICT", "1")
# ROCm's legacy KFD IPC interface is absent from in-tree amdgpu; select
# the DMA-BUF mode before HIP starts.
os.environ.setdefault("HSA_ENABLE_IPC_MODE_LEGACY", "0")

import torch  # noqa: E402

MiB = 1 << 20


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def runtime_identity():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "kernel": platform.release(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_arch"] = getattr(p, "gcnArchName", None) or (f"sm_{p.major}{p.minor}")
        info["gpu_total_bytes"] = p.total_memory
    return info


def device_to_host(dst, src, n):
    if torch.version.hip:
        lib, fn = ctypes.CDLL("libamdhip64.so"), "hipMemcpy"
    else:
        lib, fn = ctypes.CDLL("libcudart.so"), "cudaMemcpy"
    memcpy = getattr(lib, fn)
    memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    rc = memcpy(dst, src, n, 2)  # DeviceToHost in both runtimes
    if rc != 0:
        raise RuntimeError(f"{fn} device to host failed: {rc}")


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
        device_to_host(ctypes.addressof(host), ptr + off, MiB)
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


def registration_mode(args):
    from fastsafetensors import io_uring_dmabuf as io

    fd = os.open(args.registration_file, os.O_RDONLY | os.O_DIRECT)
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
            if size > fsize:
                print(f"skip {mib} MiB: the file has {fsize // MiB} MiB", flush=True)
                continue
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
                    f"close {med['close_s']*1e3:.2f} ms, "
                    f"free {med['free_s']*1e3:.2f} ms"
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


def model_mode(args, paths):
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
    from fastsafetensors.allocation import SharedDeviceAllocation

    runs = []
    for trial in range(args.model_trials):
        files = []
        t_all = time.perf_counter()
        for path in paths:
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
                "read_sum_s": sum(f["wait_s"] for f in files) - rsum,
                "release_fraction": rsum / total,
                "fallback_files": sum(f["fallback"] for f in files),
                "files": files,
            }
        )
        print(
            f"trial {trial}: {total:.3f} s total, reads {runs[-1]['read_sum_s']:.3f}"
            f" s, release {rsum*1e3:.1f} ms ({100*rsum/total:.1f}%)",
            flush=True,
        )
    return runs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--model", help="directory of *.safetensors files to load")
    ap.add_argument("--files", nargs="*", default=[], help="files to load, in order")
    ap.add_argument("--model-trials", type=int, default=3)
    ap.add_argument(
        "--registration-file",
        help="large file for the per-step registration timing",
    )
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 256, 384, 1024, 2048, 4094],
        help="registration sizes in MiB; sizes above the file size are skipped",
    )
    ap.add_argument("--trials", type=int, default=5, help="registration trials")
    ap.add_argument("--io-bytes", type=int, default=64 * MiB)
    ap.add_argument("--qd", type=int, default=128, help="registration queue depth")
    args = ap.parse_args()
    paths = list(args.files)
    if args.model:
        paths += sorted(glob.glob(os.path.join(args.model, "*.safetensors")))
    if not paths and not args.registration_file:
        ap.error("give --model, --files or --registration-file")
    torch.cuda.init()
    torch.empty(1, device="cuda")
    import fastsafetensors

    out = {
        "runtime": runtime_identity(),
        "fastsafetensors": os.path.dirname(fastsafetensors.__file__),
        "io_bytes": args.io_bytes,
        "queue_depth": args.qd,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("FASTSAFETENSORS_") or k == "HSA_ENABLE_IPC_MODE_LEGACY"
        },
    }
    if args.registration_file:
        out["registration_file"] = os.path.abspath(args.registration_file)
        out["registration"] = registration_mode(args)
        write_json(args.out, out)
    if paths:
        out["model_files"] = [os.path.abspath(p) for p in paths]
        out["model"] = model_mode(args, paths)
        runs = out["model"]
        out["model_median"] = {
            k: statistics.median(r[k] for r in runs)
            for k in ("total_s", "read_sum_s", "release_sum_s")
        }
        m = out["model_median"]
        print(
            f"median of {len(runs)}: {m['total_s']:.3f} s total, "
            f"reads {m['read_sum_s']:.3f} s, release {m['release_sum_s']*1e3:.1f} ms",
            flush=True,
        )
        write_json(args.out, out)


if __name__ == "__main__":
    main()
