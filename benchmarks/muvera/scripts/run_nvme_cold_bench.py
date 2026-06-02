#!/usr/bin/env python3
"""run_nvme_cold_bench.py — cold NVMe / O_DIRECT byte-floor benchmark.

Demonstrates that sub-4096-byte logical records suffer read amplification
when the access path is raw NVMe random reads, because storage operates
at block/page granularity and latency dominates.

Modes:
  A. page-cache cold pread()       — fadvise DONTNEED between random reads
  B. O_DIRECT 4 KiB aligned reads  — explicitly measures read amplification
  C. (optional) io_uring batched   — only if liburing-python is installed

Reports physical_read_bytes, read_amplification, latency p50/p95/p99,
IOPS, and effective logical/physical GB/s.
"""
from __future__ import annotations

import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np

import _muvera_config as cf

PAGE = 4096


def fadvise_dontneed(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception: pass


def gen_large_file(path: Path, total_bytes: int):
    if path.exists() and path.stat().st_size == total_bytes:
        return
    print(f"  generating {total_bytes/1e9:.1f} GB packed file at {path}…")
    rng = np.random.default_rng(20260502)
    chunk = 1 << 26  # 64 MiB
    with open(path, "wb") as f:
        rem = total_bytes
        while rem > 0:
            n = min(chunk, rem)
            f.write(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
            rem -= n


def percentiles(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean())}


def page_cache_cold_pread(path, b, n_records, n_ops=2000):
    """Random pread() with fadvise DONTNEED between reads. Measures
    cold-cache random access in the conventional buffered-IO path."""
    fd = os.open(path, os.O_RDONLY)
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, n_records, n_ops)
    lats = []
    try:
        for idx in idxs:
            offset = int(idx) * b
            # Best-effort cold: drop adjacent pages
            page = (offset // PAGE) * PAGE
            try: os.posix_fadvise(fd, page, PAGE * 8, os.POSIX_FADV_DONTNEED)
            except Exception: pass
            t0 = time.perf_counter_ns()
            buf = os.pread(fd, b, offset)
            lats.append(time.perf_counter_ns() - t0)
    finally:
        os.close(fd)
    return lats


def odirect_4kib_read(path, b, n_records, n_ops=2000):
    """O_DIRECT aligned 4 KiB reads — measures read amplification.
    For sub-4 KiB logical records, we still issue a 4 KiB physical read.
    Returns (lats, physical_bytes_per_op)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as e:
        print(f"  O_DIRECT not available on this filesystem: {e}")
        return None, None
    # Allocate aligned buffer. Use mmap for guaranteed page alignment.
    mm_buf = mmap.mmap(-1, PAGE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    rng = np.random.default_rng(1)
    idxs = rng.integers(0, n_records, n_ops)
    lats = []
    try:
        # Some filesystems require we pread from a page-aligned offset.
        # Round each record offset down to the containing 4 KiB page.
        for idx in idxs:
            logical_offset = int(idx) * b
            page_offset = (logical_offset // PAGE) * PAGE
            # Reset buffer position
            mm_buf.seek(0)
            t0 = time.perf_counter_ns()
            os.preadv(fd, [mm_buf], page_offset)
            lats.append(time.perf_counter_ns() - t0)
    finally:
        os.close(fd)
        mm_buf.close()
    return lats, PAGE


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH", False):
        print("CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=n; skipping (set to y to enable)")
        return

    run_dir = cf.get_run_dir(cfg)
    cf.stamp_environment(cfg, run_dir)

    bench_path_str = cfg.get("CONFIG_MUVERA_NVME_BENCH_PATH") or "/mnt/nvme/muvera_bench"
    bench_path = Path(bench_path_str)
    bench_path.mkdir(parents=True, exist_ok=True)
    large_gb = int(cfg.get("CONFIG_MUVERA_NVME_LARGE_GB", 128))
    total_bytes = large_gb * (1 << 30)
    n_random = int(cfg.get("CONFIG_MUVERA_NVME_RANDOM_READS", 100000))
    sizes = cf.parse_int_list(cfg.get("CONFIG_MUVERA_RECORD_SIZES",
        "32 64 128 256 512 1024 2048 4096 8192 16384"))

    big_file = bench_path / "muvera_bench_large.bin"
    print(f"=== NVMe cold bench: {large_gb} GB at {big_file} ===")
    gen_large_file(big_file, total_bytes)
    rows = []
    for b in sizes:
        if b > total_bytes: continue
        n_records = total_bytes // b
        print(f"\n--- record_bytes = {b}  (n_records = {n_records:.0f}) ---")

        # A. page-cache cold pread
        try:
            cold_lats = page_cache_cold_pread(big_file, b, n_records,
                                                n_ops=min(n_random, 2000))
            cp = percentiles(cold_lats)
            iops = 1e9 / max(cp["mean"], 1)
            log_gbs = b / (cp["mean"] * 1e-9) / 1e9
            phys_gbs = log_gbs  # buffered IO, exact read of b bytes
            print(f"  cold pread:  p50 {cp['p50']/1e3:.1f} us  p99 {cp['p99']/1e3:.1f}  iops {iops:.0f}")
            rows.append({"mode": "cold_pread",
                         "logical_record_bytes": b, "physical_read_bytes": b,
                         "read_amplification": 1.0,
                         "lat_p50_us": cp["p50"]/1e3, "lat_p95_us": cp["p95"]/1e3,
                         "lat_p99_us": cp["p99"]/1e3, "iops": iops,
                         "effective_logical_gbps": log_gbs,
                         "effective_physical_gbps": phys_gbs,
                         "notes": "fadvise DONTNEED, buffered pread"})
        except Exception as e:
            print(f"  cold pread error: {e}")

        # B. O_DIRECT 4 KiB
        odir = odirect_4kib_read(big_file, b, n_records, n_ops=min(n_random, 2000))
        if odir is not None and odir[0] is not None:
            lats, phys_bytes = odir
            op = percentiles(lats)
            iops = 1e9 / max(op["mean"], 1)
            log_gbs = b / (op["mean"] * 1e-9) / 1e9
            phys_gbs = phys_bytes / (op["mean"] * 1e-9) / 1e9
            ra = phys_bytes / b
            print(f"  O_DIRECT 4K: p50 {op['p50']/1e3:.1f} us  p99 {op['p99']/1e3:.1f}  "
                  f"iops {iops:.0f}  read_amp {ra:.1f}x")
            rows.append({"mode": "odirect_4kib",
                         "logical_record_bytes": b, "physical_read_bytes": phys_bytes,
                         "read_amplification": ra,
                         "lat_p50_us": op["p50"]/1e3, "lat_p95_us": op["p95"]/1e3,
                         "lat_p99_us": op["p99"]/1e3, "iops": iops,
                         "effective_logical_gbps": log_gbs,
                         "effective_physical_gbps": phys_gbs,
                         "notes": "O_DIRECT page-aligned, single read"})
        else:
            rows.append({"mode": "odirect_4kib",
                         "logical_record_bytes": b, "physical_read_bytes": "-",
                         "read_amplification": "-",
                         "lat_p50_us": "-", "lat_p95_us": "-", "lat_p99_us": "-",
                         "iops": "-",
                         "effective_logical_gbps": "-",
                         "effective_physical_gbps": "-",
                         "notes": "O_DIRECT not supported on this fs"})

    cols = ["mode", "logical_record_bytes", "physical_read_bytes",
             "read_amplification", "lat_p50_us", "lat_p95_us", "lat_p99_us",
             "iops", "effective_logical_gbps", "effective_physical_gbps", "notes"]
    csv_path = run_dir / "nvme_cold_byte_floor.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
