#!/usr/bin/env python3
"""run_systems_byte_bench.py — systems byte-floor microbench.

Measures:
  A. mmap warm random point reads
  B. mmap batched random reads
  C. top-k scatter pattern
  D. sequential per-document read (cold + warm)
  E. CPU → GPU H2D transfer
  F. GPU projection / expansion

Sweeps record sizes from CONFIG_MUVERA_RECORD_SIZES. Writes
systems_byte_floor.csv in the run dir.
"""
from __future__ import annotations

import argparse
import gc
import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

import _muvera_config as cf


def fadvise_dontneed(path):
    """Best-effort drop of file from OS page cache."""
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass


def gen_packed_file(path: Path, bytes_per_record: int, n_records: int):
    total = bytes_per_record * n_records
    if path.exists() and path.stat().st_size == total:
        return
    rng = np.random.default_rng(20260502)
    # Generate in chunks to avoid huge in-memory tensors for big files
    chunk = max(1, 1 << 26)  # 64 MiB
    with open(path, "wb") as f:
        remaining = total
        while remaining > 0:
            n = min(chunk, remaining)
            f.write(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
            remaining -= n


def percentiles(arr_ns):
    a = np.asarray(arr_ns, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max())}


def time_sequential_read(path, total_bytes, n_runs=3):
    cold = []
    for _ in range(n_runs):
        fadvise_dontneed(path)
        t0 = time.perf_counter_ns()
        with open(path, "rb") as f: _ = f.read()
        cold.append(time.perf_counter_ns() - t0)
    warm = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        with open(path, "rb") as f: _ = f.read()
        warm.append(time.perf_counter_ns() - t0)
    return cold, warm


def time_random_point(path, b, n_records, n_ops):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        rng = np.random.default_rng(0)
        idxs = rng.integers(0, n_records, n_ops)
        lats = []
        for idx in idxs:
            offset = int(idx) * b
            t0 = time.perf_counter_ns()
            buf = mm[offset:offset + b]
            _ = buf[0]
            lats.append(time.perf_counter_ns() - t0)
        mm.close()
    return lats


def time_batched_random(path, b, n_records, batch, n_ops):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        rng = np.random.default_rng(1)
        lats = []
        for _ in range(n_ops):
            idxs = rng.integers(0, n_records, batch)
            t0 = time.perf_counter_ns()
            for idx in idxs:
                _ = bytes(mm[int(idx) * b:int(idx) * b + b])
            lats.append(time.perf_counter_ns() - t0)
        mm.close()
    return lats


def time_topk_scatter(path, b, n_records, k, n_ops):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        rng = np.random.default_rng(2)
        stride = max(1, n_records // (k * 4))
        lats = []
        for _ in range(n_ops):
            base = int(rng.integers(0, n_records - k * stride))
            idxs = [base + i * stride for i in range(k)]
            t0 = time.perf_counter_ns()
            for idx in idxs:
                _ = bytes(mm[idx * b:idx * b + b])
            lats.append(time.perf_counter_ns() - t0)
        mm.close()
    return lats


def time_h2d(b, batch, n_ops, device):
    try:
        import torch
        if not torch.cuda.is_available(): return None
    except ImportError:
        return None
    dim_fp16 = b // 2
    if dim_fp16 == 0: return None
    src = torch.empty((batch, dim_fp16), dtype=torch.float16, pin_memory=True)
    src.uniform_(-1, 1)
    for _ in range(5):
        _ = src.to(device, non_blocking=True); torch.cuda.synchronize()
    lats = []
    for _ in range(n_ops):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        _ = src.to(device, non_blocking=True)
        torch.cuda.synchronize()
        lats.append(time.perf_counter_ns() - t0)
    return lats


def time_projection(b, batch, hidden, n_ops, device):
    try:
        import torch
        if not torch.cuda.is_available(): return None
    except ImportError:
        return None
    in_dim = b // 2
    if in_dim == 0: return None
    W = torch.randn(in_dim, hidden, device=device, dtype=torch.float16)
    src = torch.randn(batch, in_dim, device=device, dtype=torch.float16)
    for _ in range(5):
        _ = src @ W
    torch.cuda.synchronize()
    lats = []
    for _ in range(n_ops):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        _ = src @ W
        torch.cuda.synchronize()
        lats.append(time.perf_counter_ns() - t0)
    return lats


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_SYSTEMS_BENCH", True):
        print("CONFIG_MUVERA_SYSTEMS_BENCH=n; skipping")
        return

    run_dir = cf.get_run_dir(cfg)
    cf.stamp_environment(cfg, run_dir)

    sizes = cf.parse_int_list(cfg.get("CONFIG_MUVERA_RECORD_SIZES",
        "32 64 128 256 512 1024 2048 4096 8192 16384"))
    n_records = int(cfg.get("CONFIG_MUVERA_SYSTEMS_SMALL_RECORDS", 1000000))
    n_random = int(cfg.get("CONFIG_MUVERA_SYSTEMS_RANDOM_READS", 100000))
    topk = int(cfg.get("CONFIG_MUVERA_SYSTEMS_TOPK", 64))
    batches = cf.parse_int_list(cfg.get("CONFIG_MUVERA_SYSTEMS_BATCH_SIZES",
        "1 8 16 64 256"))
    hidden = int(cfg.get("CONFIG_MUVERA_SYSTEMS_PROJECTION_HIDDEN", 4096))
    device = cfg.get("CONFIG_MUVERA_DEVICE") or "cuda:0"

    workdir = run_dir / "systems_files"
    workdir.mkdir(exist_ok=True)
    rows = []
    for b in sizes:
        path = workdir / f"records_{b}.bin"
        total_bytes = b * n_records
        # Cap large files at a reasonable size to fit on prune
        # (16 KiB × 1 M = 16 GiB; the user can override n_records in .config)
        print(f"\n=== bytes_per_record = {b}  (file = {total_bytes/1e6:.1f} MB) ===")
        gen_packed_file(path, b, n_records)
        with open(path, "rb") as f: f.read()  # warm cache

        # 1. Sequential cold + warm
        cold, warm = time_sequential_read(path, total_bytes, n_runs=3)
        cold_p = percentiles(cold); warm_p = percentiles(warm)
        cold_gbs = total_bytes / (cold_p["mean"] * 1e-9) / 1e9
        warm_gbs = total_bytes / (warm_p["mean"] * 1e-9) / 1e9
        print(f"  seq:  cold {cold_p['mean']/1e6:.2f} ms ({cold_gbs:.2f} GB/s)  "
              f"warm {warm_p['mean']/1e6:.2f} ms ({warm_gbs:.2f} GB/s)")
        rows.append({"mode": "sequential_cold", "record_bytes": b, "num_records": n_records,
                     "batch_size": "-", "topk": "-", "total_file_bytes": total_bytes,
                     "lat_p50_ns": cold_p["p50"], "lat_p95_ns": cold_p["p95"],
                     "lat_p99_ns": cold_p["p99"],
                     "records_per_sec": n_records / (cold_p["mean"] * 1e-9),
                     "effective_gbps": cold_gbs, "notes": "fadvise DONTNEED"})
        rows.append({"mode": "sequential_warm", "record_bytes": b, "num_records": n_records,
                     "batch_size": "-", "topk": "-", "total_file_bytes": total_bytes,
                     "lat_p50_ns": warm_p["p50"], "lat_p95_ns": warm_p["p95"],
                     "lat_p99_ns": warm_p["p99"],
                     "records_per_sec": n_records / (warm_p["mean"] * 1e-9),
                     "effective_gbps": warm_gbs, "notes": "page cache hot"})

        # 2. Random point
        rand_lats = time_random_point(path, b, n_records, n_ops=min(n_random, 5000))
        p = percentiles(rand_lats)
        print(f"  random pt:  p50 {p['p50']:.0f} ns  p95 {p['p95']:.0f}  p99 {p['p99']:.0f}")
        rows.append({"mode": "mmap_random_point", "record_bytes": b, "num_records": n_records,
                     "batch_size": 1, "topk": "-", "total_file_bytes": total_bytes,
                     "lat_p50_ns": p["p50"], "lat_p95_ns": p["p95"], "lat_p99_ns": p["p99"],
                     "records_per_sec": 1e9 / max(p["mean"], 1),
                     "effective_gbps": b / (p["mean"] * 1e-9) / 1e9,
                     "notes": "mmap warm"})

        # 3. Batched random per batch size
        for bs in batches:
            n_ops = max(50, min(500, n_random // bs))
            blats = time_batched_random(path, b, n_records, bs, n_ops)
            bp = percentiles(blats)
            gbs = (b * bs) / (bp["mean"] * 1e-9) / 1e9
            rows.append({"mode": "mmap_batched_random", "record_bytes": b, "num_records": n_records,
                         "batch_size": bs, "topk": "-", "total_file_bytes": total_bytes,
                         "lat_p50_ns": bp["p50"], "lat_p95_ns": bp["p95"], "lat_p99_ns": bp["p99"],
                         "records_per_sec": bs * 1e9 / max(bp["mean"], 1),
                         "effective_gbps": gbs, "notes": f"mmap warm batch={bs}"})
        print(f"  batched random x{batches[-1]}: p50 {bp['p50']/1e3:.1f} us")

        # 4. Top-k scatter
        topk_lats = time_topk_scatter(path, b, n_records, topk, n_ops=200)
        tp = percentiles(topk_lats)
        rows.append({"mode": "mmap_topk_scatter", "record_bytes": b, "num_records": n_records,
                     "batch_size": "-", "topk": topk, "total_file_bytes": total_bytes,
                     "lat_p50_ns": tp["p50"], "lat_p95_ns": tp["p95"], "lat_p99_ns": tp["p99"],
                     "records_per_sec": topk * 1e9 / max(tp["mean"], 1),
                     "effective_gbps": (b * topk) / (tp["mean"] * 1e-9) / 1e9,
                     "notes": f"k={topk} stride scatter"})

        # 5. CPU → GPU H2D
        for bs in batches:
            h2d = time_h2d(b, bs, n_ops=200, device=device)
            if h2d is None: continue
            hp = percentiles(h2d)
            rows.append({"mode": "h2d_transfer", "record_bytes": b, "num_records": "-",
                         "batch_size": bs, "topk": "-", "total_file_bytes": "-",
                         "lat_p50_ns": hp["p50"], "lat_p95_ns": hp["p95"], "lat_p99_ns": hp["p99"],
                         "records_per_sec": bs * 1e9 / max(hp["mean"], 1),
                         "effective_gbps": (b * bs) / (hp["mean"] * 1e-9) / 1e9,
                         "notes": f"pinned CPU→{device} batch={bs}"})

        # 6. GPU projection
        for bs in batches:
            p_lats = time_projection(b, bs, hidden, n_ops=200, device=device)
            if p_lats is None: continue
            pp = percentiles(p_lats)
            rows.append({"mode": "gpu_projection", "record_bytes": b, "num_records": "-",
                         "batch_size": bs, "topk": "-", "total_file_bytes": "-",
                         "lat_p50_ns": pp["p50"], "lat_p95_ns": pp["p95"], "lat_p99_ns": pp["p99"],
                         "records_per_sec": bs * 1e9 / max(pp["mean"], 1),
                         "effective_gbps": "-",
                         "notes": f"linear in_dim={b//2} → hidden={hidden} fp16 batch={bs}"})

    # Persist
    cols = ["mode", "record_bytes", "num_records", "batch_size", "topk",
             "total_file_bytes", "lat_p50_ns", "lat_p95_ns", "lat_p99_ns",
             "records_per_sec", "effective_gbps", "notes"]
    csv_path = run_dir / "systems_byte_floor.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
