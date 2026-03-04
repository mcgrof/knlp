"""Storage microbenchmarking for tier latencies."""

import os
import time

import numpy as np


def benchmark_path(
    path: str,
    block_bytes: int,
    num_files: int = 64,
    num_reads: int = 256,
    sequential: bool = True,
) -> dict:
    """Benchmark read latency and throughput for a storage path.

    Creates num_files temporary payload files of block_bytes size,
    then measures read latency over num_reads random or sequential reads.
    """
    os.makedirs(path, exist_ok=True)

    # Write test files
    file_paths = []
    payload = os.urandom(block_bytes)
    for i in range(num_files):
        fp = os.path.join(path, f"bench_block_{i:04d}.bin")
        with open(fp, "wb") as f:
            f.write(payload)
        file_paths.append(fp)

    # Flush caches
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except (PermissionError, FileNotFoundError):
        pass

    # Read benchmark
    rng = np.random.RandomState(42)
    latencies_us = []

    if sequential:
        indices = list(range(num_files)) * (num_reads // num_files + 1)
        indices = indices[:num_reads]
    else:
        indices = rng.randint(0, num_files, size=num_reads).tolist()

    for idx in indices:
        fp = file_paths[idx]
        t0 = time.perf_counter()
        with open(fp, "rb") as f:
            data = f.read()
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    # Cleanup
    for fp in file_paths:
        try:
            os.remove(fp)
        except OSError:
            pass

    arr = np.array(latencies_us)
    total_bytes = block_bytes * num_reads
    total_time_s = arr.sum() / 1e6

    return {
        "path": path,
        "block_bytes": block_bytes,
        "num_reads": num_reads,
        "sequential": sequential,
        "p50_us": float(np.median(arr)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_us": float(arr.mean()),
        "std_us": float(arr.std()),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
        "throughput_mb_s": (
            (total_bytes / 1e6) / total_time_s if total_time_s > 0 else 0
        ),
    }


def run_storage_microbench(
    tmpfs_path: str,
    sfs_path: str,
    block_bytes: int,
) -> dict:
    """Run the full storage microbench suite."""
    results = {}

    for label, path in [("tmpfs", tmpfs_path), ("sfs", sfs_path)]:
        path_results = {}
        for seq_label, seq in [("sequential", True), ("random", False)]:
            try:
                r = benchmark_path(path, block_bytes, sequential=seq)
                path_results[seq_label] = r
                print(
                    f"  {label}/{seq_label}: p50={r['p50_us']:.0f}us "
                    f"p95={r['p95_us']:.0f}us "
                    f"throughput={r['throughput_mb_s']:.1f}MB/s"
                )
            except Exception as e:
                path_results[seq_label] = {"error": str(e)}
                print(f"  {label}/{seq_label}: ERROR: {e}")
        results[label] = path_results

    return results
