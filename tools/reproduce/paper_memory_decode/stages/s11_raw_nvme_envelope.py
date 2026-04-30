"""Stage 11 (storage profile): raw NVMe envelope benchmark.

This stage is NOT a paper reproduction claim.  It measures raw
open/write/fsync and read throughput for the storage device under
test, independent of any codec or LMCache API overhead.

Purpose: let a colleague running Samsung drives separate
device-bound limits from LMCache/fsync/codec overhead.

Pass criterion: none.  All results are informational.
Stage status is always "passed" if it runs without error.

Results written to stage_dir/raw_nvme_envelope.json.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import torch

from ..stages import StageContext, StageResult

N_WARMUP = 3


def _drop_cache() -> None:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass


def _kv_to_bytes(t: torch.Tensor) -> bytes:
    return bytes(t.contiguous().view(torch.uint8).numpy().tobytes())


def _make_data(target_bytes: int) -> bytes:
    n = max(1, target_bytes // 2)
    t = torch.randn(n // 2, dtype=torch.float16)
    b = _kv_to_bytes(t)
    # Pad or truncate to target_bytes.
    if len(b) < target_bytes:
        b = b * (target_bytes // len(b) + 1)
    return b[:target_bytes]


def _bench_raw(data: bytes, bench_dir: str, label: str, n_iters: int) -> dict:
    fpath = os.path.join(bench_dir, f"raw_{label}.bin")
    n_bytes = len(data)

    for _ in range(N_WARMUP):
        with open(fpath, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    lats_w = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        with open(fpath, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        lats_w.append((time.perf_counter() - t0) * 1e3)

    _drop_cache()
    lats_r = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        with open(fpath, "rb") as f:
            _ = f.read()
        lats_r.append((time.perf_counter() - t0) * 1e3)

    try:
        os.unlink(fpath)
    except OSError:
        pass

    def p(lst, pct):
        lst = sorted(lst)
        return lst[min(int(len(lst) * pct / 100), len(lst) - 1)]

    def mbps(nb, lats):
        avg_s = sum(lats) / len(lats) / 1e3
        return (nb / 1e6) / avg_s if avg_s > 0 else 0.0

    return {
        "label": label,
        "bytes": n_bytes,
        "write_p50_ms": p(lats_w, 50),
        "write_p95_ms": p(lats_w, 95),
        "read_p50_ms": p(lats_r, 50),
        "read_p95_ms": p(lats_r, 95),
        "write_MBps": mbps(n_bytes, lats_w),
        "read_MBps": mbps(n_bytes, lats_r),
    }


def run(ctx: StageContext) -> StageResult:
    cfg_nvme = getattr(ctx.cfg, "nvme_path", "") or ""
    env_nvme = os.environ.get("KNLP_NVME_PATH", "")
    nvme_root = cfg_nvme or env_nvme
    if nvme_root:
        bench_dir = os.path.join(nvme_root, "s11_raw")
        using_real_nvme = True
    else:
        bench_dir = str(ctx.stage_dir / "bench_raw")
        using_real_nvme = False
    os.makedirs(bench_dir, exist_ok=True)

    device_label = getattr(ctx.cfg, "storage_device_label", "") or "unknown"
    print(
        f"  bench_dir={bench_dir}  real_nvme={using_real_nvme}"
        f"  device={device_label}",
        flush=True,
    )

    chunk_sizes = [
        ("256KB", 256 * 1024, 64),
        ("1MB", 1 * 1024 * 1024, 64),
        ("4MB", 4 * 1024 * 1024, 32),
        ("8MB", 8 * 1024 * 1024, 32),
        ("32MB", 32 * 1024 * 1024, 16),
        ("64MB", 64 * 1024 * 1024, 16),
        ("256MB", 256 * 1024 * 1024, 8),
    ]

    rows = []
    for label, target, n_iters in chunk_sizes:
        data = _make_data(target)
        print(
            f"  [{label}] {len(data)/1e6:.1f} MB  {n_iters} iters ...",
            end=" ",
            flush=True,
        )
        try:
            r = _bench_raw(data, bench_dir, label, n_iters)
            r["device_label"] = device_label
            rows.append(r)
            print(
                f"w p50={r['write_p50_ms']:.1f} ms  "
                f"r p50={r['read_p50_ms']:.1f} ms  "
                f"w={r['write_MBps']:.1f}  r={r['read_MBps']:.1f} MB/s"
            )
            ctx.log_metric(f"write_MBps_{label}", r["write_MBps"])
            ctx.log_metric(f"read_MBps_{label}", r["read_MBps"])
            ctx.log_metric(f"write_p50_ms_{label}", r["write_p50_ms"])
            ctx.log_metric(f"read_p50_ms_{label}", r["read_p50_ms"])
        except Exception as e:
            import traceback

            msg = traceback.format_exc()
            ctx.stderr_path.open("a").write(f"WARN [{label}]: {e}\n{msg}\n")
            print(f"ERROR: {e}", flush=True)
            rows.append({"label": label, "error": str(e)})

    shutil.rmtree(bench_dir, ignore_errors=True)

    result_path = ctx.stage_dir / "raw_nvme_envelope.json"
    payload = {
        "device_label": device_label,
        "using_real_nvme": using_real_nvme,
        "note": "raw file I/O — NOT a paper reproduction claim",
        "rows": rows,
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    ctx.telemetry.log_artifact(result_path, "raw_nvme_envelope")

    ctx.mark_done({"rows": len(rows), "device_label": device_label})
    return StageResult(name=ctx.name, status="passed")
