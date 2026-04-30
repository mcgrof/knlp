"""Stage 10: LMCache split-tier storage microbench.

Measures the split-tier NVMe-traffic ratio and throughput for the
asymmetric K16/V8 codec across chunk sizes (1 MB, 8 MB, 32 MB, 64 MB).

Two policies are compared:

  ALL_NVME          -- full encoded object (header + K + V + scales) to disk
  SPLIT_K_CPU_V_NVME -- K stays in pinned CPU RAM; only header + V + scales
                        go to disk (~1/3 of the ALL_NVME byte count)

The NVMe path is taken from the environment variable KNLP_NVME_PATH
(default: a tmpdir inside the run directory).  On a pod with real NVMe
mounted at /runpod-volume, set KNLP_NVME_PATH=/runpod-volume/s10_bench.

Pass criterion: nvme_ratio within 0.005 of 1/3 on every chunk.
The ratio is a mathematical consequence of the K16/V8 layout and is
hardware-independent.  Throughput numbers are logged as informational
metrics only -- they vary with storage hardware.

The EncodedKV object returned by AsymK16V8Codec.encode() has:
  .header_bytes   -- fixed-size metadata (~213 bytes)
  .payload        -- k_payload | v_payload | scales (contiguous bytes)
  .k_payload_len  -- length of K portion
  .v_payload_len  -- length of V portion
  .scale_payload_len

ALL_NVME bytes  = len(header) + len(payload)
SPLIT NVMe bytes = len(header) + v_payload_len + scale_payload_len
                 = len(header) + len(payload) - k_payload_len
nvme_ratio      = SPLIT / ALL  ≈ 1/3

Writes results to stage_dir/split_tier_results.json.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

from ..stages import StageContext, StageResult

N_WARMUP = 3
N_MEASURE = 10
NVME_RATIO_TARGET = 1.0 / 3.0
NVME_RATIO_TOL = 0.005


def _drop_cache() -> None:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass


def _timed_write(data: bytes, path: str, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    return time.perf_counter() - t0


def _timed_read(path: str, n: int) -> float:
    _drop_cache()
    t0 = time.perf_counter()
    for _ in range(n):
        with open(path, "rb") as f:
            _ = f.read()
    return time.perf_counter() - t0


def _make_kv(fp16_target_bytes: int):
    """Return (k, v) with combined FP16 size ~ fp16_target_bytes."""
    n_heads, head_dim, seq_len = 16, 64, 256
    per_token_fp16 = n_heads * head_dim * 2  # BF16 bytes per token per K or V
    total_tokens = fp16_target_bytes // (per_token_fp16 * 2)
    n_layers = max(1, total_tokens // seq_len)
    k = torch.randn(n_layers, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(n_layers, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    return k, v, (k.numel() + v.numel()) * 2


def _bench_chunk(codec, k, v, bench_dir: str) -> dict:
    """Run ALL_NVME and SPLIT_K_CPU_V_NVME bench for one (k, v) pair."""
    enc = codec.encode(k, v)
    header: bytes = enc.header_bytes
    payload: bytes = enc.payload
    k_len: int = enc.k_payload_len

    full_bytes = header + payload
    # SPLIT NVMe payload: header + everything after K in payload
    split_bytes = header + payload[k_len:]

    total_enc = len(full_bytes)
    nvme_split = len(split_bytes)
    nvme_ratio = nvme_split / total_enc

    fpath_all = os.path.join(bench_dir, "all.bin")
    fpath_split = os.path.join(bench_dir, "split_v.bin")

    # Warmup
    for _ in range(N_WARMUP):
        with open(fpath_all, "wb") as f:
            f.write(full_bytes)
            f.flush()
            os.fsync(f.fileno())
        with open(fpath_split, "wb") as f:
            f.write(split_bytes)
            f.flush()
            os.fsync(f.fileno())

    all_write_s = _timed_write(full_bytes, fpath_all, N_MEASURE)
    split_write_s = _timed_write(split_bytes, fpath_split, N_MEASURE)
    all_read_s = _timed_read(fpath_all, N_MEASURE)
    split_read_s = _timed_read(fpath_split, N_MEASURE)

    for p in (fpath_all, fpath_split):
        try:
            os.unlink(p)
        except OSError:
            pass

    def mbps(nbytes, secs):
        return (nbytes * N_MEASURE / 1e6) / secs if secs > 0 else 0.0

    return {
        "total_enc_bytes": total_enc,
        "nvme_split_bytes": nvme_split,
        "k_payload_len": k_len,
        "v_payload_len": enc.v_payload_len,
        "scale_payload_len": enc.scale_payload_len,
        "nvme_ratio": nvme_ratio,
        "all_write_mbps": mbps(total_enc, all_write_s),
        "split_write_mbps": mbps(nvme_split, split_write_s),
        "all_read_mbps": mbps(total_enc, all_read_s),
        "split_read_mbps": mbps(nvme_split, split_read_s),
    }


def run(ctx: StageContext) -> StageResult:
    # Import from the v1 codec path (asymmetric-kv-codec branch).
    try:
        from lmcache.v1.kv_codec import AsymK16V8Codec  # type: ignore[import-not-found]
    except ImportError:
        # Fall back to the older import path used in early branch builds.
        try:
            from lmcache.storage_backend.serde.cachegen_encoder import (  # type: ignore[import-not-found]
                AsymK16V8Codec,
            )
        except ImportError as e:
            reason = f"lmcache not importable: {e}; ensure stage 04 passed"
            ctx.mark_skipped(reason)
            return StageResult(name=ctx.name, status="skipped", reason=reason)

    # Resolve bench directory.  Priority: defconfig > env var > tmpdir.
    cfg_nvme = getattr(ctx.cfg, "nvme_path", "") or ""
    env_nvme = os.environ.get("KNLP_NVME_PATH", "")
    nvme_root = cfg_nvme or env_nvme
    if nvme_root:
        bench_dir = nvme_root
        using_real_nvme = True
    else:
        bench_dir = str(ctx.stage_dir / "bench_tmp")
        using_real_nvme = False
    os.makedirs(bench_dir, exist_ok=True)

    print(
        f"  bench_dir={bench_dir}  real_nvme={using_real_nvme}",
        flush=True,
    )

    codec = AsymK16V8Codec()
    result_path = ctx.stage_dir / "split_tier_results.json"

    chunk_targets = [
        ("1MB", 1 * 1024 * 1024),
        ("8MB", 8 * 1024 * 1024),
        ("32MB", 32 * 1024 * 1024),
        ("64MB", 64 * 1024 * 1024),
    ]

    rows: list[dict] = []
    ratio_failures: list[str] = []
    any_success = False

    for label, fp16_target in chunk_targets:
        k, v, fp16_bytes = _make_kv(fp16_target)
        print(
            f"  [{label}] shape={list(k.shape)} fp16={fp16_bytes / 1e6:.1f} MB",
            flush=True,
        )
        try:
            r = _bench_chunk(codec, k, v, bench_dir)
            r["label"] = label
            r["fp16_bytes"] = fp16_bytes
            rows.append(r)
            any_success = True

            ratio_ok = abs(r["nvme_ratio"] - NVME_RATIO_TARGET) <= NVME_RATIO_TOL
            if not ratio_ok:
                ratio_failures.append(
                    f"{label}: nvme_ratio={r['nvme_ratio']:.4f} "
                    f"not within {NVME_RATIO_TOL} of {NVME_RATIO_TARGET:.4f}"
                )

            print(
                f"    nvme_ratio={r['nvme_ratio']:.4f}  "
                f"all_write={r['all_write_mbps']:.1f} MB/s  "
                f"split_write={r['split_write_mbps']:.1f} MB/s  "
                f"all_read={r['all_read_mbps']:.1f} MB/s  "
                f"split_read={r['split_read_mbps']:.1f} MB/s",
                flush=True,
            )

            # Log nvme_ratio as a pass/fail metric; throughput is informational.
            ctx.log_metric(f"nvme_ratio_{label}", r["nvme_ratio"])
            ctx.log_metric(f"all_write_mbps_{label}", r["all_write_mbps"])
            ctx.log_metric(f"split_write_mbps_{label}", r["split_write_mbps"])
            ctx.log_metric(f"all_read_mbps_{label}", r["all_read_mbps"])
            ctx.log_metric(f"split_read_mbps_{label}", r["split_read_mbps"])

        except Exception as e:
            import traceback

            msg = traceback.format_exc()
            rows.append({"label": label, "error": str(e)})
            ctx.stderr_path.open("a").write(f"WARN [{label}]: {e}\n{msg}\n")
            print(f"    ERROR: {e}", flush=True)

    shutil.rmtree(bench_dir, ignore_errors=True)

    payload = {
        "bench_dir": bench_dir,
        "using_real_nvme": using_real_nvme,
        "rows": rows,
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    ctx.telemetry.log_artifact(result_path, "split_tier_results")

    if not any_success:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason="all chunk rows errored; check lmcache installation",
        )

    if ratio_failures:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"nvme_ratio out of tolerance: {ratio_failures[0]}",
        )

    ctx.mark_done(
        {
            "rows": len(rows),
            "using_real_nvme": using_real_nvme,
            "bench_dir": bench_dir,
        }
    )
    return StageResult(name=ctx.name, status="passed")
