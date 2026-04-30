"""Stage 10: LMCache split-tier storage microbench.

Uses the real LMCache SplitTierStore engine to verify the NVMe
traffic-ratio claim from the paper and measure throughput under
three placement policies.

Three storage policies:

  FP16_BASELINE      raw BF16 K+V bytes to disk (no codec, reference)
  ALL_NVME           SplitTierStore(ALL_NVME): K16+V8+scales to disk
  SPLIT_K_CPU_V_NVME SplitTierStore(SPLIT): K in CPU pinned RAM,
                     only V8+scales on NVMe (~1/3 of ALL_NVME bytes)

Pass criteria (hardware-independent layout math):
  nvme_ratio = SPLIT.nvme_bytes / ALL_NVME.nvme_bytes
  Must be within NVME_RATIO_TOL of 1/3 on every chunk size.

Additional quality checks (warn-only, not pass/fail):
  K round-trip bit-exact (BF16 K stored at full precision)
  V relative error < V_ERR_TOL (FP8 dequant introduces ~1-3% error)

Throughput numbers are informational and device-specific.  They
are logged as metrics but do not affect pass/fail status.

NVMe path resolution: CONFIG_KNLP_NVME_PATH > KNLP_NVME_PATH env >
tmpdir inside run_dir.  On a pod with real NVMe mounted at
/runpod-volume, add CONFIG_KNLP_NVME_PATH="/runpod-volume/s10"
to .config or use the decode-nvme-tier defconfig.

Results written to stage_dir/split_tier_results.json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from ..stages import StageContext, StageResult

N_WARMUP = 3
N_MEASURE = 10
NVME_RATIO_TARGET = 1.0 / 3.0
NVME_RATIO_TOL = 0.005
# V goes through FP8 quantisation; relative error is expected ~1-3%.
V_ERR_TOL = 0.05
# CPU pinned budget for _CPUPinnedKStore: 4 GiB is enough for 10
# N_MEASURE puts of 64 MB chunks (K ≈ 32 MB each → 320 MB total).
CPU_PINNED_BUDGET = 4 * 1024**3


def _drop_cache() -> None:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass


def _run_cmd(cmd: list[str]) -> str:
    """Run a command and return stdout; return empty string on failure."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def _capture_hw_metadata(bench_dir: str) -> dict:
    """Collect storage-device info for the result record."""
    meta: dict = {}
    meta["bench_dir"] = bench_dir

    # Filesystem and mount
    meta["df"] = _run_cmd(["df", "-h", bench_dir])
    meta["mount"] = _run_cmd(
        [
            "findmnt",
            "-T",
            bench_dir,
            "-o",
            "TARGET,SOURCE,FSTYPE,OPTIONS",
            "--noheadings",
        ]
    )

    # RAID (if any)
    meta["mdstat"] = _run_cmd(["cat", "/proc/mdstat"])

    # Block device queue scheduler for the device backing bench_dir
    try:
        src = (
            subprocess.run(
                ["df", "--output=source", bench_dir],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .split("\n")[-1]
        )
        meta["block_device"] = src
        dev = src.lstrip("/").split("/")[-1]  # e.g. "md0" or "nvme0n1"
        sched_path = f"/sys/block/{dev}/queue/scheduler"
        if os.path.exists(sched_path):
            meta["scheduler"] = open(sched_path).read().strip()
    except Exception:
        pass

    # PCIe link info for each NVMe device
    nvme_links: dict = {}
    for i in range(9):
        sp = f"/sys/block/nvme{i}n1/../device/current_link_speed"
        wp = f"/sys/block/nvme{i}n1/../device/current_link_width"
        if os.path.exists(sp):
            speed = open(sp).read().strip()
            width = open(wp).read().strip() if os.path.exists(wp) else "?"
            nvme_links[f"nvme{i}n1"] = f"{speed} x{width}"
    meta["nvme_pcie_links"] = nvme_links

    # lsblk
    meta["lsblk"] = _run_cmd(["lsblk", "-o", "NAME,SIZE,TYPE,ROTA,SCHED,MOUNTPOINT"])

    # Kernel version
    meta["uname"] = _run_cmd(["uname", "-r"])

    return meta


def _kv_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a BF16 tensor to raw bytes."""
    return bytes(t.contiguous().view(torch.uint8).numpy().tobytes())


def _make_kv(fp16_target_bytes: int):
    """Return (k, v, actual_fp16_bytes).

    Combined BF16 size of (k, v) is approximately fp16_target_bytes.
    """
    n_heads, head_dim, seq_len = 16, 64, 256
    per_token_fp16 = n_heads * head_dim * 2  # BF16 bytes per token per K or V
    total_tokens = fp16_target_bytes // (per_token_fp16 * 2)
    n_layers = max(1, total_tokens // seq_len)
    k = torch.randn(n_layers, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(n_layers, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    return k, v, (k.numel() + v.numel()) * 2


# ── FP16 raw baseline ────────────────────────────────────────────────────────


def _bench_fp16_baseline(k: torch.Tensor, v: torch.Tensor, bench_dir: str) -> dict:
    """Time raw BF16 K+V writes and reads (no codec, reference tier)."""
    data = _kv_to_bytes(k) + _kv_to_bytes(v)
    n_bytes = len(data)
    fpath = os.path.join(bench_dir, "fp16_baseline.bin")

    # Warmup
    for _ in range(N_WARMUP):
        with open(fpath, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    # Write timing
    t0 = time.perf_counter()
    for _ in range(N_MEASURE):
        with open(fpath, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    write_s = time.perf_counter() - t0

    # Read timing
    _drop_cache()
    t0 = time.perf_counter()
    for _ in range(N_MEASURE):
        with open(fpath, "rb") as f:
            _ = f.read()
    read_s = time.perf_counter() - t0

    try:
        os.unlink(fpath)
    except OSError:
        pass

    def mbps(nb, s):
        return (nb * N_MEASURE / 1e6) / s if s > 0 else 0.0

    return {
        "fp16_size": n_bytes,
        "fp16_write_mbps": mbps(n_bytes, write_s),
        "fp16_read_mbps": mbps(n_bytes, read_s),
    }


# ── SplitTierStore policy bench ──────────────────────────────────────────────


def _bench_policy(
    SplitTierStore,  # class, passed in to avoid import at module level
    PlacementPolicy,
    CodecHashes,
    codec,
    policy,
    k: torch.Tensor,
    v: torch.Tensor,
    bench_dir: str,
    fp16_size: int,
) -> dict:
    """Benchmark one SplitTierStore placement policy.

    Returns a dict with byte counts, nvme_ratio, throughput, and
    quality-check results.
    """
    store_root = Path(bench_dir) / policy.name.lower()
    store_root.mkdir(parents=True, exist_ok=True)

    store = SplitTierStore(
        root=store_root,
        codec=codec,
        policy=policy,
        cpu_pinned_budget_bytes=CPU_PINNED_BUDGET,
    )

    # Warmup writes (unique chunk_ids so _CPUPinnedKStore doesn't collide)
    for i in range(N_WARMUP):
        store.put(f"warmup", 0, i, k, v)

    # Measured writes
    t0 = time.perf_counter()
    last_counts = None
    for i in range(N_MEASURE):
        last_counts = store.put(f"bench", 0, i, k, v)
    write_s = time.perf_counter() - t0

    nvme_write = last_counts.nvme_bytes
    cpu_write = last_counts.cpu_bytes
    meta_write = last_counts.meta_bytes

    # Measured reads (drop page cache first to force NVMe reads)
    _drop_cache()
    t0 = time.perf_counter()
    last_enc = None
    last_rcounts = None
    for i in range(N_MEASURE):
        last_enc, last_rcounts = store.get(f"bench", 0, i)
    read_s = time.perf_counter() - t0

    nvme_read = last_rcounts.nvme_bytes
    cpu_read = last_rcounts.cpu_bytes

    # Quality checks on the last retrieved chunk
    k_exact = False
    v_rel_err = float("inf")
    quality_warn = []
    try:
        k2, v2 = codec.decode(last_enc)
        k_exact = torch.equal(k, k2)
        if not k_exact:
            quality_warn.append("K round-trip not bit-exact")
        v_norm = torch.norm(v.float())
        diff_norm = torch.norm((v - v2).float())
        v_rel_err = (diff_norm / v_norm).item() if v_norm > 0 else 0.0
        if v_rel_err >= V_ERR_TOL:
            quality_warn.append(
                f"V relative error {v_rel_err:.4f} >= tolerance {V_ERR_TOL}"
            )
    except Exception as e:
        quality_warn.append(f"decode failed: {e}")

    # Cleanup
    shutil.rmtree(store_root, ignore_errors=True)

    def mbps(nb, s):
        return (nb * N_MEASURE / 1e6) / s if s > 0 else 0.0

    return {
        "nvme_write_bytes": nvme_write,
        "cpu_write_bytes": cpu_write,
        "meta_write_bytes": meta_write,
        "total_write_bytes": nvme_write + cpu_write + meta_write,
        "nvme_read_bytes": nvme_read,
        "cpu_read_bytes": cpu_read,
        "storage_ratio": (nvme_write + cpu_write) / fp16_size,
        "k_exact": k_exact,
        "v_rel_err": v_rel_err,
        "quality_warnings": quality_warn,
        "write_mbps": mbps(nvme_write + cpu_write, write_s),
        "read_mbps": mbps(nvme_read + cpu_read, read_s),
        # Engine-level: NVMe bytes only (excludes CPU-side K in SPLIT)
        "nvme_write_mbps": mbps(nvme_write, write_s),
        "nvme_read_mbps": mbps(nvme_read, read_s),
    }


# ── per-chunk orchestration ──────────────────────────────────────────────────


def _bench_chunk(
    SplitTierStore,
    PlacementPolicy,
    CodecHashes,
    codec,
    k: torch.Tensor,
    v: torch.Tensor,
    bench_dir: str,
) -> dict:
    """Run FP16_BASELINE, ALL_NVME, and SPLIT_K_CPU_V_NVME for one KV pair."""
    fp16_size = (k.numel() + v.numel()) * 2

    fp16 = _bench_fp16_baseline(k, v, bench_dir)

    all_nvme = _bench_policy(
        SplitTierStore,
        PlacementPolicy,
        CodecHashes,
        codec,
        PlacementPolicy.ALL_NVME,
        k,
        v,
        bench_dir,
        fp16_size,
    )

    split = _bench_policy(
        SplitTierStore,
        PlacementPolicy,
        CodecHashes,
        codec,
        PlacementPolicy.SPLIT_K_CPU_V_NVME,
        k,
        v,
        bench_dir,
        fp16_size,
    )

    # nvme_ratio: fraction of ALL_NVME disk bytes that SPLIT puts on NVMe.
    # V+scales vs K+V+scales — this is the hardware-independent layout ratio.
    nvme_ratio = (
        split["nvme_write_bytes"] / all_nvme["nvme_write_bytes"]
        if all_nvme["nvme_write_bytes"] > 0
        else float("nan")
    )

    return {
        "fp16_size": fp16_size,
        "fp16_write_mbps": fp16["fp16_write_mbps"],
        "fp16_read_mbps": fp16["fp16_read_mbps"],
        "all_nvme_write_bytes": all_nvme["nvme_write_bytes"],
        "all_nvme_read_bytes": all_nvme["nvme_read_bytes"],
        "split_nvme_write_bytes": split["nvme_write_bytes"],
        "split_cpu_write_bytes": split["cpu_write_bytes"],
        "split_nvme_read_bytes": split["nvme_read_bytes"],
        "split_cpu_read_bytes": split["cpu_read_bytes"],
        "storage_ratio_all": all_nvme["storage_ratio"],
        "storage_ratio_split": split["storage_ratio"],
        "nvme_ratio": nvme_ratio,
        "all_nvme_write_mbps": all_nvme["nvme_write_mbps"],
        "all_nvme_read_mbps": all_nvme["nvme_read_mbps"],
        "split_nvme_write_mbps": split["nvme_write_mbps"],
        "split_nvme_read_mbps": split["nvme_read_mbps"],
        "k_exact": split["k_exact"],
        "v_rel_err": split["v_rel_err"],
        "quality_warnings": split["quality_warnings"],
    }


# ── stage entry point ────────────────────────────────────────────────────────


def run(ctx: StageContext) -> StageResult:
    # Import LMCache storage engine components.
    try:
        from lmcache.v1.kv_codec import AsymK16V8Codec  # type: ignore[import-not-found]
        from lmcache.v1.kv_codec.split_tier import (  # type: ignore[import-not-found]
            SplitTierStore,
            PlacementPolicy,
        )
        from lmcache.v1.kv_codec.encoded_kv import (  # type: ignore[import-not-found]
            CodecHashes,
        )
    except ImportError as e:
        reason = f"lmcache not importable: {e}; ensure stage 04 passed"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    # Resolve bench directory. Priority: defconfig > env var > tmpdir.
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

    print(f"  bench_dir={bench_dir}  real_nvme={using_real_nvme}", flush=True)

    # Capture hardware metadata before any I/O.
    hw_meta = _capture_hw_metadata(bench_dir)
    if hw_meta.get("nvme_pcie_links"):
        for dev, info in hw_meta["nvme_pcie_links"].items():
            print(f"  {dev}: {info}", flush=True)
    if hw_meta.get("lsblk"):
        print(f"  lsblk:\n{hw_meta['lsblk']}", flush=True)

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
    quality_warns: list[str] = []
    any_success = False

    for label, fp16_target in chunk_targets:
        k, v, fp16_bytes = _make_kv(fp16_target)
        print(
            f"  [{label}] shape={list(k.shape)} fp16={fp16_bytes / 1e6:.1f} MB",
            flush=True,
        )
        try:
            r = _bench_chunk(
                SplitTierStore,
                PlacementPolicy,
                CodecHashes,
                codec,
                k,
                v,
                bench_dir,
            )
            r["label"] = label
            r["fp16_bytes"] = fp16_bytes
            rows.append(r)
            any_success = True

            # Pass/fail: byte ratio is pure layout math.
            ratio_ok = abs(r["nvme_ratio"] - NVME_RATIO_TARGET) <= NVME_RATIO_TOL
            if not ratio_ok:
                ratio_failures.append(
                    f"{label}: nvme_ratio={r['nvme_ratio']:.4f} "
                    f"not within {NVME_RATIO_TOL} of "
                    f"{NVME_RATIO_TARGET:.4f}"
                )

            # Warn-only: quality checks.
            for w in r.get("quality_warnings", []):
                quality_warns.append(f"{label}: {w}")

            # Print byte accounting.
            all_mb = r["all_nvme_write_bytes"] / 1e6
            split_mb = r["split_nvme_write_bytes"] / 1e6
            fp16_mb = r["fp16_size"] / 1e6
            print(
                f"    bytes:  fp16={fp16_mb:.2f} MB  "
                f"all_nvme={all_mb:.2f} MB "
                f"({r['storage_ratio_all']:.4f}x fp16)  "
                f"split_nvme={split_mb:.2f} MB "
                f"({r['storage_ratio_split']:.4f}x fp16)  "
                f"nvme_ratio={r['nvme_ratio']:.4f}",
                flush=True,
            )
            print(
                f"    quality: K_exact={r['k_exact']}  "
                f"V_rel_err={r['v_rel_err']:.4f}"
                + (f"  WARN: {r['quality_warnings']}" if r["quality_warnings"] else ""),
                flush=True,
            )
            print(
                f"    write:  fp16={r['fp16_write_mbps']:.1f}  "
                f"all_nvme={r['all_nvme_write_mbps']:.1f}  "
                f"split_nvme={r['split_nvme_write_mbps']:.1f} MB/s",
                flush=True,
            )
            print(
                f"    read:   fp16={r['fp16_read_mbps']:.1f}  "
                f"all_nvme={r['all_nvme_read_mbps']:.1f}  "
                f"split_nvme={r['split_nvme_read_mbps']:.1f} MB/s",
                flush=True,
            )

            # Log metrics. nvme_ratio is the pass/fail metric.
            # Throughput metrics are informational.
            ctx.log_metric(f"nvme_ratio_{label}", r["nvme_ratio"])
            ctx.log_metric(f"storage_ratio_all_{label}", r["storage_ratio_all"])
            ctx.log_metric(f"storage_ratio_split_{label}", r["storage_ratio_split"])
            ctx.log_metric(f"v_rel_err_{label}", r["v_rel_err"])
            ctx.log_metric(f"k_exact_{label}", int(r["k_exact"]))
            ctx.log_metric(f"fp16_write_mbps_{label}", r["fp16_write_mbps"])
            ctx.log_metric(f"all_nvme_write_mbps_{label}", r["all_nvme_write_mbps"])
            ctx.log_metric(f"split_nvme_write_mbps_{label}", r["split_nvme_write_mbps"])
            ctx.log_metric(f"fp16_read_mbps_{label}", r["fp16_read_mbps"])
            ctx.log_metric(f"all_nvme_read_mbps_{label}", r["all_nvme_read_mbps"])
            ctx.log_metric(f"split_nvme_read_mbps_{label}", r["split_nvme_read_mbps"])

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
        "hw_meta": hw_meta,
        "rows": rows,
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    ctx.telemetry.log_artifact(result_path, "split_tier_results")

    # Log quality warnings to stderr (non-fatal).
    if quality_warns:
        with ctx.stderr_path.open("a") as f:
            for w in quality_warns:
                f.write(f"QUALITY WARN: {w}\n")
        print(f"  Quality warnings (non-fatal): {quality_warns}", flush=True)

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
            "quality_warnings": quality_warns,
        }
    )
    return StageResult(name=ctx.name, status="passed")
