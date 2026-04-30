"""Stage 13 (storage profile): LMCache split-tier concurrency stress.

Runs concurrent SplitTierStore.put/get workers to stress pinned-CPU
memory pressure and measure p50/p95/p99 latency distributions.

Not a paper reproduction claim.  Purpose: let a colleague understand
whether K-in-pinned-CPU becomes the concurrency bottleneck.

Worker counts: 1, 4, 16.
Duration per worker count: STRESS_DURATION_S seconds.
Policies: ALL_NVME, SPLIT_K_CPU_V_NVME.

Pass criterion: none (informational).
Stage status is "passed" if it runs without error.

Results written to stage_dir/concurrency_stress.json.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import torch

from ..stages import StageContext, StageResult

CPU_PINNED_BUDGET = 8 * 1024**3
STRESS_DURATION_S = 30  # seconds per (policy, worker_count) cell
CHUNK_TARGET_BYTES = 4 * 1024 * 1024  # 4 MB synthetic KV


def _make_kv(target_bytes: int):
    n_heads, head_dim, seq_len = 16, 64, 256
    per_tok = n_heads * head_dim * 2
    total_tok = target_bytes // (per_tok * 2)
    n_layers = max(1, total_tok // seq_len)
    k = torch.randn(n_layers, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(n_layers, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    return k, v


def _worker(store_factory, k, v, duration_s, lat_q, error_q, worker_id):
    """Run put/get in a loop for duration_s, pushing latencies to lat_q."""
    try:
        store = store_factory(worker_id)
        chunk_id = 0
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            store.put(f"w{worker_id}", 0, chunk_id, k, v)
            put_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            store.get(f"w{worker_id}", 0, chunk_id)
            get_ms = (time.perf_counter() - t0) * 1e3

            lat_q.put((put_ms, get_ms))
            chunk_id += 1
    except Exception as e:
        error_q.put(str(e))


def _run_stress_cell(
    SplitTierStore,
    PlacementPolicy,
    codec,
    policy,
    k,
    v,
    bench_dir,
    n_workers,
    duration_s,
):
    cell_root = Path(bench_dir) / policy.name.lower() / f"w{n_workers}"

    def store_factory(worker_id):
        root = cell_root / f"worker_{worker_id}"
        root.mkdir(parents=True, exist_ok=True)
        return SplitTierStore(
            root=root,
            codec=codec,
            policy=policy,
            cpu_pinned_budget_bytes=CPU_PINNED_BUDGET // n_workers,
        )

    lat_q: queue.Queue = queue.Queue()
    error_q: queue.Queue = queue.Queue()

    threads = [
        threading.Thread(
            target=_worker,
            args=(store_factory, k, v, duration_s, lat_q, error_q, i),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    put_lats = []
    get_lats = []
    while not lat_q.empty():
        put_ms, get_ms = lat_q.get_nowait()
        put_lats.append(put_ms)
        get_lats.append(get_ms)

    errors = []
    while not error_q.empty():
        errors.append(error_q.get_nowait())

    shutil.rmtree(cell_root, ignore_errors=True)

    def pct(lst, p):
        if not lst:
            return 0.0
        lst = sorted(lst)
        return lst[min(int(len(lst) * p / 100), len(lst) - 1)]

    return {
        "n_workers": n_workers,
        "n_ops": len(put_lats),
        "errors": errors,
        "put_p50_ms": pct(put_lats, 50),
        "put_p95_ms": pct(put_lats, 95),
        "put_p99_ms": pct(put_lats, 99),
        "get_p50_ms": pct(get_lats, 50),
        "get_p95_ms": pct(get_lats, 95),
        "get_p99_ms": pct(get_lats, 99),
    }


def run(ctx: StageContext) -> StageResult:
    try:
        from lmcache.v1.kv_codec import AsymK16V8Codec  # type: ignore
        from lmcache.v1.kv_codec.split_tier import (  # type: ignore
            SplitTierStore,
            PlacementPolicy,
        )
    except ImportError as e:
        reason = f"lmcache not importable: {e}"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    cfg_nvme = getattr(ctx.cfg, "nvme_path", "") or ""
    env_nvme = os.environ.get("KNLP_NVME_PATH", "")
    nvme_root = cfg_nvme or env_nvme
    bench_dir = (
        os.path.join(nvme_root, "s13_stress")
        if nvme_root
        else str(ctx.stage_dir / "bench_stress")
    )
    os.makedirs(bench_dir, exist_ok=True)

    device_label = getattr(ctx.cfg, "storage_device_label", "") or "unknown"
    codec = AsymK16V8Codec()
    k, v = _make_kv(CHUNK_TARGET_BYTES)
    fp16_bytes = (k.numel() + v.numel()) * 2
    print(
        f"  chunk {fp16_bytes/1e6:.1f} MB  duration={STRESS_DURATION_S}s"
        f"  device={device_label}",
        flush=True,
    )

    policies = [PlacementPolicy.ALL_NVME, PlacementPolicy.SPLIT_K_CPU_V_NVME]
    worker_counts = [1, 4, 16]

    rows = []
    for policy in policies:
        print(f"  {policy.name}", flush=True)
        for n_w in worker_counts:
            print(f"    {n_w} workers {STRESS_DURATION_S}s ...", end=" ", flush=True)
            try:
                r = _run_stress_cell(
                    SplitTierStore,
                    PlacementPolicy,
                    codec,
                    policy,
                    k,
                    v,
                    bench_dir,
                    n_w,
                    STRESS_DURATION_S,
                )
                r["policy"] = policy.name
                r["device_label"] = device_label
                rows.append(r)
                print(
                    f"ops={r['n_ops']}  "
                    f"put p50={r['put_p50_ms']:.1f} p99={r['put_p99_ms']:.1f} ms  "
                    f"get p50={r['get_p50_ms']:.1f} p99={r['get_p99_ms']:.1f} ms"
                    + (f"  ERR={r['errors']}" if r["errors"] else ""),
                    flush=True,
                )
                pfx = f"{policy.name}_w{n_w}"
                ctx.log_metric(f"put_p50_ms_{pfx}", r["put_p50_ms"])
                ctx.log_metric(f"put_p99_ms_{pfx}", r["put_p99_ms"])
                ctx.log_metric(f"get_p50_ms_{pfx}", r["get_p50_ms"])
                ctx.log_metric(f"get_p99_ms_{pfx}", r["get_p99_ms"])
                ctx.log_metric(
                    f"throughput_ops_s_{pfx}", r["n_ops"] / STRESS_DURATION_S
                )
            except Exception as e:
                import traceback

                ctx.stderr_path.open("a").write(
                    f"WARN [{policy.name} w={n_w}]: {e}\n" f"{traceback.format_exc()}\n"
                )
                print(f"ERROR: {e}", flush=True)
                rows.append({"policy": policy.name, "n_workers": n_w, "error": str(e)})

    shutil.rmtree(bench_dir, ignore_errors=True)

    result_path = ctx.stage_dir / "concurrency_stress.json"
    with open(result_path, "w") as f:
        json.dump(
            {"device_label": device_label, "rows": rows}, f, indent=2, default=str
        )
    ctx.telemetry.log_artifact(result_path, "concurrency_stress")

    ctx.mark_done({"rows": len(rows), "device_label": device_label})
    return StageResult(name=ctx.name, status="passed")
