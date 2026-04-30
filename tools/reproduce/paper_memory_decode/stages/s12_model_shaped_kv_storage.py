"""Stage 12 (storage profile): model-shaped KV storage benchmark.

Tests SplitTierStore.put/get with realistic KV tensor shapes from
three representative model families:

  Qwen2.5-7B-like    (GQA 4 heads, head_dim=128)
  Llama-3.1-8B-like  (GQA 8 heads, head_dim=128)
  Qwen3-27B-FA-quarter (GQA 8 heads, head_dim=128, n_layers/4)

Sequence lengths: 1024, 4096.
Policies: ALL_NVME, SPLIT_K_CPU_V_NVME.

Quality checks per cell:
  K bit-exact
  V relative error < 0.05

Pass criterion: none (informational).  All results logged as metrics.
Stage status is "passed" if it runs without error.

Results written to stage_dir/model_shaped_storage.json.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import torch

from ..stages import StageContext, StageResult

CPU_PINNED_BUDGET = 8 * 1024**3
N_WARMUP = 3
N_MEASURE = 8  # fewer iters: large tensors
V_ERR_TOL = 0.05

MODEL_SHAPES = {
    "qwen25_7b": {
        "name": "Qwen2.5-7B-like",
        "n_kv_heads": 4,
        "head_dim": 128,
        "n_layers": 28,
        "seq_lens": [1024, 4096],
    },
    "llama31_8b": {
        "name": "Llama-3.1-8B-like",
        "n_kv_heads": 8,
        "head_dim": 128,
        "n_layers": 32,
        "seq_lens": [1024, 4096],
    },
    "qwen3_27b_fa_quarter": {
        "name": "Qwen3-27B-FA-quarter",
        "n_kv_heads": 8,
        "head_dim": 128,
        "n_layers": 10,
        "seq_lens": [1024, 4096],
    },
}


def _drop_cache() -> None:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass


def _bench_cell(
    SplitTierStore, PlacementPolicy, codec, policy, k, v, bench_dir, cell_key
):
    store_root = Path(bench_dir) / cell_key / policy.name.lower()
    store_root.mkdir(parents=True, exist_ok=True)

    store = SplitTierStore(
        root=store_root,
        codec=codec,
        policy=policy,
        cpu_pinned_budget_bytes=CPU_PINNED_BUDGET,
    )
    for i in range(N_WARMUP):
        store.put("warmup", 0, i, k, v)

    lats_w = []
    last_counts = None
    for i in range(N_MEASURE):
        t0 = time.perf_counter()
        last_counts = store.put("bench", 0, i, k, v)
        lats_w.append((time.perf_counter() - t0) * 1e3)

    _drop_cache()
    lats_r = []
    last_enc = last_rcounts = None
    for i in range(N_MEASURE):
        t0 = time.perf_counter()
        last_enc, last_rcounts = store.get("bench", 0, i)
        lats_r.append((time.perf_counter() - t0) * 1e3)

    # Quality check
    k_exact = False
    v_err = float("inf")
    try:
        k2, v2, *_ = codec.decode(last_enc)
        k_exact = torch.equal(k, k2)
        vn = torch.norm(v.float())
        v_err = (torch.norm((v - v2).float()) / vn).item() if vn > 0 else 0.0
    except Exception as e:
        pass

    shutil.rmtree(store_root, ignore_errors=True)

    def p(lst, pct):
        lst = sorted(lst)
        return lst[min(int(len(lst) * pct / 100), len(lst) - 1)]

    def mbps(nb, lats):
        avg_s = sum(lats) / len(lats) / 1e3
        return (nb / 1e6) / avg_s if avg_s > 0 else 0.0

    fp16_bytes = (k.numel() + v.numel()) * 2
    nvme_bytes = last_counts.nvme_bytes if last_counts else 0
    cpu_bytes = last_counts.cpu_bytes if last_counts else 0
    asym_bytes = nvme_bytes + cpu_bytes + (last_counts.meta_bytes if last_counts else 0)

    return {
        "fp16_bytes": fp16_bytes,
        "asym_bytes": asym_bytes,
        "nvme_bytes": nvme_bytes,
        "cpu_bytes": cpu_bytes,
        "nvme_ratio": nvme_bytes / asym_bytes if asym_bytes > 0 else 0,
        "k_exact": k_exact,
        "v_rel_err": v_err,
        "write_p50_ms": p(lats_w, 50),
        "write_p95_ms": p(lats_w, 95),
        "read_p50_ms": p(lats_r, 50),
        "read_p95_ms": p(lats_r, 95),
        "write_nvme_MBps": mbps(nvme_bytes, lats_w),
        "read_nvme_MBps": mbps(nvme_bytes, lats_r) if last_rcounts else 0,
        "write_logical_MBps": mbps(fp16_bytes, lats_w),
        "read_logical_MBps": mbps(fp16_bytes, lats_r),
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
        os.path.join(nvme_root, "s12_model")
        if nvme_root
        else str(ctx.stage_dir / "bench_model")
    )
    os.makedirs(bench_dir, exist_ok=True)

    device_label = getattr(ctx.cfg, "storage_device_label", "") or "unknown"
    codec = AsymK16V8Codec()
    policies = [PlacementPolicy.ALL_NVME, PlacementPolicy.SPLIT_K_CPU_V_NVME]

    rows = []
    for shape_key, shape in MODEL_SHAPES.items():
        for seq_len in shape["seq_lens"]:
            k = torch.randn(
                shape["n_layers"],
                seq_len,
                shape["n_kv_heads"],
                shape["head_dim"],
                dtype=torch.bfloat16,
            )
            v = torch.randn(
                shape["n_layers"],
                seq_len,
                shape["n_kv_heads"],
                shape["head_dim"],
                dtype=torch.bfloat16,
            )
            fp16_bytes = (k.numel() + v.numel()) * 2
            print(
                f"  {shape['name']} seq={seq_len} "
                f"shape={list(k.shape)} {fp16_bytes/1e6:.1f} MB",
                flush=True,
            )
            row = {
                "shape_key": shape_key,
                "shape_name": shape["name"],
                "seq_len": seq_len,
                "k_shape": list(k.shape),
                "fp16_bytes": fp16_bytes,
                "device_label": device_label,
                "policies": {},
            }
            for policy in policies:
                cell_key = f"{shape_key}_seq{seq_len}"
                print(f"    {policy.name} ...", end=" ", flush=True)
                try:
                    r = _bench_cell(
                        SplitTierStore,
                        PlacementPolicy,
                        codec,
                        policy,
                        k,
                        v,
                        bench_dir,
                        cell_key,
                    )
                    row["policies"][policy.name] = r
                    print(
                        f"nvme={r['nvme_bytes']/1e6:.2f} MB  "
                        f"ratio={r['nvme_ratio']:.4f}  "
                        f"K_exact={r['k_exact']}  "
                        f"V_err={r['v_rel_err']:.4f}  "
                        f"w_p50={r['write_p50_ms']:.1f} ms",
                        flush=True,
                    )
                    pfx = f"{shape_key}_seq{seq_len}_{policy.name}"
                    ctx.log_metric(f"nvme_ratio_{pfx}", r["nvme_ratio"])
                    ctx.log_metric(f"write_p50_ms_{pfx}", r["write_p50_ms"])
                    ctx.log_metric(f"read_p50_ms_{pfx}", r["read_p50_ms"])
                    ctx.log_metric(f"v_rel_err_{pfx}", r["v_rel_err"])
                    ctx.log_metric(f"k_exact_{pfx}", int(r["k_exact"]))
                except Exception as e:
                    import traceback

                    ctx.stderr_path.open("a").write(
                        f"WARN [{shape_key} seq={seq_len} {policy.name}]: "
                        f"{e}\n{traceback.format_exc()}\n"
                    )
                    print(f"ERROR: {e}", flush=True)
            rows.append(row)

    shutil.rmtree(bench_dir, ignore_errors=True)

    result_path = ctx.stage_dir / "model_shaped_storage.json"
    with open(result_path, "w") as f:
        json.dump(
            {"device_label": device_label, "rows": rows}, f, indent=2, default=str
        )
    ctx.telemetry.log_artifact(result_path, "model_shaped_storage")

    ctx.mark_done({"rows": len(rows), "device_label": device_label})
    return StageResult(name=ctx.name, status="passed")
