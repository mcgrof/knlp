#!/usr/bin/env python3
"""Decode-time KV-offload latency projection (L1 HBM / L2 host / L3 SSD).

The accuracy benches (rocketkv_eval.py, quest_tiered.py) answer the INFORMATION
question: only ~5-9% of the KV cache must stay resident in HBM for retrieval to
stay near full accuracy. They do NOT answer the SPEED question that decides
whether offloading the other ~90% is usable during generation: each generated
token must fetch its selected cold pages back from wherever they live, and that
fetch sits on the critical path of every token.

This script projects that per-token fetch cost from the bench's measured
cold-page counts plus real model geometry, across a 3-level memory hierarchy:

  L1  HBM        resident tier (the 5-9%); attended for free, sets the compute floor.
  L2  host mem   DRAM reached over PCIe5 / CXL / NVLink-C2C. Bandwidth-bound.
  L3  SSD        NVMe Gen5. Latency-bound on scattered small reads -- the killer.

Decode is memory-bound, so the per-token COMPUTE floor is just the time to stream
the weights (+ resident KV) from HBM once: t_compute = (W + KV_res) / BW_hbm. A
cold fetch is "hidden" if it can be prefetched and overlapped under t_compute;
otherwise it adds to every token. We report t_fetch per tier and whether it hides,
with an --overlap knob modelling a transient HBM cache / prefetch reuse (fraction
of a token's cold pages already resident from the previous step).

Inputs are the bench JSON files. Byte math uses the cold-pages-per-step the bench
measured, expanded by the model's (layers x kv_heads x block x head_dim x 2(K+V)
x dtype). This is a projection with explicit, tunable constants -- not a measured
benchmark -- so every assumption is a CLI flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Model geometry: params, layers, kv_heads, head_dim. KV cache size and per-token
# cold-fetch volume both scale with layers x kv_heads x head_dim.
MODELS = {
    "Qwen/Qwen2.5-1.5B": dict(params=1.54e9, layers=28, kv_heads=2, head_dim=128),
    "HuggingFaceTB/SmolLM2-1.7B": dict(
        params=1.71e9, layers=24, kv_heads=32, head_dim=64
    ),
}

# Memory tiers: bandwidth (GB/s), per-access latency (us), usable queue depth for
# scattered reads. Order-of-magnitude 2026 datacenter numbers; override via flags.
TIERS = {
    "L1_HBM_H100": dict(bw=3350.0, lat=0.0, qd=1, role="compute floor (resident)"),
    "L2_host_PCIe5": dict(bw=50.0, lat=1.0, qd=128, role="host DRAM over PCIe5 x16"),
    "L2_host_CXL": dict(bw=40.0, lat=0.3, qd=128, role="host DRAM over CXL"),
    "L2_host_C2C": dict(bw=900.0, lat=0.4, qd=256, role="host DRAM, Grace-Hopper C2C"),
    "L3_SSD_Gen5": dict(bw=12.0, lat=90.0, qd=256, role="NVMe Gen5, random 8KB"),
}
GB = 1e9
MiB = 1024 * 1024


def t_compute_ms(m, resident_pct, ctx, dtype_bytes, hbm_bw, batch):
    """Memory-bound decode floor. Weights stream ONCE per batch (shared); resident
    KV scales with batch (one cache per request)."""
    w_bytes = m["params"] * dtype_bytes
    kv_full = m["layers"] * m["kv_heads"] * ctx * m["head_dim"] * 2 * dtype_bytes
    kv_res = kv_full * resident_pct / 100.0
    return 1e3 * (w_bytes + batch * kv_res) / (hbm_bw * GB)


def fetch(m, cold_pages, block, dtype_bytes, tier, overlap, batch):
    """Per-token cold-fetch bytes, scattered reads, and projected time (ms). Cold
    fetch is PER REQUEST, so it scales with batch (unlike the shared weights)."""
    units = m["layers"] * m["kv_heads"]  # each (layer,kv-head) selects independently
    eff = max(0.0, 1.0 - overlap)
    page_bytes = block * m["head_dim"] * 2 * dtype_bytes  # one page, one (layer,head)
    bytes_tok = cold_pages * units * page_bytes * eff * batch
    reads_tok = cold_pages * units * eff * batch
    t_bw = bytes_tok / (tier["bw"] * GB)
    t_lat = (reads_tok / max(1, tier["qd"])) * (tier["lat"] * 1e-6)
    return bytes_tok, reads_tok, 1e3 * max(t_bw, t_lat)


def near_full_budget(res_method, full, frac=0.95):
    """Smallest budget reaching frac of full-KV accuracy (the deployable point)."""
    best = None
    for B in sorted(res_method, key=int):
        if res_method[B]["acc"] >= frac * full:
            best = int(B)
            break
    return (
        best
        if best is not None
        else int(max(res_method, key=lambda b: res_method[b]["acc"]))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", required=True, help="rocketkv_eval JSON(s)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--hbm-gbps", type=float, default=3350.0, help="resident-tier BW")
    ap.add_argument(
        "--overlap", type=float, default=0.0, help="prefetch reuse frac 0..1"
    )
    ap.add_argument("--method", default="rocketkv", help="which method's cold fetch")
    ap.add_argument("--full-frac", type=float, default=0.95)
    ap.add_argument("--batch", default="1,32,128", help="concurrent requests")
    args = ap.parse_args()
    batches = [int(b) for b in str(args.batch).split(",")]

    cold_tiers = [t for t in TIERS if t != "L1_HBM_H100"]
    for path in args.bench:
        j = json.loads(Path(path).read_text())
        model = j["model"]
        if model not in MODELS:
            print(f"[skip] no geometry for {model}")
            continue
        m = MODELS[model]
        block = j.get("block_size", args.block)
        res = j["results"]
        full = max(res["full"][b]["acc"] for b in res["full"])
        kv_full_mib = (
            m["layers"]
            * m["kv_heads"]
            * args.ctx
            * m["head_dim"]
            * 2
            * args.dtype_bytes
        ) / MiB
        meth = args.method if args.method in res else "rocketkv"
        B = str(near_full_budget(res[meth], full, args.full_frac))
        cell = res[meth][B]
        cold_pages = cell["cold_read_per_step"]
        if cold_pages == 0:  # snapkv/quest carry no cold fetch; fall back to read
            cold_pages = cell["read_per_step"]

        print(
            f"\n=== {model}  ctx={args.ctx}  KV_full={kv_full_mib:.0f} MiB/req "
            f"({m['kv_heads']} kv-heads) ==="
        )
        print(
            f"  method={meth}  budget={B}  acc={cell['acc']:.2f} (full {full:.2f})  "
            f"resident={cell['resident_pct']:.1f}%  "
            f"cold fetch={cold_pages}p x {m['layers']}L x {m['kv_heads']}H/req  "
            f"overlap={args.overlap:.0%}"
        )
        for batch in batches:
            tcomp = t_compute_ms(
                m,
                cell["resident_pct"],
                args.ctx,
                args.dtype_bytes,
                args.hbm_gbps,
                batch,
            )
            print(
                f"\n  batch={batch:<4d} compute floor {tcomp:.3f} ms/tok "
                f"(weights shared, KV x{batch})"
            )
            print(
                f"  {'tier':16s} {'role':30s} {'MiB/tok':>9s} {'reads/tok':>10s} "
                f"{'ms/tok':>8s} {'hides?':>7s}"
            )
            for tn in cold_tiers:
                tier = TIERS[tn]
                byt, rd, tms = fetch(
                    m, cold_pages, block, args.dtype_bytes, tier, args.overlap, batch
                )
                hides = "yes" if tms <= tcomp else "NO"
                print(
                    f"  {tn:16s} {tier['role']:30s} {byt/MiB:9.2f} {rd:10.0f} "
                    f"{tms:8.3f} {hides:>7s}"
                )


if __name__ == "__main__":
    main()
