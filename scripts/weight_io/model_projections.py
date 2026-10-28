#!/usr/bin/env python3
"""Capacity, routing and timing projections from measured storage inputs.

Every number produced here is a calculation. The inputs that come from
measurement (bandwidth at a command size, per-command latency, maximum
outstanding commands, measured layer transfer and compute times) are
named explicitly so a reader can trace each projection to its evidence.

Projections:

  storage_bound(D, N, BW, IOPS, lat, Q)
      optimistic lower bound max(D/BW, N/IOPS, N*lat/Q) for D bytes in
      N commands under a steady state; dependency chains can be worse.

  unique_experts(E, k, B)
      expected distinct experts selected by B independent requests
      choosing k of E uniformly: E * (1 - (1 - k/E)**B). A reference
      model, not router behaviour.

  reload_time(layer_bytes, n_layers, BW)
      serial no-residency reload of every layer at bandwidth BW.

  pipeline(read, copy, repack, compute)
      serial sum versus the optimistic max of overlappable stages for a
      double-buffered layer stream.

  hbf_sensitivity(ranges, page_sizes)
      union page footprint for labeled hypothetical page sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def storage_bound(D, N, BW, IOPS, lat_s, Q):
    terms = {
        "bytes/BW": D / BW if BW else None,
        "cmds/IOPS": N / IOPS if IOPS else None,
        "cmds*lat/Q": N * lat_s / Q if Q else None,
    }
    return {
        "seconds": max(v for v in terms.values() if v is not None),
        "terms_seconds": terms,
    }


def unique_experts(E, k, B):
    return E * (1 - (1 - k / E) ** B)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--medians", required=True, help="metrics_medians.csv from aggregate_runs.py"
    )
    ap.add_argument(
        "--projections", required=True, help="projections.json from projections.py"
    )
    ap.add_argument("--classes", required=True)
    ap.add_argument("--gpu-bytes", type=int, default=48 * 2**30)
    ap.add_argument(
        "--gpu-reserve-fraction",
        type=float,
        default=0.25,
        help="assumed reserve for activations, KV and workspaces",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import csv

    med = list(csv.DictReader(open(args.medians)))
    proj = json.loads(Path(args.projections).read_text())
    classes = json.loads(Path(args.classes).read_text())

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # measured inputs: best cold direct bandwidth from the chunk sweep, and latency
    chunk_rows = [r for r in med if r["group"] == "chunks"]
    bw_by = {}
    for r in chunk_rows:
        bw = f(r["target_bw_MiBps_med"])
        if bw:
            bw_by[(int(r["chunk"]), int(r["qd"]))] = {
                "MiBps": bw,
                "lat_p50_us": f(r["lat_p50_us_med"]),
                "cmds": f(r["D_target_cmds_med"]),
            }
    best = max(bw_by.items(), key=lambda kv: kv[1]["MiBps"]) if bw_by else None
    layer_rows = [r for r in med if r["group"] == "layers" and r["cache"] == "cold"]
    measured = {
        "chunk_sweep_MiBps": {f"{c}B/qd{q}": v for (c, q), v in sorted(bw_by.items())},
        "best_chunk_qd": (
            {"chunk": best[0][0], "qd": best[0][1], **best[1]} if best else None
        ),
        "layer_cold": {
            r["run"]: {
                "W": int(float(r["W_required"])),
                "elapsed_ms": f(r["elapsed_ms_med"]),
                "D": f(r["D_target_bytes_med"]),
                "cmds": f(r["D_target_cmds_med"]),
                "MiBps": f(r["target_bw_MiBps_med"]),
            }
            for r in layer_rows
        },
    }
    BW = (best[1]["MiBps"] * 2**20) if best else None
    lat = (best[1]["lat_p50_us"] * 1e-6) if best and best[1]["lat_p50_us"] else None

    out = {
        "inputs": measured,
        "assumptions": {
            "gpu_bytes": args.gpu_bytes,
            "gpu_reserve_fraction": args.gpu_reserve_fraction,
            "max_cmd_bytes": proj["max_cmd_bytes"],
        },
        "models": {},
    }
    usable = args.gpu_bytes * (1 - args.gpu_reserve_fraction)
    for repo, c in classes.items():
        m = {"classes": {}}
        for h, v in c["classes"].items():
            bpl = v["bytes_per_layer"][0]
            n = len(v["layers"])
            units = [
                u
                for u in proj["units"]
                if u["repo"] == repo and u.get("bytes_per_expert")
            ]
            exp_bytes = None
            n_exp = v["n_experts"]
            for u in units:
                if u["unit"] in (
                    f"layer{v['layers'][0]}",
                    f"layer{v['layers'][len(v['layers']) // 2]}",
                ):
                    exp_bytes = list(u["bytes_per_expert"].values())[0]
                    if not n_exp:
                        # fused: derive from selection slab count is not available here; keep config-free estimate
                        n_exp = v["n_experts"]
            cls = {
                "n_layers": n,
                "bytes_per_layer": bpl,
                "layers_fit_in_usable_gpu": int(usable // bpl) if bpl else None,
                "two_buffers_fit": bool(2 * bpl <= usable),
                "reload_all_layers_seconds_at_best_BW": (bpl * n / BW) if BW else None,
                "min_cmds_per_layer_at_max_cmd": -(-bpl // proj["max_cmd_bytes"]),
            }
            if exp_bytes and n_exp:
                cls["expert_bytes"] = exp_bytes
                cls["n_experts"] = n_exp
                cls["bank_bytes"] = exp_bytes * n_exp
                cls["unique_experts_B"] = {
                    str(B): unique_experts(n_exp, 8, B) for B in (1, 4, 16, 64, 256)
                }
                cls["unique_experts_topk_assumed"] = 8
                for hit in (0, 25, 50, 75, 90, 99):
                    cls[f"bytes_missed_top8_hit{hit}pct"] = (
                        8 * exp_bytes * (1 - hit / 100)
                    )
                if BW:
                    cls["storage_bound_top8_no_reuse_seconds"] = (
                        storage_bound(
                            8 * exp_bytes,
                            -(-8 * exp_bytes // proj["max_cmd_bytes"]),
                            BW,
                            None,
                            lat,
                            32,
                        )["seconds"]
                        if lat
                        else None
                    )
            m["classes"][h] = cls
        out["models"][repo] = m
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}; best measured chunk/qd: {measured['best_chunk_qd']}")


if __name__ == "__main__":
    main()
