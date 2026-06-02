#!/usr/bin/env python3
"""plot_stage5_results.py — generate plots from systems_byte_floor.csv,
retrieval_frontier.csv, and (optionally) nvme_cold_byte_floor.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import _muvera_config as cf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path):
    if not path.exists(): return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _to_float(s):
    try: return float(s)
    except (TypeError, ValueError): return None


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_PLOTS", True):
        print("CONFIG_MUVERA_PLOTS=n; skipping")
        return
    run_dir = cf.get_run_dir(cfg)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    sys_rows = _read_csv(run_dir / "systems_byte_floor.csv")
    ret_rows = _read_csv(run_dir / "retrieval_frontier.csv")
    nvme_rows = _read_csv(run_dir / "nvme_cold_byte_floor.csv")

    if sys_rows:
        # mmap random p50 vs record_bytes
        rb = sorted(set(int(r["record_bytes"]) for r in sys_rows))
        p50_by_b = {int(r["record_bytes"]): _to_float(r["lat_p50_ns"])
                     for r in sys_rows if r["mode"] == "mmap_random_point"}
        p99_by_b = {int(r["record_bytes"]): _to_float(r["lat_p99_ns"])
                     for r in sys_rows if r["mode"] == "mmap_random_point"}
        if p50_by_b:
            fig, ax = plt.subplots(figsize=(7, 5))
            xs = sorted(p50_by_b.keys())
            ax.plot(xs, [p50_by_b[x] for x in xs], "o-", label="p50")
            ax.plot(xs, [p99_by_b[x] for x in xs], "o-", label="p99")
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("bytes per record"); ax.set_ylabel("latency (ns)")
            ax.set_title("mmap random read latency vs record size")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "record_bytes_vs_mmap_random_p50.png", dpi=120)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(xs, [p99_by_b[x] for x in xs], "o-", color="tab:red", label="p99")
            ax.plot(xs, [p50_by_b[x] for x in xs], "o-", color="tab:blue", label="p50")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("bytes per record"); ax.set_ylabel("latency (ns)")
            ax.set_title("mmap random read latency (linear y) vs record size")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "record_bytes_vs_mmap_random_p95.png", dpi=120)
            plt.close(fig)

        # H2D + projection at batch=64 vs record_bytes
        h2d_64 = {int(r["record_bytes"]): _to_float(r["lat_p50_ns"])
                   for r in sys_rows
                   if r["mode"] == "h2d_transfer" and r["batch_size"] == "64"}
        proj_64 = {int(r["record_bytes"]): _to_float(r["lat_p50_ns"])
                    for r in sys_rows
                    if r["mode"] == "gpu_projection" and r["batch_size"] == "64"}
        if h2d_64:
            fig, ax = plt.subplots(figsize=(7, 5))
            xs = sorted(h2d_64.keys())
            ax.plot(xs, [h2d_64[x]/1e3 for x in xs], "o-", label="H2D x64 p50")
            if proj_64:
                ax.plot(xs, [proj_64[x]/1e3 for x in xs], "o-",
                         label="proj x64 p50 (→hidden=4096)")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("bytes per record"); ax.set_ylabel("μs / batch")
            ax.set_title("CPU→GPU + projection cost vs record size (batch=64)")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "record_bytes_vs_h2d_batch64.png", dpi=120)
            plt.close(fig)

        # Total per-record serving estimate
        brand_64 = {int(r["record_bytes"]): _to_float(r["lat_p50_ns"])
                     for r in sys_rows
                     if r["mode"] == "mmap_batched_random" and r["batch_size"] == "64"}
        if brand_64 and h2d_64 and proj_64:
            fig, ax = plt.subplots(figsize=(7, 5))
            xs = sorted(brand_64.keys() & h2d_64.keys() & proj_64.keys())
            total = [(brand_64[x] + h2d_64[x] + proj_64[x]) / 64 / 1e3 for x in xs]
            ax.plot(xs, total, "o-", color="black", label="total estimate per record")
            ax.plot(xs, [brand_64[x]/64/1e3 for x in xs], "o--", label="read")
            ax.plot(xs, [h2d_64[x]/64/1e3 for x in xs], "o--", label="H2D")
            ax.plot(xs, [proj_64[x]/64/1e3 for x in xs], "o--", label="proj")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("bytes per record"); ax.set_ylabel("μs / record")
            ax.set_title("Estimated per-record serving cost vs record size (batch=64)")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "record_bytes_vs_total_serving_estimate.png", dpi=120)
            plt.close(fig)

    if ret_rows:
        # Recall@10 vs bytes
        oracle = next((r for r in ret_rows if r["variant"] == "chamfer_oracle"), None)
        chamfer_r10 = _to_float(oracle["recall_at_10"]) if oracle else None
        fig, ax = plt.subplots(figsize=(8, 5))
        for prec in ("fp32", "fp16", "int8", "pq"):
            xs = []; ys = []
            for r in ret_rows:
                if r["variant"] != "FDE": continue
                if r["precision"] != prec: continue
                bpv = _to_float(r["bytes_per_vector"]); rk = _to_float(r["recall_at_10"])
                if bpv and rk is not None: xs.append(bpv); ys.append(rk)
            if xs:
                ax.plot(xs, ys, "o", alpha=0.7, label=f"FDE {prec}")
        for r in ret_rows:
            if not r["variant"].startswith("pooled-"): continue
            bpv = _to_float(r["bytes_per_vector"]); rk = _to_float(r["recall_at_10"])
            if bpv and rk is not None:
                ax.plot(bpv, rk, "s", markersize=10, label=r["variant"])
        if chamfer_r10 is not None:
            ax.axhline(chamfer_r10, color="black", linestyle="--",
                        label=f"chamfer oracle ({chamfer_r10:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("bytes per vector"); ax.set_ylabel("recall@10")
        ax.set_title("Retrieval recall@10 vs storage bytes")
        ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "bytes_vs_recall10.png", dpi=120)
        plt.close(fig)

        # Recall@100 vs bytes
        chamfer_r100 = _to_float(oracle["recall_at_100"]) if oracle else None
        fig, ax = plt.subplots(figsize=(8, 5))
        for prec in ("fp32", "fp16", "int8", "pq"):
            xs = []; ys = []
            for r in ret_rows:
                if r["variant"] != "FDE": continue
                if r["precision"] != prec: continue
                bpv = _to_float(r["bytes_per_vector"]); rk = _to_float(r["recall_at_100"])
                if bpv and rk is not None: xs.append(bpv); ys.append(rk)
            if xs:
                ax.plot(xs, ys, "o", alpha=0.7, label=f"FDE {prec}")
        for r in ret_rows:
            if not r["variant"].startswith("pooled-"): continue
            bpv = _to_float(r["bytes_per_vector"]); rk = _to_float(r["recall_at_100"])
            if bpv and rk is not None:
                ax.plot(bpv, rk, "s", markersize=10, label=r["variant"])
        if chamfer_r100 is not None:
            ax.axhline(chamfer_r100, color="black", linestyle="--",
                        label=f"chamfer oracle ({chamfer_r100:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("bytes per vector"); ax.set_ylabel("recall@100")
        ax.set_title("Retrieval recall@100 vs storage bytes")
        ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "bytes_vs_recall100.png", dpi=120)
        plt.close(fig)

        # Latency vs recall@10
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in ret_rows:
            if r["variant"] in ("chamfer_oracle",): continue
            lat = _to_float(r["lat_us_per_query"]); rk = _to_float(r["recall_at_10"])
            if lat is not None and rk is not None:
                ax.plot(lat, rk, "o", alpha=0.7, label=f"{r['variant']} ({r['precision']})")
        ax.set_xscale("log")
        ax.set_xlabel("latency μs/query"); ax.set_ylabel("recall@10")
        ax.set_title("Per-query latency vs recall@10")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "latency_vs_recall10.png", dpi=120)
        plt.close(fig)

    if nvme_rows:
        # read amplification + latency
        fig, ax = plt.subplots(figsize=(7, 5))
        odir = [r for r in nvme_rows if r["mode"] == "odirect_4kib"]
        if odir:
            xs = [_to_float(r["logical_record_bytes"]) for r in odir]
            ra = [_to_float(r["read_amplification"]) for r in odir]
            xs_v, ra_v = zip(*[(x, r) for x, r in zip(xs, ra)
                                 if x is not None and r is not None])
            ax.plot(xs_v, ra_v, "o-", color="tab:red", label="O_DIRECT 4 KiB read amp")
        cold = [r for r in nvme_rows if r["mode"] == "cold_pread"]
        if cold:
            xs = [_to_float(r["logical_record_bytes"]) for r in cold]
            ra = [_to_float(r["read_amplification"]) for r in cold]
            xs_v, ra_v = zip(*[(x, r) for x, r in zip(xs, ra)
                                 if x is not None and r is not None])
            ax.plot(xs_v, ra_v, "o-", color="tab:blue", label="cold pread amplification")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("logical record bytes"); ax.set_ylabel("read amplification (x)")
        ax.set_title("NVMe read amplification vs logical record size")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "nvme_read_amplification.png", dpi=120)
        plt.close(fig)

    print(f"plots written to {plot_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
