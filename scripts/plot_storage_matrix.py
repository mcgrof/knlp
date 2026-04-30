"""
Plot NVMe storage matrix benchmark results.

Reads storage_matrix_results.json (from /tmp/bench_storage_matrix.py or
the decode-storage pipeline) and produces four paper-ready figures:

  plots/storage_byte_budget.pdf      -- Figure A: stacked byte budget
  plots/lmcache_nvme_bytes_by_chunk.pdf  -- nvme_bytes vs chunk size
  plots/lmcache_put_latency_by_chunk.pdf -- write p50 vs chunk size
  plots/lmcache_get_latency_by_chunk.pdf -- read  p50 vs chunk size
  plots/raw_nvme_envelope.pdf        -- L1 raw write+read MB/s

Usage:
  python scripts/plot_storage_matrix.py \\
      --results /data/knlp-key-results/lmcache_nvme_storage_matrix_20260430/storage_matrix_results.json \\
      --out plots/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# colour palette
C_FP16 = "#4e79a7"  # blue  – unused FP16 headroom
C_K16 = "#59a14f"  # green – K (BF16, 2 bytes/el)
C_V8 = "#f28e2b"  # orange– V (FP8,  1 byte/el)
C_SCALES = "#e15759"  # red   – scales (1 byte/el)
C_META = "#76b7b2"  # teal  – metadata overhead

C_ALL = "#4e79a7"  # blue  – ALL_NVME series
C_SPLIT = "#f28e2b"  # orange– SPLIT_K_CPU_V_NVME series
C_RAW_W = "#59a14f"  # green – raw write
C_RAW_R = "#e15759"  # red   – raw read


# ── helpers ────────────────────────────────────────────────────────────────


def _label_to_mb(label: str) -> float:
    """'256KB' -> 0.25, '1MB' -> 1.0, '4MB' -> 4.0 …"""
    if label.endswith("KB"):
        return float(label[:-2]) / 1024
    if label.endswith("MB"):
        return float(label[:-2])
    return float(label)


def _save(fig, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / name
    fig.savefig(fpath)
    print(f"  saved {fpath}")
    plt.close(fig)


# ── Figure A: stacked byte budget ─────────────────────────────────────────


def plot_byte_budget(data: dict, out: Path) -> None:
    """
    Stacked bar chart normalised to FP16 K+V = 1.0.

    Left bar  : ALL_NVME  (K16 + V8 + scales + meta → NVMe)
    Right bar : SPLIT_K_CPU_V_NVME (K16 → pinned CPU, V8+scales → NVMe)

    Use one representative chunk (8 MB from L2).  The byte ratios are
    layout-invariant so the choice of chunk does not matter.
    """
    # pick 8 MB row from L2
    row = next(r for r in data["L2_api"] if r["label"] == "8MB")
    fp16 = row["fp16_bytes"]

    # K is BF16 (same width as FP16) → ½ of fp16_bytes
    k_bytes = fp16 / 2
    # V is FP8 → ¼ of fp16_bytes
    v_bytes = fp16 / 4
    # scales (one per 16 values of FP8) → small
    n_el = fp16 // 2  # total elements (K+V share half each)
    v_el = n_el // 2  # V elements
    scale_bytes = math.ceil(v_el / 16)  # 1 scale byte per block of 16

    # meta overhead from actual measurement
    all_nvme = row["policies"]["ALL_NVME"]
    meta_bytes = all_nvme["asym_bytes"] - all_nvme["nvme_bytes"] - all_nvme["cpu_bytes"]

    norm = fp16  # normalise to 1.0

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = [0, 1]
    bar_w = 0.55
    labels = ["ALL_NVME", "SPLIT\n(K→CPU, V→NVMe)"]

    # ALL_NVME: K16 + V8 + scales + meta, all to NVMe
    ax.bar(x[0], k_bytes / norm, bar_w, color=C_K16, label="K (BF16)")
    ax.bar(
        x[0], v_bytes / norm, bar_w, bottom=k_bytes / norm, color=C_V8, label="V (FP8)"
    )
    ax.bar(
        x[0],
        scale_bytes / norm,
        bar_w,
        bottom=(k_bytes + v_bytes) / norm,
        color=C_SCALES,
        label="FP8 scales",
    )
    ax.bar(
        x[0],
        meta_bytes / norm,
        bar_w,
        bottom=(k_bytes + v_bytes + scale_bytes) / norm,
        color=C_META,
        label="metadata",
    )

    # SPLIT: K16 → CPU (hatched), V8+scales+meta → NVMe (solid)
    split = row["policies"]["SPLIT_K_CPU_V_NVME"]
    cpu_frac = split["cpu_bytes"] / norm
    nvme_frac = split["nvme_bytes"] / norm
    meta_frac = meta_bytes / norm

    ax.bar(
        x[1],
        cpu_frac,
        bar_w,
        color=C_K16,
        hatch="//",
        edgecolor="white",
        label="K (pinned CPU)",
    )
    ax.bar(x[1], nvme_frac, bar_w, bottom=cpu_frac, color=C_V8, label="V+scales (NVMe)")
    ax.bar(x[1], meta_frac, bar_w, bottom=(cpu_frac + nvme_frac), color=C_META)

    # horizontal reference at FP16 = 1.0
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(1.5, 1.02, "FP16 K+V = 1.0", va="bottom", fontsize=8, color="black")

    # annotation: NVMe traffic ratio
    ratio = split["nvme_bytes"] / all_nvme["nvme_bytes"]
    ax.annotate(
        f"NVMe traffic ratio\n{ratio:.4f}× ≈ 1/3",
        xy=(x[1], nvme_frac / 2 + cpu_frac),
        xytext=(1.55, nvme_frac / 2 + cpu_frac),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8,
        va="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of FP16 K+V size")
    ax.set_ylim(0, 1.3)
    ax.set_title("Asymmetric KV byte budget (layout invariant)")

    handles, lbls = ax.get_legend_handles_labels()
    # de-dup legend (pinned CPU K entry repeats colour)
    seen = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = True
            dedup_h.append(h)
            dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, loc="upper right", fontsize=8)

    _save(fig, out, "storage_byte_budget.pdf")


# ── Figure B: nvme_bytes by chunk size ────────────────────────────────────


def plot_nvme_bytes_by_chunk(data: dict, out: Path) -> None:
    rows = data["L2_api"]
    sizes_mb = [_label_to_mb(r["label"]) for r in rows]
    all_nvme_mb = [r["policies"]["ALL_NVME"]["nvme_bytes"] / 1e6 for r in rows]
    split_nvme_mb = [
        r["policies"]["SPLIT_K_CPU_V_NVME"]["nvme_bytes"] / 1e6 for r in rows
    ]
    fp16_mb = [r["fp16_bytes"] / 1e6 for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(
        sizes_mb, fp16_mb, "k--", linewidth=1, label="FP16 K+V (reference)", zorder=1
    )
    ax.plot(sizes_mb, all_nvme_mb, "o-", color=C_ALL, label="ALL_NVME", linewidth=1.5)
    ax.plot(
        sizes_mb,
        split_nvme_mb,
        "s--",
        color=C_SPLIT,
        label="SPLIT_K_CPU_V_NVME",
        linewidth=1.5,
    )

    # annotate ratio at 8 MB
    row8 = next(r for r in rows if r["label"] == "8MB")
    ratio = row8["nvme_ratio_split_vs_all"]
    ax.annotate(
        f"ratio = {ratio:.4f}",
        xy=(8, row8["policies"]["SPLIT_K_CPU_V_NVME"]["nvme_bytes"] / 1e6),
        xytext=(12, row8["policies"]["SPLIT_K_CPU_V_NVME"]["nvme_bytes"] / 1e6 + 0.3),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
        fontsize=8,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chunk size (MB, logical FP16 K+V)")
    ax.set_ylabel("NVMe bytes written (MB)")
    ax.set_title("NVMe traffic per put() — layout invariant")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    _save(fig, out, "lmcache_nvme_bytes_by_chunk.pdf")


# ── Figure C: put latency by chunk size ───────────────────────────────────


def plot_put_latency_by_chunk(data: dict, out: Path) -> None:
    rows = data["L2_api"]
    sizes_mb = [_label_to_mb(r["label"]) for r in rows]
    all_w = [r["policies"]["ALL_NVME"]["write_p50_ms"] for r in rows]
    split_w = [r["policies"]["SPLIT_K_CPU_V_NVME"]["write_p50_ms"] for r in rows]

    # L1 raw for comparison
    l1 = data["L1_raw"]
    l1_mb = [_label_to_mb(r["label"]) for r in l1]
    l1_w = [r["write_p50_ms"] for r in l1]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(l1_mb, l1_w, "k:", linewidth=1, label="L1 raw file I/O", zorder=1)
    ax.plot(sizes_mb, all_w, "o-", color=C_ALL, label="ALL_NVME", linewidth=1.5)
    ax.plot(
        sizes_mb,
        split_w,
        "s--",
        color=C_SPLIT,
        label="SPLIT_K_CPU_V_NVME",
        linewidth=1.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chunk size (MB)")
    ax.set_ylabel("Write latency p50 (ms)")
    ax.set_title("put() latency — FADU 8× RAID-0 (device-specific)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    _save(fig, out, "lmcache_put_latency_by_chunk.pdf")


# ── Figure D: get latency by chunk size ───────────────────────────────────


def plot_get_latency_by_chunk(data: dict, out: Path) -> None:
    rows = data["L2_api"]
    sizes_mb = [_label_to_mb(r["label"]) for r in rows]
    all_r = [r["policies"]["ALL_NVME"]["read_p50_ms"] for r in rows]
    split_r = [r["policies"]["SPLIT_K_CPU_V_NVME"]["read_p50_ms"] for r in rows]

    l1 = data["L1_raw"]
    l1_mb = [_label_to_mb(r["label"]) for r in l1]
    l1_r = [r["read_p50_ms"] for r in l1]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(l1_mb, l1_r, "k:", linewidth=1, label="L1 raw file I/O", zorder=1)
    ax.plot(sizes_mb, all_r, "o-", color=C_ALL, label="ALL_NVME", linewidth=1.5)
    ax.plot(
        sizes_mb,
        split_r,
        "s--",
        color=C_SPLIT,
        label="SPLIT_K_CPU_V_NVME",
        linewidth=1.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chunk size (MB)")
    ax.set_ylabel("Read latency p50 (ms)")
    ax.set_title("get() latency — FADU 8× RAID-0 (device-specific)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    _save(fig, out, "lmcache_get_latency_by_chunk.pdf")


# ── Figure E: raw NVMe envelope ───────────────────────────────────────────


def plot_raw_nvme_envelope(data: dict, out: Path) -> None:
    rows = data["L1_raw"]
    sizes_mb = [_label_to_mb(r["label"]) for r in rows]
    write_mbps = [r["write_MBps"] for r in rows]
    read_mbps = [r["read_MBps"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(
        sizes_mb, write_mbps, "o-", color=C_RAW_W, label="Write (fsync)", linewidth=1.5
    )
    ax.plot(sizes_mb, read_mbps, "s--", color=C_RAW_R, label="Read", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Chunk size (MB)")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Raw NVMe envelope — 8× FADU RAID-0, 100 GB XFS slice")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # annotate peak write
    peak_w_i = int(np.argmax(write_mbps))
    ax.annotate(
        f"peak write\n{write_mbps[peak_w_i]:.0f} MB/s",
        xy=(sizes_mb[peak_w_i], write_mbps[peak_w_i]),
        xytext=(sizes_mb[peak_w_i] * 2, write_mbps[peak_w_i] * 0.9),
        arrowprops=dict(arrowstyle="->", color=C_RAW_W, lw=0.8),
        fontsize=8,
        color=C_RAW_W,
    )
    # annotate peak read
    peak_r_i = int(np.argmax(read_mbps))
    ax.annotate(
        f"peak read\n{read_mbps[peak_r_i]:.0f} MB/s",
        xy=(sizes_mb[peak_r_i], read_mbps[peak_r_i]),
        xytext=(sizes_mb[peak_r_i] * 2, read_mbps[peak_r_i] * 1.05),
        arrowprops=dict(arrowstyle="->", color=C_RAW_R, lw=0.8),
        fontsize=8,
        color=C_RAW_R,
    )

    _save(fig, out, "raw_nvme_envelope.pdf")


# ── Figure F: model-shaped KV latency (L3) ────────────────────────────────


def plot_model_shaped_latency(data: dict, out: Path) -> None:
    """
    Bar chart: write and read p50 for ALL_NVME vs SPLIT across model shapes.
    X axis: model × seq_len combinations.
    """
    rows = data.get("L3_model", [])
    if not rows:
        print("  no L3_model data, skipping model-shaped latency plot")
        return

    configs = [f"{r['shape'].split('-')[0]}\nseq={r['seq_len']//1024}k" for r in rows]
    n = len(configs)
    x = np.arange(n)
    bar_w = 0.2

    all_w = [r["policies"]["ALL_NVME"]["write_p50_ms"] for r in rows]
    all_r = [r["policies"]["ALL_NVME"]["read_p50_ms"] for r in rows]
    split_w = [r["policies"]["SPLIT_K_CPU_V_NVME"]["write_p50_ms"] for r in rows]
    split_r = [r["policies"]["SPLIT_K_CPU_V_NVME"]["read_p50_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.8))

    ax.bar(
        x - 1.5 * bar_w, all_w, bar_w, color=C_ALL, alpha=0.9, label="ALL_NVME write"
    )
    ax.bar(
        x - 0.5 * bar_w,
        all_r,
        bar_w,
        color=C_ALL,
        alpha=0.5,
        label="ALL_NVME read",
        hatch="//",
        edgecolor="white",
    )
    ax.bar(
        x + 0.5 * bar_w, split_w, bar_w, color=C_SPLIT, alpha=0.9, label="SPLIT write"
    )
    ax.bar(
        x + 1.5 * bar_w,
        split_r,
        bar_w,
        color=C_SPLIT,
        alpha=0.5,
        label="SPLIT read",
        hatch="//",
        edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Model-shaped KV latency — FADU 8× RAID-0 (device-specific)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, out, "lmcache_model_shaped_latency.pdf")


# ── main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        default="/data/knlp-key-results/lmcache_nvme_storage_matrix_20260430"
        "/storage_matrix_results.json",
        help="Path to storage_matrix_results.json",
    )
    ap.add_argument(
        "--out",
        default="plots/storage",
        help="Output directory for PDF figures",
    )
    args = ap.parse_args()

    with open(args.results) as f:
        raw = f.read()
    # JSON spec doesn't allow Infinity; patch it before parsing
    raw = raw.replace(": Infinity", ": 1e308").replace(":Infinity", ":1e308")
    data = json.loads(raw)

    out = Path(args.out)
    print(f"Writing figures to {out}/")

    plot_byte_budget(data, out)
    plot_nvme_bytes_by_chunk(data, out)
    plot_put_latency_by_chunk(data, out)
    plot_get_latency_by_chunk(data, out)
    plot_raw_nvme_envelope(data, out)
    plot_model_shaped_latency(data, out)

    print("Done.")


if __name__ == "__main__":
    main()
