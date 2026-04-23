#!/usr/bin/env python3.12
"""Hill fit on the saturation-sweep JSONL.

Reads one or more saturation_*.jsonl files, groups by (gpu, config),
fits the Hill function

    S(B) = S_max * B^gamma / (B_half^gamma + B^gamma)

to the measured speedup ratio P_compressed / P_fp16 at each batch
size, and writes a per-GPU per-config parameter table.

Inputs:
    --jsonl /workspace/results/saturation_*.jsonl   (one or more)
    --gpu h100                                      (label for output)
    --out-dir /workspace/results/hill_fits/         (default)

Outputs:
    hill_fits_<gpu>.json        per-config {S_max, B_half, gamma, R2}
    hill_fits_<gpu>.csv         flat table for the paper
    hill_fits_<gpu>.png         overlay of measured points and fitted curves
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def hill(B, S_max, B_half, gamma):
    """Three-parameter Hill curve."""
    return S_max * np.power(B, gamma) / (np.power(B_half, gamma) + np.power(B, gamma))


def load_jsonl(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_ratios(rows):
    """For each (model, B, T) collect tok_per_s per config, then compute
    speedup ratio config/fp16 at each batch."""
    # Index: rows[(model, B, T)] -> {config: tps}
    idx = defaultdict(dict)
    for r in rows:
        if r.get("tok_per_s") is None:
            continue
        idx[(r["model"], r["B"], r["T"])][r["config"]] = r["tok_per_s"]
    out = []
    for (model, B, T), cfgs in idx.items():
        fp16 = cfgs.get("fp16")
        if not fp16:
            continue
        for c, tps in cfgs.items():
            out.append(dict(
                model=model, B=B, T=T, config=c,
                tok_per_s=tps, ratio=tps / fp16,
            ))
    return out


def fit_config(points):
    """Fit Hill to (B, ratio) pairs. Returns (S_max, B_half, gamma, R2, n)."""
    from scipy.optimize import curve_fit

    B = np.array([p["B"] for p in points], dtype=float)
    y = np.array([p["ratio"] for p in points], dtype=float)
    if len(B) < 4:
        return None
    try:
        popt, _ = curve_fit(
            hill, B, y,
            p0=[max(y), 8.0, 1.0],
            bounds=([0, 0.1, 0.1], [10, 256, 5]),
            maxfev=5000,
        )
    except Exception:
        return None
    yhat = hill(B, *popt)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(
        S_max=float(popt[0]),
        B_half=float(popt[1]),
        gamma=float(popt[2]),
        R2=float(r2),
        n=len(points),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--gpu", required=True,
                    help="label for output, e.g. h100 / a100 / mi300x / b200")
    ap.add_argument("--out-dir", default="/workspace/results/hill_fits")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.jsonl)
    points = compute_ratios(rows)

    # Group by config
    by_cfg = defaultdict(list)
    for p in points:
        by_cfg[p["config"]].append(p)

    fits = {}
    for cfg, pts in sorted(by_cfg.items()):
        fit = fit_config(pts)
        fits[cfg] = fit
        if fit:
            print(f"{cfg:>16s}  S_max={fit['S_max']:.3f} "
                  f"B_half={fit['B_half']:.2f} "
                  f"gamma={fit['gamma']:.2f} "
                  f"R2={fit['R2']:.3f}  (n={fit['n']})")
        else:
            print(f"{cfg:>16s}  FIT FAILED (insufficient data)")

    out_json = Path(args.out_dir) / f"hill_fits_{args.gpu}.json"
    with open(out_json, "w") as f:
        json.dump({"gpu": args.gpu, "fits": fits}, f, indent=2)

    out_csv = Path(args.out_dir) / f"hill_fits_{args.gpu}.csv"
    with open(out_csv, "w") as f:
        f.write("config,S_max,B_half,gamma,R2,n\n")
        for cfg, fit in sorted(fits.items()):
            if fit:
                f.write(f"{cfg},{fit['S_max']:.4f},{fit['B_half']:.4f},"
                        f"{fit['gamma']:.4f},{fit['R2']:.4f},{fit['n']}\n")

    # Plot overlay
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = np.linspace(1, 128, 200)
        colors = {
            "fp16": "#6a6a6a",
            "fp8_uncalib": "#9c27b0",
            "fp8_calib_pt": "#ce93d8",
            "fp8_calib_pc": "#ba68c8",
            "asym_uncalib": "#00bcd4",
            "asym_calib": "#00838f",
        }
        for cfg, fit in sorted(fits.items()):
            pts = by_cfg[cfg]
            if fit:
                Bx = np.array([p["B"] for p in pts])
                yx = np.array([p["ratio"] for p in pts])
                ax.scatter(Bx, yx, s=16, color=colors.get(cfg, "#777"),
                           label=f"{cfg}  R²={fit['R2']:.2f}")
                ax.plot(x, hill(x, fit["S_max"], fit["B_half"], fit["gamma"]),
                        color=colors.get(cfg, "#777"), alpha=0.7, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Batch size B")
        ax.set_ylabel("Decode speedup vs FP16")
        ax.set_title(f"Hill saturation fit — {args.gpu.upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        plot_path = Path(args.out_dir) / f"hill_fits_{args.gpu}.png"
        fig.savefig(plot_path, dpi=200)
        print(f"plot: {plot_path}")
    except Exception as e:
        print(f"plot skipped: {e}")

    print(f"fits: {out_json}")
    print(f"csv:  {out_csv}")


if __name__ == "__main__":
    main()
