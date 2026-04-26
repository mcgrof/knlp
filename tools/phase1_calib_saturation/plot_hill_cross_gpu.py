#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hill saturation overlay across A100 / H100 / H200.

Reads the per-GPU saturation_*.jsonl directories and produces a 3-panel
figure (one panel per context length T) overlaying the measured points
and fitted Hill curve for each GPU.  Used in scaling.tex as the visual
companion to the cross-GPU Hill fit table.
"""
# Standard
import argparse
import json
from collections import defaultdict
from pathlib import Path

# Third Party
import numpy as np

# matplotlib loaded only if --out is set, so the script can be imported
# without a display.


def hill(B, S_max, B_half, gamma):
    return S_max * np.power(B, gamma) / (np.power(B_half, gamma) + np.power(B, gamma))


def load_rows(jsonl_paths):
    rows = []
    for p in jsonl_paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("tok_per_s") is not None:
                rows.append(r)
    return rows


def fit_per_T(rows):
    # Third Party
    from scipy.optimize import curve_fit

    by_T = defaultdict(list)
    for r in rows:
        by_T[r["T"]].append((r["B"], r["tok_per_s"]))
    fits = {}
    for T, pts in sorted(by_T.items()):
        B = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        try:
            popt, _ = curve_fit(
                hill, B, y,
                p0=[max(y) * 1.5, 16.0, 1.2],
                bounds=([0, 0.1, 0.1], [1e7, 10000, 5]),
                maxfev=20000,
            )
            yhat = hill(B, *popt)
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fits[T] = dict(
                S_max=float(popt[0]),
                B_half=float(popt[1]),
                gamma=float(popt[2]),
                R2=float(r2),
                B=B.tolist(),
                y=y.tolist(),
            )
        except Exception:
            fits[T] = None
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpu",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        required=True,
        help="repeat: --gpu A100 /path/to/sat_a100/  --gpu H100 /path/...",
    )
    ap.add_argument("--out", required=True, help="output PDF/PNG path")
    ap.add_argument("--bw-tbs", action="append", default=[], type=str,
                    help="optional bandwidth label for legend, e.g. 'A100=2.0'")
    args = ap.parse_args()

    # Third Party
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bw = {kv.split("=")[0]: kv.split("=")[1] for kv in args.bw_tbs if "=" in kv}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    colors = {"A100": "#2185d0", "H100": "#ea580c", "H200": "#15803d"}
    Ts = (1024, 4096, 16384)

    legend_handles = {}
    for label, gdir in args.gpu:
        rows = load_rows(sorted(Path(gdir).glob("saturation_*.jsonl")))
        fits = fit_per_T(rows)
        for ax, T in zip(axes, Ts):
            f = fits.get(T)
            if f is None:
                continue
            B = np.array(f["B"])
            y = np.array(f["y"])
            color = colors.get(label, "#666")
            sc = ax.scatter(B, y, s=8, color=color, alpha=0.45,
                            label=label, edgecolors="none")
            # Fitted curve
            x = np.linspace(1, max(64, B.max() * 1.05), 200)
            ax.plot(x, hill(x, f["S_max"], f["B_half"], f["gamma"]),
                    color=color, linewidth=1.7, alpha=0.9)
            legend_handles[label] = sc

    for ax, T in zip(axes, Ts):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size B")
        ax.set_title(f"T = {T:,}")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Decode throughput (tok/s)")

    # Build legend with per-GPU labels (and optionally bandwidth)
    handles, labels = [], []
    for lbl in ("A100", "H100", "H200"):
        if lbl in legend_handles:
            handles.append(legend_handles[lbl])
            if lbl in bw:
                labels.append(f"{lbl} ({bw[lbl]} TB/s)")
            else:
                labels.append(lbl)
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(handles),
               frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
