#!/usr/bin/env python3.12
"""Hill fit per context length on raw decode throughput.

Matches the methodology reported in scaling.tex for H100:

    For each context length T, aggregate (B, tok_per_s) across all
    models and all configs, fit

        S(B) = S_max * B^gamma / (B_half^gamma + B^gamma)

    and report S_max (tok/s), B_half, gamma, R^2.

This is a different knob than fit_hill.py, which groups by config
and fits per-config speedup ratios (useful when there IS separation
between configs; degenerate on H200 where configs overlap).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def hill(B, S_max, B_half, gamma):
    return S_max * np.power(B, gamma) / (np.power(B_half, gamma) + np.power(B, gamma))


def load_rows(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("tok_per_s") is not None:
                rows.append(r)
    return rows


def fit_Ts(rows, configs=None):
    """Fit per T across all models and configs (or a filtered set)."""
    from scipy.optimize import curve_fit

    by_T = defaultdict(list)
    for r in rows:
        if configs and r.get("config") not in configs:
            continue
        by_T[r["T"]].append((r["B"], r["tok_per_s"]))

    fits = {}
    for T, pts in sorted(by_T.items()):
        B = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        if len(B) < 4:
            fits[T] = None
            continue
        try:
            popt, _ = curve_fit(
                hill, B, y,
                p0=[max(y) * 1.5, 16.0, 1.2],
                bounds=([0, 0.1, 0.1], [1e6, 4096, 5]),
                maxfev=10000,
            )
        except Exception as e:
            fits[T] = {"err": str(e)[:100]}
            continue
        yhat = hill(B, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fits[T] = dict(
            S_max=float(popt[0]),
            B_half=float(popt[1]),
            gamma=float(popt[2]),
            R2=float(r2),
            n=int(len(pts)),
        )
    return fits


def fit_per_config_T(rows):
    """Per (config, T) fit — 4 configs * 3 T = 12 fits."""
    from scipy.optimize import curve_fit

    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["config"], r["T"])].append((r["B"], r["tok_per_s"]))

    fits = {}
    for key, pts in sorted(by_key.items()):
        B = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        if len(B) < 4:
            fits[key] = None
            continue
        try:
            popt, _ = curve_fit(
                hill, B, y,
                p0=[max(y) * 1.5, 16.0, 1.2],
                bounds=([0, 0.1, 0.1], [1e6, 4096, 5]),
                maxfev=10000,
            )
        except Exception:
            fits[key] = None
            continue
        yhat = hill(B, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fits[key] = dict(
            S_max=float(popt[0]),
            B_half=float(popt[1]),
            gamma=float(popt[2]),
            R2=float(r2),
            n=int(len(pts)),
        )
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--gpu", required=True,
                    help="label for output, e.g. h100 / h200 / a100 / mi300x")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--exclude-configs", nargs="*", default=[],
                    help="configs to drop before aggregating (e.g. ptpc_fp8)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.jsonl)
    if args.exclude_configs:
        rows = [r for r in rows if r["config"] not in args.exclude_configs]
    configs_seen = sorted({r["config"] for r in rows})
    models_seen = sorted({r["model"] for r in rows})
    print(f"Loaded {len(rows)} ok points across {len(configs_seen)} configs "
          f"x {len(models_seen)} models on {args.gpu}")
    print(f"  configs: {configs_seen}")

    # Aggregate per-T fits (all configs, all models lumped together)
    agg_fits = fit_Ts(rows)
    print(f"\n=== {args.gpu.upper()} — aggregated Hill fit per T ===")
    print(f"{'T':>8s} {'S_max':>10s} {'B_1/2':>8s} {'gamma':>7s} {'R^2':>7s} {'n':>4s}")
    for T, f in sorted(agg_fits.items()):
        if f and "err" not in f:
            print(f"{T:>8d} {f['S_max']:>10.1f} {f['B_half']:>8.2f} "
                  f"{f['gamma']:>7.3f} {f['R2']:>7.4f} {f['n']:>4d}")
        else:
            print(f"{T:>8d} FIT FAILED")

    # Per-(config, T) fits for the separation check
    cfg_fits = fit_per_config_T(rows)
    print(f"\n=== {args.gpu.upper()} — per-(config, T) fits ===")
    print(f"{'config':>14s} {'T':>8s} {'S_max':>10s} {'B_1/2':>8s} "
          f"{'gamma':>7s} {'R^2':>7s} {'n':>4s}")
    for (cfg, T), f in sorted(cfg_fits.items()):
        if f:
            print(f"{cfg:>14s} {T:>8d} {f['S_max']:>10.1f} {f['B_half']:>8.2f} "
                  f"{f['gamma']:>7.3f} {f['R2']:>7.4f} {f['n']:>4d}")

    out_json = Path(args.out_dir) / f"hill_fits_per_t_{args.gpu}.json"
    with open(out_json, "w") as f:
        json.dump({
            "gpu": args.gpu,
            "configs": configs_seen,
            "models": models_seen,
            "n_points": len(rows),
            "aggregated_per_T": agg_fits,
            "per_config_T": {f"{c}|{T}": v for (c, T), v in cfg_fits.items()},
        }, f, indent=2)

    out_csv = Path(args.out_dir) / f"hill_fits_per_t_{args.gpu}.csv"
    with open(out_csv, "w") as f:
        f.write("scope,config,T,S_max,B_half,gamma,R2,n\n")
        for T, fit in sorted(agg_fits.items()):
            if fit and "err" not in fit:
                f.write(f"aggregated,ALL,{T},{fit['S_max']:.2f},"
                        f"{fit['B_half']:.3f},{fit['gamma']:.4f},"
                        f"{fit['R2']:.4f},{fit['n']}\n")
        for (cfg, T), fit in sorted(cfg_fits.items()):
            if fit:
                f.write(f"per_config,{cfg},{T},{fit['S_max']:.2f},"
                        f"{fit['B_half']:.3f},{fit['gamma']:.4f},"
                        f"{fit['R2']:.4f},{fit['n']}\n")
    print(f"\nwrote: {out_json}")
    print(f"wrote: {out_csv}")


if __name__ == "__main__":
    main()
