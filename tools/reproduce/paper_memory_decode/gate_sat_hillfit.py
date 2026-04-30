#!/usr/bin/env python3
"""Hill saturation model fit gate.

Reads the sweep JSON produced by gate_sat_sweep.py and fits a Hill
saturation curve to tok/s vs batch size for each context length:

    S(B) = S_max * B / (K_m + B)

where S_max is the saturation throughput plateau and K_m is the
half-maximum batch size (S(K_m) = S_max / 2).

Also fits the alternative B*T model:

    S(B, T) = S_max_bt * (B * T) / (K_m_bt + B * T)

where T is context length (tokens).  Both models are compared by
residual sum of squares (RSS) and R².

Metrics printed for parent stage (grep-able):
  HILL_R2_MIN=<float>    minimum R² across context lengths
  HILL_R2_MEAN=<float>   mean R² across context lengths
  BT_R2=<float>          R² of the unified B*T model

Pass criterion: HILL_R2_MIN > 0.85 (poor fit flags bad sweep data).

Exits 0 on pass, 1 on failure, 2 on skip (sweep file missing).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import NamedTuple

SWEEP_PATH = os.environ.get("KNLP_SWEEP_PATH", "/tmp/sat_sweep_results.json")
RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/sat_hillfit_results.json")

# Pass threshold.
MIN_R2 = 0.85


class HillFit(NamedTuple):
    s_max: float
    k_m: float
    r2: float
    rss: float


def _hill(b: float, s_max: float, k_m: float) -> float:
    return s_max * b / (k_m + b)


def _fit_hill(bs: list[float], tps: list[float]) -> HillFit:
    """Least-squares Hill fit via gradient-free Nelder-Mead."""
    try:
        from scipy.optimize import curve_fit
        import numpy as np

        popt, _ = curve_fit(
            lambda b, s_max, k_m: s_max * b / (k_m + b),
            bs,
            tps,
            p0=[max(tps) * 1.2, float(bs[len(bs) // 2])],
            bounds=([0, 0], [1e6, 1e6]),
            maxfev=5000,
        )
        s_max, k_m = float(popt[0]), float(popt[1])
        predicted = [_hill(b, s_max, k_m) for b in bs]
        ss_res = sum((y - yh) ** 2 for y, yh in zip(tps, predicted))
        ss_tot = sum((y - sum(tps) / len(tps)) ** 2 for y in tps)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return HillFit(s_max=s_max, k_m=k_m, r2=r2, rss=ss_res)
    except Exception:
        pass

    # Fallback: closed-form two-point estimate using median pairs.
    n = len(bs)
    if n < 2:
        return HillFit(s_max=float(tps[0]) * 2, k_m=float(bs[0]), r2=0.0, rss=0.0)
    b1, s1 = bs[0], tps[0]
    b2, s2 = bs[-1], tps[-1]
    # s1 = s_max * b1 / (k_m + b1), s2 = s_max * b2 / (k_m + b2)
    # => k_m = (s2 * b2 - s1 * b1) / (s1 - s2) (simplified)
    denom = s1 - s2
    if abs(denom) < 1e-9:
        return HillFit(s_max=float(s2) * 1.1, k_m=float(b1), r2=0.0, rss=0.0)
    k_m = max((s2 * b2 - s1 * b1) / denom, 0.1)
    s_max = s1 * (k_m + b1) / b1
    predicted = [_hill(b, s_max, k_m) for b in bs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(tps, predicted))
    ss_tot = sum((y - sum(tps) / len(tps)) ** 2 for y in tps)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return HillFit(s_max=s_max, k_m=k_m, r2=r2, rss=ss_res)


def _fit_bt_model(cells: list[dict]) -> tuple[float, float, float]:
    """Fit unified S(B,T) = S_max * (B*T) / (K_m + B*T).

    Returns (s_max, k_m, r2)."""
    ok = [c for c in cells if not c.get("failed") and c.get("tps") is not None]
    if len(ok) < 4:
        return 0.0, 0.0, 0.0
    xs = [c["batch"] * c["ctx"] for c in ok]
    ys = [c["tps"] for c in ok]
    fit = _fit_hill(xs, ys)
    return fit.s_max, fit.k_m, fit.r2


def main() -> int:
    if not os.path.exists(SWEEP_PATH):
        print(f"SKIP: sweep results not found at {SWEEP_PATH}", flush=True)
        return 2

    with open(SWEEP_PATH) as f:
        sweep = json.load(f)

    cells = sweep.get("cells", [])
    ctx_lengths = sorted({c["ctx"] for c in cells})
    batch_sizes = sorted({c["batch"] for c in cells})

    per_ctx: list[dict] = []
    r2_values: list[float] = []

    for ctx in ctx_lengths:
        ctx_cells = sorted(
            [
                c
                for c in cells
                if c["ctx"] == ctx and not c.get("failed") and c.get("tps") is not None
            ],
            key=lambda c: c["batch"],
        )
        if len(ctx_cells) < 3:
            print(
                f"  ctx={ctx}: too few valid cells ({len(ctx_cells)}), skipping",
                flush=True,
            )
            continue
        bs = [float(c["batch"]) for c in ctx_cells]
        tps = [float(c["tps"]) for c in ctx_cells]
        fit = _fit_hill(bs, tps)
        print(
            f"  ctx={ctx:6d}: S_max={fit.s_max:7.1f} tok/s  "
            f"K_m={fit.k_m:5.1f}  R²={fit.r2:.3f}",
            flush=True,
        )
        r2_values.append(fit.r2)
        per_ctx.append(
            {
                "ctx": ctx,
                "s_max": fit.s_max,
                "k_m": fit.k_m,
                "r2": fit.r2,
                "rss": fit.rss,
                "n_cells": len(ctx_cells),
            }
        )

    if not r2_values:
        print("GATE FAILED: no valid context slices to fit", flush=True)
        return 1

    r2_min = min(r2_values)
    r2_mean = sum(r2_values) / len(r2_values)

    bt_smax, bt_km, bt_r2 = _fit_bt_model(cells)
    print(
        f"\nB*T unified model: S_max={bt_smax:.1f}  K_m={bt_km:.1f}  R²={bt_r2:.3f}",
        flush=True,
    )

    print(f"\nHILL_R2_MIN={r2_min:.4f}", flush=True)
    print(f"HILL_R2_MEAN={r2_mean:.4f}", flush=True)
    print(f"BT_R2={bt_r2:.4f}", flush=True)

    payload = {
        "model": sweep.get("model", ""),
        "per_ctx": per_ctx,
        "r2_min": r2_min,
        "r2_mean": r2_mean,
        "bt_model": {"s_max": bt_smax, "k_m": bt_km, "r2": bt_r2},
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {RESULT_PATH}", flush=True)

    if r2_min < MIN_R2:
        print(
            f"GATE FAILED: min R²={r2_min:.3f} < {MIN_R2}; "
            "sweep data may be noisy or model loading failed",
            flush=True,
        )
        return 1

    print("HILL FIT GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
