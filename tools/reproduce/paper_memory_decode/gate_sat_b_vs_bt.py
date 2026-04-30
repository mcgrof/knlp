#!/usr/bin/env python3
"""B vs B*T model comparison gate.

Reads the Hill fit JSON and sweep JSON, then compares two saturation
model families:

  B-model (per-ctx):   S(B | T) = S_max(T) * B / (K_m(T) + B)
  B*T-model (unified): S(B, T) = S_max_bt  * (B*T) / (K_m_bt + B*T)

Reports which model fits better (lower RSS), the ratio of residuals,
and whether the B*T model captures the full 2D surface within tolerance.

The paper claim is that B*T (memory-traffic load) is a better predictor
than B alone, because context length and batch size both scale HBM reads
linearly and interchangeably.

Metrics printed for parent stage (grep-able):
  BT_WIN_FRACTION=<float>  fraction of cells where B*T beats per-ctx B
  RSS_RATIO=<float>        RSS_B / RSS_BT (>1 means B*T wins overall)

Pass criterion: BT_WIN_FRACTION > 0.50 (B*T is at least as good as B).

Exits 0 on pass, 1 on failure, 2 on skip (fit/sweep files missing).
"""
from __future__ import annotations

import json
import os
import sys

SWEEP_PATH = os.environ.get("KNLP_SWEEP_PATH", "/tmp/sat_sweep_results.json")
FIT_PATH = os.environ.get("KNLP_FIT_PATH", "/tmp/sat_hillfit_results.json")
RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/sat_b_vs_bt_results.json")

MIN_BT_WIN_FRACTION = 0.50


def _hill(x: float, s_max: float, k_m: float) -> float:
    return s_max * x / (k_m + x)


def main() -> int:
    missing = [p for p in [SWEEP_PATH, FIT_PATH] if not os.path.exists(p)]
    if missing:
        print(f"SKIP: required files missing: {missing}", flush=True)
        return 2

    with open(SWEEP_PATH) as f:
        sweep = json.load(f)
    with open(FIT_PATH) as f:
        fit = json.load(f)

    cells = [
        c
        for c in sweep.get("cells", [])
        if not c.get("failed") and c.get("tps") is not None
    ]
    if not cells:
        print("SKIP: no valid cells in sweep", flush=True)
        return 2

    per_ctx = {e["ctx"]: e for e in fit.get("per_ctx", [])}
    bt = fit.get("bt_model", {})
    bt_smax = bt.get("s_max", 0.0)
    bt_km = bt.get("k_m", 1.0)

    n_bt_wins = 0
    total_rss_b = 0.0
    total_rss_bt = 0.0
    comparisons: list[dict] = []

    for c in cells:
        actual = c["tps"]
        b = c["batch"]
        ctx = c["ctx"]

        # B-model prediction (uses per-context fit if available).
        ctx_fit = per_ctx.get(ctx)
        if ctx_fit:
            pred_b = _hill(b, ctx_fit["s_max"], ctx_fit["k_m"])
        else:
            pred_b = actual  # no fit available; zero residual (conservative)

        # B*T-model prediction.
        pred_bt = _hill(b * ctx, bt_smax, bt_km)

        err_b = (actual - pred_b) ** 2
        err_bt = (actual - pred_bt) ** 2

        total_rss_b += err_b
        total_rss_bt += err_bt
        bt_wins = err_bt <= err_b
        if bt_wins:
            n_bt_wins += 1

        comparisons.append(
            {
                "batch": b,
                "ctx": ctx,
                "actual": actual,
                "pred_b": pred_b,
                "pred_bt": pred_bt,
                "err_b": err_b,
                "err_bt": err_bt,
                "bt_wins": bt_wins,
            }
        )

    n = len(cells)
    bt_win_frac = n_bt_wins / n if n > 0 else 0.0
    rss_ratio = total_rss_b / total_rss_bt if total_rss_bt > 1e-12 else float("inf")

    print(f"B-model   total RSS: {total_rss_b:.2f}", flush=True)
    print(f"B*T-model total RSS: {total_rss_bt:.2f}", flush=True)
    print(f"RSS ratio (B/B*T):   {rss_ratio:.3f}  (>1 means B*T wins)", flush=True)
    print(f"B*T wins on {n_bt_wins}/{n} cells = {bt_win_frac:.2%}", flush=True)

    print(f"\nBT_WIN_FRACTION={bt_win_frac:.4f}", flush=True)
    print(f"RSS_RATIO={rss_ratio:.4f}", flush=True)

    payload = {
        "n_cells": n,
        "n_bt_wins": n_bt_wins,
        "bt_win_fraction": bt_win_frac,
        "rss_b": total_rss_b,
        "rss_bt": total_rss_bt,
        "rss_ratio": rss_ratio,
        "comparisons": comparisons,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {RESULT_PATH}", flush=True)

    if bt_win_frac < MIN_BT_WIN_FRACTION:
        print(
            f"GATE FAILED: B*T win fraction {bt_win_frac:.2%} < {MIN_BT_WIN_FRACTION:.0%}",
            flush=True,
        )
        return 1

    print("B vs B*T GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
