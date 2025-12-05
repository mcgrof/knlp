#!/usr/bin/env python3
"""
rank_scheduler.py

FIM + spline-based rank allocator for KV compression.

Goal:
  Take per-head importance metrics (e.g. FIM eigenvalues) and produce a
  per-(layer, head) rank assignment, using a smooth spline mapping:

      importance  ──>  normalized importance [0,1]
                    ──> capacity fraction via spline
                    ──> discrete rank in allowed_ranks

Intended usage:
  1. Run your FIM / eigenvalue analysis to produce a CSV with per-head stats.
     The script supports two common layouts:

     (A) Wide format:
         layer,head,max_eig,fim_trace,energy,...
     (B) Long/metric format:
         run_name,layer,head,metric,value
         where metric ∈ {max_eig, fim_trace, ...}

  2. Choose which metric to use as "importance" (e.g. max_eig).

  3. Call this script:
     python rank_scheduler.py \
         --fim-csv fim_stats.csv \
         --metric max_eig \
         --d-head 64 \
         --target-compression 0.5 \
         --allowed-ranks 4 8 16 32 64 \
         --out ranks_gpt2_fim_spline.json

  4. Your KVSplice / latent-KV backend loads the JSON and
     uses rank[layer][head] when constructing projections.

Dependencies:
  - numpy
  - pandas
  - scipy (for UnivariateSpline)

If you hate SciPy, you can swap the spline for a piecewise-linear mapping
(see the build_capacity_spline() helper).
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SplinePolicy:
    """
    Describes how normalized importance ∈ [0,1] maps to capacity fraction ∈ [0,1].

    We specify a few control points (x_i, y_i) and fit a smooth spline that
    interpolates them. You can tweak these to encode your "taste":

      - At x=0.0 (least important heads), use a low capacity fraction
      - At x=1.0 (most important heads), capacity fraction ~ 1.0
    """

    x: List[float]
    y: List[float]
    min_frac: float = 0.05  # minimum capacity fraction before rank snapping
    max_frac: float = 1.0  # maximum allowed fraction before snapping


# ---------------------------------------------------------------------------
# CSV loading / importance extraction
# ---------------------------------------------------------------------------


def load_importance_table(
    path: str,
    metric: str,
    long_metric_column: str = "metric",
    long_value_column: str = "value",
) -> pd.DataFrame:
    """
    Load FIM/eigenvalue CSV and return a DataFrame with at least:
        layer, head, importance

    Supports two layouts:

      1) Wide:
          layer, head, max_eig, fim_trace, ...
         -> importance = df[metric]

      2) Long:
          run_name, layer, head, metric, value
         -> importance = df[df.metric == metric].pivot(...)

    We auto-detect based on presence of `metric` column.
    """
    df = pd.read_csv(path)

    if long_metric_column in df.columns:
        # Long format: filter by metric name, then pivot
        df_metric = df[df[long_metric_column] == metric].copy()
        if df_metric.empty:
            raise ValueError(f"No rows found for metric='{metric}' in {path}")
        # Expect at least these columns:
        for col in ["layer", "head", long_value_column]:
            if col not in df_metric.columns:
                raise ValueError(
                    f"Expected column '{col}' in long-format CSV for metric='{metric}'"
                )
        df_imp = df_metric[["layer", "head", long_value_column]].copy()
        df_imp = df_imp.rename(columns={long_value_column: "importance"})
    else:
        # Wide format: metric is a column name
        if metric not in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found as a column in {path}. "
                f"Available: {sorted(df.columns)}"
            )
        for col in ["layer", "head"]:
            if col not in df.columns:
                raise ValueError(
                    f"Expected columns 'layer' and 'head' in wide-format CSV."
                )
        df_imp = df[["layer", "head", metric]].copy()
        df_imp = df_imp.rename(columns={metric: "importance"})

    # Drop duplicates if any (e.g. multiple runs); you can choose a reduction.
    # Here we take mean per (layer, head).
    df_imp = df_imp.groupby(["layer", "head"], as_index=False)["importance"].mean()

    return df_imp


# ---------------------------------------------------------------------------
# Importance normalization and spline mapping
# ---------------------------------------------------------------------------


def normalize_importance(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Map raw importance values -> [0,1].

    If all values are equal, returns all 0.5.
    """
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < eps:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def build_capacity_spline(policy: SplinePolicy) -> UnivariateSpline:
    """
    Build a 1D spline mapping normalized importance -> capacity fraction.

    We use an interpolating cubic spline (k=3, s=0) through the control points.
    """
    x = np.array(policy.x, dtype=np.float64)
    y = np.array(policy.y, dtype=np.float64)

    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("Spline x knots must be in [0,1].")
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("Spline y values (capacity fraction) must be in [0,1].")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Spline x knots must be strictly increasing.")

    spline = UnivariateSpline(x, y, k=min(3, len(x) - 1), s=0.0)
    return spline


def map_importance_to_capacity(
    imp_norm: np.ndarray,
    spline: UnivariateSpline,
    policy: SplinePolicy,
) -> np.ndarray:
    """
    Apply spline to normalized importance to get capacity fraction ∈ [0,1],
    then clamp to [policy.min_frac, policy.max_frac].
    """
    cap = spline(imp_norm)
    cap = np.clip(cap, policy.min_frac, policy.max_frac)
    return cap.astype(np.float32)


# ---------------------------------------------------------------------------
# Rank snapping / global compression adjustment
# ---------------------------------------------------------------------------


def snap_to_allowed_ranks(
    capacity_fraction: np.ndarray,
    d_head: int,
    allowed_ranks: List[int],
) -> np.ndarray:
    """
    For each capacity fraction f, convert to continuous rank f * d_head,
    then snap to nearest allowed rank in allowed_ranks.

    Returns: array of ints of same shape as capacity_fraction.
    """
    allowed = np.array(sorted(set(allowed_ranks)), dtype=np.int32)
    cont_ranks = np.clip(capacity_fraction * d_head, 1, d_head)
    snapped = []
    for r in cont_ranks:
        idx = np.argmin(np.abs(allowed - r))
        snapped.append(int(allowed[idx]))
    return np.array(snapped, dtype=np.int32)


def adjust_for_target_compression(
    capacity_fraction: np.ndarray,
    d_head: int,
    target_compression: float,
    min_frac: float,
    max_frac: float,
) -> np.ndarray:
    """
    Scale capacity fractions so that average (rank/d_head) ≈ target_compression.

    target_compression is the desired average rank fraction:
      e.g. 0.5 -> average rank ~ 0.5 * d_head

    Returns: scaled capacity_fraction (still continuous; snapping happens later).
    """
    # current average fraction
    current_frac = float(capacity_fraction.mean())
    if current_frac <= 0.0:
        return np.full_like(capacity_fraction, target_compression, dtype=np.float32)

    scale = target_compression / current_frac
    scaled = capacity_fraction * scale
    scaled = np.clip(scaled, min_frac, max_frac)
    return scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build_rank_schedule(
    df_imp: pd.DataFrame,
    d_head: int,
    allowed_ranks: List[int],
    spline_policy: SplinePolicy,
    target_compression: float,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Given importance table and spline policy, produce:

      - DataFrame with columns:
          layer, head, importance, importance_norm, capacity_frac, rank

      - metadata dict for JSON export
    """
    if not len(df_imp):
        raise ValueError("Empty importance table.")

    values = df_imp["importance"].to_numpy(dtype=np.float64)
    imp_norm = normalize_importance(values)

    spline = build_capacity_spline(spline_policy)
    capacity_frac = map_importance_to_capacity(imp_norm, spline, spline_policy)

    # Global rescaling to hit target compression on average
    if target_compression is not None:
        capacity_frac = adjust_for_target_compression(
            capacity_frac,
            d_head=d_head,
            target_compression=target_compression,
            min_frac=spline_policy.min_frac,
            max_frac=spline_policy.max_frac,
        )

    ranks = snap_to_allowed_ranks(
        capacity_frac, d_head=d_head, allowed_ranks=allowed_ranks
    )

    df_out = df_imp.copy()
    df_out["importance_norm"] = imp_norm
    df_out["capacity_frac"] = capacity_frac
    df_out["rank"] = ranks

    meta = {
        "d_head": d_head,
        "allowed_ranks": allowed_ranks,
        "target_compression": target_compression,
        "spline_policy": {
            "x": spline_policy.x,
            "y": spline_policy.y,
            "min_frac": spline_policy.min_frac,
            "max_frac": spline_policy.max_frac,
        },
        "avg_rank": float(ranks.mean()),
        "avg_rank_fraction": float(ranks.mean() / d_head),
        "min_rank": int(ranks.min()),
        "max_rank": int(ranks.max()),
    }

    return df_out, meta


def export_rank_json(
    df_out: pd.DataFrame,
    meta: Dict,
    path: str,
) -> None:
    """
    Export per-head ranks and metadata as a JSON blob.

    Format:

    {
      "meta": { ... },
      "ranks": [
        {
          "layer": 0,
          "head": 0,
          "rank": 16,
          "importance": 123.4,
          "importance_norm": 0.42,
          "capacity_frac": 0.25
        },
        ...
      ]
    }
    """
    ranks_list = []
    for row in df_out.itertuples(index=False):
        ranks_list.append(
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "rank": int(row.rank),
                "importance": float(row.importance),
                "importance_norm": float(row.importance_norm),
                "capacity_frac": float(row.capacity_frac),
            }
        )

    blob = {
        "meta": meta,
        "ranks": ranks_list,
    }

    with open(path, "w") as f:
        json.dump(blob, f, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FIM + spline-based rank scheduler for KV compression."
    )
    p.add_argument(
        "--fim-csv",
        type=str,
        required=True,
        help="Path to CSV with per-head importance metrics.",
    )
    p.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric name to use as importance (column name OR long-format metric).",
    )
    p.add_argument(
        "--d-head",
        type=int,
        required=True,
        help="Head dimension (e.g. 64 for GPT-2).",
    )
    p.add_argument(
        "--allowed-ranks",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Discrete ranks to snap to. Default: 4 8 16 32 64",
    )
    p.add_argument(
        "--target-compression",
        type=float,
        default=0.5,
        help=(
            "Desired average rank fraction: 0.5 => avg rank ~ 0.5 * d_head. "
            "Set None or <0 to disable global rescaling."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path for rank schedule.",
    )
    # Spline control points: we allow override but provide sane defaults
    p.add_argument(
        "--spline-knots",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.7, 1.0],
        help="Normalized importance x-knots in [0,1]. Default: 0.0 0.3 0.7 1.0",
    )
    p.add_argument(
        "--spline-values",
        type=float,
        nargs="+",
        default=[0.25, 0.35, 0.7, 1.0],
        help=(
            "Capacity fractions at knots. Same length as --spline-knots. "
            "Default: 0.25 0.35 0.7 1.0"
        ),
    )
    p.add_argument(
        "--min-frac",
        type=float,
        default=0.05,
        help="Minimum capacity fraction before snapping. Default: 0.05",
    )
    p.add_argument(
        "--max-frac",
        type=float,
        default=1.0,
        help="Maximum capacity fraction before snapping. Default: 1.0",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.spline_knots) != len(args.spline_values):
        raise ValueError(
            f"--spline-knots and --spline-values must have same length, "
            f"got {len(args.spline_knots)} and {len(args.spline_values)}"
        )

    target_compression = (
        args.target_compression
        if args.target_compression is not None and args.target_compression > 0
        else None
    )

    spline_policy = SplinePolicy(
        x=args.spline_knots,
        y=args.spline_values,
        min_frac=args.min_frac,
        max_frac=args.max_frac,
    )

    df_imp = load_importance_table(args.fim_csv, metric=args.metric)

    df_out, meta = build_rank_schedule(
        df_imp=df_imp,
        d_head=args.d_head,
        allowed_ranks=args.allowed_ranks,
        spline_policy=spline_policy,
        target_compression=target_compression,
    )

    print(
        f"[rank_scheduler] Heads: {len(df_out)}, "
        f"avg rank={meta['avg_rank']:.2f}, "
        f"avg rank fraction={meta['avg_rank_fraction']:.3f}"
    )

    export_rank_json(df_out, meta, args.out)
    print(f"[rank_scheduler] Saved rank schedule to {args.out}")


if __name__ == "__main__":
    main()
