#!/usr/bin/env python3
"""
Unified FIM analysis tool: Extract from W&B → CSV → Human summary → W&B viz

Workflow:
1. Extract FIM/Fisher/trace metrics from W&B runs
2. Convert to tall-table CSV format
3. Generate human-readable interpretations
4. Save summary to file (for test matrix results)
5. Log insights back to W&B as custom metrics/tables
"""

import argparse
import csv
import json
import math
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb


# ============================================================================
# Step 1: Extract FIM metrics from W&B
# ============================================================================


def extract_fim_from_wandb(entity: str, project: str) -> List[Dict]:
    """Extract FIM/Fisher/trace metrics from all runs in a W&B project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    print(f"Found {len(runs)} runs in {entity}/{project}")

    all_fim_data = []

    for run in runs:
        print(f"\nProcessing run: {run.display_name or run.name}")

        summary = run.summary

        fim_data = {
            "run_info": {
                "entity": entity,
                "project": project,
                "run_name": run.name,
                "display_name": run.display_name or run.name,
                "run_id": run.id,
                "url": run.url,
                "state": run.state,
                "created_at": run.created_at,
            },
            "config": {},
            "summary_metrics": {},
            "history": {},
        }

        # Extract config values
        config = run.config
        for key in config.keys():
            if "fim" in key.lower():
                fim_data["config"][key] = config[key]

        # Extract summary metrics (FIM, Fisher, trace)
        for key in summary.keys():
            if any(keyword in key.lower() for keyword in ["fim", "fisher", "trace"]):
                value = summary[key]
                if isinstance(value, (int, float, str, bool, type(None))):
                    fim_data["summary_metrics"][key] = value

        # Get history for FIM-related metrics (optional, can be slow)
        # Commented out by default to speed things up
        # history_keys = [
        #     k for k in summary.keys()
        #     if any(keyword in k.lower() for keyword in ["fim", "fisher", "trace"])
        # ]
        # if history_keys:
        #     print(f"  Fetching history for {len(history_keys)} metrics...")
        #     try:
        #         history = run.history(keys=history_keys, samples=1000)
        #         for key in history_keys:
        #             if key in history.columns:
        #                 values = history[key].dropna().tolist()
        #                 if values:
        #                     fim_data["history"][key] = values
        #     except Exception as e:
        #         print(f"  Warning: Could not fetch history: {e}")

        all_fim_data.append(fim_data)

    return all_fim_data


# ============================================================================
# Step 2: Convert JSON to CSV tall-table format
# ============================================================================


def json_to_csv(runs_data: List[Dict]) -> pd.DataFrame:
    """Convert JSON FIM data to tall-table CSV format."""
    rows = []

    for run in runs_data:
        run_name = run.get("run_info", {}).get("display_name", "unknown")
        if "summary_metrics" in run:
            for key, value in run["summary_metrics"].items():
                if not key.startswith("fisher/"):
                    continue

                parts = key.split("/")
                # fisher/layerL/headH/metric or fisher/layerL/metric or fisher/metric
                if len(parts) == 2:
                    # fisher/metric (global metrics)
                    layer = "global"
                    head = "all"
                    metric = parts[1]
                elif len(parts) == 3:
                    # fisher/layerL/metric
                    layer = parts[1]  # 'layer0'
                    head = "all"
                    metric = parts[2]
                elif len(parts) == 4:
                    # fisher/layerL/headH/metric
                    layer = parts[1]  # 'layer0'
                    head = parts[2]  # 'head7'
                    metric = parts[3]
                else:
                    continue

                rows.append(
                    {
                        "run_name": run_name,
                        "layer": layer,
                        "head": head,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================================
# Step 3: Analysis functions (from fim_summary.py)
# ============================================================================


def get_scalar_metric(
    df: pd.DataFrame, run: str, layer: str, head: str, metric: str
) -> Optional[float]:
    """Return a single scalar metric for (run, layer, head, metric) if present."""
    rows = df[
        (df["run_name"] == run)
        & (df["layer"] == layer)
        & (df["head"] == head)
        & (df["metric"] == metric)
    ]
    if rows.empty:
        return None
    try:
        return float(rows["value"].iloc[0])
    except Exception:
        return None


def compute_global_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute global FIM statistics for each run."""
    runs = df["run_name"].unique()
    stats: Dict[str, Dict[str, float]] = {}

    for run in runs:
        s: Dict[str, float] = {}

        s["cond_mean"] = get_scalar_metric(
            df, run, "global", "all", "cond_global_mean"
        ) or float("nan")
        s["cond_max"] = get_scalar_metric(
            df, run, "global", "all", "cond_global_max"
        ) or float("nan")

        s["trace_mean"] = get_scalar_metric(
            df, run, "global", "all", "trace_global_mean"
        ) or float("nan")
        s["trace_max"] = get_scalar_metric(
            df, run, "global", "all", "trace_global_max"
        ) or float("nan")

        s["eigmax_mean"] = get_scalar_metric(
            df, run, "global", "all", "eigmax_global_mean"
        ) or float("nan")
        s["eigmax_max"] = get_scalar_metric(
            df, run, "global", "all", "eigmax_global_max"
        ) or float("nan")

        stats[run] = s

    return stats


def compute_low_rank_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Estimate low-rank structure using energy_rk metrics."""
    energy_metrics = [m for m in df["metric"].unique() if m.startswith("energy_r")]
    runs = df["run_name"].unique()
    result: Dict[str, Dict[str, float]] = {}

    if not energy_metrics:
        return result

    def parse_k(name: str) -> int:
        try:
            return int(name.split("energy_r", 1)[1])
        except Exception:
            return 0

    energy_metrics_sorted = sorted(energy_metrics, key=parse_k)

    for run in runs:
        sub = df[(df["run_name"] == run) & df["metric"].isin(energy_metrics_sorted)]
        if sub.empty:
            continue

        mean_energy_by_metric = sub.groupby("metric")["value"].mean().to_dict()

        stats = {}
        for m in ("energy_r8", "energy_r16"):
            if m in mean_energy_by_metric:
                stats[f"{m}_mean"] = float(mean_energy_by_metric[m])

        # Approximate effective rank: smallest k where mean energy >= 0.9
        eff_rank = None
        for m in energy_metrics_sorted:
            k = parse_k(m)
            if k <= 0:
                continue
            e = mean_energy_by_metric.get(m, None)
            if e is not None and e >= 0.9:
                eff_rank = k
                break

        if eff_rank is not None:
            stats["effective_rank_90pct"] = eff_rank

        result[run] = stats

    return result


def find_hotspots(
    df: pd.DataFrame, metric_name: str, top_k: int = 5
) -> List[Tuple[str, str, str, float]]:
    """Find top-k entries with largest value for a metric."""
    sub = df[df["metric"] == metric_name].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("value", ascending=False).head(top_k)
    return [
        (str(r["run_name"]), str(r["layer"]), str(r["head"]), float(r["value"]))
        for _, r in sub.iterrows()
    ]


def find_coldspots(
    df: pd.DataFrame, metric_name: str, bottom_k: int = 5
) -> List[Tuple[str, str, str, float]]:
    """Find bottom-k entries with smallest value for a metric."""
    sub = df[df["metric"] == metric_name].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("value", ascending=True).head(bottom_k)
    return [
        (str(r["run_name"]), str(r["layer"]), str(r["head"]), float(r["value"]))
        for _, r in sub.iterrows()
    ]


def compute_layer_mean_trace(
    df: pd.DataFrame, run_name: Optional[str] = None
) -> List[Tuple[str, str, float]]:
    """
    Compute mean FIM trace per layer across all heads.

    Returns list of (run_name, layer, mean_trace) sorted by mean_trace ascending.
    Layers with lowest mean trace are best compression candidates.
    """
    sub = df[df["metric"] == "trace"].copy()
    if sub.empty:
        return []

    if run_name:
        sub = sub[sub["run_name"] == run_name]

    # Filter to per-head metrics (exclude global)
    sub = sub[sub["layer"].str.startswith("layer")]
    sub = sub[sub["head"].str.startswith("head")]

    if sub.empty:
        return []

    # Group by run and layer, compute mean trace
    grouped = sub.groupby(["run_name", "layer"])["value"].mean().reset_index()
    grouped = grouped.sort_values("value", ascending=True)

    return [
        (str(r["run_name"]), str(r["layer"]), float(r["value"]))
        for _, r in grouped.iterrows()
    ]


def compute_head_mean_trace(
    df: pd.DataFrame, run_name: Optional[str] = None
) -> List[Tuple[str, str, str, float]]:
    """
    Get per-head FIM trace values sorted by trace ascending.

    Returns list of (run_name, layer, head, trace) sorted by trace ascending.
    Heads with lowest trace are best compression candidates.
    """
    sub = df[df["metric"] == "trace"].copy()
    if sub.empty:
        return []

    if run_name:
        sub = sub[sub["run_name"] == run_name]

    # Filter to per-head metrics (exclude global)
    sub = sub[sub["layer"].str.startswith("layer")]
    sub = sub[sub["head"].str.startswith("head")]

    if sub.empty:
        return []

    sub = sub.sort_values("value", ascending=True)

    return [
        (str(r["run_name"]), str(r["layer"]), str(r["head"]), float(r["value"]))
        for _, r in sub.iterrows()
    ]


# ============================================================================
# Step 4: Generate human-readable summary
# ============================================================================


def describe_condition_number(c: float) -> str:
    """Human-readable condition number interpretation."""
    if math.isnan(c):
        return "N/A (missing)"
    if c < 1e3:
        return f"{c:.2e} (well-conditioned / relatively isotropic)"
    if c < 1e5:
        return f"{c:.2e} (moderately anisotropic)"
    if c < 1e7:
        return f"{c:.2e} (strong anisotropy: some very stiff directions)"
    return f"{c:.2e} (extremely ill-conditioned: few stiff directions dominate)"


def describe_energy(e: float) -> str:
    """Human-readable energy interpretation."""
    if math.isnan(e):
        return "N/A (missing)"
    if e >= 0.95:
        return f"{e:.3f} (almost all energy in top modes → very low-rank)"
    if e >= 0.9:
        return f"{e:.3f} (most energy in top modes → low effective rank)"
    if e >= 0.8:
        return f"{e:.3f} (substantial low-rank structure)"
    return f"{e:.3f} (energy more spread across spectrum)"


def generate_summary_text(
    global_stats: Dict[str, Dict[str, float]],
    low_rank_stats: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
) -> str:
    """Generate human-readable FIM summary text."""
    out = StringIO()

    out.write("=" * 80 + "\n")
    out.write("FIM (Fisher Information Matrix) Analysis Summary\n")
    out.write("=" * 80 + "\n\n")

    runs = sorted(global_stats.keys())

    # Per-run summaries
    out.write("=== Per-Run FIM Summaries ===\n\n")
    for run in runs:
        g = global_stats.get(run, {})
        l = low_rank_stats.get(run, {})

        out.write(f"Run: {run}\n")
        out.write("-" * (len(run) + 5) + "\n")

        # Overall curvature scale
        trace_mean = g.get("trace_mean", float("nan"))
        eigmax_mean = g.get("eigmax_mean", float("nan"))
        if not math.isnan(trace_mean):
            out.write(f"  • Mean FIM trace (global): {trace_mean:.3e}\n")
        else:
            out.write(f"  • Mean FIM trace (global): N/A\n")

        if not math.isnan(eigmax_mean):
            out.write(f"  • Mean top eigenvalue (global): {eigmax_mean:.3e}\n")
        else:
            out.write(f"  • Mean top eigenvalue (global): N/A\n")

        # Anisotropy / condition number
        cond_mean = g.get("cond_mean", float("nan"))
        cond_desc = describe_condition_number(cond_mean)
        out.write(f"  • Mean condition number: {cond_desc}\n")

        # Low-rank structure
        if l:
            eff_rank = l.get("effective_rank_90pct", None)
            e8 = l.get("energy_r8_mean", float("nan"))
            e16 = l.get("energy_r16_mean", float("nan"))

            if not math.isnan(e8):
                out.write(f"  • Mean energy_r8:  {describe_energy(e8)}\n")
            if not math.isnan(e16):
                out.write(f"  • Mean energy_r16: {describe_energy(e16)}\n")

            if eff_rank is not None:
                out.write(
                    f"  • Approx. effective rank (90% energy): top ≈{eff_rank} eigenvalues\n"
                )
        else:
            out.write("  • Low-rank structure: energy_rk metrics not available\n")

        out.write("\n")

    # Cross-run comparisons
    if len(runs) >= 2:
        out.write("\n=== Cross-Run Comparisons ===\n\n")

        def sorted_by(key: str):
            return sorted(runs, key=lambda r: global_stats[r].get(key, float("nan")))

        # Curvature comparison
        trace_sorted = sorted_by("trace_mean")
        if trace_sorted and not math.isnan(
            global_stats[trace_sorted[0]].get("trace_mean", float("nan"))
        ):
            low = trace_sorted[0]
            high = trace_sorted[-1]
            t_low = global_stats[low]["trace_mean"]
            t_high = global_stats[high]["trace_mean"]
            if t_low > 0:
                ratio = t_high / t_low
                out.write(
                    f"  • Curvature scale (mean trace): '{high}' ≈ {ratio:.2f}× higher than '{low}'\n"
                )

        # Anisotropy comparison
        cond_sorted = sorted_by("cond_mean")
        if cond_sorted and not math.isnan(
            global_stats[cond_sorted[0]].get("cond_mean", float("nan"))
        ):
            low = cond_sorted[0]
            high = cond_sorted[-1]
            c_low = global_stats[low]["cond_mean"]
            c_high = global_stats[high]["cond_mean"]
            if c_low > 0:
                ratio = c_high / c_low
                out.write(
                    f"  • Anisotropy (mean cond): '{high}' ≈ {ratio:.2f}× more ill-conditioned than '{low}'\n"
                )

        out.write("\n")

    # Hotspots
    out.write("\n=== Hotspots: Heads/Layers with Extreme FIM Metrics ===\n\n")
    for metric in ("cond", "trace", "eigmax"):
        hotspots = find_hotspots(df, metric_name=metric, top_k=5)
        if not hotspots:
            continue

        out.write(f"Top-5 by {metric}:\n")
        for run, layer, head, val in hotspots:
            out.write(
                f"  • {metric} = {val:.3e}  @ run={run}, layer={layer}, head={head}\n"
            )
        out.write("\n")

    # Coldspots - LOWEST FIM trace (best compression candidates)
    out.write("\n" + "=" * 80 + "\n")
    out.write("COMPRESSION TARGETS: Layers/Heads with LOWEST FIM Trace\n")
    out.write("=" * 80 + "\n\n")
    out.write("Low FIM trace = minimal representational work = SAFE TO COMPRESS\n")
    out.write(
        "These heads/layers can be aggressively compressed with minimal loss.\n\n"
    )

    # Show coldspots per run
    for run in runs:
        out.write(f"--- {run} ---\n\n")

        # Layer mean trace for this run
        layer_traces = compute_layer_mean_trace(df, run_name=run)
        if layer_traces:
            out.write("Layers ranked by mean FIM trace (lowest → highest):\n")
            for _, layer, trace_val in layer_traces[:12]:  # All layers
                bar_len = int(trace_val * 40)  # Visual bar
                bar = "█" * bar_len + "░" * (40 - bar_len)
                out.write(f"  {layer:8s}: {trace_val:.4f} |{bar}|\n")
            out.write("\n")

            # Identify best compression candidates
            if len(layer_traces) >= 3:
                out.write("BEST COMPRESSION TARGETS (lowest 3 layers):\n")
                for _, layer, trace_val in layer_traces[:3]:
                    out.write(f"  ★ {layer}: trace={trace_val:.4f}\n")
                out.write("\n")

        # Head-level analysis for this run
        head_traces = compute_head_mean_trace(df, run_name=run)
        if head_traces:
            out.write("Top-10 heads with LOWEST trace (best compression targets):\n")
            for _, layer, head, trace_val in head_traces[:10]:
                out.write(f"  ✓ {layer}/{head}: trace={trace_val:.4f}\n")
            out.write("\n")

            out.write("Top-5 heads with HIGHEST trace (DO NOT COMPRESS):\n")
            for _, layer, head, trace_val in head_traces[-5:]:
                out.write(f"  ✗ {layer}/{head}: trace={trace_val:.4f}\n")
            out.write("\n")

    # Interpretation guide
    out.write("\n" + "=" * 80 + "\n")
    out.write("Quick Interpretation Guide\n")
    out.write("=" * 80 + "\n\n")

    out.write("HIGH TRACE HOTSPOTS:\n")
    out.write("  → Head/layer doing disproportionate representational work\n")
    out.write("  → Cannot be compressed aggressively (critical feature extractor)\n")
    out.write("  → DO NOT prune these heads\n")
    out.write("  → Good candidates for: adaptive precision, dual-tier KV cache\n\n")

    out.write("HIGH EIGMAX HOTSPOTS:\n")
    out.write("  → Extremely sensitive along one specific direction\n")
    out.write("  → Handles rare/high-stakes token interactions\n")
    out.write("  → Top-eigenvector projection is very powerful here\n")
    out.write(
        "  → Best candidates for: KVSplice latent compression, FIM-guided routing\n\n"
    )

    out.write("HIGH CONDITION NUMBER:\n")
    out.write("  → Highly anisotropic (few key directions matter a LOT)\n")
    out.write("  → Natural low-rank structure\n")
    out.write(
        "  → Best compression targets (can safely compress KV cache, projections)\n"
    )
    out.write("  → Prune low-energy directions with near-zero loss\n\n")

    return out.getvalue()


# ============================================================================
# Step 5: FIM-Guided Compression Config Generation
# ============================================================================


def generate_compression_config(
    df: pd.DataFrame,
    target_memory_reduction: float = 0.5,
    d_head: int = 64,
    trace_critical: float = 0.95,
    trace_moderate: float = 0.90,
    cond_excellent: float = 1e7,
    cond_good: float = 1e6,
) -> Dict:
    """
    Generate per-head compression config from FIM analysis.

    Uses FIM metrics (trace, condition number, eigmax) to recommend
    compression ranks for each layer/head combination.

    Strategy:
    - High trace (>0.95): CRITICAL heads, minimal/no compression
    - High cond (>1e7) + low trace: Excellent targets, aggressive compression
    - Moderate cond (>1e6): Good targets, moderate compression
    - Otherwise: Conservative compression

    Args:
        df: FIM metrics dataframe (from json_to_csv)
        target_memory_reduction: Target overall KV memory reduction (0.0-1.0)
        d_head: Dimensionality per head (for memory calculation)
        trace_critical: Trace threshold for critical heads
        trace_moderate: Trace threshold for moderate compression
        cond_excellent: Condition number for excellent compression targets
        cond_good: Condition number for good compression targets

    Returns:
        config: dict with keys:
            - target_memory_reduction: float
            - per_layer_head: dict[(layer, head)] -> {rank, enabled, reason, algo}
            - expected_kv_memory_savings: str (percentage)
            - compression_summary: dict with category counts
    """
    # Get unique run (assume single run for now, or use first run)
    runs = df["run_name"].unique()
    if len(runs) == 0:
        return {"error": "No runs found in dataframe"}

    run_name = runs[0]
    if len(runs) > 1:
        print(
            f"Warning: Multiple runs found, using first: {run_name}. "
            f"Others: {runs[1:]}"
        )

    # Extract all layer/head combinations
    layer_head_df = df[
        (df["run_name"] == run_name) & (df["layer"] != "global") & (df["head"] != "all")
    ].copy()

    if layer_head_df.empty:
        return {"error": "No per-head FIM metrics found"}

    # Get unique (layer, head) pairs
    layer_heads = layer_head_df[["layer", "head"]].drop_duplicates()

    config = {
        "target_memory_reduction": target_memory_reduction,
        "d_head": d_head,
        "algo_default": "kvsplice",
        "per_layer_head": {},
        "compression_summary": {
            "critical": [],
            "excellent": [],
            "good": [],
            "moderate": [],
        },
    }

    total_heads = 0
    total_original_dims = 0
    total_compressed_dims = 0

    for _, row in layer_heads.iterrows():
        layer = row["layer"]
        head = row["head"]

        # Extract metrics for this head
        trace = get_scalar_metric(df, run_name, layer, head, "trace")
        cond = get_scalar_metric(df, run_name, layer, head, "cond")
        eigmax = get_scalar_metric(df, run_name, layer, head, "eigmax")

        # Default values if missing
        if trace is None:
            trace = 0.5
        if cond is None:
            cond = 1e3
        if eigmax is None:
            eigmax = 0.05

        # Apply tiering strategy
        key = f"{layer}/{head}"

        if trace > trace_critical:
            # CRITICAL HEAD - protect from compression
            rank = d_head  # Minimal or no compression
            enabled = False  # Disable compression entirely
            category = "critical"
            reason = f"CRITICAL: trace={trace:.3f} (>{trace_critical})"
            algo = "none"

        elif cond > cond_excellent and trace < trace_moderate:
            # EXCELLENT COMPRESSION TARGET
            rank = 8
            enabled = True
            category = "excellent"
            reason = f"EXCELLENT: cond={cond:.2e}, trace={trace:.3f}"
            algo = "kvsplice"

        elif cond > cond_good:
            # GOOD COMPRESSION TARGET
            rank = 16
            enabled = True
            category = "good"
            reason = f"GOOD: cond={cond:.2e}, trace={trace:.3f}"
            algo = "kvsplice"

        else:
            # MODERATE COMPRESSION
            rank = 32
            enabled = True
            category = "moderate"
            reason = f"MODERATE: cond={cond:.2e}, trace={trace:.3f}"
            algo = "kvsplice"

        # Store config
        config["per_layer_head"][key] = {
            "enabled": enabled,
            "rank": rank,
            "algo": algo,
            "compression_ratio": 1.0 - (rank / d_head) if enabled else 0.0,
            "reason": reason,
            "metrics": {"trace": trace, "cond": cond, "eigmax": eigmax},
        }

        config["compression_summary"][category].append(key)

        # Update memory calculations
        total_heads += 1
        total_original_dims += d_head
        if enabled:
            total_compressed_dims += rank
        else:
            total_compressed_dims += d_head

    # Calculate expected memory savings
    if total_original_dims > 0:
        actual_reduction = 1.0 - (total_compressed_dims / total_original_dims)
        config["expected_kv_memory_savings"] = f"{actual_reduction * 100:.1f}%"
        config["actual_memory_reduction"] = actual_reduction
    else:
        config["expected_kv_memory_savings"] = "N/A"
        config["actual_memory_reduction"] = 0.0

    # Add summary statistics
    config["summary_stats"] = {
        "total_heads": total_heads,
        "critical_heads": len(config["compression_summary"]["critical"]),
        "excellent_targets": len(config["compression_summary"]["excellent"]),
        "good_targets": len(config["compression_summary"]["good"]),
        "moderate_targets": len(config["compression_summary"]["moderate"]),
        "total_original_params": total_original_dims,
        "total_compressed_params": total_compressed_dims,
    }

    return config


def format_compression_config_summary(config: Dict) -> str:
    """Generate human-readable summary of compression config."""
    out = StringIO()

    out.write("=" * 80 + "\n")
    out.write("FIM-Guided KV Cache Compression Configuration\n")
    out.write("=" * 80 + "\n\n")

    if "error" in config:
        out.write(f"ERROR: {config['error']}\n")
        return out.getvalue()

    stats = config.get("summary_stats", {})
    out.write(
        f"Target memory reduction: {config['target_memory_reduction'] * 100:.0f}%\n"
    )
    out.write(f"Expected KV memory savings: {config['expected_kv_memory_savings']}\n\n")

    out.write("=== Compression Summary ===\n\n")
    out.write(f"Total heads analyzed: {stats.get('total_heads', 0)}\n")
    out.write(
        f"  • Critical heads (NO compression): {stats.get('critical_heads', 0)}\n"
    )
    out.write(
        f"  • Excellent targets (rank=8, ~88% compression): {stats.get('excellent_targets', 0)}\n"
    )
    out.write(
        f"  • Good targets (rank=16, ~75% compression): {stats.get('good_targets', 0)}\n"
    )
    out.write(
        f"  • Moderate targets (rank=32, ~50% compression): {stats.get('moderate_targets', 0)}\n"
    )
    out.write("\n")

    out.write(
        f"Total parameters: {stats.get('total_original_params', 0)} → "
        f"{stats.get('total_compressed_params', 0)} "
        f"({config.get('actual_memory_reduction', 0) * 100:.1f}% reduction)\n\n"
    )

    # Show examples from each category
    for category in ["critical", "excellent", "good", "moderate"]:
        heads = config["compression_summary"].get(category, [])
        if not heads:
            continue

        out.write(f"=== {category.upper()} HEADS ===\n\n")
        for key in heads[:5]:  # Show first 5
            head_cfg = config["per_layer_head"][key]
            out.write(f"{key}:\n")
            out.write(f"  Rank: {head_cfg['rank']}\n")
            out.write(f"  Enabled: {head_cfg['enabled']}\n")
            out.write(f"  Algorithm: {head_cfg['algo']}\n")
            out.write(
                f"  Compression ratio: {head_cfg['compression_ratio'] * 100:.1f}%\n"
            )
            out.write(f"  Reason: {head_cfg['reason']}\n")
            out.write("\n")

        if len(heads) > 5:
            out.write(f"  ... and {len(heads) - 5} more\n\n")

    out.write("\n" + "=" * 80 + "\n")
    out.write("Usage Instructions\n")
    out.write("=" * 80 + "\n\n")

    out.write("To use this config for KV cache compression:\n\n")
    out.write("1. Save config to JSON:\n")
    out.write("   python scripts/analyze_fim_metrics.py \\\n")
    out.write("     --entity <entity> --project <project> \\\n")
    out.write("     --generate-compression-config \\\n")
    out.write("     --compression-config-output compression_config.json\n\n")

    out.write("2. Load config in compression plugin:\n")
    out.write("   ```python\n")
    out.write("   import json\n")
    out.write('   with open("compression_config.json") as f:\n')
    out.write("       config = json.load(f)\n")
    out.write("   compressor = KVSpliceCompressor(config)\n")
    out.write("   ```\n\n")

    out.write("3. Apply to HF model:\n")
    out.write("   ```python\n")
    out.write('   model = AutoModelForCausalLM.from_pretrained("gpt2")\n')
    out.write("   wrapped_model = CompressedKVModelWrapper(model, compressor)\n")
    out.write("   ```\n\n")

    return out.getvalue()


# ============================================================================
# Step 6: Log insights to W&B
# ============================================================================


def create_wandb_summary_table(
    global_stats: Dict[str, Dict[str, float]],
    low_rank_stats: Dict[str, Dict[str, float]],
) -> wandb.Table:
    """Create W&B table with FIM summary per run."""
    columns = [
        "run_name",
        "trace_mean",
        "eigmax_mean",
        "cond_mean",
        "energy_r8",
        "energy_r16",
        "eff_rank_90pct",
        "compression_potential",
    ]

    data = []
    for run in sorted(global_stats.keys()):
        g = global_stats[run]
        l = low_rank_stats.get(run, {})

        trace = g.get("trace_mean", float("nan"))
        eigmax = g.get("eigmax_mean", float("nan"))
        cond = g.get("cond_mean", float("nan"))
        e8 = l.get("energy_r8_mean", float("nan"))
        e16 = l.get("energy_r16_mean", float("nan"))
        eff_rank = l.get("effective_rank_90pct", float("nan"))

        # Heuristic compression potential score
        # High cond + high energy_r8 = good compression potential
        if not math.isnan(cond) and not math.isnan(e8):
            comp_score = (math.log10(cond) / 7.0) * e8 * 100  # 0-100 scale
        else:
            comp_score = float("nan")

        data.append(
            [
                run,
                trace if not math.isnan(trace) else "N/A",
                eigmax if not math.isnan(eigmax) else "N/A",
                cond if not math.isnan(cond) else "N/A",
                e8 if not math.isnan(e8) else "N/A",
                e16 if not math.isnan(e16) else "N/A",
                eff_rank if not math.isnan(eff_rank) else "N/A",
                f"{comp_score:.1f}" if not math.isnan(comp_score) else "N/A",
            ]
        )

    return wandb.Table(columns=columns, data=data)


def log_to_wandb(
    run_name: str,
    global_stats: Dict[str, Dict[str, float]],
    low_rank_stats: Dict[str, Dict[str, float]],
    summary_text: str,
):
    """Log FIM insights to W&B run."""
    # Find the run
    if run_name not in global_stats:
        print(f"Warning: Run {run_name} not found in FIM stats")
        return

    g = global_stats[run_name]
    l = low_rank_stats.get(run_name, {})

    # Log scalar metrics
    fim_metrics = {
        "fim/trace_global_mean": g.get("trace_mean", float("nan")),
        "fim/eigmax_global_mean": g.get("eigmax_mean", float("nan")),
        "fim/cond_global_mean": g.get("cond_mean", float("nan")),
        "fim/energy_r8_mean": l.get("energy_r8_mean", float("nan")),
        "fim/energy_r16_mean": l.get("energy_r16_mean", float("nan")),
        "fim/effective_rank_90pct": l.get("effective_rank_90pct", float("nan")),
    }

    # Remove NaN values
    fim_metrics = {k: v for k, v in fim_metrics.items() if not math.isnan(v)}

    print(f"\nLogging FIM metrics to W&B run: {run_name}")
    for k, v in fim_metrics.items():
        print(f"  {k}: {v}")

    # Create summary table
    table = create_wandb_summary_table(global_stats, low_rank_stats)

    # Log HTML panel with full summary
    html_summary = f"<pre>{summary_text}</pre>"

    return {
        "metrics": fim_metrics,
        "table": table,
        "html": html_summary,
    }


# ============================================================================
# Main workflow
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified FIM analysis: W&B → CSV → Summary → Viz"
    )
    parser.add_argument("--entity", type=str, required=True, help="W&B entity name")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for JSON/CSV/summary files",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="fim_summary.txt",
        help="Output file for human-readable summary",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Filter to specific run name (e.g., gpt2_mla_fineweb)",
    )
    parser.add_argument(
        "--generate-compression-config",
        action="store_true",
        help="Generate FIM-guided compression config",
    )
    parser.add_argument(
        "--compression-config-output",
        type=str,
        default="compression_config.json",
        help="Output file for compression config JSON",
    )
    parser.add_argument(
        "--target-memory-reduction",
        type=float,
        default=0.5,
        help="Target KV memory reduction (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--d-head",
        type=int,
        default=64,
        help="Head dimension for memory calculation (default: 64)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract from W&B
    print("\n" + "=" * 80)
    print("Step 1: Extracting FIM metrics from W&B")
    print("=" * 80)
    runs_data = extract_fim_from_wandb(args.entity, args.project)

    # Save JSON
    json_file = output_dir / f"fim_metrics_{args.project}.json"
    with open(json_file, "w") as f:
        json.dump(runs_data, f, indent=2)
    print(f"\n✓ Saved JSON: {json_file}")

    # Step 2: Convert to CSV
    print("\n" + "=" * 80)
    print("Step 2: Converting to CSV tall-table format")
    print("=" * 80)
    df = json_to_csv(runs_data)

    if df.empty:
        print("ERROR: No FIM metrics found in runs!")
        sys.exit(1)

    csv_file = output_dir / f"fim_summary_{args.project}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    print(f"  Rows: {len(df)}")
    print(f"  Runs: {df['run_name'].nunique()}")

    # Filter to specific run if requested
    if args.run_name:
        available_runs = df["run_name"].unique().tolist()
        if args.run_name not in available_runs:
            print(f"\nERROR: Run '{args.run_name}' not found!")
            print(f"Available runs: {available_runs}")
            sys.exit(1)
        print(f"\n✓ Filtering to run: {args.run_name}")
        df = df[df["run_name"] == args.run_name]
        print(f"  Filtered rows: {len(df)}")

    # Step 3: Analyze
    print("\n" + "=" * 80)
    print("Step 3: Analyzing FIM metrics")
    print("=" * 80)
    global_stats = compute_global_stats(df)
    low_rank_stats = compute_low_rank_stats(df)

    # Step 4: Generate summary
    print("\n" + "=" * 80)
    print("Step 4: Generating human-readable summary")
    print("=" * 80)
    summary_text = generate_summary_text(global_stats, low_rank_stats, df)

    summary_file = output_dir / args.output_summary
    with open(summary_file, "w") as f:
        f.write(summary_text)
    print(f"✓ Saved summary: {summary_file}")

    # Print summary to stdout
    print("\n" + summary_text)

    # Step 5: Generate compression config (optional)
    if args.generate_compression_config:
        print("\n" + "=" * 80)
        print("Step 5: Generating FIM-Guided Compression Config")
        print("=" * 80)

        compression_config = generate_compression_config(
            df,
            target_memory_reduction=args.target_memory_reduction,
            d_head=args.d_head,
        )

        config_file = output_dir / args.compression_config_output
        with open(config_file, "w") as f:
            json.dump(compression_config, f, indent=2)
        print(f"✓ Saved compression config: {config_file}")

        # Generate and save human-readable config summary
        config_summary = format_compression_config_summary(compression_config)
        config_summary_file = output_dir / args.compression_config_output.replace(
            ".json", "_summary.txt"
        )
        with open(config_summary_file, "w") as f:
            f.write(config_summary)
        print(f"✓ Saved config summary: {config_summary_file}")

        # Print summary to stdout
        print("\n" + config_summary)

    # Step 6: Generate W&B viz instructions
    print("\n" + "=" * 80)
    print("Step 6: W&B Visualization Recommendations")
    print("=" * 80)
    print("\nTo log FIM insights to W&B, add this to your training code:")
    print("\n```python")
    print("import wandb")
    print("from scripts.analyze_fim_metrics import create_wandb_summary_table")
    print()
    print("# Create FIM summary table")
    print(f"table = create_wandb_summary_table(global_stats, low_rank_stats)")
    print('wandb.log({"fim/summary_table": table})')
    print()
    print("# Log key metrics")
    print("wandb.log({")
    print('    "fim/trace_global_mean": trace_mean,')
    print('    "fim/cond_global_mean": cond_mean,')
    print('    "fim/effective_rank": eff_rank,')
    print('    "fim/compression_potential": comp_score,')
    print("})")
    print()
    print("# Create custom HTML panel")
    print('html_panel = wandb.Html(f"<pre>{summary_text}</pre>")')
    print('wandb.log({"fim/interpretation": html_panel})')
    print("```")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
