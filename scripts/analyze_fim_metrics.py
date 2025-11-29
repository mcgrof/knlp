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
# Step 5: Log insights to W&B
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

    # Step 5: Generate W&B viz instructions
    print("\n" + "=" * 80)
    print("Step 5: W&B Visualization Recommendations")
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
