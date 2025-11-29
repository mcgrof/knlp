#!/usr/bin/env python3
"""
Helper functions to log brilliant FIM visualizations to W&B.

Usage:
    from scripts.fim_wandb_viz import log_fim_to_wandb

    log_fim_to_wandb(
        entity="mcgrof-citizen",
        project="gpt2-ra-v2-h100",
        run_id="abc123",  # W&B run ID
        df=fim_dataframe,  # tall-table CSV as DataFrame
        global_stats=global_stats,
        low_rank_stats=low_rank_stats,
    )
"""

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import wandb


def compute_compression_score(cond: float, energy_r8: float) -> float:
    """
    Heuristic compression potential score (0-100).

    High condition number + high energy in top 8 modes = very compressible.

    Args:
        cond: Condition number
        energy_r8: Energy captured by top 8 eigenvalues

    Returns:
        Compression score (0-100)
    """
    if math.isnan(cond) or math.isnan(energy_r8):
        return float("nan")

    # Normalize condition number: cond=1e7 → 1.0, cond=1e3 → 0.0
    cond_norm = min(1.0, (math.log10(cond) - 3.0) / 4.0)

    # Combine with energy
    score = cond_norm * energy_r8 * 100
    return max(0.0, min(100.0, score))


def compute_efficiency_score(
    trace: float, baseline_trace: Optional[float] = None
) -> float:
    """
    Efficiency score: lower trace = more efficient.

    Args:
        trace: FIM trace (sum of eigenvalues)
        baseline_trace: Optional baseline for comparison

    Returns:
        Efficiency score (0-100, higher is better)
    """
    if math.isnan(trace):
        return float("nan")

    if baseline_trace is not None and not math.isnan(baseline_trace):
        # Relative efficiency vs baseline
        if baseline_trace > 0:
            efficiency = (1.0 - trace / baseline_trace) * 100
            return efficiency

    # Absolute efficiency (assume trace ~1.0 is baseline)
    return (1.0 - trace) * 100


def create_compression_heatmap(df: pd.DataFrame) -> wandb.Table:
    """
    Create layer×head heatmap showing compression potential.

    Returns W&B table with:
    - layer, head, compression_score, category
    - category: "high_potential" | "critical" | "moderate"
    """
    rows = []

    # Get all unique (run, layer, head) combinations
    # For simplicity, take the first run
    runs = df["run_name"].unique()
    if len(runs) == 0:
        return wandb.Table(columns=["layer", "head", "score", "category"], data=[])

    run = runs[0]  # Take first run

    layers = sorted([l for l in df["layer"].unique() if l.startswith("layer")])

    for layer in layers:
        # Get heads for this layer
        heads = sorted(
            [
                h
                for h in df[(df["run_name"] == run) & (df["layer"] == layer)][
                    "head"
                ].unique()
                if h.startswith("head")
            ]
        )

        for head in heads:
            # Get metrics
            cond = df[
                (df["run_name"] == run)
                & (df["layer"] == layer)
                & (df["head"] == head)
                & (df["metric"] == "cond")
            ]["value"].values

            trace = df[
                (df["run_name"] == run)
                & (df["layer"] == layer)
                & (df["head"] == head)
                & (df["metric"] == "trace")
            ]["value"].values

            energy_r8 = df[
                (df["run_name"] == run)
                & (df["layer"] == layer)
                & (df["head"] == head)
                & (df["metric"] == "energy_r8")
            ]["value"].values

            if len(cond) > 0 and len(energy_r8) > 0:
                cond_val = float(cond[0])
                e8_val = float(energy_r8[0])
                score = compute_compression_score(cond_val, e8_val)

                trace_val = float(trace[0]) if len(trace) > 0 else float("nan")

                # Categorize
                if not math.isnan(trace_val) and trace_val > 0.95:
                    category = "critical"  # High trace = critical head
                elif not math.isnan(score) and score > 70:
                    category = "high_potential"  # High compression potential
                else:
                    category = "moderate"

                rows.append([layer, head, score, category])

    return wandb.Table(
        columns=["layer", "head", "compression_score", "category"], data=rows
    )


def create_scatter_data(df: pd.DataFrame) -> wandb.Table:
    """
    Create scatter plot data: condition number vs trace.

    Returns W&B table with columns: cond, trace, layer, head, eigmax
    """
    rows = []

    runs = df["run_name"].unique()
    if len(runs) == 0:
        return wandb.Table(
            columns=["cond", "trace", "layer", "head", "eigmax"], data=[]
        )

    run = runs[0]

    # Get all per-head metrics
    for _, row in df[
        (df["run_name"] == run)
        & (df["metric"] == "cond")
        & (df["head"].str.startswith("head"))
    ].iterrows():
        layer = row["layer"]
        head = row["head"]
        cond_val = row["value"]

        # Get corresponding trace and eigmax
        trace_val = df[
            (df["run_name"] == run)
            & (df["layer"] == layer)
            & (df["head"] == head)
            & (df["metric"] == "trace")
        ]["value"].values

        eigmax_val = df[
            (df["run_name"] == run)
            & (df["layer"] == layer)
            & (df["head"] == head)
            & (df["metric"] == "eigmax")
        ]["value"].values

        if len(trace_val) > 0 and len(eigmax_val) > 0:
            rows.append(
                [cond_val, float(trace_val[0]), layer, head, float(eigmax_val[0])]
            )

    return wandb.Table(
        columns=["condition_number", "trace", "layer", "head", "eigmax"], data=rows
    )


def log_fim_to_wandb(
    entity: str,
    project: str,
    run_id: str,
    df: pd.DataFrame,
    global_stats: Dict[str, Dict[str, float]],
    low_rank_stats: Dict[str, Dict[str, float]],
    summary_text: str,
):
    """
    Log FIM analysis to W&B run.

    Args:
        entity: W&B entity
        project: W&B project
        run_id: W&B run ID to log to
        df: FIM metrics DataFrame (tall-table format)
        global_stats: Global FIM statistics
        low_rank_stats: Low-rank structure stats
        summary_text: Human-readable summary text
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get run name
    run_name = run.display_name or run.name

    if run_name not in global_stats:
        print(f"Warning: Run {run_name} not found in FIM stats")
        return

    g = global_stats[run_name]
    l = low_rank_stats.get(run_name, {})

    # Compute derived metrics
    cond = g.get("cond_mean", float("nan"))
    trace = g.get("trace_mean", float("nan"))
    eigmax = g.get("eigmax_mean", float("nan"))
    energy_r8 = l.get("energy_r8_mean", float("nan"))
    energy_r16 = l.get("energy_r16_mean", float("nan"))
    eff_rank = l.get("effective_rank_90pct", float("nan"))

    compression_score = compute_compression_score(cond, energy_r8)
    efficiency_score = compute_efficiency_score(trace)

    # Create visualizations
    heatmap = create_compression_heatmap(df)
    scatter = create_scatter_data(df)

    # Log everything using W&B SDK
    with wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
    ) as wb_run:
        # Scalar metrics
        wb_run.summary["fim/trace_global_mean"] = trace
        wb_run.summary["fim/eigmax_global_mean"] = eigmax
        wb_run.summary["fim/cond_global_mean"] = cond
        wb_run.summary["fim/energy_r8_mean"] = energy_r8
        wb_run.summary["fim/energy_r16_mean"] = energy_r16
        wb_run.summary["fim/effective_rank"] = eff_rank
        wb_run.summary["fim/compression_potential"] = compression_score
        wb_run.summary["fim/efficiency_score"] = efficiency_score

        # Tables
        wb_run.log({"fim/compression_heatmap": heatmap})
        wb_run.log({"fim/scatter_cond_vs_trace": scatter})

        # HTML summary
        html = wandb.Html(f"<pre>{summary_text}</pre>")
        wb_run.log({"fim/interpretation": html})

        print(f"\n✓ Logged FIM metrics to W&B run: {run_name}")
        print(f"  Compression potential: {compression_score:.1f}/100")
        print(f"  Efficiency score: {efficiency_score:.1f}/100")
        print(f"  Effective rank: {eff_rank}")


if __name__ == "__main__":
    print("This is a helper module. Import it in your scripts.")
    print("Example:")
    print()
    print("  from scripts.fim_wandb_viz import log_fim_to_wandb")
    print("  log_fim_to_wandb(entity, project, run_id, df, stats, lr_stats, summary)")
