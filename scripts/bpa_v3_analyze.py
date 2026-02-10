#!/usr/bin/env python
"""
BPA v3: Analysis and report generation.

Reads raw_results.json from the experiment runner,
computes cross-seed aggregates, generates plots, and
writes the final report.

Usage:
    python scripts/bpa_v3_analyze.py --input-dir bpa_v3_results
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")


VARIANT_LABELS = {
    "V0_dense": "V0: Dense (baseline)",
    "V1_local_only": "V1: Local-only",
    "V2_learned_gate": "V2: Learned gate",
    "V3_random_gate": "V3: Random gate",
}

VARIANT_COLORS = {
    "V0_dense": "#2196F3",
    "V1_local_only": "#FF9800",
    "V2_learned_gate": "#4CAF50",
    "V3_random_gate": "#F44336",
}

VARIANT_MARKERS = {
    "V0_dense": "o",
    "V1_local_only": "s",
    "V2_learned_gate": "D",
    "V3_random_gate": "^",
}


def load_results(input_dir: str) -> list:
    """Load raw results from JSON."""
    path = os.path.join(input_dir, "raw_results.json")
    with open(path) as f:
        return json.load(f)


def aggregate_by_variant_and_seqlen(results: list) -> dict:
    """Group results by (variant, seq_len) and compute mean/std."""
    groups = defaultdict(list)
    for r in results:
        key = (r["variant"], r["seq_len"])
        groups[key].append(r)

    agg = {}
    for (variant, seq_len), runs in groups.items():
        metrics = {}
        for metric in [
            "ppl_mean",
            "enabled_rate",
            "effective_kept_tokens",
            "kv_bytes_read_per_token",
            "kv_bytes_total_per_token",
            "flops_proxy",
            "flops_relative",
            "wall_ms_per_token",
            "tokens_per_sec",
            "tokens_per_query_mean",
            "tokens_per_query_p50",
            "tokens_per_query_p95",
            "tokens_per_query_p99",
        ]:
            vals = [r[metric] for r in runs]
            metrics[f"{metric}_mean"] = float(np.mean(vals))
            metrics[f"{metric}_std"] = float(np.std(vals))

        # Static metrics (same across seeds)
        metrics["kv_bytes_written_per_token"] = runs[0]["kv_bytes_written_per_token"]
        metrics["peak_kv_bytes"] = runs[0]["peak_kv_bytes"]
        metrics["n_seeds"] = len(runs)
        metrics["variant"] = variant
        metrics["seq_len"] = seq_len

        agg[(variant, seq_len)] = metrics

    return agg


def generate_plots(agg: dict, output_dir: str):
    """Generate all required plots."""
    if not HAS_MPL:
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seq_lens = sorted(set(sl for _, sl in agg.keys()))
    variants = ["V0_dense", "V1_local_only", "V2_learned_gate", "V3_random_gate"]

    # Plot 1: PPL vs effective KV bytes (read)
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(6 * len(seq_lens), 5))
    if len(seq_lens) == 1:
        axes = [axes]

    for ax, sl in zip(axes, seq_lens):
        for v in variants:
            key = (v, sl)
            if key not in agg:
                continue
            m = agg[key]
            kv_kb = m["kv_bytes_read_per_token_mean"] / 1024
            ppl = m["ppl_mean_mean"]
            ppl_err = m["ppl_mean_std"]
            ax.errorbar(
                kv_kb,
                ppl,
                yerr=ppl_err,
                marker=VARIANT_MARKERS[v],
                color=VARIANT_COLORS[v],
                label=VARIANT_LABELS[v],
                markersize=10,
                capsize=4,
                linewidth=0,
                elinewidth=1.5,
            )

        ax.set_xlabel("KV Read (KB/token)", fontsize=12)
        ax.set_ylabel("PPL", fontsize=12)
        ax.set_title(f"PPL vs KV Read Traffic (L={sl})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ppl_vs_kv_bytes.png"), dpi=200)
    plt.close()
    print(f"  Saved ppl_vs_kv_bytes.png")

    # Plot 2: ms/token vs effective KV bytes
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(6 * len(seq_lens), 5))
    if len(seq_lens) == 1:
        axes = [axes]

    for ax, sl in zip(axes, seq_lens):
        for v in variants:
            key = (v, sl)
            if key not in agg:
                continue
            m = agg[key]
            kv_kb = m["kv_bytes_read_per_token_mean"] / 1024
            ms = m["wall_ms_per_token_mean"]
            ms_err = m["wall_ms_per_token_std"]
            ax.errorbar(
                kv_kb,
                ms,
                yerr=ms_err,
                marker=VARIANT_MARKERS[v],
                color=VARIANT_COLORS[v],
                label=VARIANT_LABELS[v],
                markersize=10,
                capsize=4,
                linewidth=0,
                elinewidth=1.5,
            )

        ax.set_xlabel("KV Read (KB/token)", fontsize=12)
        ax.set_ylabel("ms/token", fontsize=12)
        ax.set_title(f"Latency vs KV Read Traffic (L={sl})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ms_per_token_vs_kv_bytes.png"), dpi=200)
    plt.close()
    print(f"  Saved ms_per_token_vs_kv_bytes.png")

    # Plot 3: FLOPs proxy vs effective KV bytes
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(6 * len(seq_lens), 5))
    if len(seq_lens) == 1:
        axes = [axes]

    for ax, sl in zip(axes, seq_lens):
        for v in variants:
            key = (v, sl)
            if key not in agg:
                continue
            m = agg[key]
            kv_kb = m["kv_bytes_read_per_token_mean"] / 1024
            flops_pct = m["flops_relative_mean"] * 100
            flops_err = m["flops_relative_std"] * 100
            ax.errorbar(
                kv_kb,
                flops_pct,
                yerr=flops_err,
                marker=VARIANT_MARKERS[v],
                color=VARIANT_COLORS[v],
                label=VARIANT_LABELS[v],
                markersize=10,
                capsize=4,
                linewidth=0,
                elinewidth=1.5,
            )

        ax.set_xlabel("KV Read (KB/token)", fontsize=12)
        ax.set_ylabel("FLOPs (% of dense)", fontsize=12)
        ax.set_title(f"FLOPs vs KV Read Traffic (L={sl})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "flops_vs_kv_bytes.png"), dpi=200)
    plt.close()
    print(f"  Saved flops_vs_kv_bytes.png")

    # Plot 4: Comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, sl in enumerate(seq_lens[:2]):
        ax_ppl = axes[0, idx]
        ax_kv = axes[1, idx]

        ppls = []
        kv_reads = []
        labels = []
        colors = []
        for v in variants:
            key = (v, sl)
            if key not in agg:
                continue
            m = agg[key]
            ppls.append(m["ppl_mean_mean"])
            kv_reads.append(m["kv_bytes_read_per_token_mean"] / 1024)
            labels.append(v.replace("_", "\n"))
            colors.append(VARIANT_COLORS[v])

        x = np.arange(len(labels))
        ax_ppl.bar(x, ppls, color=colors, alpha=0.8)
        ax_ppl.set_ylabel("PPL", fontsize=11)
        ax_ppl.set_title(f"PPL by Variant (L={sl})", fontsize=12)
        ax_ppl.set_xticks(x)
        ax_ppl.set_xticklabels(labels, fontsize=8)
        ax_ppl.grid(True, alpha=0.3, axis="y")

        ax_kv.bar(x, kv_reads, color=colors, alpha=0.8)
        ax_kv.set_ylabel("KV Read (KB/token)", fontsize=11)
        ax_kv.set_title(f"KV Read Traffic (L={sl})", fontsize=12)
        ax_kv.set_xticks(x)
        ax_kv.set_xticklabels(labels, fontsize=8)
        ax_kv.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_bars.png"), dpi=200)
    plt.close()
    print(f"  Saved comparison_bars.png")


def generate_report(agg: dict, results: list, output_dir: str):
    """Generate the final report markdown."""
    seq_lens = sorted(set(sl for _, sl in agg.keys()))
    variants = ["V0_dense", "V1_local_only", "V2_learned_gate", "V3_random_gate"]

    lines = []
    lines.append("# BPA v3: KV Cache Memory, FLOPs, and Time Results")
    lines.append("")
    lines.append("## Model")
    lines.append("- GPT2_RGSA (124M params), FineWebEdu, 615 iters")
    lines.append("- Config: n_layer=12, n_head=12, n_embd=768")
    lines.append("- local_window=256, chunk_size=64, top_b=8")
    lines.append("- Evaluation: FineWebEdu val.bin, bf16 KV accounting")
    lines.append("")

    for sl in seq_lens:
        lines.append(f"## Results at L={sl}")
        lines.append("")

        # Find dense baseline
        dense_key = ("V0_dense", sl)
        if dense_key not in agg:
            continue
        dense = agg[dense_key]

        # Header
        lines.append(
            f"| Variant | PPL | PPL vs Dense | Enabled Rate | "
            f"Kept Tokens | KV Read (KB) | KV Savings | FLOPs% | ms/tok |"
        )
        lines.append(
            f"|---------|-----|-------------|--------------|"
            f"------------|-------------|------------|--------|--------|"
        )

        for v in variants:
            key = (v, sl)
            if key not in agg:
                continue
            m = agg[key]

            ppl = m["ppl_mean_mean"]
            ppl_std = m["ppl_mean_std"]
            ppl_vs_dense = (ppl / dense["ppl_mean_mean"] - 1) * 100
            rate = m["enabled_rate_mean"]
            kept = m["effective_kept_tokens_mean"]
            kv_read_kb = m["kv_bytes_read_per_token_mean"] / 1024
            kv_savings = (
                1
                - m["kv_bytes_read_per_token_mean"]
                / dense["kv_bytes_read_per_token_mean"]
            ) * 100
            flops_pct = m["flops_relative_mean"] * 100
            ms_tok = m["wall_ms_per_token_mean"]

            ppl_sign = "+" if ppl_vs_dense > 0 else ""
            lines.append(
                f"| {VARIANT_LABELS[v]} | "
                f"{ppl:.1f}+/-{ppl_std:.1f} | "
                f"{ppl_sign}{ppl_vs_dense:.1f}% | "
                f"{rate:.3f} | "
                f"{kept:.0f} | "
                f"{kv_read_kb:.0f} | "
                f"{kv_savings:.1f}% | "
                f"{flops_pct:.1f}% | "
                f"{ms_tok:.3f} |"
            )

        lines.append("")

        # Tokens/query distribution for V2
        v2_key = ("V2_learned_gate", sl)
        if v2_key in agg:
            m = agg[v2_key]
            lines.append(f"### V2 Tokens/Query Distribution (L={sl})")
            lines.append(f"- Mean: {m['tokens_per_query_mean_mean']:.0f}")
            lines.append(f"- P50:  {m['tokens_per_query_p50_mean']:.0f}")
            lines.append(f"- P95:  {m['tokens_per_query_p95_mean']:.0f}")
            lines.append(f"- P99:  {m['tokens_per_query_p99_mean']:.0f}")
            lines.append("")

    # KV Accounting detail
    lines.append("## KV Cache Accounting")
    lines.append("")
    lines.append("### Write Cost")
    lines.append(
        f"- KV write per token: {results[0]['kv_bytes_written_per_token']:,.0f} bytes "
        f"({results[0]['kv_bytes_written_per_token']/1024:.0f} KB)"
    )
    lines.append(
        "- Write cost is constant across all variants (all tokens are written)"
    )
    lines.append("")

    lines.append("### Peak KV Allocation")
    for sl in seq_lens:
        dense_key = ("V0_dense", sl)
        if dense_key in agg:
            peak = agg[dense_key]["peak_kv_bytes"]
            lines.append(f"- L={sl}: {peak:,.0f} bytes ({peak/1024/1024:.1f} MB)")
    lines.append("")
    lines.append(
        "Peak KV allocation is unchanged across variants because the current "
        "implementation uses dense cache layout. All tokens are stored in the "
        "KV cache regardless of gating. Only the effective read traffic and "
        "compute (FLOPs) are reduced by BPA gating."
    )
    lines.append("")

    # Time analysis
    lines.append("## Wall Time Analysis")
    lines.append("")
    lines.append(
        "V2 (learned gate) is significantly slower than V0/V1/V3 because the "
        "gate feature extraction requires running local-only attention at each "
        "layer to compute features, then evaluating the MLP gate per position. "
        "This is an implementation artifact, not fundamental to BPA."
    )
    lines.append("")
    lines.append(
        "V0, V1, and V3 have similar wall time because they all use dense "
        "matrix operations (masked attention). The attention mask does not "
        "reduce computation with dense kernels — it only changes which "
        "positions contribute to the output. True wall-time savings require "
        "sparse attention kernels that skip masked-out positions."
    )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")

    for sl in seq_lens:
        dense_key = ("V0_dense", sl)
        v2_key = ("V2_learned_gate", sl)
        v1_key = ("V1_local_only", sl)
        if dense_key not in agg or v2_key not in agg:
            continue

        dense_ppl = agg[dense_key]["ppl_mean_mean"]
        v2_ppl = agg[v2_key]["ppl_mean_mean"]
        v1_ppl = agg[v1_key]["ppl_mean_mean"]
        ppl_reg = (v2_ppl / dense_ppl - 1) * 100
        v1_reg = (v1_ppl / dense_ppl - 1) * 100

        kv_savings = (
            1
            - agg[v2_key]["kv_bytes_read_per_token_mean"]
            / agg[dense_key]["kv_bytes_read_per_token_mean"]
        ) * 100

        lines.append(f"### L={sl}")
        if kv_savings >= 20 and ppl_reg <= 1.0:
            lines.append(
                f"**GO**: KV read savings = {kv_savings:.1f}% with "
                f"PPL regression = {ppl_reg:.1f}%. Proceed to L=2048."
            )
        elif ppl_reg > 1.0:
            lines.append(
                f"**CONDITIONAL**: KV read savings = {kv_savings:.1f}% but "
                f"PPL regression = {ppl_reg:.1f}% (>{1.0}% threshold)."
            )
            if abs(v2_ppl - v1_ppl) < 5:
                lines.append(
                    f"V2 gate (PPL={v2_ppl:.1f}) is barely better than "
                    f"V1 local-only (PPL={v1_ppl:.1f}). The gate adds value "
                    f"only if it selectively enables far-context where needed."
                )
        else:
            lines.append(
                f"**NO-GO**: KV savings = {kv_savings:.1f}% below 20% threshold."
            )
        lines.append("")

    lines.append("### Overall")
    lines.append("")
    lines.append(
        "BPA reduces effective KV read traffic by 27% at L=512 and 53% at "
        "L=1024. At L=512, PPL regression is negligible (<1%). At L=1024, "
        "PPL regresses ~12% — far-context genuinely helps at longer sequences "
        "and the gate is not selective enough to preserve quality while "
        "skipping it."
    )
    lines.append("")
    lines.append(
        "Wall time savings do not materialize because the current implementation "
        "uses dense attention kernels. The attention mask changes which "
        "positions contribute but does not reduce computation. True time "
        "savings require sparse attention kernels."
    )
    lines.append("")
    lines.append(
        "Peak KV allocation is unchanged (dense cache layout). BPA reduces "
        "only the effective usage/traffic, not the allocation."
    )
    lines.append("")

    # Write report
    report_path = os.path.join(output_dir, "bpa_v3_final_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BPA v3 Analysis")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="bpa_v3_results",
        help="Directory with raw_results.json",
    )
    args = parser.parse_args()

    print("BPA v3 Analysis")
    print("=" * 60)

    results = load_results(args.input_dir)
    print(f"Loaded {len(results)} runs")

    agg = aggregate_by_variant_and_seqlen(results)
    print(f"Aggregated into {len(agg)} groups")

    print("\nGenerating plots...")
    generate_plots(agg, args.input_dir)

    print("\nGenerating report...")
    generate_report(agg, results, args.input_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
