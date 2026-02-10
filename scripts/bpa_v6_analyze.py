#!/usr/bin/env python
"""
BPA v6 analysis: aggregate results, fit scaling law, generate report + plots.
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str):
    """Load raw results from JSON."""
    path = os.path.join(results_dir, "raw_results.json")
    with open(path) as f:
        return json.load(f)


def load_overhead(results_dir: str):
    """Load overhead benchmark results."""
    path = os.path.join(results_dir, "overhead_bench.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate(results):
    """Aggregate across seeds for each (variant, seq_len)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["variant"], r["seq_len"])
        groups[key].append(r)

    agg = []
    for (variant, seq_len), runs in sorted(groups.items()):
        ppls = [r["ppl_mean"] for r in runs]
        agg.append(
            {
                "variant": variant,
                "seq_len": seq_len,
                "ppl_mean": float(np.mean(ppls)),
                "ppl_std": float(np.std(ppls)),
                "n_seeds": len(runs),
                "enabled_rate": runs[0].get("enabled_rate", 0.0),
                "far_budget": runs[0].get("far_budget", 0),
                "flops_relative": runs[0].get("flops_relative", 1.0),
                "wall_ms_per_token": float(
                    np.mean([r["wall_ms_per_token"] for r in runs])
                ),
                "gate_ms_per_token": float(
                    np.mean([r.get("gate_ms_per_token", 0.0) for r in runs])
                ),
                "fwd_ms_per_token": float(
                    np.mean([r.get("fwd_ms_per_token", 0.0) for r in runs])
                ),
                "effective_kept_tokens": float(
                    np.mean([r.get("effective_kept_tokens", 0.0) for r in runs])
                ),
                "kv_bytes_read_per_token": float(
                    np.mean([r.get("kv_bytes_read_per_token", 0.0) for r in runs])
                ),
                "peak_kv_bytes": float(
                    np.mean([r.get("peak_kv_bytes", 0.0) for r in runs])
                ),
            }
        )
    return agg


def compute_regressions(agg):
    """Compute PPL regression vs V0_dense for each seq_len."""
    baselines = {}
    for a in agg:
        if a["variant"] == "V0_dense":
            baselines[a["seq_len"]] = a["ppl_mean"]

    for a in agg:
        base = baselines.get(a["seq_len"], a["ppl_mean"])
        a["ppl_regression_pct"] = (a["ppl_mean"] - base) / base * 100
    return agg


def fit_scaling_law(agg):
    """Fit KV_kept(L) = c * L^beta_eff for each variant.

    Returns dict mapping variant -> {beta_eff, c, r_squared, data_points}.
    """
    # Group by variant across seq_lens
    variant_data = defaultdict(list)
    for a in agg:
        variant_data[a["variant"]].append((a["seq_len"], a["effective_kept_tokens"]))

    fits = {}
    for variant, data in variant_data.items():
        data.sort()
        if len(data) < 2:
            continue

        log_L = np.array([math.log(d[0]) for d in data])
        log_KV = np.array([math.log(max(d[1], 1.0)) for d in data])

        # Linear regression in log-log space: log(KV) = log(c) + beta * log(L)
        if len(log_L) >= 2:
            A = np.vstack([log_L, np.ones(len(log_L))]).T
            result = np.linalg.lstsq(A, log_KV, rcond=None)
            beta_eff = float(result[0][0])
            log_c = float(result[0][1])
            c = math.exp(log_c)

            # R-squared
            y_pred = beta_eff * log_L + log_c
            ss_res = np.sum((log_KV - y_pred) ** 2)
            ss_tot = np.sum((log_KV - np.mean(log_KV)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fits[variant] = {
                "beta_eff": beta_eff,
                "c": c,
                "r_squared": r_squared,
                "data_points": [(d[0], d[1]) for d in data],
            }

    return fits


def generate_report(agg, overhead, scaling_fits, output_dir: str):
    """Generate the final markdown report."""
    report = []
    report.append("# BPA v6: Scale + Polish Results\n")

    report.append(
        "> Boundary-Pressure Attention is a conditional rate controller "
        "for the model's history channel, attempting to match memory usage "
        "to the mutual information structure of language rather than to "
        "sequence length.\n"
    )

    report.append("## Model")
    report.append("- GPT2_RGSA (124M params), FineWebEdu, 615 iters")
    report.append("- Config: n_layer=12, n_head=12, n_embd=768")
    report.append("- local_window=256, chunk_size=64")
    report.append("- Gate: v4 wbce_10x at 70% enabled_rate")
    report.append("- Surgical heads: 8 heads from layers 5-8")
    report.append("- Evaluation: FineWebEdu val.bin, bf16 KV accounting\n")

    # Overhead section
    if overhead:
        report.append("## Gate Overhead Reduction\n")
        report.append("| Config | PPL | Gate ms/tok | Fwd ms/tok | Total ms/tok |")
        report.append("|--------|-----|------------|------------|-------------|")
        baseline_total = None
        for o in overhead:
            if o["name"] == "baseline":
                baseline_total = o["wall_ms_per_token"]
            speedup = ""
            if baseline_total and o["wall_ms_per_token"] > 0:
                s = baseline_total / o["wall_ms_per_token"]
                speedup = f" ({s:.1f}x)" if s > 1.01 else ""
            report.append(
                f"| {o['name']} "
                f"| {o['ppl_mean']:.1f} "
                f"| {o['gate_ms_per_token']:.4f} "
                f"| {o['fwd_ms_per_token']:.4f} "
                f"| {o['wall_ms_per_token']:.4f}{speedup} |"
            )
        report.append("")

        if baseline_total:
            best = min(overhead, key=lambda x: x["wall_ms_per_token"])
            if best["wall_ms_per_token"] > 0:
                ratio = baseline_total / best["wall_ms_per_token"]
                report.append(
                    f"Best config: **{best['name']}** at "
                    f"{ratio:.1f}x speedup vs baseline, "
                    f"PPL={best['ppl_mean']:.1f}\n"
                )

    # Results by seq_len
    seq_lens = sorted(set(a["seq_len"] for a in agg))
    for seq_len in seq_lens:
        report.append(f"## Results at L={seq_len}\n")
        report.append(
            "| Variant | PPL | PPL vs Dense | Kept Tokens "
            "| KV Read B/tok | FLOPs% | ms/tok |"
        )
        report.append(
            "|---------|-----|-------------|-------------|"
            "--------------|--------|--------|"
        )

        rows = [a for a in agg if a["seq_len"] == seq_len]
        order = {"V0_dense": 0, "V1_local_only": 1}
        rows.sort(key=lambda x: (order.get(x["variant"], 2), x["ppl_mean"]))

        for a in rows:
            sign = "+" if a["ppl_regression_pct"] >= 0 else ""
            report.append(
                f"| {a['variant']} "
                f"| {a['ppl_mean']:.1f}+/-{a['ppl_std']:.1f} "
                f"| {sign}{a['ppl_regression_pct']:.1f}% "
                f"| {a['effective_kept_tokens']:.0f} "
                f"| {a['kv_bytes_read_per_token']:.0f} "
                f"| {a['flops_relative']*100:.1f}% "
                f"| {a['wall_ms_per_token']:.3f} |"
            )
        report.append("")

    # Scaling law
    if scaling_fits:
        report.append("## KV Cache Scaling Law\n")
        report.append("Fit: `KV_kept(L) = c * L^beta_eff` (log-log regression)\n")
        report.append("| Variant | beta_eff | c | R^2 |")
        report.append("|---------|---------|---|-----|")

        for variant, fit in sorted(scaling_fits.items()):
            report.append(
                f"| {variant} "
                f"| {fit['beta_eff']:.3f} "
                f"| {fit['c']:.1f} "
                f"| {fit['r_squared']:.3f} |"
            )
        report.append("")

        dense_beta = scaling_fits.get("V0_dense", {}).get("beta_eff")
        ra_fits = {k: v for k, v in scaling_fits.items() if "ra_value" in k}
        if dense_beta and ra_fits:
            best_ra = min(ra_fits.items(), key=lambda x: x[1]["beta_eff"])
            report.append(
                f"Dense scales as L^{dense_beta:.2f} (expected ~1.0). "
                f"Best RA variant ({best_ra[0]}) scales as "
                f"L^{best_ra[1]['beta_eff']:.2f}.\n"
            )

    # Verdict
    report.append("## Verdict\n")

    # Find best RA_value config
    l1024 = [a for a in agg if a["seq_len"] == 1024]
    ra_variants = [a for a in l1024 if "ra_value" in a["variant"]]
    recency_variants = [a for a in l1024 if "recency" in a["variant"]]

    if ra_variants and recency_variants:
        best_ra = min(ra_variants, key=lambda x: x["ppl_mean"])
        best_rec = min(recency_variants, key=lambda x: x["ppl_mean"])
        delta = best_rec["ppl_mean"] - best_ra["ppl_mean"]

        report.append(
            f"At L=1024: best RA_value ({best_ra['variant']}) "
            f"PPL={best_ra['ppl_mean']:.1f} vs "
            f"best recency ({best_rec['variant']}) "
            f"PPL={best_rec['ppl_mean']:.1f} "
            f"(delta={delta:.1f})\n"
        )

    # Check L=2048
    l2048 = [a for a in agg if a["seq_len"] == 2048]
    if l2048:
        ra_2048 = [a for a in l2048 if "ra_value" in a["variant"]]
        rec_2048 = [a for a in l2048 if "recency" in a["variant"]]
        if ra_2048 and rec_2048:
            best_ra = min(ra_2048, key=lambda x: x["ppl_mean"])
            best_rec = min(rec_2048, key=lambda x: x["ppl_mean"])
            delta = best_rec["ppl_mean"] - best_ra["ppl_mean"]
            report.append(
                f"At L=2048: RA_value ({best_ra['variant']}) "
                f"PPL={best_ra['ppl_mean']:.1f} vs "
                f"recency ({best_rec['variant']}) "
                f"PPL={best_rec['ppl_mean']:.1f} "
                f"(delta={delta:.1f})\n"
            )

    report.append("### Recommended BPA+RA Config\n")
    # Find config with best PPL at L=1024 among gated variants
    gated_1024 = [a for a in l1024 if a["variant"] not in ("V0_dense", "V1_local_only")]
    if gated_1024:
        best = min(gated_1024, key=lambda x: x["ppl_mean"])
        report.append(
            f"- Variant: {best['variant']}\n"
            f"- PPL: {best['ppl_mean']:.1f} "
            f"(+{best['ppl_regression_pct']:.1f}% vs dense)\n"
            f"- FLOPs: {best['flops_relative']*100:.1f}% of dense\n"
            f"- KV read: {best['kv_bytes_read_per_token']:.0f} B/token"
        )

    report_text = "\n".join(report) + "\n"
    report_path = os.path.join(output_dir, "bpa_v6_final_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report: {report_path}")
    return report_text


def generate_plots(agg, overhead, scaling_fits, output_dir: str):
    """Generate visualization plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: PPL regression vs dense by seq_len
    fig, ax = plt.subplots(figsize=(10, 6))
    seq_lens = sorted(set(a["seq_len"] for a in agg))
    # Only show strategy variants (not dense/local_only)
    strategy_names = sorted(
        set(
            a["variant"]
            for a in agg
            if a["variant"] not in ("V0_dense", "V1_local_only")
        )
    )

    x_pos = np.arange(len(seq_lens))
    width = 0.8 / max(len(strategy_names), 1)

    for i, name in enumerate(strategy_names):
        regs = []
        for sl in seq_lens:
            match = [a for a in agg if a["variant"] == name and a["seq_len"] == sl]
            regs.append(match[0]["ppl_regression_pct"] if match else 0.0)

        color = (
            "#4CAF50"
            if "ra_value" in name
            else (
                "#FF9800"
                if "recency" in name
                else "#8BC34A" if "blend" in name else "#9E9E9E"
            )
        )
        offset = (i - len(strategy_names) / 2 + 0.5) * width
        ax.bar(x_pos + offset, regs, width * 0.9, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("PPL Regression vs Dense (%)")
    ax.set_title("BPA v6: PPL Regression by Strategy and Length")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(sl) for sl in seq_lens])
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ppl_regression_vs_dense.png"), dpi=150)
    plt.close()

    # Plot 2: Gate overhead comparison
    if overhead:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [o["name"] for o in overhead]
        gate_ms = [o["gate_ms_per_token"] for o in overhead]
        fwd_ms = [o["fwd_ms_per_token"] for o in overhead]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, gate_ms, 0.35, label="Gate", color="#F44336", alpha=0.8)
        ax.barh(y_pos + 0.35, fwd_ms, 0.35, label="Forward", color="#2196F3", alpha=0.8)
        ax.set_yticks(y_pos + 0.175)
        ax.set_yticklabels(names)
        ax.set_xlabel("ms/token")
        ax.set_title("BPA v6: Gate Overhead Breakdown")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gate_overhead_ms_per_token.png"), dpi=150)
        plt.close()

    # Plot 3: KV kept vs L (log-log) with scaling fits
    if scaling_fits and len(seq_lens) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        for variant, fit in sorted(scaling_fits.items()):
            data = fit["data_points"]
            Ls = [d[0] for d in data]
            KVs = [d[1] for d in data]

            color = (
                "#2196F3"
                if "dense" in variant
                else (
                    "#F44336"
                    if "local" in variant
                    else (
                        "#4CAF50"
                        if "ra_value" in variant
                        else "#FF9800" if "recency" in variant else "#9E9E9E"
                    )
                )
            )

            ax.scatter(Ls, KVs, color=color, s=50, zorder=3)

            # Fitted line
            L_fit = np.linspace(min(Ls) * 0.9, max(Ls) * 1.1, 100)
            KV_fit = fit["c"] * L_fit ** fit["beta_eff"]
            label = f"{variant} (β={fit['beta_eff']:.2f})"
            ax.plot(L_fit, KV_fit, color=color, linestyle="--", alpha=0.7, label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sequence Length (L)")
        ax.set_ylabel("Effective KV Kept (tokens)")
        ax.set_title("KV Cache Scaling: KV_kept(L) = c * L^β")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "kv_kept_vs_L_loglog.png"), dpi=150)
        plt.close()

    # Plot 4: KV traffic vs L
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(set(a["variant"] for a in agg)):
        data = [
            (a["seq_len"], a["kv_bytes_read_per_token"])
            for a in agg
            if a["variant"] == variant
        ]
        data.sort()
        if len(data) < 2:
            continue
        Ls = [d[0] for d in data]
        traffic = [d[1] for d in data]

        color = (
            "#2196F3"
            if "dense" in variant
            else (
                "#F44336"
                if "local" in variant
                else (
                    "#4CAF50"
                    if "ra_value" in variant
                    else "#FF9800" if "recency" in variant else "#9E9E9E"
                )
            )
        )
        ax.plot(Ls, traffic, "o-", color=color, label=variant, alpha=0.8)

    ax.set_xlabel("Sequence Length (L)")
    ax.set_ylabel("KV Read Bytes/Token")
    ax.set_title("BPA v6: KV Read Traffic vs Sequence Length")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kv_traffic_vs_L.png"), dpi=150)
    plt.close()

    # Plot 5: PPL by strategy at each L
    n_lens = len(seq_lens)
    if n_lens > 0:
        fig, axes = plt.subplots(1, min(n_lens, 3), figsize=(5 * min(n_lens, 3), 5))
        if n_lens == 1:
            axes = [axes]
        for ax_idx, seq_len in enumerate(seq_lens[:3]):
            ax = axes[ax_idx]
            rows = [a for a in agg if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["ppl_mean"])
            names = [a["variant"] for a in rows]
            ppls = [a["ppl_mean"] for a in rows]
            stds = [a["ppl_std"] for a in rows]
            ax.barh(names, ppls, xerr=stds, capsize=3, alpha=0.8, color="#607D8B")
            ax.set_xlabel("Perplexity")
            ax.set_title(f"L={seq_len}")
        plt.suptitle("BPA v6: PPL by Variant")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ppl_by_length.png"), dpi=150)
        plt.close()

    print(f"Plots: {plots_dir}/")


def main():
    results_dir = "bpa_v6_results"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    results = load_results(results_dir)
    print(f"Loaded {len(results)} results")

    overhead = load_overhead(results_dir)
    if overhead:
        print(f"Loaded {len(overhead)} overhead benchmark entries")

    agg = aggregate(results)
    agg = compute_regressions(agg)

    print("\nAggregated results:")
    for a in agg:
        sign = "+" if a["ppl_regression_pct"] >= 0 else ""
        print(
            f"  {a['variant']:30s} L={a['seq_len']:4d} "
            f"PPL={a['ppl_mean']:6.1f}+/-{a['ppl_std']:4.1f} "
            f"({sign}{a['ppl_regression_pct']:.1f}%) "
            f"kept={a['effective_kept_tokens']:.0f}"
        )

    scaling_fits = fit_scaling_law(agg)
    if scaling_fits:
        print("\nScaling law fits:")
        for variant, fit in sorted(scaling_fits.items()):
            print(
                f"  {variant:30s} beta={fit['beta_eff']:.3f} "
                f"c={fit['c']:.1f} R²={fit['r_squared']:.3f}"
            )

    # Save scaling fits
    fits_path = os.path.join(results_dir, "scaling_fit.json")
    with open(fits_path, "w") as f:
        json.dump(scaling_fits, f, indent=2)
    print(f"\nScaling fits: {fits_path}")

    generate_report(agg, overhead, scaling_fits, results_dir)
    generate_plots(agg, overhead, scaling_fits, results_dir)


if __name__ == "__main__":
    main()
