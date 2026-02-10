#!/usr/bin/env python
"""
BPA v8 analysis: beta_eff vs (W,C) sweep, stress tests, Pareto frontier.

Generates:
  - beta_eff vs (W,C) table and plot
  - stress test region-wise PPL comparison
  - Pareto curves (PPL vs KV_kept)
  - bpa_v8_final_report.md
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


def aggregate(results):
    """Aggregate across seeds for each (variant, seq_len, W, C, stress)."""
    groups = defaultdict(list)
    for r in results:
        key = (
            r["variant"],
            r["seq_len"],
            r["local_window"],
            r["chunk_size"],
            r.get("stress_mode", "control"),
        )
        groups[key].append(r)

    agg = []
    for (variant, seq_len, W, C, stress), runs in sorted(groups.items()):
        ppls = [r["ppl_mean"] for r in runs]
        agg.append(
            {
                "variant": variant,
                "seq_len": seq_len,
                "local_window": W,
                "chunk_size": C,
                "stress_mode": stress,
                "ppl_mean": float(np.mean(ppls)),
                "ppl_std": float(np.std(ppls)),
                "n_seeds": len(runs),
                "far_budget": runs[0].get("far_budget", 0),
                "effective_kv_kept_tokens": float(
                    np.mean([r.get("effective_kv_kept_tokens", 0) for r in runs])
                ),
                "kv_read_bytes_per_token": float(
                    np.mean([r.get("kv_read_bytes_per_token", 0) for r in runs])
                ),
                "gate_ms_per_token": float(
                    np.mean([r.get("gate_ms_per_token", 0) for r in runs])
                ),
                "forward_ms_per_token": float(
                    np.mean([r.get("forward_ms_per_token", 0) for r in runs])
                ),
                "ppl_vs_dense_pct": float(
                    np.mean([r.get("ppl_vs_dense_pct", 0) for r in runs])
                ),
                "extra": runs[0].get("extra", {}),
            }
        )
    return agg


def fit_beta_eff(agg, variant, W, C, stress="control"):
    """Fit KV_kept(L) = c * L^beta for a specific (variant, W, C)."""
    data = [
        (a["seq_len"], a["effective_kv_kept_tokens"])
        for a in agg
        if a["variant"] == variant
        and a["local_window"] == W
        and a["chunk_size"] == C
        and a.get("stress_mode", "control") == stress
    ]
    data.sort()
    if len(data) < 2:
        return None

    log_L = np.array([math.log(d[0]) for d in data])
    log_KV = np.array([math.log(max(d[1], 1.0)) for d in data])

    A = np.vstack([log_L, np.ones(len(log_L))]).T
    result = np.linalg.lstsq(A, log_KV, rcond=None)
    beta = float(result[0][0])
    c = math.exp(float(result[0][1]))

    y_pred = beta * log_L + float(result[0][1])
    ss_res = np.sum((log_KV - y_pred) ** 2)
    ss_tot = np.sum((log_KV - np.mean(log_KV)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "beta_eff": beta,
        "c": c,
        "r_squared": r2,
        "n_points": len(data),
        "data": [(d[0], d[1]) for d in data],
    }


def compute_all_beta_fits(agg):
    """Compute beta_eff for all (variant, W, C) combinations."""
    combos = set()
    for a in agg:
        combos.add(
            (
                a["variant"],
                a["local_window"],
                a["chunk_size"],
                a.get("stress_mode", "control"),
            )
        )

    fits = {}
    for variant, W, C, stress in sorted(combos):
        fit = fit_beta_eff(agg, variant, W, C, stress)
        if fit:
            fits[(variant, W, C, stress)] = fit
    return fits


def generate_report(agg, beta_fits, output_dir):
    """Generate bpa_v8_final_report.md."""
    report = []
    report.append("# BPA v8: Unmask Beta_eff + Stress Tests\n")
    report.append(
        "> Goal: Determine whether BPA's observed beta_eff ~ 0.25 is "
        "(A) a real data/model property or (B) an artifact of fixed "
        "local window / chunking.\n"
    )

    report.append("## Model")
    report.append("- GPT2_RGSA (124M params), FineWebEdu, 615 iters")
    report.append("- Config: n_layer=12, n_head=12, n_embd=768")
    report.append("- Gate: v4 wbce_10x at 70% enabled_rate")
    report.append("- Surgical heads: 8 heads from layers 5-8\n")

    # Phase 1: Beta_eff vs (W, C) table
    phase1_fits = {k: v for k, v in beta_fits.items() if k[3] == "control"}
    if phase1_fits:
        report.append("## Phase 1: Beta_eff vs (W, C)\n")
        report.append("| Variant | W | C | B | beta_eff | c | R^2 | Points |")
        report.append("|---------|---|---|---|---------|---|-----|--------|")

        for (variant, W, C, _), fit in sorted(phase1_fits.items()):
            # Find far_budget for this combo
            fb = 0
            for a in agg:
                if (
                    a["variant"] == variant
                    and a["local_window"] == W
                    and a["chunk_size"] == C
                ):
                    fb = a["far_budget"]
                    break
            report.append(
                f"| {variant} | {W} | {C} | {fb} "
                f"| {fit['beta_eff']:.3f} | {fit['c']:.1f} "
                f"| {fit['r_squared']:.3f} | {fit['n_points']} |"
            )
        report.append("")

        # Summarize: does beta change with W?
        ra_fits = {k: v for k, v in phase1_fits.items() if "ra_value" in k[0]}
        if ra_fits:
            betas_by_W = defaultdict(list)
            for (_, W, C, _), fit in ra_fits.items():
                betas_by_W[W].append(fit["beta_eff"])

            report.append("### Beta_eff by W (ra_value, averaged over C)\n")
            for W in sorted(betas_by_W.keys()):
                betas = betas_by_W[W]
                report.append(
                    f"- W={W}: beta_eff = "
                    f"{np.mean(betas):.3f} +/- {np.std(betas):.3f} "
                    f"(n={len(betas)})"
                )
            report.append("")

            # Does beta vary with C?
            betas_by_C = defaultdict(list)
            for (_, W, C, _), fit in ra_fits.items():
                betas_by_C[C].append(fit["beta_eff"])

            report.append("### Beta_eff by C (ra_value, averaged over W)\n")
            for C in sorted(betas_by_C.keys()):
                betas = betas_by_C[C]
                report.append(
                    f"- C={C}: beta_eff = "
                    f"{np.mean(betas):.3f} +/- {np.std(betas):.3f} "
                    f"(n={len(betas)})"
                )
            report.append("")

            # Overall assessment
            all_betas = [fit["beta_eff"] for fit in ra_fits.values()]
            beta_range = max(all_betas) - min(all_betas)
            report.append("### Assessment\n")
            if beta_range < 0.1:
                report.append(
                    f"Beta_eff is **stable** across W and C "
                    f"(range={beta_range:.3f}). "
                    f"Mean beta={np.mean(all_betas):.3f}. "
                    f"This suggests beta is a real property of the "
                    f"data/model interaction, not an artifact of "
                    f"implementation parameters.\n"
                )
            else:
                report.append(
                    f"Beta_eff **varies** across W and C "
                    f"(range={beta_range:.3f}). "
                    f"This suggests beta is at least partially "
                    f"driven by the local window floor.\n"
                )

    # Phase 2: Stress test results
    stress_results = [a for a in agg if a.get("stress_mode") != "control"]
    control_results = [a for a in agg if a.get("stress_mode") == "control"]

    # Get phase 2 results (fixed W=256, C=64)
    stress_phase2 = [
        a
        for a in stress_results
        if a.get("local_window") == 256 and a.get("chunk_size") == 64
    ]
    control_phase2 = [
        a
        for a in control_results
        if a.get("local_window") == 256 and a.get("chunk_size") == 64
    ]

    if stress_phase2:
        report.append("## Phase 2: Adversarial Stress Tests\n")

        for stress_mode in sorted(set(a["stress_mode"] for a in stress_phase2)):
            report.append(f"### {stress_mode}\n")
            report.append(
                "| Variant | L | PPL | PPL vs Dense "
                "| Early PPL | Late PPL | KV Kept |"
            )
            report.append(
                "|---------|---|-----|-------------|"
                "-----------|----------|---------|"
            )

            rows = [a for a in stress_phase2 if a["stress_mode"] == stress_mode]
            # Add corresponding control for comparison
            for a in control_phase2:
                if a["variant"] in [r["variant"] for r in rows]:
                    if a["seq_len"] in [r["seq_len"] for r in rows]:
                        pass  # Control already there implicitly

            rows.sort(key=lambda x: (x["seq_len"], x["ppl_mean"]))

            for a in rows:
                extra = a.get("extra", {})
                early = extra.get("ppl_early_mean", 0)
                late = extra.get("ppl_late_mean", 0)
                ppl_vs = a.get("ppl_vs_dense_pct", 0)
                sign = "+" if ppl_vs >= 0 else ""
                report.append(
                    f"| {a['variant']} | {a['seq_len']} "
                    f"| {a['ppl_mean']:.1f} "
                    f"| {sign}{ppl_vs:.1f}% "
                    f"| {early:.1f} "
                    f"| {late:.1f} "
                    f"| {a['effective_kv_kept_tokens']:.0f} |"
                )
            report.append("")

    # Phase 4: Hardware metrics
    report.append("## Phase 4: Hardware Metrics\n")
    report.append(
        "| Variant | L | W | C | Gate ms/tok | Fwd ms/tok "
        "| KV Read B/tok | KV Kept |"
    )
    report.append(
        "|---------|---|---|---|------------|------------|" "--------------|---------|"
    )

    # Show a representative subset
    seen = set()
    for a in sorted(agg, key=lambda x: (x["seq_len"], x["local_window"])):
        if a.get("stress_mode", "control") != "control":
            continue
        key = (a["variant"], a["seq_len"], a["local_window"])
        if key in seen:
            continue
        seen.add(key)
        report.append(
            f"| {a['variant']} | {a['seq_len']} "
            f"| {a['local_window']} | {a['chunk_size']} "
            f"| {a['gate_ms_per_token']:.4f} "
            f"| {a['forward_ms_per_token']:.4f} "
            f"| {a['kv_read_bytes_per_token']:.0f} "
            f"| {a['effective_kv_kept_tokens']:.0f} |"
        )
    report.append("")

    # Pareto frontier
    report.append("## Pareto Frontier (PPL vs KV Kept)\n")
    for seq_len in sorted(set(a["seq_len"] for a in agg)):
        points = [
            a
            for a in agg
            if a["seq_len"] == seq_len and a.get("stress_mode", "control") == "control"
        ]
        if not points:
            continue

        points.sort(key=lambda x: x["effective_kv_kept_tokens"])

        # Find Pareto-optimal points
        frontier = []
        best_ppl = float("inf")
        for p in points:
            if p["ppl_mean"] < best_ppl:
                best_ppl = p["ppl_mean"]
                frontier.append(p)

        report.append(f"### L={seq_len}\n")
        report.append("| Variant | W | C | PPL | KV Kept |")
        report.append("|---------|---|---|-----|---------|")
        for p in frontier:
            report.append(
                f"| {p['variant']} | {p['local_window']} "
                f"| {p['chunk_size']} "
                f"| {p['ppl_mean']:.1f} "
                f"| {p['effective_kv_kept_tokens']:.0f} |"
            )
        report.append("")

    # Conclusions
    report.append("## Conclusions\n")
    report.append("### (1) Is beta_eff an artifact?\n")
    report.append("### (2) Does BPA fail under topic switch?\n")
    report.append("### (3) Does sparse selection translate to real savings?\n")

    report_text = "\n".join(report) + "\n"
    report_path = os.path.join(output_dir, "bpa_v8_final_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report: {report_path}")
    return report_text


def generate_plots(agg, beta_fits, output_dir):
    """Generate v8 plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: PPL vs L for each (W, C) — ra_value only
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    Ws = sorted(set(a["local_window"] for a in agg if a["variant"] != "V0_dense"))

    colors_W = {64: "#F44336", 128: "#FF9800", 256: "#4CAF50", 512: "#2196F3"}
    markers_C = {16: "o", 32: "s", 64: "^", 128: "D"}

    # PPL vs L
    ax = axes[0]
    for a_variant in ["V8_ra_value"]:
        for W in Ws:
            for C in sorted(
                set(
                    a["chunk_size"]
                    for a in agg
                    if a["variant"] == a_variant and a["local_window"] == W
                )
            ):
                data = [
                    (a["seq_len"], a["ppl_mean"])
                    for a in agg
                    if a["variant"] == a_variant
                    and a["local_window"] == W
                    and a["chunk_size"] == C
                    and a.get("stress_mode", "control") == "control"
                ]
                data.sort()
                if len(data) < 2:
                    continue
                Ls = [d[0] for d in data]
                ppls = [d[1] for d in data]
                ax.plot(
                    Ls,
                    ppls,
                    marker=markers_C.get(C, "o"),
                    color=colors_W.get(W, "#9E9E9E"),
                    label=f"W={W},C={C}",
                    alpha=0.8,
                )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("PPL")
    ax.set_title("PPL vs L (ra_value)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # KV kept vs L (log-log)
    ax = axes[1]
    for a_variant in ["V8_ra_value"]:
        for W in Ws:
            for C in sorted(
                set(
                    a["chunk_size"]
                    for a in agg
                    if a["variant"] == a_variant and a["local_window"] == W
                )
            ):
                data = [
                    (a["seq_len"], a["effective_kv_kept_tokens"])
                    for a in agg
                    if a["variant"] == a_variant
                    and a["local_window"] == W
                    and a["chunk_size"] == C
                    and a.get("stress_mode", "control") == "control"
                ]
                data.sort()
                if len(data) < 2:
                    continue
                Ls = [d[0] for d in data]
                kvs = [d[1] for d in data]
                ax.plot(
                    Ls,
                    kvs,
                    marker=markers_C.get(C, "o"),
                    color=colors_W.get(W, "#9E9E9E"),
                    label=f"W={W},C={C}",
                    alpha=0.8,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("KV Kept (tokens)")
    ax.set_title("KV Kept vs L (log-log)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, which="both")

    # Beta_eff heatmap by (W, C)
    ax = axes[2]
    ra_fits = {
        k: v for k, v in beta_fits.items() if "ra_value" in k[0] and k[3] == "control"
    }
    if ra_fits:
        Ws_plot = sorted(set(k[1] for k in ra_fits.keys()))
        Cs_plot = sorted(set(k[2] for k in ra_fits.keys()))
        beta_grid = np.full((len(Ws_plot), len(Cs_plot)), np.nan)
        for (_, W, C, _), fit in ra_fits.items():
            wi = Ws_plot.index(W)
            ci = Cs_plot.index(C)
            beta_grid[wi, ci] = fit["beta_eff"]

        im = ax.imshow(
            beta_grid,
            cmap="RdYlGn_r",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        ax.set_xticks(range(len(Cs_plot)))
        ax.set_xticklabels(Cs_plot)
        ax.set_yticks(range(len(Ws_plot)))
        ax.set_yticklabels(Ws_plot)
        ax.set_xlabel("Chunk Size C")
        ax.set_ylabel("Local Window W")
        ax.set_title("beta_eff (ra_value)")
        plt.colorbar(im, ax=ax, label="beta_eff")

        # Annotate
        for wi in range(len(Ws_plot)):
            for ci in range(len(Cs_plot)):
                val = beta_grid[wi, ci]
                if not np.isnan(val):
                    ax.text(
                        ci,
                        wi,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                    )

    plt.suptitle("BPA v8: Phase 1 — W/C Sweep")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase1_wc_sweep.png"), dpi=150)
    plt.close()

    # Plot 2: Pareto frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    seq_lens = sorted(set(a["seq_len"] for a in agg))
    markers_seq = ["o", "s", "^", "D"]
    for idx, seq_len in enumerate(seq_lens):
        points = [
            a
            for a in agg
            if a["seq_len"] == seq_len and a.get("stress_mode", "control") == "control"
        ]
        if not points:
            continue
        kv_vals = [p["effective_kv_kept_tokens"] for p in points]
        ppl_vals = [p["ppl_mean"] for p in points]

        # Color by variant
        for p, kv, ppl in zip(points, kv_vals, ppl_vals):
            color = (
                "#2196F3"
                if "dense" in p["variant"]
                else ("#4CAF50" if "ra_value" in p["variant"] else "#FF9800")
            )
            ax.scatter(
                kv,
                ppl,
                marker=markers_seq[idx % len(markers_seq)],
                color=color,
                alpha=0.6,
                s=40,
            )

    # Legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", label="dense"),
        Patch(facecolor="#4CAF50", label="ra_value"),
        Patch(facecolor="#FF9800", label="recency"),
    ]
    for idx, sl in enumerate(seq_lens):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers_seq[idx % len(markers_seq)],
                color="gray",
                label=f"L={sl}",
                linestyle="None",
            )
        )
    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_xlabel("Effective KV Kept (tokens)")
    ax.set_ylabel("Perplexity")
    ax.set_title("BPA v8: Pareto Frontier (PPL vs KV Kept)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pareto_frontier.png"), dpi=150)
    plt.close()

    # Plot 3: Stress test comparison
    stress_data = [a for a in agg if a.get("stress_mode") != "control"]
    if stress_data:
        stress_modes = sorted(set(a["stress_mode"] for a in stress_data))
        fig, axes = plt.subplots(
            1, len(stress_modes), figsize=(6 * len(stress_modes), 5)
        )
        if len(stress_modes) == 1:
            axes = [axes]

        for ax_idx, sm in enumerate(stress_modes):
            ax = axes[ax_idx]
            sm_data = [a for a in stress_data if a["stress_mode"] == sm]

            for variant in sorted(set(a["variant"] for a in sm_data)):
                v_data = [a for a in sm_data if a["variant"] == variant]
                v_data.sort(key=lambda x: x["seq_len"])
                Ls = [a["seq_len"] for a in v_data]
                ppls = [a["ppl_mean"] for a in v_data]

                color = (
                    "#2196F3"
                    if "dense" in variant
                    else ("#4CAF50" if "ra_value" in variant else "#FF9800")
                )
                ax.plot(Ls, ppls, "o-", color=color, label=variant, alpha=0.8)

            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("PPL")
            ax.set_title(f"Stress: {sm}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("BPA v8: Phase 2 — Stress Tests")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "phase2_stress_tests.png"), dpi=150)
        plt.close()

        # Plot 4: Region-wise PPL (early vs late)
        fig, ax = plt.subplots(figsize=(10, 6))
        for sm in stress_modes:
            for variant in sorted(
                set(a["variant"] for a in stress_data if a["stress_mode"] == sm)
            ):
                v_data = [
                    a
                    for a in stress_data
                    if a["variant"] == variant and a["stress_mode"] == sm
                ]
                v_data.sort(key=lambda x: x["seq_len"])
                for a in v_data:
                    extra = a.get("extra", {})
                    early = extra.get("ppl_early_mean", 0)
                    late = extra.get("ppl_late_mean", 0)
                    if early > 0 and late > 0:
                        color = (
                            "#2196F3"
                            if "dense" in variant
                            else ("#4CAF50" if "ra_value" in variant else "#FF9800")
                        )
                        marker = "o" if sm == "control" else "^"
                        ax.scatter(
                            early,
                            late,
                            marker=marker,
                            color=color,
                            alpha=0.7,
                            s=50,
                        )
                        ax.annotate(
                            f"{variant.replace('V8_', '')}\nL={a['seq_len']}\n{sm}",
                            (early, late),
                            fontsize=5,
                            alpha=0.6,
                        )

        ax.plot([0, 1000], [0, 1000], "k--", alpha=0.3, label="early=late")
        ax.set_xlabel("Early PPL (first half)")
        ax.set_ylabel("Late PPL (second half)")
        ax.set_title("BPA v8: Region-wise PPL")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "region_wise_ppl.png"), dpi=150)
        plt.close()

    # Plot 5: KV read traffic vs L
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(
        set(a["variant"] for a in agg if a.get("stress_mode", "control") == "control")
    ):
        # Use W=256, C=64 for traffic comparison
        data = [
            (a["seq_len"], a["kv_read_bytes_per_token"])
            for a in agg
            if a["variant"] == variant
            and a["local_window"] == 256
            and a["chunk_size"] == 64
            and a.get("stress_mode", "control") == "control"
        ]
        if not data:
            # Use any available W/C
            data = [
                (a["seq_len"], a["kv_read_bytes_per_token"])
                for a in agg
                if a["variant"] == variant
                and a.get("stress_mode", "control") == "control"
            ]
        if not data:
            continue

        # Deduplicate by taking first per seq_len
        by_L = {}
        for L, traffic in sorted(data):
            if L not in by_L:
                by_L[L] = traffic
        data = sorted(by_L.items())
        if len(data) < 2:
            continue

        Ls = [d[0] for d in data]
        traffic = [d[1] for d in data]
        color = (
            "#2196F3"
            if "dense" in variant
            else ("#4CAF50" if "ra_value" in variant else "#FF9800")
        )
        ax.plot(Ls, traffic, "o-", color=color, label=variant, alpha=0.8)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("KV Read Bytes/Token")
    ax.set_title("BPA v8: KV Read Traffic vs L")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kv_traffic_vs_L.png"), dpi=150)
    plt.close()

    print(f"Plots: {plots_dir}/")


def main():
    results_dir = "bpa_v8_results"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    results = load_results(results_dir)
    print(f"Loaded {len(results)} results")

    agg = aggregate(results)

    print("\nAggregated results:")
    for a in sorted(
        agg,
        key=lambda x: (
            x.get("stress_mode", "control"),
            x["seq_len"],
            x["local_window"],
            x["chunk_size"],
            x["ppl_mean"],
        ),
    ):
        stress = a.get("stress_mode", "control")
        print(
            f"  {a['variant']:20s} L={a['seq_len']:4d} "
            f"W={a['local_window']:3d} C={a['chunk_size']:3d} "
            f"PPL={a['ppl_mean']:6.1f}+/-{a['ppl_std']:4.1f} "
            f"kept={a['effective_kv_kept_tokens']:.0f} "
            f"stress={stress}"
        )

    beta_fits = compute_all_beta_fits(agg)
    if beta_fits:
        print("\nBeta_eff fits:")
        for (variant, W, C, stress), fit in sorted(beta_fits.items()):
            print(
                f"  {variant:20s} W={W:3d} C={C:3d} "
                f"beta={fit['beta_eff']:.3f} c={fit['c']:.1f} "
                f"R2={fit['r_squared']:.3f} stress={stress}"
            )

    # Save fits
    fits_path = os.path.join(results_dir, "beta_fits.json")
    fits_serializable = {
        f"{k[0]}_W{k[1]}_C{k[2]}_{k[3]}": v for k, v in beta_fits.items()
    }
    with open(fits_path, "w") as f:
        json.dump(fits_serializable, f, indent=2)

    generate_report(agg, beta_fits, results_dir)
    generate_plots(agg, beta_fits, results_dir)


if __name__ == "__main__":
    main()
