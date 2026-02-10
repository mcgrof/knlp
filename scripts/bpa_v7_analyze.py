#!/usr/bin/env python
"""
BPA v7 analysis: stress-test results, dynamic budget, cheap features,
bootstrap CI, Pareto frontier, failure mode table.
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
    """Aggregate across seeds for each (variant, seq_len, mode)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["variant"], r["seq_len"], r.get("mode", "unknown"))
        groups[key].append(r)

    agg = []
    for (variant, seq_len, mode), runs in sorted(groups.items()):
        ppls = [r["ppl_mean"] for r in runs]
        agg.append(
            {
                "variant": variant,
                "seq_len": seq_len,
                "mode": mode,
                "ppl_mean": float(np.mean(ppls)),
                "ppl_std": float(np.std(ppls)),
                "n_seeds": len(runs),
                "enabled_rate": runs[0].get("enabled_rate", 0.0),
                "far_budget": runs[0].get("far_budget", 0),
                "flops_relative": float(
                    np.mean([r.get("flops_relative", 1.0) for r in runs])
                ),
                "effective_kept_tokens": float(
                    np.mean([r.get("effective_kept_tokens", 0.0) for r in runs])
                ),
                "kv_bytes_read_per_token": float(
                    np.mean([r.get("kv_bytes_read_per_token", 0.0) for r in runs])
                ),
                "wall_ms_per_token": float(
                    np.mean([r.get("wall_ms_per_token", 0.0) for r in runs])
                ),
                "budget_mean": float(
                    np.mean([r.get("budget_mean", 0.0) for r in runs])
                ),
                "budget_p95": float(np.mean([r.get("budget_p95", 0.0) for r in runs])),
                "budget_p99": float(np.mean([r.get("budget_p99", 0.0) for r in runs])),
                "extra": runs[0].get("extra", {}),
                "all_ppls": ppls,
            }
        )
    return agg


def bootstrap_ci(ppls, n_boot=1000, ci=0.95, rng_seed=42):
    """Bootstrap confidence interval for mean PPL."""
    rng = np.random.RandomState(rng_seed)
    n = len(ppls)
    if n < 2:
        return ppls[0], ppls[0], ppls[0]

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(ppls, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(ppls)), lo, hi


def compute_bootstrap_cis(results):
    """Compute bootstrap CIs for each (variant, seq_len, mode)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["variant"], r["seq_len"], r.get("mode", "unknown"))
        groups[key].append(r["ppl_mean"])

    cis = {}
    for key, ppls in groups.items():
        mean, lo, hi = bootstrap_ci(ppls)
        cis[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(ppls)}
    return cis


def compute_pareto_frontier(agg, seq_len):
    """Compute PPL vs KV_kept Pareto frontier at given seq_len."""
    points = []
    for a in agg:
        if a["seq_len"] != seq_len:
            continue
        points.append(
            {
                "variant": a["variant"],
                "mode": a["mode"],
                "ppl": a["ppl_mean"],
                "kv_kept": a["effective_kept_tokens"],
                "flops_rel": a["flops_relative"],
            }
        )

    # Sort by KV kept (ascending)
    points.sort(key=lambda x: x["kv_kept"])

    # Find Pareto frontier: lowest PPL at each KV level
    frontier = []
    best_ppl = float("inf")
    for p in sorted(points, key=lambda x: x["kv_kept"]):
        if p["ppl"] < best_ppl:
            best_ppl = p["ppl"]
            frontier.append(p)

    return points, frontier


def find_failure_modes(agg, results):
    """Find cases where BPA underperforms dense baseline."""
    failures = []
    dense_ppls = {}
    for a in agg:
        if a["variant"] in ("V0_dense", "stress_control") and a["mode"] == "stress":
            dense_ppls[a["seq_len"]] = a["ppl_mean"]

    # If no dense baseline in stress results, use control
    if not dense_ppls:
        for a in agg:
            if a["variant"] == "stress_control":
                dense_ppls[a["seq_len"]] = a["ppl_mean"]

    for a in agg:
        if a["variant"] in ("V0_dense", "stress_control"):
            continue
        base = dense_ppls.get(a["seq_len"])
        if base is None:
            continue

        regression = (a["ppl_mean"] - base) / base * 100
        if regression > 5.0:
            failures.append(
                {
                    "variant": a["variant"],
                    "mode": a["mode"],
                    "seq_len": a["seq_len"],
                    "ppl": a["ppl_mean"],
                    "base_ppl": base,
                    "regression_pct": regression,
                    "extra": a.get("extra", {}),
                }
            )

    failures.sort(key=lambda x: x["regression_pct"], reverse=True)
    return failures


def fit_scaling_law(agg):
    """Fit KV_kept(L) = c * L^beta_eff."""
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

        A = np.vstack([log_L, np.ones(len(log_L))]).T
        result = np.linalg.lstsq(A, log_KV, rcond=None)
        beta_eff = float(result[0][0])
        log_c = float(result[0][1])
        c = math.exp(log_c)

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


def generate_report(agg, results, bootstrap_cis, output_dir: str):
    """Generate the v7 final report."""
    report = []
    report.append("# BPA v7: Stress-Test and Refinement Results\n")
    report.append(
        "> Does BPA + RA provide stable, information-aligned KV scaling "
        "as context length grows, or is the observed beta~0.25 an artifact "
        "of budget heuristics, layer choice, or evaluation range?\n"
    )

    report.append("## Model")
    report.append("- GPT2_RGSA (124M params), FineWebEdu, 615 iters")
    report.append("- Config: n_layer=12, n_head=12, n_embd=768")
    report.append("- local_window=256, chunk_size=64, far_budget=4")
    report.append("- Gate: v4 wbce_10x at 70% enabled_rate")
    report.append("- Surgical heads: 8 heads from layers 5-8\n")

    # Section 2: Stress test results
    stress_results = [a for a in agg if a["mode"] == "stress"]
    if stress_results:
        report.append("## Section 2: Stress-Test RA Assumptions\n")
        seq_lens = sorted(set(a["seq_len"] for a in stress_results))

        for seq_len in seq_lens:
            report.append(f"### L={seq_len}\n")
            report.append("| Variant | PPL | 95% CI | KV Kept | RA Mode |")
            report.append("|---------|-----|--------|---------|---------|")

            rows = [a for a in stress_results if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["ppl_mean"])

            for a in rows:
                ci_key = (a["variant"], a["seq_len"], a["mode"])
                ci = bootstrap_cis.get(ci_key, {})
                ci_lo = ci.get("ci_lo", a["ppl_mean"])
                ci_hi = ci.get("ci_hi", a["ppl_mean"])
                ra_mode = a.get("extra", {}).get("ra_mode", "?")
                report.append(
                    f"| {a['variant']} "
                    f"| {a['ppl_mean']:.1f}+/-{a['ppl_std']:.1f} "
                    f"| [{ci_lo:.1f}, {ci_hi:.1f}] "
                    f"| {a['effective_kept_tokens']:.0f} "
                    f"| {ra_mode} |"
                )
            report.append("")

        # Stress test interpretation
        report.append("### Stress-Test Interpretation\n")
        control_2048 = [
            a
            for a in stress_results
            if a["variant"] == "stress_control" and a["seq_len"] == 2048
        ]
        frozen_2048 = [
            a
            for a in stress_results
            if a["variant"] == "stress_frozen" and a["seq_len"] == 2048
        ]
        shuffled_2048 = [
            a
            for a in stress_results
            if a["variant"] == "stress_shuffled" and a["seq_len"] == 2048
        ]

        if control_2048 and frozen_2048:
            delta = frozen_2048[0]["ppl_mean"] - control_2048[0]["ppl_mean"]
            report.append(
                f"- **Frozen RA**: PPL delta={delta:+.1f} vs control at L=2048. "
            )
            if abs(delta) < 2.0:
                report.append(
                    "  Small delta suggests RA scores are relatively stable "
                    "and don't need frequent updates.\n"
                )
            else:
                report.append(
                    "  Significant delta confirms RA scores contain "
                    "dynamic, position-dependent signal.\n"
                )

        if control_2048 and shuffled_2048:
            delta = shuffled_2048[0]["ppl_mean"] - control_2048[0]["ppl_mean"]
            report.append(
                f"- **Shuffled RA**: PPL delta={delta:+.1f} vs control at L=2048. "
            )
            if abs(delta) > 2.0:
                report.append(
                    "  Performance collapse under shuffling confirms "
                    "RA signal is semantically aligned, not just a "
                    "proxy for recency or marginal statistics.\n"
                )
            else:
                report.append(
                    "  Minimal impact suggests RA may be acting as a "
                    "proxy for simpler statistics.\n"
                )

    # Section 3: Dynamic budget results
    dynbudget_results = [a for a in agg if a["mode"] == "dynamic-budget"]
    if dynbudget_results:
        report.append("## Section 3: Dynamic Budget\n")
        seq_lens = sorted(set(a["seq_len"] for a in dynbudget_results))

        for seq_len in seq_lens:
            report.append(f"### L={seq_len}\n")
            report.append(
                "| Variant | PPL | Budget Mean | Budget p95 " "| Budget p99 | KV Kept |"
            )
            report.append(
                "|---------|-----|------------|------------|" "------------|---------|"
            )

            rows = [a for a in dynbudget_results if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["ppl_mean"])

            for a in rows:
                report.append(
                    f"| {a['variant']} "
                    f"| {a['ppl_mean']:.1f}+/-{a['ppl_std']:.1f} "
                    f"| {a['budget_mean']:.1f} "
                    f"| {a['budget_p95']:.1f} "
                    f"| {a['budget_p99']:.1f} "
                    f"| {a['effective_kept_tokens']:.0f} |"
                )
            report.append("")

        # Dynamic budget interpretation
        report.append("### Dynamic Budget Interpretation\n")
        const_ra = [
            a
            for a in dynbudget_results
            if "constant_ra" in a["variant"] and a["seq_len"] == max(seq_lens)
        ]
        entropy_ra = [
            a
            for a in dynbudget_results
            if "entropy_ra" in a["variant"] and a["seq_len"] == max(seq_lens)
        ]
        conc_ra = [
            a
            for a in dynbudget_results
            if "concentration_ra" in a["variant"] and a["seq_len"] == max(seq_lens)
        ]

        if const_ra and entropy_ra:
            ppl_delta = entropy_ra[0]["ppl_mean"] - const_ra[0]["ppl_mean"]
            budget_delta = entropy_ra[0]["budget_mean"] - const_ra[0]["budget_mean"]
            report.append(
                f"- **Entropy budget**: PPL delta={ppl_delta:+.1f}, "
                f"budget delta={budget_delta:+.1f} vs constant"
            )
        if const_ra and conc_ra:
            ppl_delta = conc_ra[0]["ppl_mean"] - const_ra[0]["ppl_mean"]
            budget_delta = conc_ra[0]["budget_mean"] - const_ra[0]["budget_mean"]
            report.append(
                f"- **Concentration budget**: PPL delta={ppl_delta:+.1f}, "
                f"budget delta={budget_delta:+.1f} vs constant"
            )
        report.append("")

    # Section 4: Cheap features results
    cheap_results = [a for a in agg if a["mode"] == "cheap-features"]
    if cheap_results:
        report.append("## Section 4: Cheap RA-Derived Features\n")
        seq_lens = sorted(set(a["seq_len"] for a in cheap_results))

        for seq_len in seq_lens:
            report.append(f"### L={seq_len}\n")
            report.append("| Variant | PPL | 95% CI | KV Kept | Strategy |")
            report.append("|---------|-----|--------|---------|----------|")

            rows = [a for a in cheap_results if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["ppl_mean"])

            for a in rows:
                ci_key = (a["variant"], a["seq_len"], a["mode"])
                ci = bootstrap_cis.get(ci_key, {})
                ci_lo = ci.get("ci_lo", a["ppl_mean"])
                ci_hi = ci.get("ci_hi", a["ppl_mean"])
                strat = a.get("extra", {}).get("cheap_strategy", "?")
                report.append(
                    f"| {a['variant']} "
                    f"| {a['ppl_mean']:.1f}+/-{a['ppl_std']:.1f} "
                    f"| [{ci_lo:.1f}, {ci_hi:.1f}] "
                    f"| {a['effective_kept_tokens']:.0f} "
                    f"| {strat} |"
                )
            report.append("")

    # Section 6: Pareto frontier
    report.append("## PPL vs KV_Kept Pareto Frontier\n")
    for seq_len in sorted(set(a["seq_len"] for a in agg)):
        points, frontier = compute_pareto_frontier(agg, seq_len)
        if frontier:
            report.append(f"### Pareto Frontier at L={seq_len}\n")
            report.append("| Variant | PPL | KV Kept | FLOPs% |")
            report.append("|---------|-----|---------|--------|")
            for p in frontier:
                report.append(
                    f"| {p['variant']} "
                    f"| {p['ppl']:.1f} "
                    f"| {p['kv_kept']:.0f} "
                    f"| {p['flops_rel']*100:.1f}% |"
                )
            report.append("")

    # Section 6: Failure modes
    failures = find_failure_modes(agg, results)
    report.append("## Failure Mode Table\n")
    if failures:
        report.append("Cases where BPA variant PPL regresses >5% vs control:\n")
        report.append("| Variant | Mode | L | PPL | Base PPL | Regression | Signal |")
        report.append("|---------|------|---|-----|----------|------------|--------|")
        for f in failures[:20]:
            signal = "?"
            extra = f.get("extra", {})
            if extra.get("ra_mode") == "shuffled":
                signal = "alignment broken"
            elif extra.get("ra_mode") == "frozen":
                signal = "stale RA scores"
            elif extra.get("ra_mode") == "corrupt_nonsurgical":
                signal = "RA noise injection"
            elif extra.get("ra_mode") == "none":
                signal = "no RA signal"
            elif "recency" in f["variant"]:
                signal = "recency fallback"
            elif "random" in f["variant"]:
                signal = "random selection"
            report.append(
                f"| {f['variant']} "
                f"| {f['mode']} "
                f"| {f['seq_len']} "
                f"| {f['ppl']:.1f} "
                f"| {f['base_ppl']:.1f} "
                f"| +{f['regression_pct']:.1f}% "
                f"| {signal} |"
            )
        report.append("")
    else:
        report.append("No BPA variants show >5% regression vs control.\n")

    # Conclusions
    report.append("## Conclusions\n")
    report.append("### What assumption survived?\n")
    report.append("### What assumption broke?\n")
    report.append("### What remains uncertain?\n")

    report_text = "\n".join(report) + "\n"
    report_path = os.path.join(output_dir, "bpa_v7_final_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report: {report_path}")
    return report_text


def generate_plots(agg, results, bootstrap_cis, output_dir: str):
    """Generate v7 visualization plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Stress test comparison
    stress = [a for a in agg if a["mode"] == "stress"]
    if stress:
        seq_lens = sorted(set(a["seq_len"] for a in stress))
        n_lens = len(seq_lens)
        fig, axes = plt.subplots(1, min(n_lens, 3), figsize=(5 * min(n_lens, 3), 5))
        if n_lens == 1:
            axes = [axes]
        for ax_idx, seq_len in enumerate(seq_lens[:3]):
            ax = axes[ax_idx]
            rows = [a for a in stress if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["ppl_mean"])
            names = [a["variant"].replace("stress_", "") for a in rows]
            ppls = [a["ppl_mean"] for a in rows]

            # Color by RA mode
            colors = []
            for a in rows:
                rm = a.get("extra", {}).get("ra_mode", "")
                if rm == "normal":
                    colors.append("#4CAF50")
                elif rm == "frozen":
                    colors.append("#2196F3")
                elif rm == "shuffled":
                    colors.append("#FF9800")
                elif rm == "corrupt_nonsurgical":
                    colors.append("#F44336")
                else:
                    colors.append("#9E9E9E")

            # CI error bars
            ci_errs = []
            for a in rows:
                ci_key = (a["variant"], a["seq_len"], a["mode"])
                ci = bootstrap_cis.get(ci_key, {})
                lo = ci.get("ci_lo", a["ppl_mean"])
                hi = ci.get("ci_hi", a["ppl_mean"])
                ci_errs.append([a["ppl_mean"] - lo, hi - a["ppl_mean"]])

            ci_errs_arr = np.array(ci_errs).T
            ax.barh(
                names,
                ppls,
                xerr=ci_errs_arr,
                capsize=3,
                color=colors,
                alpha=0.8,
            )
            ax.set_xlabel("Perplexity")
            ax.set_title(f"Stress Tests L={seq_len}")
        plt.suptitle("BPA v7: RA Stress Tests (with 95% CI)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "stress_test_comparison.png"), dpi=150)
        plt.close()

    # Plot 2: Dynamic budget comparison
    dynbudget = [a for a in agg if a["mode"] == "dynamic-budget"]
    if dynbudget:
        seq_lens = sorted(set(a["seq_len"] for a in dynbudget))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: PPL comparison
        ax = axes[0]
        for seq_len in seq_lens:
            rows = [a for a in dynbudget if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["variant"])
            names = [a["variant"].replace("dynbudget_", "") for a in rows]
            ppls = [a["ppl_mean"] for a in rows]
            ax.plot(
                range(len(names)),
                ppls,
                "o-",
                label=f"L={seq_len}",
                alpha=0.8,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("PPL")
        ax.set_title("Dynamic Budget: PPL")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Budget distribution
        ax = axes[1]
        for seq_len in seq_lens:
            rows = [a for a in dynbudget if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["variant"])
            names = [a["variant"].replace("dynbudget_", "") for a in rows]
            budgets = [a["budget_mean"] for a in rows]
            ax.plot(
                range(len(names)),
                budgets,
                "s-",
                label=f"L={seq_len}",
                alpha=0.8,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Budget (chunks)")
        ax.set_title("Dynamic Budget: Mean Budget Used")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("BPA v7: Dynamic Budget Experiment")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "dynamic_budget_comparison.png"),
            dpi=150,
        )
        plt.close()

    # Plot 3: Cheap features comparison
    cheap = [a for a in agg if a["mode"] == "cheap-features"]
    if cheap:
        seq_lens = sorted(set(a["seq_len"] for a in cheap))
        n_lens = len(seq_lens)
        fig, axes = plt.subplots(
            1,
            min(n_lens, 3),
            figsize=(5 * min(n_lens, 3), 5),
        )
        if n_lens == 1:
            axes = [axes]

        for ax_idx, seq_len in enumerate(seq_lens[:3]):
            ax = axes[ax_idx]
            rows = [a for a in cheap if a["seq_len"] == seq_len]
            rows.sort(key=lambda x: x["ppl_mean"])
            names = [a["variant"].replace("cheap_", "") for a in rows]
            ppls = [a["ppl_mean"] for a in rows]

            colors = []
            for a in rows:
                s = a.get("extra", {}).get("cheap_strategy", "")
                if s == "ra_value":
                    colors.append("#4CAF50")
                elif s == "ra_ema":
                    colors.append("#2196F3")
                elif s == "ra_rank":
                    colors.append("#FF9800")
                else:
                    colors.append("#9E9E9E")

            ci_errs = []
            for a in rows:
                ci_key = (a["variant"], a["seq_len"], a["mode"])
                ci = bootstrap_cis.get(ci_key, {})
                lo = ci.get("ci_lo", a["ppl_mean"])
                hi = ci.get("ci_hi", a["ppl_mean"])
                ci_errs.append([a["ppl_mean"] - lo, hi - a["ppl_mean"]])

            ci_errs_arr = np.array(ci_errs).T
            ax.barh(
                names,
                ppls,
                xerr=ci_errs_arr,
                capsize=3,
                color=colors,
                alpha=0.8,
            )
            ax.set_xlabel("Perplexity")
            ax.set_title(f"Cheap Features L={seq_len}")
        plt.suptitle("BPA v7: Cheap RA Features (with 95% CI)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "cheap_features_comparison.png"),
            dpi=150,
        )
        plt.close()

    # Plot 4: Pareto frontier
    all_seq_lens = sorted(set(a["seq_len"] for a in agg))
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D"]
    for idx, seq_len in enumerate(all_seq_lens):
        points, frontier = compute_pareto_frontier(agg, seq_len)
        if not points:
            continue

        kv_vals = [p["kv_kept"] for p in points]
        ppl_vals = [p["ppl"] for p in points]
        ax.scatter(
            kv_vals,
            ppl_vals,
            marker=markers[idx % len(markers)],
            alpha=0.5,
            s=30,
            label=f"L={seq_len} (all)",
        )

        if frontier:
            f_kv = [p["kv_kept"] for p in frontier]
            f_ppl = [p["ppl"] for p in frontier]
            ax.plot(
                f_kv,
                f_ppl,
                "-",
                marker=markers[idx % len(markers)],
                alpha=0.9,
                linewidth=2,
                label=f"L={seq_len} (Pareto)",
            )

    ax.set_xlabel("Effective KV Kept (tokens)")
    ax.set_ylabel("Perplexity")
    ax.set_title("BPA v7: PPL vs KV Kept Pareto Frontier")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pareto_frontier.png"), dpi=150)
    plt.close()

    # Plot 5: Beta confidence intervals (scaling law)
    stress = [a for a in agg if a["mode"] == "stress"]
    if stress:
        fits = fit_scaling_law(stress)
        if fits:
            fig, ax = plt.subplots(figsize=(8, 5))
            variants = sorted(fits.keys())
            betas = [fits[v]["beta_eff"] for v in variants]
            r2s = [fits[v]["r_squared"] for v in variants]

            colors = []
            for v in variants:
                if "control" in v:
                    colors.append("#4CAF50")
                elif "frozen" in v:
                    colors.append("#2196F3")
                elif "shuffled" in v:
                    colors.append("#FF9800")
                elif "corrupt" in v:
                    colors.append("#F44336")
                else:
                    colors.append("#9E9E9E")

            ax.barh(variants, betas, color=colors, alpha=0.8)
            ax.set_xlabel("beta_eff")
            ax.set_title("BPA v7: Scaling Exponent by Stress Variant")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, "beta_by_variant.png"),
                dpi=150,
            )
            plt.close()

    print(f"Plots: {plots_dir}/")


def main():
    results_dir = "bpa_v7_results"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    results = load_results(results_dir)
    print(f"Loaded {len(results)} results")

    agg = aggregate(results)
    bootstrap_cis = compute_bootstrap_cis(results)

    print("\nAggregated results by mode:")
    for mode in sorted(set(a["mode"] for a in agg)):
        print(f"\n  === {mode} ===")
        mode_results = [a for a in agg if a["mode"] == mode]
        for a in sorted(mode_results, key=lambda x: (x["seq_len"], x["ppl_mean"])):
            ci_key = (a["variant"], a["seq_len"], a["mode"])
            ci = bootstrap_cis.get(ci_key, {})
            ci_lo = ci.get("ci_lo", a["ppl_mean"])
            ci_hi = ci.get("ci_hi", a["ppl_mean"])
            print(
                f"    {a['variant']:35s} L={a['seq_len']:4d} "
                f"PPL={a['ppl_mean']:6.1f}+/-{a['ppl_std']:4.1f} "
                f"CI=[{ci_lo:.1f},{ci_hi:.1f}] "
                f"kept={a['effective_kept_tokens']:.0f}"
            )

    # Pareto frontiers
    print("\nPareto frontiers:")
    for seq_len in sorted(set(a["seq_len"] for a in agg)):
        points, frontier = compute_pareto_frontier(agg, seq_len)
        print(f"\n  L={seq_len}: {len(frontier)} Pareto points")
        for p in frontier:
            print(
                f"    {p['variant']:35s} PPL={p['ppl']:.1f} "
                f"KV={p['kv_kept']:.0f} FLOPs={p['flops_rel']*100:.1f}%"
            )

    # Failure modes
    failures = find_failure_modes(agg, results)
    if failures:
        print(f"\nFailure modes ({len(failures)} cases >5% regression):")
        for f in failures[:10]:
            print(
                f"  {f['variant']:35s} L={f['seq_len']:4d} "
                f"PPL={f['ppl']:.1f} (+{f['regression_pct']:.1f}%)"
            )

    # Scaling fits
    scaling_fits = fit_scaling_law(agg)
    if scaling_fits:
        print("\nScaling law fits:")
        for variant, fit in sorted(scaling_fits.items()):
            print(
                f"  {variant:35s} beta={fit['beta_eff']:.3f} "
                f"c={fit['c']:.1f} R2={fit['r_squared']:.3f}"
            )

    # Save fits
    fits_path = os.path.join(results_dir, "scaling_fit.json")
    with open(fits_path, "w") as f:
        json.dump(scaling_fits, f, indent=2)

    # Save bootstrap CIs
    ci_path = os.path.join(results_dir, "bootstrap_cis.json")
    ci_serializable = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in bootstrap_cis.items()}
    with open(ci_path, "w") as f:
        json.dump(ci_serializable, f, indent=2)

    generate_report(agg, results, bootstrap_cis, results_dir)
    generate_plots(agg, results, bootstrap_cis, results_dir)


if __name__ == "__main__":
    main()
