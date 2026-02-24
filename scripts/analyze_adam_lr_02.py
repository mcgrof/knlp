#!/usr/bin/env python3
"""Analyze and plot adam-lr-02 Phase A results.

Produces:
- val_ppl_vs_tokens.png: validation PPL vs tokens (mean + shaded CI)
- tokens_to_threshold.png: tokens to reach target PPL thresholds
- lr_mult_vs_layer.png: LR multiplier by layer depth at checkpoints
- throughput.png: throughput bar chart
- RESULTS.md: narrative results
- STATS.md: statistical analysis

Usage:
    python scripts/analyze_adam_lr_02.py [--phase A|B]
"""

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR_A = "runs/adam-lr-02/phaseA"
OUTDIR_B = "runs/adam-lr-02/phaseB"
PLOT_DIR = "runs/adam-lr-02/plots"

COLORS = {
    "R0_baseline": "#333333",
    "R1_fisher_p1": "#2196F3",
    "R2_fisher_p2": "#F44336",
    "C1_random_shuffle": "#FF9800",
    "C2_depth_ramp": "#4CAF50",
    "C3_frozen_200": "#9C27B0",
}

LABELS = {
    "R0_baseline": "Baseline",
    "R1_fisher_p1": "Fisher p=1",
    "R2_fisher_p2": "Fisher p=2",
    "C1_random_shuffle": "Random shuffle (C1)",
    "C2_depth_ramp": "Depth ramp (C2)",
    "C3_frozen_200": "Frozen@200 (C3)",
}

LINE_STYLES = {
    "R0_baseline": "-",
    "R1_fisher_p1": "-",
    "R2_fisher_p2": "-",
    "C1_random_shuffle": "--",
    "C2_depth_ramp": "--",
    "C3_frozen_200": ":",
}


def load_results(outdir):
    """Load all run results from a phase directory."""
    results_path = os.path.join(outdir, "experiment_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)

    # Fallback: load individual result.json files
    results = []
    for entry in sorted(os.listdir(outdir)):
        rp = os.path.join(outdir, entry, "result.json")
        if os.path.isfile(rp):
            with open(rp) as f:
                results.append(json.load(f))
    return results


def load_trajectories(outdir):
    """Load val PPL trajectories, preferring result.json over stdout parsing."""
    data = {}  # config -> seed -> {val_iters, val_tokens, val_ppls, ...}
    for entry in sorted(os.listdir(outdir)):
        entry_dir = os.path.join(outdir, entry)
        if not os.path.isdir(entry_dir):
            continue
        m = re.match(r"(.+)_s(\d+)$", entry)
        if not m:
            continue
        config = m.group(1)
        seed = int(m.group(2))

        # Try result.json first (has val_trajectory)
        rp = os.path.join(entry_dir, "result.json")
        if os.path.isfile(rp):
            with open(rp) as f:
                result = json.load(f)
            vt = result.get("val_trajectory", [])
            eb = result.get("effective_batch") or 240
            if vt:
                tpi = eb * 1024
                data.setdefault(config, {})[seed] = {
                    "val_iters": [e["iter"] for e in vt],
                    "val_tokens": [e["iter"] * tpi for e in vt],
                    "val_ppls": [e["val_ppl"] for e in vt],
                    "train_iters": [e["iter"] for e in vt],
                    "train_tokens": [e["iter"] * tpi for e in vt],
                    "train_losses": [e["train_loss"] for e in vt],
                    "effective_batch": eb,
                }
                continue

        # Fallback: parse stdout.log
        path = os.path.join(entry_dir, "stdout.log")
        if not os.path.isfile(path):
            continue

        effective_batch = None
        val_iters = []
        val_ppls = []
        train_iters = []
        train_losses = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if "Effective batch size" in line:
                    try:
                        effective_batch = int(line.split(":")[-1].strip())
                    except (ValueError, IndexError):
                        pass

                em = re.match(
                    r"Eval @ iter (\d+): train ([\d.]+), val ([\d.]+), ppl ([\d.]+)",
                    line,
                )
                if em:
                    val_iters.append(int(em.group(1)))
                    val_ppls.append(float(em.group(4)))

                tm = re.match(
                    r"Iter\s+(\d+)\s+\|\s+loss\s+([\d.]+)",
                    line,
                )
                if tm:
                    train_iters.append(int(tm.group(1)))
                    train_losses.append(float(tm.group(2)))

        if not val_iters:
            continue

        tpi = (effective_batch or 240) * 1024
        data.setdefault(config, {})[seed] = {
            "val_iters": val_iters,
            "val_tokens": [it * tpi for it in val_iters],
            "val_ppls": val_ppls,
            "train_iters": train_iters,
            "train_tokens": [it * tpi for it in train_iters],
            "train_losses": train_losses,
            "effective_batch": effective_batch,
        }

    return data


def load_layer_lr_jsonl(outdir):
    """Load layer LR JSONL files per config."""
    data = {}
    for entry in sorted(os.listdir(outdir)):
        path = os.path.join(outdir, entry, "layer_lr.jsonl")
        if not os.path.isfile(path):
            continue
        m = re.match(r"(.+)_s(\d+)$", entry)
        if not m:
            continue
        config = m.group(1)
        seed = int(m.group(2))
        entries = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        if entries:
            data.setdefault(config, {})[seed] = entries
    return data


def plot_val_ppl_vs_tokens(trajectories, output_path):
    """Plot validation PPL vs tokens with mean and shaded CI."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for config in [
        "R0_baseline",
        "R1_fisher_p1",
        "R2_fisher_p2",
        "C1_random_shuffle",
        "C2_depth_ramp",
        "C3_frozen_200",
    ]:
        if config not in trajectories:
            continue
        seeds = trajectories[config]
        color = COLORS.get(config, "#999999")
        label = LABELS.get(config, config)
        style = LINE_STYLES.get(config, "-")

        if len(seeds) == 1:
            s = list(seeds.values())[0]
            ax.plot(
                s["val_tokens"],
                s["val_ppls"],
                style,
                color=color,
                linewidth=2,
                label=label,
            )
        else:
            # Align to common token axis
            all_ppls = []
            min_len = min(len(s["val_ppls"]) for s in seeds.values())
            tokens = list(seeds.values())[0]["val_tokens"][:min_len]
            for s in seeds.values():
                all_ppls.append(s["val_ppls"][:min_len])
            all_ppls = np.array(all_ppls)
            mean = np.mean(all_ppls, axis=0)
            std = np.std(all_ppls, axis=0)

            ax.plot(tokens, mean, style, color=color, linewidth=2, label=label)
            ax.fill_between(
                tokens,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
            )

    ax.set_xlabel("Tokens Processed", fontsize=13)
    ax.set_ylabel("Validation PPL", fontsize=13)
    ax.set_title(
        "adam-lr-02 Phase A: Validation PPL vs Tokens (250M budget)",
        fontsize=14,
    )
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tokens_to_threshold(trajectories, output_path):
    """Box/violin plot of tokens needed to reach target PPL thresholds."""
    thresholds = [500, 400, 300]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 6))

    config_order = [
        "R0_baseline",
        "R1_fisher_p1",
        "R2_fisher_p2",
        "C1_random_shuffle",
        "C2_depth_ramp",
        "C3_frozen_200",
    ]

    for ax_idx, threshold in enumerate(thresholds):
        ax = axes[ax_idx] if len(thresholds) > 1 else axes
        positions = []
        values = []
        labels = []
        colors_list = []

        for i, config in enumerate(config_order):
            if config not in trajectories:
                continue
            seeds = trajectories[config]
            tokens_to_thresh = []
            for s in seeds.values():
                for t, p in zip(s["val_tokens"], s["val_ppls"]):
                    if p <= threshold:
                        tokens_to_thresh.append(t)
                        break
                else:
                    tokens_to_thresh.append(float("inf"))

            finite = [t for t in tokens_to_thresh if t != float("inf")]
            if finite:
                positions.append(i)
                values.append(finite)
                labels.append(LABELS.get(config, config))
                colors_list.append(COLORS.get(config, "#999999"))

        if values:
            bp = ax.boxplot(values, positions=positions, widths=0.6, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax.set_title(f"Tokens to PPL <= {threshold}", fontsize=12)
        ax.set_ylabel("Tokens", fontsize=11)
        if labels:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_lr_mult_vs_layer(layer_lr_data, output_path):
    """Plot LR multipliers vs layer depth for Fisher configs."""
    configs_to_plot = [
        ("R1_fisher_p1", "Fisher p=1"),
        ("R2_fisher_p2", "Fisher p=2"),
        ("C1_random_shuffle", "Random shuffle"),
        ("C2_depth_ramp", "Depth ramp"),
        ("C3_frozen_200", "Frozen@200"),
    ]

    n_plots = sum(1 for c, _ in configs_to_plot if c in layer_lr_data)
    if n_plots == 0:
        print("No layer LR data found, skipping lr_mult plot")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    for config, title in configs_to_plot:
        if config not in layer_lr_data:
            continue
        ax = axes[ax_idx]
        ax_idx += 1

        # Use seed 0
        seed0_entries = layer_lr_data[config].get(0, [])
        if not seed0_entries:
            seed0_entries = list(layer_lr_data[config].values())[0]

        total_steps = seed0_entries[-1]["step"]
        checkpoints = [
            total_steps // 4,
            total_steps // 2,
            3 * total_steps // 4,
            total_steps,
        ]

        for cp_step in checkpoints:
            closest = min(seed0_entries, key=lambda e: abs(e["step"] - cp_step))
            layers = closest["layers"]

            def sort_key(name):
                m = re.match(r"h\.(\d+)", name)
                if m:
                    return (1, int(m.group(1)))
                if name == "embed":
                    return (0, 0)
                if name == "ln_f":
                    return (2, 0)
                if name == "lm_head":
                    return (3, 0)
                return (4, 0)

            sorted_layers = sorted(layers.items(), key=lambda x: sort_key(x[0]))
            names = [n for n, _ in sorted_layers]
            mults = [v for _, v in sorted_layers]

            ax.plot(
                range(len(names)),
                mults,
                "o-",
                markersize=3,
                label=f"step={closest['step']}",
            )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("LR Multiplier", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_throughput(results, output_path):
    """Bar chart of throughput per config."""
    fig, ax = plt.subplots(figsize=(10, 5))

    config_order = [
        "R0_baseline",
        "R1_fisher_p1",
        "R2_fisher_p2",
        "C1_random_shuffle",
        "C2_depth_ramp",
        "C3_frozen_200",
    ]

    means = []
    stds = []
    labels = []
    colors_list = []

    for config in config_order:
        tps_vals = [
            r["toks_per_sec"]
            for r in results
            if r["config"] == config and r.get("toks_per_sec")
        ]
        if tps_vals:
            means.append(np.mean(tps_vals))
            stds.append(np.std(tps_vals))
            labels.append(LABELS.get(config, config))
            colors_list.append(COLORS.get(config, "#999999"))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_list, alpha=0.8)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{mean:.0f}",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title("Throughput Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def compute_auc(tokens, ppls, token_range=None):
    """Compute AUC of log(ppl) vs tokens using trapezoidal rule."""
    tokens = np.array(tokens, dtype=float)
    ppls = np.array(ppls, dtype=float)
    log_ppls = np.log(ppls)

    if token_range:
        mask = (tokens >= token_range[0]) & (tokens <= token_range[1])
        tokens = tokens[mask]
        log_ppls = log_ppls[mask]

    if len(tokens) < 2:
        return float("nan")

    return np.trapezoid(log_ppls, tokens)


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)
    if n < 2:
        return np.mean(values), np.mean(values), np.mean(values)
    boot_means = np.array(
        [
            np.mean(np.random.choice(values, size=n, replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.mean(values), lo, hi


def write_stats(results, trajectories, outdir):
    """Write STATS.md with statistical analysis."""
    stats_path = os.path.join(outdir, "..", "STATS.md")
    config_order = [
        "R0_baseline",
        "R1_fisher_p1",
        "R2_fisher_p2",
        "C1_random_shuffle",
        "C2_depth_ramp",
        "C3_frozen_200",
    ]

    lines = [
        "# adam-lr-02 Statistical Analysis\n",
        "## AUC(log(val_ppl)) over Token Interval\n",
        "AUC = integral of log(val_ppl) over tokens, computed via trapezoidal rule.\n",
        "Lower AUC = better convergence efficiency.\n",
    ]

    # Compute AUC per config per seed
    auc_data = {}
    for config in config_order:
        if config not in trajectories:
            continue
        aucs = []
        for seed, s in trajectories[config].items():
            auc = compute_auc(s["val_tokens"], s["val_ppls"])
            if not np.isnan(auc):
                aucs.append(auc)
        if aucs:
            auc_data[config] = aucs

    lines.append("\n| Config | Mean AUC | StdDev | 95% CI | N |\n")
    lines.append("|--------|----------|--------|--------|---|\n")
    for config in config_order:
        if config not in auc_data:
            continue
        vals = auc_data[config]
        mean, lo, hi = bootstrap_ci(vals)
        std = np.std(vals)
        lines.append(
            f"| {LABELS.get(config, config)} | {mean:.2e} | {std:.2e} | "
            f"[{lo:.2e}, {hi:.2e}] | {len(vals)} |\n"
        )

    # Final val PPL stats
    lines.append("\n## Final Validation PPL\n")
    lines.append("\n| Config | Mean | StdDev | 95% CI | N | vs Baseline |\n")
    lines.append("|--------|------|--------|--------|---|-------------|\n")

    ppl_data = {}
    for config in config_order:
        ppls = [
            r["best_val_ppl"]
            for r in results
            if r["config"] == config and r.get("best_val_ppl") is not None
        ]
        if ppls:
            ppl_data[config] = ppls

    baseline_mean = np.mean(ppl_data.get("R0_baseline", [0]))
    for config in config_order:
        if config not in ppl_data:
            continue
        vals = ppl_data[config]
        mean, lo, hi = bootstrap_ci(vals)
        std = np.std(vals)
        improvement = ""
        if baseline_mean and "baseline" not in config:
            pct = (mean - baseline_mean) / baseline_mean * 100
            improvement = f"{pct:+.1f}%"
        lines.append(
            f"| {LABELS.get(config, config)} | {mean:.2f} | {std:.2f} | "
            f"[{lo:.2f}, {hi:.2f}] | {len(vals)} | {improvement} |\n"
        )

    lines.append("\n## Bootstrap Procedure\n")
    lines.append("- 10,000 bootstrap resamples\n")
    lines.append("- 95% confidence intervals (percentile method)\n")
    lines.append(f"- Seeds used: {sorted(set(r['seed'] for r in results))}\n")

    with open(stats_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {stats_path}")


def write_results(results, trajectories, outdir, phase):
    """Write RESULTS.md with narrative and decision gate."""
    results_path = os.path.join(outdir, "..", "RESULTS.md")
    config_order = [
        "R0_baseline",
        "R1_fisher_p1",
        "R2_fisher_p2",
        "C1_random_shuffle",
        "C2_depth_ramp",
        "C3_frozen_200",
    ]

    # Collect per-config PPL data
    ppl_data = {}
    for config in config_order:
        ppls = [
            r["best_val_ppl"]
            for r in results
            if r["config"] == config and r.get("best_val_ppl") is not None
        ]
        if ppls:
            ppl_data[config] = ppls

    baseline_mean = np.mean(ppl_data.get("R0_baseline", [0]))
    baseline_std = np.std(ppl_data.get("R0_baseline", [0]))

    # Compute AUC per config
    auc_data = {}
    for config in config_order:
        if config not in trajectories:
            continue
        aucs = []
        for seed, s in trajectories[config].items():
            auc = compute_auc(s["val_tokens"], s["val_ppls"])
            if not np.isnan(auc):
                aucs.append(auc)
        if aucs:
            auc_data[config] = aucs

    lines = [
        "# adam-lr-02: Per-Layer LR Scaling Falsification Study\n\n",
    ]

    # Summary
    lines.append("## Summary\n\n")
    lines.append(
        "This experiment tests whether per-layer learning rate scaling\n"
        "derived from Adam's second moment (diagonal Fisher proxy) provides\n"
        "a genuine optimization benefit, or whether the gains can be\n"
        "explained by simpler alternatives.\n\n"
    )
    lines.append(f"**Phase {phase}**: 250M tokens, 3 seeds, 6 configurations.\n\n")

    # Results table
    lines.append("## Results Table\n\n")
    lines.append("| Config | Seed | Iters | Best Val PPL | Toks/s | Time (s) |\n")
    lines.append("|--------|------|-------|-------------|--------|----------|\n")
    for r in sorted(results, key=lambda x: (x["config"], x["seed"])):
        vppl = f"{r['best_val_ppl']:.2f}" if r.get("best_val_ppl") else "N/A"
        tps = f"{r['toks_per_sec']:.0f}" if r.get("toks_per_sec") else "N/A"
        lines.append(
            f"| {r['config']} | {r['seed']} | {r['final_iter']} | "
            f"{vppl} | {tps} | {r['elapsed_s']:.0f} |\n"
        )

    # Mean table
    lines.append("\n### Mean over seeds\n\n")
    lines.append("| Config | Mean Val PPL | StdDev | 95% CI | vs Baseline |\n")
    lines.append("|--------|-------------|--------|--------|-------------|\n")

    for config in config_order:
        if config not in ppl_data:
            continue
        vals = ppl_data[config]
        mean, lo, hi = bootstrap_ci(vals)
        std = np.std(vals)
        improvement = "-"
        if baseline_mean and "baseline" not in config:
            pct = (mean - baseline_mean) / baseline_mean * 100
            improvement = f"{pct:+.1f}%"
        lines.append(
            f"| {LABELS.get(config, config)} | {mean:.2f} | {std:.2f} | "
            f"[{lo:.2f}, {hi:.2f}] | {improvement} |\n"
        )

    # Decision gate
    lines.append("\n## Decision Gate\n\n")

    # Evaluate gate criteria
    fisher_configs = ["R1_fisher_p1", "R2_fisher_p2"]
    control_configs = ["C1_random_shuffle", "C2_depth_ramp", "C3_frozen_200"]

    best_fisher = None
    best_fisher_mean = float("inf")
    for fc in fisher_configs:
        if fc in ppl_data:
            m = np.mean(ppl_data[fc])
            if m < best_fisher_mean:
                best_fisher = fc
                best_fisher_mean = m

    best_control = None
    best_control_mean = float("inf")
    for cc in control_configs:
        if cc in ppl_data:
            m = np.mean(ppl_data[cc])
            if m < best_control_mean:
                best_control = cc
                best_control_mean = m

    if best_fisher and baseline_mean:
        fisher_vs_base = (best_fisher_mean - baseline_mean) / baseline_mean * 100
        lines.append(
            f"Best Fisher config: **{LABELS.get(best_fisher, best_fisher)}** "
            f"(mean PPL={best_fisher_mean:.2f}, {fisher_vs_base:+.1f}% vs baseline)\n\n"
        )

    if best_control:
        control_vs_base = (best_control_mean - baseline_mean) / baseline_mean * 100
        lines.append(
            f"Best control: **{LABELS.get(best_control, best_control)}** "
            f"(mean PPL={best_control_mean:.2f}, {control_vs_base:+.1f}% vs baseline)\n\n"
        )

    gate_pass = False
    if best_fisher and best_control and baseline_mean:
        fisher_beats_baseline = best_fisher_mean < baseline_mean
        fisher_beats_controls = best_fisher_mean < best_control_mean

        if fisher_beats_baseline and fisher_beats_controls:
            lines.append(
                "**GATE: PASS** - Fisher-LR beats both baseline and all controls.\n\n"
            )
            lines.append(
                f"Proceed to Phase B with {LABELS.get(best_fisher, best_fisher)} "
                f"vs baseline vs {LABELS.get(best_control, best_control)} "
                f"at 1B tokens, 5 seeds.\n\n"
            )
            gate_pass = True
        elif fisher_beats_baseline and not fisher_beats_controls:
            lines.append(
                "**GATE: PARTIAL** - Fisher-LR beats baseline but not all controls.\n\n"
            )
            lines.append(
                "The improvement may be explained by LR heterogeneity alone, "
                "not Fisher-specific information.\n\n"
            )
        else:
            lines.append("**GATE: FAIL** - Fisher-LR does not beat baseline.\n\n")
            lines.append("Effect may have been a confound in adam-lr-01.\n\n")

    # Skeptic section
    lines.append("## Addressing Skepticism\n\n")
    lines.append("### Could random LR heterogeneity explain the gain?\n\n")
    if "C1_random_shuffle" in ppl_data and best_fisher in ppl_data:
        c1_mean = np.mean(ppl_data["C1_random_shuffle"])
        delta = (c1_mean - best_fisher_mean) / best_fisher_mean * 100
        if c1_mean > best_fisher_mean:
            lines.append(
                f"No. Random shuffle achieves PPL={c1_mean:.2f}, "
                f"which is {delta:+.1f}% worse than Fisher. "
                "The layer-assignment matters.\n\n"
            )
        else:
            lines.append(
                f"Possibly. Random shuffle achieves PPL={c1_mean:.2f}, "
                f"competitive with Fisher ({best_fisher_mean:.2f}). "
                "Fisher-specificity not confirmed.\n\n"
            )

    lines.append("### Could a simple depth heuristic suffice?\n\n")
    if "C2_depth_ramp" in ppl_data and best_fisher in ppl_data:
        c2_mean = np.mean(ppl_data["C2_depth_ramp"])
        delta = (c2_mean - best_fisher_mean) / best_fisher_mean * 100
        if c2_mean > best_fisher_mean:
            lines.append(
                f"No. Depth ramp achieves PPL={c2_mean:.2f}, "
                f"{delta:+.1f}% worse than Fisher.\n\n"
            )
        else:
            lines.append(
                f"Yes. Depth ramp achieves PPL={c2_mean:.2f}, "
                f"competitive with Fisher.\n\n"
            )

    lines.append("### Is dynamic adaptation necessary?\n\n")
    if "C3_frozen_200" in ppl_data and best_fisher in ppl_data:
        c3_mean = np.mean(ppl_data["C3_frozen_200"])
        delta = (c3_mean - best_fisher_mean) / best_fisher_mean * 100
        if c3_mean > best_fisher_mean * 1.02:
            lines.append(
                f"Yes. Frozen multipliers achieve PPL={c3_mean:.2f}, "
                f"{delta:+.1f}% worse. Dynamic updates provide benefit.\n\n"
            )
        else:
            lines.append(
                f"No. Frozen multipliers achieve PPL={c3_mean:.2f}, "
                f"within 2% of dynamic Fisher ({best_fisher_mean:.2f}). "
                "Early snapshot may suffice.\n\n"
            )

    lines.append("## Plots\n\n")
    lines.append("- `plots/val_ppl_vs_tokens.png`: Val PPL vs tokens\n")
    lines.append("- `plots/tokens_to_threshold.png`: Tokens to reach PPL targets\n")
    lines.append("- `plots/lr_mult_vs_layer.png`: LR multipliers by layer\n")
    lines.append("- `plots/throughput.png`: Throughput comparison\n")

    with open(results_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {results_path}")

    return gate_pass


def main():
    parser = argparse.ArgumentParser(description="Analyze adam-lr-02 results")
    parser.add_argument(
        "--phase",
        type=str,
        default="A",
        choices=["A", "B"],
    )
    args = parser.parse_args()

    outdir = OUTDIR_A if args.phase == "A" else OUTDIR_B
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Loading results from {outdir}...")
    results = load_results(outdir)
    if not results:
        print("No results found.")
        sys.exit(1)

    print(f"Found {len(results)} runs")

    trajectories = load_trajectories(outdir)
    layer_lr_data = load_layer_lr_jsonl(outdir)

    # Generate plots
    plot_val_ppl_vs_tokens(
        trajectories, os.path.join(PLOT_DIR, "val_ppl_vs_tokens.png")
    )
    plot_tokens_to_threshold(
        trajectories, os.path.join(PLOT_DIR, "tokens_to_threshold.png")
    )
    plot_lr_mult_vs_layer(layer_lr_data, os.path.join(PLOT_DIR, "lr_mult_vs_layer.png"))
    plot_throughput(results, os.path.join(PLOT_DIR, "throughput.png"))

    # Write reports
    write_stats(results, trajectories, outdir)
    gate_pass = write_results(results, trajectories, outdir, args.phase)

    print(f"\nAll outputs saved to {PLOT_DIR}/ and runs/adam-lr-02/")
    if gate_pass:
        print("\nDECISION GATE: PASS - proceed to Phase B")
    elif gate_pass is False:
        print("\nDECISION GATE: FAIL or PARTIAL - see RESULTS.md")
    else:
        print("\nDECISION GATE: insufficient data")


if __name__ == "__main__":
    main()
