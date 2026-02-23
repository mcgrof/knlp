#!/usr/bin/env python3
"""Generate plots for adam-lr-01 experiment results (Step 5).

Produces:
- loss_vs_iter.png: training loss curves
- val_ppl_vs_iter.png: validation PPL at checkpoints
- toks_per_sec.png: throughput bar chart
- lr_mult_vs_layer.png: LR multiplier vs layer depth at select checkpoints
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "runs/adam-lr-01/main_v2"
PLOT_DIR = "runs/adam-lr-01/plots"


def parse_stdout_log(log_path):
    """Extract training metrics from stdout log."""
    iters, losses, ppls, lrs = [], [], [], []
    val_iters, val_ppls = [], []
    toks_per_sec_samples = []
    effective_batch = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # Extract effective batch size
            if "Effective batch size" in line:
                try:
                    effective_batch = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass

            # Training log lines
            m = re.match(
                r"Iter\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+ppl\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)",
                line,
            )
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                ppls.append(float(m.group(3)))
                lrs.append(float(m.group(4)))

                ms_match = re.search(r"([\d.]+)ms/iter", line)
                if ms_match:
                    ms = float(ms_match.group(1))
                    tokens_per_iter = (effective_batch or 240) * 1024
                    toks_per_sec_samples.append(tokens_per_iter / (ms / 1000))

            # Eval lines: Eval @ iter N: train X, val Y, ppl Z
            em = re.match(
                r"Eval @ iter (\d+): train ([\d.]+), val ([\d.]+), ppl ([\d.]+)",
                line,
            )
            if em:
                val_iters.append(int(em.group(1)))
                val_ppls.append(float(em.group(4)))

    avg_toks_per_sec = (
        np.mean(toks_per_sec_samples[-50:]) if toks_per_sec_samples else 0
    )

    return {
        "iters": iters,
        "losses": losses,
        "ppls": ppls,
        "lrs": lrs,
        "val_iters": val_iters,
        "val_ppls": val_ppls,
        "avg_toks_per_sec": avg_toks_per_sec,
    }


def parse_layer_lr_jsonl(path):
    """Parse JSONL log of per-layer LR multipliers."""
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def plot_training_loss(all_data, output_path):
    """Plot training loss vs iteration for all runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "R0_baseline": "#333333",
        "R1_power1": "#2196F3",
        "R2_power2": "#F44336",
    }
    styles = {"R0_baseline": "-", "R1_power1": "--", "R2_power2": ":"}

    for config_name, seed_data in all_data.items():
        color = colors.get(config_name, "#999999")
        style = styles.get(config_name, "-")

        for seed, data in seed_data.items():
            alpha = 0.4 if len(seed_data) > 1 else 1.0
            ax.plot(
                data["iters"],
                data["losses"],
                style,
                color=color,
                alpha=alpha,
                linewidth=1.0,
            )

        # Plot mean if multiple seeds
        if len(seed_data) > 1:
            seeds = list(seed_data.values())
            min_len = min(len(s["iters"]) for s in seeds)
            mean_loss = np.mean([s["losses"][:min_len] for s in seeds], axis=0)
            ax.plot(
                seeds[0]["iters"][:min_len],
                mean_loss,
                style,
                color=color,
                linewidth=2.5,
                label=f"{config_name} (mean)",
            )
        elif seed_data:
            data = list(seed_data.values())[0]
            ax.plot([], [], style, color=color, linewidth=2.5, label=config_name)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss: Baseline vs Per-Layer LR Scaling", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_val_ppl(all_data, output_path):
    """Plot validation PPL vs iteration for all runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "R0_baseline": "#333333",
        "R1_power1": "#2196F3",
        "R2_power2": "#F44336",
    }
    markers = {"R0_baseline": "o", "R1_power1": "s", "R2_power2": "^"}

    for config_name, seed_data in all_data.items():
        color = colors.get(config_name, "#999999")
        marker = markers.get(config_name, "o")

        for seed, data in seed_data.items():
            if data["val_iters"]:
                alpha = 0.4 if len(seed_data) > 1 else 1.0
                ax.plot(
                    data["val_iters"],
                    data["val_ppls"],
                    marker=marker,
                    color=color,
                    alpha=alpha,
                    linewidth=1.0,
                    markersize=4,
                )

        # Plot mean if multiple seeds
        if len(seed_data) > 1:
            seeds_with_val = [s for s in seed_data.values() if len(s["val_iters"]) > 0]
            if len(seeds_with_val) > 1:
                min_len = min(len(s["val_ppls"]) for s in seeds_with_val)
                mean_ppl = np.mean(
                    [s["val_ppls"][:min_len] for s in seeds_with_val], axis=0
                )
                ax.plot(
                    seeds_with_val[0]["val_iters"][:min_len],
                    mean_ppl,
                    marker=marker,
                    color=color,
                    linewidth=2.5,
                    markersize=6,
                    label=f"{config_name} (mean)",
                )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Validation PPL", fontsize=12)
    ax.set_title("Validation PPL: Baseline vs Per-Layer LR Scaling", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_throughput(all_data, output_path):
    """Bar chart of average tokens/sec per config."""
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = []
    tps_means = []
    tps_stds = []
    colors = ["#333333", "#2196F3", "#F44336"]

    for config_name, seed_data in all_data.items():
        tps_vals = [
            d["avg_toks_per_sec"]
            for d in seed_data.values()
            if d["avg_toks_per_sec"] > 0
        ]
        configs.append(config_name)
        tps_means.append(np.mean(tps_vals) if tps_vals else 0)
        tps_stds.append(np.std(tps_vals) if len(tps_vals) > 1 else 0)

    x = np.arange(len(configs))
    bars = ax.bar(x, tps_means, yerr=tps_stds, capsize=5, color=colors[: len(configs)])

    for bar, mean in zip(bars, tps_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{mean:.0f}",
            ha="center",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title("Throughput Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_lr_mult_vs_layer(outdir, output_path):
    """Plot LR multiplier vs layer depth at select checkpoints."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (config_name, power_label) in enumerate(
        [("R1_power1", "power=1"), ("R2_power2", "power=2")]
    ):
        ax = axes[ax_idx]
        jsonl_path = os.path.join(outdir, f"{config_name}_s0", "layer_lr.jsonl")
        entries = parse_layer_lr_jsonl(jsonl_path)

        if not entries:
            ax.text(
                0.5,
                0.5,
                "No JSONL data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"LR Mult vs Layer ({power_label})")
            continue

        total_steps = entries[-1]["step"]
        checkpoints = [
            total_steps // 4,
            total_steps // 2,
            3 * total_steps // 4,
            total_steps,
        ]

        for cp_step in checkpoints:
            closest = min(entries, key=lambda e: abs(e["step"] - cp_step))
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
                markersize=4,
                label=f"step={closest['step']}",
            )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("LR Multiplier", fontsize=11)
        ax.set_title(f"LR Multiplier by Layer ({power_label})", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_data = {}
    for config_name in ["R0_baseline", "R1_power1", "R2_power2"]:
        all_data[config_name] = {}
        for seed in [0, 1]:
            run_name = f"{config_name}_s{seed}"
            log_path = os.path.join(OUTDIR, run_name, "stdout.log")
            if os.path.exists(log_path):
                data = parse_stdout_log(log_path)
                if data["iters"]:
                    all_data[config_name][seed] = data

    total_runs = sum(len(v) for v in all_data.values())
    if total_runs == 0:
        print("No data found. Run experiments first.")
        sys.exit(1)
    print(f"Found {total_runs} runs across {len(all_data)} configs")

    plot_training_loss(all_data, os.path.join(PLOT_DIR, "loss_vs_iter.png"))
    plot_val_ppl(all_data, os.path.join(PLOT_DIR, "val_ppl_vs_iter.png"))
    plot_throughput(all_data, os.path.join(PLOT_DIR, "toks_per_sec.png"))
    plot_lr_mult_vs_layer(OUTDIR, os.path.join(PLOT_DIR, "lr_mult_vs_layer.png"))

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
