#!/usr/bin/env python3
"""BPA v27: Generate publication-grade figures and tables from source artifacts."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths
ARTIFACTS = "artifacts/v27"
FIGURES = os.path.join(ARTIFACTS, "figures")
TABLES = os.path.join(ARTIFACTS, "tables")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

# Style
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Source data
# ============================================================

# v24 data (W7900)
v24_scoreboard = load_json("bpa_v24_scoreboard.json")
v24_params = load_json("results/v24/artifacts/v24/parameter_estimates.json")

# v26 data (H100)
v26_scoreboard = load_json("bpa_v26_scoreboard.json")
v26_oracle_qwen7b = load_json(
    "results/v26/artifacts/v26/oracle_sensitivity_qwen7b.json"
)
v26_oracle_mistral7b = load_json(
    "results/v26/artifacts/v26/oracle_sensitivity_mistral7b.json"
)
v26_kstar_qwen7b = load_json("results/v26/artifacts/v26/k_star_qwen7b.json")
v26_kstar_mistral7b = load_json("results/v26/artifacts/v26/k_star_mistral7b.json")

# v27 data (H100 confirmatory)
v27_kstar_llama = load_json("results/v27/h100_confirmatory/k_star_llama2_7b.json")

# Canonical model data
MODELS = [
    {
        "name": "Qwen2.5-0.5B",
        "short": "Q-0.5B",
        "D": 24,
        "k_star_3": 2,
        "k_star_1": None,
        "kv_ratio": 0.3008,
        "max_delta": 2.85,
        "layer0_sens": 23.48,
        "tail_frac": 0.336,
        "arch": "Qwen2",
        "gpu": "W7900",
        "alpha": v24_params["qwen05b"]["alpha"],
        "k_configs": v24_scoreboard["models"]["qwen05b"]["configs"],
    },
    {
        "name": "Qwen2.5-1.5B",
        "short": "Q-1.5B",
        "D": 28,
        "k_star_3": 2,
        "k_star_1": 3,
        "kv_ratio": 0.2974,
        "max_delta": 1.05,
        "layer0_sens": 824.55,
        "tail_frac": 0.009,
        "arch": "Qwen2",
        "gpu": "W7900",
        "alpha": v24_params["qwen15b"]["alpha"],
        "k_configs": v24_scoreboard["models"]["qwen15b"]["configs"],
    },
    {
        "name": "Qwen2.5-7B",
        "short": "Q-7B",
        "D": 28,
        "k_star_3": 2,
        "k_star_1": 3,
        "kv_ratio": 0.2974,
        "max_delta": 1.05,
        "layer0_sens": 140154.49,
        "tail_frac": 0.0001,
        "arch": "Qwen2",
        "gpu": "H100",
        "alpha": [s["max_delta"] for s in v26_oracle_qwen7b["oracle_scores"]],
        "k_configs": v26_kstar_qwen7b["k_results"],
    },
    {
        "name": "Mistral-7B",
        "short": "M-7B",
        "D": 32,
        "k_star_3": 0,
        "k_star_1": 0,
        "kv_ratio": 0.2812,
        "max_delta": 0.48,
        "layer0_sens": 0.41,
        "tail_frac": 0.140,
        "arch": "Mistral",
        "gpu": "H100",
        "alpha": [s["max_delta"] for s in v26_oracle_mistral7b["oracle_scores"]],
        "k_configs": v26_kstar_mistral7b["k_results"],
    },
    {
        "name": "Llama-2-7b",
        "short": "L-7B",
        "D": 32,
        "k_star_3": 0,
        "k_star_1": 2,
        "kv_ratio": 0.2812,
        "max_delta": 1.01,
        "layer0_sens": 0.76,
        "tail_frac": 0.406,
        "arch": "Llama",
        "gpu": "H100",
        "alpha": [s["max_delta"] for s in v27_kstar_llama["oracle_scores"]],
        "k_configs": v27_kstar_llama["k_results"],
    },
]


# ============================================================
# Table 2.1: Canonical k* results
# ============================================================
def write_canonical_table():
    header = (
        "| Model | Arch | D | GPU | Quant | Ranking | eps | "
        "k*(3%) | kv_ratio | max_delta | PASS_3% | PASS_1% |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    rows = [header, sep]
    for m in MODELS:
        p1 = "Y" if m["k_star_1"] is not None else "N"
        rows.append(
            f"| {m['name']} | {m['arch']} | {m['D']} | {m['gpu']} | "
            f"g32 INT4/INT8 | oracle | 3% | "
            f"{m['k_star_3']} | {m['kv_ratio']:.4f} | "
            f"{m['max_delta']:.2f}% | Y | {p1} |"
        )
    path = os.path.join(TABLES, "table_1_canonical_kstar.md")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  Table 1 -> {path}")


# ============================================================
# Plot 2.2: k* vs D (and k*/D vs D)
# ============================================================
def plot_kstar_vs_D():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    arch_style = {
        "Qwen2": ("o", "#2196F3"),
        "Mistral": ("s", "#FF5722"),
        "Llama": ("D", "#4CAF50"),
    }

    for m in MODELS:
        marker, color = arch_style.get(m["arch"], ("o", "#999"))
        ax1.scatter(
            m["D"],
            m["k_star_3"],
            s=100,
            marker=marker,
            color=color,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.annotate(
            m["short"],
            (m["D"], m["k_star_3"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

        kd = m["k_star_3"] / m["D"]
        ax2.scatter(
            m["D"],
            kd,
            s=100,
            marker=marker,
            color=color,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax2.annotate(
            m["short"],
            (m["D"], kd),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    # Reference lines
    ax1.axhline(y=2, color="gray", linestyle="--", alpha=0.5, label="k*=2")
    ax1.set_xlabel("Depth D (layers)")
    ax1.set_ylabel("k*(3%)")
    ax1.set_title("k* vs Model Depth")
    ax1.set_xlim(20, 36)
    ax1.set_ylim(-0.5, 5)
    ax1.set_xticks([24, 28, 32])
    ax1.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="#2196F3", linestyle="", label="Qwen2"
            ),
            plt.Line2D(
                [0], [0], marker="s", color="#FF5722", linestyle="", label="Mistral"
            ),
            plt.Line2D(
                [0], [0], marker="D", color="#4CAF50", linestyle="", label="Llama"
            ),
        ]
    )

    ax2.set_xlabel("Depth D (layers)")
    ax2.set_ylabel("k*/D")
    ax2.set_title("Protected Fraction vs Depth")
    ax2.set_xlim(20, 36)
    ax2.set_ylim(-0.01, 0.12)
    ax2.set_xticks([24, 28, 32])

    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_1_kstar_vs_D.png")
    plt.savefig(path)
    plt.close()
    print(f"  Fig 1 -> {path}")


# ============================================================
# Plot 2.3: Oracle sensitivity distributions
# ============================================================
def plot_sensitivity_distributions():
    n = len(MODELS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    arch_colors = {"Qwen2": "#2196F3", "Mistral": "#FF5722", "Llama": "#4CAF50"}

    for idx, m in enumerate(MODELS):
        ax = axes_flat[idx]
        alpha = sorted(m["alpha"], reverse=True)
        x = np.arange(len(alpha))

        # Use log scale for Qwen models due to extreme sink
        use_log = m["layer0_sens"] > 10
        bar_color = arch_colors.get(m["arch"], "#999")

        if use_log:
            alpha_plot = [max(a, 0.01) for a in alpha]
            ax.bar(x, alpha_plot, color=bar_color)
            ax.set_yscale("log")
            ax.set_ylabel("max |delta%| (log)")
        else:
            ax.bar(x, alpha, color=bar_color)
            ax.set_ylabel("max |delta%|")

        # Mark k* boundary
        k = m["k_star_3"]
        if k > 0 and k < len(alpha):
            ax.axvline(
                x=k - 0.5, color="red", linestyle="--", linewidth=1.5, label=f"k*={k}"
            )
            ax.legend()

        ax.set_xlabel("Layer rank (by sensitivity)")
        ax.set_title(f"{m['name']} (D={m['D']})")
        ax.set_xlim(-0.5, len(alpha) - 0.5)

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_2_sensitivity_distributions.png")
    plt.savefig(path)
    plt.close()
    print(f"  Fig 2 -> {path}")


# ============================================================
# Plot 2.4: kv_ratio vs k for each model
# ============================================================
def plot_kv_ratio_vs_k():
    n = len(MODELS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, m in enumerate(MODELS):
        ax = axes_flat[idx]
        configs = m["k_configs"]

        ks = []
        ratios = []
        deltas = []
        passes = []

        for key in sorted(configs.keys(), key=lambda x: configs[x]["k"]):
            c = configs[key]
            ks.append(c["k"])
            ratios.append(c["kv_ratio"])
            deltas.append(c["max_delta"])
            passes.append(c["pass_3pct"])

        colors = ["#4CAF50" if p else "#F44336" for p in passes]
        ax.scatter(
            ks, ratios, c=colors, s=80, zorder=5, edgecolors="black", linewidths=0.5
        )

        # Highlight k*
        kstar = m["k_star_3"]
        kstar_ratio = m["kv_ratio"]
        ax.scatter(
            [kstar],
            [kstar_ratio],
            s=200,
            marker="*",
            color="gold",
            zorder=6,
            edgecolors="black",
            linewidths=1,
        )

        # Reference lines
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, label="INT8-all")
        ax.axhline(y=1.0, color="gray", linestyle="-", alpha=0.3, label="Dense")

        ax.set_xlabel("k (protected layers)")
        ax.set_ylabel("kv_ratio")
        ax.set_title(f"{m['name']} (D={m['D']}, k*={kstar})")
        ax.set_ylim(0.25, 0.55)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_3_kv_ratio_vs_k.png")
    plt.savefig(path)
    plt.close()
    print(f"  Fig 3 -> {path}")


# ============================================================
# Plot 2.5: Empirical lower boundary vs D
# ============================================================
def plot_empirical_lower_boundary():
    fig, ax = plt.subplots(figsize=(7, 5))

    arch_style = {
        "Qwen2": ("o", "#2196F3"),
        "Mistral": ("s", "#FF5722"),
        "Llama": ("D", "#4CAF50"),
    }

    Ds = [m["D"] for m in MODELS]
    ratios = [m["kv_ratio"] for m in MODELS]
    names = [m["short"] for m in MODELS]
    colors = [arch_style.get(m["arch"], ("o", "#999"))[1] for m in MODELS]
    markers = [arch_style.get(m["arch"], ("o", "#999"))[0] for m in MODELS]

    for D, r, n, c, mk in zip(Ds, ratios, names, colors, markers):
        ax.scatter(
            D,
            r,
            s=120,
            color=c,
            marker=mk,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(n, (D, r), textcoords="offset points", xytext=(8, 5), fontsize=9)

    # INT4-all floor
    ax.axhline(
        y=0.28125,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="INT4-all floor (g32, hd=128)",
    )

    # Extrapolation band (clearly labeled)
    D_ext = np.array([24, 28, 32, 36, 40, 48])
    floor = 0.28125
    ax.fill_between(
        D_ext,
        floor,
        floor + 0.02,
        alpha=0.1,
        color="blue",
        label="Extrapolation (if k* stays O(1))",
    )

    ax.set_xlabel("Depth D (layers)")
    ax.set_ylabel("Best kv_ratio at eps=3%")
    ax.set_title("Empirical Lower Boundary vs Model Depth")
    ax.set_xlim(20, 50)
    ax.set_ylim(0.26, 0.35)
    ax.set_xticks([24, 28, 32, 36, 40, 48])
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_4_empirical_lower_boundary.png")
    plt.savefig(path)
    plt.close()
    print(f"  Fig 4 -> {path}")


# ============================================================
# Table 2.6: Capacity/throughput impact
# ============================================================
def write_capacity_table():
    header = "| Model | D | k*(3%) | kv_ratio | Capacity mult | Notes |"
    sep = "|" + "|".join(["---"] * 6) + "|"
    rows = [header, sep]
    for m in MODELS:
        cap = 1.0 / m["kv_ratio"]
        rows.append(
            f"| {m['name']} | {m['D']} | {m['k_star_3']} | "
            f"{m['kv_ratio']:.4f} | {cap:.2f}x | "
            f"Theoretical (no fused kernel) |"
        )
    path = os.path.join(TABLES, "table_2_capacity.md")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  Table 2 -> {path}")


# ============================================================
# Plot: Sink dominance contrast (supplementary)
# ============================================================
def plot_sink_dominance():
    fig, ax = plt.subplots(figsize=(7, 5))

    names = [m["short"] for m in MODELS]
    layer0 = [m["layer0_sens"] for m in MODELS]
    tail = [m["tail_frac"] * 100 for m in MODELS]

    x = np.arange(len(names))
    width = 0.35

    # Use log scale for layer0 sensitivity
    ax2 = ax.twinx()
    bars1 = ax.bar(
        x - width / 2,
        tail,
        width,
        color="#4CAF50",
        alpha=0.8,
        label="Tail fraction (%)",
    )
    bars2 = ax2.bar(
        x + width / 2,
        layer0,
        width,
        color="#F44336",
        alpha=0.8,
        label="Layer 0 sensitivity (%)",
    )
    ax2.set_yscale("log")

    ax.set_xlabel("Model")
    ax.set_ylabel("Tail fraction (%)", color="#4CAF50")
    ax2.set_ylabel("Layer 0 max|delta%| (log)", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Sink Dominance vs Uniform Robustness")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_5_sink_dominance.png")
    plt.savefig(path)
    plt.close()
    print(f"  Fig 5 -> {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("BPA v27: Generating paper-grade figures and tables")
    print("=" * 50)

    print("\nTables:")
    write_canonical_table()
    write_capacity_table()

    print("\nFigures:")
    plot_kstar_vs_D()
    plot_sensitivity_distributions()
    plot_kv_ratio_vs_k()
    plot_empirical_lower_boundary()
    plot_sink_dominance()

    print("\nDone. All outputs in artifacts/v27/")


if __name__ == "__main__":
    main()
