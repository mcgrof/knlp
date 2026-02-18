#!/usr/bin/env python
"""Generate figures for BPA v25 technical narrative."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "artifacts/v25/figures"
os.makedirs(OUTDIR, exist_ok=True)

# Data from v24 experiments
D_VALUES = [24, 28]
KSTAR_3PCT = [2, 2]
KSTAR_1PCT = [None, 3]
KD_3PCT = [2 / 24, 2 / 28]

# k-sweep data from v24 scoreboard
K_VALUES_05B = [0, 1, 2, 3, 4, 6, 8, 12]
DELTA_05B = [428.69, 3.05, 2.85, 2.36, 2.24, 2.33, 2.10, 1.42]
RATIO_05B = [0.2812, 0.2910, 0.3008, 0.3105, 0.3203, 0.3398, 0.3594, 0.3984]
PASS3_05B = [False, False, True, True, True, True, True, True]

K_VALUES_15B = [0, 1, 2, 3, 4, 6, 8, 12]
DELTA_15B = [67928.54, 3.85, 1.05, 0.73, 0.42, 0.48, 0.47, 0.30]
RATIO_15B = [0.2812, 0.2893, 0.2974, 0.3055, 0.3136, 0.3298, 0.3460, 0.3783]
PASS3_15B = [False, False, True, True, True, True, True, True]

# Oracle sensitivity data (from v24 Phase 3)
SENS_05B = {
    0: 23.48,
    1: 0.07,
    2: 1.34,
    3: 0.32,
    4: 0.50,
    5: 0.50,
    6: 0.43,
    7: 0.48,
    8: 0.49,
    9: 0.27,
    10: 0.46,
    11: 1.30,
    12: 0.44,
    13: 0.25,
    14: 0.41,
    15: 0.43,
    16: 0.77,
    17: 0.54,
    18: 0.37,
    19: 0.44,
    20: 0.47,
    21: 0.77,
    22: 0.38,
    23: 0.47,
}
SENS_15B = {
    0: 824.55,
    1: 0.79,
    2: 0.16,
    3: 0.05,
    4: 0.05,
    5: 0.10,
    6: 0.09,
    7: 0.10,
    8: 0.16,
    9: 0.12,
    10: 0.05,
    11: 0.08,
    12: 0.16,
    13: 0.06,
    14: 0.22,
    15: 3.20,
    16: 0.12,
    17: 0.04,
    18: 0.04,
    19: 0.43,
    20: 0.07,
    21: 0.13,
    22: 0.09,
    23: 0.20,
    24: 0.08,
    25: 0.23,
    26: 0.19,
    27: 0.07,
}


def fig_k_star_vs_D():
    """k* vs model depth D."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        D_VALUES,
        KSTAR_3PCT,
        "o-",
        color="#2563eb",
        markersize=10,
        linewidth=2,
        label="k*(3%)",
    )
    # Extrapolation
    ax.plot(
        [32, 56, 80],
        [2, 2, 2],
        "s--",
        color="#2563eb",
        alpha=0.4,
        markersize=8,
        label="Predicted (O(1))",
    )
    ax.axhline(y=2, color="#94a3b8", linestyle=":", alpha=0.5)
    ax.set_xlabel("Model Depth D (layers)", fontsize=12)
    ax.set_ylabel("k* (min INT8 layers for PASS@3%)", fontsize=12)
    ax.set_title("k* vs Model Depth: O(1) Evidence", fontsize=13)
    ax.set_xlim(20, 85)
    ax.set_ylim(0, 6)
    ax.legend(fontsize=10)
    ax.annotate(
        "0.5B\n(D=24)",
        (24, 2),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
    )
    ax.annotate(
        "1.5B\n(D=28)",
        (28, 2),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "k_star_vs_D.png"), dpi=200)
    plt.close(fig)
    print("  k_star_vs_D.png")


def fig_k_over_D_vs_D():
    """k*/D vs D."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        D_VALUES,
        KD_3PCT,
        "o-",
        color="#dc2626",
        markersize=10,
        linewidth=2,
        label="k*/D (observed)",
    )
    # Extrapolation
    ext_D = [32, 56, 80]
    ext_kd = [2 / d for d in ext_D]
    ax.plot(
        ext_D,
        ext_kd,
        "s--",
        color="#dc2626",
        alpha=0.4,
        markersize=8,
        label="Predicted (k*=2)",
    )
    ax.set_xlabel("Model Depth D (layers)", fontsize=12)
    ax.set_ylabel("k*/D (protected fraction)", fontsize=12)
    ax.set_title("Protected Fraction Decreases with Scale", fontsize=13)
    ax.set_xlim(20, 85)
    ax.set_ylim(0, 0.12)
    ax.legend(fontsize=10)
    for d, kd in zip(D_VALUES, KD_3PCT):
        ax.annotate(
            f"{kd:.3f}", (d, kd), textcoords="offset points", xytext=(8, 5), fontsize=9
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "k_over_D_vs_D.png"), dpi=200)
    plt.close(fig)
    print("  k_over_D_vs_D.png")


def fig_oracle_sensitivity(model_name, sens_data, D, outname):
    """Per-layer INT4 sensitivity bar chart."""
    layers = sorted(sens_data.keys())
    vals = [sens_data[l] for l in layers]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 3]}
    )

    # Left: full scale (shows layer 0)
    ax1 = axes[0]
    colors1 = [
        "#dc2626" if v > 3.0 else "#f59e0b" if v > 1.0 else "#2563eb" for v in vals
    ]
    ax1.barh(layers, vals, color=colors1, height=0.7)
    ax1.set_ylabel("Layer", fontsize=11)
    ax1.set_xlabel("max |delta| (%)", fontsize=11)
    ax1.set_title(f"{model_name} (D={D})\nFull Scale", fontsize=11)
    ax1.invert_yaxis()

    # Right: zoomed (exclude layer 0)
    ax2 = axes[1]
    zoomed_layers = [l for l in layers if sens_data[l] < 10]
    zoomed_vals = [sens_data[l] for l in zoomed_layers]
    colors2 = [
        "#dc2626" if v > 3.0 else "#f59e0b" if v > 1.0 else "#2563eb"
        for v in zoomed_vals
    ]
    ax2.barh(zoomed_layers, zoomed_vals, color=colors2, height=0.7)
    ax2.axvline(x=3.0, color="#dc2626", linestyle="--", alpha=0.5, label="eps=3%")
    ax2.axvline(x=1.0, color="#f59e0b", linestyle="--", alpha=0.5, label="eps=1%")
    ax2.set_xlabel("max |delta| (%)", fontsize=11)
    ax2.set_title("Zoomed (excl. layer 0)", fontsize=11)
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)

    fig.suptitle(f"Oracle INT4 Sensitivity: {model_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {outname}")


def fig_kv_ratio_vs_k():
    """kv_ratio vs k for both models."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # 0.5B
    colors_05b = ["#22c55e" if p else "#ef4444" for p in PASS3_05B]
    ax.scatter(
        K_VALUES_05B,
        RATIO_05B,
        c=colors_05b,
        s=80,
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.plot(
        K_VALUES_05B,
        RATIO_05B,
        "-",
        color="#2563eb",
        alpha=0.5,
        linewidth=1.5,
        label="0.5B (D=24)",
    )

    # 1.5B (skip k=0 catastrophic for cleaner plot)
    k15_plot = K_VALUES_15B[1:]
    r15_plot = RATIO_15B[1:]
    p15_plot = PASS3_15B[1:]
    colors_15b = ["#22c55e" if p else "#ef4444" for p in p15_plot]
    ax.scatter(
        k15_plot,
        r15_plot,
        c=colors_15b,
        s=80,
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
        marker="D",
    )
    ax.plot(
        k15_plot,
        r15_plot,
        "-",
        color="#dc2626",
        alpha=0.5,
        linewidth=1.5,
        label="1.5B (D=28)",
    )

    # Annotations
    ax.axhline(y=0.30, color="#94a3b8", linestyle=":", alpha=0.5)
    ax.annotate(
        "0.30 target",
        (0, 0.30),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        color="#64748b",
    )

    # Legend for PASS/FAIL
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#22c55e",
            markersize=8,
            label="PASS@3%",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ef4444",
            markersize=8,
            label="FAIL@3%",
        ),
        Line2D([0], [0], color="#2563eb", linewidth=1.5, label="0.5B"),
        Line2D([0], [0], color="#dc2626", linewidth=1.5, label="1.5B"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    ax.set_xlabel("k (INT8 protected layers)", fontsize=12)
    ax.set_ylabel("kv_ratio (incl. metadata)", fontsize=12)
    ax.set_title("kv_ratio vs k: More Protection = More Bytes", fontsize=13)
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(0.27, 0.42)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "kv_ratio_vs_k.png"), dpi=200)
    plt.close(fig)
    print("  kv_ratio_vs_k.png")


def fig_compute_bound():
    """Decision graph: why W7900 shows capacity not latency."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: latency is flat
    methods = ["dense", "INT8", "g32_k4", "amort"]
    ms_tok = [21.85, 21.22, 21.22, 22.92]
    colors = ["#64748b", "#2563eb", "#22c55e", "#f59e0b"]
    ax1 = axes[0]
    bars = ax1.bar(methods, ms_tok, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("ms/token (L=8K, batch=1)", fontsize=11)
    ax1.set_title("Latency: Flat (Compute-Bound)", fontsize=12)
    ax1.set_ylim(0, 30)
    for bar, val in zip(bars, ms_tok):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}",
            ha="center",
            fontsize=9,
        )

    # Right: capacity gain
    L_vals = ["8K", "16K", "32K"]
    dense_seqs = [494, 247, 123]
    amort_seqs = [1608, 804, 402]
    x = np.arange(len(L_vals))
    width = 0.35
    ax2 = axes[1]
    ax2.bar(
        x - width / 2,
        dense_seqs,
        width,
        label="Dense",
        color="#64748b",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.bar(
        x + width / 2,
        amort_seqs,
        width,
        label="INT4 (amort)",
        color="#22c55e",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(L_vals)
    ax2.set_xlabel("Context Length L", fontsize=11)
    ax2.set_ylabel("Max Concurrent Sequences", fontsize=11)
    ax2.set_title("Capacity: 3.26x Gain", fontsize=12)
    ax2.legend(fontsize=10)
    for i, (d, a) in enumerate(zip(dense_seqs, amort_seqs)):
        ax2.text(
            i + width / 2,
            a + 15,
            f"{a/d:.1f}x",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#16a34a",
        )

    fig.suptitle(
        "W7900: Compute-Bound = Capacity Win, Not Latency Win", fontsize=13, y=1.02
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTDIR, "compute_bound_decision.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    print("  compute_bound_decision.png")


if __name__ == "__main__":
    print("Generating v25 figures...")
    fig_k_star_vs_D()
    fig_k_over_D_vs_D()
    fig_oracle_sensitivity("Qwen2.5-0.5B", SENS_05B, 24, "oracle_sensitivity_0.5B.png")
    fig_oracle_sensitivity("Qwen2.5-1.5B", SENS_15B, 28, "oracle_sensitivity_1.5B.png")
    fig_kv_ratio_vs_k()
    fig_compute_bound()
    print(f"\nAll figures saved to {OUTDIR}/")
