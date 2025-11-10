#!/usr/bin/env python3
"""
Generate visual diagrams for Unified RA documentation.

Creates SVG/PNG images to replace ASCII art in docs/ra.md.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.size"] = 10


def create_performance_comparison():
    """Create performance comparison bar charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Forward time comparison
    versions = ["V0\nBaseline", "V1\nUnified RA"]
    times = [1555.23, 1522.17]
    colors = ["#d73027", "#4575b4"]

    bars1 = ax1.bar(
        versions, times, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax1.set_ylabel("Forward Time (ms/iteration)", fontsize=12, fontweight="bold")
    ax1.set_title("Forward Time Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim([1450, 1600])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{time:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add speedup annotation
    ax1.annotate(
        "2.17% FASTER!",
        xy=(1, times[1]),
        xytext=(0.5, times[1] - 20),
        arrowprops=dict(arrowstyle="->", lw=2, color="green"),
        fontsize=11,
        fontweight="bold",
        color="green",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    # Memory comparison
    memory = [3176.56, 3175.87]
    bars2 = ax2.bar(
        versions, memory, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax2.set_ylabel("Memory Usage (MB)", fontsize=12, fontweight="bold")
    ax2.set_title("Memory Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylim([3170, 3180])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for bar, mem in zip(bars2, memory):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{mem:.2f} MB",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add identical annotation
    ax2.text(
        0.5,
        3177.5,
        "IDENTICAL",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="darkblue",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig("docs/images/performance_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Created: docs/images/performance_comparison.png")
    plt.close()


def create_evolution_timeline():
    """Create evolution timeline showing RA v2 → Unified RA."""
    fig, ax = plt.subplots(figsize=(14, 6))

    versions = [
        "RA v2\n(2 GEMMs)",
        "RA v3\n(Fused)",
        "RA v4\n(Zero-cat)",
        "Unified RA\n(Folded)",
    ]
    times = [2000, 2230, 1960, 1522]
    slowdowns = [66, 85, 48, -2]  # Percent slower than baseline (1522 is faster)
    colors = ["#d73027", "#fc8d59", "#fee090", "#4575b4"]
    status = ["❌", "❌", "❌", "✅"]

    bars = ax.bar(
        versions, times, color=colors, alpha=0.8, edgecolor="black", linewidth=2
    )
    ax.set_ylabel("Forward Time (ms/iteration)", fontsize=12, fontweight="bold")
    ax.set_title("RA Evolution: From Complex to Simple", fontsize=14, fontweight="bold")
    ax.axhline(
        y=1555,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Baseline (1555 ms)",
        alpha=0.7,
    )
    ax.set_ylim([1400, 2300])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)

    # Add value labels and slowdown percentages
    for i, (bar, time, slowdown, stat) in enumerate(
        zip(bars, times, slowdowns, status)
    ):
        height = bar.get_height()
        # Time value
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 30,
            f"{time} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
        # Slowdown percentage
        sign = "+" if slowdown > 0 else ""
        color = "red" if slowdown > 0 else "green"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            f"{sign}{slowdown}%",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=13,
            color=color,
        )
        # Status
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            1450,
            stat,
            ha="center",
            va="center",
            fontsize=20,
        )

    # Add insight annotation
    ax.text(
        3,
        2100,
        "Key insight:\nPre-fold layout,\nsingle SDPA → WIN",
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("docs/images/evolution_timeline.png", dpi=150, bbox_inches="tight")
    print("✓ Created: docs/images/evolution_timeline.png")
    plt.close()


def create_folded_layout_diagram():
    """Create folded Q/K layout diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5,
        9.5,
        "Folded Q/K Layout: The Core Insight",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    # Head dimension split
    ax.text(
        5,
        8.8,
        "Head Dimension Split (D = 64)",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # D_std box
    rect1 = FancyBboxPatch(
        (1, 8.0),
        6,
        0.4,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#4575b4",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect1)
    ax.text(
        4,
        8.2,
        "D_std = 60",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="white",
    )

    # R box
    rect2 = FancyBboxPatch(
        (7, 8.0),
        1.5,
        0.4,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#d73027",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect2)
    ax.text(
        7.75,
        8.2,
        "R = 4",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="white",
    )

    # Qf layout
    ax.text(5, 7.3, "Qf Layout (Per-Head)", ha="center", fontsize=12, fontweight="bold")

    rect3 = FancyBboxPatch(
        (1.5, 6.5),
        3,
        0.5,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#4575b4",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect3)
    ax.text(
        3,
        6.75,
        "Q_std (60)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="white",
    )

    rect4 = FancyBboxPatch(
        (4.5, 6.5),
        1.5,
        0.5,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#d73027",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect4)
    ax.text(
        5.25,
        6.75,
        "K_low (4)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="white",
    )

    ax.text(0.8, 6.75, "Qf =", ha="right", va="center", fontweight="bold", fontsize=11)
    ax.text(
        6.5,
        6.75,
        "(Standard | Reciprocal)",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Kf layout
    ax.text(5, 5.8, "Kf Layout (Per-Head)", ha="center", fontsize=12, fontweight="bold")

    rect5 = FancyBboxPatch(
        (1.5, 5.0),
        3,
        0.5,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#4575b4",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect5)
    ax.text(
        3,
        5.25,
        "K_std (60)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="white",
    )

    rect6 = FancyBboxPatch(
        (4.5, 5.0),
        1.5,
        0.5,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#d73027",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect6)
    ax.text(
        5.25,
        5.25,
        "Q_low (4)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="white",
    )

    ax.text(0.8, 5.25, "Kf =", ha="right", va="center", fontweight="bold", fontsize=11)
    ax.text(
        6.5,
        5.25,
        "(Standard | Reciprocal)",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Matmul visualization
    ax.text(
        5,
        4.2,
        "When we compute: Qf @ Kf^T",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # Matrix multiplication breakdown
    components = [
        ("Q_std @ K_std^T", "Standard attention", "#4575b4"),
        ("Q_std @ Q_low^T", "Cross-term (reciprocal)", "#91bfdb"),
        ("K_low @ K_std^T", "Cross-term (reciprocal)", "#91bfdb"),
        ("K_low @ Q_low^T", "Low-rank reciprocal", "#d73027"),
    ]

    y_start = 3.3
    for i, (term, desc, color) in enumerate(components):
        y = y_start - i * 0.5
        ax.text(2.5, y, term, ha="left", fontsize=10, fontweight="bold", color=color)
        ax.text(5.5, y, f"← {desc}", ha="left", fontsize=9, style="italic", color=color)

    # Key insight box
    insight_box = FancyBboxPatch(
        (2, 0.5),
        6,
        0.8,
        boxstyle="round,pad=0.1",
        edgecolor="green",
        facecolor="lightgreen",
        linewidth=3,
        alpha=0.8,
    )
    ax.add_patch(insight_box)
    ax.text(
        5,
        0.9,
        "All in ONE matmul!",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=13,
        color="darkgreen",
    )
    ax.text(
        5,
        0.6,
        "No routing, no copies, no concatenations",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="darkgreen",
    )

    plt.tight_layout()
    plt.savefig("docs/images/folded_layout.png", dpi=150, bbox_inches="tight")
    print("✓ Created: docs/images/folded_layout.png")
    plt.close()


def create_rwr_concept_diagram():
    """Create One-Step RWR concept diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5,
        9.5,
        "One-Step RWR (Self-Restart)",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    # Standard SDPA path
    ax.text(2.5, 8.5, "Standard SDPA", ha="center", fontsize=12, fontweight="bold")
    rect1 = FancyBboxPatch(
        (1.2, 7.0),
        2.6,
        1.2,
        boxstyle="round,pad=0.1",
        edgecolor="black",
        facecolor="#fee090",
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(rect1)
    ax.text(
        2.5,
        7.6,
        "SDPA(Q,K,V)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )

    arrow1 = FancyArrowPatch(
        (2.5, 7.0), (2.5, 6.0), arrowstyle="->", lw=3, color="black", mutation_scale=30
    )
    ax.add_patch(arrow1)
    ax.text(2.5, 5.7, "Output", ha="center", fontweight="bold", fontsize=11)

    # One-Step RWR path
    ax.text(7.5, 8.5, "One-Step RWR", ha="center", fontsize=12, fontweight="bold")
    rect2 = FancyBboxPatch(
        (5.7, 7.0),
        3.6,
        1.2,
        boxstyle="round,pad=0.1",
        edgecolor="green",
        facecolor="lightgreen",
        linewidth=3,
        alpha=0.8,
    )
    ax.add_patch(rect2)
    ax.text(
        7.5,
        7.8,
        "(1-α)·SDPA(Q,K,V)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        7.5, 7.35, "+ α·V", ha="center", va="center", fontweight="bold", fontsize=10
    )
    ax.text(8.8, 7.8, "← Attention", ha="left", va="center", fontsize=8, style="italic")
    ax.text(8.3, 7.35, "← Identity", ha="left", va="center", fontsize=8, style="italic")

    arrow2 = FancyArrowPatch(
        (7.5, 7.0), (7.5, 6.0), arrowstyle="->", lw=3, color="green", mutation_scale=30
    )
    ax.add_patch(arrow2)
    ax.text(
        7.5,
        5.7,
        "Stabilized Output",
        ha="center",
        fontweight="bold",
        fontsize=11,
        color="green",
    )

    # Alpha parameter info
    ax.text(
        5,
        4.8,
        "α = 0.05 (per-head, learnable, clamped [0, 0.5])",
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Benefits section
    ax.text(5, 4.0, "Benefits:", ha="center", fontsize=12, fontweight="bold")

    benefits = [
        "✓ Stabilizes training (prevents runaway hubs)",
        "✓ Improves long-tail connectivity (diffusion)",
        "✓ Zero overhead (single element-wise mix)",
        "✓ Learnable per-head (adapts to data)",
    ]

    y_start = 3.3
    for i, benefit in enumerate(benefits):
        y = y_start - i * 0.5
        ax.text(5, y, benefit, ha="center", fontsize=10, color="darkgreen")

    # Cost comparison
    ax.text(2.5, 0.8, "Full RWR (T=4):", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        2.5, 0.4, "4× cost", ha="center", fontsize=10, color="red", fontweight="bold"
    )

    ax.text(5, 0.6, "vs", ha="center", fontsize=12, fontweight="bold")

    ax.text(7.5, 0.8, "One-Step RWR:", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        7.5, 0.4, "~0% cost", ha="center", fontsize=10, color="green", fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("docs/images/rwr_concept.png", dpi=150, bbox_inches="tight")
    print("✓ Created: docs/images/rwr_concept.png")
    plt.close()


def create_forward_pass_flow():
    """Create forward pass flow diagram."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(
        5, 11.5, "Unified RA Forward Pass", ha="center", fontsize=16, fontweight="bold"
    )

    # Input
    box_y = 10.5
    rect = FancyBboxPatch(
        (3.5, box_y),
        3,
        0.5,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#4575b4",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect)
    ax.text(
        5,
        box_y + 0.25,
        "INPUT x [B, T, C]",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="white",
    )

    # Arrow
    arrow = FancyArrowPatch(
        (5, box_y),
        (5, box_y - 0.5),
        arrowstyle="->",
        lw=2,
        color="black",
        mutation_scale=20,
    )
    ax.add_patch(arrow)

    # c_attn
    box_y = 9.3
    rect = FancyBboxPatch(
        (3.2, box_y),
        3.6,
        0.7,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#91bfdb",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect)
    ax.text(
        5,
        box_y + 0.5,
        "c_attn (3C → 3C)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )
    ax.text(
        5,
        box_y + 0.2,
        "Single GEMM",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Arrow splits into 3
    arrow = FancyArrowPatch(
        (5, box_y),
        (5, box_y - 0.5),
        arrowstyle="->",
        lw=2,
        color="black",
        mutation_scale=20,
    )
    ax.add_patch(arrow)

    # Split into Qf, Kf, V
    box_y = 7.8
    positions = [2, 5, 8]
    labels = ["Qf [B,T,C]", "Kf [B,T,C]", "V [B,T,C]"]
    colors = ["#d73027", "#fee090", "#4575b4"]

    for pos, label, color in zip(positions, labels, colors):
        rect = FancyBboxPatch(
            (pos - 0.8, box_y),
            1.6,
            0.5,
            boxstyle="round,pad=0.05",
            edgecolor="black",
            facecolor=color,
            linewidth=2,
            alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            pos,
            box_y + 0.25,
            label,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

        # Arrows from c_attn to splits
        arrow = FancyArrowPatch(
            (5, 8.3),
            (pos, box_y + 0.5),
            arrowstyle="->",
            lw=1.5,
            color="black",
            mutation_scale=15,
        )
        ax.add_patch(arrow)

    # view + transpose
    ax.text(5, 7.1, "view + transpose", ha="center", fontsize=9, style="italic")

    # Arrow down
    arrow = FancyArrowPatch(
        (5, 7.0), (5, 6.5), arrowstyle="->", lw=2, color="black", mutation_scale=20
    )
    ax.add_patch(arrow)

    # Reshaped tensors
    box_y = 5.8
    for pos, label in zip(positions, ["[B,H,T,D]", "[B,H,T,D]", "[B,H,T,D]"]):
        ax.text(
            pos,
            box_y,
            label,
            ha="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1,
            ),
        )

    # Arrows converging to SDPA
    for pos in positions:
        arrow = FancyArrowPatch(
            (pos, 5.6),
            (5, 5.0),
            arrowstyle="->",
            lw=1.5,
            color="black",
            mutation_scale=15,
        )
        ax.add_patch(arrow)

    # SDPA
    box_y = 4.0
    rect = FancyBboxPatch(
        (3.2, box_y),
        3.6,
        0.9,
        boxstyle="round,pad=0.05",
        edgecolor="green",
        facecolor="lightgreen",
        linewidth=3,
        alpha=0.8,
    )
    ax.add_patch(rect)
    ax.text(
        5,
        box_y + 0.6,
        "SDPA (causal)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
    )
    ax.text(
        5,
        box_y + 0.2,
        "Single Flash Attention",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Arrow
    arrow = FancyArrowPatch(
        (5, box_y),
        (5, box_y - 0.7),
        arrowstyle="->",
        lw=2,
        color="black",
        mutation_scale=20,
    )
    ax.add_patch(arrow)

    # Output shape
    ax.text(
        5,
        3.0,
        "[B, H, T, D]",
        ha="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1
        ),
    )

    # transpose + reshape
    arrow = FancyArrowPatch(
        (5, 2.8), (5, 2.3), arrowstyle="->", lw=2, color="black", mutation_scale=20
    )
    ax.add_patch(arrow)
    ax.text(5, 2.55, "transpose + reshape", ha="center", fontsize=9, style="italic")

    # Reshaped output
    ax.text(
        5,
        2.0,
        "[B, T, C]",
        ha="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1
        ),
    )

    # c_proj
    arrow = FancyArrowPatch(
        (5, 1.8), (5, 1.3), arrowstyle="->", lw=2, color="black", mutation_scale=20
    )
    ax.add_patch(arrow)

    box_y = 0.5
    rect = FancyBboxPatch(
        (3.5, box_y),
        3,
        0.7,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#91bfdb",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(rect)
    ax.text(
        5,
        box_y + 0.5,
        "c_proj",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )
    ax.text(
        5,
        box_y + 0.2,
        "Output GEMM",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Statistics box
    stats_box = FancyBboxPatch(
        (0.3, 8.5),
        2.2,
        1.2,
        boxstyle="round,pad=0.1",
        edgecolor="blue",
        facecolor="lightblue",
        linewidth=2,
        alpha=0.7,
    )
    ax.add_patch(stats_box)
    ax.text(1.4, 9.5, "Statistics:", ha="center", fontweight="bold", fontsize=10)
    ax.text(1.4, 9.1, "Allocations: 2", ha="center", fontsize=8)
    ax.text(1.4, 8.8, "SDPA calls: 1", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("docs/images/forward_pass.png", dpi=150, bbox_inches="tight")
    print("✓ Created: docs/images/forward_pass.png")
    plt.close()


if __name__ == "__main__":
    print("\nGenerating visual diagrams for docs/ra.md...")
    print("=" * 60)

    create_performance_comparison()
    create_evolution_timeline()
    create_folded_layout_diagram()
    create_rwr_concept_diagram()
    create_forward_pass_flow()

    print("=" * 60)
    print("✓ All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - docs/images/performance_comparison.png")
    print("  - docs/images/evolution_timeline.png")
    print("  - docs/images/folded_layout.png")
    print("  - docs/images/rwr_concept.png")
    print("  - docs/images/forward_pass.png")
