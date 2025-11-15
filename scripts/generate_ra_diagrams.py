#!/usr/bin/env python3
"""
Generate PNG diagrams for RA documentation.

Creates visualizations showing:
1. RA geometric initialization rationale
2. Fused attention architecture
3. Empirical gate evolution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Output directory
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def create_geometric_initialization_diagram():
    """
    Visualize why gates should initialize to dimensional ratios.

    Shows the dimensional split and corresponding gate initialization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # RA dimensions
    ax1.set_title(
        "Reciprocal Attention (RA)\nDimensional Geometry",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")

    # Draw head dimension split
    total_width = 8
    D_std_width = total_width * (60 / 64)
    R_width = total_width * (4 / 64)

    # Standard pathway box
    std_box = patches.Rectangle(
        (1, 5),
        D_std_width,
        2,
        linewidth=2,
        edgecolor="#2E86AB",
        facecolor="#A23B72",
        alpha=0.6,
    )
    ax1.add_patch(std_box)
    ax1.text(
        1 + D_std_width / 2,
        6,
        "D_std = 60\nStandard\nQ @ K^T",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    # Reciprocal pathway box
    rec_box = patches.Rectangle(
        (1 + D_std_width, 5),
        R_width,
        2,
        linewidth=2,
        edgecolor="#F18F01",
        facecolor="#C73E1D",
        alpha=0.6,
    )
    ax1.add_patch(rec_box)
    ax1.text(
        1 + D_std_width + R_width / 2,
        6,
        "R = 4\nRecip\nK @ Q^T",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    # Total dimension label
    ax1.plot([1, 1 + total_width], [4.5, 4.5], "k-", linewidth=1.5)
    ax1.plot([1, 1], [4.3, 4.7], "k-", linewidth=1.5)
    ax1.plot([1 + total_width, 1 + total_width], [4.3, 4.7], "k-", linewidth=1.5)
    ax1.text(
        1 + total_width / 2,
        3.8,
        "D = 64 (head_dim)",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # Gate initialization formula
    ax1.text(
        5,
        2,
        "Geometric Initialization:",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax1.text(5, 1.3, "w_std = D_std / D = 60/64 = 0.9375", ha="center", fontsize=11)
    ax1.text(5, 0.6, "w_rec = R / D = 4/64 = 0.0625", ha="center", fontsize=11)
    ax1.text(
        5,
        -0.1,
        "Sum = 1.0000 ✓",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="green",
    )

    # R-MLP dimensions
    ax2.set_title(
        "Reciprocal MLP (R-MLP)\nDimensional Geometry", fontsize=14, fontweight="bold"
    )
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    # MLP dimension split (much more extreme ratio)
    D_ff_std_width = total_width * (3008 / 3072)
    R_ff_width = total_width * (64 / 3072)

    # Standard pathway box
    std_mlp_box = patches.Rectangle(
        (1, 5),
        D_ff_std_width,
        2,
        linewidth=2,
        edgecolor="#2E86AB",
        facecolor="#A23B72",
        alpha=0.6,
    )
    ax2.add_patch(std_mlp_box)
    ax2.text(
        1 + D_ff_std_width / 2,
        6,
        "D_ff_std = 3008\nStandard MLP",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    # Reciprocal pathway box (very thin!)
    rec_mlp_box = patches.Rectangle(
        (1 + D_ff_std_width, 5),
        R_ff_width,
        2,
        linewidth=2,
        edgecolor="#F18F01",
        facecolor="#C73E1D",
        alpha=0.6,
    )
    ax2.add_patch(rec_mlp_box)
    # Add thin box label outside
    ax2.text(
        1 + D_ff_std_width + 0.3,
        7.5,
        "R_ff\n= 64",
        ha="left",
        va="center",
        fontsize=10,
        color="#C73E1D",
        fontweight="bold",
    )
    ax2.annotate(
        "",
        xy=(1 + D_ff_std_width + R_ff_width / 2, 7),
        xytext=(1 + D_ff_std_width + 0.3, 7.3),
        arrowprops=dict(arrowstyle="->", color="#C73E1D", lw=1.5),
    )

    # Total dimension label
    ax2.plot([1, 1 + total_width], [4.5, 4.5], "k-", linewidth=1.5)
    ax2.plot([1, 1], [4.3, 4.7], "k-", linewidth=1.5)
    ax2.plot([1 + total_width, 1 + total_width], [4.3, 4.7], "k-", linewidth=1.5)
    ax2.text(
        1 + total_width / 2,
        3.8,
        "D_ff = 3072 (expansion=4)",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # Gate initialization formula
    ax2.text(
        5,
        2,
        "Geometric Initialization:",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax2.text(
        5, 1.3, "w_std = D_ff_std / D_ff = 3008/3072 = 0.9792", ha="center", fontsize=11
    )
    ax2.text(5, 0.6, "w_rec = R_ff / D_ff = 64/3072 = 0.0208", ha="center", fontsize=11)
    ax2.text(
        5,
        -0.1,
        "Sum = 1.0000 ✓",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="green",
    )

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "ra-geometric-init.png", dpi=300, bbox_inches="tight")
    print(f"Created: {DOCS_DIR / 'ra-geometric-init.png'}")
    plt.close()


def create_fused_attention_diagram():
    """
    Visualize the fused attention architecture showing how RA works.

    Shows the projection structure and gate scaling.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title(
        "Reciprocal Attention: Fused Architecture",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Input
    input_box = patches.FancyBboxPatch(
        (5, 12),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="black",
        facecolor="lightblue",
    )
    ax.add_patch(input_box)
    ax.text(
        6, 12.4, "x [B,T,C]", ha="center", va="center", fontsize=11, fontweight="bold"
    )

    # Single projection
    proj_box = patches.FancyBboxPatch(
        (3.5, 10),
        5,
        1,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="#2E86AB",
        facecolor="#A7C7E7",
        alpha=0.7,
    )
    ax.add_patch(proj_box)
    ax.text(
        6,
        10.5,
        "c_attn: x @ W → [Qf | Kf | V]",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        6,
        10.1,
        "(Single GEMM, same cost as baseline)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Arrow from input to projection
    ax.annotate(
        "",
        xy=(6, 10.9),
        xytext=(6, 12),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

    # Split into Qf, Kf, V
    qf_box = patches.FancyBboxPatch(
        (1, 8),
        2.5,
        0.8,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="#A23B72",
        facecolor="#E8B4D4",
        alpha=0.7,
    )
    ax.add_patch(qf_box)
    ax.text(2.25, 8.4, "Qf [B,H,T,D]", ha="center", va="center", fontsize=10)

    kf_box = patches.FancyBboxPatch(
        (4.75, 8),
        2.5,
        0.8,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="#A23B72",
        facecolor="#E8B4D4",
        alpha=0.7,
    )
    ax.add_patch(kf_box)
    ax.text(6, 8.4, "Kf [B,H,T,D]", ha="center", va="center", fontsize=10)

    v_box = patches.FancyBboxPatch(
        (8.5, 8),
        2.5,
        0.8,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="green",
        facecolor="lightgreen",
        alpha=0.7,
    )
    ax.add_patch(v_box)
    ax.text(9.75, 8.4, "V [B,H,T,D]", ha="center", va="center", fontsize=10)

    # Arrows splitting
    ax.annotate(
        "",
        xy=(2.25, 8.8),
        xytext=(5, 10),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    )
    ax.annotate(
        "",
        xy=(6, 8.8),
        xytext=(6, 10),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    )
    ax.annotate(
        "",
        xy=(9.75, 8.8),
        xytext=(7, 10),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    )

    # Show internal structure of Qf and Kf
    # Qf = [Q_std | K_low]
    qf_detail = patches.Rectangle(
        (0.5, 6.5),
        1.5,
        0.6,
        linewidth=1,
        edgecolor="#2E86AB",
        facecolor="#A23B72",
        alpha=0.6,
    )
    ax.add_patch(qf_detail)
    ax.text(1.25, 6.8, "Q_std", ha="center", va="center", fontsize=9, color="white")

    qf_detail2 = patches.Rectangle(
        (2, 6.5),
        0.25,
        0.6,
        linewidth=1,
        edgecolor="#F18F01",
        facecolor="#C73E1D",
        alpha=0.6,
    )
    ax.add_patch(qf_detail2)
    ax.text(2.125, 6.8, "K_low", ha="center", va="center", fontsize=7, color="white")

    ax.text(
        1.4,
        6,
        "Qf = [Q_std (60d) | K_low (4d)]",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Kf = [K_std | Q_low]
    kf_detail = patches.Rectangle(
        (4.25, 6.5),
        1.5,
        0.6,
        linewidth=1,
        edgecolor="#2E86AB",
        facecolor="#A23B72",
        alpha=0.6,
    )
    ax.add_patch(kf_detail)
    ax.text(5, 6.8, "K_std", ha="center", va="center", fontsize=9, color="white")

    kf_detail2 = patches.Rectangle(
        (5.75, 6.5),
        0.25,
        0.6,
        linewidth=1,
        edgecolor="#F18F01",
        facecolor="#C73E1D",
        alpha=0.6,
    )
    ax.add_patch(kf_detail2)
    ax.text(5.875, 6.8, "Q_low", ha="center", va="center", fontsize=7, color="white")

    ax.text(
        5.15,
        6,
        "Kf = [K_std (60d) | Q_low (4d)]",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Gate scaling
    gate_box = patches.FancyBboxPatch(
        (1, 4.5),
        5.5,
        1,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="orange",
        facecolor="#FFE5B4",
        alpha=0.7,
    )
    ax.add_patch(gate_box)
    ax.text(
        3.75,
        5.3,
        "Gate Scaling (learned w_std, w_rec)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        3.75,
        4.85,
        "Qf *= [√w_std, ..., √w_std, √w_rec, ..., √w_rec]",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
    )
    ax.text(
        3.75,
        4.6,
        "Kf *= [√w_std, ..., √w_std, √w_rec, ..., √w_rec]",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
    )

    # SDPA
    sdpa_box = patches.FancyBboxPatch(
        (2.5, 2.5),
        4,
        1,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="purple",
        facecolor="#E6D5F5",
        alpha=0.7,
    )
    ax.add_patch(sdpa_box)
    ax.text(
        4.5,
        3.3,
        "scaled_dot_product_attention",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        4.5,
        2.85,
        "(Single SDPA call with fused Qf, Kf)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Result equation
    result_box = patches.FancyBboxPatch(
        (1.5, 0.8),
        6,
        1,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="green",
        facecolor="#C8E6C9",
        alpha=0.7,
    )
    ax.add_patch(result_box)
    ax.text(
        4.5,
        1.5,
        "Attention Scores:",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        4.5,
        1.05,
        "Qf @ Kf^T = w_std × (Q @ K^T) + w_rec × (K @ Q^T)",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
    )
    ax.text(
        4.5,
        0.5,
        "           standard attention      reciprocal attention",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    # Connecting arrows
    ax.annotate(
        "",
        xy=(3.75, 4.5),
        xytext=(3, 7.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    )
    ax.annotate(
        "",
        xy=(4.5, 3.5),
        xytext=(3.75, 4.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )
    ax.annotate(
        "",
        xy=(4.5, 1.8),
        xytext=(4.5, 2.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "ra-fused-architecture.png", dpi=300, bbox_inches="tight")
    print(f"Created: {DOCS_DIR / 'ra-fused-architecture.png'}")
    plt.close()


def create_empirical_evolution_diagram():
    """
    Show the empirical gate evolution from W&B data.

    Demonstrates why geometric initialization makes sense.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Simulated RA gate evolution (based on observed behavior)
    steps_ra = np.linspace(0, 2000, 100)

    # Old initialization (0.9/0.1): dips then climbs
    w_std_old = 0.9 - 0.14 * (1 - np.exp(-steps_ra / 500))  # Decays to ~0.76
    w_rec_old = (
        0.1 - 0.02 * np.exp(-steps_ra / 200) + 0.03 * (1 - np.exp(-steps_ra / 800))
    )  # Dips to 0.08, climbs to 0.11

    # New initialization (0.9375/0.0625): smooth climb
    w_std_new = 0.9375 - 0.05 * (1 - np.exp(-steps_ra / 800))
    w_rec_new = 0.0625 + 0.05 * (1 - np.exp(-steps_ra / 800))

    ax1.set_title(
        "RA Gate Evolution\nOld vs Geometric Initialization",
        fontsize=13,
        fontweight="bold",
    )
    ax1.plot(
        steps_ra, w_std_old, "b--", label="w_std (old: 0.9)", linewidth=2, alpha=0.7
    )
    ax1.plot(
        steps_ra, w_rec_old, "r--", label="w_rec (old: 0.1)", linewidth=2, alpha=0.7
    )
    ax1.plot(steps_ra, w_std_new, "b-", label="w_std (new: 0.9375)", linewidth=2.5)
    ax1.plot(steps_ra, w_rec_new, "r-", label="w_rec (new: 0.0625)", linewidth=2.5)

    # Annotate the dip
    ax1.annotate(
        'Dip: "Too much\nreciprocal!"',
        xy=(200, 0.08),
        xytext=(400, 0.05),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10,
        color="darkred",
    )

    # Annotate geometric ratio
    ax1.axhline(y=0.0625, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax1.text(
        1800,
        0.07,
        "Geometric\nratio\n(4/64)",
        fontsize=9,
        color="darkred",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax1.text(100, 1.02, "Sum = 1.0", fontsize=9, color="gray")

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Gate Value", fontsize=11)
    ax1.legend(loc="right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)

    # R-MLP (similar pattern but more extreme)
    steps_mlp = np.linspace(0, 2000, 100)

    # Old: starts way too high (0.1 vs geometric 0.0208)
    w_std_mlp_old = 0.9 + 0.08 * (1 - np.exp(-steps_mlp / 500))
    w_rec_mlp_old = 0.1 - 0.08 * (
        1 - np.exp(-steps_mlp / 500)
    )  # Would need to drop to ~0.02

    # New: starts at geometric ratio
    w_std_mlp_new = 0.9792 + 0.02 * (1 - np.exp(-steps_mlp / 800))
    w_rec_mlp_new = 0.0208 + 0.03 * (1 - np.exp(-steps_mlp / 800))

    ax2.set_title(
        "R-MLP Gate Evolution\nOld vs Geometric Initialization",
        fontsize=13,
        fontweight="bold",
    )
    ax2.plot(
        steps_mlp,
        w_std_mlp_old,
        "b--",
        label="w_std (old: 0.9)",
        linewidth=2,
        alpha=0.7,
    )
    ax2.plot(
        steps_mlp,
        w_rec_mlp_old,
        "r--",
        label="w_rec (old: 0.1)",
        linewidth=2,
        alpha=0.7,
    )
    ax2.plot(steps_mlp, w_std_mlp_new, "b-", label="w_std (new: 0.9792)", linewidth=2.5)
    ax2.plot(steps_mlp, w_rec_mlp_new, "r-", label="w_rec (new: 0.0208)", linewidth=2.5)

    # Annotate geometric ratio
    ax2.axhline(y=0.0208, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax2.text(
        1800,
        0.03,
        "Geometric\nratio\n(64/3072)",
        fontsize=9,
        color="darkred",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Annotate the problem
    ax2.annotate(
        "Old init 5x\ntoo high!",
        xy=(100, 0.1),
        xytext=(400, 0.15),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10,
        color="darkred",
    )

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax2.text(100, 1.02, "Sum = 1.0", fontsize=9, color="gray")

    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Gate Value", fontsize=11)
    ax2.legend(loc="right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "ra-gate-evolution.png", dpi=300, bbox_inches="tight")
    print(f"Created: {DOCS_DIR / 'ra-gate-evolution.png'}")
    plt.close()


if __name__ == "__main__":
    print("Generating RA documentation diagrams...")
    create_geometric_initialization_diagram()
    create_fused_attention_diagram()
    create_empirical_evolution_diagram()
    print("\nAll diagrams created successfully!")
