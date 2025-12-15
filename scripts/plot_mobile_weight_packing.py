#!/usr/bin/env python3
"""
Generate publication-quality visualizations for mobile weight packing documentation.

Creates 5 figures explaining FIM-guided quantization:
1. FIM computation pipeline (g² = squared gradients)
2. GGUF K-Quantization types table
3. Tensor/layer sensitivity ranking
4. FIM score → quantization decision tree
5. Uniform vs FIM-guided comparison

Usage:
    python scripts/plot_mobile_weight_packing.py
    # Outputs to docs/mobile-weight-packing/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

# Style configuration - dark theme matching inspiration images
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#f0883e"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#a371f7"
ACCENT_YELLOW = "#d29922"
ACCENT_CYAN = "#39c5cf"

# Tier colors
CRITICAL_COLOR = "#f85149"
HIGH_COLOR = "#f0883e"
MEDIUM_COLOR = "#d29922"
LOW_COLOR = "#3fb950"

plt.rcParams.update(
    {
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_BG,
        "axes.edgecolor": BORDER_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": BORDER_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
    }
)


def create_output_dir():
    """Create output directory for images."""
    output_dir = "docs/mobile-weight-packing"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def draw_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    color=ACCENT_BLUE,
    fontsize=10,
    text_color=TEXT_COLOR,
    alpha=0.9,
):
    """Draw a styled box with text."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
        wrap=True,
    )


def draw_arrow(ax, start, end, color=TEXT_COLOR):
    """Draw an arrow between two points."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->", color=color, lw=2, connectionstyle="arc3,rad=0"
        ),
    )


def fig1_fim_pipeline(output_dir):
    """
    Figure 1: FIM Computation Pipeline
    Shows: Load Data → Forward → Backward → Square Gradients → Accumulate
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        7,
        7.5,
        "FIM Computation Pipeline for Quantization",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=ACCENT_CYAN,
    )

    # Subtitle with formula
    ax.text(
        7,
        6.8,
        r"FIM$_{diag}$($\theta$) = E[($\partial$L/$\partial\theta$)$^2$] = Expected Squared Gradient",
        ha="center",
        va="center",
        fontsize=12,
        color=TEXT_COLOR,
        style="italic",
    )

    # Pipeline boxes - row 1
    stages = [
        ("1. Load\nCalibration Data", 2, 5.2, ACCENT_PURPLE),
        ("2. Forward Pass", 5, 5.2, ACCENT_BLUE),
        ("3. Backward Pass", 8, 5.2, ACCENT_BLUE),
        ("4. Square Gradients", 11, 5.2, ACCENT_GREEN),
    ]

    for text, x, y, color in stages:
        draw_box(ax, x, y, 2.2, 1.0, text, color=color, fontsize=9)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 1.1
        x2 = stages[i + 1][1] - 1.1
        draw_arrow(ax, (x1, 5.2), (x2, 5.2))

    # Detail boxes below
    details = [
        (
            "sequences = tokenizer(text[:1k])\nN samples, seq_length tokens each",
            2,
            4.0,
            PANEL_BG,
        ),
        ("loss = model(x, labels=x)\nCompute cross-entropy loss", 5, 4.0, PANEL_BG),
        ("loss.backward()\nCompute ∂L/∂θ for all params", 8, 4.0, PANEL_BG),
        ("g² = (param.grad)**2\nElement-wise squaring", 11, 4.0, PANEL_BG),
    ]

    for text, x, y, color in details:
        box = FancyBboxPatch(
            (x - 1.3, y - 0.6),
            2.6,
            1.2,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color,
            edgecolor=BORDER_COLOR,
            alpha=0.8,
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
            family="monospace",
        )

    # Accumulation step (row 2)
    ax.text(
        7,
        2.5,
        "5. Accumulate Over Batches",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=ACCENT_ORANGE,
    )

    # Accumulation formula box
    accum_box = FancyBboxPatch(
        (3.5, 1.4),
        7,
        1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=PANEL_BG,
        edgecolor=ACCENT_ORANGE,
        alpha=0.9,
        linewidth=2,
    )
    ax.add_patch(accum_box)

    accum_text = """for batch in calibration_data:
    grad_sq_sum[name] += (param.grad ** 2).sum()
fim_score[name] = grad_sq_sum[name] / num_batches"""

    ax.text(
        7,
        2.0,
        accum_text,
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_COLOR,
        family="monospace",
    )

    # Final result box
    result_box = FancyBboxPatch(
        (4.5, -0.1),
        5,
        0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=DARK_BG,
        edgecolor=ACCENT_CYAN,
        alpha=0.9,
        linewidth=3,
    )
    ax.add_patch(result_box)

    ax.text(
        7,
        0.3,
        r"FIM$_{diag}$($\theta$) = E[($\partial$L/$\partial\theta$)$^2$]",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=ACCENT_CYAN,
    )

    # Arrow from accumulation to result
    draw_arrow(ax, (7, 1.4), (7, 0.7), color=ACCENT_CYAN)

    # Insight note
    ax.text(
        7,
        -0.5,
        "Higher FIM score → more sensitive to quantization → use higher precision",
        ha="center",
        va="center",
        fontsize=10,
        color=ACCENT_GREEN,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig1_fim_pipeline.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Created {output_dir}/fig1_fim_pipeline.png")


def fig2_gguf_quant_types(output_dir):
    """
    Figure 2: GGUF K-Quantization Types
    Table showing type, bits/weight, PPL delta, and mixed precision strategy.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        6,
        9.5,
        "GGUF K-Quantization Types",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=ACCENT_CYAN,
    )
    ax.text(
        6,
        9.0,
        "Mixed-precision quantization with K-means super-blocks (256 values)",
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_COLOR,
        style="italic",
    )

    # Table header
    headers = ["Type", "Bits/Weight", "PPL Δ", "Mixed Precision Strategy"]
    header_x = [1.5, 4, 6, 9]
    header_colors = [ACCENT_PURPLE, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN]

    for i, (header, x, color) in enumerate(zip(headers, header_x, header_colors)):
        ax.text(
            x,
            8.2,
            header,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    # Horizontal line
    ax.plot([0.5, 11.5], [7.9, 7.9], color=BORDER_COLOR, lw=1)

    # Data rows
    rows = [
        (
            "Q2_K",
            "~2.56",
            "+0.87",
            "Extreme compression, not recommended",
            CRITICAL_COLOR,
        ),
        ("Q3_K_S", "~2.75", "+0.55", "Uniform Q3_K all tensors", HIGH_COLOR),
        ("Q3_K_M", "~3.07", "+0.25", "Q4_K for attn.wv/wo, ffn.w2", MEDIUM_COLOR),
        ("Q3_K_L", "~3.35", "+0.18", "Q5_K for critical tensors", MEDIUM_COLOR),
        ("Q4_K_S", "~3.56", "+0.11", "Smaller Q4_K variant", LOW_COLOR),
        ("Q4_K_M", "~3.8", "+0.05", "Q6_K for half critical tensors ★", LOW_COLOR),
        ("Q5_K_S", "~4.03", "+0.02", "Q6_K for half critical tensors", LOW_COLOR),
        ("Q5_K_M", "~4.45", "+0.01", "Q6_K for half critical tensors", LOW_COLOR),
        ("Q6_K", "~5.5", "+0.01", "6-bit with F16 scales, high quality", LOW_COLOR),
        ("Q8_0", "~8.5", "~0", "Near-lossless reference", LOW_COLOR),
    ]

    for i, (qtype, bits, ppl, strategy, color) in enumerate(rows):
        y = 7.4 - i * 0.7

        # Type with color indicator
        box = FancyBboxPatch(
            (0.7, y - 0.25),
            1.6,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor=color,
            alpha=0.3,
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(
            1.5,
            y,
            qtype,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

        # Other columns
        ax.text(4, y, bits, ha="center", va="center", fontsize=10, color=TEXT_COLOR)
        ax.text(6, y, ppl, ha="center", va="center", fontsize=10, color=ACCENT_ORANGE)
        ax.text(9, y, strategy, ha="center", va="center", fontsize=9, color=TEXT_COLOR)

    # Separator line
    ax.plot([0.5, 11.5], [0.3, 0.3], color=BORDER_COLOR, lw=1)

    # Naming convention legend
    ax.text(
        6,
        -0.1,
        "Naming Convention",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=ACCENT_YELLOW,
    )
    ax.text(
        6,
        -0.5,
        "Q = Quantized  |  3/4/5/6/8 = bits/weight  |  K = K-means super-blocks  |  S/M/L = Small/Medium/Large precision boost",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_COLOR,
    )

    # Note about _M variants
    ax.text(
        6,
        -0.9,
        "_M variants upgrade critical tensors (attn.wv, attn.wo, ffn.w2) to higher precision",
        ha="center",
        va="center",
        fontsize=9,
        color=ACCENT_GREEN,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig2_gguf_quant_types.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Created {output_dir}/fig2_gguf_quant_types.png")


def fig3_tensor_sensitivity(output_dir):
    """
    Figure 3: Tensor Group FIM Sensitivity Ranking
    Horizontal bar chart showing relative FIM scores by tensor type.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Data - relative FIM scores by tensor type (normalized, higher = more sensitive)
    tensors = [
        ("ffn_down", 1.00, "CRITICAL", "Most sensitive per llama.cpp #12741"),
        ("ffn_gate", 0.85, "HIGH", "Gate projection in MLP"),
        ("ffn_up", 0.82, "HIGH", "Up projection in MLP"),
        ("attn_v", 0.78, "HIGH", "Value projection"),
        ("attn_output", 0.65, "MEDIUM", "Output projection"),
        ("attn_q", 0.55, "MEDIUM", "Query projection"),
        ("attn_k", 0.45, "LOW", "Key projection"),
        ("token_emb", 0.25, "LOW", "Embedding layer"),
    ]

    names = [t[0] for t in tensors]
    scores = [t[1] for t in tensors]
    tiers = [t[2] for t in tensors]
    descriptions = [t[3] for t in tensors]

    # Color mapping
    tier_colors = {
        "CRITICAL": CRITICAL_COLOR,
        "HIGH": HIGH_COLOR,
        "MEDIUM": MEDIUM_COLOR,
        "LOW": LOW_COLOR,
    }
    colors = [tier_colors[t] for t in tiers]

    y_pos = np.arange(len(names))

    # Create horizontal bars
    bars = ax.barh(y_pos, scores, color=colors, height=0.6, alpha=0.9)

    # Add score labels on bars
    for i, (bar, score, tier) in enumerate(zip(bars, scores, tiers)):
        width = bar.get_width()
        ax.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        ax.text(
            width + 0.15,
            bar.get_y() + bar.get_height() / 2,
            tier,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=tier_colors[tier],
        )

    # Add descriptions on the right
    for i, desc in enumerate(descriptions):
        ax.text(
            1.55,
            i,
            desc,
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
            alpha=0.8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 2.2)
    ax.set_xlabel("Relative FIM Score", fontsize=12, labelpad=10)
    ax.invert_yaxis()

    # Title
    ax.set_title(
        "Tensor Group FIM Sensitivity Ranking",
        fontsize=16,
        fontweight="bold",
        color=ACCENT_CYAN,
        pad=20,
    )

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.spines["left"].set_color(BORDER_COLOR)

    # Quantization guidance legend at bottom
    legend_text = "Quantization Guidance:   CRITICAL/HIGH → Q6_K or Q5_K  |  MEDIUM → Q4_K  |  LOW → Q3_K"
    ax.text(
        0.5,
        -0.12,
        legend_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig3_tensor_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Created {output_dir}/fig3_tensor_sensitivity.png")


def fig4_fim_quant_mapping(output_dir):
    """
    Figure 4: FIM Score → Quantization Type Decision Tree
    Shows how FIM scores map to quantization choices.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        6,
        7.5,
        "FIM Score → Quantization Type Decision Tree",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=ACCENT_CYAN,
    )

    # Root node
    draw_box(ax, 6, 6.2, 2.0, 0.7, "FIM Score", ACCENT_BLUE, fontsize=11)

    # Tier nodes
    tiers = [
        (1.5, 4.5, "> 0.85\nCRITICAL", CRITICAL_COLOR),
        (4, 4.5, "0.70-0.85\nHIGH", HIGH_COLOR),
        (6.5, 4.5, "0.50-0.70\nMEDIUM", MEDIUM_COLOR),
        (9, 4.5, "< 0.50\nLOW", LOW_COLOR),
    ]

    for x, y, text, color in tiers:
        draw_box(ax, x, y, 1.8, 0.9, text, color, fontsize=9)
        # Arrow from root
        draw_arrow(ax, (6, 5.85), (x, 5.0), color=color)

    # Quantization type nodes
    quant_types = [
        (1.5, 2.8, "Q6_K\n(6.5 bpw)", CRITICAL_COLOR),
        (4, 2.8, "Q5_K\n(5.5 bpw)", HIGH_COLOR),
        (6.5, 2.8, "Q4_K\n(4.5 bpw)", MEDIUM_COLOR),
        (9, 2.8, "Q3_K\n(3.4 bpw)", LOW_COLOR),
    ]

    for x, y, text, color in quant_types:
        draw_box(ax, x, y, 1.6, 0.9, text, color, fontsize=10, alpha=0.7)
        # Arrow from tier
        draw_arrow(ax, (x, 4.0), (x, 3.3), color=color)

    # Example box at bottom
    example_box = FancyBboxPatch(
        (1.5, 0.5),
        9,
        1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=PANEL_BG,
        edgecolor=ACCENT_PURPLE,
        alpha=0.8,
        linewidth=2,
    )
    ax.add_patch(example_box)

    ax.text(
        6,
        1.6,
        "Example: Qwen2.5-1.5B with Q3_K_M base",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=ACCENT_PURPLE,
    )
    ax.text(
        6,
        1.1,
        "→ Upgrade layers 1,2,26,27 to Q6_K = 1.26% better PPL @ +1.8% size",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig4_fim_quant_mapping.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Created {output_dir}/fig4_fim_quant_mapping.png")


def fig5_mixed_precision(output_dir):
    """
    Figure 5: Uniform vs FIM-Guided Mixed Precision Comparison
    Side-by-side comparison showing precision allocation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis("off")

    # Title for whole figure
    fig.suptitle(
        "Mixed Precision Strategy: Critical Tensors Get Higher Precision",
        fontsize=16,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=0.96,
    )

    # Left panel: Uniform quantization (Q3_K_S)
    ax1.text(
        3,
        7.3,
        "Q3_K_S (Small)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=HIGH_COLOR,
    )
    ax1.text(
        3,
        6.8,
        "Uniform quantization",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_COLOR,
        style="italic",
    )

    # Tensor rows - all same color for uniform
    tensors_left = [
        ("attn.wv", "Q3_K"),
        ("attn.wo", "Q3_K"),
        ("ffn.w2", "Q3_K"),
        ("attn.q", "Q3_K"),
        ("attn.k", "Q3_K"),
        ("ffn.gate", "Q3_K"),
    ]

    for i, (name, qtype) in enumerate(tensors_left):
        y = 6.0 - i * 0.8
        # Box for tensor name
        box = FancyBboxPatch(
            (0.5, y - 0.25),
            2.2,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=PANEL_BG,
            edgecolor=HIGH_COLOR,
            alpha=0.8,
            linewidth=2,
        )
        ax1.add_patch(box)
        ax1.text(1.6, y, name, ha="center", va="center", fontsize=10, color=TEXT_COLOR)
        # Quant type
        ax1.text(
            4.5,
            y,
            qtype,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=HIGH_COLOR,
        )

    # Size and PPL
    ax1.text(
        3, 1.0, "~2.75 GB @ 7B", ha="center", va="center", fontsize=11, color=TEXT_COLOR
    )
    ax1.text(
        3,
        0.5,
        "PPL: +0.55",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=CRITICAL_COLOR,
    )

    # Right panel: FIM-guided (Q3_K_M)
    ax2.text(
        3,
        7.3,
        "Q3_K_M (Medium)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=ACCENT_GREEN,
    )
    ax2.text(
        3,
        6.8,
        "FIM-guided mixed precision",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_COLOR,
        style="italic",
    )

    # Tensor rows - critical tensors get higher precision
    tensors_right = [
        ("attn.wv", "Q4_K ★", ACCENT_GREEN, True),
        ("attn.wo", "Q4_K ★", ACCENT_GREEN, True),
        ("ffn.w2", "Q4_K ★", ACCENT_GREEN, True),
        ("attn.q", "Q3_K", HIGH_COLOR, False),
        ("attn.k", "Q3_K", HIGH_COLOR, False),
        ("ffn.gate", "Q3_K", HIGH_COLOR, False),
    ]

    for i, (name, qtype, color, is_upgraded) in enumerate(tensors_right):
        y = 6.0 - i * 0.8
        # Box for tensor name
        box = FancyBboxPatch(
            (0.5, y - 0.25),
            2.2,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=PANEL_BG,
            edgecolor=color,
            alpha=0.8,
            linewidth=2,
        )
        ax2.add_patch(box)
        ax2.text(1.6, y, name, ha="center", va="center", fontsize=10, color=TEXT_COLOR)
        # Quant type with color
        ax2.text(
            4.5,
            y,
            qtype,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    # Size and PPL
    ax2.text(
        3, 1.0, "~3.07 GB @ 7B", ha="center", va="center", fontsize=11, color=TEXT_COLOR
    )
    ax2.text(
        3,
        0.5,
        "PPL: +0.25",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=ACCENT_GREEN,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig5_mixed_precision.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Created {output_dir}/fig5_mixed_precision.png")


def fig6_g_squared_explanation(output_dir):
    """
    Figure 6: What is g² (Squared Gradient)?
    Visual explanation of the g² concept.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        6,
        7.3,
        "Understanding g² (Squared Gradient)",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=ACCENT_CYAN,
    )

    # Main equation box
    eq_box = FancyBboxPatch(
        (2, 5.5),
        8,
        1.2,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=PANEL_BG,
        edgecolor=ACCENT_CYAN,
        alpha=0.9,
        linewidth=3,
    )
    ax.add_patch(eq_box)

    ax.text(
        6,
        6.1,
        r"g² = ($\partial$L / $\partial$θ)² = (gradient)²",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=ACCENT_CYAN,
    )

    # Three interpretation boxes
    interpretations = [
        (
            2,
            4.0,
            "Sensitivity",
            "How much loss changes\nwhen weight changes",
            ACCENT_PURPLE,
            "High g² = small change\ncauses large loss increase",
        ),
        (
            6,
            4.0,
            "Importance",
            "Which weights matter\nmost for accuracy",
            ACCENT_ORANGE,
            "High g² = critical for\nmodel performance",
        ),
        (
            10,
            4.0,
            "Precision Need",
            "Quantization\nsensitivity",
            ACCENT_GREEN,
            "High g² = needs more bits\nto preserve accuracy",
        ),
    ]

    for x, y, title, desc, color, detail in interpretations:
        # Title box
        draw_box(ax, x, y + 0.3, 2.5, 0.6, title, color, fontsize=11)
        # Description
        ax.text(
            x, y - 0.5, desc, ha="center", va="center", fontsize=9, color=TEXT_COLOR
        )
        # Detail box
        detail_box = FancyBboxPatch(
            (x - 1.4, y - 1.8),
            2.8,
            0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color,
            edgecolor=color,
            alpha=0.15,
            linewidth=1,
        )
        ax.add_patch(detail_box)
        ax.text(x, y - 1.35, detail, ha="center", va="center", fontsize=8, color=color)

    # Why squaring?
    why_box = FancyBboxPatch(
        (1, 0.3),
        10,
        1.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=PANEL_BG,
        edgecolor=ACCENT_YELLOW,
        alpha=0.8,
        linewidth=2,
    )
    ax.add_patch(why_box)

    ax.text(
        6,
        1.3,
        "Why square the gradient?",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=ACCENT_YELLOW,
    )
    ax.text(
        6,
        0.7,
        "• Makes all values positive (magnitude matters, not direction)\n"
        "• Emphasizes large gradients (2x gradient = 4x importance)\n"
        "• Connects to Fisher Information theory (variance of gradient)",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig6_g_squared.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Created {output_dir}/fig6_g_squared.png")


def main():
    output_dir = create_output_dir()
    print(f"Generating visualizations to {output_dir}/\n")

    fig1_fim_pipeline(output_dir)
    fig2_gguf_quant_types(output_dir)
    fig3_tensor_sensitivity(output_dir)
    fig4_fim_quant_mapping(output_dir)
    fig5_mixed_precision(output_dir)
    fig6_g_squared_explanation(output_dir)

    print(f"\nAll visualizations generated successfully!")
    print(f"Update docs/mobile-weight-packing.md to reference these images.")


if __name__ == "__main__":
    main()
