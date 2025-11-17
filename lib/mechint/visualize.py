# SPDX-License-Identifier: MIT
"""
Visualization tools for mechanistic interpretability analysis.

Creates publication-quality figures showing:
- KV channel importance heatmaps per layer
- Sparsity vs faithfulness curves
- Circuit topology graphs
- Attention pattern comparisons

Integrates with W&B for interactive exploration.
"""

import os
from typing import Dict, List, Optional, Any

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def visualize_kv_masks(
    masks_dict: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str = ".",
    show: bool = False,
) -> List[str]:
    """
    Create heatmaps showing learned KV channel importance per layer.

    Args:
        masks_dict: Dictionary mapping layer names to mask data
                   (should contain 'importance' tensor [H, D])
        output_dir: Directory to save figures
        show: Whether to display figures interactively

    Returns:
        List of saved figure paths
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualization")
        return []

    saved_files = []
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots for each layer
    num_layers = len(masks_dict)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3 * num_layers), squeeze=False)

    for idx, (layer_name, mask_data) in enumerate(sorted(masks_dict.items())):
        importance = mask_data["importance"].cpu().numpy()  # [H, D]
        sparsity = mask_data.get("sparsity", 0.0)

        ax = axes[idx, 0]

        # Create heatmap
        im = ax.imshow(importance, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(
            f"{layer_name} (sparsity: {sparsity:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Head", fontsize=10)
        ax.set_xlabel("Channel Dimension", fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Importance Score", fontsize=10)

    plt.tight_layout()

    # Save figure
    filepath = os.path.join(output_dir, "kv_channel_importance.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    saved_files.append(filepath)
    print(f"Saved KV importance heatmap to {filepath}")

    if show:
        plt.show()
    else:
        plt.close()

    # Create per-layer individual figures
    for layer_name, mask_data in masks_dict.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        importance = mask_data["importance"].cpu().numpy()
        sparsity = mask_data.get("sparsity", 0.0)

        im = ax.imshow(importance, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(
            f"{layer_name} KV Channel Importance\n(Sparsity: {sparsity:.1%})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Head Index", fontsize=12)
        ax.set_xlabel("Channel Dimension", fontsize=12)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Importance Score", fontsize=12)

        # Save
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        filepath = os.path.join(output_dir, f"{safe_name}_importance.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        saved_files.append(filepath)

        if not show:
            plt.close()

    return saved_files


def plot_sparsity_curves(
    metrics_history: List[Dict[str, float]],
    output_dir: str = ".",
    show: bool = False,
) -> str:
    """
    Plot sparsity vs loss curves during optimization.

    Args:
        metrics_history: List of metric dictionaries from optimization
        output_dir: Directory to save figure
        show: Whether to display interactively

    Returns:
        Path to saved figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping sparsity curves")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    steps = [m["step"] for m in metrics_history]
    losses = [m["loss"] for m in metrics_history]
    sparsities = [m["current_sparsity"] for m in metrics_history]
    temperatures = [m["temperature"] for m in metrics_history]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Loss curve
    ax1.plot(steps, losses, label="Training Loss", color="#2ecc71", linewidth=2)
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("KV Circuit Optimization Progress", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Sparsity curve
    ax2.plot(steps, sparsities, label="Current Sparsity", color="#3498db", linewidth=2)
    if "target_sparsity" in metrics_history[0]:
        targets = [m["target_sparsity"] for m in metrics_history]
        ax2.plot(
            steps, targets, "--", label="Target Sparsity", color="#e74c3c", linewidth=2
        )
    ax2.set_ylabel("Sparsity", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Temperature curve
    ax3.plot(steps, temperatures, label="Temperature", color="#9b59b6", linewidth=2)
    ax3.set_ylabel("Temperature", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Optimization Step", fontsize=12, fontweight="bold")
    ax3.grid(alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    filepath = os.path.join(output_dir, "sparsity_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved sparsity curves to {filepath}")

    if show:
        plt.show()
    else:
        plt.close()

    return filepath


def plot_circuit_faithfulness(
    sparsity_levels: List[float],
    loss_values: List[float],
    baseline_loss: float,
    output_dir: str = ".",
    show: bool = False,
) -> str:
    """
    Plot faithfulness (loss recovery) vs sparsity.

    Args:
        sparsity_levels: List of sparsity levels tested
        loss_values: Corresponding loss values
        baseline_loss: Original model loss (unpruned)
        output_dir: Directory to save figure
        show: Whether to display interactively

    Returns:
        Path to saved figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot loss vs sparsity
    ax.plot(
        sparsity_levels,
        loss_values,
        "o-",
        color="#3498db",
        linewidth=2,
        markersize=8,
        label="Pruned Model",
    )

    # Baseline
    ax.axhline(
        baseline_loss,
        color="#2ecc71",
        linestyle="--",
        linewidth=2,
        label=f"Baseline (unpruned): {baseline_loss:.4f}",
    )

    ax.set_xlabel("Sparsity (fraction pruned)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax.set_title(
        "Circuit Faithfulness: Loss vs Sparsity", fontsize=15, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    # Add annotations
    for s, l in zip(sparsity_levels[::2], loss_values[::2]):
        degradation = ((l - baseline_loss) / baseline_loss) * 100
        ax.annotate(
            f"{degradation:+.1f}%",
            xy=(s, l),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    filepath = os.path.join(output_dir, "circuit_faithfulness.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved faithfulness curve to {filepath}")

    if show:
        plt.show()
    else:
        plt.close()

    return filepath


def log_circuit_to_wandb(
    masks_dict: Dict[str, Dict[str, torch.Tensor]],
    metrics_history: List[Dict[str, float]],
    config: Any,
    project_name: str = "mechint-analysis",
) -> None:
    """
    Log mechanistic interpretability analysis to W&B.

    Args:
        masks_dict: Dictionary of learned masks
        metrics_history: Optimization history
        config: Analysis configuration
        project_name: W&B project name
    """
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping W&B logging")
        return

    # Initialize W&B run
    run = wandb.init(project=project_name, config=vars(config), reinit=True)

    # Log configuration
    wandb.config.update(
        {
            "target_sparsity": config.target_sparsity,
            "num_steps": config.num_steps,
            "learning_rate": config.learning_rate,
            "target_metric": config.target_metric,
        }
    )

    # Log metrics over time
    for metrics in metrics_history:
        wandb.log(
            {
                "mechint/loss": metrics["loss"],
                "mechint/sparsity": metrics["current_sparsity"],
                "mechint/temperature": metrics["temperature"],
                "mechint/l1_loss": metrics.get("l1_loss", 0.0),
            },
            step=metrics["step"],
        )

    # Create and log visualizations
    temp_dir = "/tmp/mechint_viz"
    os.makedirs(temp_dir, exist_ok=True)

    # KV importance heatmaps
    heatmap_files = visualize_kv_masks(masks_dict, output_dir=temp_dir)
    for filepath in heatmap_files:
        wandb.log({f"mechint/viz/{os.path.basename(filepath)}": wandb.Image(filepath)})

    # Sparsity curves
    curves_file = plot_sparsity_curves(metrics_history, output_dir=temp_dir)
    if curves_file:
        wandb.log({"mechint/viz/sparsity_curves": wandb.Image(curves_file)})

    # Log final sparsity per layer
    for layer_name, mask_data in masks_dict.items():
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        wandb.log({f"mechint/sparsity/{safe_name}": mask_data["sparsity"]})

    # Create summary table
    layer_data = []
    for layer_name, mask_data in sorted(masks_dict.items()):
        importance = mask_data["importance"]
        layer_data.append(
            {
                "layer": layer_name,
                "sparsity": mask_data["sparsity"],
                "mean_importance": importance.mean().item(),
                "std_importance": importance.std().item(),
                "n_channels": importance.numel(),
                "n_pruned": (importance <= 0.5).sum().item(),
            }
        )

    # Create table with column names from dict keys
    if layer_data:
        columns = list(layer_data[0].keys())
        rows = [[row[col] for col in columns] for row in layer_data]
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({"mechint/layer_summary": table})

    # Log masks as artifacts
    artifact = wandb.Artifact("kv_masks", type="model")
    masks_path = os.path.join(temp_dir, "masks.pt")
    torch.save(masks_dict, masks_path)
    artifact.add_file(masks_path)
    run.log_artifact(artifact)

    print(f"Logged mechanistic analysis to W&B project: {project_name}")
    wandb.finish()


def create_circuit_summary_report(
    masks_dict: Dict[str, Dict[str, torch.Tensor]],
    initial_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    config: Any,
    output_dir: str = ".",
) -> str:
    """
    Generate markdown summary report of circuit analysis.

    Args:
        masks_dict: Learned masks
        initial_metrics: Metrics before optimization
        final_metrics: Metrics after optimization
        config: Analysis configuration
        output_dir: Where to save report

    Returns:
        Path to saved report
    """
    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("# KV Feature-Circuit Analysis Report\n")
    report.append(f"## Configuration\n")
    report.append(f"- Target sparsity: {config.target_sparsity:.1%}\n")
    report.append(f"- Optimization steps: {config.num_steps}\n")
    report.append(f"- Learning rate: {config.learning_rate}\n")
    report.append(f"- Target metric: {config.target_metric}\n\n")

    report.append(f"## Results\n")
    report.append(f"### Initial (Unpruned)\n")
    report.append(f"- Loss: {initial_metrics['loss']:.4f}\n")
    report.append(f"- Perplexity: {initial_metrics['perplexity']:.2f}\n")
    report.append(f"- Bits/Byte: {initial_metrics.get('bits_per_byte', 0):.3f}\n")
    report.append(f"- Sparsity: {initial_metrics['sparsity']:.1%}\n\n")

    report.append(f"### Final (Pruned)\n")
    report.append(f"- Loss: {final_metrics['loss']:.4f}\n")
    report.append(f"- Perplexity: {final_metrics['perplexity']:.2f}\n")
    report.append(f"- Bits/Byte: {final_metrics.get('bits_per_byte', 0):.3f}\n")
    report.append(f"- Sparsity: {final_metrics['sparsity']:.1%}\n\n")

    # Degradation analysis
    loss_degradation = (
        (final_metrics["loss"] - initial_metrics["loss"]) / initial_metrics["loss"]
    ) * 100
    report.append(f"### Faithfulness\n")
    report.append(f"- Loss degradation: {loss_degradation:+.2f}%\n")
    report.append(
        f"- Sparsity achieved: {final_metrics['sparsity']:.1%} "
        f"(target: {config.target_sparsity:.1%})\n\n"
    )

    # Per-layer breakdown
    report.append(f"## Per-Layer Analysis\n\n")
    report.append("| Layer | Sparsity | Mean Importance | Channels Kept |\n")
    report.append("|-------|----------|----------------|---------------|\n")

    for layer_name, mask_data in sorted(masks_dict.items()):
        importance = mask_data["importance"]
        sparsity = mask_data["sparsity"]
        mean_imp = importance.mean().item()
        n_total = importance.numel()
        n_kept = (importance > 0.5).sum().item()

        report.append(
            f"| {layer_name} | {sparsity:.1%} | {mean_imp:.3f} | "
            f"{n_kept}/{n_total} |\n"
        )

    filepath = os.path.join(output_dir, "circuit_analysis_report.md")
    with open(filepath, "w") as f:
        f.write("".join(report))

    print(f"Saved analysis report to {filepath}")
    return filepath


def compare_variants(
    variant_dirs: Dict[str, str],
    output_dir: str = ".",
    project_name: str = "mechint-analysis",
    use_wandb: bool = True,
) -> None:
    """
    Compare multiple mechint analysis variants and create delta
    visualizations.

    Args:
        variant_dirs: Dict mapping variant names to their output
                     directories
        output_dir: Directory to save comparison plots
        project_name: W&B project name for logging
        use_wandb: Whether to log to W&B
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping comparison")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load masks and metrics from each variant
    variants_data = {}
    for variant_name, variant_dir in variant_dirs.items():
        masks_path = os.path.join(variant_dir, "final_masks.pt")
        report_path = os.path.join(variant_dir, "circuit_analysis_report.md")

        if not os.path.exists(masks_path):
            print(f"Warning: masks not found for {variant_name} at" f" {masks_path}")
            continue

        masks_dict = torch.load(masks_path)

        # Extract metrics from report if available
        metrics = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                content = f.read()
                # Parse loss degradation and sparsity from report
                import re

                loss_match = re.search(r"Loss degradation: ([+-]?\d+\.\d+)%", content)
                sparsity_match = re.search(r"Sparsity achieved: (\d+\.\d+)%", content)
                bpb_match = re.search(r"Bits/Byte: (\d+\.\d+)", content)
                if loss_match:
                    metrics["loss_degradation"] = float(loss_match.group(1))
                if sparsity_match:
                    metrics["sparsity_achieved"] = float(sparsity_match.group(1))
                if bpb_match:
                    metrics["bits_per_byte"] = float(bpb_match.group(1))

        variants_data[variant_name] = {
            "masks": masks_dict,
            "metrics": metrics,
        }

    if len(variants_data) < 2:
        print(
            "Warning: need at least 2 variants to compare, found"
            f" {len(variants_data)}"
        )
        return

    # Create comparison visualizations
    variant_names = sorted(variants_data.keys())

    # 1. Per-layer sparsity comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_names = sorted(variants_data[variant_names[0]]["masks"].keys())
    x = np.arange(len(layer_names))
    width = 0.8 / len(variant_names)

    for i, variant_name in enumerate(variant_names):
        sparsities = [
            variants_data[variant_name]["masks"][layer]["sparsity"]
            for layer in layer_names
        ]
        offset = (i - len(variant_names) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            sparsities,
            width,
            label=variant_name,
            alpha=0.8,
        )

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sparsity", fontsize=12, fontweight="bold")
    ax.set_title("Per-Layer Sparsity Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ln.split(".")[-1] for ln in layer_names], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    sparsity_path = os.path.join(output_dir, "sparsity_comparison.png")
    plt.savefig(sparsity_path, dpi=300, bbox_inches="tight")
    print(f"Saved sparsity comparison to {sparsity_path}")
    plt.close()

    # 2. Per-layer mean importance comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, variant_name in enumerate(variant_names):
        mean_importances = [
            variants_data[variant_name]["masks"][layer]["importance"].mean().item()
            for layer in layer_names
        ]
        offset = (i - len(variant_names) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            mean_importances,
            width,
            label=variant_name,
            alpha=0.8,
        )

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Importance", fontsize=12, fontweight="bold")
    ax.set_title(
        "Per-Layer Mean Importance Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ln.split(".")[-1] for ln in layer_names], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    importance_path = os.path.join(output_dir, "importance_comparison.png")
    plt.savefig(importance_path, dpi=300, bbox_inches="tight")
    print(f"Saved importance comparison to {importance_path}")
    plt.close()

    # 3. Channels kept comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, variant_name in enumerate(variant_names):
        channels_kept = [
            (variants_data[variant_name]["masks"][layer]["importance"] > 0.5)
            .sum()
            .item()
            for layer in layer_names
        ]
        offset = (i - len(variant_names) / 2 + 0.5) * width
        ax.bar(x + offset, channels_kept, width, label=variant_name, alpha=0.8)

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Channels Kept (>0.5)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Per-Layer Channels Kept Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ln.split(".")[-1] for ln in layer_names], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    channels_path = os.path.join(output_dir, "channels_comparison.png")
    plt.savefig(channels_path, dpi=300, bbox_inches="tight")
    print(f"Saved channels comparison to {channels_path}")
    plt.close()

    # 4. Create delta report
    report = []
    report.append("# Mechint Variant Comparison\n\n")
    report.append(f"Variants: {', '.join(variant_names)}\n\n")

    report.append("## Overall Metrics\n\n")
    report.append("| Variant | Loss Degradation | Sparsity Achieved | Bits/Byte |\n")
    report.append("|---------|------------------|-------------------|------------|\n")
    for variant_name in variant_names:
        metrics = variants_data[variant_name]["metrics"]
        loss_deg = metrics.get("loss_degradation", "N/A")
        sparsity = metrics.get("sparsity_achieved", "N/A")
        bpb = metrics.get("bits_per_byte", "N/A")
        loss_str = f"{loss_deg:+.2f}%" if isinstance(loss_deg, float) else loss_deg
        sparsity_str = f"{sparsity:.1f}%" if isinstance(sparsity, float) else sparsity
        bpb_str = f"{bpb:.3f}" if isinstance(bpb, float) else bpb
        report.append(f"| {variant_name} | {loss_str} | {sparsity_str} | {bpb_str} |\n")

    report.append("\n## Per-Layer Delta\n\n")
    report.append("| Layer | " + " | ".join([f"{v} Sparsity" for v in variant_names]))
    if len(variant_names) == 2:
        report.append(" | Delta |")
    report.append("\n")
    report.append("|-------|" + "----------|" * len(variant_names))
    if len(variant_names) == 2:
        report.append("-------|")
    report.append("\n")

    for layer in layer_names:
        row = [layer.split(".")[-1]]
        sparsities = []
        for variant_name in variant_names:
            sparsity = variants_data[variant_name]["masks"][layer]["sparsity"]
            sparsities.append(sparsity)
            row.append(f"{sparsity:.1%}")

        if len(variant_names) == 2:
            delta = sparsities[1] - sparsities[0]
            row.append(f"{delta:+.1%}")

        report.append("| " + " | ".join(row) + " |\n")

    delta_report_path = os.path.join(output_dir, "variant_comparison.md")
    with open(delta_report_path, "w") as f:
        f.write("".join(report))
    print(f"Saved comparison report to {delta_report_path}")

    # 5. Create summary visualizations
    # Overall metrics comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss degradation comparison
    loss_degs = []
    for variant_name in variant_names:
        loss_deg = variants_data[variant_name]["metrics"].get("loss_degradation", 0)
        loss_degs.append(loss_deg)

    colors = ["#3498db" if i == 0 else "#e74c3c" for i in range(len(variant_names))]
    ax1.bar(variant_names, loss_degs, color=colors, alpha=0.8)
    ax1.set_ylabel("Loss Degradation (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Loss Degradation Comparison", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.3)
    ax1.grid(axis="y", alpha=0.3)

    # Sparsity achieved comparison
    sparsity_achieved = []
    for variant_name in variant_names:
        sparsity = variants_data[variant_name]["metrics"].get("sparsity_achieved", 0)
        sparsity_achieved.append(sparsity)

    ax2.bar(variant_names, sparsity_achieved, color=colors, alpha=0.8)
    ax2.set_ylabel("Sparsity Achieved (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Sparsity Achieved Comparison", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "overall_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    print(f"Saved overall summary to {summary_path}")
    plt.close()

    # 6. Create delta heatmap (difference from V0 baseline)
    if len(variant_names) >= 2:
        baseline_name = variant_names[0]  # V0 is baseline
        comparison_names = variant_names[1:]  # Compare all others to V0

        # Build delta matrix: rows=layers, cols=variants (excluding baseline)
        delta_matrix = []
        for layer in layer_names:
            baseline_sparsity = variants_data[baseline_name]["masks"][layer]["sparsity"]
            row = []
            for variant_name in comparison_names:
                variant_sparsity = variants_data[variant_name]["masks"][layer][
                    "sparsity"
                ]
                delta = variant_sparsity - baseline_sparsity
                row.append(delta)
            delta_matrix.append(row)

        delta_matrix = np.array(delta_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15
        )

        ax.set_xticks(np.arange(len(comparison_names)))
        ax.set_yticks(np.arange(len(layer_names)))
        ax.set_xticklabels(comparison_names, fontsize=11)
        ax.set_yticklabels([ln.split(".")[-1] for ln in layer_names], fontsize=10)

        ax.set_xlabel("Variant", fontsize=12, fontweight="bold")
        ax.set_ylabel("Layer", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Sparsity Delta from {baseline_name} (Baseline)",
            fontsize=14,
            fontweight="bold",
        )

        # Add text annotations
        for i in range(len(layer_names)):
            for j in range(len(comparison_names)):
                text = ax.text(
                    j,
                    i,
                    f"{delta_matrix[i, j]:+.1%}",
                    ha="center",
                    va="center",
                    color="white" if abs(delta_matrix[i, j]) > 0.075 else "black",
                    fontsize=9,
                )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Sparsity Delta", fontsize=11, fontweight="bold")

        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "delta_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        print(f"Saved delta heatmap to {heatmap_path}")
        plt.close()

    # 7. Generate automated findings summary
    findings = []
    findings.append("KEY FINDINGS")
    findings.append("=" * 60)
    findings.append("")

    # Overall metrics comparison
    if len(variant_names) == 2:
        v0_name, v1_name = variant_names
        v0_loss = variants_data[v0_name]["metrics"].get("loss_degradation", 0)
        v1_loss = variants_data[v1_name]["metrics"].get("loss_degradation", 0)
        loss_diff = v1_loss - v0_loss

        if abs(loss_diff) > 0.5:
            if loss_diff > 0:
                findings.append(
                    f"• {v1_name} has WORSE loss degradation: "
                    f"{v1_loss:+.2f}% vs {v0_loss:+.2f}%"
                )
            else:
                findings.append(
                    f"• {v1_name} has BETTER loss degradation: "
                    f"{v1_loss:+.2f}% vs {v0_loss:+.2f}%"
                )
        else:
            findings.append(
                f"• Similar loss degradation: "
                f"{v1_name}={v1_loss:+.2f}%, {v0_name}={v0_loss:+.2f}%"
            )

        findings.append("")

        # Per-layer sparsity analysis
        early_layers = layer_names[:4]
        middle_layers = layer_names[4:8]
        late_layers = layer_names[8:]

        def avg_delta(layers):
            deltas = []
            for layer in layers:
                v0_sp = variants_data[v0_name]["masks"][layer]["sparsity"]
                v1_sp = variants_data[v1_name]["masks"][layer]["sparsity"]
                deltas.append(v1_sp - v0_sp)
            return np.mean(deltas)

        early_delta = avg_delta(early_layers)
        middle_delta = avg_delta(middle_layers)
        late_delta = avg_delta(late_layers)

        findings.append("Sparsity Pattern Differences:")
        findings.append(
            f"  Early layers (0-3):  {v1_name} is "
            f"{early_delta:+.1%} {'more' if early_delta > 0 else 'less'} sparse"
        )
        findings.append(
            f"  Middle layers (4-7): {v1_name} is "
            f"{middle_delta:+.1%} {'more' if middle_delta > 0 else 'less'} sparse"
        )
        findings.append(
            f"  Late layers (8-11):  {v1_name} is "
            f"{late_delta:+.1%} {'more' if late_delta > 0 else 'less'} sparse"
        )

        findings.append("")

        # Find most different layers
        deltas_per_layer = []
        for layer in layer_names:
            v0_sp = variants_data[v0_name]["masks"][layer]["sparsity"]
            v1_sp = variants_data[v1_name]["masks"][layer]["sparsity"]
            deltas_per_layer.append((layer, v1_sp - v0_sp))

        # Sort by absolute delta
        deltas_per_layer.sort(key=lambda x: abs(x[1]), reverse=True)

        findings.append("Biggest Differences:")
        for layer, delta in deltas_per_layer[:3]:
            layer_short = layer.split(".")[-2]  # e.g., "h.7"
            findings.append(f"  {layer_short}: {delta:+.1%}")

    findings.append("")
    findings.append("=" * 60)

    # Create text image from findings
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    findings_text = "\n".join(findings)
    ax.text(
        0.05,
        0.95,
        findings_text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    findings_path = os.path.join(output_dir, "key_findings.png")
    plt.savefig(findings_path, dpi=300, bbox_inches="tight")
    print(f"Saved key findings to {findings_path}")
    plt.close()

    # Also save as text file
    findings_txt_path = os.path.join(output_dir, "key_findings.txt")
    with open(findings_txt_path, "w") as f:
        f.write(findings_text)
    print(f"Saved key findings text to {findings_txt_path}")

    # Log to W&B if enabled
    if use_wandb and WANDB_AVAILABLE:
        print(f"Logging comparison to W&B project: {project_name}")
        run = wandb.init(
            project=project_name,
            name="variant_comparison",
            tags=["comparison", "delta"],
            reinit=True,
        )

        # Log comparison images
        log_dict = {
            "comparison/sparsity": wandb.Image(sparsity_path),
            "comparison/importance": wandb.Image(importance_path),
            "comparison/channels": wandb.Image(channels_path),
            "comparison/overall_summary": wandb.Image(summary_path),
            "comparison/key_findings": wandb.Image(findings_path),
        }

        # Add heatmap if available
        if len(variant_names) >= 2:
            log_dict["comparison/delta_heatmap"] = wandb.Image(heatmap_path)

        wandb.log(log_dict)

        # Create comparison table
        table_data = []
        for layer in layer_names:
            row = {"layer": layer.split(".")[-1]}
            for variant_name in variant_names:
                mask_data = variants_data[variant_name]["masks"][layer]
                row[f"{variant_name}_sparsity"] = mask_data["sparsity"]
                row[f"{variant_name}_importance"] = (
                    mask_data["importance"].mean().item()
                )
                row[f"{variant_name}_channels_kept"] = (
                    (mask_data["importance"] > 0.5).sum().item()
                )

            if len(variant_names) == 2:
                v0, v1 = variant_names
                row["sparsity_delta"] = (
                    variants_data[v1]["masks"][layer]["sparsity"]
                    - variants_data[v0]["masks"][layer]["sparsity"]
                )
                row["importance_delta"] = (
                    variants_data[v1]["masks"][layer]["importance"].mean().item()
                    - variants_data[v0]["masks"][layer]["importance"].mean().item()
                )

            table_data.append(row)

        if table_data:
            columns = list(table_data[0].keys())
            rows = [[row[col] for col in columns] for row in table_data]
            table = wandb.Table(columns=columns, data=rows)
            wandb.log({"comparison/per_layer": table})

        wandb.finish()
        print("Comparison logged to W&B")
