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
    report.append(f"- Sparsity: {initial_metrics['sparsity']:.1%}\n\n")

    report.append(f"### Final (Pruned)\n")
    report.append(f"- Loss: {final_metrics['loss']:.4f}\n")
    report.append(f"- Perplexity: {final_metrics['perplexity']:.2f}\n")
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
