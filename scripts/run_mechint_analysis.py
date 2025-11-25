#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Run mechanistic interpretability analysis on trained models.

This script performs KV feature-circuit analysis to identify sparse
circuits that drive specific behaviors or metrics.

Usage:
    python scripts/run_mechint_analysis.py --checkpoint model.pt --config mechint_config

Or via defconfig:
    make defconfig-<name>-mechint
    make mechint
"""

import argparse
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import DataLoader

from lib.mechint import (
    AnalysisConfig,
    KVCircuitAnalyzer,
    run_kv_circuit_analysis,
)
from lib.mechint.visualize import (
    visualize_kv_masks,
    plot_sparsity_curves,
    log_circuit_to_wandb,
    create_circuit_summary_report,
)


def load_config_from_kconfig():
    """Load mechanistic analysis config from Kconfig settings."""
    try:
        from config import config

        if not getattr(config, "KNLP_MECHINT", False):
            print("Mechanistic interpretability not enabled in config")
            return None

        # Build AnalysisConfig from Kconfig options
        cfg = AnalysisConfig(
            target_sparsity=float(
                getattr(config, "KNLP_MECHINT_KV_TARGET_SPARSITY", "0.95")
            ),
            num_steps=getattr(config, "KNLP_MECHINT_KV_STEPS", 500),
            learning_rate=float(getattr(config, "KNLP_MECHINT_KV_LR", "0.01")),
            target_metric=getattr(config, "KNLP_MECHINT_KV_TARGET_METRIC", "loss"),
            temp_schedule=getattr(
                config, "KNLP_MECHINT_KV_TEMP_SCHEDULE", "linear:1.0:0.1"
            ),
            output_dir=getattr(config, "KNLP_MECHINT_OUTPUT_DIR", "mechint_analysis"),
        )

        checkpoint_path = getattr(config, "KNLP_MECHINT_KV_CHECKPOINT", "")
        run_dir = getattr(config, "KNLP_MECHINT_KV_RUN_DIR", "")

        use_wandb = getattr(config, "KNLP_MECHINT_VISUALIZE_WANDB", True)
        save_masks = getattr(config, "KNLP_MECHINT_SAVE_MASKS", True)
        tracker_project = getattr(config, "TRACKER_PROJECT", "mechint-analysis")

        return {
            "config": cfg,
            "checkpoint": checkpoint_path,
            "run_dir": run_dir,
            "use_wandb": use_wandb,
            "save_masks": save_masks,
            "tracker_project": tracker_project,
        }

    except ImportError:
        print("Warning: config.py not found, using command-line args only")
        return None


def load_model_and_data(checkpoint_path: str, dataset_name: str = "finewebedu"):
    """
    Load trained model and dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_name: Name of dataset to use

    Returns:
        Tuple of (model, train_loader, val_loader)
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Infer model type from checkpoint
    # This is a simplified version - you'd need to adapt based on your
    # checkpoint structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load model (example for GPT-2)
    from gpt2.model import GPT2, GPTConfig

    # Load config from checkpoint if available
    if "config" in checkpoint:
        saved_args = checkpoint["config"]
        # Create GPTConfig from saved args
        config = GPTConfig.from_name("gpt2")  # start with defaults
        # Override with saved config
        if hasattr(saved_args, "kv_tying"):
            config.kv_tying = saved_args.kv_tying
        if hasattr(saved_args, "block_size"):
            config.block_size = saved_args.block_size
        print(f"Loaded config from checkpoint: kv_tying={config.kv_tying}")
    else:
        config = GPTConfig.from_name("gpt2")  # default
        print("No config in checkpoint, using default GPT-2 config")

    model = GPT2(config)
    model.load_state_dict(state_dict, strict=False)
    # Keep model in train mode for mechint optimization (need gradients)
    # KV masks will be optimized while model params stay frozen
    model.train()

    print(f"Loaded model: {model.get_num_params() / 1e6:.2f}M parameters")

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    # This is simplified - you'd use your actual data loading code
    from torch.utils.data import TensorDataset

    # Dummy data for demonstration
    # Replace with actual data loading
    block_size = 1024
    batch_size = 8

    dummy_x = torch.randint(0, 50257, (100, block_size))
    dummy_y = torch.randint(0, 50257, (100, block_size))

    train_dataset = TensorDataset(dummy_x[:80], dummy_y[:80])
    val_dataset = TensorDataset(dummy_x[80:], dummy_y[80:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return model, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Run mechanistic interpretability analysis"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (overrides Kconfig)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="finewebedu",
        help="Dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides Kconfig)",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        help="Target sparsity (overrides Kconfig)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Optimization steps (overrides Kconfig)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides Kconfig)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--visualize-only",
        type=str,
        help="Path to existing run directory to visualize (skip optimization)",
    )

    args = parser.parse_args()

    # Load config from Kconfig if available
    kconfig_data = load_config_from_kconfig()

    # Visualization-only mode
    if args.visualize_only:
        print(f"Visualization mode: loading results from {args.visualize_only}")
        masks_path = os.path.join(args.visualize_only, "final_masks.pt")

        if not os.path.exists(masks_path):
            print(f"Error: masks not found at {masks_path}")
            sys.exit(1)

        masks_dict = torch.load(masks_path)
        print(f"Loaded {len(masks_dict)} layer masks")

        # Create visualizations
        visualize_kv_masks(masks_dict, output_dir=args.visualize_only)
        print("Visualization complete!")
        return

    # Build config (command-line args override Kconfig)
    if kconfig_data:
        config = kconfig_data["config"]
        checkpoint_path = args.checkpoint or kconfig_data["checkpoint"]
        use_wandb = not args.no_wandb and kconfig_data["use_wandb"]
        tracker_project = kconfig_data["tracker_project"]
    else:
        # Use defaults if no Kconfig
        config = AnalysisConfig()
        checkpoint_path = args.checkpoint
        use_wandb = not args.no_wandb
        tracker_project = "mechint-analysis"

    # Apply command-line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.target_sparsity is not None:
        config.target_sparsity = args.target_sparsity
    if args.steps:
        config.num_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr

    # Validate checkpoint
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        print("Please specify --checkpoint or set KNLP_MECHINT_KV_CHECKPOINT")
        sys.exit(1)

    # Make output directory unique per checkpoint to avoid overwriting
    # Extract variant name from checkpoint path (e.g., "stepV0", "stepV1")
    checkpoint_name = os.path.basename(checkpoint_path).replace(".pt", "")
    config.output_dir = f"{config.output_dir}_{checkpoint_name}"
    print(f"Output directory: {config.output_dir}")

    # Load model and data
    model, train_loader, val_loader = load_model_and_data(checkpoint_path, args.dataset)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run analysis
    print("\n" + "=" * 80)
    print("Starting KV Feature-Circuit Analysis")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Target sparsity: {config.target_sparsity:.1%}")
    print(f"  Optimization steps: {config.num_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output directory: {config.output_dir}")
    print("=" * 80 + "\n")

    # Run analysis
    analyzer = KVCircuitAnalyzer(model, config, device)
    results = analyzer.run_analysis(train_loader, val_loader)

    # Save results
    print("\nSaving results...")
    analyzer.save_masks()

    # Create visualizations
    print("\nGenerating visualizations...")
    masks_dict = torch.load(os.path.join(config.output_dir, "final_masks.pt"))

    visualize_kv_masks(masks_dict, output_dir=config.output_dir)
    plot_sparsity_curves(results["history"], output_dir=config.output_dir)

    # Generate summary report
    create_circuit_summary_report(
        masks_dict,
        results["initial_metrics"],
        results["final_metrics"],
        config,
        output_dir=config.output_dir,
    )

    # Log to W&B if enabled
    if use_wandb:
        print(f"\nLogging to W&B project: {tracker_project}...")
        log_circuit_to_wandb(
            masks_dict,
            results["history"],
            config,
            project_name=tracker_project,
        )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
