#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Training wrapper that adds GPU monitoring to any training script.
Supports AB testing and performance comparison.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory for lib imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from lib.gpu_monitoring import GPUMonitoringContext, TrainingGPUMonitor

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for the training wrapper."""
    parser = argparse.ArgumentParser(
        description="Training wrapper with GPU monitoring and AB testing support"
    )

    # Training script arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lenet5", "resnet18", "resnet50", "gpt2"],
        help="Model to train",
    )

    parser.add_argument(
        "--config-name",
        type=str,
        help="Configuration name for file naming (auto-generated if not provided)",
    )

    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for output files"
    )

    parser.add_argument(
        "--compare-with",
        type=str,
        help="Path to previous GPU stats file for comparison",
    )

    parser.add_argument(
        "--generate-graphs",
        action="store_true",
        help="Generate performance graphs after training",
    )

    # Pass through arguments to training script
    parser.add_argument(
        "training_args", nargs="*", help="Arguments to pass to the training script"
    )

    return parser.parse_args()


def build_config_name(model: str, training_args: List[str]) -> str:
    """Build a configuration name from model and training arguments."""
    config_parts = [model]

    # Parse common arguments from training_args
    i = 0
    while i < len(training_args):
        arg = training_args[i]

        if arg == "--optimizer" and i + 1 < len(training_args):
            config_parts.append(training_args[i + 1])
            i += 2
        elif arg == "--pruning-method" and i + 1 < len(training_args):
            pruning = training_args[i + 1]
            if pruning != "none":
                config_parts.append(pruning)
            i += 2
        elif arg == "--target-sparsity" and i + 1 < len(training_args):
            sparsity = training_args[i + 1]
            if float(sparsity) > 0:
                config_parts.append(f"s{sparsity}")
            i += 2
        else:
            i += 1

    return "_".join(config_parts)


def extract_training_metadata(
    model: str, training_args: List[str], use_ra_mla: bool = False
) -> Dict[str, Any]:
    """Extract training metadata from arguments."""
    script_name = "train_ra_mla.py" if use_ra_mla else "train.py"
    metadata = {
        "model": model,
        "training_command": f"python {model}/{script_name} " + " ".join(training_args),
        "timestamp": datetime.now().isoformat(),
    }

    # Parse arguments
    i = 0
    while i < len(training_args):
        arg = training_args[i]

        if arg.startswith("--") and i + 1 < len(training_args):
            key = arg[2:].replace("-", "_")
            value = training_args[i + 1]

            # Convert numeric values
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string

            metadata[key] = value
            i += 2
        else:
            i += 1

    return metadata


def run_training_with_monitoring(
    model: str, training_args: List[str], config_name: str, output_dir: str
) -> Optional[Path]:
    """
    Run training script with GPU monitoring.

    Returns:
        Path to GPU stats file if successful, None otherwise
    """
    # Set up paths
    model_dir = parent_dir / model
    train_script = model_dir / "train.py"

    # Check if RA+MLA is enabled for GPT-2
    use_ra_mla = False
    if model == "gpt2":
        try:
            config_path = parent_dir / "config.py"
            if config_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)

                if hasattr(config_module, "config"):
                    # Check for ablation mode first - ablation mode always uses train_ra_mla.py
                    ablation_mode = getattr(
                        config_module.config, "RA_MLA_ABLATION_MODE", False
                    )
                    if ablation_mode is True or ablation_mode == "y":
                        use_ra_mla = True
                        train_script = model_dir / "train_ra_mla.py"
                        logger.info("Ablation mode enabled - using train_ra_mla.py")
                    else:
                        # Fall back to checking ENABLE_RA_MLA
                        enable_ra_mla = getattr(
                            config_module.config, "ENABLE_RA_MLA", False
                        )
                        use_ra_mla = enable_ra_mla is True or enable_ra_mla == "y"
                        if use_ra_mla:
                            train_script = model_dir / "train_ra_mla.py"
                            logger.info("RA+MLA enabled - using train_ra_mla.py")
        except Exception as e:
            logger.warning(f"Could not check RA+MLA config: {e}")

    if not train_script.exists():
        logger.error(f"Training script not found: {train_script}")
        return None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract training metadata
    training_metadata = extract_training_metadata(model, training_args, use_ra_mla)

    # Check if DDP is enabled in kconfig (respect configuration, don't auto-detect)
    use_ddp = False
    num_gpus = 1

    # For GPT-2, check if DDP is enabled in config
    if model == "gpt2":
        try:
            config_path = parent_dir / "config.py"
            if config_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)

                if hasattr(config_module, "config"):
                    gpt2_use_ddp = getattr(config_module.config, "GPT2_USE_DDP", False)
                    use_ddp = gpt2_use_ddp is True or gpt2_use_ddp == "y"

                    # Only detect GPU count if DDP is enabled in config
                    if use_ddp:
                        try:
                            # Use detect_gpus.sh for vendor-agnostic GPU detection
                            detect_script = parent_dir / "scripts" / "detect_gpus.sh"
                            if detect_script.exists():
                                result = subprocess.run(
                                    [str(detect_script), "count"],
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                )
                                num_gpus = int(result.stdout.strip())
                                logger.info(
                                    f"DDP enabled in config - detected {num_gpus} GPUs"
                                )
                            else:
                                # Fallback to PyTorch detection
                                import torch

                                num_gpus = torch.cuda.device_count()
                                logger.info(
                                    f"DDP enabled in config - detected {num_gpus} GPUs via PyTorch"
                                )
                        except Exception as e:
                            logger.warning(
                                f"GPU detection failed, defaulting to 1 GPU: {e}"
                            )
                            num_gpus = 1
                    else:
                        logger.info("DDP not enabled in config - using single-GPU mode")
        except Exception as e:
            logger.warning(f"Could not check DDP config, using single-GPU mode: {e}")

    # Start GPU monitoring
    logger.info(f"Starting training with GPU monitoring: {config_name}")

    try:
        with GPUMonitoringContext(
            config_name, output_dir, training_metadata
        ) as gpu_monitor:

            if use_ddp and num_gpus > 1:
                # Use torchrun for DDP
                logger.info(f"Launching DDP training on {num_gpus} GPUs")
                cmd = [
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node",
                    str(num_gpus),
                    str(train_script),
                ] + training_args
                logger.info(f"DDP Command: {' '.join(cmd)}")
            else:
                # Regular single GPU/CPU training
                cmd = [sys.executable, str(train_script)] + training_args
                logger.info(f"Command: {' '.join(cmd)}")

            # Change to model directory for training
            # Set environment to disable Python buffering
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            result = subprocess.run(
                cmd,
                cwd=model_dir,
                capture_output=False,  # Allow real-time output
                text=True,
                env=env,
            )

            if result.returncode != 0:
                logger.error(f"Training failed with return code: {result.returncode}")
                return None

            logger.info("Training completed successfully")
            return gpu_monitor.get_stats_file()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None


def main():
    """Main function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    args = parse_arguments()

    # Build config name if not provided
    config_name = args.config_name
    if not config_name:
        config_name = build_config_name(args.model, args.training_args)

    logger.info(f"Training configuration: {config_name}")

    # Run training with monitoring
    stats_file = run_training_with_monitoring(
        args.model, args.training_args, config_name, args.output_dir
    )

    if not stats_file:
        logger.error("Training failed or no stats generated")
        sys.exit(1)

    logger.info(f"GPU stats saved to: {stats_file}")

    # Generate graphs if requested
    if args.generate_graphs:
        logger.info("Generating performance graphs...")
        monitor = TrainingGPUMonitor()
        monitor.stats_file = stats_file
        graph_file = monitor.generate_graphs()

        if graph_file:
            logger.info(f"Performance graphs saved to: {graph_file}")
        else:
            logger.warning("Failed to generate graphs")

    # Compare with previous run if requested
    if args.compare_with:
        compare_file = Path(args.compare_with)
        if compare_file.exists():
            logger.info(f"Comparing with previous run: {compare_file}")
            TrainingGPUMonitor.compare_runs(
                stats_file, compare_file, config_name, compare_file.stem
            )
        else:
            logger.warning(f"Comparison file not found: {compare_file}")

    logger.info("Training with monitoring completed successfully!")


if __name__ == "__main__":
    main()
