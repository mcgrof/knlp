#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
ResNet-18 training script with CIFAR-10 dataset.
Supports multiple optimizers and pruning methods.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add parent directory to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from lib.experiment_tracker import ExperimentTracker

# Import pruning methods and optimizers from parent
from lib.magnitude_pruning import MagnitudePruning
from lib.movement_pruning import MovementPruning
from lib.optimizers import (
    create_optimizer,
    apply_adamprune_masking,
    update_adamprune_masks,
)

# Import config if available
try:
    import config

    BATCH_SIZE = getattr(config, "BATCH_SIZE", 128)
    NUM_EPOCHS = getattr(config, "NUM_EPOCHS", 100)
    LEARNING_RATE = float(getattr(config, "LEARNING_RATE", 0.1))
    NUM_WORKERS = getattr(config, "NUM_WORKERS", 4)
    DEVICE = getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    # Default values if no config
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 data loaders with augmentation."""

    # Data augmentation for training
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # No augmentation for validation
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Download and load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def create_resnet_optimizer(
    model,
    optimizer_name,
    lr,
    pruning_method=None,
    pruning_warmup=100,
    target_sparsity=0.0,
    num_epochs=100,
    **kwargs,
):
    """Create optimizer for ResNet-18, uses shared create_optimizer with ResNet-specific LR scaling."""

    # Use shared optimizer creation with ResNet-specific parameters
    # Scale learning rates for Adam-based optimizers on ResNet
    if optimizer_name in ["adam", "adamw", "adamwadv", "adamwspam", "adamwprune"]:
        scaled_lr = lr * 0.001  # Scale LR for Adam-based optimizers
    else:
        scaled_lr = lr

    # Create a mock args object with all parameters
    class MockArgs:
        def __init__(self, **kwargs):
            # Set default values for all expected attributes
            self.spam_theta = 50.0
            self.spam_interval = 0
            self.spam_warmup_steps = 0
            self.spam_enable_clip = False
            self.pruning_method = pruning_method or "none"
            self.pruning_warmup = pruning_warmup
            self.pruning_frequency = 100
            self.target_sparsity = target_sparsity

            # AdamWPrune-specific defaults
            self.adamwprune_base_optimizer_name = "adamw"
            self.adamwprune_enable_pruning = False

            # Override with provided kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock_args = MockArgs(**kwargs)

    # Call shared create_optimizer - returns (optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state)
    optimizer_tuple = create_optimizer(
        model,
        optimizer_name,
        scaled_lr,
        args=mock_args,
        num_epochs=num_epochs,
        model_type="resnet",
    )

    if isinstance(optimizer_tuple, tuple) and len(optimizer_tuple) >= 5:
        optimizer, _, _, _, adamprune_state = optimizer_tuple
        return optimizer, adamprune_state, optimizer_name
    else:
        # Fallback if return format is different
        optimizer = (
            optimizer_tuple[0]
            if isinstance(optimizer_tuple, tuple)
            else optimizer_tuple
        )
        return optimizer, None, optimizer_name


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    pruner,
    device,
    adamprune_state=None,
    optimizer_name=None,
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Apply AdamWPrune gradient masking before optimizer step
        if optimizer_name == "adamwprune" and adamprune_state:
            apply_adamprune_masking(optimizer, adamprune_state)

        optimizer.step()

        # Update AdamWPrune masks periodically (per batch)
        if optimizer_name == "adamwprune" and adamprune_state:
            update_adamprune_masks(optimizer, adamprune_state, train_loader, 0)

        # Apply external pruning if enabled (for non-AdamWPrune optimizers)
        if pruner is not None:
            pruner.step_pruning()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)

    return avg_loss, accuracy


def get_sparsity(model):
    """Calculate current model sparsity."""
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight"):
                weight = module.weight.data
                total_params += weight.numel()
                pruned_params += (weight == 0).sum().item()

    sparsity = pruned_params / total_params if total_params > 0 else 0
    return sparsity


def main():
    # Get config defaults if available
    try:
        import config as cfg

        # Apply hyperparameter auto-detection if enabled
        from lib.hyperparams import apply_hyperparams

        apply_hyperparams(cfg, verbose=True, model_type="resnet18")

        pruning_method_default = getattr(cfg, "PRUNING_METHOD", "none")
        target_sparsity_default = getattr(cfg, "TARGET_SPARSITY", 0.9)
        pruning_warmup_default = getattr(cfg, "PRUNING_WARMUP", 500)
        optimizer_default = getattr(cfg, "OPTIMIZER", "sgd")
        spam_theta_default = getattr(cfg, "SPAM_THETA", 50.0)
        spam_interval_default = getattr(cfg, "SPAM_INTERVAL", 0)
        spam_warmup_default = getattr(cfg, "SPAM_WARMUP_STEPS", 0)
        # AdamWPrune-specific configs
        adamwprune_base_optimizer_name_default = getattr(
            cfg, "ADAMWPRUNE_BASE_OPTIMIZER_NAME", "adamw"
        )
        adamwprune_enable_pruning_default = getattr(
            cfg, "ADAMWPRUNE_ENABLE_PRUNING", False
        )
        adamwprune_pruning_method_default = getattr(
            cfg, "ADAMWPRUNE_PRUNING_METHOD", "state"
        )
        adamwprune_target_sparsity_default = getattr(
            cfg, "ADAMWPRUNE_TARGET_SPARSITY", "0.7"
        )
        adamwprune_warmup_steps_default = getattr(cfg, "ADAMWPRUNE_WARMUP_STEPS", 100)
        adamwprune_frequency_default = getattr(cfg, "ADAMWPRUNE_FREQUENCY", 50)
        adamwprune_ramp_end_epoch_default = getattr(
            cfg, "ADAMWPRUNE_RAMP_END_EPOCH", 75
        )
    except ImportError:
        pruning_method_default = "none"
        target_sparsity_default = 0.9
        pruning_warmup_default = 500
        optimizer_default = "sgd"
        spam_theta_default = 50.0
        spam_interval_default = 0
        spam_warmup_default = 0
        # AdamWPrune-specific defaults
        adamwprune_base_optimizer_name_default = "adamw"
        adamwprune_enable_pruning_default = False
        adamwprune_pruning_method_default = "state"
        adamwprune_target_sparsity_default = "0.7"
        adamwprune_warmup_steps_default = 100
        adamwprune_frequency_default = 50
        adamwprune_ramp_end_epoch_default = 75

    parser = argparse.ArgumentParser(description="ResNet-18 training on CIFAR-10")
    parser.add_argument(
        "--pruning-method",
        type=str,
        default=pruning_method_default,
        choices=["none", "movement", "magnitude", "state"],
        help="Pruning method to use (state = AdamWPrune built-in)",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=target_sparsity_default,
        help="Target sparsity for pruning",
    )
    parser.add_argument(
        "--pruning-warmup",
        type=int,
        default=pruning_warmup_default,
        help="Number of warmup steps before pruning starts",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=optimizer_default,
        choices=["sgd", "adam", "adamw", "adamwadv", "adamwspam", "adamwprune"],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--spam-theta",
        type=float,
        default=spam_theta_default,
        help="SPAM spike threshold theta",
    )
    parser.add_argument(
        "--spam-interval",
        type=int,
        default=spam_interval_default,
        help="SPAM periodic reset interval",
    )
    parser.add_argument(
        "--spam-warmup-steps",
        type=int,
        default=spam_warmup_default,
        help="SPAM warmup steps after reset",
    )
    parser.add_argument(
        "--spam-enable-clip",
        action="store_true",
        help="Enable SPAM spike-aware clipping",
    )

    # AdamWPrune tuning parameters
    parser.add_argument(
        "--adamwprune-beta1",
        type=float,
        default=0.9,
        help="Beta1 coefficient for AdamWPrune (default: 0.9)",
    )
    parser.add_argument(
        "--adamwprune-beta2",
        type=float,
        default=0.999,
        help="Beta2 coefficient for AdamWPrune (default: 0.999)",
    )
    parser.add_argument(
        "--adamwprune-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamWPrune (default: 0.01)",
    )
    parser.add_argument(
        "--adamwprune-amsgrad",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Enable AMSGrad for AdamWPrune (default: True)",
    )
    parser.add_argument(
        "--adamwprune-variant",
        type=str,
        default="bitter0",
        choices=["bitter0", "bitter7"],
        help="AdamWPrune variant: bitter0 (original) or bitter7 (variance-based)",
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default="training_metrics.json",
        help="JSON output file for metrics",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (AdamW/SGD). If None, choose a sane default.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="",
        help="Comma-separated list of trackers: wandb,trackio (default: none)",
    )
    parser.add_argument(
        "--tracker-project",
        type=str,
        default="resnet18-training",
        help="Project name for experiment tracking",
    )
    parser.add_argument(
        "--tracker-run-name",
        type=str,
        default=None,
        help="Run name for experiment tracking (default: auto-generated)",
    )

    args = parser.parse_args()

    # Add AdamWPrune-specific configs to args
    # For AdamWPrune, check if pruning is enabled via ADAMWPRUNE_ENABLE_PRUNING config
    # or if the general pruning method is "none"
    if args.optimizer == "adamwprune":
        # Base configuration
        args.adamwprune_base_optimizer_name = adamwprune_base_optimizer_name_default
        args.adamwprune_enable_pruning = adamwprune_enable_pruning_default

        # Check if AdamWPrune pruning is explicitly disabled
        if not adamwprune_enable_pruning_default or args.pruning_method == "none":
            args.adamwprune_enable_pruning = False
            args.adamwprune_pruning_method = "none"
            args.adamwprune_target_sparsity = 0.0
        else:
            args.adamwprune_pruning_method = adamwprune_pruning_method_default
            args.adamwprune_target_sparsity = float(adamwprune_target_sparsity_default)
    else:
        # For other optimizers, set defaults (won't be used)
        args.adamwprune_base_optimizer_name = "adamw"
        args.adamwprune_enable_pruning = False
        args.adamwprune_pruning_method = "none"
        args.adamwprune_target_sparsity = 0.0

    args.adamwprune_warmup_steps = adamwprune_warmup_steps_default
    args.adamwprune_frequency = adamwprune_frequency_default
    args.adamwprune_ramp_end_epoch = adamwprune_ramp_end_epoch_default

    print("=" * 60)
    print("ResNet-18 Training on CIFAR-10")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Optimizer: {args.optimizer}")
    # Show pruning info based on optimizer type
    if args.optimizer == "adamwprune":
        print(f"AdamWPrune Pruning: {args.adamwprune_pruning_method}")
        if args.adamwprune_pruning_method == "state":
            print(f"AdamWPrune Target Sparsity: {args.adamwprune_target_sparsity:.1%}")
    else:
        print(f"Pruning: {args.pruning_method}")
        if args.pruning_method != "none":
            print(f"Target Sparsity: {args.target_sparsity:.1%}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    # Set device
    device = torch.device(DEVICE)

    # Start GPU monitoring for training if enabled
    gpu_monitor = None
    gpu_monitor_file = None
    try:
        import config as cfg

        gpu_monitor_enabled = getattr(cfg, "GPU_MONITOR", False)
    except ImportError:
        gpu_monitor_enabled = False

    if gpu_monitor_enabled:
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from scripts.monitor_gpu import GPUMonitor

            # Save GPU monitoring in the same directory as other outputs
            output_dir = (
                Path(args.json_output).parent if args.json_output else Path(".")
            )
            gpu_monitor_file = (
                output_dir
                / f"gpu_training_{args.optimizer}_{args.pruning_method}_{int(args.target_sparsity*100)}.json"
            )
            gpu_monitor = GPUMonitor(
                str(gpu_monitor_file), model_name=f"training_{args.optimizer}"
            )

            if gpu_monitor.start():
                print(
                    f"GPU monitoring started for training (output: {gpu_monitor_file})"
                )
                time.sleep(0.5)  # Let monitor initialize
        except Exception as e:
            print(f"GPU monitoring not available: {e}")
            gpu_monitor = None

    # Initialize experiment trackers (W&B and/or Trackio)
    tracker_config = {
        "optimizer": args.optimizer,
        "pruning_method": args.pruning_method,
        "target_sparsity": args.target_sparsity,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "model": "resnet18",
        "dataset": "cifar10",
    }

    tracker = ExperimentTracker(
        tracker_list=args.tracker,
        project=args.tracker_project,
        run_name=args.tracker_run_name,
        config=tracker_config,
        resume="allow",
    )

    # Create model
    model = create_model(num_classes=10).to(device)

    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(args.batch_size, NUM_WORKERS)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer, adamprune_state, optimizer_name = create_resnet_optimizer(
        model,
        args.optimizer,
        args.lr,
        pruning_method=args.pruning_method,
        pruning_warmup=args.pruning_warmup,
        target_sparsity=args.target_sparsity,
        num_epochs=args.epochs,
        spam_theta=args.spam_theta,
        spam_interval=args.spam_interval,
        spam_warmup_steps=getattr(args, "spam_warmup_steps", 0),
        spam_enable_clip=getattr(args, "spam_enable_clip", False),
        # AdamWPrune-specific params
        adamwprune_pruning_method=args.adamwprune_pruning_method,
        adamwprune_target_sparsity=args.adamwprune_target_sparsity,
        adamwprune_warmup_steps=args.adamwprune_warmup_steps,
        adamwprune_frequency=args.adamwprune_frequency,
        adamwprune_ramp_end_epoch=args.adamwprune_ramp_end_epoch,
    )

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Setup pruning if requested
    pruner = None
    if (
        args.optimizer == "adamwprune"
        and adamprune_state
        and adamprune_state.get("pruning_enabled", False)
    ):
        # AdamWPrune has built-in state-based pruning for movement/state methods
        print(f"Using built-in state-based pruning in AdamWPrune")
        print(f"  Target sparsity: {adamprune_state['target_sparsity']:.1%}")
        print(f"  Warmup steps: {adamprune_state['warmup_steps']}")
        # No external pruner needed - using built-in
    elif args.pruning_method == "magnitude":
        pruner = MagnitudePruning(
            model,
            target_sparsity=args.target_sparsity,
            warmup_steps=args.pruning_warmup,
        )
    elif args.pruning_method == "movement":
        pruner = MovementPruning(
            model,
            target_sparsity=args.target_sparsity,
            warmup_steps=args.pruning_warmup,
        )

    # Training metrics
    # For adamwprune, always record as 'state' since it uses built-in state pruning
    actual_pruning_method = (
        "state"
        if args.optimizer == "adamwprune" and args.pruning_method != "none"
        else args.pruning_method
    )
    metrics = {
        "model": "resnet18",
        "dataset": "cifar10",
        "optimizer": args.optimizer,
        "pruning_method": actual_pruning_method,
        "target_sparsity": args.target_sparsity,
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "sparsity": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    best_accuracy = 0.0
    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            pruner,
            device,
            adamprune_state=adamprune_state,
            optimizer_name=optimizer_name,
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Get current sparsity
        sparsity = get_sparsity(model)

        # Step scheduler
        scheduler.step()

        # Record metrics
        epoch_time = time.time() - epoch_start
        metrics["epochs"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_accuracy"].append(test_acc)
        metrics["sparsity"].append(sparsity)
        metrics["learning_rate"].append(current_lr)
        metrics["epoch_time"].append(epoch_time)

        # Update best accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc

        # Print progress
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.2f}% "
            f"Sparsity: {sparsity:.1%} "
            f"LR: {current_lr:.6f} "
            f"Time: {epoch_time:.2f}s"
        )

        # Log to experiment trackers
        tracker.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "sparsity": sparsity,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
        )

    # Training complete
    total_time = time.time() - start_time

    # Stop GPU monitoring for training
    if gpu_monitor:
        time.sleep(0.5)  # Ensure final measurements are captured
        samples = gpu_monitor.stop()
        print(f"GPU monitoring stopped (captured {samples} samples during training)")

    print("=" * 60)
    print(f"Training Complete!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Final Sparsity: {sparsity:.1%}")

    # Add summary metrics
    metrics["total_time"] = total_time
    metrics["best_accuracy"] = best_accuracy
    metrics["final_accuracy"] = test_acc
    metrics["final_sparsity"] = sparsity

    # Create output directory if it doesn't exist
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(args.json_output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.json_output}")

    # Try to generate plots automatically
    try:
        import subprocess

        # Use the shared generate_training_plots.py from scripts directory
        plot_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "generate_training_plots.py",
        )
        if os.path.exists(plot_script):
            output_name = f"{args.optimizer}_{args.pruning_method}_{int(args.target_sparsity*100)}"
            plot_cmd = [
                "python3",
                plot_script,
                args.json_output,
                "--output",
                output_name,
            ]
            result = subprocess.run(
                plot_cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"Training plots generated: {output_name}_plot.png")
    except Exception as e:
        # Silently skip if plotting fails
        pass

    # Save model checkpoint if enabled
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config

        save_checkpoint = getattr(config, "SAVE_CHECKPOINT", True)
        run_inference_test = getattr(config, "INFERENCE_TEST", False)
        inference_batch_sizes = getattr(config, "INFERENCE_BATCH_SIZES", "1,32,128")
    except ImportError as e:
        save_checkpoint = True
        run_inference_test = False
        inference_batch_sizes = "1,32,128"

    if save_checkpoint:
        checkpoint = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "final_sparsity": sparsity,
        }
        checkpoint_path = "resnet18_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # Run inference testing if enabled
    if run_inference_test:
        try:
            from lib.inference_tester import run_inference_test

            print("\n" + "=" * 60)
            print("Running Inference Memory Test")
            print("=" * 60)

            inference_results = run_inference_test(
                model=model,
                model_name=f"resnet18_{args.optimizer}_{args.pruning_method}_{int(args.target_sparsity*100)}",
                input_shape=(3, 32, 32),  # CIFAR-10 input shape
                batch_sizes=inference_batch_sizes,
                save_path=Path(args.json_output).parent / "inference_results.json",
                enable_gpu_monitor=gpu_monitor_enabled,  # Use same config as training
            )

            print(f"Inference test complete!")

            # Automatically analyze GPU usage if monitoring was enabled
            if gpu_monitor_enabled:
                print("\n" + "=" * 60)
                print("Analyzing GPU Memory Usage")
                print("=" * 60)
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "python",
                            "../scripts/analyze_compare_results.py",
                            "--dir",
                            ".",
                            "--output",
                            "gpu_memory_analysis.png",
                        ],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent,
                    )
                    if result.returncode == 0:
                        print(
                            "GPU memory analysis complete. View gpu_memory_analysis.png"
                        )
                        if result.stdout:
                            print(result.stdout)
                    else:
                        print(f"GPU analysis failed: {result.stderr}")
                except Exception as e:
                    print(f"Could not run GPU analysis: {e}")

        except Exception as e:
            print(f"Warning: Inference test failed: {e}")

    # Finish experiment tracking
    tracker.finish()


if __name__ == "__main__":
    main()
