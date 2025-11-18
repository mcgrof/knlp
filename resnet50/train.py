#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Training script for ResNet-50 with pruning support."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gpu_monitoring import GPUMonitor
from lib.optimizers import (
    create_optimizer,
    apply_adamprune_masking,
    update_adamprune_masks,
)
from lib.hyperparams import apply_hyperparams
from lib.movement_pruning import MovementPruning
from lib.magnitude_pruning import MagnitudePruning
from resnet50.model import ResNet50, ResNet50WithPCA, ResNet50WithSplinePCA
import numpy as np

try:
    from config import config as GENERATED_CONFIG
except ImportError:
    GENERATED_CONFIG = None


def get_data_loaders(args):
    """Create data loaders for training and testing."""

    if args.dataset == "imagenet":
        # ImageNet data augmentation
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Assuming ImageNet is in standard format
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, "train"), transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, "val"), transform=test_transform
        )
        num_classes = 1000

    elif args.dataset == "cifar100":
        # CIFAR-100 for testing/development
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),  # Resize for ResNet-50
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(224),  # Resize for ResNet-50
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        train_dataset = datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 100

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_classes


def train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    epoch,
    pruning_method=None,
    adamprune_state=None,
    args=None,
    scaler=None,
    ctx=nullcontext(),
):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Apply pruning masks if using pruning
        if pruning_method:
            pruning_method.apply_masks()

        optimizer.zero_grad()
        with ctx:
            output = model(data)
            loss = criterion(output, target)

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Apply AdamWPrune gradient masking before optimizer step
        if args and args.optimizer == "adamwprune" and adamprune_state:
            apply_adamprune_masking(optimizer, adamprune_state)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update AdamWPrune masks periodically (per batch)
        if args and args.optimizer == "adamwprune" and adamprune_state:
            update_adamprune_masks(optimizer, adamprune_state, train_loader, epoch)

        # Update pruning (step counter and potentially masks)
        if pruning_method and hasattr(pruning_method, "step_pruning"):
            pruning_method.step_pruning()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if args and batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}\t"
                f"Acc: {100. * correct / total:.2f}%"
            )

    return train_loss / len(train_loader), 100.0 * correct / total


def test(model, device, test_loader, criterion, ctx=nullcontext()):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with ctx:
                output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total} ({accuracy:.2f}%)\n"
    )

    return test_loss, accuracy


def calculate_sparsity(model):
    """Calculate model sparsity."""
    total_params = 0
    zero_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_params += module.weight.numel()
            zero_params += (module.weight == 0).sum().item()

    return zero_params / total_params if total_params > 0 else 0


def main(args):
    """Main training loop."""
    cfg = GENERATED_CONFIG

    # Apply hyperparameter auto-detection if enabled
    if cfg is not None:
        apply_hyperparams(cfg, verbose=True, model_type="resnet50")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Initialize experiment trackers
    trackers_enabled = []
    wandb_run = None
    trackio_run = None
    run_name = args.tracker_run_name

    if args.tracker:
        tracker_list = [t.strip() for t in args.tracker.split(",")]
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"resnet50_{args.optimizer}_{args.pruning_method}_{timestamp}"

        if "wandb" in tracker_list:
            try:
                import wandb

                wandb_run = wandb.init(
                    project=args.tracker_project, name=run_name, config=vars(args)
                )
                trackers_enabled.append("wandb")
                print(f"Initialized WandB tracking for project: {args.tracker_project}")
            except ImportError:
                print("wandb not installed, skipping W&B tracking")

        if "trackio" in tracker_list:
            try:
                import trackio

                trackio.init(
                    project=args.tracker_project, name=run_name, config=vars(args)
                )
                trackio_run = True
                trackers_enabled.append("trackio")
                print(
                    f"Initialized Trackio tracking for project: {args.tracker_project}"
                )
            except ImportError:
                print("trackio not installed, skipping Trackio tracking")

    # Setup mixed precision
    scaler = None
    ctx = nullcontext()
    if device.type == "cuda":
        if cfg and cfg.get("MIXED_PRECISION") in ("y", True):
            print("Mixed precision enabled.")
            scaler = GradScaler()
            # Determine dtype for autocast
            if torch.cuda.is_bf16_supported():
                ptdtype = torch.bfloat16
                print("Using bfloat16 for mixed precision.")
            else:
                ptdtype = torch.float16
                print("Using float16 for mixed precision.")
            ctx = autocast(device_type=device.type, dtype=ptdtype)
        else:
            print("Mixed precision disabled by config or not specified.")
    else:
        print("Mixed precision not applicable for CPU training.")

    # Create data loaders
    train_loader, test_loader, num_classes = get_data_loaders(args)

    # Create model with optional tokenization
    print(f"Using tokenization method: {args.tokenizer_method}")

    if args.tokenizer_method == "pca":
        # PCA tokenization: spatial tiering by variance
        n_components = cfg.get("RESNET50_PCA_COMPONENTS", 256) if cfg else 256
        whiten = cfg.get("RESNET50_PCA_WHITEN", False) if cfg else False
        print(f"PCA tokenization: {n_components} components, whiten={whiten}")
        model = ResNet50WithPCA(
            num_classes=num_classes, n_components=n_components, whiten=whiten
        ).to(device)

        # Fit PCA on training data
        print("Fitting PCA tokenizer on training data...")
        train_images = []
        for images, _ in train_loader:
            train_images.append(images.cpu().numpy())
        train_images = np.concatenate(train_images, axis=0)
        model.fit_pca(train_images)
        print(f"PCA fitted on {len(train_images)} training images")

    elif args.tokenizer_method == "spline-pca":
        # Spline-PCA tokenization: spatial + temporal tiering
        n_components = cfg.get("RESNET50_PCA_COMPONENTS", 256) if cfg else 256
        n_control_points = cfg.get("RESNET50_SPLINE_CONTROL_POINTS", 8) if cfg else 8
        whiten = cfg.get("RESNET50_PCA_WHITEN", False) if cfg else False
        print(
            f"Spline-PCA tokenization: {n_components} components, {n_control_points} control points, whiten={whiten}"
        )
        model = ResNet50WithSplinePCA(
            num_classes=num_classes,
            n_components=n_components,
            n_control_points=n_control_points,
            whiten=whiten,
        ).to(device)

        # Fit PCA on training data
        print("Fitting PCA tokenizer on training data...")
        train_images = []
        for images, _ in train_loader:
            train_images.append(images.cpu().numpy())
        train_images = np.concatenate(train_images, axis=0)
        model.fit_pca(train_images)
        print(f"PCA fitted on {len(train_images)} training images")

    else:
        # Baseline: no tokenization
        print("Baseline mode: no tokenization")
        model = ResNet50(num_classes=num_classes).to(device)

    print(
        f"Model: ResNet-50 with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Set AdamWPrune pruning flag if using state pruning
    if args.optimizer == "adamwprune" and args.pruning_method == "state":
        args.adamwprune_enable_pruning = True
        args.adamwprune_target_sparsity = args.target_sparsity
        args.adamwprune_warmup_steps = args.pruning_warmup
        args.adamwprune_ramp_end_epoch = args.pruning_end_epoch

    # Create optimizer
    optimizer_tuple = create_optimizer(
        model,
        args.optimizer,
        args.lr,
        args=args,
        num_epochs=args.epochs,
        model_type="resnet",
    )

    adamprune_state = None
    if isinstance(optimizer_tuple, tuple):
        if len(optimizer_tuple) == 5:
            optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state = (
                optimizer_tuple
            )
        else:
            optimizer = optimizer_tuple[0]
            adamprune_state = optimizer_tuple[-1] if len(optimizer_tuple) > 4 else None
    else:
        optimizer = optimizer_tuple

    criterion = nn.CrossEntropyLoss()

    pruning_method = None
    if args.pruning_method == "state" and args.optimizer == "adamwprune":
        if adamprune_state and adamprune_state.get("pruning_enabled"):
            print(f"Using built-in state-based pruning in AdamWPrune")
    elif args.pruning_method in ["movement", "magnitude"]:
        steps_per_epoch = len(train_loader)
        ramp_end_step = int(steps_per_epoch * args.pruning_end_epoch)
        if args.pruning_method == "movement":
            pruning_method = MovementPruning(
                model,
                target_sparsity=args.target_sparsity,
                warmup_steps=args.pruning_warmup,
                pruning_frequency=50,
                ramp_end_step=ramp_end_step,
            )
        elif args.pruning_method == "magnitude":
            pruning_method = MagnitudePruning(
                model,
                target_sparsity=args.target_sparsity,
                warmup_steps=args.pruning_warmup,
                pruning_frequency=50,
                ramp_end_step=ramp_end_step,
            )

    gpu_monitor = None
    if args.monitor_gpu:
        gpu_monitor = GPUMonitor(output_dir=".")
        gpu_monitor.start()

    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "sparsity": [],
        "epoch_times": [],
    }

    start_time = time.time()
    best_accuracy = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch,
            pruning_method,
            adamprune_state,
            args,
            scaler,
            ctx,
        )
        test_loss, test_acc = test(model, device, test_loader, criterion, ctx)
        sparsity = calculate_sparsity(model)
        epoch_time = time.time() - epoch_start

        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_accuracy"].append(test_acc)
        metrics["sparsity"].append(sparsity)
        metrics["epoch_times"].append(epoch_time)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if args.save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": test_acc,
                        "sparsity": sparsity,
                    },
                    "best_model.pth",
                )

        print(
            f"Epoch {epoch}: Test Acc: {test_acc:.2f}%, Sparsity: {sparsity:.2%}, Time: {epoch_time:.1f}s"
        )

        # Log to trackers
        log_metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "sparsity": sparsity,
            "epoch_time": epoch_time,
        }
        if wandb_run:
            wandb.log(log_metrics)
        if trackio_run:
            trackio.log(log_metrics)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Final sparsity: {sparsity:.2%}")

    if gpu_monitor:
        gpu_monitor.stop()

    metrics["total_time"] = total_time
    metrics["best_accuracy"] = best_accuracy
    metrics["final_sparsity"] = sparsity

    with open(args.json_output, "w") as f:
        json.dump(metrics, f, indent=2)

    if wandb_run:
        wandb.finish()
    if trackio_run:
        trackio.finish()

    return best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-50 Training")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Path to dataset"
    )

    # Training
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (AdamW/SGD). If None, choose a sane default.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamwprune", help="Optimizer to use"
    )

    # Optimizer-specific arguments
    parser.add_argument(
        "--spam-theta", type=float, default=50.0, help="SPAM theta parameter"
    )
    parser.add_argument(
        "--spam-interval", type=int, default=0, help="SPAM interval (0=disabled)"
    )
    parser.add_argument(
        "--spam-warmup-steps", type=int, default=0, help="SPAM warmup steps"
    )
    parser.add_argument(
        "--spam-enable-clip", action="store_true", help="Enable SPAM gradient clipping"
    )
    parser.add_argument(
        "--adamwprune-beta1", type=float, default=0.9, help="AdamWPrune beta1"
    )
    parser.add_argument(
        "--adamwprune-beta2", type=float, default=0.999, help="AdamWPrune beta2"
    )
    parser.add_argument(
        "--adamwprune-weight-decay",
        type=float,
        default=0.01,
        help="AdamWPrune weight decay",
    )
    parser.add_argument(
        "--adamwprune-amsgrad", type=str, default="true", help="AdamWPrune AMSGrad"
    )
    parser.add_argument(
        "--adamwprune-base-optimizer-name",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "adamwadv", "adamwspam"],
        help="Base optimizer for AdamWPrune (default: adamw)",
    )
    parser.add_argument(
        "--adamwprune-variant",
        type=str,
        default="bitter0",
        choices=["bitter0", "bitter7"],
        help="AdamWPrune variant: bitter0 (original) or bitter7 (variance-based)",
    )

    # Pruning
    parser.add_argument(
        "--pruning-method",
        type=str,
        default="none",
        choices=["none", "magnitude", "movement", "state"],
        help="Pruning method",
    )
    parser.add_argument(
        "--target-sparsity", type=float, default=0.7, help="Target sparsity level"
    )
    parser.add_argument(
        "--pruning-warmup",
        type=int,
        default=100,
        help="Warmup steps before pruning starts",
    )
    parser.add_argument(
        "--pruning-start-epoch", type=int, default=10, help="Epoch to start pruning"
    )
    parser.add_argument(
        "--pruning-end-epoch", type=int, default=80, help="Epoch to end pruning"
    )

    # Other
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How many batches to wait before logging",
    )
    parser.add_argument("--save-model", action="store_true", help="Save the best model")
    parser.add_argument("--monitor-gpu", action="store_true", help="Monitor GPU usage")
    parser.add_argument(
        "--json-output",
        type=str,
        default="training_metrics.json",
        help="Path to save training metrics in JSON format",
    )

    # Tracker arguments
    parser.add_argument(
        "--tracker",
        type=str,
        default="",
        help="Comma-separated list of trackers: wandb,trackio (default: none)",
    )
    parser.add_argument(
        "--tracker-project",
        type=str,
        default="resnet50-experiments",
        help="Project name for experiment tracking",
    )
    parser.add_argument(
        "--tracker-run-name",
        type=str,
        default=None,
        help="Run name for experiment tracking (default: auto-generated)",
    )

    # Tokenization
    parser.add_argument(
        "--tokenizer-method",
        type=str,
        default="none",
        choices=["none", "pca", "spline-pca"],
        help="Image tokenization method",
    )

    args = parser.parse_args()

    # Load tokenizer configuration from config.py if available
    if GENERATED_CONFIG is not None and hasattr(
        GENERATED_CONFIG, "RESNET50_TOKENIZER_METHOD"
    ):
        args.tokenizer_method = GENERATED_CONFIG.RESNET50_TOKENIZER_METHOD

    main(args)
