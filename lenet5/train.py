# SPDX-License-Identifier: MIT

# Load in relevant libraries, and alias where appropriate
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import logging
import json
from datetime import datetime
import argparse
import numpy as np
from collections import deque

# Import shared library modules
from lib.optimizers import (
    create_optimizer,
    apply_spam_gradient_processing,
    apply_periodic_spam_reset,
    apply_adamprune_masking,
    update_adamprune_masks,
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="LeNet-5 training with optional pruning")
parser.add_argument(
    "--pruning-method",
    type=str,
    default="none",
    choices=["none", "movement", "magnitude", "state"],
    help="Pruning method to use (state = AdamWPrune built-in, default: none)",
)
parser.add_argument(
    "--target-sparsity",
    type=float,
    default=0.9,
    help="Target sparsity for pruning (default: 0.9)",
)
parser.add_argument(
    "--pruning-warmup",
    type=int,
    default=100,
    help="Number of warmup steps before pruning starts (default: 100)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adam", "adamw", "adamwadv", "adamwspam", "adamwprune"],
    help="Optimizer to use for training (default: sgd)",
)
parser.add_argument(
    "--spam-theta",
    type=float,
    default=50.0,
    help="SPAM spike threshold theta for approximate GSS (default: 50.0)",
)
parser.add_argument(
    "--spam-interval",
    type=int,
    default=0,
    help="SPAM periodic momentum reset interval in steps (0 disables)",
)
parser.add_argument(
    "--spam-warmup-steps",
    type=int,
    default=0,
    help="SPAM cosine warmup steps after each reset (default: 0)",
)
parser.add_argument(
    "--spam-enable-clip",
    action="store_true",
    help="Enable SPAM spike-aware clipping using Adam's second moment",
)

# AdamWPrune tuning parameters
parser.add_argument(
    "--adamwprune-beta1",
    type=float,
    default=None,
    help="Beta1 coefficient for AdamWPrune (default: 0.9)",
)
parser.add_argument(
    "--adamwprune-beta2",
    type=float,
    default=None,
    help="Beta2 coefficient for AdamWPrune (default: 0.999)",
)
parser.add_argument(
    "--adamwprune-weight-decay",
    type=float,
    default=None,
    help="Weight decay for AdamWPrune (default: 0.01)",
)
parser.add_argument(
    "--adamwprune-amsgrad",
    type=lambda x: x.lower() in ['true', '1', 'yes'],
    default=None,
    help="Enable AMSGrad for AdamWPrune (default: True)",
)
parser.add_argument(
    "--adamwprune-base-optimizer-name",
    type=str,
    default="adamw",
    choices=["adamw", "adamwspam"],
    help="Base optimizer for AdamWPrune (default: adamw)",
)

# AdamWPrune-specific pruning configuration
parser.add_argument(
    "--adamwprune-pruning-method",
    type=str,
    default="state",
    choices=["none", "state"],
    help="AdamWPrune pruning method (state or none, default: state)",
)
parser.add_argument(
    "--adamwprune-target-sparsity",
    type=float,
    default=0.7,
    help="AdamWPrune target sparsity (default: 0.7)",
)
parser.add_argument(
    "--adamwprune-warmup-steps",
    type=int,
    default=100,
    help="AdamWPrune pruning warmup steps (default: 100)",
)
parser.add_argument(
    "--adamwprune-frequency",
    type=int,
    default=50,
    help="AdamWPrune pruning update frequency (default: 50)",
)
parser.add_argument(
    "--adamwprune-ramp-end-epoch",
    type=int,
    default=75,
    help="AdamWPrune pruning ramp end epoch (default: 75)",
)

parser.add_argument(
    "--json-output",
    type=str,
    default="training_metrics.json",
    help="json output file to use for stats, deafult is training_metrics.json",
)
parser.add_argument("--weight-decay", type=float, default=None,
                    help="Weight decay (AdamW/SGD). If None, choose a sane default.")
parser.add_argument(
    "--adamwprune-variant",
    type=str,
    default="bitter0",
    choices=["bitter0", "bitter1", "bitter2", "bitter3", "bitter4",
             "bitter5", "bitter6", "bitter7", "bitter8", "bitter9"],
    help="AdamWPrune variant to use (default: bitter0)",
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
    default="lenet5-experiments",
    help="Project name for experiment tracking (default: lenet5-experiments)",
)
parser.add_argument(
    "--tracker-run-name",
    type=str,
    default=None,
    help="Run name for experiment tracking (default: auto-generated)",
)
args = parser.parse_args()

# Try to load config.py if it exists (from Kconfig)
config = None
try:
    import config as cfg
    config = cfg.Config if hasattr(cfg, 'Config') else cfg
    logger_msg = "Loaded configuration from config.py (Kconfig-generated)"
except ImportError:
    logger_msg = "No config.py found, using command-line arguments only"

# Conditionally import pruning module
if args.pruning_method == "movement":
    from lib.movement_pruning import MovementPruning
elif args.pruning_method == "magnitude":
    from lib.magnitude_pruning import MagnitudePruning
elif args.pruning_method == "state":
    # State pruning is built into AdamWPrune, no external import needed
    pass

# Define relevant variables for the ML task
batch_size = 512
num_classes = 10
learning_rate = 0.001
num_epochs = 10
num_workers = 16  # Use multiple workers for data loading

# Movement pruning hyperparameters (when enabled)
enable_pruning = args.pruning_method != "none"
initial_sparsity = 0.0  # Start with no pruning
target_sparsity = args.target_sparsity
warmup_steps = args.pruning_warmup
pruning_frequency = 50  # Update masks every 50 steps
ramp_end_epoch = 8  # Reach target sparsity by epoch 8

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
log_filename = "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)

# Initialize metrics tracking
training_metrics = {
    "start_time": datetime.now().isoformat(),
    "config": {
        "batch_size": batch_size,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "pruning_method": args.pruning_method,
        "target_sparsity": target_sparsity if enable_pruning else None,
        "pruning_warmup": warmup_steps if enable_pruning else None,
        "optimizer": args.optimizer,
    },
    "epochs": [],
}

if device.type == "cuda":
    # Enable TensorFloat32 for faster matrix multiplication on GPUs that support it
    torch.set_float32_matmul_precision("high")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"Using GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    training_metrics["device"] = {
        "type": "cuda",
        "name": gpu_name,
        "memory_gb": gpu_memory,
    }
else:
    logger.info("Using CPU")
    training_metrics["device"] = {"type": "cpu"}

# Initialize experiment trackers (W&B and/or Trackio)
trackers_enabled = []
wandb_run = None
trackio_run = None

if args.tracker:
    tracker_list = [t.strip() for t in args.tracker.split(",")]

    # Generate run name if not provided
    run_name = args.tracker_run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lenet5_{args.optimizer}_{args.pruning_method}_{timestamp}"

    # Initialize W&B
    if "wandb" in tracker_list:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.tracker_project,
                name=run_name,
                config={
                    "model": "lenet5",
                    "optimizer": args.optimizer,
                    "pruning_method": args.pruning_method,
                    "target_sparsity": target_sparsity,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "adamwprune_variant": args.adamwprune_variant if args.optimizer == "adamwprune" else None,
                },
            )
            trackers_enabled.append("wandb")
            logger.info(f"Initialized WandB tracking for project: {args.tracker_project}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B tracking")

    # Initialize Trackio
    if "trackio" in tracker_list:
        try:
            import trackio
            trackio.init(project=args.tracker_project)
            trackio_run = trackio.start_run(name=run_name)
            trackio.log_params({
                "model": "lenet5",
                "optimizer": args.optimizer,
                "pruning_method": args.pruning_method,
                "target_sparsity": target_sparsity,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "adamwprune_variant": args.adamwprune_variant if args.optimizer == "adamwprune" else None,
            })
            trackers_enabled.append("trackio")
            logger.info(f"Initialized Trackio tracking for project: {args.tracker_project}")
        except ImportError:
            logger.warning("trackio not installed, skipping Trackio tracking")

if trackers_enabled:
    logger.info(f"Tracking enabled: {', '.join(trackers_enabled)}")
    training_metrics["tracking"] = {
        "enabled": trackers_enabled,
        "project": args.tracker_project,
        "run_name": run_name,
    }

# Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(
    root="../data",
    train=True,
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    ),
    download=True,
)


test_dataset = torchvision.datasets.MNIST(
    root="../data",
    train=False,
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
        ]
    ),
    download=True,
)


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Pin memory for faster GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2,  # Prefetch batches
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size * 2,  # Larger batch for testing since no gradients
    shuffle=False,  # No need to shuffle test data
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = LeNet5(num_classes).to(device)

# Compile the model for faster execution (PyTorch 2.0+)
if torch.__version__ >= "2.0.0" and device.type == "cuda":
    logger.info("Compiling model with torch.compile()...")
    compile_start_time = time.time()
    model = torch.compile(model, mode="reduce-overhead")
    training_metrics["compile_time"] = time.time() - compile_start_time
    logger.info(
        "Model compilation completed after %.2fs", training_metrics["compile_time"]
    )
    training_metrics["model_compiled"] = True
else:
    training_metrics["model_compiled"] = False

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Enable state pruning for AdamWPrune when requested
if args.optimizer == "adamwprune" and args.pruning_method == "state":
    args.adamwprune_enable_pruning = True
    args.adamwprune_target_sparsity = args.target_sparsity
    args.adamwprune_warmup_steps = args.pruning_warmup
    # LeNet-5 trains for fewer epochs, so ramp up pruning faster
    args.adamwprune_ramp_end_epoch = min(8, num_epochs - 2)

# Create optimizer using the library function
optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state = (
    create_optimizer(
        model=model,
        optimizer_type=args.optimizer,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        args=args,
        model_type="lenet",
    )
)

# Initialize GradScaler for mixed precision training
scaler = GradScaler("cuda")

# Initialize pruning if enabled
pruner = None
# For AdamWPrune with state pruning, don't create external pruner
if enable_pruning and not (
    args.optimizer == "adamwprune" and args.pruning_method == "state"
):
    # Calculate total training steps for pruning schedule
    total_training_steps = len(train_loader) * ramp_end_epoch

    if args.pruning_method == "movement":
        pruner = MovementPruning(
            model=model,
            initial_sparsity=initial_sparsity,
            target_sparsity=target_sparsity,
            warmup_steps=warmup_steps,
            pruning_frequency=pruning_frequency,
            ramp_end_step=total_training_steps,
        )
        logger.info(f"Movement pruning enabled with target sparsity: {target_sparsity}")
    elif args.pruning_method == "magnitude":
        pruner = MagnitudePruning(
            model=model,
            initial_sparsity=initial_sparsity,
            target_sparsity=target_sparsity,
            warmup_steps=warmup_steps,
            pruning_frequency=pruning_frequency,
            ramp_end_step=total_training_steps,
        )
        logger.info(
            f"Magnitude pruning enabled with target sparsity: {target_sparsity}"
        )
    logger.info(
        f"Pruning warmup steps: {warmup_steps}, ramp end: {total_training_steps}"
    )
    training_metrics["pruning"] = {
        "method": "movement",
        "initial_sparsity": initial_sparsity,
        "target_sparsity": target_sparsity,
        "warmup_steps": warmup_steps,
        "ramp_end_step": total_training_steps,
    }

# this is defined to print how many steps are remaining when training
total_step = len(train_loader)

# GPU warmup
if device.type == "cuda":
    logger.info("Warming up GPU...")
    gpu_warmup_start_time = time.time()
    dummy_input = torch.randn(batch_size, 1, 32, 32, dtype=torch.float).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    training_metrics["gpu_warmup_time"] = time.time() - gpu_warmup_start_time
    logger.info("GPU warmed up for %.2fs", training_metrics["gpu_warmup_time"])

logger.info(f"Starting training with batch size: {batch_size}")
logger.info(f"Total training samples: {len(train_dataset)}")
logger.info(f"Total test samples: {len(test_dataset)}")
training_metrics["dataset"] = {
    "train_samples": len(train_dataset),
    "test_samples": len(test_dataset),
}

total_step = len(train_loader)
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    running_loss = 0.0
    epoch_metrics = {
        "epoch": epoch + 1,
        "losses": [],
        "batch_times": [],
    }

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass (use autocast only on CUDA)
        if device.type == "cuda":
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = cost(outputs, labels)
        else:
            outputs = model(images)
            loss = cost(outputs, labels)

        # Backward and optimize with gradient scaling
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        scaler.scale(loss).backward()

        # Apply gradient masking (AdamWPrune) and gradient clipping if enabled
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)

            # Apply AdamWPrune gradient masking
            apply_adamprune_masking(optimizer, adamprune_state)

            # Apply SPAM gradient processing
            apply_spam_gradient_processing(
                optimizer, model, spam_state, gradient_clip_norm
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        # Periodic SPAM momentum reset with optional warmup
        apply_periodic_spam_reset(optimizer, spam_state)

        scaler.step(optimizer)
        scaler.update()

        # Apply AdamWPrune state-based pruning
        update_adamprune_masks(optimizer, adamprune_state, train_loader, epoch)

        # Apply movement pruning if enabled (for non-AdamWPrune optimizers)
        if pruner is not None:
            pruner.step_pruning()

        running_loss += loss.item()

        # Let's see the hockey stick.
        #
        # We want to be able to track the first epoch well. The ideal
        # loss will depend on the number of classes. For Lenet-5 we have
        # 10 classes, so we have a uniform probability (1/10 chance for each
        # class), the expected the cross-entropy loss would be:
        #
        # -log(1/10) = -log(0.1) = 2.303
        #
        # So that's the worst case los swe expect on initialization.
        if epoch == 0:
            if i < 10:
                print_at_steps = 1
            else:
                print_at_steps = 10
        else:
            print_at_steps = 118

        if (i + 1) % print_at_steps == 0:
            avg_loss = running_loss / print_at_steps
            batch_time = time.time() - epoch_start
            logger.info(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f}s".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    total_step,
                    avg_loss,
                    batch_time,
                )
            )
            epoch_metrics["losses"].append(avg_loss)
            epoch_metrics["batch_times"].append(batch_time)
            running_loss = 0.0

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start

        # Log AdamWPrune sparsity if enabled
        if adamprune_state is not None and adamprune_state["pruning_enabled"]:
            total_params = 0
            zero_params = 0
            for mask in adamprune_state["masks"].values():
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()

            global_sparsity = zero_params / total_params if total_params > 0 else 0
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, "
                f"Test Accuracy: {accuracy:.2f}%, Sparsity: {global_sparsity:.2%}"
            )
            epoch_metrics["sparsity"] = global_sparsity
            epoch_metrics["sparsity_stats"] = {"global": {"sparsity": global_sparsity}}
        # Log pruning statistics if enabled
        elif pruner is not None:
            sparsity_stats = pruner.get_sparsity_stats()
            global_sparsity = sparsity_stats["global"]["sparsity"]
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, "
                f"Test Accuracy: {accuracy:.2f}%, Sparsity: {global_sparsity:.2%}"
            )
            epoch_metrics["sparsity"] = global_sparsity
            epoch_metrics["sparsity_stats"] = sparsity_stats
        else:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, Test Accuracy: {accuracy:.2f}%"
            )

        # Store epoch metrics
        epoch_metrics["accuracy"] = accuracy
        epoch_metrics["epoch_time"] = epoch_time
        epoch_metrics["avg_loss"] = (
            sum(epoch_metrics["losses"]) / len(epoch_metrics["losses"])
            if epoch_metrics["losses"]
            else 0
        )
        training_metrics["epochs"].append(epoch_metrics)

        # Log to trackers
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_metrics["avg_loss"],
                "test/accuracy": accuracy,
                "train/epoch_time": epoch_time,
                "train/sparsity": epoch_metrics.get("sparsity", 0),
            })
        if trackio_run is not None:
            trackio.log_metrics({
                "epoch": epoch + 1,
                "loss": epoch_metrics["avg_loss"],
                "accuracy": accuracy,
                "epoch_time": epoch_time,
                "sparsity": epoch_metrics.get("sparsity", 0),
            })

        # Step the learning rate scheduler if using AdamWAdv
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Learning rate adjusted to: {current_lr:.6f}")

total_time = time.time() - start_time
logger.info(f"Training completed in {total_time:.2f} seconds")
logger.info(f"Average time per epoch: {total_time/num_epochs:.2f} seconds")

# Log SPAM statistics if using SPAM optimizer
if spam_state is not None:
    logger.info(f"SPAM Statistics:")
    logger.info(f"  - Total momentum resets: {spam_state['momentum_reset_count']}")
    logger.info(f"  - Spike events: {len(spam_state['spike_events'])}")
    training_metrics["spam_stats"] = {
        "momentum_resets": spam_state["momentum_reset_count"],
        "spike_events": len(spam_state["spike_events"]),
        "spike_steps": spam_state["spike_events"][:10],  # First 10 spikes for analysis
    }

# Log final AdamWPrune statistics
if adamprune_state is not None and adamprune_state["pruning_enabled"]:
    # Apply final pruning
    total_params = 0
    zero_params = 0
    for module, mask in adamprune_state["masks"].items():
        module.weight.data.mul_(mask.to(module.weight.dtype))
        total_params += mask.numel()
        zero_params += (mask == 0).sum().item()

    final_sparsity = zero_params / total_params if total_params > 0 else 0
    logger.info(f"Final AdamWPrune sparsity: {final_sparsity:.2%}")

    # Count remaining parameters
    total_model_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    logger.info(f"Total parameters: {total_model_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info(f"Compression ratio: {total_model_params/non_zero_params:.2f}x")

    training_metrics["final_sparsity"] = final_sparsity
    training_metrics["total_params"] = total_model_params
    training_metrics["non_zero_params"] = non_zero_params
# Log final pruning statistics
elif pruner is not None:
    final_sparsity = pruner.prune_model_final()
    logger.info(f"Final model sparsity: {final_sparsity:.2%}")

    # Count remaining parameters
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info(f"Compression ratio: {total_params/non_zero_params:.2f}x")

    training_metrics["final_sparsity"] = final_sparsity
    training_metrics["total_params"] = total_params
    training_metrics["non_zero_params"] = non_zero_params

# Final metrics
training_metrics["total_training_time"] = total_time
training_metrics["avg_time_per_epoch"] = total_time / num_epochs
training_metrics["final_accuracy"] = accuracy
training_metrics["end_time"] = datetime.now().isoformat()

# Save the model. This uses buffered IO, so we flush to measure how long
# it takes to save with writeback incurred to the filesystem.
model_filename = "lenet5_pruned.pth" if enable_pruning else "lenet5_optimized.pth"
save_start_time = time.time()
with open(model_filename, "wb") as f:
    torch.save(model.state_dict(), f)
    f.flush()
    os.fsync(f.fileno())
training_metrics["save_time"] = time.time() - save_start_time
training_metrics["save_size"] = os.path.getsize(model_filename)

logger.info(
    "Model saved as %s (%.2f MB) in %.2fs",
    model_filename,
    training_metrics["save_size"] / (1024 * 1024),
    training_metrics["save_time"],
)

# Save metrics to JSON for plotting
with open(args.json_output, "w") as f:
    json.dump(training_metrics, f, indent=2)
logger.info(f"Training metrics saved to {args.json_output}")

# Try to generate plots automatically
try:
    import subprocess
    import os

    # Try shared script first
    plot_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "generate_training_plots.py",
    )
    if not os.path.exists(plot_script):
        plot_script = "plot_training.py"  # Fallback to local

    if os.path.exists(plot_script):
        output_name = (
            f"{args.optimizer}_{args.pruning_method}_{int(args.target_sparsity*100)}"
        )
        plot_cmd = ["python3", plot_script, args.json_output, "--output", output_name]
        result = subprocess.run(plot_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Training plots generated: {output_name}_plot.png")
        else:
            logger.debug(f"Could not generate plots: {result.stderr}")
except Exception as e:
    logger.debug(f"Plot generation skipped: {e}")

# Training complete
logger.info("Training script finished successfully")

# We use persistent_workers=True to keep workers accross epochs, so to not
# have to spawn  new ones. When we finish triajning we just need to clean up
# persistent workers explicitly, otherwise this will delay the exit.
del train_loader
del test_loader
