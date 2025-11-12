"""
GPT-2 training script with AdamWPrune support.
Adapted from nanoGPT by Andrej Karpathy: https://github.com/karpathy/nanoGPT

Integrates with the AdamWPrune optimizer for state-based pruning experiments.
"""

import os
import sys

# CRITICAL: Set environment variables before importing torch
# Read from config.py if available, otherwise use defaults
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from config import Config

    config = Config()
    if hasattr(config, "PYTORCH_CUDA_ALLOC_CONF"):
        # Set both old and new variable names for compatibility
        alloc_conf = config.PYTORCH_CUDA_ALLOC_CONF
        os.environ.setdefault("PYTORCH_ALLOC_CONF", alloc_conf)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)

    # Detect GPU type and clear ROCm variables if on NVIDIA
    # This prevents spurious "No ROCm runtime found" messages on NVIDIA GPUs
    try:
        import subprocess

        # Try nvidia-smi first (fast check for NVIDIA GPUs)
        nvidia_result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if nvidia_result.returncode == 0:
            # NVIDIA GPU detected - clear ROCm environment variables
            for var in ["ROCM_PATH", "ROCM_HOME", "HIP_PATH", "HIP_PLATFORM"]:
                if var in os.environ:
                    del os.environ[var]
            # Also remove ROCm paths from LD_LIBRARY_PATH
            if "LD_LIBRARY_PATH" in os.environ:
                ld_paths = os.environ["LD_LIBRARY_PATH"].split(":")
                ld_paths = [
                    p for p in ld_paths if "/opt/rocm" not in p and "/hip/" not in p
                ]
                os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # nvidia-smi not available

    # Enable experimental flash attention for Navi31/32/33 when explicitly requested
    # Only set environment variable if CONFIG_GPT2_FLASH_ATTENTION=y AND we detect Navi3x
    flash_attention_enabled = (
        hasattr(config, "GPT2_FLASH_ATTENTION")
        and config.GPT2_FLASH_ATTENTION
        and config.GPT2_FLASH_ATTENTION != "n"
    )

    if flash_attention_enabled:
        # Detect Navi31/32/33 before importing torch using rocm-smi
        try:
            import subprocess

            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.lower()
                is_navi3x = any(
                    x in gpu_name for x in ["navi31", "navi32", "navi33", "w7900"]
                )
                if is_navi3x:
                    os.environ.setdefault(
                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # rocm-smi not available or timed out
except (ImportError, AttributeError):
    # Fallback to safe defaults if config.py doesn't exist or doesn't have the settings
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import math
import pickle
import argparse
from contextlib import nullcontext
from datetime import datetime
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Suppress wandb weave warning
try:
    import weave
except ImportError:
    pass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig

# Import from parent directory's lib
try:
    from lib.optimizers import (
        create_optimizer,
        apply_spam_gradient_processing,
        apply_periodic_spam_reset,
        apply_adamprune_masking,
        update_adamprune_masks,
    )
    from lib.movement_pruning import MovementPruning
    from lib.magnitude_pruning import MagnitudePruning
except ImportError:
    # If direct import fails, try adding parent to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lib.optimizers import (
        create_optimizer,
        apply_spam_gradient_processing,
        apply_periodic_spam_reset,
        apply_adamprune_masking,
        update_adamprune_masks,
    )
    from lib.movement_pruning import MovementPruning
    from lib.magnitude_pruning import MagnitudePruning


# -----------------------------------------------------------------------------
# Helper functions
def supports_tensorcore_fp32():
    """
    Detect if the current device supports tensor cores (NVIDIA) or WMMA (AMD).
    Works on both CUDA and ROCm backends.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check device capability for NVIDIA GPUs
        cap = torch.cuda.get_device_capability(0)
        # Tensor cores require SM >= 7.0 (Volta and newer)
        if cap and cap[0] >= 7:
            return True
    except Exception:
        pass

    # ROCm heuristic: look for RDNA3 / gfx11+ which have WMMA ISA
    if torch.version.hip:
        try:
            arch = torch.cuda.get_device_name(0).lower()
            # Check for RDNA3 (gfx110x, Navi3x, W7900) or MI series
            return any(
                x in arch
                for x in [
                    "gfx110",
                    "navi31",
                    "navi32",
                    "navi33",
                    "w7900",
                    "mi300",
                    "mi250",
                ]
            )
        except Exception:
            pass

    return False


def is_navi3x_gpu():
    """
    Detect if the current device is AMD RDNA3 (Navi31/32/33).
    Flash attention support is experimental on these GPUs.
    """
    if not torch.cuda.is_available():
        return False

    if torch.version.hip:
        try:
            arch = torch.cuda.get_device_name(0).lower()
            return any(
                x in arch for x in ["gfx110", "navi31", "navi32", "navi33", "w7900"]
            )
        except Exception:
            pass

    return False


# -----------------------------------------------------------------------------
# Argument parsing
parser = argparse.ArgumentParser(description="GPT-2 training with AdamWPrune")

# Model configuration
parser.add_argument(
    "--model-name",
    type=str,
    default="gpt2",
    choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    help="GPT-2 model size",
)
parser.add_argument("--block-size", type=int, default=1024, help="Context length")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--bias", action="store_true", default=True, help="Use bias in Linear/LayerNorm"
)

# Dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="shakespeare",
    choices=["shakespeare", "finewebedu", "openwebtext"],
    help="Dataset to use",
)
parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

# Training
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size (default optimized for 24GB+ GPUs)",
)
parser.add_argument(
    "--gradient-accumulation",
    type=int,
    default=4,
    help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
)
parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override for number of epochs (alias for num-epochs)",
)
parser.add_argument(
    "--max-iters",
    type=int,
    default=10000,
    help="Maximum iterations (for better convergence)",
)
parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
parser.add_argument(
    "--decay-lr", action="store_true", default=True, help="Use learning rate decay"
)
parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")

# Optimizer
parser.add_argument(
    "--optimizer",
    type=str,
    default="adamw",
    choices=["sgd", "adam", "adamw", "adamwadv", "adamwspam", "adamwprune"],
    help="Optimizer to use",
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
    "--target-sparsity", type=float, default=0.5, help="Target sparsity"
)
parser.add_argument(
    "--pruning-warmup", type=int, default=1000, help="Pruning warmup steps"
)

# AdamWPrune specific
parser.add_argument(
    "--adamwprune-base-optimizer-name",
    type=str,
    default="adamw",
    help="Base optimizer for AdamWPrune",
)
parser.add_argument(
    "--adamwprune-beta1", type=float, default=0.9, help="AdamWPrune beta1"
)
parser.add_argument(
    "--adamwprune-beta2", type=float, default=0.999, help="AdamWPrune beta2"
)
parser.add_argument(
    "--adamwprune-weight-decay", type=float, default=0.1, help="AdamWPrune weight decay"
)
parser.add_argument(
    "--adamwprune-amsgrad", type=str, default="false", help="Use AMSGrad for AdamWPrune"
)
parser.add_argument(
    "--adamwprune-variant",
    type=str,
    default="bitter0",
    choices=[
        "bitter0",
        "bitter1",
        "bitter2",
        "bitter3",
        "bitter4",
        "bitter5",
        "bitter6",
        "bitter7",
        "bitter8",
        "bitter9",
    ],
    help="AdamWPrune variants: 0=original, 1=magnitude, 2=scale-aware, 3=grad-mag, 4=layer-adaptive, "
    "5=movement-to-zero, 6=coherence, 7=second-moment, 8=bias-corrected, 9=hybrid",
)

# SPAM configuration
parser.add_argument("--spam-theta", type=float, default=50.0, help="SPAM theta")
parser.add_argument(
    "--spam-interval", type=int, default=1000, help="SPAM reset interval"
)
parser.add_argument(
    "--spam-warmup-steps", type=int, default=100, help="SPAM warmup steps"
)
parser.add_argument(
    "--spam-enable-clip", action="store_true", help="Enable SPAM clipping"
)

# Evaluation
parser.add_argument(
    "--eval-interval", type=int, default=100, help="Evaluation interval"
)
parser.add_argument(
    "--eval-samples", type=int, default=200, help="Number of evaluation samples"
)
parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

# Output
parser.add_argument(
    "--json-output", type=str, default=None, help="Path to save training metrics JSON"
)

# System
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["float32", "float16", "bfloat16"],
    help="Data type for training",
)
parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
parser.add_argument(
    "--flash-attention", action="store_true", default=True, help="Use Flash Attention"
)

# Output
parser.add_argument(
    "--output-dir", type=str, default="gpt2/outputs", help="Output directory"
)

# Experiment tracking
parser.add_argument(
    "--tracker",
    type=str,
    default="none",
    help="Experiment tracker(s) to use (none, trackio, wandb, or comma-separated combination)",
)
parser.add_argument(
    "--tracker-project",
    type=str,
    default="adamwprune-gpt2",
    help="Project name for experiment tracker",
)
parser.add_argument(
    "--tracker-run-name",
    type=str,
    default=None,
    help="Run name for experiment tracker (auto-generated if not provided)",
)


def main():
    """Main training function."""
    args = parser.parse_args()

    # Handle epochs alias
    if args.epochs is not None:
        args.num_epochs = args.epochs

    # Environment variable overrides (highest precedence)
    # These override both command-line args and config file settings
    if "TRACKER" in os.environ:
        args.tracker = os.environ["TRACKER"]
    if "TRACKER_PROJECT" in os.environ:
        args.tracker_project = os.environ["TRACKER_PROJECT"]
    if "TRACKER_RUN_NAME" in os.environ:
        args.tracker_run_name = os.environ["TRACKER_RUN_NAME"]

    # -----------------------------------------------------------------------------
    # Setup

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup - auto-detect if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("RANK", "0") == "0":
            print("CUDA not available, falling back to CPU", flush=True)
        device = "cpu"
    else:
        device = args.device

    # Enable TensorFloat32 for matmul operations if supported
    # WMMA on AMD RDNA3+, Tensor Cores on NVIDIA Volta+
    if device == "cuda" and supports_tensorcore_fp32():
        torch.set_float32_matmul_precision("high")
        if os.environ.get("RANK", "0") == "0":
            print(
                "Enabled TensorFloat32 matmul precision (WMMA/Tensor Cores)", flush=True
            )
    elif device == "cuda":
        if os.environ.get("RANK", "0") == "0":
            print("Tensor cores/WMMA not detected, using default precision", flush=True)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    ptdtype = dtype

    # Only use autocast if on CUDA and not using float32
    if device == "cpu" or args.dtype == "float32":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

    # Fix random seeds
    torch.manual_seed(1337)
    np.random.seed(1337)

    # -----------------------------------------------------------------------------
    # Data loading

    def get_batch(split, data_dir, dataset, block_size, batch_size, device):
        """Get a batch of data."""
        # Load data
        if split == "train":
            data_path = os.path.join(data_dir, dataset, "train.bin")
        else:
            data_path = os.path.join(data_dir, dataset, "val.bin")

        data = np.memmap(data_path, dtype=np.uint16, mode="r")

        # Generate random positions
        ix = torch.randint(len(data) - block_size, (batch_size,))

        # Create batch
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )

        if device == "cuda":
            # Pin arrays for async transfer
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)

        return x, y

    # -----------------------------------------------------------------------------
    # DDP setup (if enabled)
    ddp = False
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    # Check if DDP is enabled via config
    try:
        import config as cfg

        if hasattr(cfg, "config") and hasattr(cfg.config, "GPT2_USE_DDP"):
            use_ddp = cfg.config.GPT2_USE_DDP == "y"
            ddp_backend = getattr(cfg.config, "GPT2_DDP_BACKEND", "nccl")
            ddp_find_unused = (
                getattr(cfg.config, "GPT2_DDP_FIND_UNUSED_PARAMS", "y") == "y"
            )
        else:
            use_ddp = False
            ddp_backend = "nccl"
            ddp_find_unused = True
    except ImportError:
        use_ddp = True
        ddp_backend = "nccl"
        ddp_find_unused = True

    # Initialize DDP if enabled and environment variables are set
    # DEBUG: Print environment check
    print(f"DEBUG: use_ddp={use_ddp}, RANK in env={'RANK' in os.environ}, RANK={os.environ.get('RANK', 'NOT SET')}", flush=True)
    if use_ddp and "RANK" in os.environ:
        ddp = True
        init_process_group(backend=ddp_backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        # Debug: print rank info (will appear 4 times, but helps debug)
        print(
            f"[RANK {ddp_rank}] master_process={master_process}, device={device}",
            flush=True,
        )
        if master_process:
            print(
                f"DDP initialized: {ddp_world_size} ranks, backend: {ddp_backend}",
                flush=True,
            )
    else:
        master_process = True
        seed_offset = 0
        if use_ddp:
            print(
                "DDP enabled in config but RANK environment variable not set. Running in single GPU mode.",
                flush=True,
            )

    # -----------------------------------------------------------------------------
    # Initialize experiment tracker(s) - only on master process to avoid duplicates
    trackers = set()  # Use set to track active trackers
    if master_process:
        if args.tracker != "none":
            # Parse comma-separated trackers
            tracker_names = [t.strip() for t in args.tracker.split(",")]

            # Auto-generate project name if not provided
            if not args.tracker_project:
                import hashlib

                cwd = os.getcwd()
                dir_name = os.path.basename(cwd)
                # Create a short checksum of the full path for uniqueness
                path_hash = hashlib.md5(cwd.encode()).hexdigest()[:8]
                args.tracker_project = f"{dir_name}-{path_hash}"
                print(
                    f"Auto-generated project name: {args.tracker_project}", flush=True
                )
        else:
            tracker_names = []

        if "trackio" in tracker_names:
            try:
                import trackio

                run_name = (
                    args.tracker_run_name
                    or f"gpt2_{args.optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                trackio.init(
                    project=args.tracker_project,
                    config=vars(args),
                    name=run_name,
                )
                trackers.add("trackio")
                print(
                    f"Initialized Trackio tracking for project: {args.tracker_project}",
                    flush=True,
                )
            except ImportError:
                print(
                    "Warning: trackio not installed. Install with: pip install trackio",
                    flush=True,
                )

        if "wandb" in tracker_names:
            try:
                import wandb

                run_name = (
                    args.tracker_run_name
                    or f"gpt2_{args.optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                wandb.init(
                    project=args.tracker_project,
                    config=vars(args),
                    name=run_name,
                )
                trackers.add("wandb")
                print(
                    f"Initialized WandB tracking for project: {args.tracker_project}",
                    flush=True,
                )
            except ImportError:
                print(
                    "Warning: wandb not installed. Install with: pip install wandb",
                    flush=True,
                )

    # -----------------------------------------------------------------------------
    # Model initialization

    if master_process:
        print(f"Initializing GPT-2 model: {args.model_name}", flush=True)
    config = GPTConfig.from_name(args.model_name)
    config.block_size = args.block_size
    config.dropout = args.dropout
    config.bias = args.bias

    model = GPT(config)
    model.to(device)

    # Wrap model in DDP if enabled
    if ddp:
        model = DDP(
            model, device_ids=[ddp_local_rank], find_unused_parameters=ddp_find_unused
        )

    # Compile model if requested (only compile the base model, not DDP wrapper)
    if args.compile and hasattr(torch, "compile") and not ddp:
        print("Compiling model with torch.compile()...", flush=True)
        model = torch.compile(model)

    # -----------------------------------------------------------------------------
    # Optimizer setup

    print(f"Setting up {args.optimizer} optimizer...", flush=True)
    print(f"Weight decay: {args.weight_decay}", flush=True)

    # Enable state pruning for AdamWPrune when requested
    if args.optimizer == "adamwprune" and args.pruning_method == "state":
        args.adamwprune_enable_pruning = True
        args.adamwprune_target_sparsity = args.target_sparsity
        args.adamwprune_warmup_steps = args.pruning_warmup
        args.adamwprune_ramp_end_epoch = min(8, args.num_epochs - 1)
        args.adamwprune_ramp_end_step = args.max_iters

        # Handle bitter2/bitter3/bitter4 variants: increase iterations to use saved compute
        if args.adamwprune_variant == "bitter2" and args.max_iters == 10000:
            args.max_iters = 12100
            print(
                f"Bitter2 variant: Increased max_iters to {args.max_iters} (+21%)",
                flush=True,
            )
        elif (
            args.adamwprune_variant
            in ["bitter3", "bitter4", "bitter5", "bitter6", "bitter8", "bitter9"]
            and args.max_iters == 10000
        ):
            args.max_iters = 13000
            print(
                f"{args.adamwprune_variant.capitalize()} variant: Increased max_iters to {args.max_iters} (+30%)",
                flush=True,
            )

    # Create optimizer using the library function
    optimizer, scheduler, gradient_clip_norm, spam_state, adamwprune_state = (
        create_optimizer(
            model=model,
            optimizer_type=args.optimizer,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            args=args,
            model_type="gpt2",
        )
    )

    # Pruning setup
    pruner = None
    if args.pruning_method != "none" and args.pruning_method != "state":
        print(f"Setting up {args.pruning_method} pruning...", flush=True)
        if args.pruning_method == "magnitude":
            pruner = MagnitudePruning(
                model=model,
                target_sparsity=args.target_sparsity,
                warmup_steps=args.pruning_warmup,
                ramp_end_step=args.max_iters,
            )
        elif args.pruning_method == "movement":
            pruner = MovementPruning(
                model=model,
                target_sparsity=args.target_sparsity,
                warmup_steps=args.pruning_warmup,
                pruning_frequency=50,
                ramp_end_step=args.max_iters,
            )

    # -----------------------------------------------------------------------------
    # Training loop

    @torch.no_grad()
    def evaluate(
        model, data_dir, dataset, block_size, batch_size, device, eval_samples=200
    ):
        """Evaluate the model."""
        model.eval()
        losses = []

        for _ in range(eval_samples):
            x, y = get_batch("val", data_dir, dataset, block_size, batch_size, device)
            with ctx:
                logits, loss = model(x, y)
            losses.append(loss.item())

        model.train()
        return np.mean(losses)

    @torch.no_grad()
    def measure_latency(model, seq_length, batch_size=1, num_iterations=100, warmup=10):
        """Measure inference latency at different sequence lengths."""
        model.eval()
        device = next(model.parameters()).device

        # Create dummy input
        input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=device)

        # Warmup
        for _ in range(warmup):
            with ctx:
                _ = model(input_ids)

        # Measure latency
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with ctx:
                _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]

        model.train()
        return p50, p95

    def measure_memory():
        """Measure current GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            return allocated, reserved
        return 0, 0

    def get_lr(it, warmup_steps, learning_rate, min_lr, max_iters):
        """Learning rate schedule with warmup and cosine decay."""
        # Warmup
        if it < warmup_steps:
            return learning_rate * it / warmup_steps
        # Cosine decay
        if it > max_iters:
            return min_lr
        # Cosine decay
        decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # Training metrics
    metrics = {
        "config": vars(args),
        "model_params": model.get_num_params(),
        "train_losses": [],
        "val_losses": [],
        "train_perplexities": [],
        "val_perplexities": [],
        "learning_rates": [],
        "sparsities": [],
        "timestamps": [],
        "iterations": [],
    }

    # Initialize gradient scaler for mixed precision (only for CUDA)
    if device == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    else:
        # CPU doesn't support GradScaler, use a dummy that just passes through
        class DummyScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        scaler = DummyScaler()

    if master_process:
        print(f"\nStarting training...", flush=True)
        print(f"Parameters: {model.get_num_params()/1e6:.2f}M", flush=True)
        print(f"Device: {device}, dtype: {dtype}", flush=True)
        print(
            f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation}",
            flush=True,
        )
        print(
            f"Effective batch size: {args.batch_size * args.gradient_accumulation}",
            flush=True,
        )
        print("-" * 50, flush=True)

    # Training loop
    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    running_loss = 0.0
    best_val_loss = float("inf")

    for iter_num in range(args.max_iters):

        # Determine learning rate
        if args.decay_lr:
            lr = get_lr(
                iter_num,
                args.warmup_steps,
                args.learning_rate,
                args.min_lr,
                args.max_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = args.learning_rate

        # Accumulate gradients
        for micro_step in range(args.gradient_accumulation):
            x, y = get_batch(
                "train",
                args.data_dir,
                args.dataset,
                args.block_size,
                args.batch_size,
                device,
            )

            with ctx:
                logits, loss = model(x, y)
                loss = loss / args.gradient_accumulation

            # Backward pass
            scaler.scale(loss).backward()
            running_loss += loss.item()

        # Gradient processing for special optimizers
        if args.optimizer != "sgd":
            # Only unscale if using actual CUDA scaler
            if device == "cuda":
                scaler.unscale_(optimizer)

            # Apply AdamWPrune gradient masking
            apply_adamprune_masking(optimizer, adamwprune_state)

            # Apply SPAM gradient processing
            apply_spam_gradient_processing(
                optimizer, model, spam_state, gradient_clip_norm
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Periodic SPAM momentum reset with optional warmup
        apply_periodic_spam_reset(optimizer, spam_state)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Apply masks again after optimizer step to ensure pruned weights stay zero
        if pruner is not None:
            pruner.apply_masks()

        # Update AdamWPrune state-based pruning
        if adamwprune_state is not None and adamwprune_state.get("enabled", False):
            # For GPT-2, we're using iteration-based training, not epochs
            # Pass the current iteration as the step count
            update_adamprune_masks(optimizer, adamwprune_state, None, iter_num)

        # Update pruning for external pruners
        if pruner is not None:
            pruner.update_masks(iter_num)
            pruner.apply_masks()  # Apply masks to zero out pruned weights

        # Logging (only on master process)
        if iter_num % args.log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            avg_loss = running_loss / args.log_interval
            avg_perplexity = math.exp(min(avg_loss, 20))  # Cap at 20 to avoid overflow

            # Calculate sparsity
            if pruner is not None:
                sparsity = pruner.get_sparsity()
            elif args.optimizer == "adamwprune" and args.pruning_method == "state":
                # Get sparsity from AdamWPrune masks
                if adamwprune_state is not None and "masks" in adamwprune_state:
                    total_params = 0
                    total_pruned = 0
                    for module, mask in adamwprune_state["masks"].items():
                        total_params += mask.numel()
                        total_pruned += (mask == 0).sum().item()
                    sparsity = total_pruned / total_params if total_params > 0 else 0.0
                else:
                    sparsity = 0.0
            else:
                sparsity = 0.0

            print(
                f"Iter {iter_num:5d} | loss {avg_loss:.4f} | ppl {avg_perplexity:7.2f} | "
                f"lr {lr:.2e} | sparsity {sparsity:.1%} | {dt*1000/args.log_interval:.1f}ms/iter",
                flush=True,
            )

            metrics["train_losses"].append(avg_loss)
            metrics["train_perplexities"].append(avg_perplexity)
            metrics["learning_rates"].append(lr)
            metrics["sparsities"].append(sparsity)
            metrics["iterations"].append(iter_num)
            metrics["timestamps"].append(time.time())

            # Log to experiment tracker(s)
            if "trackio" in trackers:
                import trackio

                trackio.log(
                    {
                        "iteration": iter_num,
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "sparsity": sparsity,
                    }
                )
            if "wandb" in trackers:
                import wandb

                wandb.log(
                    {
                        "iteration": iter_num,
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "sparsity": sparsity,
                    }
                )

            running_loss = 0.0

        # Evaluation (only on master process)
        if iter_num % args.eval_interval == 0 and master_process:
            val_loss = evaluate(
                model,
                args.data_dir,
                args.dataset,
                args.block_size,
                args.batch_size,
                device,
                args.eval_samples,
            )

            val_perplexity = math.exp(min(val_loss, 20))  # Cap at 20 to avoid overflow
            print(
                f"Validation loss: {val_loss:.4f} | ppl: {val_perplexity:.2f}",
                flush=True,
            )
            metrics["val_losses"].append(val_loss)
            metrics["val_perplexities"].append(val_perplexity)

            # Log validation to experiment tracker(s)
            if "trackio" in trackers:
                import trackio

                trackio.log(
                    {
                        "iteration": iter_num,
                        "val_loss": val_loss,
                        "val_perplexity": val_perplexity,
                    }
                )
            if "wandb" in trackers:
                import wandb

                wandb.log(
                    {
                        "iteration": iter_num,
                        "val_loss": val_loss,
                        "val_perplexity": val_perplexity,
                    }
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "args": args,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
                print(f"Saved best model (val_loss: {val_loss:.4f})", flush=True)

    # -----------------------------------------------------------------------------
    # Final evaluation and saving (only on master process)

    if master_process:
        print("\n" + "=" * 50, flush=True)
        print("Training complete!", flush=True)

        # Final evaluation
        final_val_loss = evaluate(
            model,
            args.data_dir,
            args.dataset,
            args.block_size,
            args.batch_size,
            device,
            args.eval_samples * 2,
        )

    final_perplexity = math.exp(min(final_val_loss, 20))
    best_perplexity = math.exp(min(best_val_loss, 20))

    # Calculate ΔPPL (change in perplexity from first to best)
    if metrics["val_perplexities"]:
        initial_perplexity = metrics["val_perplexities"][0]
        delta_ppl = best_perplexity - initial_perplexity
    else:
        initial_perplexity = float("inf")
        delta_ppl = 0.0

    print(
        f"Final validation loss: {final_val_loss:.4f} | ppl: {final_perplexity:.2f}",
        flush=True,
    )
    print(
        f"Best validation loss: {best_val_loss:.4f} | ppl: {best_perplexity:.2f}",
        flush=True,
    )
    print(f"ΔPPL (improvement): {delta_ppl:.2f}", flush=True)

    # Measure inference latency at different sequence lengths
    print("\nMeasuring inference latency...", flush=True)
    latency_results = {}
    for seq_len in [512, 1024]:
        try:
            p50, p95 = measure_latency(model, seq_len, batch_size=1, num_iterations=50)
            latency_results[f"latency_seq{seq_len}_p50"] = p50
            latency_results[f"latency_seq{seq_len}_p95"] = p95
            print(f"  Seq {seq_len}: p50={p50:.2f}ms, p95={p95:.2f}ms", flush=True)
        except Exception as e:
            print(f"  Seq {seq_len}: measurement failed - {e}", flush=True)
            latency_results[f"latency_seq{seq_len}_p50"] = -1
            latency_results[f"latency_seq{seq_len}_p95"] = -1

    # Measure final memory usage
    allocated_mb, reserved_mb = measure_memory()
    print(
        f"\nGPU Memory: {allocated_mb:.1f}MB allocated, {reserved_mb:.1f}MB reserved",
        flush=True,
    )

    # Save final model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": args.max_iters,
        "val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "config": config,
        "args": args,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "final_model.pt"))

    # Save metrics
    metrics["final_val_loss"] = final_val_loss
    metrics["best_val_loss"] = best_val_loss
    metrics["final_perplexity"] = final_perplexity
    metrics["best_perplexity"] = best_perplexity
    metrics["initial_perplexity"] = initial_perplexity
    metrics["delta_ppl"] = delta_ppl
    metrics["total_time"] = (
        time.time() - metrics["timestamps"][0] if metrics["timestamps"] else 0
    )

    # Add latency and memory metrics
    metrics.update(latency_results)
    metrics["gpu_memory_allocated_mb"] = allocated_mb
    metrics["gpu_memory_reserved_mb"] = reserved_mb

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.json_output}", flush=True)

        # Save detailed metrics
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved detailed metrics to {metrics_path}", flush=True)

        print("\nTraining complete!", flush=True)

        # Finish experiment tracking
        if "trackio" in trackers:
            import trackio

            trackio.log(
                {
                    "final_val_loss": final_val_loss,
                    "best_val_loss": best_val_loss,
                    "total_time": metrics["total_time"],
                }
            )
            trackio.finish()
            print(
                "Trackio tracking finished. Run 'trackio show' to view results.",
                flush=True,
            )
        if "wandb" in trackers:
            import wandb

            wandb.log(
                {
                    "final_val_loss": final_val_loss,
                    "best_val_loss": best_val_loss,
                    "total_time": metrics["total_time"],
                }
            )
            wandb.finish()
            print(
                "WandB tracking finished. Check your WandB dashboard for results.",
                flush=True,
            )

    # Cleanup DDP
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
