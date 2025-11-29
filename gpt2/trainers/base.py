"""
Base GPT-2 Trainer

Provides common functionality for all GPT-2 training variants.
Subclasses implement specific architecture initialization and
training loop customization.
"""

import os
import sys
import time
import math
import json
from typing import Dict, Tuple, Optional
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
from torch.distributed import init_process_group, destroy_process_group

# Add parent directory to path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from gpt2.model import GPT2, GPTConfig


class BaseGPT2Trainer:
    """
    Base trainer for GPT-2 variants.

    Provides common functionality:
    - Data loading (get_batch)
    - Learning rate scheduling
    - Loss estimation
    - Checkpoint management
    - DDP setup
    - Metrics tracking

    Subclasses override:
    - create_model(): Model initialization
    - create_optimizer(): Optimizer setup
    - train_step(): Single training iteration
    - should_save_checkpoint(): Custom checkpoint logic
    """

    def __init__(self, args, config):
        """
        Initialize base trainer.

        Args:
            args: Parsed command-line arguments
            config: Config object from config.py
        """
        self.args = args
        self.config = config

        # Device setup
        self.setup_device()

        # DDP setup (if applicable)
        self.setup_ddp()

        # Data loading
        self.setup_data()

        # Tracker setup
        self.trackers = set()
        if self.master_process:
            self.setup_trackers()

        # Model and optimizer (to be created by subclass)
        self.model = None
        self.raw_model = None  # Non-DDP wrapped model
        self.optimizer = None
        self.scaler = None
        self.scheduler = None

        # Training state
        self.iter_num = 0
        self.best_val_loss = float("inf")
        self.best_perplexity = float("inf")
        self.training_start_time = None

        # Metrics
        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "train_perplexities": [],
            "val_perplexities": [],
            "learning_rates": [],
            "timestamps": [],
            "iterations": [],
        }

    def setup_device(self):
        """Setup device (CPU/CUDA) and dtype."""
        self.device = self.args.device
        self.dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )

        # Set up precision context
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type=self.device, dtype=self.ptdtype)
        )

    def setup_ddp(self):
        """Setup Distributed Data Parallel or FSDP if applicable."""
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        # Check if FSDP is requested (via config or environment)
        self.use_fsdp = (
            getattr(self.config, "USE_FSDP", False) if self.config else False
        ) or (os.environ.get("USE_FSDP", "0") == "1")

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert (
                self.args.gradient_accumulation % self.ddp_world_size == 0
            ), f"gradient_accumulation ({self.args.gradient_accumulation}) must be divisible by world_size ({self.ddp_world_size})"
            self.args.gradient_accumulation //= self.ddp_world_size

            if self.master_process:
                mode = "FSDP" if self.use_fsdp else "DDP"
                print(f"Initialized {mode}: rank {self.ddp_rank}/{self.ddp_world_size}")
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.use_fsdp = False

    def setup_data(self):
        """Setup data loading (to be implemented by subclass if needed)."""
        pass

    def setup_trackers(self):
        """Setup experiment tracking (trackio, wandb)."""
        if not hasattr(self.args, "tracker") or self.args.tracker == "none":
            return

        # Parse comma-separated trackers
        tracker_names = [t.strip() for t in self.args.tracker.split(",")]

        # Auto-generate project name if not provided
        if not hasattr(self.args, "tracker_project") or not self.args.tracker_project:
            import hashlib

            cwd = os.getcwd()
            dir_name = os.path.basename(cwd)
            path_hash = hashlib.md5(cwd.encode()).hexdigest()[:8]
            self.args.tracker_project = f"{dir_name}-{path_hash}"
            if self.master_process:
                print(f"Auto-generated project name: {self.args.tracker_project}")

        if "trackio" in tracker_names:
            try:
                import trackio
                import shutil
                from pathlib import Path

                # Clear Trackio cache for this specific project
                trackio_cache = Path.home() / ".cache" / "huggingface" / "trackio"
                project_cache = trackio_cache / self.args.tracker_project
                if project_cache.exists():
                    if self.master_process:
                        print(f"Clearing Trackio project cache: {project_cache}")
                    shutil.rmtree(project_cache, ignore_errors=True)

                run_name = (
                    getattr(self.args, "tracker_run_name", None)
                    or f"gpt2_{self.args.optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                # Use resume='allow' to reuse existing run if present, avoiding duplicate warnings
                trackio.init(
                    project=self.args.tracker_project,
                    config=vars(self.args),
                    name=run_name,
                    resume="allow",
                )
                self.trackers.add("trackio")
                print(
                    f"Initialized Trackio tracking for project: {self.args.tracker_project}"
                )
            except ImportError:
                print(
                    "Warning: trackio not installed. Install with: pip install trackio"
                )

        if "wandb" in tracker_names:
            try:
                import wandb

                # Import weave to suppress wandb warning about LLM tracing
                try:
                    import weave
                except ImportError:
                    pass

                run_name = (
                    getattr(self.args, "tracker_run_name", None)
                    or f"gpt2_{self.args.optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                wandb.init(
                    project=self.args.tracker_project,
                    config=vars(self.args),
                    name=run_name,
                )
                self.trackers.add("wandb")
                print(
                    f"Initialized WandB tracking for project: {self.args.tracker_project}"
                )
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.

        Args:
            split: 'train' or 'val'

        Returns:
            (x, y) tensors
        """
        # Load data
        if split == "train":
            data_path = os.path.join(self.args.data_dir, self.args.dataset, "train.bin")
        else:
            data_path = os.path.join(self.args.data_dir, self.args.dataset, "val.bin")

        data = np.memmap(data_path, dtype=np.uint16, mode="r")

        # Generate random positions
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))

        # Create batch
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.args.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.args.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if self.device == "cuda":
            # Pin arrays for async transfer
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y

    def get_lr(self, it: int) -> float:
        """
        Compute learning rate for iteration.

        Args:
            it: Current iteration number

        Returns:
            Learning rate
        """
        # Linear warmup
        if it < self.args.warmup_steps:
            return self.args.learning_rate * it / self.args.warmup_steps
        # Cosine decay
        if it > self.args.max_iters:
            return self.args.min_lr
        decay_ratio = (it - self.args.warmup_steps) / (
            self.args.max_iters - self.args.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Estimate loss on train and val splits.

        Returns:
            Dictionary with 'train' and 'val' losses
        """
        out = {}
        # Unwrap compiled model for eval (torch.compile can have issues in eval mode)
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.args.eval_samples)
            for k in range(self.args.eval_samples):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = model(X, Y)
                loss_val = loss.item()
                if math.isnan(loss_val):
                    print(f"Warning: NaN loss in {split} eval sample {k}")
                    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
                    print(f"  X range: [{X.min()}, {X.max()}]")
                losses[k] = loss_val
            out[split] = losses.mean().item()
        model.train()
        return out

    def save_checkpoint(self, checkpoint_path: str):
        """
        Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.args,
        }
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_num = checkpoint["iter_num"]
        self.best_val_loss = checkpoint["best_val_loss"]
        if "scaler" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

    # Abstract methods to be implemented by subclasses

    def create_model(self) -> nn.Module:
        """Create and return the model. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement create_model()")

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create and return the optimizer. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement create_optimizer()")

    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            X: Input tensor
            Y: Target tensor

        Returns:
            Dictionary with metrics (at minimum 'loss')
        """
        raise NotImplementedError("Subclass must implement train_step()")

    def should_save_checkpoint(self) -> bool:
        """
        Determine if checkpoint should be saved at current iteration.

        Returns:
            True if checkpoint should be saved
        """
        if not hasattr(self.args, "checkpoint_interval"):
            return False
        return self.iter_num % self.args.checkpoint_interval == 0

    def wrap_model_ddp(self, model: nn.Module) -> nn.Module:
        """
        Wrap model in DDP or FSDP if distributed training is enabled.

        Args:
            model: Model to wrap

        Returns:
            DDP/FSDP-wrapped model or original model
        """
        if self.ddp:
            if self.use_fsdp:
                # FSDP wrapping - auto-wrap transformer blocks
                # Determine transformer block class based on model type
                try:
                    # Try to get the block class from model
                    if hasattr(model, "transformer") and hasattr(
                        model.transformer, "h"
                    ):
                        # Standard GPT-2 style: transformer.h[i] are blocks
                        block_class = type(model.transformer.h[0])
                    elif hasattr(model, "blocks"):
                        # MLA style: blocks[i]
                        block_class = type(model.blocks[0])
                    else:
                        block_class = None

                    if block_class and self.master_process:
                        print(f"FSDP auto-wrap policy: {block_class.__name__}")

                    # Create auto-wrap policy
                    auto_wrap_policy = (
                        functools.partial(
                            transformer_auto_wrap_policy,
                            transformer_layer_cls={block_class},
                        )
                        if block_class
                        else None
                    )

                    # FSDP configuration
                    model = FSDP(
                        model,
                        auto_wrap_policy=auto_wrap_policy,
                        device_id=self.ddp_local_rank,
                        # cpu_offload=CPUOffload(offload_params=True),  # Uncomment for CPU offload
                    )

                    if self.master_process:
                        print("Model wrapped with FSDP")

                except Exception as e:
                    if self.master_process:
                        print(
                            f"Warning: FSDP wrapping failed ({e}), falling back to DDP"
                        )
                    self.use_fsdp = False

            if not self.use_fsdp:
                # DDP wrapping
                find_unused = getattr(self.args, "ddp_find_unused_params", False)
                model = DDP(
                    model,
                    device_ids=[self.ddp_local_rank],
                    find_unused_parameters=find_unused,
                )
                if self.master_process:
                    print("Model wrapped with DDP")

        return model

    def setup_mixed_precision(self):
        """Setup mixed precision training scaler."""
        if self.device == "cuda":
            enabled = self.dtype == "float16"  # Only for FP16, not BF16
            self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        else:
            # CPU doesn't support GradScaler
            class DummyScaler:
                def scale(self, loss):
                    return loss

                def unscale_(self, optimizer):
                    pass

                def step(self, optimizer):
                    optimizer.step()

                def update(self):
                    pass

                def state_dict(self):
                    return {}

                def load_state_dict(self, state_dict):
                    pass

            self.scaler = DummyScaler()

    def _collect_aggregate_gpu_metrics(self):
        """
        Collect aggregate GPU metrics (averaged across all visible GPUs).

        Returns:
            Dictionary with aggregate metrics or empty dict if unavailable
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            # Collect metrics from all GPUs
            memory_utils = []
            compute_utils = []
            memory_used = []
            memory_total = []

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Utilization rates (returns utilization object with gpu and memory fields)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    compute_utils.append(util.gpu)
                    memory_utils.append(util.memory)

                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used.append(mem_info.used / (1024**3))  # GB
                    memory_total.append(mem_info.total / (1024**3))  # GB

                except pynvml.NVMLError:
                    continue

            pynvml.nvmlShutdown()

            # Compute aggregates
            agg_metrics = {}
            if compute_utils:
                agg_metrics["gpu/compute_util_avg"] = sum(compute_utils) / len(
                    compute_utils
                )
            if memory_utils:
                agg_metrics["gpu/memory_util_avg"] = sum(memory_utils) / len(
                    memory_utils
                )
            if memory_used:
                agg_metrics["gpu/memory_used_avg_gb"] = sum(memory_used) / len(
                    memory_used
                )
                agg_metrics["gpu/memory_used_total_gb"] = sum(memory_used)
            if memory_total:
                agg_metrics["gpu/memory_total_gb"] = memory_total[
                    0
                ]  # Same for all GPUs

            return agg_metrics

        except (ImportError, Exception):
            # pynvml not available or error occurred
            return {}

    def log_metrics(self, metrics_dict: Dict[str, float]):
        """
        Log metrics to configured trackers.

        Args:
            metrics_dict: Dictionary of metric name -> value
        """
        if not self.master_process:
            return

        # Add iteration number
        metrics_dict["iteration"] = self.iter_num

        # Add aggregate GPU metrics
        gpu_metrics = self._collect_aggregate_gpu_metrics()
        metrics_dict.update(gpu_metrics)

        # Sanitize metrics: convert tensors to Python scalars
        # Separate wandb-specific objects (Tables) from JSON-serializable metrics
        sanitized_metrics = {}
        wandb_only_metrics = {}

        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                sanitized_metrics[key] = value.item()
            else:
                # Check if this is a wandb-specific type (Table, Image, etc.)
                type_name = type(value).__name__
                if type_name in ("Table", "Image", "Video", "Audio", "Html"):
                    # These are W&B-specific types, don't send to trackio
                    wandb_only_metrics[key] = value
                else:
                    sanitized_metrics[key] = value

        # Log to trackio (only JSON-serializable metrics)
        if "trackio" in self.trackers:
            try:
                import trackio

                trackio.log(sanitized_metrics)
            except Exception as e:
                print(f"Warning: Failed to log to trackio: {e}")

        # Log to wandb (all metrics including Tables)
        if "wandb" in self.trackers:
            try:
                import wandb

                # Combine both metric types for wandb
                all_wandb_metrics = {**sanitized_metrics, **wandb_only_metrics}
                wandb.log(all_wandb_metrics, step=self.iter_num)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

    def _sanitize_for_json(self, obj):
        """
        Recursively convert PyTorch Tensors to Python scalars for JSON
        serialization.

        Args:
            obj: Object to sanitize (can be dict, list, Tensor, or scalar)

        Returns:
            JSON-serializable version of obj
        """
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        else:
            return obj

    def save_metrics_json(self, output_path: str):
        """
        Save training metrics to JSON file.

        Args:
            output_path: Path to save JSON file
        """
        if not self.master_process:
            return

        metrics_data = {
            "final_iteration": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "metrics": self.metrics,
        }

        # Sanitize all values to ensure JSON serializability
        sanitized_data = self._sanitize_for_json(metrics_data)

        try:
            with open(output_path, "w") as f:
                json.dump(sanitized_data, f, indent=2)
            print(f"Metrics saved to {output_path}")
        except Exception as e:
            print(f"Warning: Failed to save metrics to {output_path}: {e}")

    def run_dry_run(self, exit_on_completion=True):
        """
        Dry-run validation: tests architecture with minimal forward/backward
        pass.

        Args:
            exit_on_completion: If True, exits with status 0 if successful, 1 if error.
                               If False, returns status instead (for ablation mode).

        Returns:
            If exit_on_completion=False, returns 0 on success, 1 on error.
        """
        if not self.master_process:
            return 0 if not exit_on_completion else None

        print("\n" + "=" * 60)
        print("DRY-RUN MODE: Architecture Validation")
        print("=" * 60)
        print(f"  Model: {self.args.model_name}")
        print(f"  Parameters: {self.raw_model.get_num_params() / 1e6:.2f}M")
        print(f"  Device: {self.device}")

        # Create minimal dummy batch (batch_size=2, seq_len=32)
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)
        y = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)

        print(f"  Dummy batch: {batch_size}x{seq_len}")

        try:
            # Forward pass
            print("  ✓ Testing forward pass...")
            with self.ctx:
                logits, loss = self.model(x, y)
            print(f"    Output shape: {logits.shape}, Loss: {loss.item():.4f}")

            # Backward pass
            print("  ✓ Testing backward pass...")
            self.scaler.scale(loss).backward()
            print("    Gradients computed")

            # Optimizer step
            print("  ✓ Testing optimizer step...")
            if self.device == "cuda":
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            print("    Parameters updated")

            # Success
            print("\n" + "=" * 60)
            print("✓ DRY-RUN PASSED: Architecture is valid")
            print("=" * 60 + "\n")

            if exit_on_completion:
                sys.exit(0)
            else:
                return 0

        except Exception as e:
            print("\n" + "=" * 60)
            print("✗ DRY-RUN FAILED: Architecture validation error")
            print("=" * 60)
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

            if exit_on_completion:
                sys.exit(1)
            else:
                return 1

    def run_batch_overfit_sanity_check(
        self,
        max_steps: int = 100,
        batch_size: int = 8,
        print_every: int = 25,
        log_to_wandb: bool = True,
    ):
        """
        Single-batch overfit sanity test.

        Trains on ONE batch for max_steps to verify:
        1. Model can learn (loss decreases)
        2. Gradients flow correctly
        3. W&B logging works
        4. No obvious bugs in forward/backward pass

        This catches issues before wasting hours on full training.

        Args:
            max_steps: Number of steps to train on single batch
            batch_size: Batch size for sanity test (keep small for speed)
            print_every: Print progress every N steps
            log_to_wandb: Whether to log metrics to W&B

        Returns:
            True if sanity check passed, False otherwise
        """
        if not self.master_process:
            return True  # Only run on master process

        print("\n" + "=" * 80)
        print("BATCH OVERFIT SANITY CHECK")
        print("=" * 80)
        print(
            f"Training on ONE batch for {max_steps} steps to verify model can learn..."
        )
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}, dtype: {self.dtype}")

        # Get one batch and reuse it
        original_batch_size = self.args.batch_size
        self.args.batch_size = batch_size  # Temporarily override
        X, Y = self.get_batch("train")
        self.args.batch_size = original_batch_size  # Restore

        print(f"Batch shape: X={tuple(X.shape)}, Y={tuple(Y.shape)}\n")

        # Put model in train mode
        self.model.train()

        # Create temporary optimizer with no weight decay (easier to overfit)
        temp_params = self.model.parameters()
        temp_optimizer = torch.optim.AdamW(
            temp_params, lr=3e-4, weight_decay=0.0, betas=(0.9, 0.95)
        )

        loss_history = []
        start_time = time.time()

        for step in range(1, max_steps + 1):
            temp_optimizer.zero_grad(set_to_none=True)

            # Forward pass with autocast
            with self.ctx:
                logits, loss = self.model(X, Y)

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(temp_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                temp_optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Print progress
            if step % print_every == 0 or step == 1:
                avg_loss = sum(loss_history[-print_every:]) / min(
                    len(loss_history), print_every
                )
                print(
                    f"Step {step:4d}/{max_steps} | loss={loss_val:.4f} | avg={avg_loss:.4f}"
                )

            # Log to W&B if enabled
            if log_to_wandb and "wandb" in self.trackers:
                try:
                    import wandb

                    wandb.log(
                        {
                            "sanity/step": step,
                            "sanity/loss": loss_val,
                            "sanity/avg_loss": (
                                avg_loss if step % print_every == 0 else None
                            ),
                        }
                    )
                except Exception as e:
                    print(f"Warning: Failed to log to W&B: {e}")

        elapsed = time.time() - start_time

        # Evaluate results
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        reduction = (initial_loss - final_loss) / initial_loss * 100

        print("\n" + "=" * 80)
        print("SANITY CHECK RESULTS")
        print("=" * 80)
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss:   {final_loss:.4f}")
        print(f"Reduction:    {reduction:.1f}%")
        print(f"Time:         {elapsed:.1f}s")

        # Verdict
        passed = final_loss < 0.3 * initial_loss
        if passed:
            print(
                "\n✅ PASSED: Loss dropped significantly - model/gradients OK, safe to train!"
            )
        else:
            print(
                "\n❌ FAILED: Loss did NOT drop enough - check model, loss, optimizer, etc."
            )
            print(
                "   Possible issues: wrong loss function, no gradients, bad init, etc."
            )
            print("   DO NOT PROCEED with full training until this is fixed!")

        print("=" * 80 + "\n")

        # Restore model to clean state (reset any changed parameters)
        # This is best-effort; optimizer state is separate from model
        del temp_optimizer
        self.model.train()

        return passed

    def train(self):
        """
        Main training loop. Must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement train()")
