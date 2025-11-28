"""
Vanilla GPT-2 Trainer

Standard GPT-2 training without architectural modifications.
Supports AdamW, AdamWSPAM, and AdamWPrune optimizers.
"""

import os
import sys
import time
import math
from typing import Dict, Optional

import torch
import numpy as np

# Add parent to path for imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from gpt2.model import GPT2, GPTConfig
from lib.optimizers import create_optimizer
from lib.pruning import create_pruner
from .base import BaseGPT2Trainer


class VanillaGPT2Trainer(BaseGPT2Trainer):
    """
    Trainer for standard GPT-2 models.

    Implements:
    - Standard GPT-2 model initialization
    - Optimizer setup (AdamW, AdamWSPAM, AdamWPrune)
    - Standard training loop
    - Pruning evaluation (when using AdamWPrune)
    - Vanilla ablation support (KV tying, etc.)
    """

    def __init__(self, args, config, ablation_step: Optional[str] = None):
        # Configure ablation step if provided
        self.ablation_step = ablation_step or getattr(args, "vanilla_step", "V0")
        if ablation_step:
            self._configure_step(args, self.ablation_step)

        super().__init__(args, config)

        # Initialize model
        self.model = self.create_model()
        self.raw_model = self.model.module if self.ddp else self.model

        # Initialize optimizer and related components
        (
            self.optimizer,
            self.scheduler,
            self.gradient_clip_norm,
            self.spam_state,
            self.adamwprune_state,
        ) = self.create_optimizer()

        # Setup mixed precision
        self.setup_mixed_precision()

        # Setup pruner (if applicable)
        self.pruner = self.create_pruner()

        # Fetch baseline metrics if configured
        self.baseline_metrics = None
        if config is not None:
            baseline_run_id = getattr(config, "BASELINE_RUN_ID", None)
            if baseline_run_id and baseline_run_id.strip():
                self.baseline_metrics = self._fetch_baseline_metrics(baseline_run_id)

    def _configure_step(self, args, step: str):
        """Configure args based on vanilla ablation step (no-op after KV tying removal)."""
        # KV tying ablation steps (V0/V1/V2) removed - no longer supported
        # This method kept for compatibility but does nothing
        pass

    def create_model(self):
        """Create standard GPT-2 model."""
        if self.master_process:
            print(f"Initializing GPT-2 model: {self.args.model_name}")

        # Create GPT config
        config = GPTConfig.from_name(self.args.model_name)
        config.block_size = self.args.block_size
        config.dropout = self.args.dropout
        config.bias = getattr(self.args, "bias", True)

        # Create model
        model = GPT2(config)
        model.to(self.device)

        # Compile if requested (must be before DDP)
        if getattr(self.args, "compile", False) and hasattr(torch, "compile"):
            if self.master_process:
                print("Compiling model with torch.compile(dynamic=True)...")
            model = torch.compile(model, dynamic=True)

        # Wrap in DDP if needed
        model = self.wrap_model_ddp(model)

        return model

    def create_optimizer(self):
        """Create optimizer (AdamW, AdamWSPAM, or AdamWPrune)."""
        if self.master_process:
            print(f"Setting up {self.args.optimizer} optimizer...")
            print(f"Weight decay: {self.args.weight_decay}")

        # Handle state pruning for AdamWPrune
        if (
            self.args.optimizer == "adamwprune"
            and getattr(self.args, "pruning_method", "none") == "state"
        ):
            self.args.adamwprune_enable_pruning = True
            self.args.adamwprune_target_sparsity = getattr(
                self.args, "target_sparsity", 0.5
            )
            self.args.adamwprune_warmup_steps = getattr(
                self.args, "pruning_warmup", 1000
            )
            self.args.adamwprune_ramp_end_epoch = min(8, self.args.num_epochs - 1)
            self.args.adamwprune_ramp_end_step = getattr(
                self.args, "adamwprune_ramp_end_step", 3000
            )

            # Handle bitter variants
            variant = getattr(self.args, "adamwprune_variant", None)
            if variant == "bitter2" and self.args.max_iters == 10000:
                self.args.max_iters = 12100
                if self.master_process:
                    print(
                        f"Bitter2 variant: Increased max_iters to {self.args.max_iters} (+21%)"
                    )
            elif (
                variant
                in ["bitter3", "bitter4", "bitter5", "bitter6", "bitter8", "bitter9"]
                and self.args.max_iters == 10000
            ):
                self.args.max_iters = 13000
                if self.master_process:
                    print(
                        f"{variant.capitalize()} variant: Increased max_iters to {self.args.max_iters} (+30%)"
                    )

        # Create optimizer using library function
        return create_optimizer(
            model=self.model,
            optimizer_type=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            num_epochs=self.args.num_epochs,
            args=self.args,
            model_type="gpt2",
        )

    def create_pruner(self):
        """Create pruner if pruning method is specified."""
        pruning_method = getattr(self.args, "pruning_method", "none")
        if pruning_method != "none" and pruning_method != "state":
            if self.master_process:
                print(f"Setting up {pruning_method} pruning...")
            return create_pruner(
                model=self.raw_model,
                pruning_method=pruning_method,
                target_sparsity=getattr(self.args, "target_sparsity", 0.5),
                warmup_steps=getattr(self.args, "pruning_warmup", 1000),
            )
        return None

    def train(self):
        """Main training loop."""
        if self.master_process:
            print(f"\nStarting training...")
            print(f"Parameters: {self.raw_model.get_num_params()/1e6:.2f}M")
            print(f"Device: {self.device}, dtype: {self.dtype}")
            print(
                f"Batch size: {self.args.batch_size}, Gradient accumulation: {self.args.gradient_accumulation}"
            )
            print(
                f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation}"
            )
            save_enabled = getattr(self.args, "save_checkpoint", False)
            output_dir = getattr(self.args, "output_dir", ".")
            print(f"Save checkpoint: {save_enabled}, Output: {output_dir}")
            print("-" * 50)

        # Training setup
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.training_start_time = time.time()

        t0 = time.time()
        running_loss = 0.0

        # Training loop
        while self.iter_num < self.args.max_iters:
            # Check time limit
            if getattr(self.args, "max_time", 0) > 0:
                elapsed = time.time() - self.training_start_time
                if elapsed >= self.args.max_time:
                    if self.master_process:
                        print(
                            f"\nReached max training time of {self.args.max_time}s ({elapsed:.1f}s elapsed)"
                        )
                    break

            # Update learning rate
            if getattr(self.args, "decay_lr", True):
                lr = self.get_lr(self.iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = self.args.learning_rate

            # Gradient accumulation
            for micro_step in range(self.args.gradient_accumulation):
                X, Y = self.get_batch("train")

                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.args.gradient_accumulation

                # Backward
                self.scaler.scale(loss).backward()
                running_loss += loss.item()

            # Gradient processing
            if self.args.optimizer != "sgd":
                if self.device == "cuda":
                    self.scaler.unscale_(self.optimizer)

                # Apply AdamWPrune masking
                self._apply_adamprune_masking()

                # Apply SPAM processing
                self._apply_spam_processing()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Periodic SPAM reset
            self._apply_spam_reset()

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Apply pruning masks
            if self.pruner is not None:
                self.pruner.apply_masks()

            # Update pruning
            if self.adamwprune_state and self.adamwprune_state.get("enabled", False):
                self._update_adamprune_masks()

            if self.pruner is not None:
                self.pruner.update_masks(self.iter_num)
                self.pruner.apply_masks()

            # Logging
            if self.iter_num % self.args.log_interval == 0 and self.master_process:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                avg_loss = running_loss / self.args.log_interval
                avg_ppl = math.exp(min(avg_loss, 20))

                # Calculate sparsity
                sparsity = self._get_sparsity()

                print(
                    f"Iter {self.iter_num:5d} | loss {avg_loss:.4f} | ppl {avg_ppl:7.2f} | "
                    f"lr {lr:.2e} | sparsity {sparsity:.1%} | {dt*1000/self.args.log_interval:.1f}ms/iter"
                )

                # Update metrics
                self.metrics["train_losses"].append(avg_loss)
                self.metrics["train_perplexities"].append(avg_ppl)
                self.metrics["learning_rates"].append(lr)
                self.metrics["iterations"].append(self.iter_num)
                self.metrics["timestamps"].append(time.time())

                # Log to trackers
                self.log_metrics(
                    {
                        "train_loss": avg_loss,
                        "train_perplexity": avg_ppl,
                        "learning_rate": lr,
                        "sparsity": sparsity,
                    }
                )

                running_loss = 0.0

            # Evaluation
            if self.iter_num % self.args.eval_interval == 0:
                losses = self.estimate_loss()
                if self.master_process:
                    val_ppl = math.exp(min(losses["val"], 20))
                    print(
                        f"\nEval @ iter {self.iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, ppl {val_ppl:.2f}"
                    )

                    self.metrics["val_losses"].append(losses["val"])
                    self.metrics["val_perplexities"].append(val_ppl)

                    # Update best metrics
                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                    if val_ppl < self.best_perplexity:
                        self.best_perplexity = val_ppl

                    metrics_to_log = {
                        "val_loss": losses["val"],
                        "val_perplexity": val_ppl,
                    }

                    # Add baseline metrics if available
                    if self.baseline_metrics:
                        metrics_to_log.update(self.baseline_metrics)

                    # Add Fisher Information metrics if available
                    fisher_metrics = self._compute_fisher_metrics()
                    if fisher_metrics:
                        metrics_to_log.update(fisher_metrics)

                    # Add KV cache memory metrics
                    kv_cache_metrics = self._compute_kv_cache_metrics()
                    if kv_cache_metrics:
                        metrics_to_log.update(kv_cache_metrics)

                    # Add KVSplice parameter metrics
                    kvsplice_metrics = self._compute_kvsplice_param_metrics()
                    if kvsplice_metrics:
                        metrics_to_log.update(kvsplice_metrics)

                    self.log_metrics(metrics_to_log)

                    # Save best model (step-specific for ablation runs)
                    if losses["val"] < self.best_val_loss:
                        if getattr(self.args, "save_checkpoint", False):
                            output_dir = getattr(self.args, "output_dir", ".")
                            if self.ablation_step:
                                checkpoint_path = os.path.join(
                                    output_dir,
                                    f"best_model_step{self.ablation_step}.pt",
                                )
                            else:
                                checkpoint_path = os.path.join(
                                    output_dir, "best_model.pt"
                                )
                            self.save_checkpoint(checkpoint_path)
                            print(
                                f"Saved best model (val_loss={self.best_val_loss:.4f})"
                            )

            # Periodic checkpoint
            if self.should_save_checkpoint() and self.master_process:
                checkpoint_path = os.path.join(
                    getattr(self.args, "output_dir", "."),
                    f"checkpoint_iter_{self.iter_num}.pt",
                )
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint at iteration {self.iter_num}")

            self.iter_num += 1

        # Training complete
        if self.master_process:
            total_time = time.time() - self.training_start_time
            print(f"\nTraining complete! Total time: {total_time/60:.2f} minutes")

            # Skip final evaluation (estimate_loss() can hang after training ends)
            # Final metrics are already logged from the last evaluation during training

            # Log final best perplexity as summary
            if self.best_perplexity < float("inf"):
                print(
                    f"Best validation perplexity achieved: {self.best_perplexity:.2f}"
                )
                self.log_metrics(
                    {
                        "final/best_val_perplexity": self.best_perplexity,
                        "final/best_val_loss": self.best_val_loss,
                    }
                )

            # Save metrics to JSON if requested (step-specific for ablation runs)
            if hasattr(self.args, "json_output") and self.args.json_output:
                json_path = self.args.json_output
                if self.ablation_step and not json_path.endswith(
                    f"_{self.ablation_step}.json"
                ):
                    # Insert step suffix before .json extension
                    json_path = json_path.replace(
                        ".json", f"_step{self.ablation_step}.json"
                    )
                self.save_metrics_json(json_path)

            # Save final model (step-specific for ablation runs)
            save_checkpoint_val = getattr(self.args, "save_checkpoint", False)
            if save_checkpoint_val:
                output_dir = getattr(self.args, "output_dir", ".")
                os.makedirs(output_dir, exist_ok=True)

                # Save with step-specific name for ablation runs
                if self.ablation_step:
                    final_path = os.path.join(
                        output_dir, f"final_model_step{self.ablation_step}.pt"
                    )
                else:
                    final_path = os.path.join(output_dir, "final_model.pt")

                self.save_checkpoint(final_path)
                print(f"Saved final model: {final_path}")

            # Hook for subclasses to run code before trackers finish
            self.on_train_end()

            # Finish experiment tracking gracefully before exit
            if "trackio" in self.trackers:
                try:
                    import trackio

                    trackio.finish()
                    print("Trackio tracking finished")
                except Exception as e:
                    print(f"Warning: Failed to finish trackio: {e}")

            if "wandb" in self.trackers:
                try:
                    import wandb

                    wandb.finish()
                    print("W&B tracking finished")
                except Exception as e:
                    print(f"Warning: Failed to finish wandb: {e}")

    def on_train_end(self):
        """Hook for subclasses to run code before trackers finish."""
        pass

    def _apply_adamprune_masking(self):
        """Apply AdamWPrune gradient masking."""
        if self.adamwprune_state is None:
            return
        # Import helper
        try:
            from lib.optimizers import apply_adamprune_masking

            apply_adamprune_masking(self.optimizer, self.adamwprune_state)
        except ImportError:
            pass

    def _apply_spam_processing(self):
        """Apply SPAM gradient processing."""
        if self.spam_state is None:
            return
        try:
            from lib.optimizers import apply_spam_gradient_processing

            apply_spam_gradient_processing(
                self.optimizer, self.model, self.spam_state, self.gradient_clip_norm
            )
        except ImportError:
            pass

    def _apply_spam_reset(self):
        """Apply periodic SPAM momentum reset."""
        if self.spam_state is None:
            return
        try:
            from lib.optimizers import apply_periodic_spam_reset

            apply_periodic_spam_reset(self.optimizer, self.spam_state)
        except ImportError:
            pass

    def _update_adamprune_masks(self):
        """Update AdamWPrune state-based pruning masks."""
        try:
            from lib.optimizers import update_adamprune_masks

            update_adamprune_masks(
                self.optimizer, self.adamwprune_state, None, self.iter_num
            )
        except ImportError:
            pass

    def _get_sparsity(self) -> float:
        """Calculate current model sparsity."""
        if self.pruner is not None:
            return self.pruner.get_sparsity()
        elif (
            self.args.optimizer == "adamwprune"
            and getattr(self.args, "pruning_method", "none") == "state"
        ):
            if self.adamwprune_state and "masks" in self.adamwprune_state:
                total_params = 0
                total_pruned = 0
                for module, mask in self.adamwprune_state["masks"].items():
                    total_params += mask.numel()
                    total_pruned += (mask == 0).sum().item()
                return total_pruned / total_params if total_params > 0 else 0.0
        return 0.0

    def _fetch_baseline_metrics(self, baseline_run_id: str):
        """
        Fetch baseline metrics from W&B for comparison.

        Args:
            baseline_run_id: W&B run ID in format "entity/project/run_id"

        Returns:
            Dictionary with baseline metrics or None if fetch fails
        """
        try:
            import wandb

            api = wandb.Api()
            run = api.run(baseline_run_id)

            # Extract key metrics from the baseline run
            baseline_metrics = {
                "baseline/val_loss": run.summary.get("val_loss"),
                "baseline/val_perplexity": run.summary.get("val_perplexity"),
                "baseline/train_loss": run.summary.get("train_loss"),
                "baseline/train_perplexity": run.summary.get("train_perplexity"),
            }

            # Filter out None values
            baseline_metrics = {
                k: v for k, v in baseline_metrics.items() if v is not None
            }

            if self.master_process and baseline_metrics:
                print(f"\nFetched baseline metrics from {baseline_run_id}:")
                for k, v in baseline_metrics.items():
                    print(f"  {k}: {v:.4f}")

            return baseline_metrics

        except Exception as e:
            if self.master_process:
                print(f"\nWarning: Failed to fetch baseline metrics: {e}")
                print("Continuing without baseline reference...")
            return None

    def _compute_fisher_metrics(self):
        """
        Compute Fisher Information Matrix metrics for the model.

        Returns:
            Dictionary with FIM metrics or None if computation fails
        """
        if not hasattr(self.raw_model, "compute_fisher_metrics"):
            return None

        try:
            # Create a small batch for Fisher computation
            batch_size = 4
            seq_len = 128
            device = next(self.raw_model.parameters()).device

            x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

            # Compute FIM metrics
            metrics = self.raw_model.compute_fisher_metrics(x, n_samples=64, topk=8)

            return metrics if metrics else None

        except Exception as e:
            if self.master_process:
                print(f"\nWarning: Failed to compute Fisher metrics: {e}")
            return None

    def _compute_kv_cache_metrics(self):
        """
        Compute KV cache memory metrics for different sequence lengths.

        Returns:
            Dictionary with KV cache metrics or None if not applicable
        """
        # Get model config
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        if not hasattr(raw_model, "cfg"):
            return None

        cfg = raw_model.cfg
        if not hasattr(cfg, "n_layers"):
            return None

        n_layers = cfg.n_layers
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        # Test sequence lengths
        seq_lengths = [512, 1024, 2048, 4096]
        batch_size = 1

        metrics = {}

        for seq_len in seq_lengths:
            # Standard KV cache: K and V for each layer (2 bytes per fp16 element)
            standard_cache_bytes = (
                batch_size * n_layers * 2 * n_heads * seq_len * head_dim * 2
            )
            standard_cache_mb = standard_cache_bytes / 1024**2

            # Check if model uses compressed KV cache
            if hasattr(cfg, "d_latent"):
                # MLA model - cache stores latent instead of full K/V
                d_latent = cfg.d_latent

                # Check if KVSplice compression is used
                if hasattr(raw_model, "compression_ratio"):
                    compression_ratio = raw_model.compression_ratio
                    d_compressed = int(d_latent * compression_ratio)
                    compressed_cache_bytes = (
                        batch_size * n_layers * seq_len * d_compressed * 2
                    )
                    actual_cache_mb = compressed_cache_bytes / 1024**2
                    cache_type = "kvsplice"
                else:
                    # MLA without KVSplice - cache stores d_latent
                    latent_cache_bytes = batch_size * n_layers * seq_len * d_latent * 2
                    actual_cache_mb = latent_cache_bytes / 1024**2
                    cache_type = "mla"

                savings_pct = (1 - actual_cache_mb / standard_cache_mb) * 100
            else:
                # Standard GPT-2
                actual_cache_mb = standard_cache_mb
                cache_type = "standard"
                savings_pct = 0.0

            metrics[f"kv_cache/seq{seq_len}_standard_mb"] = standard_cache_mb
            metrics[f"kv_cache/seq{seq_len}_actual_mb"] = actual_cache_mb
            metrics[f"kv_cache/seq{seq_len}_savings_pct"] = savings_pct

        # Summary metrics
        if cache_type == "kvsplice":
            metrics["kv_cache/type"] = 2.0  # KVSplice
            if hasattr(raw_model, "compression_ratio"):
                metrics["kv_cache/compression_ratio"] = raw_model.compression_ratio
        elif cache_type == "mla":
            metrics["kv_cache/type"] = 1.0  # MLA
        else:
            metrics["kv_cache/type"] = 0.0  # Standard

        # Print summary (once at first eval)
        if self.master_process and self.iter_num == self.args.eval_interval:
            print("\n--- KV Cache Memory ---")
            print(f"  Cache type: {cache_type}")
            for seq_len in seq_lengths:
                actual = metrics[f"kv_cache/seq{seq_len}_actual_mb"]
                savings = metrics[f"kv_cache/seq{seq_len}_savings_pct"]
                if savings > 0:
                    print(f"  seq={seq_len}: {actual:.1f} MB ({savings:.0f}% savings)")
                else:
                    print(f"  seq={seq_len}: {actual:.1f} MB")

        return metrics

    def _compute_kvsplice_param_metrics(self):
        """
        Compute metrics on KVSplice transform_scale and transform_shift parameters.

        These parameters control the learned monotonic transform before compression:
            x_transformed = x * softplus(scale) + shift

        Returns:
            Dictionary with KVSplice parameter metrics or None if not applicable
        """
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Check if this is an MLA model with KVSplice
        if not hasattr(raw_model, "transformer"):
            return None

        # Get first layer's attention to check for KVSplice
        first_layer = raw_model.transformer.h[0]
        if not hasattr(first_layer, "attn"):
            return None

        attn = first_layer.attn
        if not hasattr(attn, "kvsplice"):
            return None

        kvsplice = attn.kvsplice

        # Extract scale and shift parameters
        scale_raw = kvsplice.transform_scale.data  # [d_in]
        shift = kvsplice.transform_shift.data  # [d_in]

        # Apply softplus to get actual scale values
        import torch.nn.functional as F

        scale = F.softplus(scale_raw)

        # Compute statistics
        metrics = {}

        # Scale statistics
        metrics["kvsplice/scale_mean"] = scale.mean().item()
        metrics["kvsplice/scale_std"] = scale.std().item()
        metrics["kvsplice/scale_min"] = scale.min().item()
        metrics["kvsplice/scale_max"] = scale.max().item()

        # Shift statistics
        metrics["kvsplice/shift_mean"] = shift.mean().item()
        metrics["kvsplice/shift_std"] = shift.std().item()
        metrics["kvsplice/shift_min"] = shift.min().item()
        metrics["kvsplice/shift_max"] = shift.max().item()

        # Count dimensions with extreme scaling (important for pruning)
        scale_threshold_low = 0.1  # Nearly zero scale = unimportant dimension
        scale_threshold_high = 10.0  # Very high scale = important dimension

        low_scale_dims = (scale < scale_threshold_low).sum().item()
        high_scale_dims = (scale > scale_threshold_high).sum().item()

        low_scale_pct = 100 * low_scale_dims / scale.numel()
        high_scale_pct = 100 * high_scale_dims / scale.numel()

        metrics["kvsplice/low_scale_dims"] = low_scale_dims
        metrics["kvsplice/high_scale_dims"] = high_scale_dims
        metrics["kvsplice/low_scale_pct"] = low_scale_pct
        metrics["kvsplice/high_scale_pct"] = high_scale_pct

        # Histogram of scale values (for visualization in W&B)
        import wandb

        if "wandb" in self.trackers and self.master_process:
            try:
                # Create histogram of actual scale values
                scale_np = scale.cpu().numpy()
                shift_np = shift.cpu().numpy()

                metrics["kvsplice/scale_histogram"] = wandb.Histogram(scale_np)
                metrics["kvsplice/shift_histogram"] = wandb.Histogram(shift_np)

                # Create scatter plot: scale vs dimension index
                # This shows which dimensions are being scaled up/down
                import matplotlib.pyplot as plt
                import numpy as np

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Plot 1: Scale values by dimension
                dims = np.arange(len(scale_np))
                ax1.scatter(dims, scale_np, alpha=0.6, s=10)
                ax1.axhline(
                    y=1.0,
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label="scale=1 (identity)",
                )
                ax1.axhline(
                    y=scale_threshold_low,
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    label=f"low (<{scale_threshold_low})",
                )
                ax1.axhline(
                    y=scale_threshold_high,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    label=f"high (>{scale_threshold_high})",
                )
                ax1.set_xlabel("Dimension Index")
                ax1.set_ylabel("Scale (softplus)")
                ax1.set_title("KVSplice Scale by Dimension")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Shift values by dimension
                ax2.scatter(dims, shift_np, alpha=0.6, s=10, color="purple")
                ax2.axhline(
                    y=0.0, color="r", linestyle="--", alpha=0.5, label="shift=0"
                )
                ax2.set_xlabel("Dimension Index")
                ax2.set_ylabel("Shift")
                ax2.set_title("KVSplice Shift by Dimension")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Log to W&B
                metrics["kvsplice/scale_shift_plot"] = wandb.Image(fig)
                plt.close(fig)

                # Create a second visualization: sorted scale values
                # This shows the "importance" distribution
                fig2, ax = plt.subplots(1, 1, figsize=(10, 5))

                sorted_scales = np.sort(scale_np)[::-1]  # Sort descending
                ax.plot(sorted_scales, linewidth=2)
                ax.axhline(
                    y=1.0,
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label="scale=1 (identity)",
                )
                ax.axhline(
                    y=scale_threshold_low,
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    label=f"low (<{scale_threshold_low})",
                )
                ax.fill_between(
                    range(len(sorted_scales)),
                    0,
                    sorted_scales,
                    where=(sorted_scales < scale_threshold_low),
                    alpha=0.3,
                    color="orange",
                    label=f"Prunable ({low_scale_dims} dims)",
                )
                ax.set_xlabel("Dimension (sorted by importance)")
                ax.set_ylabel("Scale (softplus)")
                ax.set_title(
                    f"KVSplice Dimension Importance ({low_scale_pct:.1f}% prunable)"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                metrics["kvsplice/importance_plot"] = wandb.Image(fig2)
                plt.close(fig2)

            except Exception as e:
                if self.master_process:
                    print(f"\nWarning: Failed to create KVSplice visualizations: {e}")

        # Print summary (once at first eval)
        if self.master_process and self.iter_num == self.args.eval_interval:
            print("\n--- KVSplice Parameters ---")
            print(
                f"  Scale: {metrics['kvsplice/scale_mean']:.3f} ± {metrics['kvsplice/scale_std']:.3f}"
            )
            print(
                f"         range=[{metrics['kvsplice/scale_min']:.3f}, {metrics['kvsplice/scale_max']:.3f}]"
            )
            print(
                f"  Shift: {metrics['kvsplice/shift_mean']:.3f} ± {metrics['kvsplice/shift_std']:.3f}"
            )
            print(
                f"         range=[{metrics['kvsplice/shift_min']:.3f}, {metrics['kvsplice/shift_max']:.3f}]"
            )
            print(
                f"  Low-scale dims (<{scale_threshold_low}): {low_scale_dims} ({metrics['kvsplice/low_scale_pct']:.1f}%)"
            )
            print(
                f"  High-scale dims (>{scale_threshold_high}): {high_scale_dims} ({metrics['kvsplice/high_scale_pct']:.1f}%)"
            )

        return metrics
