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
from gpt2.mla import GPT2_MLA, GPT2_MLA_KV, MLA_Config
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

        # Track scale parameter history for stability analysis
        self.scale_history = []  # List of (iteration, scale_params) tuples
        self.max_scale_history = 10  # Keep last 10 checkpoints

    def _configure_step(self, args, step: str):
        """Configure args based on vanilla ablation step (no-op after KV tying removal)."""
        # KV tying ablation steps (V0/V1/V2) removed - no longer supported
        # This method kept for compatibility but does nothing
        pass

    def create_model(self):
        """Create GPT-2 model (standard or MLA variant)."""
        # Check if MLA is enabled via config or environment
        enable_mla = getattr(self.config, "ENABLE_MLA", False)
        mla_variant = os.environ.get("CONFIG_MLA_VARIANT") or getattr(
            self.config, "MLA_VARIANT", ""
        )

        if enable_mla and mla_variant:
            # Create MLA model
            if self.master_process:
                print(f"Initializing MLA model: {mla_variant}")

            # Get base model config
            base_config = GPTConfig.from_name(self.args.model_name)

            # Create MLA config
            config = MLA_Config(
                d_model=base_config.n_embd,
                n_heads=base_config.n_head,
                head_dim=base_config.n_embd // base_config.n_head,
                d_latent=getattr(self.config, "MLA_D_LATENT", 256),
                block_size=self.args.block_size,
                n_layers=base_config.n_layer,
                dropout=self.args.dropout,
            )

            # Create appropriate MLA model
            if "mla_kv" in mla_variant.lower():
                compression_ratio = float(
                    getattr(self.config, "MLA_COMPRESSION_RATIO", 0.5)
                )
                model = GPT2_MLA_KV(config, compression_ratio=compression_ratio)
                if self.master_process:
                    print(
                        f"  MLA+KVSplice: d_latent={config.d_latent}, "
                        f"compression_ratio={compression_ratio}"
                    )
            else:
                model = GPT2_MLA(config)
                if self.master_process:
                    print(f"  MLA: d_latent={config.d_latent}")

            model.to(self.device)

        else:
            # Create standard GPT-2 model
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
                        # Add global Fisher summaries for easier interpretation
                        from gpt2.model import aggregate_fisher_metrics

                        fisher_summaries = aggregate_fisher_metrics(fisher_metrics)
                        if fisher_summaries:
                            metrics_to_log.update(fisher_summaries)

                    # Add KV cache memory metrics
                    kv_cache_metrics = self._compute_kv_cache_metrics()
                    if kv_cache_metrics:
                        metrics_to_log.update(kv_cache_metrics)

                    # Add KVSplice parameter metrics
                    kvsplice_metrics = self._compute_kvsplice_param_metrics()
                    if kvsplice_metrics:
                        metrics_to_log.update(kvsplice_metrics)

                    # Add pruning sensitivity analysis (test reconstruction error)
                    pruning_metrics = self._compute_pruning_sensitivity()
                    if pruning_metrics:
                        metrics_to_log.update(pruning_metrics)

                    # Add scale stability tracking (convergence monitoring)
                    stability_metrics = self._compute_scale_stability()
                    if stability_metrics:
                        metrics_to_log.update(stability_metrics)

                    # Add inference benchmarks (if at benchmark interval)
                    inference_metrics = self._compute_inference_benchmarks()
                    if inference_metrics:
                        metrics_to_log.update(inference_metrics)

                    # Run lm-eval if requested
                    lm_eval_metrics = self._run_lm_eval()
                    if lm_eval_metrics:
                        metrics_to_log.update(lm_eval_metrics)

                    # Run inference benchmarks if requested
                    inference_benchmark_metrics = self._run_inference_benchmark()
                    if inference_benchmark_metrics:
                        metrics_to_log.update(inference_benchmark_metrics)

                    # Generate sample text
                    sample_outputs = self._generate_sample_text()
                    if sample_outputs and "wandb" in self.trackers:
                        import wandb

                        # Create W&B Table for sample text visibility
                        columns = ["iteration", "prompt", "generated_text"]
                        table_data = []
                        for prompt, generated in sample_outputs:
                            table_data.append([self.iter_num, prompt, generated])

                        sample_table = wandb.Table(columns=columns, data=table_data)
                        metrics_to_log["samples/generated_text"] = sample_table

                        if self.master_process:
                            print(f"\nGenerated {len(sample_outputs)} text samples")
                    elif self.master_process and sample_outputs is None:
                        print("\nWarning: Sample text generation returned None")

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
            if self.master_process and self.iter_num == self.args.eval_interval:
                print(
                    f"DEBUG: Model has no cfg attribute (type: {type(raw_model).__name__})"
                )
            return None

        cfg = raw_model.cfg
        if not hasattr(cfg, "n_layers"):
            if self.master_process and self.iter_num == self.args.eval_interval:
                print(
                    f"DEBUG: cfg has no n_layers attribute (cfg type: {type(cfg).__name__})"
                )
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

        # Detect model architecture (standard GPT-2 vs MLA)
        if hasattr(raw_model, "blocks"):
            # MLA model structure
            layers = raw_model.blocks
        elif hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
            # Standard GPT-2 structure
            layers = raw_model.transformer.h
        else:
            return None

        # Get first layer to check for KVSplice
        first_layer = layers[0]
        if not hasattr(first_layer, "attn"):
            return None

        attn = first_layer.attn
        if not hasattr(attn, "kvsplice"):
            return None

        # Collect scale/shift from ALL layers for per-layer analysis
        import torch.nn.functional as F
        import numpy as np

        all_scales = []
        all_shifts = []

        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer.attn, "kvsplice"):
                continue

            kvsplice = layer.attn.kvsplice
            scale_raw = kvsplice.transform_scale.data
            shift = kvsplice.transform_shift.data

            scale = F.softplus(scale_raw)
            all_scales.append(scale.cpu().numpy())
            all_shifts.append(shift.cpu().numpy())

        if not all_scales:
            return None

        all_scales = np.array(all_scales)  # [n_layers, d_in]
        all_shifts = np.array(all_shifts)  # [n_layers, d_in]
        n_layers = len(all_scales)

        # Compute average across layers
        avg_scale = np.mean(all_scales, axis=0)  # [d_in]
        avg_shift = np.mean(all_shifts, axis=0)  # [d_in]

        metrics = {}

        # Average scale statistics across all layers
        metrics["kvsplice/scale_mean"] = float(np.mean(avg_scale))
        metrics["kvsplice/scale_std"] = float(np.std(avg_scale))
        metrics["kvsplice/scale_min"] = float(np.min(avg_scale))
        metrics["kvsplice/scale_max"] = float(np.max(avg_scale))

        # Average shift statistics
        metrics["kvsplice/shift_mean"] = float(np.mean(avg_shift))
        metrics["kvsplice/shift_std"] = float(np.std(avg_shift))
        metrics["kvsplice/shift_min"] = float(np.min(avg_shift))
        metrics["kvsplice/shift_max"] = float(np.max(avg_shift))

        # Per-dimension tracking for first layer (sample of dimensions)
        # Track every 32nd dimension to avoid overwhelming W&B
        if len(all_scales) > 0:
            first_layer_scale = all_scales[0]
            first_layer_shift = all_shifts[0]
            d_in = len(first_layer_scale)

            # Sample dimensions (every 32nd, max 8 samples)
            sample_interval = max(1, d_in // 8)
            dim_indices = range(0, d_in, sample_interval)[:8]

            for dim_idx in dim_indices:
                metrics[f"kvsplice/dim{dim_idx}_scale"] = float(
                    first_layer_scale[dim_idx]
                )
                metrics[f"kvsplice/dim{dim_idx}_shift"] = float(
                    first_layer_shift[dim_idx]
                )

        # Per-layer statistics (track first, middle, last layers)
        layers_to_track = [0, n_layers // 2, n_layers - 1]
        layer_names = ["first", "middle", "last"]

        for layer_idx, layer_name in zip(layers_to_track, layer_names):
            if layer_idx < len(all_scales):
                layer_scale = all_scales[layer_idx]
                layer_shift = all_shifts[layer_idx]

                metrics[f"kvsplice/{layer_name}_layer_scale_mean"] = float(
                    np.mean(layer_scale)
                )
                metrics[f"kvsplice/{layer_name}_layer_shift_mean"] = float(
                    np.mean(layer_shift)
                )

                # Count low-scale dims per layer
                scale_threshold_low = 0.1
                low_scale_dims = np.sum(layer_scale < scale_threshold_low)
                low_scale_pct = 100 * low_scale_dims / len(layer_scale)
                metrics[f"kvsplice/{layer_name}_layer_low_scale_pct"] = float(
                    low_scale_pct
                )

        # Count dimensions with extreme scaling (important for pruning)
        # Use averaged scale across all layers
        scale_threshold_low = 0.1  # Nearly zero scale = unimportant dimension
        scale_threshold_high = 10.0  # Very high scale = important dimension

        low_scale_dims = int(np.sum(avg_scale < scale_threshold_low))
        high_scale_dims = int(np.sum(avg_scale > scale_threshold_high))

        low_scale_pct = 100 * low_scale_dims / len(avg_scale)
        high_scale_pct = 100 * high_scale_dims / len(avg_scale)

        metrics["kvsplice/low_scale_dims"] = low_scale_dims
        metrics["kvsplice/high_scale_dims"] = high_scale_dims
        metrics["kvsplice/low_scale_pct"] = low_scale_pct
        metrics["kvsplice/high_scale_pct"] = high_scale_pct

        # Histogram of scale values (for visualization in W&B)
        import wandb

        # Only add W&B-specific objects (Histogram, Image) when using actual W&B
        # trackio can't serialize these objects to JSON
        use_wandb_objects = "wandb" in self.trackers and "trackio" not in self.trackers

        if use_wandb_objects and self.master_process:
            try:
                # Create histogram of actual scale values (averaged across layers)
                metrics["kvsplice/scale_histogram"] = wandb.Histogram(avg_scale)
                metrics["kvsplice/shift_histogram"] = wandb.Histogram(avg_shift)

                # Create scatter plot: scale vs dimension index
                # This shows which dimensions are being scaled up/down
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Plot 1: Scale values by dimension
                dims = np.arange(len(avg_scale))
                ax1.scatter(dims, avg_scale, alpha=0.6, s=10)
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
                ax2.scatter(dims, avg_shift, alpha=0.6, s=10, color="purple")
                ax2.axhline(
                    y=0.0, color="r", linestyle="--", alpha=0.5, label="shift=0"
                )
                ax2.set_xlabel("Dimension Index")
                ax2.set_ylabel("Shift")
                ax2.set_title("KVSplice Shift by Dimension")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Log to W&B (only if using actual W&B, not trackio)
                if use_wandb_objects:
                    metrics["kvsplice/scale_shift_plot"] = wandb.Image(fig)
                plt.close(fig)

                # Create a second visualization: sorted scale values
                # This shows the "importance" distribution
                fig2, ax = plt.subplots(1, 1, figsize=(10, 5))

                sorted_scales = np.sort(avg_scale)[::-1]  # Sort descending
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

                # Log to W&B (only if using actual W&B, not trackio)
                if use_wandb_objects:
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

    def _compute_pruning_sensitivity(self):
        """
        Test reconstruction error and perplexity impact when pruning low-scale
        dimensions. This simulates pruning at inference time by temporarily
        zeroing out dimensions with scale below various thresholds.

        Returns:
            Dictionary with pruning sensitivity metrics or None if not applicable
        """
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Check if this is an MLA model with KVSplice
        if not hasattr(raw_model, "transformer"):
            return None

        # Verify all layers have KVSplice
        has_kvsplice = True
        for layer in raw_model.transformer.h:
            if not hasattr(layer, "attn") or not hasattr(layer.attn, "kvsplice"):
                has_kvsplice = False
                break

        if not has_kvsplice:
            return None

        # Get validation batch for testing
        try:
            # Create small validation batch (reuse existing data)
            batch_size = 4
            seq_len = 128
            device = next(raw_model.parameters()).device

            x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            targets = x.clone()

            # Baseline: compute loss with no pruning
            raw_model.eval()
            with torch.no_grad():
                logits, _ = raw_model(x)
                baseline_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                baseline_ppl = torch.exp(baseline_loss).item()

            metrics = {}
            metrics["pruning/baseline_ppl"] = baseline_ppl

            # Test different pruning thresholds
            thresholds = [0.05, 0.1, 0.2, 0.5]

            for thresh in thresholds:
                # Temporarily mask low-scale dimensions in all layers
                original_scales = []

                for layer in raw_model.transformer.h:
                    kvsplice = layer.attn.kvsplice
                    scale_raw = kvsplice.transform_scale.data.clone()
                    original_scales.append(scale_raw.clone())

                    # Apply softplus to get actual scale
                    scale = F.softplus(scale_raw)

                    # Create mask: zero out dimensions with scale < threshold
                    mask = (scale >= thresh).float()

                    # Temporarily modify scale parameters
                    # Set low-scale dims to very negative value so softplus -> 0
                    kvsplice.transform_scale.data[scale < thresh] = -10.0

                # Compute loss with pruned dimensions
                with torch.no_grad():
                    logits_pruned, _ = raw_model(x)
                    pruned_loss = F.cross_entropy(
                        logits_pruned.view(-1, logits_pruned.size(-1)),
                        targets.view(-1),
                    )
                    pruned_ppl = torch.exp(pruned_loss).item()

                # Restore original scales
                for layer, orig_scale in zip(raw_model.transformer.h, original_scales):
                    layer.attn.kvsplice.transform_scale.data.copy_(orig_scale)

                # Compute metrics
                ppl_increase = pruned_ppl - baseline_ppl
                ppl_increase_pct = 100 * (pruned_ppl / baseline_ppl - 1)

                # Count pruned dimensions (from first layer as proxy)
                first_scale = F.softplus(original_scales[0])
                pruned_dims = (first_scale < thresh).sum().item()
                pruned_pct = 100 * pruned_dims / first_scale.numel()

                metrics[f"pruning/thresh_{thresh}_ppl"] = pruned_ppl
                metrics[f"pruning/thresh_{thresh}_ppl_increase"] = ppl_increase
                metrics[f"pruning/thresh_{thresh}_ppl_increase_pct"] = ppl_increase_pct
                metrics[f"pruning/thresh_{thresh}_pruned_pct"] = pruned_pct

            # Determine safe pruning threshold (< 5% PPL increase)
            safe_threshold = None
            max_prunable_pct = 0.0

            for thresh in thresholds:
                ppl_increase_pct = metrics[f"pruning/thresh_{thresh}_ppl_increase_pct"]
                pruned_pct = metrics[f"pruning/thresh_{thresh}_pruned_pct"]

                if ppl_increase_pct < 5.0:  # Safe if < 5% degradation
                    safe_threshold = thresh
                    max_prunable_pct = pruned_pct

            metrics["pruning/safe_threshold"] = (
                safe_threshold if safe_threshold else 0.0
            )
            metrics["pruning/max_prunable_pct"] = max_prunable_pct

            # Print summary
            if self.master_process:
                print("\n--- Pruning Sensitivity Analysis ---")
                print(f"  Baseline PPL: {baseline_ppl:.2f}")
                for thresh in thresholds:
                    ppl = metrics[f"pruning/thresh_{thresh}_ppl"]
                    increase = metrics[f"pruning/thresh_{thresh}_ppl_increase_pct"]
                    pruned = metrics[f"pruning/thresh_{thresh}_pruned_pct"]
                    status = "✓ SAFE" if increase < 5.0 else "✗ UNSAFE"
                    print(
                        f"  Threshold {thresh}: {ppl:.2f} PPL (+{increase:.1f}%), {pruned:.0f}% pruned {status}"
                    )
                if safe_threshold:
                    print(
                        f"\n  Safe pruning: threshold={safe_threshold}, up to {max_prunable_pct:.0f}% dims"
                    )
                else:
                    print("  ⚠ No safe pruning threshold found (all degrade >5%)")

            return metrics

        except Exception as e:
            if self.master_process:
                print(f"\nWarning: Failed to compute pruning sensitivity: {e}")
            return None

    def _compute_scale_stability(self):
        """
        Compute scale parameter stability and convergence metrics.

        Tracks scale parameters over time and determines if they have
        converged (safe to start pruning).

        Returns:
            Dictionary with stability metrics or None if not applicable
        """
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Check if this is an MLA model with KVSplice
        if not hasattr(raw_model, "transformer"):
            return None

        first_layer = raw_model.transformer.h[0]
        if not hasattr(first_layer, "attn") or not hasattr(
            first_layer.attn, "kvsplice"
        ):
            return None

        # Get current scale parameters (averaged across all layers)
        scales_per_layer = []
        for layer in raw_model.transformer.h:
            kvsplice = layer.attn.kvsplice
            scale_raw = kvsplice.transform_scale.data
            scale = F.softplus(scale_raw)
            scales_per_layer.append(scale.cpu().numpy())

        # Average across layers for stability tracking
        import numpy as np

        avg_scales = np.mean(scales_per_layer, axis=0)  # [d_in]

        # Add to history
        self.scale_history.append((self.iter_num, avg_scales.copy()))

        # Keep only last N checkpoints
        if len(self.scale_history) > self.max_scale_history:
            self.scale_history.pop(0)

        metrics = {}

        # Need at least 3 checkpoints for stability analysis
        if len(self.scale_history) >= 3:
            # Extract recent scales
            recent_scales = np.array([s for _, s in self.scale_history])  # [N, d_in]

            # Compute variance across checkpoints for each dimension
            scale_variance = np.var(recent_scales, axis=0)  # [d_in]
            avg_variance = np.mean(scale_variance)
            max_variance = np.max(scale_variance)

            metrics["stability/scale_variance_mean"] = float(avg_variance)
            metrics["stability/scale_variance_max"] = float(max_variance)

            # Compute ranking stability (Spearman correlation between consecutive checkpoints)
            from scipy.stats import spearmanr

            if len(self.scale_history) >= 2:
                prev_scales = self.scale_history[-2][1]
                curr_scales = self.scale_history[-1][1]

                # Rank correlation (do important dims stay important?)
                rank_corr, _ = spearmanr(prev_scales, curr_scales)
                metrics["stability/rank_correlation"] = float(rank_corr)

            # Determine convergence status
            # Converged if: variance is low AND ranking is stable
            variance_threshold = 0.01
            rank_corr_threshold = 0.95

            is_converged = (
                avg_variance < variance_threshold
                and metrics.get("stability/rank_correlation", 0) > rank_corr_threshold
            )

            metrics["stability/converged"] = 1.0 if is_converged else 0.0

            # Estimate gradient norms on scales (proxy: change from last checkpoint)
            if len(self.scale_history) >= 2:
                prev_scales = self.scale_history[-2][1]
                curr_scales = self.scale_history[-1][1]
                delta = curr_scales - prev_scales
                avg_change = np.mean(np.abs(delta))
                max_change = np.max(np.abs(delta))

                metrics["stability/avg_scale_change"] = float(avg_change)
                metrics["stability/max_scale_change"] = float(max_change)

            # Print summary
            if self.master_process:
                print("\n--- Scale Parameter Stability ---")
                print(
                    f"  History: {len(self.scale_history)} checkpoints (last {self.max_scale_history})"
                )
                print(
                    f"  Variance: {avg_variance:.6f} (threshold: {variance_threshold})"
                )
                if "stability/rank_correlation" in metrics:
                    print(
                        f"  Rank correlation: {metrics['stability/rank_correlation']:.4f} (threshold: {rank_corr_threshold})"
                    )
                if "stability/avg_scale_change" in metrics:
                    print(f"  Avg change: {metrics['stability/avg_scale_change']:.6f}")
                print(
                    f"  Status: {'✓ CONVERGED' if is_converged else '⧗ TRAINING'} (safe to prune: {is_converged})"
                )

        else:
            # Not enough history yet
            metrics["stability/converged"] = 0.0
            if self.master_process:
                print(
                    f"\n--- Scale Parameter Stability: {len(self.scale_history)}/{max(3, self.max_scale_history)} checkpoints (need 3 min) ---"
                )

        return metrics

    def _compute_inference_benchmarks(self):
        """
        Measure actual inference performance metrics.

        Tests throughput and latency at different batch sizes to validate
        that theoretical cache compression translates to real speedup.

        Returns:
            Dictionary with inference benchmark metrics or None if not applicable
        """
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Run benchmarks only at specific intervals to avoid overhead
        # Default to 1000 iterations (same as lm_eval) if INFERENCE_BENCHMARK is enabled
        if not getattr(self.args, "inference_benchmark", False):
            return None

        benchmark_interval = getattr(self.args, "inference_benchmark_interval", 1000)
        if self.iter_num % benchmark_interval != 0:
            return None

        try:
            import time
            import numpy as np

            raw_model.eval()
            device = next(raw_model.parameters()).device

            metrics = {}

            # Test different batch sizes
            batch_sizes = [1, 4, 16]  # Skip 64 for 124M model (might OOM)
            seq_len = 128  # Fixed sequence length for benchmarking
            n_trials = 10  # Number of trials for latency measurement

            if self.master_process:
                print("\n--- Inference Benchmarks ---")

            for batch_size in batch_sizes:
                # Generate random input
                x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

                # Warmup (3 iterations to prime GPU)
                with torch.no_grad():
                    for _ in range(3):
                        _ = raw_model(x)

                # Synchronize before timing
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # Measure throughput (single long run)
                start_time = time.perf_counter()
                with torch.no_grad():
                    for _ in range(n_trials):
                        _ = raw_model(x)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time

                # Compute metrics
                total_tokens = batch_size * seq_len * n_trials
                tokens_per_sec = total_tokens / elapsed
                latency_per_token_ms = (elapsed / total_tokens) * 1000

                # Collect latency samples for percentiles
                latencies = []
                for _ in range(n_trials):
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = raw_model(x)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    latency_ms = (time.perf_counter() - start) * 1000
                    latencies.append(latency_ms)

                latency_median = np.median(latencies)
                latency_p95 = np.percentile(latencies, 95)
                latency_p99 = np.percentile(latencies, 99)

                # Log metrics
                metrics[f"inference/bs{batch_size}_tokens_per_sec"] = tokens_per_sec
                metrics[f"inference/bs{batch_size}_latency_median_ms"] = latency_median
                metrics[f"inference/bs{batch_size}_latency_p95_ms"] = latency_p95
                metrics[f"inference/bs{batch_size}_latency_p99_ms"] = latency_p99

                if self.master_process:
                    print(
                        f"  Batch size {batch_size}: {tokens_per_sec:.0f} tok/s, "
                        f"latency {latency_median:.2f}ms (p95: {latency_p95:.2f}ms)"
                    )

            # Get GPU memory bandwidth utilization during inference
            if device.type == "cuda":
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)

                    # Run inference while sampling GPU metrics
                    x = torch.randint(0, 50257, (16, 128), device=device)
                    memory_utils = []
                    compute_utils = []

                    for _ in range(10):
                        with torch.no_grad():
                            _ = raw_model(x)

                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_utils.append(util.memory)
                        compute_utils.append(util.gpu)

                    pynvml.nvmlShutdown()

                    avg_memory_util = np.mean(memory_utils)
                    avg_compute_util = np.mean(compute_utils)

                    metrics["inference/memory_bandwidth_util_pct"] = avg_memory_util
                    metrics["inference/compute_util_pct"] = avg_compute_util

                    if self.master_process:
                        print(
                            f"  GPU utilization: {avg_compute_util:.1f}% compute, "
                            f"{avg_memory_util:.1f}% memory bandwidth"
                        )

                except Exception:
                    pass  # pynvml not available or error

            raw_model.train()
            return metrics

        except Exception as e:
            if self.master_process:
                print(f"\nWarning: Failed to run inference benchmarks: {e}")
            raw_model.train()
            return None

    def _run_lm_eval(self):
        """Run lm-eval benchmarks on the model."""
        if not getattr(self.args, "run_lm_eval", False):
            return None

        # Run lm-eval only at specific intervals to avoid overhead
        # Default to 1000 iterations if not specified (eval_interval is typically 50-100)
        lm_eval_interval = getattr(self.args, "lm_eval_interval", 1000)
        if self.iter_num % lm_eval_interval != 0:
            return None

        try:
            from lm_eval import evaluator
            from lm_eval.api.model import LM
            import tiktoken
            import torch.nn.functional as F
        except ImportError as e:
            if self.master_process:
                print(f"lm-eval not available (pip install lm-eval tiktoken): {e}")
            return None

        if not self.master_process:
            return None

        print("\n--- lm-eval Benchmarks ---")

        # Get tasks from args
        tasks = getattr(self.args, "lm_eval_tasks", "hellaswag").split(",")
        tasks = [t.strip() for t in tasks]

        # Load GPT-2 tokenizer
        enc = tiktoken.get_encoding("gpt2")
        model = self.model
        device = self.device

        # Create a wrapper for our model
        class GPT2ModelWrapper(LM):
            def __init__(wrapper_self, model, device, tokenizer, block_size):
                super().__init__()
                wrapper_self._model = model
                wrapper_self._device = device
                wrapper_self._tokenizer = tokenizer
                wrapper_self._block_size = block_size
                wrapper_self.batch_size_per_gpu = 1

            @property
            def eot_token_id(wrapper_self):
                return wrapper_self._tokenizer.eot_token

            @property
            def max_length(wrapper_self):
                return wrapper_self._block_size

            @property
            def max_gen_toks(wrapper_self):
                return 256

            @property
            def batch_size(wrapper_self):
                return 1

            @property
            def device(wrapper_self):
                return wrapper_self._device

            def tok_encode(wrapper_self, string, **kwargs):
                return wrapper_self._tokenizer.encode(
                    string, allowed_special={"<|endoftext|>"}
                )

            def tok_decode(wrapper_self, tokens, **kwargs):
                return wrapper_self._tokenizer.decode(tokens)

            def _loglikelihood_tokens(wrapper_self, requests, disable_tqdm=False):
                results = []
                for context, continuation in requests:
                    ctx_tensor = torch.tensor([context], device=wrapper_self._device)
                    with torch.no_grad():
                        logits, _ = wrapper_self._model(ctx_tensor)
                    # Compute log likelihood of continuation
                    log_probs = F.log_softmax(
                        logits[0, -len(continuation) - 1 : -1], dim=-1
                    )
                    ll = sum(
                        log_probs[i, continuation[i]].item()
                        for i in range(min(len(continuation), log_probs.size(0)))
                    )
                    results.append((ll, True))
                return results

            def loglikelihood(wrapper_self, requests):
                new_reqs = []
                for req in requests:
                    context = wrapper_self.tok_encode(req.args[0])
                    continuation = wrapper_self.tok_encode(req.args[1])
                    new_reqs.append((context, continuation))
                return wrapper_self._loglikelihood_tokens(new_reqs)

            def loglikelihood_rolling(wrapper_self, requests):
                results = []
                for req in requests:
                    tokens = wrapper_self.tok_encode(req.args[0])
                    if len(tokens) < 2:
                        results.append((0.0, True))
                        continue
                    ctx = tokens[:-1]
                    cont = tokens[1:]
                    ll, _ = wrapper_self._loglikelihood_tokens([(ctx, cont)])[0]
                    results.append((ll, True))
                return results

            def generate_until(wrapper_self, requests):
                results = []
                for req in requests:
                    context = wrapper_self.tok_encode(req.args[0])
                    gen_kwargs = req.args[1]
                    max_gen = gen_kwargs.get("max_gen_toks", 100)

                    ctx_tensor = torch.tensor(
                        [context[-wrapper_self.max_length :]],
                        device=wrapper_self._device,
                    )
                    generated = []

                    with torch.no_grad():
                        for _ in range(max_gen):
                            logits, _ = wrapper_self._model(ctx_tensor)
                            next_token = logits[0, -1].argmax().item()
                            generated.append(next_token)
                            ctx_tensor = torch.cat(
                                [
                                    ctx_tensor,
                                    torch.tensor(
                                        [[next_token]], device=wrapper_self._device
                                    ),
                                ],
                                dim=1,
                            )
                            if ctx_tensor.size(1) > wrapper_self.max_length:
                                ctx_tensor = ctx_tensor[:, -wrapper_self.max_length :]

                    results.append(wrapper_self.tok_decode(generated))
                return results

        # Create wrapper and run evaluation
        try:
            wrapper = GPT2ModelWrapper(model, device, enc, self.args.block_size)

            # Get limit from config (None = all samples)
            limit = getattr(self.args, "lm_eval_limit", None)
            if limit:
                print(f"Running lm-eval with limit={limit} samples per task")

            results = evaluator.simple_evaluate(
                model=wrapper,
                tasks=tasks,
                num_fewshot=0,
                batch_size=1,
                device=str(device),
                limit=limit,
            )

            # Extract and print metrics
            lm_eval_metrics = {}
            table_rows = []  # For W&B Table

            for task_name, task_results in results.get("results", {}).items():
                row_data = {"task": task_name, "iteration": self.iter_num}
                for metric_name, value in task_results.items():
                    if isinstance(value, (int, float)) and not metric_name.endswith(
                        "_stderr"
                    ):
                        key = f"lm_eval/{task_name}_{metric_name}"
                        lm_eval_metrics[key] = value
                        row_data[metric_name] = value
                        print(f"{task_name}/{metric_name}: {value:.4f}")
                table_rows.append(row_data)

            # Create W&B Table for better visualization
            if "wandb" in self.trackers and table_rows:
                import wandb

                # Get all metric columns (excluding task and iteration)
                metric_cols = set()
                for row in table_rows:
                    metric_cols.update(
                        k for k in row.keys() if k not in ["task", "iteration"]
                    )

                # Create table with columns
                columns = ["iteration", "task"] + sorted(metric_cols)
                table_data = []
                for row in table_rows:
                    table_data.append([row.get(col, None) for col in columns])

                lm_eval_table = wandb.Table(columns=columns, data=table_data)
                lm_eval_metrics["lm_eval/results_table"] = lm_eval_table
                print(f"Created lm-eval results table with {len(table_rows)} tasks")

            return lm_eval_metrics

        except Exception as e:
            print(f"lm-eval failed: {e}")
            return None

    def _run_inference_benchmark(self):
        """Run inference benchmarks: throughput, latency, GPU utilization."""
        if not getattr(self.args, "inference_benchmark", False):
            return None

        # Run benchmarks only at specific intervals to avoid overhead
        benchmark_interval = getattr(self.args, "inference_benchmark_interval", 1000)
        if self.iter_num % benchmark_interval != 0:
            return None

        if not self.master_process:
            return None

        print("\n--- Inference Benchmarks ---")

        # Use unwrapped model for benchmarks
        model = self.raw_model if hasattr(self, "raw_model") else self.model
        model.eval()

        # Benchmark parameters
        warmup_runs = 10
        measure_runs = 50
        batch_sizes = [1, 4, 8]  # Test multiple batch sizes
        seq_len = self.args.block_size

        benchmark_results = []

        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randint(
                0, 50257, (batch_size, seq_len), device=self.device
            )

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(dummy_input)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Measure inference time
            latencies = []
            with torch.no_grad():
                for _ in range(measure_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    _ = model(dummy_input)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)

            # Calculate statistics
            import numpy as np

            latency_mean = np.mean(latencies)
            latency_std = np.std(latencies)
            latency_min = np.min(latencies)
            latency_max = np.max(latencies)

            # Throughput calculations
            tokens_per_batch = batch_size * seq_len
            throughput_tokens_sec = (tokens_per_batch / latency_mean) * 1000
            samples_per_sec = (batch_size / latency_mean) * 1000

            result = {
                "iteration": self.iter_num,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "latency_mean_ms": latency_mean,
                "latency_std_ms": latency_std,
                "latency_min_ms": latency_min,
                "latency_max_ms": latency_max,
                "throughput_tokens_sec": throughput_tokens_sec,
                "throughput_samples_sec": samples_per_sec,
            }

            benchmark_results.append(result)

            print(
                f"Batch {batch_size:2d}: {latency_mean:6.2f}±{latency_std:5.2f} ms, "
                f"{throughput_tokens_sec:8.1f} tokens/sec, {samples_per_sec:6.1f} samples/sec"
            )

        # Convert to metrics dict for logging
        inference_metrics = {}

        # Log metrics for each batch size
        for result in benchmark_results:
            bs = result["batch_size"]
            inference_metrics[f"inference/batch{bs}_latency_ms"] = result[
                "latency_mean_ms"
            ]
            inference_metrics[f"inference/batch{bs}_throughput_tokens_sec"] = result[
                "throughput_tokens_sec"
            ]
            inference_metrics[f"inference/batch{bs}_throughput_samples_sec"] = result[
                "throughput_samples_sec"
            ]

        # Create W&B Table for better visualization
        if "wandb" in self.trackers and benchmark_results:
            import wandb

            columns = [
                "iteration",
                "batch_size",
                "seq_len",
                "latency_mean_ms",
                "latency_std_ms",
                "throughput_tokens_sec",
                "throughput_samples_sec",
            ]

            table_data = []
            for result in benchmark_results:
                table_data.append([result.get(col, None) for col in columns])

            inference_table = wandb.Table(columns=columns, data=table_data)
            inference_metrics["inference/benchmark_table"] = inference_table
            print(
                f"Created inference benchmark table with {len(benchmark_results)} batch sizes"
            )

        model.train()
        return inference_metrics

    def _generate_sample_text(self):
        """
        Generate sample text from the model for W&B logging.

        Returns:
            List of (prompt, generated_text) tuples or None if generation fails
        """
        try:
            import tiktoken

            # Load tokenizer
            enc = tiktoken.get_encoding("gpt2")

            # Sample prompts to test the model
            prompts = [
                "Once upon a time",
                "The quick brown fox",
                "In a world where",
            ]

            self.model.eval()
            samples = []

            for prompt in prompts:
                # Encode prompt
                tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
                x = torch.tensor([tokens], dtype=torch.long, device=self.device)

                # Generate
                with torch.no_grad():
                    # Use model's generate method if available
                    if hasattr(self.raw_model, "generate"):
                        y = self.raw_model.generate(
                            x,
                            max_new_tokens=50,
                            temperature=0.8,
                            top_k=200,
                        )
                        generated = enc.decode(y[0].tolist())
                    else:
                        # Fallback: manual generation
                        generated_tokens = tokens.copy()
                        for _ in range(50):
                            logits, _ = self.model(x)
                            logits = logits[:, -1, :] / 0.8  # temperature
                            probs = torch.softmax(logits, dim=-1)
                            # Top-k sampling
                            topk_probs, topk_indices = torch.topk(probs[0], 200)
                            # Sample from top-k distribution
                            idx = torch.multinomial(topk_probs, 1)
                            next_token = topk_indices[idx].item()
                            generated_tokens.append(next_token)
                            # Append to sequence
                            next_token_tensor = torch.tensor(
                                [[next_token]], device=self.device
                            )
                            x = torch.cat([x, next_token_tensor], dim=1)
                        generated = enc.decode(generated_tokens)

                samples.append((prompt, generated))

            self.model.train()

            # Return list of (prompt, generated_text) tuples
            return samples

        except Exception as e:
            if self.master_process:
                print(f"Warning: Failed to generate sample text: {e}")
            return None
