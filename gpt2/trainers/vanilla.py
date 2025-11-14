"""
Vanilla GPT-2 Trainer

Standard GPT-2 training without architectural modifications.
Supports AdamW, AdamWSPAM, and AdamWPrune optimizers.
"""

import os
import sys
import time
import math
from typing import Dict

import torch
import numpy as np

# Add parent to path for imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from gpt2.model import GPT, GPTConfig
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
    """

    def __init__(self, args, config):
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
        model = GPT(config)
        model.to(self.device)

        # Compile if requested (must be before DDP)
        if getattr(self.args, "compile", False) and hasattr(torch, "compile"):
            if self.master_process:
                print("Compiling model with torch.compile()...")
            model = torch.compile(model)

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
            self.args.adamwprune_ramp_end_step = self.args.max_iters

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
                args=self.args,
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
                    print(
                        f"\nEval @ iter {self.iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}"
                    )

                    self.metrics["val_losses"].append(losses["val"])
                    self.metrics["val_perplexities"].append(
                        math.exp(min(losses["val"], 20))
                    )

                    val_ppl = math.exp(min(losses["val"], 20))

                    # Update best perplexity
                    if val_ppl < self.best_perplexity:
                        self.best_perplexity = val_ppl

                    self.log_metrics(
                        {
                            "val_loss": losses["val"],
                            "val_perplexity": val_ppl,
                            "best_perplexity": self.best_perplexity,
                        }
                    )

                    # Save best model
                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                        if getattr(self.args, "save_checkpoint", False):
                            checkpoint_path = os.path.join(
                                getattr(self.args, "output_dir", "."), "best_model.pt"
                            )
                            self.save_checkpoint(checkpoint_path)
                            print(
                                f"Saved best model (val_loss={self.best_val_loss:.4f}, best_ppl={self.best_perplexity:.2f})"
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

            # Final evaluation
            final_losses = self.estimate_loss()
            print(
                f"Final: train {final_losses['train']:.4f}, val {final_losses['val']:.4f}"
            )

            # Save final model
            if getattr(self.args, "save_checkpoint", False):
                final_path = os.path.join(
                    getattr(self.args, "output_dir", "."), "final_model.pt"
                )
                self.save_checkpoint(final_path)
                print(f"Saved final model")

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
