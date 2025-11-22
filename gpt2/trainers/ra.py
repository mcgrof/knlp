"""
RA Trainer

Trainer for Reciprocal Attention with compute routing.
Supports two modes: Baseline (no routing) vs RA (router enabled).

The inductive bias: route compute based on |x - E(x)| to use cheap attention
for easy tokens and full attention for hard tokens.
"""

import os
import sys
import time
import math
from typing import Optional

import torch

# Add parent to path for imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from gpt2.model import GPT, GPTConfig
from gpt2.ra_patch import patch_gpt2_with_ra, set_ra_phase
from lib.optimizers import create_optimizer
from .base import BaseGPT2Trainer


class RATrainer(BaseGPT2Trainer):
    """
    Trainer for RA with compute routing.

    Supports two steps:
      - Step 0: Baseline (all heads as FULL, no routing)
      - Step 1: RA with routing (router decides compute tier)

    Both use same code path through RABlock for fair comparison.
    """

    def __init__(self, args, config, ablation_step: Optional[str] = None):
        """
        Initialize RA trainer.

        Args:
            args: Command-line arguments
            config: Config object
            ablation_step: "0" for baseline, "1" for RA
        """
        # Configure ablation step
        self.ablation_step = ablation_step or getattr(args, "ra_step", "0")
        self._configure_step(args, self.ablation_step)

        super().__init__(args, config)

        # Initialize model
        self.model = self.create_model()
        self.raw_model = self.model.module if self.ddp else self.model

        # Initialize optimizer
        (
            self.optimizer,
            self.scheduler,
            self.gradient_clip_norm,
            self.spam_state,
            self.adamwprune_state,
        ) = self.create_optimizer()

        # Setup mixed precision
        self.setup_mixed_precision()

        # Track phase transition
        self.transitioned = False

    def _configure_step(self, args, step: str):
        """Configure args based on ablation step."""
        # RA configuration
        args.ra_head_frac = getattr(args, "ra_head_frac", 0.25)
        args.router_hidden = getattr(args, "router_hidden", 16)
        args.router_bias_full = getattr(args, "router_bias_full", -1.0)
        args.warmup_loss_drop = getattr(args, "warmup_loss_drop", 0.15)
        args.compute_penalty_weight = getattr(args, "compute_penalty_weight", 0.01)

        if step == "0":
            # Baseline: no routing
            args.enable_routing = False
            print(f"Step {step}: Baseline (no routing)")
        elif step == "1":
            # RA with routing
            args.enable_routing = True
            print(f"Step {step}: RA with routing")
        else:
            raise ValueError(f"Unknown step: {step}. Use '0' or '1'.")

    def create_model(self):
        """Create and patch GPT-2 model with RA."""
        # Create base model
        gpt_config = GPTConfig(
            block_size=self.config.GPT2_BLOCK_SIZE,
            vocab_size=self.config.TOKENIZER_VOCAB_SIZE,
            n_layer=self.config.GPT2_N_LAYER,
            n_head=self.config.GPT2_N_HEAD,
            n_embd=self.config.GPT2_N_EMBD,
        )

        model = GPT(gpt_config)

        # Patch with RA
        args = self.args
        model, self.warmup_scheduler = patch_gpt2_with_ra(
            model,
            ra_head_frac=args.ra_head_frac,
            router_hidden=args.router_hidden,
            router_bias_full=args.router_bias_full,
            warmup_loss_drop=args.warmup_loss_drop,
            enable_routing=args.enable_routing,
        )

        # Move to device
        model = model.to(self.device)

        # Compile if requested
        if getattr(self.config, "COMPILE_MODEL", False):
            print("Compiling model...")
            model = torch.compile(model)

        # Setup DDP if needed
        if self.ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
            )

        return model

    def train_step(self, batch):
        """
        Execute one training step.

        Returns:
            dict with loss, lr, and other metrics
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # Need to pass input_ids for routing
            # This requires modifying the forward to accept them
            logits, loss = self.raw_model(x, y)

            # Add compute penalty if in phase 2
            penalty = torch.tensor(0.0, device=self.device)
            if self.args.enable_routing and not self.transitioned:
                # Only compute penalty after transition
                pass
            elif self.args.enable_routing and self.transitioned:
                # Compute penalty across layers
                penalty = self._compute_penalty(x)
                loss = loss + self.args.compute_penalty_weight * penalty

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.gradient_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # LR scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss.item(),
            "penalty": penalty.item() if isinstance(penalty, torch.Tensor) else penalty,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _compute_penalty(self, x):
        """Compute total compute penalty across layers."""
        # This is a placeholder - actual implementation needs
        # access to hidden states at each layer
        return torch.tensor(0.0, device=self.device)

    def on_eval(self, eval_loss, eval_metrics):
        """
        Called after each evaluation.

        Checks for phase transition if routing is enabled.
        """
        if not self.args.enable_routing:
            return

        if self.transitioned:
            return

        # Check for transition
        if self.warmup_scheduler.should_transition(eval_loss):
            print(f"\n{'='*60}")
            print("PHASE TRANSITION: Enabling routing")
            print(f"  Eval loss: {eval_loss:.4f}")
            print(f"  Initial loss: {self.warmup_scheduler.initial_loss:.4f}")
            rel_drop = (
                self.warmup_scheduler.initial_loss - eval_loss
            ) / self.warmup_scheduler.initial_loss
            print(f"  Relative drop: {rel_drop*100:.1f}%")
            print(f"{'='*60}\n")

            set_ra_phase(self.raw_model, phase1=False)
            self.transitioned = True

    def get_run_name(self):
        """Get run name for logging."""
        step = self.ablation_step
        mode = "baseline" if step == "0" else "ra_routing"
        optimizer = getattr(self.args, "optimizer", "adamw")
        return f"gpt2_{optimizer}_{mode}"

    def get_step_description(self):
        """Get human-readable step description."""
        if self.ablation_step == "0":
            return "Baseline (no routing)"
        else:
            return f"RA with routing (ra_frac={self.args.ra_head_frac})"
