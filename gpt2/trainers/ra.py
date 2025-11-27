"""
RA (Reciprocal Attention) Trainer

Trains GPT-2 Tiny with RA (double attention: Q@K.T + K@Q.T).
Supports ablation studies comparing baseline vs RA.
"""

import os
import sys
from typing import Optional

import torch

# Add parent to path for imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from ra import GPT2TinyRA
from lib.optimizers import create_optimizer
from lib.pruning import create_pruner
from .vanilla import VanillaGPT2Trainer


class RATrainer(VanillaGPT2Trainer):
    """
    Trainer for RA (Reciprocal Attention) models.

    Implements:
    - GPT2TinyRA model (6L/512d/8H with double attention)
    - Ablation support (baseline vs RA)
    - Standard training loop from VanillaGPT2Trainer
    """

    def __init__(self, args, config, ablation_step: Optional[str] = None):
        # Store ablation step before calling super().__init__
        self.ablation_step = ablation_step or "ra"

        # Don't call VanillaGPT2Trainer.__init__ yet - we need to override
        # model creation. Call BaseGPT2Trainer.__init__ directly.
        from .base import BaseGPT2Trainer

        BaseGPT2Trainer.__init__(self, args, config)

        # Initialize model (RA or baseline depending on step)
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

    def create_model(self):
        """Create RA model or baseline GPT-2 Tiny depending on ablation step."""
        if self.ablation_step == "baseline":
            # Use baseline GPT-2 Tiny (same architecture, no RA)
            if self.master_process:
                print("Creating baseline GPT-2 Tiny (6L/512d/8H, single attention)")

            from gpt2.model import GPT2, GPTConfig

            model_config = GPTConfig(
                block_size=self.args.block_size,
                vocab_size=50257,  # GPT-2 vocab size
                n_layer=6,
                n_head=8,
                n_embd=512,
                dropout=self.args.dropout,
                bias=self.args.bias,
            )
            model = GPT2(model_config)
        else:
            # Use RA model (double attention)
            if self.master_process:
                print(f"Creating GPT-2 Tiny RA (6L/512d/8H, double attention)")

            model = GPT2TinyRA(
                vocab_size=50257,
                block_size=self.args.block_size,
                n_layers=6,
                d_model=512,
                n_heads=8,
                dropout=self.args.dropout,
                bias=self.args.bias,
            )

        if self.master_process:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")

        model = model.to(self.device)

        # Compile model if enabled
        if self.args.compile_model:
            if self.master_process:
                print("Compiling model with torch.compile()...")
            model = torch.compile(model)

        # Setup DDP if applicable
        if self.ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.ddp_local_rank]
            )

        return model
