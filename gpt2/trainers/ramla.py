"""
RAMLA Trainer - RA + MLA Learning Rate Ablation

Handles all architecture variants for the LR ablation study:
  B0, B1       - Baseline GPT-2 at 6e-4 / 1.2e-3
  MLA0, MLA1   - MLA at 6e-4 / 1.2e-3
  RA0, RA1     - RA routing at 6e-4 / 1.2e-3
  RAMLA0, RAMLA1 - RA+MLA at 6e-4 / 1.2e-3
  RAMLAKV0, RAMLAKV1 - RA+MLA+KVSplice at 6e-4 / 1.2e-3
"""

import sys
import os

# Add parent to path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from typing import Optional, Dict
from gpt2.trainers.base import VanillaGPT2Trainer
from gpt2.trainers.ra import RATrainer


# Learning rates for ablation
LR_STANDARD = 6e-4
LR_AGGRESSIVE = 1.2e-3


def parse_step(step: str) -> Dict:
    """
    Parse ablation step name into configuration.

    Returns:
        dict with keys: arch, lr, lr_name
    """
    step = step.upper()

    # Determine learning rate from suffix
    if step.endswith("1"):
        lr = LR_AGGRESSIVE
        lr_name = "aggressive"
        base = step[:-1]
    elif step.endswith("0"):
        lr = LR_STANDARD
        lr_name = "standard"
        base = step[:-1]
    else:
        raise ValueError(f"Step must end with 0 or 1: {step}")

    # Determine architecture from prefix
    if base == "B":
        arch = "baseline"
    elif base == "MLA":
        arch = "mla"
    elif base == "RA":
        arch = "ra"
    elif base == "RAMLA":
        arch = "ramla"
    elif base == "RAMLAKV":
        arch = "ramlakv"
    else:
        raise ValueError(f"Unknown architecture prefix: {base}")

    return {
        "arch": arch,
        "lr": lr,
        "lr_name": lr_name,
    }


class RAMLATrainer(VanillaGPT2Trainer):
    """
    Trainer for RA-MLA LR ablation study.

    Dynamically selects architecture based on step name and configures
    appropriate learning rate.
    """

    def __init__(self, args, config, ablation_step: str = "B0"):
        self.ablation_step = ablation_step
        self.step_config = parse_step(ablation_step)

        # Override learning rate based on step
        args.learning_rate = self.step_config["lr"]

        print(
            f"Step {ablation_step}: {self.step_config['arch']} architecture, "
            f"{self.step_config['lr_name']} LR ({self.step_config['lr']:.1e})"
        )

        # Initialize based on architecture
        if self.step_config["arch"] == "baseline":
            # Standard GPT-2
            super().__init__(args, config)

        elif self.step_config["arch"] == "ra":
            # RA routing - delegate to RATrainer
            # We need to use composition here since RATrainer has different model
            self._ra_trainer = RATrainer(args, config, ablation_step="1")
            self.model = self._ra_trainer.model
            self.optimizer = self._ra_trainer.optimizer
            self.scheduler = self._ra_trainer.scheduler
            self.ctx = self._ra_trainer.ctx
            self.scaler = self._ra_trainer.scaler
            self.args = args
            self.config = config
            self.trackers = self._ra_trainer.trackers

        elif self.step_config["arch"] in ["mla", "ramla", "ramlakv"]:
            # MLA-based architectures
            super().__init__(args, config)
            # Replace model with MLA variant
            self._setup_mla_model()

        else:
            raise ValueError(f"Unknown architecture: {self.step_config['arch']}")

    def _setup_mla_model(self):
        """Replace model with MLA/RAMLA/RAMLAKV variant."""
        import torch
        from ra import RA_MLA_Config, MLAGPT, RAMLAGPT, RAMLAKV_GPT

        arch = self.step_config["arch"]

        # Get MLA config from args or defaults
        d_latent = getattr(self.args, "mla_d_latent", 256)
        compression_ratio = getattr(self.args, "mla_compression_ratio", 0.5)

        cfg = RA_MLA_Config(
            d_model=768,
            n_heads=12,
            head_dim=64,
            d_latent=d_latent,
            block_size=self.args.block_size,
            n_layers=12,
        )

        # Create appropriate full GPT model
        if arch == "mla":
            self.model = MLAGPT(cfg)
            print(f"Created MLAGPT with d_latent={d_latent}")
        elif arch == "ramla":
            self.model = RAMLAGPT(cfg)
            print(f"Created RAMLAGPT with d_latent={d_latent}")
            print(
                f"  Layer directions: {self.model.get_alternation_distribution()[:3].tolist()}..."
            )
        elif arch == "ramlakv":
            self.model = RAMLAKV_GPT(cfg, compression_ratio=compression_ratio)
            print(f"Created RAMLAKV_GPT")
            print(f"  Compression: {self.model.get_compression_stats()}")

        # Move to device
        self.model = self.model.to(self.args.device)

        # Store config for balance loss
        self._mla_config = cfg

        print(f"Number of parameters: {self.model.get_num_params()/1e6:.2f}M")

        # Recreate optimizer with new model
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup optimizer for the MLA model."""
        import torch

        # Use AdamW with weight decay
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }

        # Separate weight decay and no-decay params
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
        )

        print(f"Created optimizer with LR={self.args.learning_rate:.1e}")

    def train(self):
        """Run training."""
        if self.step_config["arch"] == "ra":
            # Use RATrainer's train method
            return self._ra_trainer.train()
        else:
            # Use base trainer
            return super().train()

    def run_dry_run(self):
        """Run architecture validation."""
        if self.step_config["arch"] == "ra":
            return self._ra_trainer.run_dry_run()
        else:
            return super().run_dry_run()


class RAMLACoordinator:
    """Coordinates running multiple RAMLA ablation steps."""

    def __init__(self, args, config, steps):
        self.args = args
        self.config = config
        self.steps = steps

    def run(self):
        """Run all ablation steps sequentially."""
        for step in self.steps:
            print(f"\n{'=' * 80}")
            print(f"Running RAMLA ablation step: {step}")
            print(f"{'=' * 80}\n")

            trainer = RAMLATrainer(self.args, self.config, ablation_step=step)

            if getattr(self.args, "dry_run", False):
                trainer.run_dry_run()
            else:
                trainer.train()

            print(f"\nCompleted RAMLA ablation step: {step}")
