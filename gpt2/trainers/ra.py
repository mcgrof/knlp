"""
RA Trainer

Trainer for Reciprocal Attention (RA) and Reciprocal MLP (R-MLP) ablation studies.
Supports V0-V19 ablation steps with coupling warmup to prevent MLP collapse.
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

from gpt2.model import GPT, GPTConfig
from gpt2.ra_patch import patch_gpt2_with_ra_v5, patch_gpt2_with_unified_ra_and_rmlp
from lib.optimizers import create_optimizer
from .base import BaseGPT2Trainer


class RATrainer(BaseGPT2Trainer):
    """
    Trainer for RA+R-MLP ablation studies with coupling warmup.

    Implements:
    - Unified RA model patching (ra.py)
    - R-MLP support
    - KV pruning/compression variants
    - V-series step configuration (V0-V19)
    - Gate analysis (RA gates, R-MLP gates)
    - Delayed activation (variance-guided)
    """

    def __init__(self, args, config, ablation_step: Optional[str] = None):
        """
        Initialize Unified RA trainer.

        Args:
            args: Command-line arguments
            config: Config object
            ablation_step: Ablation step (e.g., "V1", "V3"). If None, uses args.ra_step
        """
        # Configure ablation step
        self.ablation_step = ablation_step or getattr(args, "ra_step", "V0")
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

    def _configure_step(self, args, step: str):
        """Configure args based on ablation step."""
        # Default values
        args.use_ra_v5 = False
        args.use_rmlp = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0

        if step == "V0":
            # Baseline GPT-2
            pass
        elif step == "V1":
            # Unified RA (R=4)
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = False
        elif step == "V2":
            # Unified RA + Self-Restart
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = True
        elif step == "V3":
            # Unified RA + R-MLP (basic)
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = False
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = False
        elif step == "V4":
            # Unified RA + R-MLP + Mixer
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = True
            args.rmlp_use_gates = False
        elif step == "V5":
            # Unified RA + R-MLP + Gates
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = True
        elif step == "V6":
            # Unified RA + R-MLP + Mixer + Gates
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = True
            args.rmlp_use_gates = True
        elif step == "V7":
            # Unified RA (R=8)
            args.use_ra_v5 = True
            args.ra_v5_R = 8
        elif step == "V8":
            # Unified RA (R=8) + Self-Restart
            args.use_ra_v5 = True
            args.ra_v5_R = 8
            args.ra_v5_use_self_restart = True
        elif step == "V9":
            # Unified RA (R=2)
            args.use_ra_v5 = True
            args.ra_v5_R = 2
        elif step == "V10":
            # Unified RA + Self-Restart + 6x MLP
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = True
            args.mlp_expansion_ratio = 6.0
        elif step == "V16":
            # Unified RA (R=4) with per-head gates
            # TODO: Add variance-guided activation (currently uses standard coupling warmup)
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_per_head_gates = True
            args.use_rmlp = False
        elif step == "V17":
            # R-MLP basic (R_ff=64) + KV pruning (k=391)
            # TODO: Add variance-guided activation (currently uses standard coupling warmup)
            args.use_ra_v5 = False
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = False
            # KV pruning config
            args.kv_cache_prune = True
            args.kv_prune_k = 391  # Golden ratio: 391/1024 ≈ 0.382
            args.kv_prune_recency = 64
        elif step == "V18":
            # R-MLP golden (R_ff=1152) + KV pruning (learned ratio)
            # TODO: Add variance-guided activation (currently uses standard coupling warmup)
            args.use_ra_v5 = False
            args.use_rmlp = True
            args.rmlp_R_ff = 1152  # Golden ratio split of expansion
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = False
            # KV pruning with learned ratio
            args.kv_cache_prune = True
            args.kv_prune_learned = True
            args.kv_prune_init_ratio = 0.382  # Start at golden ratio
            args.kv_prune_recency = 64
        else:
            if self.master_process:
                print(f"Warning: Unknown ablation step {step}, using baseline V0")

    def create_model(self):
        """Create Unified RA model based on ablation step."""
        if self.master_process:
            print(f"Initializing model for ablation step {self.ablation_step}")

        # Create base GPT config
        config = GPTConfig.from_name(self.args.model_name)
        config.block_size = self.args.block_size
        config.dropout = self.args.dropout
        config.bias = getattr(self.args, "bias", True)

        # Create base model
        model = GPT(config)
        model.to(self.device)

        # Apply Unified RA / R-MLP patching
        if getattr(self.args, "use_ra_v5", False) and getattr(
            self.args, "use_rmlp", False
        ):
            # Both RA and R-MLP
            if self.master_process:
                print(
                    f"Patching with Unified RA (R={self.args.ra_v5_R}) + R-MLP (R_ff={self.args.rmlp_R_ff})"
                )
            model = patch_gpt2_with_unified_ra_and_rmlp(
                model,
                R=getattr(self.args, "ra_v5_R", 4),
                attn_dropout=self.args.dropout,
                use_self_restart=getattr(self.args, "ra_v5_use_self_restart", False),
                mlp_expansion=self.args.mlp_expansion_ratio,
                R_ff=getattr(self.args, "rmlp_R_ff", 64),
                mlp_dropout=self.args.dropout,
                use_mixer=getattr(self.args, "rmlp_use_mixer", False),
                use_gates=getattr(self.args, "rmlp_use_gates", False),
                tie_up_low=getattr(self.args, "rmlp_tie_up_low", False),
                per_head_gates=getattr(self.args, "ra_v5_per_head_gates", True),
            )
        elif getattr(self.args, "use_ra_v5", False):
            # Unified RA only
            if self.master_process:
                print(f"Patching with Unified RA (R={self.args.ra_v5_R})")
            model = patch_gpt2_with_ra_v5(
                model,
                R=getattr(self.args, "ra_v5_R", 4),
                dropout=self.args.dropout,
                use_self_restart=getattr(self.args, "ra_v5_use_self_restart", False),
                per_head_gates=getattr(self.args, "ra_v5_per_head_gates", True),
            )

        # Compile if requested (must be before DDP)
        if getattr(self.args, "compile", False) and hasattr(torch, "compile"):
            if self.master_process:
                print("Compiling model with torch.compile()...")
            model = torch.compile(model)

        # Wrap in DDP if needed
        model = self.wrap_model_ddp(model)

        return model

    def create_optimizer(self):
        """Create optimizer (typically AdamWSPAM for RA experiments)."""
        if self.master_process:
            print(f"Setting up {self.args.optimizer} optimizer...")

        return create_optimizer(
            model=self.model,
            optimizer_type=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            num_epochs=self.args.num_epochs,
            args=self.args,
            model_type="gpt2",
        )

    def train(self):
        """Main training loop (similar to VanillaGPT2Trainer but with gate analysis)."""
        if self.master_process:
            print(f"\nStarting training for step {self.ablation_step}...")
            print(f"Parameters: {self.raw_model.get_num_params()/1e6:.2f}M")
            print(f"Device: {self.device}, dtype: {self.dtype}")
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
                        print(f"\nReached max training time ({elapsed:.1f}s)")
                    break

            # Update learning rate
            lr = (
                self.get_lr(self.iter_num)
                if getattr(self.args, "decay_lr", True)
                else self.args.learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            for micro_step in range(self.args.gradient_accumulation):
                X, Y = self.get_batch("train")
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.args.gradient_accumulation
                self.scaler.scale(loss).backward()
                running_loss += loss.item()

            # Gradient processing
            if self.device == "cuda":
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Coupling warmup: ramp reciprocal pathways from 0 to 1
            # Prevents MLP collapse by starting with vanilla GPT-2 pathways
            warmup_steps = getattr(self.args, 'warmup_steps', 200)
            if self.iter_num < warmup_steps:
                coupling_scale = self.iter_num / warmup_steps
            else:
                coupling_scale = 1.0

            # Import and apply coupling warmup
            from ra import set_coupling_scale
            set_coupling_scale(self.raw_model, coupling_scale)

            # Logging
            if self.iter_num % self.args.log_interval == 0 and self.master_process:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                avg_loss = running_loss / self.args.log_interval
                avg_ppl = math.exp(min(avg_loss, 20))

                print(
                    f"Iter {self.iter_num:5d} | loss {avg_loss:.4f} | ppl {avg_ppl:7.2f} | "
                    f"lr {lr:.2e} | {dt*1000/self.args.log_interval:.1f}ms/iter"
                )

                # Analyze gates if RA/R-MLP enabled
                gate_stats = {}
                if getattr(self.args, "use_ra_v5", False):
                    gate_stats = self._analyze_ra_gates()
                    if gate_stats:
                        print(
                            f"  RA gates: w_std={gate_stats.get('ra_gate/w_std_mean', 0):.3f}, "
                            f"w_rec={gate_stats.get('ra_gate/w_rec_mean', 0):.3f}"
                        )

                # Analyze R-MLP gates
                rmlp_stats = self._analyze_rmlp_gates()
                if rmlp_stats:
                    print(
                        f"  R-MLP gates: w_std={rmlp_stats.get('rmlp_gate/w_std_mean', 0):.3f}, "
                        f"w_rec={rmlp_stats.get('rmlp_gate/w_rec_mean', 0):.3f}"
                    )

                # Combine all metrics
                metrics = {
                    "train_loss": avg_loss,
                    "train_perplexity": avg_ppl,
                    "learning_rate": lr,
                }
                metrics.update(gate_stats)
                metrics.update(rmlp_stats)

                self.log_metrics(metrics)

                running_loss = 0.0

            # Evaluation
            if self.iter_num % self.args.eval_interval == 0:
                losses = self.estimate_loss()
                if self.master_process:
                    val_ppl = math.exp(min(losses["val"], 20))
                    print(
                        f"\nEval @ iter {self.iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, ppl {val_ppl:.2f}"
                    )

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

                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]

            self.iter_num += 1

        # Training complete
        if self.master_process:
            print(f"\nTraining complete for step {self.ablation_step}!")
            final_losses = self.estimate_loss()
            print(
                f"Final: train {final_losses['train']:.4f}, val {final_losses['val']:.4f}"
            )

    def _analyze_ra_gates(self) -> Dict[str, float]:
        """Analyze Unified RA gate values."""
        try:
            from ra import UnifiedRAttention

            w_std_list = []
            w_rec_list = []

            for name, module in self.raw_model.named_modules():
                if isinstance(module, UnifiedRAttention):
                    with torch.no_grad():
                        w_std = module.w_std.cpu()
                        w_rec = module.w_rec.cpu()

                        if w_std.dim() == 0:
                            w_std_list.append(w_std.item())
                            w_rec_list.append(w_rec.item())
                        else:
                            w_std_list.extend(w_std.tolist())
                            w_rec_list.extend(w_rec.tolist())

            if w_std_list:
                return {
                    "ra_gate/w_std_mean": np.mean(w_std_list),
                    "ra_gate/w_std_min": np.min(w_std_list),
                    "ra_gate/w_std_max": np.max(w_std_list),
                    "ra_gate/w_std_std": np.std(w_std_list),
                    "ra_gate/w_rec_mean": np.mean(w_rec_list),
                    "ra_gate/w_rec_min": np.min(w_rec_list),
                    "ra_gate/w_rec_max": np.max(w_rec_list),
                    "ra_gate/w_rec_std": np.std(w_rec_list),
                }
        except Exception as e:
            pass

        return {}

    def _analyze_rmlp_gates(self) -> Dict[str, float]:
        """Analyze R-MLP gate values."""
        try:
            from ra import ReciprocalMLP

            w_std_list = []
            w_rec_list = []

            for name, module in self.raw_model.named_modules():
                if isinstance(module, ReciprocalMLP):
                    with torch.no_grad():
                        w_std = module.w_std.cpu()
                        w_rec = module.w_rec.cpu()

                        if w_std.dim() == 0:
                            w_std_list.append(w_std.item())
                            w_rec_list.append(w_rec.item())
                        else:
                            w_std_list.extend(w_std.tolist())
                            w_rec_list.extend(w_rec.tolist())

            if w_std_list:
                return {
                    "rmlp_gate/w_std_mean": np.mean(w_std_list),
                    "rmlp_gate/w_std_min": np.min(w_std_list),
                    "rmlp_gate/w_std_max": np.max(w_std_list),
                    "rmlp_gate/w_std_std": np.std(w_std_list),
                    "rmlp_gate/w_rec_mean": np.mean(w_rec_list),
                    "rmlp_gate/w_rec_min": np.min(w_rec_list),
                    "rmlp_gate/w_rec_max": np.max(w_rec_list),
                    "rmlp_gate/w_rec_std": np.std(w_rec_list),
                }
        except Exception as e:
            pass

        return {}
