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
    - RA model patching (ra.py)
    - R-MLP support
    - KV pruning/compression variants
    - V-series step configuration (V0-V19)
    - Gate analysis (RA gates, R-MLP gates)
    - Delayed activation (variance-guided)
    """

    def __init__(self, args, config, ablation_step: Optional[str] = None):
        """
        Initialize RA trainer.

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

        # Variance-guided activation tracking
        self.variance_activated = False
        self.loss_history = []
        self.activation_step = None

        # Fetch baseline metrics if configured
        self.baseline_metrics = None
        baseline_run_id = getattr(config, "BASELINE_RUN_ID", None)
        if baseline_run_id and baseline_run_id.strip():
            self.baseline_metrics = self._fetch_baseline_metrics(baseline_run_id)

    def _configure_step(self, args, step: str):
        """Configure args based on ablation step."""
        # Default values
        args.use_ra_v5 = False
        args.use_rmlp = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0

        # Variance-guided activation defaults (disabled by default)
        args.use_variance_guided = False
        args.variance_check_interval = 10  # Check every 10 steps (~30 seconds)
        args.variance_min_step = 50  # Start checking after 50 steps (~2.5 minutes)
        args.variance_window = 50  # Smaller window for faster response
        # Aggressive hybrid stability thresholds (activate within 10 minutes)
        args.stability_cv_threshold = 0.20  # Coefficient of variation < 20%
        args.stability_rate_threshold = 0.10  # Rate of change < 0.1 per step

        if step == "V0":
            # Baseline GPT-2
            pass
        elif step == "V1":
            # RA (R=4)
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = False
        elif step == "V2":
            # RA + Self-Restart
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = True
        elif step == "V3":
            # RA + R-MLP (basic)
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = False
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = False
        elif step == "V4":
            # RA + R-MLP + Mixer
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = True
            args.rmlp_use_gates = False
        elif step == "V5":
            # RA + R-MLP + Gates
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = True
        elif step == "V6":
            # RA + R-MLP + Mixer + Gates
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = True
            args.rmlp_use_gates = True
        elif step == "V7":
            # RA (R=8)
            args.use_ra_v5 = True
            args.ra_v5_R = 8
        elif step == "V8":
            # RA (R=8) + Self-Restart
            args.use_ra_v5 = True
            args.ra_v5_R = 8
            args.ra_v5_use_self_restart = True
        elif step == "V9":
            # RA (R=2)
            args.use_ra_v5 = True
            args.ra_v5_R = 2
        elif step == "V10":
            # RA + Self-Restart + 6x MLP
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_use_self_restart = True
            args.mlp_expansion_ratio = 6.0
        elif step == "V16":
            # RA (R=4) with per-head gates + variance-guided activation
            args.use_ra_v5 = True
            args.ra_v5_R = 4
            args.ra_v5_per_head_gates = True
            args.use_rmlp = False
            # Variance-guided activation (uses aggressive defaults: activates within 10 min)
            args.use_variance_guided = True
        elif step == "V17":
            # R-MLP basic (R_ff=64) + KV pruning (k=391) + variance-guided activation
            args.use_ra_v5 = False
            args.use_rmlp = True
            args.rmlp_R_ff = 64
            args.rmlp_use_mixer = False
            args.rmlp_use_gates = False
            # KV pruning config
            args.kv_cache_prune = True
            args.kv_prune_k = 391  # Golden ratio: 391/1024 ≈ 0.382
            args.kv_prune_recency = 64
            # Variance-guided activation (uses aggressive defaults: activates within 10 min)
            args.use_variance_guided = True
        elif step == "V18":
            # R-MLP golden (R_ff=1152) + KV pruning (learned ratio) + variance-guided
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
            # Variance-guided activation (uses aggressive defaults: activates within 10 min)
            args.use_variance_guided = True
        else:
            if self.master_process:
                print(f"Warning: Unknown ablation step {step}, using baseline V0")

    def create_model(self):
        """Create RA model based on ablation step."""
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

        # Apply RA / R-MLP patching
        if getattr(self.args, "use_ra_v5", False) and getattr(
            self.args, "use_rmlp", False
        ):
            # Both RA and R-MLP
            if self.master_process:
                print(
                    f"Patching with RA (R={self.args.ra_v5_R}) + R-MLP (R_ff={self.args.rmlp_R_ff})"
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
                per_head_gates=getattr(self.args, "ra_v5_per_head_gates", True),
            )
        elif getattr(self.args, "use_ra_v5", False):
            # RA only
            if self.master_process:
                print(f"Patching with RA (R={self.args.ra_v5_R})")
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

            # Variance-guided activation or standard coupling warmup
            use_variance = getattr(self.args, "use_variance_guided", False)

            if use_variance and not self.variance_activated:
                # Variance-guided mode: check if we should activate
                if (
                    self.iter_num % self.args.variance_check_interval == 0
                    and self.iter_num >= self.args.variance_min_step
                ):

                    if len(self.loss_history) >= self.args.variance_window:
                        # Hybrid stability metric: CV + Rate of Change
                        recent_losses = self.loss_history[-self.args.variance_window :]
                        losses_tensor = torch.tensor(recent_losses)

                        mean_loss = losses_tensor.mean().item()
                        std_loss = losses_tensor.std().item()
                        cv = std_loss / mean_loss if mean_loss > 0 else float("inf")

                        rate_of_change = abs(
                            recent_losses[-1] - recent_losses[0]
                        ) / len(recent_losses)

                        # Check both conditions
                        cv_threshold = getattr(
                            self.args, "stability_cv_threshold", 0.05
                        )
                        rate_threshold = getattr(
                            self.args, "stability_rate_threshold", 0.01
                        )

                        if cv < cv_threshold and rate_of_change < rate_threshold:
                            self.variance_activated = True
                            self.activation_step = self.iter_num
                            if self.master_process:
                                print(f"\n{'='*60}")
                                print(
                                    f"✓ Variance-guided activation triggered at step {self.iter_num}"
                                )
                                print(
                                    f"  Coefficient of Variation: {cv:.6f} < {cv_threshold}"
                                )
                                print(
                                    f"  Rate of Change: {rate_of_change:.6f} < {rate_threshold}"
                                )
                                print(
                                    f"  Loss mean: {mean_loss:.4f}, std: {std_loss:.4f}"
                                )
                                print(
                                    f"  Beginning {self.args.warmup_steps}-step coupling warmup"
                                )
                                print(f"{'='*60}\n")

                # Before activation: keep reciprocal pathways disabled
                coupling_scale = 0.0

            elif use_variance and self.variance_activated:
                # After activation: ramp from 0 to 1
                warmup_steps = getattr(self.args, "warmup_steps", 200)
                steps_since_activation = self.iter_num - self.activation_step
                if steps_since_activation < warmup_steps:
                    coupling_scale = steps_since_activation / warmup_steps
                else:
                    coupling_scale = 1.0

            else:
                # Standard coupling warmup (no variance-guided delay)
                warmup_steps = getattr(self.args, "warmup_steps", 200)
                if self.iter_num < warmup_steps:
                    coupling_scale = self.iter_num / warmup_steps
                else:
                    coupling_scale = 1.0

            # Apply coupling warmup to RA/R-MLP modules
            from ra import set_coupling_scale

            set_coupling_scale(self.raw_model, coupling_scale)

            # Logging
            if self.iter_num % self.args.log_interval == 0:
                avg_loss = running_loss / self.args.log_interval

                # Track loss for variance-guided activation
                if getattr(self.args, "use_variance_guided", False):
                    self.loss_history.append(avg_loss)

                if self.master_process:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1

                    avg_ppl = math.exp(min(avg_loss, 20))

                    print(
                        f"Iter {self.iter_num:5d} | loss {avg_loss:.4f} | ppl {avg_ppl:7.2f} | "
                        f"lr {lr:.2e} | {dt*1000/self.args.log_interval:.1f}ms/iter"
                    )

                    # Collect gate metrics for logging
                    gate_metrics = {}

                    # Analyze RA gates if enabled
                    if getattr(self.args, "use_ra_v5", False):
                        ra_gate_stats = self._analyze_ra_gates()
                        if ra_gate_stats:
                            gate_metrics.update(
                                {f"ra_gates/{k}": v for k, v in ra_gate_stats.items()}
                            )
                            print(
                                f"  RA gates: w_std={ra_gate_stats.get('w_std_mean', 0):.3f}, "
                                f"w_rec={ra_gate_stats.get('w_rec_mean', 0):.3f}"
                            )

                    # Analyze R-MLP gates if enabled
                    if getattr(self.args, "use_rmlp", False):
                        rmlp_gate_stats = self._analyze_rmlp_gates()
                        if rmlp_gate_stats:
                            gate_metrics.update(
                                {
                                    f"rmlp_gates/{k}": v
                                    for k, v in rmlp_gate_stats.items()
                                }
                            )
                            print(
                                f"  R-MLP gates: w_std={rmlp_gate_stats.get('w_std_mean', 0):.3f}, "
                                f"w_rec={rmlp_gate_stats.get('w_rec_mean', 0):.3f}"
                            )

                    # Analyze KV pruning stats if enabled
                    kv_pruning_stats = self._analyze_kv_pruning()
                    if kv_pruning_stats:
                        gate_metrics.update(
                            {f"kv_pruning/{k}": v for k, v in kv_pruning_stats.items()}
                        )
                        # Print either fixed or learned ratio stats
                        if "kv_keep_ratio" in kv_pruning_stats:
                            print(
                                f"  KV pruning: keep={kv_pruning_stats['kv_keep_ratio']:.3f} "
                                f"({kv_pruning_stats['kv_memory_reduction_pct']:.1f}% reduction)"
                            )
                        elif "kv_learned_ratio" in kv_pruning_stats:
                            print(
                                f"  KV pruning (learned): ratio={kv_pruning_stats['kv_learned_ratio']:.3f} "
                                f"({kv_pruning_stats['kv_learned_memory_reduction_pct']:.1f}% reduction)"
                            )

                    # Add variance-guided monitoring if enabled
                    if (
                        getattr(self.args, "use_variance_guided", False)
                        and len(self.loss_history) >= 10
                    ):
                        recent = self.loss_history[-min(100, len(self.loss_history)) :]
                        losses_tensor = torch.tensor(recent)
                        mean_loss = losses_tensor.mean().item()
                        std_loss = losses_tensor.std().item()
                        cv = std_loss / mean_loss if mean_loss > 0 else 0
                        rate = (
                            abs(recent[-1] - recent[0]) / len(recent)
                            if len(recent) > 1
                            else 0
                        )

                        gate_metrics.update(
                            {
                                "variance/cv": cv,
                                "variance/rate_of_change": rate,
                                "variance/coupling_scale": coupling_scale,
                                "variance/activated": float(self.variance_activated),
                                "variance/loss_mean": mean_loss,
                                "variance/loss_std": std_loss,
                            }
                        )

                        # Print variance monitoring stats to console
                        print(
                            f"  Variance: CV={cv:.4f}, rate={rate:.4f}, "
                            f"coupling={coupling_scale:.3f}, activated={self.variance_activated}"
                        )

                    # Combine all metrics and log
                    metrics = {
                        "train_loss": avg_loss,
                        "train_perplexity": avg_ppl,
                        "learning_rate": lr,
                    }
                    metrics.update(gate_metrics)

                    # Add baseline metrics as reference lines if available
                    if self.baseline_metrics:
                        metrics.update(self.baseline_metrics)

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

                    # Update best metrics
                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                    if val_ppl < self.best_perplexity:
                        self.best_perplexity = val_ppl

                    self.log_metrics(
                        {
                            "val_loss": losses["val"],
                            "val_perplexity": val_ppl,
                            "best_perplexity": self.best_perplexity,
                        }
                    )

            self.iter_num += 1

        # Training complete
        if self.master_process:
            print(f"\nTraining complete for step {self.ablation_step}!")
            final_losses = self.estimate_loss()
            print(
                f"Final: train {final_losses['train']:.4f}, val {final_losses['val']:.4f}"
            )

            # Generate tier hints if configured
            self._generate_tier_hints()

            # Save metrics to JSON if requested
            if hasattr(self.args, "json_output") and self.args.json_output:
                self.save_metrics_json(self.args.json_output)

    def _analyze_ra_gates(self) -> Dict[str, float]:
        """Analyze RA gate values."""
        try:
            from ra import ReciprocalAttention

            w_std_list = []
            w_rec_list = []
            count = 0

            for name, module in self.raw_model.named_modules():
                if isinstance(module, ReciprocalAttention):
                    count += 1
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
                    "w_std_mean": np.mean(w_std_list),
                    "w_rec_mean": np.mean(w_rec_list),
                    "w_std_std": np.std(w_std_list),
                    "w_rec_std": np.std(w_rec_list),
                    "module_count": float(count),
                }
        except Exception as e:
            if self.master_process and self.iter_num < 20:
                print(f"  Warning: RA gate analysis failed: {e}")

        return {}

    def _analyze_rmlp_gates(self) -> Dict[str, float]:
        """Analyze R-MLP gate values."""
        try:
            from ra import ReciprocalMLP

            w_std_list = []
            w_rec_list = []
            count = 0

            for name, module in self.raw_model.named_modules():
                if isinstance(module, ReciprocalMLP):
                    count += 1
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
                    "w_std_mean": np.mean(w_std_list),
                    "w_rec_mean": np.mean(w_rec_list),
                    "w_std_std": np.std(w_std_list),
                    "w_rec_std": np.std(w_rec_list),
                    "module_count": float(count),
                }
        except Exception as e:
            if self.master_process and self.iter_num < 20:
                print(f"  Warning: R-MLP gate analysis failed: {e}")

        return {}

    def _analyze_kv_pruning(self) -> Dict[str, float]:
        """
        Analyze KV cache pruning statistics.

        Returns metrics for both fixed ratio (V17) and learned ratio (V18) pruning.
        """
        try:
            from ra import PrunedKVAttention

            fixed_ratios = []
            learned_ratios = []
            ratio_grads = []

            for name, module in self.raw_model.named_modules():
                if isinstance(module, PrunedKVAttention):
                    with torch.no_grad():
                        # Calculate keep ratio
                        k_keep = module.k_keep
                        block_size = module.block_size
                        keep_ratio = k_keep / block_size

                        if module.learn_ratio and hasattr(module, "keep_ratio"):
                            # V18: Learned ratio pruning
                            learned_ratio = module.keep_ratio.cpu().item()
                            learned_ratios.append(learned_ratio)

                            # Track gradient norm if available
                            if module.keep_ratio.grad is not None:
                                grad_norm = module.keep_ratio.grad.norm().cpu().item()
                                ratio_grads.append(grad_norm)
                        else:
                            # V17: Fixed ratio pruning
                            fixed_ratios.append(keep_ratio)

            metrics = {}

            # Fixed ratio stats (V17)
            if fixed_ratios:
                keep_ratio = np.mean(fixed_ratios)
                metrics.update(
                    {
                        "kv_keep_ratio": keep_ratio,
                        "kv_memory_reduction_pct": (1.0 - keep_ratio) * 100,
                    }
                )

            # Learned ratio stats (V18)
            if learned_ratios:
                learned_ratio = np.mean(learned_ratios)
                metrics.update(
                    {
                        "kv_learned_ratio": learned_ratio,
                        "kv_learned_memory_reduction_pct": (1.0 - learned_ratio) * 100,
                    }
                )

                if ratio_grads:
                    metrics["kv_ratio_grad_norm"] = np.mean(ratio_grads)

            return metrics

        except Exception as e:
            pass

        return {}

    def _fetch_baseline_metrics(
        self, baseline_run_id: str
    ) -> Optional[Dict[str, float]]:
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

    def _generate_tier_hints(self):
        """
        Generate tier hints JSON from Adam optimizer states.

        Called at the end of training if tiering is enabled in config.
        """
        # Check if tiering is enabled
        tiering_enabled = getattr(self.config, "ENABLE_HIERARCHICAL_TIERING", False)
        generate_json = getattr(self.config, "TIERING_GENERATE_JSON", False)

        if not (tiering_enabled and generate_json):
            return

        print("\nGenerating tier hints from Adam optimizer states...")

        try:
            from lib.tiering import AdamStateTierAnalyzer

            # Get thresholds from config
            hbm_threshold = float(
                getattr(self.config, "TIERING_HBM_THRESHOLD", "0.3").strip('"')
            )
            cpu_threshold = float(
                getattr(self.config, "TIERING_CPU_THRESHOLD", "0.5").strip('"')
            )

            # Create analyzer
            analyzer = AdamStateTierAnalyzer(
                hbm_threshold=hbm_threshold, cpu_threshold=cpu_threshold
            )

            # Analyze optimizer states
            tier_assignments = analyzer.analyze_optimizer_states(
                self.optimizer, self.raw_model
            )

            # Count tier distribution
            tier_counts = {}
            for tier in tier_assignments.values():
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            print(f"  Tier distribution:")
            for tier in ["HBM", "CPU", "SSD"]:
                count = tier_counts.get(tier, 0)
                if count > 0:
                    pct = 100 * count / len(tier_assignments)
                    print(f"    {tier}: {count} modules ({pct:.1f}%)")

            # Save to JSON
            output_path = getattr(
                self.config, "TIERING_JSON_OUTPUT", "tier_hints.json"
            ).strip('"')

            analyzer.save_tier_hints(tier_assignments, output_path)
            print(f"  Saved tier hints to {output_path}")

        except Exception as e:
            print(f"  Warning: Failed to generate tier hints: {e}")
            import traceback

            traceback.print_exc()
