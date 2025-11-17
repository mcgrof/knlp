# SPDX-License-Identifier: MIT
"""
KV Feature-Circuit Analysis via Binary Channel Masking

Implements sparse circuit discovery by optimizing binary masks over
attention K/V channels while keeping model weights frozen. Follows the
continuous sparsification approach from "Scaling Sparse Feature Circuit
Finding to Gemma 9B".

Usage:
    analyzer = KVCircuitAnalyzer(model, config)
    results = analyzer.run_analysis(dataloader)
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class KVMaskConfig:
    """Configuration for KV feature mask."""

    n_heads: int
    head_dim: int
    # Initial probability that a channel is "on"
    init_keep_prob: float = 0.5
    # Temperature for sigmoid relaxation
    temperature: float = 1.0
    # Use straight-through estimator for hard {0,1} mask
    hard: bool = True


@dataclass
class AnalysisConfig:
    """Configuration for KV circuit analysis."""

    # Target sparsity (fraction of channels to prune)
    target_sparsity: float = 0.95
    # Number of optimization steps
    num_steps: int = 500
    # Learning rate for mask optimization
    learning_rate: float = 0.01
    # Target metric to maintain ("loss", "accuracy", "perplexity", etc.)
    target_metric: str = "loss"
    # Temperature schedule: "linear:1.0:0.1", "constant:0.5", "exp:1.0:0.01"
    temp_schedule: str = "linear:1.0:0.1"
    # Sparsity schedule: how to gradually increase sparsity
    sparsity_schedule: str = "linear"  # or "exponential", "cosine"
    # L1 regularization weight for mask logits
    l1_weight: float = 0.0
    # Evaluation batch size
    eval_batch_size: int = 32
    # Number of eval batches per step
    eval_batches: int = 10
    # Save mask checkpoints every N steps
    save_interval: int = 100
    # Output directory for results
    output_dir: str = "mechint_analysis"


class KVFeatureMask(nn.Module):
    """
    Learnable binary feature mask over K/V channels.

    Uses straight-through estimator: hard {0,1} mask in forward pass,
    gradient flows through relaxed sigmoid probabilities.

    Args:
        cfg: KVMaskConfig with n_heads, head_dim, and optimization settings
    """

    def __init__(self, cfg: KVMaskConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize logits so sigmoid(logits) ≈ init_keep_prob
        p = cfg.init_keep_prob
        p = min(max(p, 1e-4), 1.0 - 1e-4)  # clamp to avoid log(0)
        init_logit = math.log(p / (1.0 - p))

        self.mask_logits = nn.Parameter(
            torch.full(
                (cfg.n_heads, cfg.head_dim),
                fill_value=init_logit,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply feature mask to K and V.

        Args:
            k: [B, H, T, D] key tensor
            v: [B, H, T, D] value tensor
            temperature: override for sigmoid temperature

        Returns:
            k_masked: [B, H, T, D] masked keys
            v_masked: [B, H, T, D] masked values
            mask: [H, D] current mask (post-STE if hard=True)
        """
        B, H, T, D = k.shape
        assert k.shape == v.shape, "K and V must have same shape"
        assert H == self.cfg.n_heads and D == self.cfg.head_dim

        temp = temperature if temperature is not None else self.cfg.temperature
        probs = torch.sigmoid(self.mask_logits / temp)  # [H, D]

        if self.cfg.hard:
            # Straight-through: forward is {0,1}, backward through probs
            hard_mask = (probs > 0.5).float()
            mask = hard_mask + (probs - probs.detach())
        else:
            mask = probs

        # Broadcast over batch and sequence: [1, H, 1, D]
        mask_expanded = mask.view(1, H, 1, D)
        k_masked = k * mask_expanded
        v_masked = v * mask_expanded

        return k_masked, v_masked, mask

    def get_sparsity(self) -> float:
        """Get current fraction of channels pruned."""
        with torch.no_grad():
            probs = torch.sigmoid(self.mask_logits)
            mask = (probs > 0.5).float()
            return 1.0 - mask.mean().item()

    def get_importance_scores(self) -> torch.Tensor:
        """Get per-channel importance scores."""
        with torch.no_grad():
            return torch.sigmoid(self.mask_logits)


class SparsitySchedule:
    """Manages gradual increase in sparsity during optimization."""

    def __init__(self, schedule_type: str, target_sparsity: float, num_steps: int):
        self.schedule_type = schedule_type
        self.target = target_sparsity
        self.num_steps = num_steps

    def get_sparsity(self, step: int) -> float:
        """Get target sparsity for current step."""
        t = min(step / self.num_steps, 1.0)

        if self.schedule_type == "linear":
            return t * self.target
        elif self.schedule_type == "exponential":
            return self.target * (1.0 - math.exp(-5 * t))
        elif self.schedule_type == "cosine":
            return self.target * (1.0 - math.cos(t * math.pi / 2))
        else:
            return self.target


class TemperatureSchedule:
    """Manages temperature annealing for mask relaxation."""

    def __init__(self, schedule_str: str, num_steps: int):
        # Parse format: "type:start:end"
        parts = schedule_str.split(":")
        self.schedule_type = parts[0]
        self.start_temp = float(parts[1]) if len(parts) > 1 else 1.0
        self.end_temp = float(parts[2]) if len(parts) > 2 else 0.1
        self.num_steps = num_steps

    def get_temperature(self, step: int) -> float:
        """Get temperature for current step."""
        t = min(step / self.num_steps, 1.0)

        if self.schedule_type == "constant":
            return self.start_temp
        elif self.schedule_type == "linear":
            return self.start_temp + t * (self.end_temp - self.start_temp)
        elif self.schedule_type == "exp":
            log_start = math.log(self.start_temp)
            log_end = math.log(self.end_temp)
            return math.exp(log_start + t * (log_end - log_start))
        else:
            return self.start_temp


class KVCircuitAnalyzer:
    """
    Analyzes trained models to identify sparse KV feature circuits.

    Freezes model weights and optimizes binary masks over attention K/V
    channels to find minimal circuits that maintain target metric.

    Args:
        model: Trained model with attention layers
        config: AnalysisConfig with optimization settings
        device: Device to run analysis on
    """

    def __init__(
        self,
        model: nn.Module,
        config: AnalysisConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device and freeze weights
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        # Inject KV masks into attention layers
        self.masks = self._inject_kv_masks()

        # Create optimizer for mask logits only
        mask_params = [m.mask_logits for m in self.masks.values()]
        self.optimizer = torch.optim.Adam(mask_params, lr=config.learning_rate)

        # Create schedules
        self.sparsity_schedule = SparsitySchedule(
            config.sparsity_schedule, config.target_sparsity, config.num_steps
        )
        self.temp_schedule = TemperatureSchedule(config.temp_schedule, config.num_steps)

        # Logging
        self.metrics_history = []
        os.makedirs(config.output_dir, exist_ok=True)

    def _inject_kv_masks(self) -> Dict[str, KVFeatureMask]:
        """
        Inject KV feature masks into all attention layers.

        Returns:
            Dictionary mapping layer names to KV mask modules
        """
        masks = {}

        # Find all attention modules
        for name, module in self.model.named_modules():
            if self._is_attention_layer(name, module):
                # Get attention config
                n_heads, head_dim = self._get_attention_dims(module)

                # Create mask config
                mask_cfg = KVMaskConfig(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    init_keep_prob=0.5,
                    temperature=1.0,
                    hard=True,
                )

                # Create and register mask
                mask = KVFeatureMask(mask_cfg).to(self.device)
                masks[name] = mask

                # Hook the mask into the forward pass
                self._hook_mask_into_layer(module, mask)

        print(f"Injected KV masks into {len(masks)} attention layers")
        return masks

    def _is_attention_layer(self, name: str, module: nn.Module) -> bool:
        """Check if module is an attention layer."""
        # Only match the CausalSelfAttention module itself, not submodules
        # Check for class name to avoid matching c_attn (Linear) or attn_dropout (Dropout)
        return module.__class__.__name__ == "CausalSelfAttention"

    def _get_attention_dims(self, module: nn.Module) -> Tuple[int, int]:
        """Extract n_heads and head_dim from attention module."""
        # Try to infer from module attributes
        if hasattr(module, "n_head"):
            n_heads = module.n_head
        elif hasattr(module, "num_heads"):
            n_heads = module.num_heads
        else:
            n_heads = 12  # default for GPT-2

        if hasattr(module, "head_dim"):
            head_dim = module.head_dim
        elif hasattr(module, "n_embd") and hasattr(module, "n_head"):
            head_dim = module.n_embd // module.n_head
        else:
            head_dim = 64  # default

        return n_heads, head_dim

    def _hook_mask_into_layer(self, module: nn.Module, mask: KVFeatureMask):
        """Hook mask application into attention layer forward pass."""
        # Store original forward
        original_forward = module.forward

        # Store reference to self for temperature access
        analyzer = self

        def masked_forward(x, *args, **kwargs):
            # Pass mask to attention forward with current temperature
            # The mask will be applied inside the attention module
            kwargs["mechint_kv_mask"] = mask
            # Get current temperature from analyzer
            if hasattr(analyzer, "_current_temperature"):
                mask.cfg.temperature = analyzer._current_temperature
            return original_forward(x, *args, **kwargs)

        # Replace forward method
        module.forward = masked_forward

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model with current masks.

        Returns:
            Dictionary with metrics: loss, perplexity, sparsity, bits_per_byte, etc.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_tokens = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= self.config.eval_batches:
                break

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass (with masks applied)
            logits, loss = self.model(x, y)
            total_loss += loss.item()
            total_tokens += y.numel()  # Count tokens for BPB calculation
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))

        # Compute bits-per-byte (BPB)
        # Assume GPT-2 tokenizer compression: ~4.8 bytes/token
        # BPB = log2(perplexity) = log(perplexity) / log(2)
        bytes_per_token = 4.8
        bits_per_token = avg_loss / math.log(2)  # Convert nats to bits
        bits_per_byte = bits_per_token / bytes_per_token

        # Compute overall sparsity
        total_pruned = 0
        total_params = 0
        for mask in self.masks.values():
            scores = mask.get_importance_scores()
            total_pruned += (scores <= 0.5).sum().item()
            total_params += scores.numel()

        sparsity = total_pruned / max(total_params, 1)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "bits_per_byte": bits_per_byte,
            "sparsity": sparsity,
            "num_batches": num_batches,
        }

    def optimization_step(self, dataloader: DataLoader, step: int) -> Dict[str, float]:
        """
        Single optimization step for mask learning.

        Args:
            dataloader: DataLoader for training data
            step: Current optimization step

        Returns:
            Dictionary with step metrics
        """
        self.model.train()

        # Get current temperature and target sparsity
        temperature = self.temp_schedule.get_temperature(step)
        target_sparsity = self.sparsity_schedule.get_sparsity(step)

        # Store temperature for mask hooks to access
        self._current_temperature = temperature

        # Sample batch
        batch_iter = iter(dataloader)
        x, y = next(batch_iter)
        x, y = x.to(self.device), y.to(self.device)

        # Forward pass with current masks (temperature passed via hooks)
        logits, loss = self.model(x, y)

        # Sparsity regularization: L1 on mask probabilities
        l1_loss = 0.0
        if self.config.l1_weight > 0:
            for mask in self.masks.values():
                probs = torch.sigmoid(mask.mask_logits)
                l1_loss += probs.sum()
            l1_loss = self.config.l1_weight * l1_loss

        # Total loss
        total_loss = loss + l1_loss

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Collect metrics
        current_sparsity = sum(m.get_sparsity() for m in self.masks.values()) / len(
            self.masks
        )

        return {
            "step": step,
            "loss": loss.item(),
            "l1_loss": l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss,
            "total_loss": total_loss.item(),
            "temperature": temperature,
            "target_sparsity": target_sparsity,
            "current_sparsity": current_sparsity,
        }

    def run_analysis(
        self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Run complete KV circuit analysis.

        Args:
            train_dataloader: DataLoader for optimization
            val_dataloader: Optional validation DataLoader

        Returns:
            Dictionary with analysis results
        """
        print(f"Starting KV circuit analysis for {self.config.num_steps} steps")
        print(f"Target sparsity: {self.config.target_sparsity:.1%}")
        print(f"Output directory: {self.config.output_dir}")

        # Initial evaluation
        print("\nInitial evaluation...")
        initial_metrics = self.evaluate(val_dataloader or train_dataloader)
        print(f"Initial loss: {initial_metrics['loss']:.4f}")
        print(f"Initial sparsity: {initial_metrics['sparsity']:.1%}")

        # Optimization loop
        for step in range(self.config.num_steps):
            metrics = self.optimization_step(train_dataloader, step)
            self.metrics_history.append(metrics)

            if step % 10 == 0:
                print(
                    f"Step {step:4d} | loss {metrics['loss']:.4f} | "
                    f"sparsity {metrics['current_sparsity']:.1%} | "
                    f"temp {metrics['temperature']:.3f}"
                )

            # Periodic evaluation
            if step % 50 == 0 and val_dataloader is not None:
                eval_metrics = self.evaluate(val_dataloader)
                metrics.update({f"val_{k}": v for k, v in eval_metrics.items()})

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self.save_checkpoint(step)

        # Final evaluation
        print("\nFinal evaluation...")
        final_metrics = self.evaluate(val_dataloader or train_dataloader)
        print(f"Final loss: {final_metrics['loss']:.4f}")
        print(f"Final sparsity: {final_metrics['sparsity']:.1%}")

        # Save final masks
        self.save_masks()

        return {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "history": self.metrics_history,
        }

    def save_masks(self, filename: str = "final_masks.pt"):
        """Save learned masks to disk."""
        masks_dict = {}
        for name, mask in self.masks.items():
            masks_dict[name] = {
                "logits": mask.mask_logits.detach().cpu(),
                "mask": (torch.sigmoid(mask.mask_logits) > 0.5).float().cpu(),
                "importance": mask.get_importance_scores().cpu(),
                "sparsity": mask.get_sparsity(),
            }

        filepath = os.path.join(self.config.output_dir, filename)
        torch.save(masks_dict, filepath)
        print(f"Saved masks to {filepath}")

    def save_checkpoint(self, step: int):
        """Save checkpoint during optimization."""
        checkpoint = {
            "step": step,
            "mask_logits": {
                name: mask.mask_logits.detach().cpu()
                for name, mask in self.masks.items()
            },
            "optimizer_state": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }

        filepath = os.path.join(self.config.output_dir, f"checkpoint_step{step}.pt")
        torch.save(checkpoint, filepath)


def run_kv_circuit_analysis(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Optional[AnalysisConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run KV circuit analysis.

    Args:
        model: Trained model to analyze
        train_dataloader: Training data for optimization
        val_dataloader: Optional validation data
        config: Optional AnalysisConfig (uses defaults if None)
        device: Device to run on (auto-detects if None)

    Returns:
        Analysis results dictionary
    """
    if config is None:
        config = AnalysisConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analyzer = KVCircuitAnalyzer(model, config, device)
    results = analyzer.run_analysis(train_dataloader, val_dataloader)

    return results
