# SPDX-License-Identifier: MIT

"""
Magnitude Pruning implementation - the baseline pruning method.
Prunes weights based on their absolute magnitude (smallest weights are pruned).
This is the simplest and most widely used pruning technique.
"""

import torch
import torch.nn as nn


class MagnitudePruning:
    """
    Implements Magnitude Pruning - prunes weights with smallest absolute values.

    This is the standard baseline pruning method that removes weights
    closest to zero, regardless of their movement during training.
    """

    def __init__(
        self,
        model,
        initial_sparsity=0.0,
        target_sparsity=0.9,
        warmup_steps=0,
        pruning_frequency=100,
        ramp_end_step=3000,
        schedule="cubic",
    ):
        """
        Initialize Magnitude Pruning.

        Args:
            model: PyTorch model to prune
            initial_sparsity: Starting sparsity level (0 to 1)
            target_sparsity: Final target sparsity level (0 to 1)
            warmup_steps: Number of steps before pruning starts
            pruning_frequency: Update masks every N steps
            ramp_end_step: Step at which target sparsity is reached
            schedule: Sparsity schedule ("linear" or "cubic"). Default is "cubic"
                     to match state-of-art and bitter3-9 variants.
        """
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.warmup_steps = warmup_steps
        self.pruning_frequency = pruning_frequency
        self.ramp_end_step = ramp_end_step
        self.schedule = schedule
        self.step = 0

        # Initialize masks for prunable layers
        self.masks = {}

        # Get prunable layers (Conv2d and Linear layers)
        self.prunable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable_layers.append((name, module))

                # Initialize binary masks (1 = keep, 0 = prune)
                weight_shape = module.weight.shape
                self.masks[name] = torch.ones(
                    weight_shape, dtype=torch.float32, device=module.weight.device
                )

                # Register mask as buffer in the module
                module.register_buffer(f"pruning_mask", self.masks[name])

    def get_current_sparsity(self):
        """Calculate current sparsity level based on schedule."""
        if self.step < self.warmup_steps:
            return 0.0

        if self.step >= self.ramp_end_step:
            return self.target_sparsity

        # Calculate ramp progress
        ramp_progress = (self.step - self.warmup_steps) / (
            self.ramp_end_step - self.warmup_steps
        )

        # Apply schedule transformation
        if self.schedule == "cubic":
            # Cubic schedule: slower initial pruning, faster at the end
            # Matches bitter3-9 variants and state-of-art methods
            ramp_progress = ramp_progress**3
        # else: linear (no transformation)

        current_sparsity = (
            self.initial_sparsity
            + (self.target_sparsity - self.initial_sparsity) * ramp_progress
        )

        return current_sparsity

    def update_masks(self, iter_num=None):
        """Update binary masks based on weight magnitudes and target sparsity.

        Args:
            iter_num: Current iteration number (used to track progress)
        """
        # Use provided iter_num if available, otherwise use internal counter
        if iter_num is not None:
            self.step = iter_num

        current_sparsity = self.get_current_sparsity()

        if current_sparsity == 0.0:
            return

        # Collect all weight magnitudes to determine global threshold
        all_magnitudes = []
        for name, module in self.prunable_layers:
            # Use absolute values of weights
            magnitudes = module.weight.data.abs()
            all_magnitudes.append(magnitudes.flatten())

        all_magnitudes = torch.cat(all_magnitudes)

        # Find threshold for target sparsity
        # Prune weights with smallest magnitudes
        k = int(current_sparsity * all_magnitudes.numel())
        if k > 0:
            # Get the k-th smallest magnitude as threshold
            threshold = torch.kthvalue(all_magnitudes, k).values

            # Update masks
            for name, module in self.prunable_layers:
                # Weights with magnitude below threshold are pruned
                self.masks[name] = (module.weight.data.abs() > threshold).float()

                # Update module's mask buffer
                module.pruning_mask.data = self.masks[name]

    def apply_masks(self):
        """Apply binary masks to weights."""
        for name, module in self.prunable_layers:
            module.weight.data *= self.masks[name]

    def step_pruning(self):
        """Called at each training step to update pruning."""
        self.step += 1

        # Update masks at specified frequency
        if self.step % self.pruning_frequency == 0:
            self.update_masks()

        # Always apply masks to ensure pruned weights stay zero
        self.apply_masks()

    def get_sparsity(self):
        """Get overall sparsity of the model."""
        total_params = 0
        total_pruned = 0

        for name in self.masks:
            mask = self.masks[name]
            total_params += mask.numel()
            total_pruned += (mask == 0).sum().item()

        return total_pruned / total_params if total_params > 0 else 0.0

    def get_sparsity_stats(self):
        """Get current sparsity statistics for monitoring."""
        stats = {}
        total_params = 0
        total_pruned = 0

        for name, module in self.prunable_layers:
            mask = self.masks[name]
            num_params = mask.numel()
            num_pruned = (mask == 0).sum().item()

            total_params += num_params
            total_pruned += num_pruned

            stats[name] = {
                "sparsity": num_pruned / num_params,
                "pruned": num_pruned,
                "total": num_params,
            }

        stats["global"] = {
            "sparsity": total_pruned / total_params if total_params > 0 else 0,
            "pruned": total_pruned,
            "total": total_params,
            "target_sparsity": self.get_current_sparsity(),
        }

        return stats

    def prune_model_final(self):
        """
        Final pruning to permanently remove pruned weights.
        """
        # Apply final masks
        self.apply_masks()

        # Return final sparsity
        stats = self.get_sparsity_stats()
        return stats["global"]["sparsity"]


class PrunedForward:
    """
    Context manager to ensure masked weights during forward pass.
    """

    def __init__(self, pruning_module):
        self.pruning = pruning_module

    def __enter__(self):
        self.pruning.apply_masks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
