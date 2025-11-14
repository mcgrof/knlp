"""
Pruning utilities for neural network weight pruning.

Provides factory function to create pruners based on pruning method.
"""

import torch
import torch.nn as nn
from typing import Any


def create_pruner(
    model: nn.Module,
    pruning_method: str,
    target_sparsity: float,
    args: Any,
):
    """
    Create a pruner based on the specified pruning method.

    Args:
        model: The model to prune
        pruning_method: Pruning method ("magnitude", "movement", etc.)
        target_sparsity: Target sparsity level (0.0 to 1.0)
        args: Additional arguments

    Returns:
        Pruner instance or None
    """
    if pruning_method == "magnitude":
        from lib.magnitude_pruning import MagnitudePruning

        pruning_warmup = getattr(args, "pruning_warmup", 1000)
        return MagnitudePruning(
            model=model,
            target_sparsity=target_sparsity,
            warmup_steps=pruning_warmup,
        )

    elif pruning_method == "movement":
        from lib.movement_pruning import MovementPruning

        pruning_warmup = getattr(args, "pruning_warmup", 1000)
        return MovementPruning(
            model=model,
            target_sparsity=target_sparsity,
            warmup_steps=pruning_warmup,
        )

    elif pruning_method == "state":
        # State pruning is handled by AdamWPrune optimizer, not here
        return None

    elif pruning_method == "none":
        return None

    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
