"""
Pruning factory module.

Provides factory function for creating pruners based on pruning method.
"""

from lib.magnitude_pruning import MagnitudePruning
from lib.movement_pruning import MovementPruning


def create_pruner(
    model, pruning_method: str, target_sparsity: float = 0.5, warmup_steps: int = 1000
):
    """
    Create pruner based on pruning method.

    Args:
        model: PyTorch model to prune
        pruning_method: Pruning method ("magnitude", "movement")
        target_sparsity: Target sparsity level (0.0 to 1.0)
        warmup_steps: Number of warmup steps before pruning starts

    Returns:
        Pruner object (MagnitudePruning or MovementPruning)

    Raises:
        ValueError: If pruning method is unknown
    """
    if pruning_method == "magnitude":
        return MagnitudePruning(
            model=model,
            target_sparsity=target_sparsity,
            warmup_steps=warmup_steps,
        )
    elif pruning_method == "movement":
        return MovementPruning(
            model=model,
            target_sparsity=target_sparsity,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
