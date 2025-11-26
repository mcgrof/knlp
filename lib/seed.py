#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Deterministic seeding utilities for reproducible experiments.

Provides functions to set random seeds across Python, NumPy, and PyTorch
for reproducible training runs.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic ops (may hurt performance)

    Example:
        >>> set_seed(42)  # All runs with seed 42 will be identical
        >>> set_seed(43)  # Different seed, different run
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic operations (slower but reproducible)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations still have non-deterministic behavior
        # See: https://pytorch.org/docs/stable/notes/randomness.html
    else:
        # Allow cuDNN to optimize (faster but not deterministic)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_dataloader_seed(base_seed: int, epoch: int = 0, worker_id: int = 0) -> int:
    """
    Generate deterministic seed for dataloader workers.

    Each combination of (base_seed, epoch, worker_id) produces a unique seed,
    ensuring reproducible data ordering across runs.

    Args:
        base_seed: Base random seed for the experiment
        epoch: Current epoch number
        worker_id: DataLoader worker ID

    Returns:
        Deterministic seed for this worker

    Example:
        >>> def worker_init_fn(worker_id):
        ...     seed = get_dataloader_seed(args.seed, epoch, worker_id)
        ...     np.random.seed(seed)
        ...     random.seed(seed)
    """
    # Combine seeds in a way that avoids collisions
    return base_seed + epoch * 1000 + worker_id


def seed_everything(seed: int, deterministic: bool = True):
    """
    Alias for set_seed() for compatibility with common ML libraries.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic ops
    """
    set_seed(seed, deterministic)
