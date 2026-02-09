"""
Sensitivity extraction from Adam optimizer state for variance-weighted RGSA.

This module extracts per-layer sensitivity summaries S from Adam's exp_avg_sq (v̂),
which serves as a diagonal Fisher proxy (per Squisher). The sensitivity values
indicate which attention pathways are expensive to compress and should receive
more budget in variance-weighted RGSA allocation.

Usage:
    from utils.sensitivity import extract_sensitivity, save_sensitivity_json

    # After training checkpoint
    sensitivity = extract_sensitivity(model, optimizer)
    save_sensitivity_json(sensitivity, "sensitivity.json", step=1000, tokens_seen=100000)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


def extract_sensitivity(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    param_patterns: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract per-layer sensitivity summaries from Adam optimizer exp_avg_sq.

    The sensitivity S_layer[l] is computed as the sum of exp_avg_sq for attention
    projection parameters (Wq, Wk, Wv, Wo) and any RGSA router/gating params.

    Args:
        model: The model being trained (used to map param names to layers)
        optimizer: Adam or AdamW optimizer with exp_avg_sq in state
        param_patterns: Optional list of patterns to match parameter names.
            Defaults to ["attn", "c_attn", "c_proj", "router", "chunk_projector"]

    Returns:
        Dictionary with:
            - 'S_layer': Tensor of shape [n_layer] with per-layer sensitivity
            - 'param_sensitivity': Dict mapping param names to their sensitivity
            - 'layer_param_map': Dict mapping layer index to param names
    """
    if param_patterns is None:
        param_patterns = ["attn", "c_attn", "c_proj", "router", "chunk_projector"]

    # Get model state dict to map param names
    param_to_layer = {}
    n_layer = 12  # Default for GPT-2 base

    # Detect n_layer from model config if available
    if hasattr(model, "config") and hasattr(model.config, "n_layer"):
        n_layer = model.config.n_layer
    elif hasattr(model, "n_layer"):
        n_layer = model.n_layer
    elif hasattr(model, "module"):
        # Handle DDP wrapped models
        if hasattr(model.module, "config") and hasattr(model.module.config, "n_layer"):
            n_layer = model.module.config.n_layer

    # Map parameters to layers
    for name, param in model.named_parameters():
        # Extract layer number from name (e.g., "transformer.h.0.attn.c_attn.weight")
        parts = name.split(".")
        layer_idx = None
        for i, part in enumerate(parts):
            if part.isdigit():
                layer_idx = int(part)
                break
            # Handle "h.N" pattern (GPT-2 style)
            if part == "h" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
            # Handle "blocks.N" pattern
            if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break

        # Check if param matches our patterns
        matches_pattern = any(pattern in name for pattern in param_patterns)
        if layer_idx is not None and matches_pattern:
            param_to_layer[name] = layer_idx

    # Extract exp_avg_sq from optimizer state
    S_layer = torch.zeros(n_layer, dtype=torch.float32)
    param_sensitivity = {}
    layer_param_map = {l: [] for l in range(n_layer)}

    for group in optimizer.param_groups:
        for param in group["params"]:
            # Find param name
            param_name = None
            for name, p in model.named_parameters():
                if p is param:
                    param_name = name
                    break

            if param_name is None:
                continue

            # Check if this param is in our mapping
            if param_name not in param_to_layer:
                continue

            layer_idx = param_to_layer[param_name]

            # Get optimizer state
            state = optimizer.state.get(param, {})
            if "exp_avg_sq" not in state:
                continue

            # Sum exp_avg_sq for this parameter
            exp_avg_sq = state["exp_avg_sq"]
            sensitivity = exp_avg_sq.sum().item()

            param_sensitivity[param_name] = sensitivity
            layer_param_map[layer_idx].append(param_name)
            S_layer[layer_idx] += sensitivity

    return {
        "S_layer": S_layer,
        "param_sensitivity": param_sensitivity,
        "layer_param_map": layer_param_map,
        "n_layer": n_layer,
    }


def save_sensitivity_json(
    sensitivity: Dict,
    filepath: str,
    step: int,
    tokens_seen: int,
    extra_info: Optional[Dict] = None,
):
    """
    Save sensitivity data to JSON file.

    Args:
        sensitivity: Output from extract_sensitivity()
        filepath: Path to save JSON
        step: Training step/iteration
        tokens_seen: Total tokens seen during training
        extra_info: Optional additional info to include
    """
    S_layer = sensitivity["S_layer"]

    # Compute summary stats
    S_np = S_layer.numpy()
    data = {
        "step": step,
        "tokens_seen": tokens_seen,
        "n_layer": sensitivity["n_layer"],
        "S_layer": S_layer.tolist(),
        "summary": {
            "mean": float(S_np.mean()),
            "std": float(S_np.std()),
            "min": float(S_np.min()),
            "max": float(S_np.max()),
            "p50": float(sorted(S_np)[len(S_np) // 2]),
            "p90": float(sorted(S_np)[int(len(S_np) * 0.9)]),
        },
        "layer_param_map": sensitivity["layer_param_map"],
    }

    if extra_info:
        data.update(extra_info)

    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True
    )
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def compute_variance_weights(
    S_layer: torch.Tensor,
    alpha: float = 1.0,
    normalize_by_median: bool = True,
    invert: bool = False,
) -> torch.Tensor:
    """
    Compute per-layer budget weights from sensitivity.

    w_l = S_l^alpha / sum(S^alpha)

    If invert=True, uses 1/S instead (protect low-sensitivity layers).

    Args:
        S_layer: Per-layer sensitivity values
        alpha: Exponent for sensitivity weighting (0 = uniform, 1 = linear)
        normalize_by_median: If True, normalize S by median for numeric stability
        invert: If True, use 1/S (allocate more to low-sensitivity layers)

    Returns:
        Tensor of weights summing to 1.0
    """
    S = S_layer.clone()

    if normalize_by_median:
        median = S.median()
        if median > 0:
            S = S / median

    # Apply alpha exponent
    if alpha == 0:
        # Uniform weights
        weights = torch.ones_like(S) / len(S)
    else:
        # Clamp to avoid numerical issues
        S = S.clamp(min=1e-8)

        # Invert if requested (protect low-sensitivity layers)
        if invert:
            S = 1.0 / S

        S_alpha = S.pow(alpha)
        weights = S_alpha / S_alpha.sum()

    return weights


def compute_per_layer_top_b(
    weights: torch.Tensor,
    top_b_base: int = 8,
    n_layer: int = 12,
    top_b_min: int = 2,
    top_b_max: int = 16,
    exact_total: Optional[int] = None,
) -> List[int]:
    """
    Compute per-layer top_b values from variance weights.

    top_b_l = clamp(round(top_b_base * w_l * n_layer), min, max)

    The scaling by n_layer ensures that average top_b equals top_b_base.

    Args:
        weights: Per-layer weights from compute_variance_weights()
        top_b_base: Base top_b value (average across layers)
        n_layer: Number of layers
        top_b_min: Minimum top_b per layer
        top_b_max: Maximum top_b per layer
        exact_total: If set, guarantee sum(top_b_l) == exact_total exactly

    Returns:
        List of per-layer top_b values
    """
    if exact_total is not None:
        return compute_per_layer_top_b_exact(
            weights, exact_total, n_layer, top_b_min, top_b_max
        )

    # Scale weights by n_layer so average allocation equals top_b_base
    scaled = weights * n_layer * top_b_base

    top_b_per_layer = []
    for i in range(n_layer):
        top_b = int(round(scaled[i].item()))
        top_b = max(top_b_min, min(top_b, top_b_max))
        top_b_per_layer.append(top_b)

    return top_b_per_layer


def compute_per_layer_top_b_exact(
    weights: torch.Tensor,
    total_budget: int,
    n_layer: int,
    top_b_min: int = 2,
    top_b_max: int = 16,
) -> List[int]:
    """
    Compute per-layer top_b with EXACT budget matching (no rounding drift).

    Algorithm:
    1. Compute raw allocations a_l = w_l * total_budget
    2. Take floor: top_b_l = floor(a_l)
    3. Distribute remaining tokens to layers with largest fractional remainders
    4. Apply min/max caps and rebalance to preserve exact sum

    Args:
        weights: Per-layer weights (must sum to 1.0)
        total_budget: Exact total budget sum(top_b_l) must equal
        n_layer: Number of layers
        top_b_min: Minimum top_b per layer
        top_b_max: Maximum top_b per layer

    Returns:
        List of per-layer top_b values with sum == total_budget
    """
    # Step 1: Compute raw allocations
    raw_alloc = (weights * total_budget).tolist()

    # Step 2: Take floor
    top_b = [int(a) for a in raw_alloc]
    remainders = [raw_alloc[i] - top_b[i] for i in range(n_layer)]

    # Step 3: Distribute remaining budget to layers with largest remainders
    remaining = total_budget - sum(top_b)
    if remaining > 0:
        # Sort by remainder descending
        sorted_indices = sorted(range(n_layer), key=lambda i: -remainders[i])
        for i in range(remaining):
            top_b[sorted_indices[i]] += 1

    # Step 4: Apply caps and rebalance
    # First pass: apply caps
    excess = 0
    deficit = 0
    for i in range(n_layer):
        if top_b[i] < top_b_min:
            deficit += top_b_min - top_b[i]
            top_b[i] = top_b_min
        elif top_b[i] > top_b_max:
            excess += top_b[i] - top_b_max
            top_b[i] = top_b_max

    # Second pass: redistribute excess/deficit
    # If we had to raise some to min, take from those above min
    if deficit > 0:
        for i in range(n_layer):
            if top_b[i] > top_b_min and deficit > 0:
                take = min(top_b[i] - top_b_min, deficit)
                top_b[i] -= take
                deficit -= take

    # If we had to lower some to max, give to those below max
    if excess > 0:
        for i in range(n_layer):
            if top_b[i] < top_b_max and excess > 0:
                give = min(top_b_max - top_b[i], excess)
                top_b[i] += give
                excess -= give

    # Final check: if caps make exact matching impossible, warn but continue
    actual_sum = sum(top_b)
    if actual_sum != total_budget:
        # Try one more rebalance pass
        diff = total_budget - actual_sum
        if diff > 0:
            # Need to add more
            for i in range(n_layer):
                if top_b[i] < top_b_max and diff > 0:
                    add = min(top_b_max - top_b[i], diff)
                    top_b[i] += add
                    diff -= add
        elif diff < 0:
            # Need to remove
            diff = -diff
            for i in range(n_layer):
                if top_b[i] > top_b_min and diff > 0:
                    remove = min(top_b[i] - top_b_min, diff)
                    top_b[i] -= remove
                    diff -= remove

    return top_b


def compute_per_head_top_b_exact(
    weights: torch.Tensor,
    total_budget: int,
    n_layer: int,
    n_head: int,
    top_b_min: int = 0,
    top_b_max: int = 16,
) -> torch.Tensor:
    """
    Compute per-head top_b with EXACT budget matching for RGSA v18.

    Given importance weights w[l,h], allocate integer top_b_{l,h} such that
    sum(top_b_{l,h}) == total_budget exactly.

    Algorithm:
    1. Compute raw allocations a_{l,h} = w_{l,h} * total_budget
    2. Take floor: top_b_{l,h} = floor(a_{l,h})
    3. Distribute remaining tokens to heads with largest fractional remainders
    4. Apply min/max caps and rebalance to preserve exact sum

    Args:
        weights: Per-head weights [n_layer, n_head] (should sum to 1.0)
        total_budget: Exact total budget sum(top_b_{l,h}) must equal
        n_layer: Number of layers
        n_head: Number of heads per layer
        top_b_min: Minimum top_b per head (default 0 = can drop far-context entirely)
        top_b_max: Maximum top_b per head

    Returns:
        Tensor [n_layer, n_head] of integer top_b values with sum == total_budget
    """
    assert weights.shape == (n_layer, n_head), f"weights shape mismatch: {weights.shape}"

    # Step 1: Compute raw allocations
    raw_alloc = weights * total_budget  # [n_layer, n_head]

    # Step 2: Take floor
    top_b = raw_alloc.floor().int()
    remainders = raw_alloc - top_b.float()

    # Step 3: Distribute remaining budget to heads with largest remainders
    remaining = total_budget - top_b.sum().item()
    if remaining > 0:
        # Flatten, sort by remainder descending, add 1 to top 'remaining' entries
        flat_remainders = remainders.view(-1)
        flat_top_b = top_b.view(-1)
        _, sorted_indices = torch.sort(flat_remainders, descending=True)
        for i in range(int(remaining)):
            flat_top_b[sorted_indices[i]] += 1

    # Step 4: Apply caps and rebalance
    # First pass: apply caps
    excess = 0
    deficit = 0
    for l in range(n_layer):
        for h in range(n_head):
            if top_b[l, h] < top_b_min:
                deficit += top_b_min - top_b[l, h].item()
                top_b[l, h] = top_b_min
            elif top_b[l, h] > top_b_max:
                excess += top_b[l, h].item() - top_b_max
                top_b[l, h] = top_b_max

    # Second pass: redistribute excess/deficit
    if deficit > 0:
        for l in range(n_layer):
            for h in range(n_head):
                if top_b[l, h] > top_b_min and deficit > 0:
                    take = min(top_b[l, h].item() - top_b_min, deficit)
                    top_b[l, h] -= take
                    deficit -= take

    if excess > 0:
        for l in range(n_layer):
            for h in range(n_head):
                if top_b[l, h] < top_b_max and excess > 0:
                    give = min(top_b_max - top_b[l, h].item(), excess)
                    top_b[l, h] += give
                    excess -= give

    # Final check and fixup
    actual_sum = top_b.sum().item()
    diff = total_budget - actual_sum
    if diff > 0:
        for l in range(n_layer):
            for h in range(n_head):
                if top_b[l, h] < top_b_max and diff > 0:
                    add = min(top_b_max - top_b[l, h].item(), diff)
                    top_b[l, h] += add
                    diff -= add
    elif diff < 0:
        diff = -diff
        for l in range(n_layer):
            for h in range(n_head):
                if top_b[l, h] > top_b_min and diff > 0:
                    remove = min(top_b[l, h].item() - top_b_min, diff)
                    top_b[l, h] -= remove
                    diff -= remove

    return top_b


def compute_head_weights_from_metrics(
    metrics_tensor: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert head-level metrics to allocation weights.

    w[l,h] = (metrics[l,h] + eps)^gamma / sum((metrics + eps)^gamma)

    Args:
        metrics_tensor: Per-head metric values [n_layer, n_head]
        gamma: Exponent (1.0 = linear, 0.5 = sqrt)
        eps: Small constant for numerical stability

    Returns:
        Weights tensor [n_layer, n_head] summing to 1.0
    """
    m = (metrics_tensor + eps).pow(gamma)
    return m / m.sum()


def log_sensitivity_to_wandb(
    sensitivity: Dict,
    step: int,
    prefix: str = "sensitivity",
):
    """
    Log sensitivity data to Weights & Biases.

    Args:
        sensitivity: Output from extract_sensitivity()
        step: Training step for logging
        prefix: Metric prefix
    """
    try:
        import wandb

        if wandb.run is None:
            return

        S_layer = sensitivity["S_layer"]
        n_layer = sensitivity["n_layer"]

        # Log per-layer values
        for i in range(n_layer):
            wandb.log({f"{prefix}/S_layer_{i}": S_layer[i].item()}, step=step)

        # Log summary stats
        S_np = S_layer.numpy()
        wandb.log(
            {
                f"{prefix}/mean": S_np.mean(),
                f"{prefix}/std": S_np.std(),
                f"{prefix}/max": S_np.max(),
                f"{prefix}/min": S_np.min(),
                f"{prefix}/ratio_max_min": S_np.max() / max(S_np.min(), 1e-8),
            },
            step=step,
        )

        # Log heatmap as a bar chart
        import numpy as np

        data = [[i, S_layer[i].item()] for i in range(n_layer)]
        table = wandb.Table(data=data, columns=["layer", "sensitivity"])
        wandb.log(
            {
                f"{prefix}/heatmap": wandb.plot.bar(
                    table, "layer", "sensitivity", title="Per-Layer Sensitivity"
                )
            },
            step=step,
        )

    except ImportError:
        pass  # wandb not available
    except Exception as e:
        print(f"Warning: Failed to log sensitivity to W&B: {e}")


def load_sensitivity_json(filepath: str) -> Dict:
    """Load sensitivity data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Convert S_layer back to tensor
    data["S_layer"] = torch.tensor(data["S_layer"], dtype=torch.float32)
    return data


def compute_top_b_per_layer_from_file(
    sensitivity_path: str,
    top_b_base: int = 8,
    alpha: float = 1.0,
    top_b_min: int = 2,
    top_b_max: int = 16,
    invert: bool = False,
) -> List[int]:
    """
    Load sensitivity from file and compute per-layer top_b allocation.

    Args:
        sensitivity_path: Path to sensitivity.json file
        top_b_base: Base top_b value (average across layers)
        alpha: Variance weighting exponent (0=uniform, 1=linear)
        top_b_min: Minimum top_b per layer
        top_b_max: Maximum top_b per layer
        invert: If True, use 1/S (allocate more to low-sensitivity layers)

    Returns:
        List of per-layer top_b values
    """
    data = load_sensitivity_json(sensitivity_path)
    S_layer = data["S_layer"]
    n_layer = data["n_layer"]

    weights = compute_variance_weights(S_layer, alpha=alpha, invert=invert)
    top_b_per_layer = compute_per_layer_top_b(
        weights,
        top_b_base=top_b_base,
        n_layer=n_layer,
        top_b_min=top_b_min,
        top_b_max=top_b_max,
    )

    return top_b_per_layer
