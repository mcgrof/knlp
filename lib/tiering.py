"""
Hierarchical Memory Tiering

Implements memory tiering strategies to place model weights across
different memory tiers (HBM, CPU RAM, SSD) based on Adam optimizer
state analysis.

Supports:
- Adam state-based tier assignment (momentum/variance analysis)
- Emulated tiering with realistic latency injection
- Real offloading using PyTorch hooks and device_map
- Tier hints JSON generation for inference benchmarking
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np


@dataclass
class TierSpec:
    """Specification for a memory tier."""

    name: str
    setup_us: float  # Setup/latency overhead in microseconds
    bandwidth_gb_s: float  # Effective bandwidth in GB/s

    def latency_for_bytes(self, n_bytes: int) -> float:
        """
        Calculate simulated latency in seconds for this tier.

        Args:
            n_bytes: Number of bytes to transfer

        Returns:
            Latency in seconds (setup + transfer)
        """
        transfer_s = n_bytes / (self.bandwidth_gb_s * 1e9)
        setup_s = self.setup_us * 1e-6
        return setup_s + transfer_s


# Conservative tier specifications based on real hardware
# HBM: Modern GPU memory (A100/H100/W7900)
# CPU: DDR5 over PCIe 4.0/5.0
# SSD: NVMe Gen4 SSD over PCIe
TIER_HBM = TierSpec(name="HBM", setup_us=1.0, bandwidth_gb_s=800.0)
TIER_CPU = TierSpec(name="CPU", setup_us=5.0, bandwidth_gb_s=150.0)
TIER_SSD = TierSpec(name="SSD", setup_us=30.0, bandwidth_gb_s=10.0)


class AdamStateTierAnalyzer:
    """
    Analyze Adam optimizer states to determine tier placement.

    Strategy: Weights with high momentum/variance magnitude are frequently
    updated and should stay in fast memory (HBM). Stable weights can be
    offloaded to slower tiers (CPU, SSD).
    """

    def __init__(
        self,
        hbm_threshold: float = 0.3,
        cpu_threshold: float = 0.5,
    ):
        """
        Initialize tier analyzer.

        Args:
            hbm_threshold: Fraction of weights to keep in HBM (top percentile)
            cpu_threshold: Fraction of weights in CPU tier (middle percentile)
                          Remainder goes to SSD
        """
        self.hbm_threshold = hbm_threshold
        self.cpu_threshold = cpu_threshold

    def analyze_optimizer_states(
        self, optimizer: torch.optim.Optimizer, model: nn.Module
    ) -> Dict[str, str]:
        """
        Analyze Adam optimizer states to assign tiers to modules.

        Args:
            optimizer: Adam/AdamW optimizer with state
            model: Model being optimized

        Returns:
            Dictionary mapping module names to tier names
            {"transformer.h.0.attn": "HBM", "transformer.h.5.mlp": "CPU", ...}
        """
        # Collect per-parameter activity scores
        param_scores = {}

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in optimizer.state:
                    continue

                state = optimizer.state[p]

                # Use momentum and variance magnitude as activity proxy
                # High momentum/variance = frequently updated = keep in HBM
                score = 0.0

                if "exp_avg" in state:  # momentum (first moment)
                    score += state["exp_avg"].abs().mean().item()

                if "exp_avg_sq" in state:  # variance (second moment)
                    score += state["exp_avg_sq"].sqrt().mean().item()

                param_scores[p] = score

        # Map parameters to module names
        param_to_module = {}
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                param_to_module[param] = name

        # Aggregate scores by module
        module_scores = {}
        for param, score in param_scores.items():
            module_name = param_to_module.get(param)
            if module_name:
                if module_name not in module_scores:
                    module_scores[module_name] = []
                module_scores[module_name].append(score)

        # Average scores per module
        module_avg_scores = {
            name: np.mean(scores) for name, scores in module_scores.items()
        }

        # Sort modules by score (descending)
        sorted_modules = sorted(
            module_avg_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Assign tiers based on thresholds
        tier_assignments = {}
        n_modules = len(sorted_modules)
        hbm_cutoff = int(n_modules * self.hbm_threshold)
        cpu_cutoff = int(n_modules * self.cpu_threshold)

        for idx, (module_name, score) in enumerate(sorted_modules):
            if idx < hbm_cutoff:
                tier_assignments[module_name] = "HBM"
            elif idx < cpu_cutoff:
                tier_assignments[module_name] = "CPU"
            else:
                tier_assignments[module_name] = "SSD"

        return tier_assignments

    def save_tier_hints(self, tier_assignments: Dict[str, str], output_path: str):
        """
        Save tier assignments to JSON file.

        Args:
            tier_assignments: Module name -> tier name mapping
            output_path: Path to write JSON file
        """
        with open(output_path, "w") as f:
            json.dump(tier_assignments, f, indent=2)


def load_tier_hints(json_path: str) -> Dict[str, str]:
    """
    Load tier assignments from JSON file.

    Args:
        json_path: Path to tier hints JSON

    Returns:
        Dictionary mapping module names to tier names
    """
    with open(json_path, "r") as f:
        return json.load(f)


class EmulatedTiering:
    """
    Emulated tiering with realistic latency injection.

    Uses forward hooks to inject delays based on tier placement,
    without actually moving weights. Allows testing tier strategies
    without specialized hardware.
    """

    def __init__(self, tier_assignments: Dict[str, str]):
        """
        Initialize emulated tiering.

        Args:
            tier_assignments: Module name -> tier name mapping
        """
        self.tier_assignments = tier_assignments
        self.tier_specs = {"HBM": TIER_HBM, "CPU": TIER_CPU, "SSD": TIER_SSD}
        self.hooks = []

    def _tier_latency_hook(self, module, inputs, output):
        """Forward hook that injects tier latency."""
        tier_name = getattr(module, "_tier_name", None)
        if tier_name is None or tier_name == "HBM":
            return  # No delay for HBM or unassigned

        tier_spec = self.tier_specs.get(tier_name)
        if tier_spec is None:
            return

        # Calculate bytes from output tensor(s)
        if isinstance(output, torch.Tensor):
            n_bytes = output.numel() * output.element_size()
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            n_bytes = sum(t.numel() * t.element_size() for t in output)
        else:
            return

        # Inject delay (blocks CPU, simulates wall-clock impact)
        delay_s = tier_spec.latency_for_bytes(n_bytes)
        time.sleep(delay_s)

    def install(self, model: nn.Module):
        """
        Install emulated tiering hooks on model.

        Args:
            model: Model to instrument
        """
        for name, module in model.named_modules():
            tier_name = self.tier_assignments.get(name)
            if tier_name:
                # Attach tier metadata
                module._tier_name = tier_name

                # Register forward hook
                handle = module.register_forward_hook(self._tier_latency_hook)
                self.hooks.append(handle)

    def remove(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


class RealOffloadTiering:
    """
    Real offloading using PyTorch device_map and hooks.

    Actually moves weights to CPU/disk and loads them just before use.
    Reduces GPU memory but may impact performance.
    """

    def __init__(self, tier_assignments: Dict[str, str], offload_dir: str = "offload"):
        """
        Initialize real offloading.

        Args:
            tier_assignments: Module name -> tier name mapping
            offload_dir: Directory for disk offloading
        """
        self.tier_assignments = tier_assignments
        self.offload_dir = offload_dir
        self.hooks = []

    def _create_offload_hook(self, module, device_map: str):
        """
        Create hook to move weights to/from GPU.

        Args:
            module: Module to offload
            device_map: Target device ("cpu" or "disk")
        """
        original_device = next(module.parameters()).device

        def pre_forward_hook(mod, inputs):
            # Move weights to GPU before forward
            mod.to(original_device)

        def post_forward_hook(mod, inputs, output):
            # Move weights back to offload device after forward
            if device_map == "cpu":
                mod.to("cpu")
            elif device_map == "disk":
                # For disk offloading, move to CPU for now
                # Full disk offload requires more sophisticated handling
                mod.to("cpu")

        return pre_forward_hook, post_forward_hook

    def install(self, model: nn.Module):
        """
        Install real offloading hooks on model.

        Args:
            model: Model to instrument
        """
        for name, module in model.named_modules():
            tier_name = self.tier_assignments.get(name)

            # Only offload CPU and SSD tiers
            if tier_name in ("CPU", "SSD"):
                device_map = "cpu" if tier_name == "CPU" else "disk"

                pre_hook, post_hook = self._create_offload_hook(module, device_map)

                # Register hooks
                pre_handle = module.register_forward_pre_hook(pre_hook)
                post_handle = module.register_forward_hook(post_hook)

                self.hooks.append(pre_handle)
                self.hooks.append(post_handle)

    def remove(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


def create_tiering_system(
    tier_assignments: Dict[str, str],
    mode: str = "emulated",
    offload_dir: str = "offload",
):
    """
    Factory function to create appropriate tiering system.

    Args:
        tier_assignments: Module name -> tier name mapping
        mode: "emulated" or "real"
        offload_dir: Directory for disk offloading (real mode only)

    Returns:
        Tiering system (EmulatedTiering or RealOffloadTiering)
    """
    if mode == "emulated":
        return EmulatedTiering(tier_assignments)
    elif mode == "real":
        return RealOffloadTiering(tier_assignments, offload_dir)
    else:
        raise ValueError(f"Unknown tiering mode: {mode}")
