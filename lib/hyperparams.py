#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Automatic hyperparameter detection based on GPU hardware.

This module automatically selects optimal batch size and gradient accumulation
based on detected GPU type, memory, count, and torch.compile status. This
eliminates the need for GPU-specific defconfigs and prevents accidentally
using wrong hyperparameters for available hardware.

Supports different model types with appropriate memory scaling factors.
"""

import torch


def get_gpu_info():
    """Get GPU information for auto-detection.

    Returns:
        dict: GPU info with keys:
            - gpu_name: str (e.g., "NVIDIA B200")
            - gpu_count: int (number of homogeneous GPUs for training)
            - gpu_mem_gb: float (total memory per GPU in GB)
            - gpu_free_gb: float (free memory per GPU in GB)
            - has_gpu: bool (True if CUDA available)
            - all_gpu_names: list (names of all detected GPUs)
    """
    if not torch.cuda.is_available():
        return {
            "gpu_name": "CPU",
            "gpu_count": 0,
            "gpu_mem_gb": 0.0,
            "gpu_free_gb": 0.0,
            "has_gpu": False,
            "all_gpu_names": [],
        }

    # Check all GPUs to determine if they're homogeneous
    total_gpus = torch.cuda.device_count()
    all_gpu_names = [
        torch.cuda.get_device_properties(i).name for i in range(total_gpus)
    ]

    # Only count GPUs for multi-GPU training if they're all identical
    # This handles cases like W7900 + integrated GPU (only use W7900)
    primary_gpu = torch.cuda.get_device_properties(0)
    if total_gpus > 1:
        # Check if all GPUs have the same name
        if len(set(all_gpu_names)) == 1:
            # All GPUs identical - use all for training
            gpu_count = total_gpus
        else:
            # Mixed GPUs - assume single-GPU training on GPU 0
            gpu_count = 1
    else:
        gpu_count = 1

    # Get free memory (accounts for other processes using GPU)
    free_mem, total_mem = torch.cuda.mem_get_info(0)

    return {
        "gpu_name": primary_gpu.name,
        "gpu_count": gpu_count,
        "gpu_mem_gb": total_mem / 1e9,
        "gpu_free_gb": free_mem / 1e9,
        "has_gpu": True,
        "all_gpu_names": all_gpu_names,
    }


def auto_detect_hyperparams(config, target_effective_batch=None, model_type="gpt2"):
    """Automatically detect optimal hyperparameters based on GPU.

    Args:
        config: Config object with COMPILE_MODEL attribute
        target_effective_batch: Target effective batch size (default varies by model)
        model_type: Model type for memory scaling ("gpt2", "resnet18", "resnet50", "lenet5")

    Returns:
        dict: Hyperparameters with keys:
            - batch_size: int
            - gradient_accumulation: int
            - effective_batch: int (batch_size × gradient_accumulation × gpu_count)
            - gpu_info: dict (from get_gpu_info())
            - rationale: str (explanation of choices)
    """
    gpu_info = get_gpu_info()

    # Check if MLA is enabled (requires ~40% more memory than standard GPT-2)
    # MLA uses extra latent projection layers and different activation patterns
    enable_mla = getattr(config, "ENABLE_MLA", False)
    if enable_mla in ("y", True):
        mla_scale = 0.6  # Conservative: reduce batch by 40% for MLA
    else:
        mla_scale = 1.0

    # Model-specific defaults and memory scaling
    # GPT-2: Large memory footprint per sample (activations, attention matrices)
    # ResNet: Medium memory footprint (image batches, conv activations)
    # LeNet: Small memory footprint (tiny model, small images)
    #
    # Strategy: Target per-GPU effective batch of ~256 for optimal convergence
    # Empirical data (finewebedu, 2h training, GPT-2 124M):
    #   - A100 (per_gpu=512): 201 iters, ppl=964 (POOR)
    #   - W7900 (per_gpu=256): 601 iters, ppl=336 (GOOD)
    #   - B200x4 (per_gpu=256): 4470 iters, ppl=38 (EXCELLENT)
    model_configs = {
        "gpt2": {
            "default_per_gpu_target": 256,  # Target per-GPU effective batch
            "scale_factor": 1.0,  # Baseline
            "grad_acc_attr": "GPT2_GRADIENT_ACCUMULATION",
        },
        "resnet18": {
            "default_per_gpu_target": 256,
            "scale_factor": 2.0,  # ResNet uses ~half memory per sample vs GPT-2
            "grad_acc_attr": "GRADIENT_ACCUMULATION",
        },
        "resnet50": {
            "default_per_gpu_target": 128,
            "scale_factor": 1.5,  # ResNet-50 larger than ResNet-18 but smaller than GPT-2
            "grad_acc_attr": "GRADIENT_ACCUMULATION",
        },
        "lenet5": {
            "default_per_gpu_target": 512,
            "scale_factor": 4.0,  # LeNet very small, can fit huge batches
            "grad_acc_attr": "GRADIENT_ACCUMULATION",
        },
    }

    model_cfg = model_configs.get(model_type.lower(), model_configs["gpt2"])

    # Calculate target effective batch
    # If user provides explicit target (non-zero), use it; otherwise use per-GPU target × gpu_count
    gpu_count = gpu_info["gpu_count"]
    if target_effective_batch is not None and target_effective_batch > 0:
        target_eff = target_effective_batch
    else:
        # Use per-GPU target scaled by number of GPUs (default: 0 or None)
        per_gpu_target = model_cfg["default_per_gpu_target"]
        target_eff = per_gpu_target * gpu_count
    scale = model_cfg["scale_factor"]
    compile_on = getattr(config, "COMPILE_MODEL", "y") in ("y", True)

    # CPU fallback - CPUs have much more RAM than GPUs (32-256GB typical)
    # Use model scale factor: larger models need smaller batches
    if not gpu_info["has_gpu"]:
        # Base CPU batch size (for GPT-2 baseline)
        cpu_base_batch = 16
        # Scale by model factor (LeNet-5 gets 64, ResNet-18 gets 32, etc.)
        cpu_batch = int(cpu_base_batch * scale)
        cpu_grad_acc = max(1, target_eff // cpu_batch)
        return {
            "batch_size": cpu_batch,
            "gradient_accumulation": cpu_grad_acc,
            "effective_batch": cpu_batch * cpu_grad_acc,
            "gpu_info": gpu_info,
            "rationale": f"CPU mode: batch={cpu_batch} (base={cpu_base_batch} × {scale}x scale), grad_acc={cpu_grad_acc}",
            "grad_acc_attr": model_cfg["grad_acc_attr"],
        }

    gpu_name = gpu_info["gpu_name"]
    gpu_mem_gb = gpu_info["gpu_mem_gb"]
    gpu_free_gb = gpu_info["gpu_free_gb"]
    gpu_count = gpu_info["gpu_count"]

    # Use FREE memory for heuristics to account for other GPU processes
    # Conservative: use 80% of free memory as safety margin
    usable_mem_gb = gpu_free_gb * 0.8

    # Heuristic table for batch size selection (for GPT-2 baseline)
    # Format: (gpu_pattern, min_free_mem_gb) -> (batch_with_compile, batch_without_compile)
    # CRITICAL: These are based on AVAILABLE memory, not total memory
    # Note: torch.compile() creates significant memory overhead during compilation phase
    # Compiled batches must be conservative to avoid OOM during graph construction
    # These will be scaled by model_type scale_factor
    base_heuristics = [
        # Very high free memory (128GB+)
        (None, 128, (128, 128)),
        # High free memory (64GB+) - H100/A100 80GB
        (None, 64, (64, 64)),
        # Good free memory (32GB+) - W7900/A100 if mostly free
        (None, 32, (24, 16)),
        # Medium free memory (16GB+) - A10G or W7900 with other processes
        (None, 16, (12, 8)),
        # Low free memory (8GB+)
        (None, 8, (8, 4)),
        # Very low free memory (4GB+)
        (None, 4, (4, 2)),
        # Fallback for minimal memory
        (None, 0, (2, 1)),
    ]

    # Scale batch sizes based on model type and MLA
    # Apply both model scale (ResNet vs GPT-2) and MLA scale (MLA vs vanilla GPT-2)
    combined_scale = scale * mla_scale
    heuristics = [
        (patterns, min_free, (int(bc * combined_scale), int(bnc * combined_scale)))
        for patterns, min_free, (bc, bnc) in base_heuristics
    ]

    # Find matching heuristic based on USABLE memory
    batch_size = None
    for patterns, min_free, (batch_compile, batch_no_compile) in heuristics:
        # Check GPU name pattern match
        if patterns is not None:
            if not any(pattern in gpu_name for pattern in patterns):
                continue

        # Check FREE memory requirement (with safety margin)
        if usable_mem_gb >= min_free:
            batch_size = batch_compile if compile_on else batch_no_compile
            break

    # Fallback (shouldn't happen with table above)
    if batch_size is None:
        batch_size = int(4 * scale) if compile_on else int(8 * scale)

    # Compute gradient accumulation to hit target effective batch
    # effective_batch = batch_size × gradient_accumulation × gpu_count
    per_gpu_batch = target_eff // gpu_count
    gradient_accumulation = max(1, per_gpu_batch // batch_size)

    # Recompute actual effective batch (may differ slightly due to rounding)
    effective_batch = batch_size * gradient_accumulation * gpu_count

    # Build rationale with GPU info
    all_gpu_names = gpu_info.get("all_gpu_names", [])
    total_gpus = len(all_gpu_names)
    if total_gpus > 1 and gpu_count == 1:
        # Mixed GPUs - show warning
        gpu_str = (
            f"{gpu_name} (detected {total_gpus} GPUs but using 1 due to mixed models)"
        )
    elif gpu_count > 1:
        gpu_str = f"{gpu_name} × {gpu_count}"
    else:
        gpu_str = gpu_name

    # Build rationale with MLA info if applicable
    mla_info = "+MLA" if mla_scale < 1.0 else ""
    rationale = (
        f"GPU: {gpu_str} ({gpu_mem_gb:.1f}GB total, {gpu_free_gb:.1f}GB free), "
        f"model={model_type}{mla_info}, compile={'ON' if compile_on else 'OFF'} → "
        f"batch={batch_size}, grad_acc={gradient_accumulation} "
        f"(per_gpu_eff={per_gpu_batch}, total_eff={effective_batch}, target={target_eff})"
    )

    return {
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch": effective_batch,
        "gpu_info": gpu_info,
        "rationale": rationale,
        "grad_acc_attr": model_cfg["grad_acc_attr"],
    }


def auto_detect_compile(config, verbose=True):
    """Automatically detect whether to enable torch.compile() based on GPU.

    Args:
        config: Config object with COMPILE_AUTO attribute
        verbose: If True, print compile detection rationale

    Returns:
        bool: True if compile should be enabled, False otherwise
    """
    gpu_info = get_gpu_info()

    if not gpu_info["has_gpu"]:
        if verbose:
            print("Compile: Disabled (CPU mode)")
        return False

    gpu_name = gpu_info["gpu_name"]

    # Blacklist: GPUs with known torch.compile issues
    blacklist = [
        "MI210",  # AMD: ROCm torch.compile instability
    ]

    for pattern in blacklist:
        if pattern in gpu_name:
            if verbose:
                print(
                    f"Compile: Disabled (GPU '{gpu_name}' has known torch.compile issues)"
                )
            return False

    # Default: Enable for NVIDIA and other AMD GPUs
    if verbose:
        print(f"Compile: Enabled (GPU '{gpu_name}' has good torch.compile support)")
    return True


def apply_hyperparams(config, verbose=True, model_type="gpt2"):
    """Apply hyperparameters to config based on AUTO/MANUAL mode.

    If AUTO mode: Detect and set batch_size and gradient_accumulation
    If MANUAL mode: Use existing config values

    Args:
        config: Config object to modify in-place
        verbose: If True, print hyperparameter selection rationale
        model_type: Model type for memory scaling ("gpt2", "resnet18", "resnet50", "lenet5")

    Returns:
        dict: Hyperparameter info (only in AUTO mode, None otherwise)
    """
    # Check if AUTO mode is enabled for hyperparams
    hyper_auto = getattr(config, "HYPER_PARAM_AUTO", "y") in ("y", True)

    if not hyper_auto:
        if verbose:
            batch = getattr(config, "BATCH_SIZE", "?")
            # Try model-specific grad_acc attribute first, fall back to generic
            grad_acc_attrs = [
                "GPT2_GRADIENT_ACCUMULATION",
                "GRADIENT_ACCUMULATION",
            ]
            grad_acc = "?"
            for attr in grad_acc_attrs:
                grad_acc = getattr(config, attr, None)
                if grad_acc is not None:
                    break
            if grad_acc is None:
                grad_acc = "?"
            print(f"Hyperparams: MANUAL mode (batch={batch}, grad_acc={grad_acc})")
        return None
    else:
        # AUTO mode: detect and apply
        target_eff = getattr(config, "TARGET_EFFECTIVE_BATCH", None)
        if isinstance(target_eff, str):
            target_eff = int(target_eff)

        params = auto_detect_hyperparams(
            config, target_effective_batch=target_eff, model_type=model_type
        )

        # Apply to config
        config.BATCH_SIZE = params["batch_size"]

        # Set gradient accumulation using model-specific attribute
        setattr(config, params["grad_acc_attr"], params["gradient_accumulation"])

        if verbose:
            print(f"Hyperparams: AUTO mode - {params['rationale']}")

    # Check if AUTO mode is enabled for compile
    compile_auto = getattr(config, "COMPILE_AUTO", "y") in ("y", True)

    if compile_auto:
        # AUTO mode: detect and apply
        should_compile = auto_detect_compile(config, verbose=verbose)
        config.COMPILE_MODEL = "y" if should_compile else "n"
    else:
        if verbose:
            compile_status = getattr(config, "COMPILE_MODEL", "?")
            print(f"Compile: MANUAL mode (compile={compile_status})")

    return params if hyper_auto else None
