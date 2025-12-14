#!/usr/bin/env python3
"""
Extract weight importance scores for mobile weight packing.

Supports multiple importance metrics:
1. Weight magnitude (baseline, no optimizer needed)
2. Adam state-based (bitter7 formula) from checkpoints
3. Gradient-based importance (requires forward/backward pass)

IMPORTANT: This script enforces CPU-only execution to avoid GPU contention.
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional
from collections import defaultdict

# Force CPU before importing torch (covers NVIDIA CUDA and AMD ROCm)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

import torch

# Verify CPU-only mode
assert not torch.cuda.is_available(), "CUDA should be disabled but is available!"
DEVICE = torch.device("cpu")
print(f"[CPU ENFORCED] Running on: {DEVICE}")
print(f"[CPU ENFORCED] torch.cuda.is_available() = {torch.cuda.is_available()}")


def load_public_model(model_name: str = "openai-community/gpt2"):
    """
    Load a public HuggingFace model on CPU.

    Args:
        model_name: HuggingFace model identifier
                    Use full repo names like 'openai-community/gpt2' to avoid
                    conflicts with local directories.

    Returns:
        model, tokenizer
    """
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    print(f"Loading {model_name} on CPU...")

    # Use specific GPT2 classes for gpt2 models, Auto classes for others
    if "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    # Ensure model is on CPU
    model = model.to(DEVICE)

    # Verify all parameters are on CPU
    for name, param in model.named_parameters():
        assert (
            param.device == DEVICE
        ), f"Parameter {name} is on {param.device}, expected CPU"

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer


def compute_magnitude_importance(model) -> Dict[str, float]:
    """
    Compute importance scores based on weight magnitude.

    This is the simplest baseline - larger weights are assumed more important.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to importance scores
    """
    importance = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # L1 norm normalized by number of elements
            score = param.data.abs().mean().item()
            importance[name] = score

    return importance


def compute_bitter7_importance(model, optimizer_state: Dict) -> Dict[str, float]:
    """
    Compute importance using bitter7 formula from Adam state.

    Formula: importance = |weight| × (exp_avg_sq + ε)^0.25

    The 4th root provides conservative damping - only truly stable
    low-activity weights get low scores.

    Args:
        model: PyTorch model
        optimizer_state: Adam optimizer state dict

    Returns:
        Dictionary mapping parameter names to importance scores
    """
    importance = {}

    # Build mapping from param to state
    param_to_state = {}
    for param_id, state in optimizer_state.items():
        if isinstance(param_id, int):
            param_to_state[param_id] = state

    # Get parameter order from model
    param_list = list(model.parameters())

    for idx, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue

        state = param_to_state.get(idx, {})

        if "exp_avg_sq" in state:
            # bitter7 formula
            v = state["exp_avg_sq"]
            if isinstance(v, torch.Tensor):
                score = (param.data.abs() * (v.abs() + 1e-8) ** 0.25).mean().item()
            else:
                # Fallback if state is not a tensor
                score = param.data.abs().mean().item()
        else:
            # Fallback to magnitude
            score = param.data.abs().mean().item()

        importance[name] = score

    return importance


def compute_gradient_importance(
    model,
    tokenizer,
    num_samples: int = 10,
    seq_len: int = 64,
) -> Dict[str, float]:
    """
    Compute importance based on gradient magnitude.

    Runs forward/backward passes on random text and accumulates
    gradient magnitudes. Higher gradient = more important for loss.

    WARNING: Slow on CPU! ~10-30 seconds per sample for GPT-2.

    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        num_samples: Number of samples to accumulate gradients over
        seq_len: Sequence length for each sample

    Returns:
        Dictionary mapping parameter names to importance scores
    """
    import time

    print(
        f"Computing gradient importance ({num_samples} samples, seq_len={seq_len})..."
    )
    print("WARNING: This is slow on CPU. Expect ~10-30 seconds per sample.")

    model.train()
    gradient_accum = defaultdict(float)

    # Generate random input tokens
    vocab_size = tokenizer.vocab_size

    for i in range(num_samples):
        start = time.time()

        # Random tokens as input
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)

        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Accumulate gradient magnitudes
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_accum[name] += param.grad.abs().mean().item()

        elapsed = time.time() - start
        print(
            f"  Sample {i+1}/{num_samples}: loss={loss.item():.4f}, time={elapsed:.1f}s"
        )

    # Average over samples
    importance = {name: score / num_samples for name, score in gradient_accum.items()}

    model.eval()
    return importance


def aggregate_by_module(
    param_importance: Dict[str, float],
    aggregation: str = "mean",
) -> Dict[str, float]:
    """
    Aggregate parameter-level importance to module-level.

    Args:
        param_importance: Parameter name -> importance score
        aggregation: "mean", "max", or "sum"

    Returns:
        Module name -> importance score
    """
    module_scores = defaultdict(list)

    for param_name, score in param_importance.items():
        # Extract module name (remove .weight, .bias suffix)
        parts = param_name.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in ("weight", "bias"):
            module_name = parts[0]
        else:
            module_name = param_name

        module_scores[module_name].append(score)

    # Aggregate
    if aggregation == "mean":
        return {
            name: sum(scores) / len(scores) for name, scores in module_scores.items()
        }
    elif aggregation == "max":
        return {name: max(scores) for name, scores in module_scores.items()}
    elif aggregation == "sum":
        return {name: sum(scores) for name, scores in module_scores.items()}
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def load_checkpoint_optimizer_state(checkpoint_path: str) -> Optional[Dict]:
    """
    Load optimizer state from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file

    Returns:
        Optimizer state dict or None if not found
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "optimizer_state_dict" in checkpoint:
        return checkpoint["optimizer_state_dict"].get("state", {})
    elif "optimizer" in checkpoint:
        return checkpoint["optimizer"].get("state", {})
    else:
        print("WARNING: No optimizer state found in checkpoint")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract weight importance scores for mobile packing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name (default: openai-community/gpt2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint with optimizer state (optional)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "bitter7", "gradient", "all"],
        default="magnitude",
        help="Importance computation method",
    )
    parser.add_argument(
        "--gradient-samples",
        type=int,
        default=5,
        help="Number of samples for gradient importance (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weight_importance.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max", "sum", "none"],
        default="none",
        help="Aggregate to module level (default: none = per-parameter)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_public_model(args.model)

    results = {
        "model": args.model,
        "device": str(DEVICE),
        "method": args.method,
        "aggregation": args.aggregation,
    }

    # Compute importance based on method
    if args.method == "magnitude" or args.method == "all":
        print("\n=== Computing magnitude importance ===")
        mag_importance = compute_magnitude_importance(model)
        if args.aggregation != "none":
            mag_importance = aggregate_by_module(mag_importance, args.aggregation)
        results["magnitude"] = mag_importance
        print(f"Computed importance for {len(mag_importance)} entries")

    if args.method == "bitter7" or args.method == "all":
        print("\n=== Computing bitter7 importance ===")
        if args.checkpoint:
            opt_state = load_checkpoint_optimizer_state(args.checkpoint)
            if opt_state:
                bitter7_importance = compute_bitter7_importance(model, opt_state)
                if args.aggregation != "none":
                    bitter7_importance = aggregate_by_module(
                        bitter7_importance, args.aggregation
                    )
                results["bitter7"] = bitter7_importance
                print(f"Computed importance for {len(bitter7_importance)} entries")
            else:
                print("Skipping bitter7: no optimizer state available")
        else:
            print("Skipping bitter7: no checkpoint provided (use --checkpoint)")

    if args.method == "gradient" or args.method == "all":
        print("\n=== Computing gradient importance ===")
        grad_importance = compute_gradient_importance(
            model, tokenizer, num_samples=args.gradient_samples
        )
        if args.aggregation != "none":
            grad_importance = aggregate_by_module(grad_importance, args.aggregation)
        results["gradient"] = grad_importance
        print(f"Computed importance for {len(grad_importance)} entries")

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== Summary ===")
    for method_name in ["magnitude", "bitter7", "gradient"]:
        if method_name in results:
            scores = list(results[method_name].values())
            print(f"{method_name}:")
            print(f"  entries: {len(scores)}")
            print(f"  min: {min(scores):.6f}")
            print(f"  max: {max(scores):.6f}")
            print(f"  mean: {sum(scores)/len(scores):.6f}")


if __name__ == "__main__":
    main()
