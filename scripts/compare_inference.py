#!/usr/bin/env python3
"""
Compare inference speed across different GPT-2 model architectures.

Supports three modes:
  1. No checkpoints: Compare all model types with random initialization
  2. One checkpoint: Compare loaded model against all types with random init
  3. Two checkpoints: Compare two specific loaded models

Automatically detects model type from checkpoint metadata.
"""

import torch
import time
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import torch.nn as nn

from gpt2.model import GPT2, GPTConfig
from ra import RA_MLA_Config
import ra as ra_module


def discover_gpt2_models():
    """
    Auto-discover all GPT2 model classes using convention-based introspection.

    Models must follow these conventions:
    1. Named GPT2* (enforces naming convention)
    2. Inherit from nn.Module
    3. Implement get_num_params() method
    4. Use 'config: GPTConfig' or 'cfg: RA_MLA_Config' as first __init__ parameter

    Returns:
        dict: Maps model name to (model_class, config_type)
    """
    models = {}

    # Discover from gpt2.model module
    models["GPT2"] = (GPT2, "gpt")

    # Discover from ra module
    for name, obj in inspect.getmembers(ra_module, inspect.isclass):
        # Must match naming convention
        if not name.startswith("GPT2"):
            continue
        # Must be a PyTorch model
        if not issubclass(obj, nn.Module):
            continue
        # Must have standard interface
        if not hasattr(obj, "get_num_params"):
            continue

        # Detect config type from __init__ signature
        try:
            sig = inspect.signature(obj.__init__)
            params = list(sig.parameters.keys())

            if "config" in params:
                config_type = "gpt"
            elif "cfg" in params:
                config_type = "mla"
            else:
                # Skip models without standard config parameter
                continue

            models[name] = (obj, config_type)
        except Exception:
            # Skip if we can't inspect the signature
            continue

    return models


# Auto-discover all GPT2 models
MODEL_REGISTRY = discover_gpt2_models()


def create_config(config_type: str, **kwargs):
    """Create appropriate config object based on type."""
    if config_type == "gpt":
        config = GPTConfig.from_name(kwargs.get("model_name", "gpt2"))
        config.block_size = kwargs.get("block_size", 1024)
        config.dropout = kwargs.get("dropout", 0.0)
        config.bias = kwargs.get("bias", True)
        return config
    elif config_type == "mla":
        return RA_MLA_Config(
            d_model=kwargs.get("d_model", 768),
            n_heads=kwargs.get("n_heads", 12),
            head_dim=kwargs.get("head_dim", 64),
            d_latent=kwargs.get("d_latent", 256),
            block_size=kwargs.get("block_size", 1024),
            n_layers=kwargs.get("n_layers", 12),
            dropout=kwargs.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def detect_model_type(checkpoint: Dict) -> Tuple[str, Dict]:
    """
    Detect model type from checkpoint.

    Returns:
        (model_type_name, config_dict)
    """
    # Try to get model type from checkpoint metadata
    if "model_type" in checkpoint:
        return checkpoint["model_type"], checkpoint.get("config", {})

    # Infer from state dict keys
    state_dict = checkpoint.get("model", checkpoint)
    keys = list(state_dict.keys())

    # Check for MLA-specific keys
    has_mla = any("kv_latent" in k for k in keys)
    has_ra = any("alternation_logits" in k for k in keys)
    has_mlakv = any("k_compress" in k or "v_compress" in k for k in keys)

    if has_mla and has_ra and has_mlakv:
        # Could be GPT2_MLA_RA_KV or variants
        if any("mlp_gate_proj" in k for k in keys):
            return "GPT2_MLA_RA_KVM", {}
        return "GPT2_MLA_RA_KV", {}
    elif has_mla and has_ra:
        return "GPT2_MLA_RA", {}
    elif has_mla and has_mlakv:
        return "GPT2_MLA_KV", {}
    elif has_mla:
        return "GPT2_MLA", {}
    elif has_ra:
        return "GPT2_RA", {}
    else:
        return "GPT", {}


def load_model(
    checkpoint_path: str, device: str = "cuda"
) -> Tuple[torch.nn.Module, str]:
    """
    Load model from checkpoint, auto-detecting type.

    Returns:
        (model, model_type_name)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type, config_dict = detect_model_type(checkpoint)

    print(f"Detected model type: {model_type}")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class, config_type = MODEL_REGISTRY[model_type]

    # Extract config from checkpoint or use defaults
    if "config" in checkpoint:
        # Try to use saved config
        saved_config = checkpoint["config"]
        if hasattr(saved_config, "__dict__"):
            config_dict = vars(saved_config)
        elif isinstance(saved_config, dict):
            config_dict = saved_config

    # Create config
    config = create_config(config_type, **config_dict)

    # Instantiate model
    if model_type in ["GPT2_MLA_RA_KV", "GPT2_MLA_KV", "GPT2_MLA_RA_KVM"]:
        # These models need compression_ratio
        compression_ratio = config_dict.get("compression_ratio", 0.5)
        model = model_class(config, compression_ratio=compression_ratio)
    elif model_type in ["GPT2_MLA_KV2", "GPT2_MLA_KV2M"]:
        compression_ratio = config_dict.get("compression_ratio", 0.5)
        if model_type == "GPT2_MLA_KV2M":
            mlp_d_latent = config_dict.get("mlp_d_latent", 256)
            model = model_class(config, compression_ratio, mlp_d_latent)
        else:
            model = model_class(config, compression_ratio)
    else:
        model = model_class(config)

    # Load state dict
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return model, model_type


def create_model_random(
    model_type: str, device: str = "cuda", vocab_size: int = 50257
) -> torch.nn.Module:
    """Create model with random initialization."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class, config_type = MODEL_REGISTRY[model_type]

    # Use default config
    config = create_config(config_type)

    # Instantiate model with defaults
    # Note: vocab_size comes before other parameters in MLA model signatures
    if model_type == "GPT2_MLA_RA_KV":
        model = model_class(config, vocab_size, compression_ratio=0.5)
    elif model_type == "GPT2_MLA_KV":
        model = model_class(config, vocab_size, compression_ratio=0.5)
    elif model_type == "GPT2_MLA_RA_KVM":
        model = model_class(
            config, vocab_size, compression_ratio=0.5, mlp_d_latent=256, tie_mlp=True
        )
    elif model_type == "GPT2_MLA_KV2":
        model = model_class(config, vocab_size, compression_ratio=0.5)
    elif model_type == "GPT2_MLA_KV2M":
        model = model_class(config, vocab_size, compression_ratio=0.5, mlp_d_latent=256)
    elif model_type in ["GPT2_MLA", "GPT2_MLA_RA"]:
        model = model_class(config, vocab_size)
    else:
        # GPT and GPT2_RA use GPTConfig which includes vocab_size
        model = model_class(config)

    model.to(device)
    return model


def benchmark_inference(
    model, num_tokens=100, num_runs=5, batch_size=1
) -> Dict[str, float]:
    """Benchmark inference speed using autoregressive generation."""
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    for _ in range(3):
        prompt = torch.randint(0, 50257, (batch_size, 32), device=device)
        with torch.no_grad():
            for _ in range(10):
                logits, _ = model(prompt)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                prompt = torch.cat([prompt, next_token], dim=1)

    # Benchmark
    times = []
    for run in range(num_runs):
        prompt = torch.randint(0, 50257, (batch_size, 32), device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_tokens):
                logits, _ = model(prompt)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                prompt = torch.cat([prompt, next_token], dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        tokens_per_sec = num_tokens / elapsed
        print(f"  Run {run+1}: {elapsed:.3f}s ({tokens_per_sec:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    avg_tokens_per_sec = num_tokens / avg_time

    return {
        "avg_time": avg_time,
        "tokens_per_sec": avg_tokens_per_sec,
        "times": times,
    }


def print_comparison(results: Dict[str, Dict], baseline_name: str = None):
    """Print comparison table of all results."""
    print("\n" + "=" * 80)
    print("INFERENCE SPEED COMPARISON")
    print("=" * 80)
    print(
        f"\n{'Model Type':<25} {'Parameters':<15} {'Throughput':<20} {'vs Baseline':<15}"
    )
    print("-" * 80)

    # Get baseline for comparison
    if baseline_name and baseline_name in results:
        baseline_speed = results[baseline_name]["tokens_per_sec"]
    else:
        baseline_name = list(results.keys())[0]
        baseline_speed = results[baseline_name]["tokens_per_sec"]

    for model_name, result in results.items():
        params = result["params"]
        speed = result["tokens_per_sec"]
        speedup = speed / baseline_speed

        if model_name == baseline_name:
            comparison = "(baseline)"
        elif speedup > 1:
            comparison = f"{speedup:.2f}x faster"
        else:
            comparison = f"{1/speedup:.2f}x slower"

        print(
            f"{model_name:<25} {params:>7.2f}M       {speed:>10.1f} tok/s     {comparison:<15}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference speed across GPT-2 model architectures"
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        help="Model checkpoint files (0-2). If 0: compare all types with random init. "
        "If 1: compare loaded model vs all types. If 2: compare two loaded models.",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=100, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detect if None)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported model types and exit",
    )
    parser.add_argument(
        "--test-models",
        type=str,
        default=None,
        help="Comma-separated list of model types to test (only for mode 0: random init comparison). "
        "Example: --test-models='GPT,GPT2_RA,GPT2_MLA'",
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        print("Supported model architectures:")
        print("-" * 80)
        # Sort models: first by config type (gpt before mla), then alphabetically
        sorted_models = sorted(MODEL_REGISTRY.items(), key=lambda x: (x[1][1], x[0]))
        for i, (model_name, (model_class, config_type)) in enumerate(sorted_models, 1):
            print(f"  {i:2d}. {model_name:<25} (config: {config_type})")
        print("-" * 80)
        print(f"Total: {len(MODEL_REGISTRY)} architectures")
        sys.exit(0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GPT-2 Architecture Inference Speed Comparison")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if len(args.checkpoints) == 0:
        # Mode 1: Compare all model types with random init
        print("\nMode: Comparing all model types with random initialization")
        print("-" * 80)

        # Filter models if --test-models specified
        if args.test_models:
            requested_models = [m.strip() for m in args.test_models.split(",")]
            # Validate requested models
            invalid_models = [m for m in requested_models if m not in MODEL_REGISTRY]
            if invalid_models:
                print(f"Error: Unknown model types: {', '.join(invalid_models)}")
                print(f"Use --list to see all supported models")
                sys.exit(1)
            models_to_test = requested_models
            print(f"Testing models: {', '.join(models_to_test)}")
        else:
            models_to_test = list(MODEL_REGISTRY.keys())

        for model_type in models_to_test:
            print(f"\n{'='*80}")
            print(f"{model_type}")
            print(f"{'='*80}")
            try:
                model = create_model_random(model_type, device)
                params = model.get_num_params() / 1e6
                print(f"Parameters: {params:.2f}M")

                bench_results = benchmark_inference(
                    model, args.num_tokens, args.num_runs, args.batch_size
                )
                results[model_type] = {**bench_results, "params": params}

                # Free memory
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error benchmarking {model_type}: {e}")
                import traceback

                traceback.print_exc()

    elif len(args.checkpoints) == 1:
        # Mode 2: Compare loaded model vs all types with random init
        print(f"\nMode: Comparing loaded model against all types")
        print("-" * 80)

        # Load the checkpoint
        print(f"\nLoading checkpoint: {args.checkpoints[0]}")
        print("-" * 80)
        loaded_model, loaded_type = load_model(args.checkpoints[0], device)
        params = loaded_model.get_num_params() / 1e6
        print(f"Parameters: {params:.2f}M")

        bench_results = benchmark_inference(
            loaded_model, args.num_tokens, args.num_runs, args.batch_size
        )
        results[f"{loaded_type} (loaded)"] = {**bench_results, "params": params}

        del loaded_model
        if device == "cuda":
            torch.cuda.empty_cache()

        # Compare against other types with random init
        print("\n" + "=" * 80)
        print("Comparing against other architectures (random init)")
        print("=" * 80)

        for model_type in MODEL_REGISTRY.keys():
            if model_type == loaded_type:
                continue  # Skip the type we already loaded

            print(f"\n{'-'*80}")
            print(f"{model_type}")
            print(f"{'-'*80}")
            try:
                model = create_model_random(model_type, device)
                params = model.get_num_params() / 1e6
                print(f"Parameters: {params:.2f}M")

                bench_results = benchmark_inference(
                    model, args.num_tokens, args.num_runs, args.batch_size
                )
                results[model_type] = {**bench_results, "params": params}

                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error benchmarking {model_type}: {e}")

    elif len(args.checkpoints) == 2:
        # Mode 3: Compare two loaded models
        print(f"\nMode: Comparing two loaded models")
        print("-" * 80)

        for i, ckpt_path in enumerate(args.checkpoints, 1):
            print(f"\nLoading checkpoint {i}: {ckpt_path}")
            print("-" * 80)
            model, model_type = load_model(ckpt_path, device)
            params = model.get_num_params() / 1e6
            print(f"Parameters: {params:.2f}M")

            bench_results = benchmark_inference(
                model, args.num_tokens, args.num_runs, args.batch_size
            )
            results[f"{model_type} (ckpt{i})"] = {**bench_results, "params": params}

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    else:
        print(
            f"Error: Too many checkpoints provided ({len(args.checkpoints)}). Max is 2."
        )
        sys.exit(1)

    # Print comparison table
    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
