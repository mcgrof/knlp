#!/usr/bin/env python3
"""
Test SVD vs Random projection methods for KVSplice.

This tests the projection computation without full model integration,
to validate the projection quality before patching into inference.
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Tuple


def create_random_orthogonal_projection(
    d_in: int, d_compressed: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random orthogonal projection matrices.

    Returns:
        compress_weight: [d_compressed, d_in]
        expand_weight: [d_in, d_compressed]
    """
    print(f"\nCreating random orthogonal projection: {d_in} → {d_compressed}")

    # Random orthogonal matrix via QR decomposition
    Q, _ = torch.linalg.qr(torch.randn(d_in, d_in))

    # Use top d_compressed rows
    compress_weight = Q[:d_compressed, :].to(device)
    expand_weight = compress_weight.T.to(device)

    # Verify orthogonality
    identity_approx = compress_weight @ expand_weight
    identity_error = (
        (identity_approx - torch.eye(d_compressed, device=device)).abs().max()
    )

    print(f"  Orthogonality error: {identity_error:.6f}")

    return compress_weight, expand_weight


def create_svd_projection(
    activations: torch.Tensor, d_compressed: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Create SVD-based projection from activations.

    Args:
        activations: [N, d_in] - collected activations
        d_compressed: Target compressed dimension

    Returns:
        compress_weight: [d_compressed, d_in]
        expand_weight: [d_in, d_compressed]
        stats: Dictionary with SVD statistics
    """
    print(f"\nCreating SVD projection: {activations.shape[1]} → {d_compressed}")
    print(f"  Samples: {activations.shape[0]}")

    # Center the data
    mean = activations.mean(dim=0, keepdim=True)
    activations_centered = activations - mean

    print(f"  Computing SVD...")
    # Compute SVD
    U, S, Vh = torch.svd(activations_centered.cpu())

    # Top k components
    compress_weight = Vh[:d_compressed, :].to(device)
    expand_weight = compress_weight.T.to(device)

    # Compute explained variance
    total_var = S.pow(2).sum()
    kept_var = S[:d_compressed].pow(2).sum()
    explained = (kept_var / total_var * 100).item()

    # Compute reconstruction error on sample
    sample = activations[:100].to(device)
    compressed = sample @ expand_weight
    reconstructed = compressed @ compress_weight.T
    recon_error = (sample - reconstructed).pow(2).mean().sqrt().item()

    stats = {
        "explained_variance_pct": explained,
        "reconstruction_error": recon_error,
        "singular_values": S[:d_compressed].cpu().numpy(),
    }

    print(f"  Explained variance: {explained:.2f}%")
    print(f"  Reconstruction error (RMSE): {recon_error:.6f}")
    print(f"  Top 5 singular values: {S[:5].numpy()}")

    return compress_weight, expand_weight, stats


def collect_activations(
    model,
    tokenizer,
    layer_idx: int = 0,
    n_samples: int = 2000,
    max_length: int = 512,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Collect hidden state activations from a specific layer.

    Returns:
        activations: [N, hidden_size]
    """
    print(f"\nCollecting activations from layer {layer_idx}...")

    # Load calibration data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    activations = []
    samples_collected = 0

    # Hook to capture activations
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Flatten batch and sequence: [B, T, D] → [B*T, D]
        activations.append(hidden.detach().cpu().reshape(-1, hidden.shape[-1]))

    # Find target layer
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Unsupported architecture")

    if layer_idx >= len(layers):
        raise ValueError(
            f"Layer {layer_idx} not found (model has {len(layers)} layers)"
        )

    layer = layers[layer_idx]

    # Hook attention output
    if hasattr(layer, "self_attn"):
        hook = layer.self_attn.register_forward_hook(hook_fn)
    elif hasattr(layer, "attn"):
        hook = layer.attn.register_forward_hook(hook_fn)
    else:
        raise ValueError("Cannot find attention module")

    # Collect activations
    model.eval()
    with torch.no_grad():
        for text in texts:
            if samples_collected >= n_samples:
                break

            inputs = tokenizer(
                text, max_length=max_length, truncation=True, return_tensors="pt"
            ).to(device)

            if inputs["input_ids"].shape[1] < 10:
                continue

            model(**inputs)

            samples_collected += inputs["input_ids"].shape[1]

            if samples_collected % 500 == 0:
                print(f"  Progress: {samples_collected}/{n_samples} tokens")

    hook.remove()

    # Concatenate and subsample
    all_activations = torch.cat(activations, dim=0)
    if all_activations.shape[0] > n_samples:
        all_activations = all_activations[:n_samples]

    print(f"  Collected: {all_activations.shape}")

    return all_activations


def compare_projection_methods(
    activations: torch.Tensor,
    compression_ratio: float = 0.5,
    device: str = "cuda",
):
    """Compare SVD vs Random projection on same activations."""

    d_in = activations.shape[1]
    d_compressed = max(1, int(d_in * compression_ratio))

    print(f"\n{'=' * 80}")
    print(f"COMPARING PROJECTION METHODS")
    print(f"{'=' * 80}")
    print(f"Input dimension: {d_in}")
    print(f"Compressed dimension: {d_compressed}")
    print(f"Compression ratio: {compression_ratio} ({d_compressed}/{d_in})")

    # Test data (held out from training)
    # Convert to float32 for projection operations
    test_size = min(500, activations.shape[0] // 4)
    test_data = activations[-test_size:].to(device).float()
    train_data = activations[:-test_size].float()

    # Method 1: Random Orthogonal
    print(f"\n{'=' * 80}")
    print("METHOD 1: Random Orthogonal Projection")
    print(f"{'=' * 80}")

    compress_rand, expand_rand = create_random_orthogonal_projection(
        d_in, d_compressed, device
    )

    # Test reconstruction
    # Forward: data @ expand -> compressed latent
    # Backward: compressed @ compress -> reconstructed data
    compressed_rand = test_data @ expand_rand
    reconstructed_rand = compressed_rand @ compress_rand
    error_rand = (test_data - reconstructed_rand).pow(2).mean().sqrt().item()

    # Measure dimension preservation
    original_norm = test_data.norm(dim=1).mean().item()
    compressed_norm = compressed_rand.norm(dim=1).mean().item()
    norm_ratio_rand = compressed_norm / original_norm

    print(f"\nTest Metrics:")
    print(f"  Reconstruction error (RMSE): {error_rand:.6f}")
    print(f"  Original norm: {original_norm:.4f}")
    print(f"  Compressed norm: {compressed_norm:.4f}")
    print(f"  Norm ratio: {norm_ratio_rand:.4f}")

    # Method 2: SVD
    print(f"\n{'=' * 80}")
    print("METHOD 2: SVD-based Projection")
    print(f"{'=' * 80}")

    compress_svd, expand_svd, svd_stats = create_svd_projection(
        train_data, d_compressed, device
    )

    # Test reconstruction
    compressed_svd = test_data @ expand_svd
    reconstructed_svd = compressed_svd @ compress_svd
    error_svd = (test_data - reconstructed_svd).pow(2).mean().sqrt().item()

    # Measure dimension preservation
    compressed_norm_svd = compressed_svd.norm(dim=1).mean().item()
    norm_ratio_svd = compressed_norm_svd / original_norm

    print(f"\nTest Metrics:")
    print(f"  Reconstruction error (RMSE): {error_svd:.6f}")
    print(f"  Original norm: {original_norm:.4f}")
    print(f"  Compressed norm: {compressed_norm_svd:.4f}")
    print(f"  Norm ratio: {norm_ratio_svd:.4f}")

    # Comparison
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}")

    improvement = (error_rand - error_svd) / error_rand * 100

    print(f"\nReconstruction Error:")
    print(f"  Random: {error_rand:.6f}")
    print(f"  SVD:    {error_svd:.6f}")
    print(f"  SVD improvement: {improvement:.2f}%")

    print(f"\nNorm Preservation:")
    print(f"  Random: {norm_ratio_rand:.4f}")
    print(f"  SVD:    {norm_ratio_svd:.4f}")

    # Save projection matrices
    results = {
        "random": {
            "compress": compress_rand.cpu(),
            "expand": expand_rand.cpu(),
            "error": error_rand,
            "norm_ratio": norm_ratio_rand,
        },
        "svd": {
            "compress": compress_svd.cpu(),
            "expand": expand_svd.cpu(),
            "error": error_svd,
            "norm_ratio": norm_ratio_svd,
            "explained_variance": svd_stats["explained_variance_pct"],
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Test projection methods for KVSplice")
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="Model to use for activation collection",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index to collect activations from",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of activation samples to collect",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="Compression ratio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save projection matrices",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("KVSplice Projection Method Comparison")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Samples: {args.samples}")
    print(f"Compression: {args.compression_ratio}")

    # Load model and tokenizer
    print(f"\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except (TypeError, OSError) as e:
        print(f"  Failed with fast tokenizer, trying slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)

    # Collect activations
    activations = collect_activations(
        model,
        tokenizer,
        layer_idx=args.layer,
        n_samples=args.samples,
        device=args.device,
    )

    # Free model memory
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Compare methods
    results = compare_projection_methods(
        activations, compression_ratio=args.compression_ratio, device=args.device
    )

    # Save if requested
    if args.save:
        torch.save(results, args.save)
        print(f"\nSaved projection matrices to: {args.save}")


if __name__ == "__main__":
    main()
