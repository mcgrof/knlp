#!/usr/bin/env python3
"""
KV Low-Rank Calibration Script.

Collects real K/V activations from a model and computes PCA-based
orthogonal projections for each layer. Saves projections to disk
for use during inference.

Usage:
    python scripts/calibrate_kv_lowrank.py --model Qwen/Qwen2.5-0.5B
    python scripts/calibrate_kv_lowrank.py --model Qwen/Qwen2.5-0.5B --rank 32
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_kv_from_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: str = "cuda",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract K, V tensors from each layer."""
    kv_pairs = []

    # Forward pass with cache
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    # Get K, V from DynamicCache
    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
        pkv = outputs.past_key_values
        if hasattr(pkv, "key_cache"):
            # DynamicCache object
            for i in range(len(pkv.key_cache)):
                kv_pairs.append((pkv.key_cache[i].clone(), pkv.value_cache[i].clone()))
        else:
            # Tuple of (k, v) pairs
            for k, v in pkv:
                kv_pairs.append((k.clone(), v.clone()))

    return kv_pairs


def compute_pca_basis(
    data: torch.Tensor,
    rank: int,
    center: bool = True,
    compute_gamma_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute PCA basis for data with optional γ-aware stats.

    Args:
        data: [N, D] tensor of samples
        rank: Number of principal components to keep
        center: Whether to center data before PCA
        compute_gamma_stats: Whether to compute per-latent-dim stats for γ-aware quant

    Returns:
        Dict with:
            U: [D, rank] orthonormal projection matrix
            mean: [D] mean vector (zeros if not centered)
            latent_std: [rank] per-dim std in latent space (if compute_gamma_stats)
            latent_max_abs: [rank] per-dim max_abs in latent space (if compute_gamma_stats)
            energy_fraction: [rank] normalized importance scores (if compute_gamma_stats)
    """
    N, D = data.shape
    rank = min(rank, D, N)

    # Convert to float32 for numerical stability
    data_f32 = data.float()

    # Center data
    if center:
        mean = data_f32.mean(dim=0)
        centered = data_f32 - mean
    else:
        mean = torch.zeros(D, device=data.device, dtype=torch.float32)
        centered = data_f32

    # Compute SVD (low-rank approximation)
    # Use torch.pca_lowrank for efficiency
    try:
        U, S, V = torch.pca_lowrank(centered, q=rank, center=False, niter=2)
        # V is [D, rank] - the principal components
        basis = V
    except Exception:
        # Fallback to full SVD
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        basis = Vt[:rank, :].T  # [D, rank]

    result = {"U": basis, "mean": mean}

    # Compute γ-aware stats in latent space
    if compute_gamma_stats:
        # Project centered data to latent space
        latent = centered @ basis  # [N, rank]

        # Per-latent-dim stats
        latent_std = latent.std(dim=0)  # [rank]
        latent_max_abs = latent.abs().max(dim=0).values  # [rank]

        # Compute energy fraction for channel pruning (v19)
        # Energy is proportional to variance (std^2), but we use std for stability
        energy = latent_std.clamp(min=1e-8)
        energy_fraction = energy / energy.sum()

        result["latent_std"] = latent_std
        result["latent_max_abs"] = latent_max_abs
        result["energy_fraction"] = energy_fraction

    return result


def calibrate_model(
    model_name: str,
    rank: int,
    cal_texts: List[str],
    device: str = "cuda",
    max_length: int = 512,
) -> Dict:
    """
    Run calibration and compute PCA projections for each layer.

    Returns dict with structure:
    {
        "model": model_name,
        "rank": rank,
        "head_dim": int,
        "n_layers": int,
        "n_heads": int,
        "layers": [
            {
                "K": {"U": tensor, "mean": tensor},
                "V": {"U": tensor, "mean": tensor},
            },
            ...
        ]
    }
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Collect KV activations from calibration texts
    print(f"\nCollecting KV activations from {len(cal_texts)} texts...")
    all_kv_per_layer = None

    for i, text in enumerate(cal_texts):
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(device)

        kv_pairs = extract_kv_from_model(model, input_ids, device)

        if all_kv_per_layer is None:
            # Initialize storage
            n_layers = len(kv_pairs)
            all_kv_per_layer = [{"K": [], "V": []} for _ in range(n_layers)]

        for layer_idx, (k, v) in enumerate(kv_pairs):
            # k, v shape: [B, n_heads, T, head_dim]
            # Flatten to [B*n_heads*T, head_dim]
            B, n_heads, T, head_dim = k.shape
            k_flat = k.view(-1, head_dim)
            v_flat = v.view(-1, head_dim)
            all_kv_per_layer[layer_idx]["K"].append(k_flat)
            all_kv_per_layer[layer_idx]["V"].append(v_flat)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(cal_texts)} texts")

    # Get dimensions
    k_sample = all_kv_per_layer[0]["K"][0]
    head_dim = k_sample.shape[-1]
    n_layers = len(all_kv_per_layer)
    n_heads = kv_pairs[0][0].shape[1]
    effective_rank = min(rank, head_dim)

    print(f"\nModel info:")
    print(f"  Layers: {n_layers}")
    print(f"  Heads: {n_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Target rank: {effective_rank}")

    # Compute PCA for each layer
    print(f"\nComputing PCA projections...")
    layers_data = []

    for layer_idx in range(n_layers):
        # Concatenate all K samples for this layer
        K_all = torch.cat(all_kv_per_layer[layer_idx]["K"], dim=0)
        V_all = torch.cat(all_kv_per_layer[layer_idx]["V"], dim=0)

        print(
            f"  Layer {layer_idx}: K={K_all.shape}, V={V_all.shape}",
            end=" ",
            flush=True,
        )

        # Compute PCA basis with γ-aware stats
        K_stats = compute_pca_basis(K_all, effective_rank, compute_gamma_stats=True)
        V_stats = compute_pca_basis(V_all, effective_rank, compute_gamma_stats=True)

        K_U = K_stats["U"]
        V_U = V_stats["U"]

        # Verify orthonormality
        K_check = (
            (K_U.T @ K_U - torch.eye(effective_rank, device=K_U.device)).abs().max()
        )
        V_check = (
            (V_U.T @ V_U - torch.eye(effective_rank, device=V_U.device)).abs().max()
        )

        # Print γ stats summary
        K_std_range = (
            f"[{K_stats['latent_std'].min():.2f}, {K_stats['latent_std'].max():.2f}]"
        )
        V_std_range = (
            f"[{V_stats['latent_std'].min():.2f}, {V_stats['latent_std'].max():.2f}]"
        )
        print(
            f"(K_orth={K_check:.2e}, V_orth={V_check:.2e}, K_std={K_std_range}, V_std={V_std_range})"
        )

        # Store all stats including γ-aware and energy fraction (v19)
        layers_data.append(
            {
                "K": {
                    "U": K_stats["U"].half().cpu(),
                    "mean": K_stats["mean"].half().cpu(),
                    "latent_std": K_stats["latent_std"].half().cpu(),
                    "latent_max_abs": K_stats["latent_max_abs"].half().cpu(),
                    "energy_fraction": K_stats["energy_fraction"].half().cpu(),
                },
                "V": {
                    "U": V_stats["U"].half().cpu(),
                    "mean": V_stats["mean"].half().cpu(),
                    "latent_std": V_stats["latent_std"].half().cpu(),
                    "latent_max_abs": V_stats["latent_max_abs"].half().cpu(),
                    "energy_fraction": V_stats["energy_fraction"].half().cpu(),
                },
            }
        )

        # Free memory
        del K_all, V_all
        torch.cuda.empty_cache()

    return {
        "version": 3,  # v3 includes energy_fraction for channel pruning
        "model": model_name,
        "rank": effective_rank,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "gamma_aware": True,
        "channel_pruning": True,
        "layers": layers_data,
    }


def get_calibration_texts() -> List[str]:
    """Return calibration texts."""
    texts = [
        """Machine learning has transformed how we approach complex problems.
        Neural networks can learn patterns from data without explicit programming.
        Deep learning models have achieved remarkable success in computer vision,
        natural language processing, and many other domains.""",
        """The transformer architecture revolutionized natural language processing.
        Self-attention mechanisms allow models to capture long-range dependencies.
        Key-value caching enables efficient autoregressive generation by storing
        previously computed keys and values for reuse in subsequent tokens.""",
        """Large language models have become increasingly powerful over the past decade.
        They can generate coherent text, answer questions, translate languages,
        and assist with various tasks. The computational requirements continue to grow,
        driving research into efficient inference techniques.""",
        """Compression techniques for neural network inference are essential for
        practical deployment. Methods include quantization, pruning, and low-rank
        approximation. The goal is to reduce memory usage and computational cost
        while maintaining model quality.""",
        """Attention mechanisms compute weighted sums of value vectors based on
        query-key similarity. In multi-head attention, multiple attention heads
        process the input in parallel, each capturing different aspects of
        relationships in the data.""",
        """The quick brown fox jumps over the lazy dog. This pangram contains every
        letter of the English alphabet at least once. It is commonly used for
        testing fonts and keyboard layouts, as well as for practicing typing.""",
        """In mathematics, a function is a relation between sets that associates
        every element of a first set to exactly one element of a second set.
        Functions are fundamental to many areas of mathematics and its applications.""",
        """The history of computing spans from ancient calculating devices to modern
        supercomputers. Key milestones include the development of the transistor,
        integrated circuits, and the internet. Today's computers process billions
        of operations per second.""",
    ]
    # Repeat to get more samples
    return texts * 5


def main():
    parser = argparse.ArgumentParser(description="Calibrate KV low-rank projections")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to calibrate",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Target rank for low-rank projection",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: kv_lowrank_calib_{model}.pt)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for calibration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    args = parser.parse_args()

    # Get calibration texts
    cal_texts = get_calibration_texts()
    print(f"Using {len(cal_texts)} calibration texts")

    # Run calibration
    calib_data = calibrate_model(
        args.model,
        args.rank,
        cal_texts,
        args.device,
        args.max_length,
    )

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = args.model.replace("/", "-").lower()
        output_path = Path(f"kv_lowrank_calib_{model_short}_r{args.rank}.pt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calib_data, output_path)
    print(f"\nCalibration saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Version: {calib_data.get('version', 1)}")
    print(f"Model: {calib_data['model']}")
    print(f"Layers: {calib_data['n_layers']}")
    print(f"Heads: {calib_data['n_heads']}")
    print(f"Head dim: {calib_data['head_dim']}")
    print(f"Rank: {calib_data['rank']}")
    print(f"Compression ratio: {calib_data['head_dim'] / calib_data['rank']:.2f}x")
    print(f"γ-aware stats: {calib_data.get('gamma_aware', False)}")


if __name__ == "__main__":
    main()
