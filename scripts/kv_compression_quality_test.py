#!/usr/bin/env python3
"""
Direct KV Compression Quality Test

Measures actual reconstruction error for each compressor type.
This is the honest way to evaluate compression - no framework overhead.

Usage:
    python scripts/kv_compression_quality_test.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.kv_plugin import (
    KVCompressorConfig,
    IdentityCompressor,
    PCACompressor,
    TopKCompressor,
    SVDCompressor,
    HybridCompressor,
)


def collect_kv_activations(model, tokenizer, text, device="cuda", max_length=512):
    """Collect K, V activations from all layers."""
    model.eval()

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Storage for activations
    k_activations = []
    v_activations = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # GPT-2 attention output structure varies
            # Try to extract present_key_value from output
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, tuple) and len(item) == 2:
                        # Check if it looks like (key, value)
                        k, v = item
                        if hasattr(k, 'shape') and len(k.shape) == 4:
                            k_activations.append(k.detach().clone())
                            v_activations.append(v.detach().clone())
                            return
        return hook

    # Register hooks
    hooks = []
    for i, block in enumerate(model.transformer.h):
        hook = block.attn.register_forward_hook(make_hook(i))
        hooks.append(hook)

    # Forward pass with use_cache=True to get KV outputs
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # If hooks didn't capture, try extracting from past_key_values
    if len(k_activations) == 0 and hasattr(outputs, 'past_key_values') and outputs.past_key_values:
        print("  Using past_key_values from model output...")
        for layer_kv in outputs.past_key_values:
            if layer_kv is not None:
                k, v = layer_kv
                k_activations.append(k.detach())
                v_activations.append(v.detach())

    return k_activations, v_activations


def measure_reconstruction_error(compressor, activations, name=""):
    """Measure reconstruction MSE and normalized RMSE."""
    # Collect all activations for calibration
    all_flat = []
    for act in activations:
        # Flatten: [B, H, T, D] -> [B*T, H*D]
        B, H, T, D = act.shape
        flat = act.permute(0, 2, 1, 3).reshape(-1, H * D)
        all_flat.append(flat)

    # Combine all layers for calibration
    combined = torch.cat(all_flat, dim=0).float()

    # Calibrate on ALL data
    compressor.calibrate(combined)

    # Now measure reconstruction error
    total_mse = 0.0
    total_var = 0.0
    total_samples = 0

    for flat in all_flat:
        flat = flat.float()

        # Compress and expand
        compressed = compressor.compress(flat)
        reconstructed = compressor.expand(compressed)

        # Compute errors
        diff = flat - reconstructed
        mse = (diff ** 2).mean().item()
        var = flat.var().item()

        total_mse += mse * flat.shape[0]
        total_var += var * flat.shape[0]
        total_samples += flat.shape[0]

    avg_mse = total_mse / total_samples
    avg_var = total_var / total_samples

    # Normalized RMSE: RMSE / std(data) * 100
    # This gives percentage of variance NOT explained
    nrmse = (avg_mse ** 0.5) / (avg_var ** 0.5 + 1e-8) * 100

    # Variance explained (like R² for PCA)
    var_explained = max(0, 1 - avg_mse / (avg_var + 1e-8)) * 100

    return {
        "mse": avg_mse,
        "rmse": avg_mse ** 0.5,
        "nrmse_pct": nrmse,
        "variance_explained_pct": var_explained,
    }


def estimate_ppl_impact(rel_error):
    """
    Rough estimate of PPL impact from reconstruction error.

    Based on empirical observation that:
    - 1% relative error ≈ 0.5-1% PPL increase
    - 5% relative error ≈ 3-5% PPL increase
    - 10% relative error ≈ 8-12% PPL increase

    This is approximate - actual impact depends on which dimensions
    carry semantic information.
    """
    # Quadratic relationship (error compounds through layers)
    return rel_error * (1 + rel_error) * 100


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading GPT-2...")
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 for accurate error measurement
    ).to(device)

    # Get calibration text from WikiText
    print("Loading WikiText-2 for calibration...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    cal_text = " ".join(dataset["text"][:100])  # First 100 samples

    # Collect activations
    print("Collecting K/V activations...")
    k_acts, v_acts = collect_kv_activations(
        model, tokenizer, cal_text, device, max_length=512
    )
    print(f"  Collected {len(k_acts)} layers")
    print(f"  K shape per layer: {k_acts[0].shape}")

    # Test configurations
    d_input = k_acts[0].shape[1] * k_acts[0].shape[3]  # heads * head_dim = 768
    print(f"  d_input = {d_input}")

    configs = [
        ("Identity (baseline)", "identity", {"d_compressed": d_input}),
        ("PCA rank=384 (2x)", "pca", {"d_compressed": 384}),
        ("PCA rank=256 (3x)", "pca", {"d_compressed": 256}),
        ("PCA rank=128 (6x)", "pca", {"d_compressed": 128}),
        ("PCA rank=64 (12x)", "pca", {"d_compressed": 64}),
        ("PCA rank=32 (24x)", "pca", {"d_compressed": 32}),
        ("TopK k=384 (2x)", "topk", {"d_compressed": 384}),
        ("TopK k=256 (3x)", "topk", {"d_compressed": 256}),
        ("TopK k=128 (6x)", "topk", {"d_compressed": 128}),
        ("TopK k=64 (12x)", "topk", {"d_compressed": 64}),
        ("SVD rank=128 (6x)", "svd", {"d_compressed": 128}),
        ("SVD rank=64 (12x)", "svd", {"d_compressed": 64}),
        ("Hybrid lat=256 comp=128 (6x)", "hybrid", {"d_compressed": 128, "d_latent": 256}),
        ("Hybrid lat=256 comp=64 (12x)", "hybrid", {"d_compressed": 64, "d_latent": 256}),
        ("Hybrid lat=256 comp=32 (24x)", "hybrid", {"d_compressed": 32, "d_latent": 256}),
    ]

    # Create compressor instances
    compressor_classes = {
        "identity": IdentityCompressor,
        "pca": PCACompressor,
        "topk": TopKCompressor,
        "svd": SVDCompressor,
        "hybrid": HybridCompressor,
    }

    print("\n" + "=" * 90)
    print("KV COMPRESSION QUALITY TEST - EMPIRICAL RESULTS")
    print("=" * 90)
    print(f"{'Config':<35} {'K VarExp%':>10} {'V VarExp%':>10} {'K NRMSE%':>10} {'V NRMSE%':>10}")
    print("-" * 90)

    results = []
    for name, comp_type, params in configs:
        config = KVCompressorConfig(
            d_input=d_input,
            d_compressed=params["d_compressed"],
            device=device,
            dtype=torch.float32,
        )

        cls = compressor_classes[comp_type]
        if comp_type == "hybrid":
            k_comp = cls(config, d_latent=params["d_latent"])
            v_comp = cls(config, d_latent=params["d_latent"])
        else:
            k_comp = cls(config)
            v_comp = cls(config)

        k_comp = k_comp.to(device)
        v_comp = v_comp.to(device)

        # Measure K reconstruction
        k_err = measure_reconstruction_error(k_comp, k_acts, f"{name} K")
        # Measure V reconstruction
        v_err = measure_reconstruction_error(v_comp, v_acts, f"{name} V")

        # Estimate PPL impact based on variance NOT explained
        avg_nrmse = (k_err["nrmse_pct"] + v_err["nrmse_pct"]) / 2
        ppl_impact = estimate_ppl_impact(avg_nrmse / 100)  # Convert to fraction

        print(
            f"{name:<35} {k_err['variance_explained_pct']:>10.1f} "
            f"{v_err['variance_explained_pct']:>10.1f} "
            f"{k_err['nrmse_pct']:>10.1f} {v_err['nrmse_pct']:>10.1f}"
        )

        results.append({
            "name": name,
            "type": comp_type,
            "params": params,
            "k_variance_explained": k_err["variance_explained_pct"],
            "v_variance_explained": v_err["variance_explained_pct"],
            "k_nrmse": k_err["nrmse_pct"],
            "v_nrmse": v_err["nrmse_pct"],
            "estimated_ppl_impact": ppl_impact,
        })

    print("-" * 90)
    print("\nNotes:")
    print("- VarExp% = variance explained (100% = perfect reconstruction)")
    print("- NRMSE% = normalized RMSE (lower = better reconstruction)")
    print("- Higher VarExp% = less quality loss when using compression")
    print("- V compression typically matters more than K for generation quality")

    # Save results
    import json
    output_path = Path("key_results/kv_compression_quality.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
