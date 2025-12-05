#!/usr/bin/env python3
"""
Test γ-aware quantization vs standard quantization.

Compares:
1. Standard QuantizedCalibratedCompressor
2. GammaAwareQuantizedCompressor (per-dim scale normalization)

The γ-aware version should give better reconstruction quality at the same bitwidth
because it normalizes per-dim variance before quantizing.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    QuantizedCalibratedCompressor,
    GammaAwareQuantizedCompressor,
    load_calibrated_compressors,
)


def test_reconstruction_quality(
    model_name: str,
    calib_path: str,
    device: str = "cuda",
    num_samples: int = 1000,
):
    """Compare reconstruction quality of standard vs γ-aware quantization."""
    print(f"\n=== Reconstruction Quality Test ===")
    print(f"Model: {model_name}")
    print(f"Calibration: {calib_path}")

    # Load calibration data
    calib_data = torch.load(calib_path, map_location=device)

    # Get a sample layer's data
    layer_data = calib_data["layers"][0]
    V_U = layer_data["V"]["U"].to(device).to(torch.float16)
    V_mean = layer_data["V"]["mean"].to(device).to(torch.float16)

    d_input = V_U.shape[0]
    rank = V_U.shape[1]
    print(f"Input dim: {d_input}, Rank: {rank}")

    # Generate test data (simulate LN'd vectors with mean ~0)
    test_data = torch.randn(num_samples, d_input, device=device, dtype=torch.float16)
    test_data = test_data - test_data.mean(dim=-1, keepdim=True)

    # Standard quantized compressor
    std_comp = QuantizedCalibratedCompressor(V_U, V_mean, bits=8)

    # γ-aware compressor (compute scale from test data)
    centered = test_data - V_mean
    latent = centered @ V_U
    scale = latent.std(dim=0).clamp(min=1e-6)
    gamma_comp = GammaAwareQuantizedCompressor(V_U, V_mean, scale, bits=8)

    # Compress and reconstruct
    std_compressed = std_comp.compress(test_data)
    std_recon = std_comp.expand(std_compressed)

    gamma_compressed = gamma_comp.compress(test_data)
    gamma_recon = gamma_comp.expand(gamma_compressed)

    # Measure errors
    std_mse = ((test_data - std_recon) ** 2).mean().item()
    gamma_mse = ((test_data - gamma_recon) ** 2).mean().item()

    std_cos = torch.nn.functional.cosine_similarity(
        test_data.flatten(), std_recon.flatten(), dim=0
    ).item()
    gamma_cos = torch.nn.functional.cosine_similarity(
        test_data.flatten(), gamma_recon.flatten(), dim=0
    ).item()

    print(f"\nStandard Quantization:")
    print(f"  MSE: {std_mse:.6f}")
    print(f"  Cosine sim: {std_cos:.6f}")

    print(f"\nγ-Aware Quantization:")
    print(f"  MSE: {gamma_mse:.6f}")
    print(f"  Cosine sim: {gamma_cos:.6f}")

    improvement = (std_mse - gamma_mse) / std_mse * 100
    print(f"\nImprovement: {improvement:+.2f}% MSE reduction")

    return std_mse, gamma_mse


def eval_ppl_comparison(
    model_name: str,
    calib_path: str,
    device: str = "cuda",
    max_samples: int = 50,
):
    """Compare PPL with standard vs γ-aware quantization."""
    print(f"\n=== PPL Comparison Test ===")
    print(f"Model: {model_name}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers

    # Load calibration data
    calib_data = torch.load(calib_path, map_location=device)

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:max_samples]

    def eval_with_compressor(compressor_type: str):
        """Evaluate PPL with given compressor type."""
        k_compressors = []
        v_compressors = []

        for layer_data in calib_data["layers"]:
            V_U = layer_data["V"]["U"].to(device).to(torch.float16)
            V_mean = layer_data["V"]["mean"].to(device).to(torch.float16)

            # K: identity
            k_compressors.append(IdentityCompressor())

            # V: standard or γ-aware
            if compressor_type == "standard":
                v_comp = QuantizedCalibratedCompressor(V_U, V_mean, bits=8)
            else:
                # For γ-aware, compute scale from stored calibration stats
                # Check if scale is stored, otherwise compute from U's singular values
                if "scale" in layer_data.get("V", {}):
                    scale = layer_data["V"]["scale"].to(device).to(torch.float16)
                else:
                    # Estimate scale from projection matrix singular values
                    # This approximates the variance along each latent dimension
                    _, s, _ = torch.linalg.svd(V_U, full_matrices=False)
                    scale = s[: V_U.shape[1]].to(device).to(torch.float16)
                v_comp = GammaAwareQuantizedCompressor(V_U, V_mean, scale, bits=8)

            v_compressors.append(v_comp)

        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = inputs.input_ids.to(device)

            if input_ids.shape[1] < 2:
                continue

            cache.reset()

            with torch.no_grad():
                outputs = model(input_ids, past_key_values=cache, use_cache=True)

                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += targets.numel()

        ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        return ppl

    # Baseline (identity)
    print("\nEvaluating baseline (no compression)...")
    k_comp_id = [IdentityCompressor() for _ in range(num_layers)]
    v_comp_id = [IdentityCompressor() for _ in range(num_layers)]
    cache_id = CompressedDynamicCache(k_comp_id, v_comp_id, num_layers)

    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        if input_ids.shape[1] < 2:
            continue
        cache_id.reset()
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=cache_id, use_cache=True)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    ppl_baseline = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    # Standard quantization
    print("Evaluating standard quantization...")
    ppl_std = eval_with_compressor("standard")

    # γ-aware quantization
    print("Evaluating γ-aware quantization...")
    ppl_gamma = eval_with_compressor("gamma")

    print(f"\n=== Results ===")
    print(f"Baseline PPL: {ppl_baseline:.4f}")
    print(f"Standard int8 PPL: {ppl_std:.4f} ({(ppl_std/ppl_baseline - 1)*100:+.2f}%)")
    print(
        f"γ-aware int8 PPL: {ppl_gamma:.4f} ({(ppl_gamma/ppl_baseline - 1)*100:+.2f}%)"
    )

    return ppl_baseline, ppl_std, ppl_gamma


def main():
    parser = argparse.ArgumentParser(description="Test γ-aware quantization")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B", help="Model to test"
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="kv_lowrank_calib_qwen-qwen2.5-7b_r96.pt",
        help="Calibration file",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--max-samples", type=int, default=30, help="Max samples for PPL eval"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("γ-AWARE QUANTIZATION TEST")
    print("=" * 70)

    # Test 1: Reconstruction quality
    test_reconstruction_quality(args.model, args.calib, args.device)

    # Test 2: PPL comparison
    eval_ppl_comparison(args.model, args.calib, args.device, args.max_samples)


if __name__ == "__main__":
    main()
