#!/usr/bin/env python3
"""
Direct KV Compression Quality Test

Measures compress→expand round-trip quality by:
1. Running forward to get real K/V tensors
2. Compressing at various ranks
3. Expanding back
4. Measuring reconstruction error and attention output difference

This bypasses the plugin's inference path to directly measure compression quality.

Usage:
    python scripts/kv_compression_direct_test.py --model Qwen/Qwen2.5-0.5B
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.kv_plugin import (
    OrthogonalCompressor,
    KVCompressorConfig,
    quantize_to_int8,
    quantize_to_int4,
)


def extract_kv_from_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: str = "cuda",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract K, V tensors from each layer."""
    kv_pairs = []

    # Register hooks to capture K, V
    handles = []
    captured_kv = {}

    def make_hook(layer_idx):
        def hook(module, args, output):
            # output is typically (hidden_states, present_key_value, ...)
            if isinstance(output, tuple) and len(output) > 1:
                present = output[1]
                if present is not None:
                    if hasattr(present, "key_cache"):
                        # DynamicCache
                        k = present.key_cache[layer_idx]
                        v = present.value_cache[layer_idx]
                    elif isinstance(present, tuple):
                        k, v = present
                    else:
                        return
                    captured_kv[layer_idx] = (k.clone(), v.clone())

        return hook

    # Find attention layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        for i, layer in enumerate(layers):
            h = layer.self_attn.register_forward_hook(make_hook(i))
            handles.append(h)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        for i, layer in enumerate(layers):
            h = layer.attn.register_forward_hook(make_hook(i))
            handles.append(h)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, output_attentions=True)

    # Remove hooks
    for h in handles:
        h.remove()

    # Get K, V from DynamicCache if available
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

    return kv_pairs, outputs


def measure_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """Measure reconstruction quality."""
    # MSE
    mse = F.mse_loss(original.float(), reconstructed.float()).item()

    # Relative error
    rel_error = (
        torch.norm(original.float() - reconstructed.float())
        / torch.norm(original.float())
    ).item()

    # Cosine similarity (flatten and compare)
    cos_sim = F.cosine_similarity(
        original.float().flatten().unsqueeze(0),
        reconstructed.float().flatten().unsqueeze(0),
    ).item()

    # Max absolute error
    max_error = torch.max(torch.abs(original.float() - reconstructed.float())).item()

    return {
        "mse": mse,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
        "max_error": max_error,
    }


def test_compression_rank(
    kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    d_compressed: int,
    quant_bits: int = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Test compression at a specific rank."""
    # Get dimensions from first layer
    k_sample, v_sample = kv_pairs[0]
    # Shape: [B, n_heads, T, head_dim]
    B, n_heads, T, head_dim = k_sample.shape
    d_input = head_dim

    # Create compressor config
    config = KVCompressorConfig(
        d_input=d_input,
        d_compressed=min(d_compressed, d_input),
        n_heads=n_heads,
        device=device,
        dtype=torch.float16,
        quant_bits=quant_bits,
        quant_storage=quant_bits is not None,
    )

    # Create compressor
    compressor = OrthogonalCompressor(config)

    # Calibrate with concatenated K, V
    all_kv = []
    for k, v in kv_pairs:
        # Reshape: [B, n_heads, T, head_dim] -> [B*n_heads*T, head_dim]
        k_flat = k.view(-1, head_dim)
        v_flat = v.view(-1, head_dim)
        all_kv.append(k_flat)
        all_kv.append(v_flat)

    cal_data = torch.cat(all_kv, dim=0)
    compressor.calibrate(cal_data)

    # Test round-trip on each layer
    k_errors = []
    v_errors = []

    for k, v in kv_pairs:
        # Reshape for compression
        orig_shape = k.shape
        k_flat = k.view(-1, head_dim)
        v_flat = v.view(-1, head_dim)

        # Compress
        k_compressed = compressor.compress(k_flat)
        v_compressed = compressor.compress(v_flat)

        # Expand
        k_reconstructed = compressor.expand(k_compressed)
        v_reconstructed = compressor.expand(v_compressed)

        # Reshape back
        k_reconstructed = k_reconstructed.view(orig_shape)
        v_reconstructed = v_reconstructed.view(orig_shape)

        # Measure errors
        k_err = measure_reconstruction_error(k, k_reconstructed)
        v_err = measure_reconstruction_error(v, v_reconstructed)

        k_errors.append(k_err)
        v_errors.append(v_err)

    # Average across layers
    avg_k_error = {
        k: sum(e[k] for e in k_errors) / len(k_errors) for k in k_errors[0].keys()
    }
    avg_v_error = {
        k: sum(e[k] for e in v_errors) / len(v_errors) for k in v_errors[0].keys()
    }

    compression_ratio = d_input / config.d_compressed

    return {
        "d_compressed": config.d_compressed,
        "compression_ratio": compression_ratio,
        "quant_bits": quant_bits or 16,
        "k_mse": avg_k_error["mse"],
        "k_cosine_sim": avg_k_error["cosine_similarity"],
        "k_rel_error": avg_k_error["relative_error"],
        "v_mse": avg_v_error["mse"],
        "v_cosine_sim": avg_v_error["cosine_similarity"],
        "v_rel_error": avg_v_error["relative_error"],
    }


def main():
    parser = argparse.ArgumentParser(description="Direct KV compression quality test")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to test",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="64,32,16,8,4,2,1",
        help="Ranks to test",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length for test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = [int(r) for r in args.ranks.split(",")]

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Ranks to test: {ranks}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Generate test input
    test_text = "The quick brown fox jumps over the lazy dog. " * 20
    input_ids = tokenizer(
        test_text, return_tensors="pt", truncation=True, max_length=args.seq_len
    ).input_ids.to(device)
    print(f"Input shape: {input_ids.shape}")

    # Extract K, V
    print("\nExtracting K, V from model...")
    kv_pairs, outputs = extract_kv_from_model(model, input_ids, device)
    print(f"Extracted {len(kv_pairs)} layers")
    if kv_pairs:
        k0, v0 = kv_pairs[0]
        print(f"K shape: {k0.shape}, V shape: {v0.shape}")
        head_dim = k0.shape[-1]
        print(f"Head dimension: {head_dim}")

    # Test each rank
    results = []
    print("\n" + "=" * 70)
    print("COMPRESSION QUALITY TEST")
    print("=" * 70)
    print(
        f"{'Rank':<6} {'Ratio':<8} {'Bits':<6} {'K_MSE':<12} {'K_CosSim':<10} {'V_MSE':<12} {'V_CosSim':<10}"
    )
    print("-" * 70)

    for rank in ranks:
        for bits in [None, 8, 4]:  # None = fp16
            try:
                result = test_compression_rank(kv_pairs, rank, bits, device)
                results.append(result)

                bits_str = str(bits) if bits else "16"
                print(
                    f"{result['d_compressed']:<6} "
                    f"{result['compression_ratio']:<8.1f}x "
                    f"{bits_str:<6} "
                    f"{result['k_mse']:<12.6f} "
                    f"{result['k_cosine_sim']:<10.6f} "
                    f"{result['v_mse']:<12.6f} "
                    f"{result['v_cosine_sim']:<10.6f}"
                )
            except Exception as e:
                print(f"Rank {rank}, bits {bits}: ERROR - {e}")

    # Find failure points
    print("\n" + "=" * 70)
    print("FAILURE BOUNDARY ANALYSIS")
    print("=" * 70)

    for threshold, name in [
        (0.99, "99% cosine similarity"),
        (0.95, "95% cosine similarity"),
        (0.90, "90% cosine similarity"),
    ]:
        failed_k = [r for r in results if r["k_cosine_sim"] < threshold]
        failed_v = [r for r in results if r["v_cosine_sim"] < threshold]

        if failed_k:
            worst = min(failed_k, key=lambda x: x["k_cosine_sim"])
            print(
                f"K below {name}: rank={worst['d_compressed']}, "
                f"ratio={worst['compression_ratio']:.1f}x, "
                f"bits={worst['quant_bits']}, "
                f"cos_sim={worst['k_cosine_sim']:.4f}"
            )
        else:
            print(f"K below {name}: NO FAILURE")

        if failed_v:
            worst = min(failed_v, key=lambda x: x["v_cosine_sim"])
            print(
                f"V below {name}: rank={worst['d_compressed']}, "
                f"ratio={worst['compression_ratio']:.1f}x, "
                f"bits={worst['quant_bits']}, "
                f"cos_sim={worst['v_cosine_sim']:.4f}"
            )
        else:
            print(f"V below {name}: NO FAILURE")
        print()

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "seq_len": args.seq_len,
                    "head_dim": head_dim if kv_pairs else None,
                    "n_layers": len(kv_pairs),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
