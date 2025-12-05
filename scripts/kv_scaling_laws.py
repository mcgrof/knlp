#!/usr/bin/env python3
"""
KV Compression Scaling Laws Analysis.

Establishes scaling laws for KV compressibility across model sizes.

Key hypothesis: Larger models are more compressible (lower ΔPPL at same compression ratio).

Usage:
    python scripts/kv_scaling_laws.py --output results/scaling_laws.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


# Models to analyze (ordered by size)
MODELS = [
    ("Qwen/Qwen2.5-0.5B", 0.5),
    ("Qwen/Qwen2-1.5B", 1.5),
    ("Qwen/Qwen2.5-7B", 7.0),
    ("mistralai/Mistral-7B-v0.1", 7.0),
]


def get_compression_ratios(head_dim: int) -> List[float]:
    """Generate compression ratios to test."""
    # Test 1.0x (baseline), 1.1x, 1.25x, 1.33x, 1.5x, 2.0x
    ratios = [1.0, 1.1, 1.25, 1.33, 1.5, 2.0]
    return ratios


def ratio_to_rank(ratio: float, head_dim: int) -> int:
    """Convert compression ratio to rank."""
    return max(1, int(head_dim / ratio))


def measure_ppl(
    model,
    tokenizer,
    text: str,
    device: str,
    cache=None,
) -> float:
    """Measure perplexity on text."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        if cache is not None:
            outputs = model(input_ids, labels=input_ids, past_key_values=cache)
        else:
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def find_calibration_file(model_name: str, rank: int) -> str:
    """Find calibration file for model and rank."""
    model_short = model_name.replace("/", "-").lower()
    path = f"kv_lowrank_calib_{model_short}_r{rank}.pt"
    if Path(path).exists():
        return path
    return None


def analyze_model(
    model_name: str,
    model_size_b: float,
    device: str = "cuda",
    eval_text: str = None,
) -> Dict:
    """
    Analyze compression scaling for a single model.

    Returns dict with results for each compression ratio.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name} ({model_size_b}B)")
    print("=" * 70)

    # Get model config
    config = AutoConfig.from_pretrained(model_name)
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers

    print(f"  Head dim: {head_dim}")
    print(f"  Layers: {num_layers}")

    # Load model
    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Evaluation text
    if eval_text is None:
        eval_text = (
            "Machine learning has transformed how we approach complex problems. "
            "Neural networks can learn patterns from data without explicit programming. "
            "Deep learning models use multiple layers to extract hierarchical features. "
            "The field continues to advance with new architectures and training methods. "
        ) * 10

    # Baseline PPL
    print(f"  Measuring baseline PPL...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_text, device)
    print(f"    Baseline PPL: {baseline_ppl:.4f}")

    # Test each compression ratio
    results = {
        "model": model_name,
        "size_b": model_size_b,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "baseline_ppl": baseline_ppl,
        "compression_results": [],
    }

    ratios = get_compression_ratios(head_dim)

    for ratio in ratios:
        rank = ratio_to_rank(ratio, head_dim)
        actual_ratio = head_dim / rank

        print(f"\n  Testing compression ratio {actual_ratio:.2f}x (rank={rank})...")

        if actual_ratio == 1.0:
            # Baseline (no compression)
            ppl = baseline_ppl
            delta = 0.0
        else:
            # Find calibration file
            calib_path = find_calibration_file(model_name, rank)

            if calib_path is None:
                print(f"    No calibration file found, skipping")
                continue

            try:
                # Load compressors
                k_comp, v_comp, metadata = load_calibrated_compressors(
                    calib_path,
                    device=torch.device(device),
                    dtype=torch.float16,
                    quantize_bits=8,  # Use int8 for V
                )

                # V-only compression (most robust)
                k_comp = [IdentityCompressor() for _ in range(num_layers)]

                cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
                ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
                delta = (ppl - baseline_ppl) / baseline_ppl

                del cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error: {e}")
                continue

        # With int8, effective compression is 2x the rank compression
        total_compression = actual_ratio * 2 if actual_ratio > 1.0 else 1.0

        result = {
            "rank": rank,
            "rank_ratio": actual_ratio,
            "total_compression": total_compression,
            "ppl": ppl,
            "delta_ppl": delta,
        }
        results["compression_results"].append(result)

        print(f"    Total compression: {total_compression:.2f}x")
        print(f"    PPL: {ppl:.4f} ({delta*100:+.2f}%)")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def plot_scaling_laws(all_results: List[Dict], output_dir: str = "plots/scaling"):
    """Generate scaling law plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: ΔPPL vs Compression Ratio (all models)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))

    for i, result in enumerate(all_results):
        model_name = result["model"].split("/")[-1]
        size = result["size_b"]

        compressions = [r["total_compression"] for r in result["compression_results"]]
        deltas = [r["delta_ppl"] * 100 for r in result["compression_results"]]

        ax.plot(
            compressions,
            deltas,
            "o-",
            color=colors[i],
            label=f"{model_name} ({size}B)",
            markersize=8,
        )

    ax.set_xlabel("Total Compression Ratio", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title("KV Compression Scaling: Larger Models Compress Better", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.9)
    ax.set_ylim(bottom=-0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ppl_vs_compression.png", dpi=300)
    plt.close()

    # Plot 2: Compression Ratio at Fixed ΔPPL vs Model Size
    fig, ax = plt.subplots(figsize=(10, 6))

    target_deltas = [0.01, 0.03, 0.05]  # 1%, 3%, 5% degradation

    for target in target_deltas:
        sizes = []
        max_compressions = []

        for result in all_results:
            size = result["size_b"]

            # Find max compression at this delta
            max_comp = 1.0
            for r in result["compression_results"]:
                if r["delta_ppl"] <= target and r["total_compression"] > max_comp:
                    max_comp = r["total_compression"]

            sizes.append(size)
            max_compressions.append(max_comp)

        ax.plot(
            sizes,
            max_compressions,
            "o-",
            label=f"ΔPPL ≤ {target*100:.0f}%",
            markersize=10,
        )

    ax.set_xlabel("Model Size (B parameters)", fontsize=12)
    ax.set_ylabel("Max Compression Ratio", fontsize=12)
    ax.set_title("Maximum Compression at Fixed Quality Loss", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/compression_vs_model_size.png", dpi=300)
    plt.close()

    # Plot 3: ΔPPL at Fixed Compression vs Model Size
    fig, ax = plt.subplots(figsize=(10, 6))

    target_compressions = [1.5, 2.0, 2.5]

    for target in target_compressions:
        sizes = []
        deltas = []

        for result in all_results:
            size = result["size_b"]

            # Find delta at this compression (interpolate)
            compressions = [
                r["total_compression"] for r in result["compression_results"]
            ]
            ppl_deltas = [r["delta_ppl"] * 100 for r in result["compression_results"]]

            if target <= max(compressions):
                # Linear interpolation
                delta = np.interp(target, compressions, ppl_deltas)
                sizes.append(size)
                deltas.append(delta)

        if sizes:
            ax.plot(
                sizes, deltas, "o-", label=f"{target:.1f}x compression", markersize=10
            )

    ax.set_xlabel("Model Size (B parameters)", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title("Quality Loss at Fixed Compression vs Model Size", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/delta_ppl_vs_model_size.png", dpi=300)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="KV compression scaling laws")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all supported)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scaling_laws.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/scaling",
        help="Plot output directory",
    )
    args = parser.parse_args()

    # Select models
    if args.models:
        model_names = args.models.split(",")
        models = [(m.strip(), 0.0) for m in model_names]  # Size unknown
    else:
        models = MODELS

    print("KV Compression Scaling Laws Analysis")
    print("=" * 70)
    print(f"Models: {[m[0] for m in models]}")

    # Analyze each model
    all_results = []

    for model_name, size in models:
        try:
            results = analyze_model(model_name, size, args.device)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Generate plots
    if all_results:
        plot_scaling_laws(all_results, args.plot_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SCALING LAWS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Size':<8} {'Best Compress':<15} {'ΔPPL'}")
    print("-" * 70)

    for result in all_results:
        model = result["model"].split("/")[-1]
        size = f"{result['size_b']}B"

        # Find best compression under 5% degradation
        best = None
        for r in result["compression_results"]:
            if r["delta_ppl"] <= 0.05:
                if best is None or r["total_compression"] > best["total_compression"]:
                    best = r

        if best:
            print(
                f"{model:<30} {size:<8} {best['total_compression']:.2f}x{'':<10} "
                f"{best['delta_ppl']*100:+.2f}%"
            )


if __name__ == "__main__":
    main()
