#!/usr/bin/env python3
"""
End-to-End FIM-Guided Heterogeneous Compression Pipeline

Validates the complete pipeline from PCA eigenvalue extraction through
heterogeneous rank optimization to final KVSplice calibration and
evaluation.

Pipeline stages:
1. PCA calibration → eigenvalue spectra
2. Spline smoothing + rank optimization → heterogeneous config
3. KVSplice calibration with heterogeneous ranks
4. Evaluation: heterogeneous vs uniform compression
5. Visualization: compression-PPL trade-off curves

Usage:
    python scripts/validate_fim_pipeline.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --calibration-samples 500 \
        --eval-samples 100 \
        --output fim_validation_results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.kvsplice import KVSpliceCompressor
from gpt2.compression.pca import PCACompressor
from gpt2.compression.spline_smoother import EigenvalueSplineSmoother
from gpt2.compression.wrapper_hooks import CompressedKVModelWrapper


def load_calibration_data(tokenizer, num_samples, max_length=512):
    """Load calibration data from FineWeb-Edu."""
    print(f"\nLoading {num_samples} calibration samples...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        text = example["text"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        samples.append(tokens["input_ids"])

    print(f"  Loaded {len(samples)} samples")
    return samples


def calibrate_pca_and_extract_eigenvalues(
    model, tokenizer, samples, output_dir, device="cuda"
):
    """
    Stage 1: PCA calibration to extract eigenvalue spectra.

    Returns path to eigenvalue_spectra.json
    """
    print("\n" + "=" * 70)
    print("Stage 1: PCA Calibration → Eigenvalue Extraction")
    print("=" * 70)

    # Get model config
    n_layer = model.config.num_hidden_layers
    n_head = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_head)
    d_head = model.config.hidden_size // n_head

    print(f"\nModel config:")
    print(f"  Layers: {n_layer}")
    print(f"  Q heads: {n_head}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Head dim: {d_head}")

    # Create PCA config (full rank for eigenvalue extraction)
    pca_config = {
        "global": {"d_head": d_head, "algo_default": "pca"},
        "per_layer_head": {},
    }

    # Enable PCA for all KV heads (GQA-aware)
    for layer_idx in range(n_layer):
        for head_idx in range(n_kv_heads):
            pca_config["per_layer_head"][(layer_idx, head_idx)] = {
                "enabled": True,
                "rank": d_head,  # Full rank for eigenvalue extraction
                "d_k": d_head,
                "d_v": d_head,
                "algo": "pca",
            }

    # Create PCA compressor and wrapper
    pca_compressor = PCACompressor(pca_config)
    wrapped_model = CompressedKVModelWrapper(model, pca_compressor, auto_patch=True)

    # Calibrate
    print(f"\nCalibrating PCA on {len(samples)} samples...")
    wrapped_model.start_calibration()

    model.to(device)
    for i, sample in enumerate(tqdm(samples, desc="PCA calibration")):
        with torch.no_grad():
            _ = wrapped_model(sample.to(device), use_cache=True)

    wrapped_model.end_calibration()
    model.to("cpu")

    # Save eigenvalue spectra
    eigenvalue_path = output_dir / "eigenvalue_spectra.json"
    pca_compressor.save_eigenvalue_spectra(str(eigenvalue_path))

    print(f"\n✓ Eigenvalue spectra saved to {eigenvalue_path}")

    # Clean up
    wrapped_model.remove_hooks()
    del wrapped_model
    del pca_compressor
    torch.cuda.empty_cache()

    return eigenvalue_path


def generate_heterogeneous_config(
    eigenvalue_path, model_config, output_dir, smoothing_factor=0.3
):
    """
    Stage 2: Spline smoothing + rank optimization → heterogeneous config.

    Returns path to fim_guided_config.json
    """
    print("\n" + "=" * 70)
    print("Stage 2: Eigenvalues → Heterogeneous Rank Selection")
    print("=" * 70)

    # Load eigenvalue spectra
    with open(eigenvalue_path) as f:
        spectra = json.load(f)

    print(f"\nLoaded eigenvalue spectra for {len(spectra)} heads")

    # Create spline smoother
    smoother = EigenvalueSplineSmoother(smoothing_factor=smoothing_factor)

    # Get model parameters
    n_head = model_config.num_attention_heads
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_head)
    d_head = model_config.hidden_size // n_head

    # Initialize config
    config = {
        "global": {
            "d_head": d_head,
            "algo_default": "kvsplice",
            "n_head": n_head,
            "n_kv_heads": n_kv_heads,
        },
        "per_layer_head": {},
        "stats": {"critical": 0, "good": 0, "moderate": 0, "excellent": 0},
    }

    ranks = []

    # Process each head
    for key, data in spectra.items():
        layer_idx, head_idx = map(int, key.split("/"))

        # Get eigenvalues for K
        eigenvalues = np.array(data["eigenvalues_k"])

        # Analyze with spline smoothing
        analysis = smoother.analyze_eigenvalue_curve(eigenvalues)

        # Select rank (moderate threshold by default)
        rank = analysis["rank_moderate"]

        # Classify head importance
        max_eigenvalue = analysis["max_eigenvalue"]
        total_variance = analysis["total_variance"]
        compression_ratio = rank / d_head

        if max_eigenvalue > 1000 or total_variance > 5000:
            category = "critical"
        elif max_eigenvalue > 100 or compression_ratio > 0.7:
            category = "good"
        elif max_eigenvalue > 10 or compression_ratio > 0.4:
            category = "moderate"
        else:
            category = "excellent"

        # Store config for this head (use string keys for JSON compatibility)
        head_key = f"layer{layer_idx}/head{head_idx}"
        config["per_layer_head"][head_key] = {
            "enabled": rank < d_head,
            "rank": int(rank),
            "algo": "kvsplice",
            "d_k": d_head,
            "d_v": d_head,
            # Diagnostic info
            "max_eigenvalue": float(max_eigenvalue),
            "total_variance": float(total_variance),
            "category": category,
            "rank_conservative": int(analysis["rank_conservative"]),
            "rank_moderate": int(analysis["rank_moderate"]),
            "rank_aggressive": int(analysis["rank_aggressive"]),
        }

        config["stats"][category] += 1
        ranks.append(rank)

    # Compute actual average compression
    avg_rank = np.mean(ranks)
    avg_compression = 1.0 - (avg_rank / d_head)
    config["global"]["actual_avg_compression"] = float(avg_compression)
    config["global"]["actual_avg_rank"] = float(avg_rank)
    config["global"]["min_rank"] = int(np.min(ranks))
    config["global"]["max_rank"] = int(np.max(ranks))

    # Save config
    config_path = output_dir / "fim_guided_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Heterogeneous config saved to {config_path}")
    print(f"\nConfig summary:")
    print(f"  Average rank: {avg_rank:.1f}")
    print(f"  Average compression: {avg_compression * 100:.1f}%")
    print(
        f"  Rank range: {config['global']['min_rank']} - {config['global']['max_rank']}"
    )
    print(f"\nHead categories:")
    for category, count in config["stats"].items():
        pct = count / len(spectra) * 100
        print(f"  {category:12s}: {count:3d} heads ({pct:5.1f}%)")

    return config_path


def calibrate_kvsplice(
    model, tokenizer, config_path, samples, output_dir, device="cuda"
):
    """
    Stage 3: KVSplice calibration with heterogeneous ranks.

    Returns calibrated compressor and wrapper.
    """
    print("\n" + "=" * 70)
    print("Stage 3: KVSplice Calibration with Heterogeneous Ranks")
    print("=" * 70)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"\nCalibrating KVSplice with heterogeneous ranks...")
    print(f"  Average rank: {config['global']['actual_avg_rank']:.1f}")
    print(
        f"  Average compression: {config['global']['actual_avg_compression'] * 100:.1f}%"
    )

    # Create KVSplice compressor and wrapper
    kvsplice_compressor = KVSpliceCompressor(config)
    wrapped_model = CompressedKVModelWrapper(
        model, kvsplice_compressor, auto_patch=True
    )

    # Calibrate
    print(f"\nCalibrating on {len(samples)} samples...")
    wrapped_model.start_calibration()

    model.to(device)
    for i, sample in enumerate(tqdm(samples, desc="KVSplice calibration")):
        with torch.no_grad():
            _ = wrapped_model(sample.to(device), use_cache=True)

    wrapped_model.end_calibration()

    print(f"\n✓ KVSplice calibration complete")

    return wrapped_model, kvsplice_compressor


def evaluate_perplexity(model, tokenizer, samples, device="cuda", desc="Evaluation"):
    """Evaluate perplexity on samples."""
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc=desc):
            sample = sample.to(device)

            # Forward pass
            outputs = model(sample, labels=sample, use_cache=False)
            loss = outputs.loss

            # Accumulate
            seq_len = sample.size(1)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    model.to("cpu")

    return perplexity, avg_loss


def compare_compression_methods(
    model,
    tokenizer,
    calib_samples,
    eval_samples,
    fim_config_path,
    output_dir,
    device="cuda",
):
    """
    Stage 4: Compare heterogeneous vs uniform compression.

    Evaluates:
    - Baseline (no compression)
    - Uniform compression (same rank for all heads)
    - Heterogeneous compression (FIM-guided per-head ranks)
    """
    print("\n" + "=" * 70)
    print("Stage 4: Compression Method Comparison")
    print("=" * 70)

    results = {}

    # 1. Baseline (no compression)
    print("\n1. Baseline (no compression)")
    ppl_baseline, loss_baseline = evaluate_perplexity(
        model, tokenizer, eval_samples, device, desc="Baseline PPL"
    )
    results["baseline"] = {
        "perplexity": ppl_baseline,
        "loss": loss_baseline,
        "compression": 0.0,
    }
    print(f"   Baseline PPL: {ppl_baseline:.4f}")

    # 2. Heterogeneous compression (FIM-guided)
    print("\n2. Heterogeneous compression (FIM-guided)")
    with open(fim_config_path) as f:
        fim_config = json.load(f)

    avg_compression = fim_config["global"]["actual_avg_compression"]
    print(f"   Average compression: {avg_compression * 100:.1f}%")

    # Calibrate and evaluate
    wrapped_hetero, _ = calibrate_kvsplice(
        model, tokenizer, fim_config_path, calib_samples, output_dir, device
    )

    ppl_hetero, loss_hetero = evaluate_perplexity(
        wrapped_hetero, tokenizer, eval_samples, device, desc="Heterogeneous PPL"
    )
    results["heterogeneous"] = {
        "perplexity": ppl_hetero,
        "loss": loss_hetero,
        "compression": avg_compression,
        "config": str(fim_config_path),
    }
    print(f"   Heterogeneous PPL: {ppl_hetero:.4f}")
    print(f"   PPL degradation: {((ppl_hetero / ppl_baseline) - 1) * 100:.2f}%")

    wrapped_hetero.remove_hooks()
    del wrapped_hetero
    torch.cuda.empty_cache()

    # 3. Uniform compression (for comparison)
    print("\n3. Uniform compression (same compression ratio)")

    # Create uniform config with same average compression
    uniform_rank = int(fim_config["global"]["actual_avg_rank"])
    n_layer = model.config.num_hidden_layers
    n_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )
    d_head = model.config.hidden_size // model.config.num_attention_heads

    uniform_config = {
        "global": {
            "d_head": d_head,
            "algo_default": "kvsplice",
            "actual_avg_rank": uniform_rank,
            "actual_avg_compression": 1.0 - (uniform_rank / d_head),
        },
        "per_layer_head": {},
    }

    for layer_idx in range(n_layer):
        for head_idx in range(n_kv_heads):
            head_key = f"layer{layer_idx}/head{head_idx}"
            uniform_config["per_layer_head"][head_key] = {
                "enabled": True,
                "rank": uniform_rank,
                "algo": "kvsplice",
                "d_k": d_head,
                "d_v": d_head,
            }

    uniform_config_path = output_dir / "uniform_config.json"
    with open(uniform_config_path, "w") as f:
        json.dump(uniform_config, f, indent=2)

    print(f"   Uniform rank: {uniform_rank}")

    # Calibrate and evaluate
    wrapped_uniform, _ = calibrate_kvsplice(
        model, tokenizer, uniform_config_path, calib_samples, output_dir, device
    )

    ppl_uniform, loss_uniform = evaluate_perplexity(
        wrapped_uniform, tokenizer, eval_samples, device, desc="Uniform PPL"
    )
    results["uniform"] = {
        "perplexity": ppl_uniform,
        "loss": loss_uniform,
        "compression": 1.0 - (uniform_rank / d_head),
        "config": str(uniform_config_path),
    }
    print(f"   Uniform PPL: {ppl_uniform:.4f}")
    print(f"   PPL degradation: {((ppl_uniform / ppl_baseline) - 1) * 100:.2f}%")

    wrapped_uniform.remove_hooks()
    del wrapped_uniform
    torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Comparison results saved to {results_path}")

    return results


def generate_visualization(results, output_dir):
    """
    Stage 5: Generate publication-quality comparison plots.
    """
    print("\n" + "=" * 70)
    print("Stage 5: Visualization")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Compression vs PPL
    ax = axes[0]

    methods = ["baseline", "heterogeneous", "uniform"]
    colors = {"baseline": "green", "heterogeneous": "blue", "uniform": "orange"}
    labels = {
        "baseline": "Baseline (no compression)",
        "heterogeneous": "Heterogeneous (FIM-guided)",
        "uniform": "Uniform (same rank)",
    }

    for method in methods:
        if method in results:
            compression = results[method]["compression"] * 100
            ppl = results[method]["perplexity"]
            ax.scatter(
                compression,
                ppl,
                s=200,
                c=colors[method],
                label=labels[method],
                marker="o",
                edgecolors="black",
                linewidths=1.5,
            )

    ax.set_xlabel("Compression Ratio (%)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Compression vs Perplexity Trade-off", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Relative PPL degradation
    ax = axes[1]

    baseline_ppl = results["baseline"]["perplexity"]
    methods_compare = ["heterogeneous", "uniform"]

    compressions = []
    degradations = []
    method_labels = []

    for method in methods_compare:
        if method in results:
            compression = results[method]["compression"] * 100
            ppl = results[method]["perplexity"]
            degradation = ((ppl / baseline_ppl) - 1) * 100

            compressions.append(compression)
            degradations.append(degradation)
            method_labels.append(labels[method].split(" (")[0])

    bars = ax.bar(
        method_labels,
        degradations,
        color=[colors["heterogeneous"], colors["uniform"]],
        edgecolor="black",
        linewidth=1.5,
    )

    # Annotate bars
    for bar, comp in zip(bars, compressions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{height:.2f}%\n({comp:.1f}% comp)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title(
        "Perplexity Degradation by Compression Method", fontsize=14, fontweight="bold"
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "fim_validation_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization saved to {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end FIM-guided heterogeneous compression validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to validate",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=500,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=100, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.3,
        help="Spline smoothing factor",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fim_validation_results",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computation"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIM-Guided Heterogeneous Compression Pipeline Validation")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Evaluation samples: {args.eval_samples}")
    print(f"Output directory: {output_dir}")

    # Load model and tokenizer
    print(f"\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded: {model.config.model_type}")

    # Load data
    all_samples = load_calibration_data(
        tokenizer,
        args.calibration_samples + args.eval_samples,
        args.max_length,
    )

    calib_samples = all_samples[: args.calibration_samples]
    eval_samples = all_samples[args.calibration_samples :]

    # Stage 1: PCA calibration → eigenvalue spectra
    eigenvalue_path = calibrate_pca_and_extract_eigenvalues(
        model, tokenizer, calib_samples, output_dir, args.device
    )

    # Stage 2: Spline smoothing + rank optimization
    config_path = generate_heterogeneous_config(
        eigenvalue_path, model.config, output_dir, args.smoothing_factor
    )

    # Stage 3-4: Compare compression methods
    results = compare_compression_methods(
        model,
        tokenizer,
        calib_samples,
        eval_samples,
        config_path,
        output_dir,
        args.device,
    )

    # Stage 5: Generate visualizations
    generate_visualization(results, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nResults summary:")
    print(f"  Baseline PPL: {results['baseline']['perplexity']:.4f}")
    print(
        f"  Heterogeneous PPL: {results['heterogeneous']['perplexity']:.4f} "
        f"({results['heterogeneous']['compression'] * 100:.1f}% compression)"
    )
    print(
        f"  Uniform PPL: {results['uniform']['perplexity']:.4f} "
        f"({results['uniform']['compression'] * 100:.1f}% compression)"
    )

    baseline_ppl = results["baseline"]["perplexity"]
    hetero_deg = ((results["heterogeneous"]["perplexity"] / baseline_ppl) - 1) * 100
    uniform_deg = ((results["uniform"]["perplexity"] / baseline_ppl) - 1) * 100

    print(f"\nPPL degradation:")
    print(f"  Heterogeneous: {hetero_deg:.2f}%")
    print(f"  Uniform: {uniform_deg:.2f}%")

    if hetero_deg < uniform_deg:
        improvement = uniform_deg - hetero_deg
        print(
            f"\n✓ Heterogeneous compression is {improvement:.2f}% better than uniform!"
        )
    else:
        print(
            f"\n⚠ Uniform compression performed better by {uniform_deg - hetero_deg:.2f}%"
        )

    print(f"\nAll results saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
