#!/usr/bin/env python3
"""
KV Cache Compressor Calibration Script

Calibrates KV compressor on held-out data and benchmarks quality.

Workflow:
1. Load HF model and wrap with compression
2. Collect KV samples from calibration data
3. Fit compressor parameters (minimize reconstruction loss)
4. Benchmark: memory usage and perplexity vs baseline

Usage:
    python scripts/calibrate_kv_compressor.py \
        --model gpt2 \
        --compressor kvsplice \
        --config compression_config.json \
        --calibration-dataset wikitext \
        --calibration-samples 1000 \
        --eval-dataset wikitext
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.kvsplice import KVSpliceCompressor
from gpt2.compression.pca import PCACompressor
from gpt2.compression.wrapper_hooks import CompressedKVModelWrapper


def load_compressor(compressor_type: str, config_path: str, device: str):
    """Load compressor from config file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Add global config if not present
    if "global" not in config:
        config["global"] = {
            "target_memory_reduction": config.get("target_memory_reduction", 0.5),
            "d_head": config.get("d_head", 64),
            "algo_default": config.get("algo_default", compressor_type),
        }

    if compressor_type == "kvsplice":
        compressor = KVSpliceCompressor(config)
    elif compressor_type == "pca":
        compressor = PCACompressor(config)
    else:
        raise ValueError(f"Unknown compressor type: {compressor_type}")

    return compressor.to(device)


def collect_calibration_samples(
    model,
    tokenizer,
    dataset_name: str,
    num_samples: int,
    max_length: int = 512,
    batch_size: int = 4,
):
    """
    Collect KV samples by running model on calibration data.

    Args:
        model: CompressedKVModelWrapper with compressor in calibration mode
        tokenizer: HF tokenizer
        dataset_name: Dataset name (e.g., "wikitext")
        num_samples: Number of samples to collect
        max_length: Max sequence length
        batch_size: Batch size for processing

    Returns:
        Number of samples collected
    """
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Dataset {dataset_name} not yet supported")

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 0)

    # Start calibration mode
    model.compressor.start_calibration()

    samples_collected = 0
    pbar = tqdm(total=num_samples, desc="Collecting calibration samples")

    for i in range(0, len(dataset), batch_size):
        if samples_collected >= num_samples:
            break

        batch_texts = [
            dataset[j][text_column] for j in range(i, min(i + batch_size, len(dataset)))
        ]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.model.device) for k, v in inputs.items()}

        # Forward pass (will collect KV samples via hooks)
        with torch.no_grad():
            _ = model(**inputs, use_cache=True)

        samples_collected += len(batch_texts)
        pbar.update(len(batch_texts))

    pbar.close()

    # End calibration and fit parameters
    print("\nFitting compressor parameters...")
    model.compressor.end_calibration()

    return samples_collected


def evaluate_perplexity(
    model, tokenizer, dataset_name: str, num_samples: int = 100, max_length: int = 512
):
    """
    Evaluate perplexity on held-out data.

    Args:
        model: Model to evaluate (baseline or compressed)
        tokenizer: HF tokenizer
        dataset_name: Dataset name
        num_samples: Number of samples for evaluation
        max_length: Max sequence length

    Returns:
        perplexity: Float
    """
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text_column = "text"
    else:
        raise ValueError(f"Dataset {dataset_name} not yet supported")

    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 0)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating perplexity"):
            text = example[text_column]

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {
                k: v.to(model.model.device if hasattr(model, "model") else model.device)
                for k, v in inputs.items()
            }

            # Compute loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Accumulate
            seq_len = inputs["input_ids"].size(1)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def benchmark_memory(model, compressor):
    """
    Estimate KV cache memory savings.

    Args:
        model: Wrapped model
        compressor: KVCompressor instance

    Returns:
        stats: Dict with memory statistics
    """
    stats = compressor.memory_stats()

    # Estimate bytes per token for KV cache
    # Each head: 2 * d_head * sizeof(float16) bytes per token
    bytes_per_token_orig = stats["total_original_params"] * 2 * 2  # K + V, fp16
    bytes_per_token_comp = stats["total_compressed_params"] * 2 * 2

    stats["bytes_per_token_original"] = bytes_per_token_orig
    stats["bytes_per_token_compressed"] = bytes_per_token_comp

    return stats


def main():
    parser = argparse.ArgumentParser(description="Calibrate KV compressor")
    parser.add_argument(
        "--model", type=str, default="gpt2", help="HF model name (default: gpt2)"
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="kvsplice",
        choices=["kvsplice", "pca"],
        help="Compressor type",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Compression config JSON (from analyze_fim_metrics.py)",
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="wikitext",
        help="Calibration dataset",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1000,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--eval-dataset", type=str, default="wikitext", help="Evaluation dataset"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=100, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Calibration batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compression_results",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KV Cache Compressor Calibration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compressor: {args.compressor}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = model.to(args.device)
    model.eval()

    # Evaluate baseline perplexity
    print("\n" + "=" * 80)
    print("Baseline Evaluation (No Compression)")
    print("=" * 80)
    baseline_ppl = evaluate_perplexity(
        model, tokenizer, args.eval_dataset, args.eval_samples, args.max_length
    )
    print(f"Baseline perplexity: {baseline_ppl:.2f}")

    # Load compressor
    print("\n" + "=" * 80)
    print("Loading Compressor")
    print("=" * 80)
    compressor = load_compressor(args.compressor, args.config, args.device)

    # Wrap model with compression
    print("Wrapping model with compression...")
    wrapped_model = CompressedKVModelWrapper(model, compressor)

    # Calibrate compressor
    print("\n" + "=" * 80)
    print("Calibrating Compressor")
    print("=" * 80)
    start_time = time.time()
    samples_collected = collect_calibration_samples(
        wrapped_model,
        tokenizer,
        args.calibration_dataset,
        args.calibration_samples,
        args.max_length,
        args.batch_size,
    )
    calibration_time = time.time() - start_time
    print(f"Calibration completed in {calibration_time:.1f}s")
    print(f"Samples collected: {samples_collected}")

    # Benchmark memory
    print("\n" + "=" * 80)
    print("Memory Statistics")
    print("=" * 80)
    mem_stats = benchmark_memory(wrapped_model, compressor)
    print(f"Total heads: {len(compressor.layer_head_configs)}")
    print(f"Original KV params: {mem_stats['total_original_params']}")
    print(f"Compressed KV params: {mem_stats['total_compressed_params']}")
    print(f"Compression ratio: {mem_stats['compression_ratio']:.3f}")
    print(f"Memory savings: {mem_stats['memory_savings_pct']:.1f}%")
    print(f"Bytes per token (original): {mem_stats['bytes_per_token_original']}")
    print(f"Bytes per token (compressed): {mem_stats['bytes_per_token_compressed']}")

    # Evaluate compressed perplexity
    print("\n" + "=" * 80)
    print("Compressed Model Evaluation")
    print("=" * 80)
    compressed_ppl = evaluate_perplexity(
        wrapped_model, tokenizer, args.eval_dataset, args.eval_samples, args.max_length
    )
    print(f"Compressed perplexity: {compressed_ppl:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    ppl_degradation = ((compressed_ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"Baseline perplexity:    {baseline_ppl:.2f}")
    print(f"Compressed perplexity:  {compressed_ppl:.2f}")
    print(f"Perplexity degradation: {ppl_degradation:+.2f}%")
    print(f"Memory savings:         {mem_stats['memory_savings_pct']:.1f}%")
    print(f"Calibration time:       {calibration_time:.1f}s")

    # Save results
    results = {
        "model": args.model,
        "compressor": args.compressor,
        "config": args.config,
        "baseline_perplexity": baseline_ppl,
        "compressed_perplexity": compressed_ppl,
        "perplexity_degradation_pct": ppl_degradation,
        "memory_stats": mem_stats,
        "calibration_time_sec": calibration_time,
        "calibration_samples": samples_collected,
    }

    results_file = output_dir / f"calibration_results_{args.compressor}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Save compressor state
    compressor_state_file = output_dir / f"compressor_{args.compressor}.pt"
    torch.save(compressor.state_dict(), compressor_state_file)
    print(f"Compressor state saved to: {compressor_state_file}")


if __name__ == "__main__":
    main()
