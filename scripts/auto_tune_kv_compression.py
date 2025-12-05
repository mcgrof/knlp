#!/usr/bin/env python3
"""
Auto-Tuner for KV Cache Compression.

Automatically finds the best compression configuration under a PPL budget.

Usage:
    # Basic auto-tune
    python scripts/auto_tune_kv_compression.py --model Qwen/Qwen2.5-0.5B --target-ppl-delta 0.05

    # With custom config
    python scripts/auto_tune_kv_compression.py --model Qwen/Qwen2.5-7B \
        --target-ppl-delta 0.05 \
        --rank-grid 96,80,64 \
        --bits-options 16,8

    # Generate preset only (use existing calibrations)
    python scripts/auto_tune_kv_compression.py --model Qwen/Qwen2.5-0.5B \
        --target-ppl-delta 0.05 --skip-calibration
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


def get_model_info(model_name: str) -> Dict:
    """Get model configuration info."""
    config = AutoConfig.from_pretrained(model_name)
    return {
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        "head_dim": config.hidden_size // config.num_attention_heads,
        "num_layers": config.num_hidden_layers,
    }


def get_calibration_path(model_name: str, rank: int) -> str:
    """Get calibration file path for model and rank."""
    model_short = model_name.replace("/", "-").lower()
    return f"kv_lowrank_calib_{model_short}_r{rank}.pt"


def run_calibration(model_name: str, rank: int, device: str = "cuda") -> str:
    """Run calibration if needed, return calibration file path."""
    calib_path = get_calibration_path(model_name, rank)

    if os.path.exists(calib_path):
        print(f"  Using existing calibration: {calib_path}")
        return calib_path

    print(f"  Running calibration for rank {rank}...")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_kv_lowrank.py",
            "--model",
            model_name,
            "--rank",
            str(rank),
            "--device",
            device,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Calibration failed: {result.stderr}")
        return None

    return calib_path


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


def evaluate_config(
    model,
    tokenizer,
    calib_path: str,
    target: str,
    quantize_bits: Optional[int],
    device: str,
    eval_text: str,
    num_layers: int,
) -> Tuple[float, float]:
    """
    Evaluate a compression configuration.

    Returns:
        (ppl, delta_pct) relative to baseline
    """
    # Load compressors
    k_comp, v_comp, metadata = load_calibrated_compressors(
        calib_path,
        device=torch.device(device),
        dtype=torch.float16,
        quantize_bits=quantize_bits,
    )

    # Apply target filter
    if target == "k":
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
    elif target == "v":
        k_comp = [IdentityCompressor() for _ in range(num_layers)]

    # Create cache and measure
    cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
    ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)

    del cache, k_comp, v_comp
    torch.cuda.empty_cache()

    return ppl


def auto_tune(
    model_name: str,
    target_ppl_delta: float = 0.05,
    rank_grid: List[int] = None,
    bits_options: List[int] = None,
    target: str = "v",
    device: str = "cuda",
    skip_calibration: bool = False,
    output_preset: str = None,
) -> Dict:
    """
    Auto-tune compression configuration.

    Args:
        model_name: HuggingFace model name
        target_ppl_delta: Maximum allowed PPL degradation (e.g., 0.05 = +5%)
        rank_grid: List of ranks to try (auto-computed if None)
        bits_options: List of bit widths to try [16, 8]
        target: "k", "v", or "kv"
        device: Device to use
        skip_calibration: Skip calibration, use existing files only
        output_preset: Output preset file path

    Returns:
        Best configuration dict
    """
    print(f"Auto-Tuning KV Compression")
    print(f"  Model: {model_name}")
    print(f"  Target PPL delta: {target_ppl_delta * 100:.1f}%")
    print(f"  Target: {target.upper()}")
    print("=" * 70)

    # Get model info
    model_info = get_model_info(model_name)
    head_dim = model_info["head_dim"]
    num_layers = model_info["num_layers"]
    print(f"\nModel info:")
    print(f"  Head dim: {head_dim}")
    print(f"  Layers: {num_layers}")

    # Auto-compute rank grid if not provided
    if rank_grid is None:
        # Generate ranks from head_dim down to ~50%
        rank_grid = []
        for ratio in [1.0, 1.1, 1.2, 1.33, 1.5, 1.6, 2.0]:
            rank = int(head_dim / ratio)
            if rank >= head_dim // 2:  # Don't go below 50%
                rank_grid.append(rank)
        rank_grid = sorted(set(rank_grid), reverse=True)

    if bits_options is None:
        bits_options = [16, 8]

    print(f"  Rank grid: {rank_grid}")
    print(f"  Bits options: {bits_options}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Evaluation text
    eval_text = (
        "Machine learning has transformed how we approach complex problems. "
        "Neural networks can learn patterns from data without explicit programming. "
    ) * 10

    # Baseline PPL
    print(f"\nMeasuring baseline PPL...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_text, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Run calibrations
    print(f"\nPreparing calibrations...")
    calib_files = {}
    for rank in rank_grid:
        if skip_calibration:
            path = get_calibration_path(model_name, rank)
            if os.path.exists(path):
                calib_files[rank] = path
            else:
                print(f"  Skipping rank {rank} (no calibration file)")
        else:
            path = run_calibration(model_name, rank, device)
            if path:
                calib_files[rank] = path

    # Evaluate all configurations
    print(f"\nEvaluating configurations...")
    results = []

    for rank in sorted(calib_files.keys(), reverse=True):
        calib_path = calib_files[rank]
        compression_ratio = head_dim / rank

        for bits in bits_options:
            quantize_bits = bits if bits < 16 else None

            # Calculate total compression
            total_compression = compression_ratio
            if quantize_bits == 8:
                total_compression *= 2  # 16->8 bits
            elif quantize_bits == 4:
                total_compression *= 4  # 16->4 bits

            config_name = f"r={rank}, {target.upper()}, bits={bits}"
            print(f"  Testing {config_name}...")

            try:
                ppl = evaluate_config(
                    model,
                    tokenizer,
                    calib_path,
                    target,
                    quantize_bits,
                    device,
                    eval_text,
                    num_layers,
                )
                delta = (ppl - baseline_ppl) / baseline_ppl

                results.append(
                    {
                        "rank": rank,
                        "target": target,
                        "bits": bits,
                        "compression_ratio": compression_ratio,
                        "total_compression": total_compression,
                        "ppl": ppl,
                        "delta": delta,
                        "calib_path": calib_path,
                    }
                )

                status = "PASS" if delta <= target_ppl_delta else "FAIL"
                print(f"    PPL: {ppl:.4f} ({delta*100:+.2f}%) [{status}]")

            except Exception as e:
                print(f"    Error: {e}")

    # Find best configuration
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    passing = [r for r in results if r["delta"] <= target_ppl_delta]

    if not passing:
        print(
            f"No configuration meets the target PPL delta of {target_ppl_delta*100:.1f}%"
        )
        print(f"Best available:")
        best = min(results, key=lambda x: x["delta"])
    else:
        # Select config with maximum compression among passing
        best = max(passing, key=lambda x: x["total_compression"])
        print(f"Best configuration meeting target:")

    print(f"  Rank: {best['rank']}")
    print(f"  Target: {best['target'].upper()}")
    print(f"  Bits: {best['bits']}")
    print(f"  Rank compression: {best['compression_ratio']:.2f}x")
    print(f"  Total compression: {best['total_compression']:.2f}x")
    print(f"  PPL: {best['ppl']:.4f}")
    print(f"  PPL delta: {best['delta']*100:+.2f}%")

    # Create preset
    preset = {
        "model": model_name,
        "version": "v9",
        "rank": best["rank"],
        "target": best["target"],
        "bits": best["bits"],
        "compression_ratio": best["compression_ratio"],
        "total_compression": best["total_compression"],
        "baseline_ppl": baseline_ppl,
        "compressed_ppl": best["ppl"],
        "ppl_delta": best["delta"],
        "calibration_file": best["calib_path"],
        "head_dim": head_dim,
        "num_layers": num_layers,
    }

    # Save preset
    if output_preset is None:
        model_short = model_name.replace("/", "-").lower()
        output_preset = f"kv_preset_{model_short}_v9.json"

    with open(output_preset, "w") as f:
        json.dump(preset, f, indent=2)

    print(f"\nPreset saved to: {output_preset}")

    # Summary table
    print(f"\n" + "=" * 70)
    print("ALL RESULTS")
    print("=" * 70)
    print(
        f"{'Rank':<8} {'Target':<6} {'Bits':<6} {'Compress':<10} {'PPL':<10} {'Delta':<10} {'Status'}"
    )
    print("-" * 70)

    for r in sorted(results, key=lambda x: -x["total_compression"]):
        status = "PASS" if r["delta"] <= target_ppl_delta else "FAIL"
        selected = " *BEST*" if r == best else ""
        print(
            f"{r['rank']:<8} {r['target'].upper():<6} {r['bits']:<6} {r['total_compression']:.2f}x{'':<5} "
            f"{r['ppl']:<10.4f} {r['delta']*100:+.2f}%{'':<5} {status}{selected}"
        )

    return preset


def main():
    parser = argparse.ArgumentParser(description="Auto-tune KV cache compression")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to tune",
    )
    parser.add_argument(
        "--target-ppl-delta",
        type=float,
        default=0.05,
        help="Maximum allowed PPL degradation (default: 0.05 = +5%%)",
    )
    parser.add_argument(
        "--rank-grid",
        type=str,
        default=None,
        help="Comma-separated ranks to try (auto-computed if not specified)",
    )
    parser.add_argument(
        "--bits-options",
        type=str,
        default="16,8",
        help="Comma-separated bit widths to try",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="v",
        choices=["k", "v", "kv"],
        help="What to compress (default: v)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration, use existing files only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output preset file path",
    )
    args = parser.parse_args()

    # Parse rank grid
    rank_grid = None
    if args.rank_grid:
        rank_grid = [int(x) for x in args.rank_grid.split(",")]

    bits_options = [int(x) for x in args.bits_options.split(",")]

    auto_tune(
        model_name=args.model,
        target_ppl_delta=args.target_ppl_delta,
        rank_grid=rank_grid,
        bits_options=bits_options,
        target=args.target,
        device=args.device,
        skip_calibration=args.skip_calibration,
        output_preset=args.output,
    )


if __name__ == "__main__":
    main()
