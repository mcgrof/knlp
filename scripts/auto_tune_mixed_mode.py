#!/usr/bin/env python3
"""
Mixed-Mode Auto-Tuner for KV Cache Compression.

Searches for optimal K/V compression configurations using:
- K_strategy: none, low-rank, low-rank-int8
- V_strategy: low-rank, low-rank-int8, low-rank-int4
- Per-layer toggles
- Even/odd layer strategies

Goal: Find configs reaching 3.0-3.5x with <3% ΔPPL.

Usage:
    python scripts/auto_tune_mixed_mode.py --model Qwen/Qwen2.5-7B --target-ppl-delta 0.03
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    create_mixed_mode_compressors,
)


@dataclass
class MixedModeConfig:
    """Configuration for mixed-mode compression."""

    name: str
    k_mode: str  # "identity", "full"
    v_mode: str  # "identity", "full"
    k_bits: Optional[int]  # None (FP16), 8, 4
    v_bits: Optional[int]  # None (FP16), 8, 4
    layer_strategy: str  # "all", "even", "odd", "middle", "first_half", "last_half"

    def get_layer_mask(self, num_layers: int) -> List[bool]:
        """Generate layer mask based on strategy."""
        if self.layer_strategy == "all":
            return [True] * num_layers
        elif self.layer_strategy == "even":
            return [i % 2 == 0 for i in range(num_layers)]
        elif self.layer_strategy == "odd":
            return [i % 2 == 1 for i in range(num_layers)]
        elif self.layer_strategy == "middle":
            # Skip first and last 4 layers
            skip = min(4, num_layers // 4)
            return [skip <= i < num_layers - skip for i in range(num_layers)]
        elif self.layer_strategy == "first_half":
            return [i < num_layers // 2 for i in range(num_layers)]
        elif self.layer_strategy == "last_half":
            return [i >= num_layers // 2 for i in range(num_layers)]
        else:
            return [True] * num_layers


def generate_search_space() -> List[MixedModeConfig]:
    """Generate all configurations to search."""
    configs = []

    # K strategies
    k_strategies = [
        ("identity", None),  # No K compression
        ("full", None),  # K FP16
        ("full", 8),  # K int8
    ]

    # V strategies
    v_strategies = [
        ("full", None),  # V FP16
        ("full", 8),  # V int8
    ]

    # Layer strategies
    layer_strategies = ["all", "even", "odd", "middle"]

    for (k_mode, k_bits), (v_mode, v_bits), layer_strat in itertools.product(
        k_strategies, v_strategies, layer_strategies
    ):
        # Skip identity+identity (no compression)
        if k_mode == "identity" and v_mode == "identity":
            continue

        name = f"K:{k_mode}"
        if k_bits:
            name += f"_int{k_bits}"
        name += f"_V:{v_mode}"
        if v_bits:
            name += f"_int{v_bits}"
        name += f"_{layer_strat}"

        configs.append(
            MixedModeConfig(
                name=name,
                k_mode=k_mode,
                v_mode=v_mode,
                k_bits=k_bits,
                v_bits=v_bits,
                layer_strategy=layer_strat,
            )
        )

    return configs


def calculate_compression_ratio(
    config: MixedModeConfig,
    head_dim: int,
    rank: int,
    num_layers: int,
) -> float:
    """Calculate effective compression ratio for a config."""
    layer_mask = config.get_layer_mask(num_layers)
    layers_compressed = sum(layer_mask)
    layer_ratio = layers_compressed / num_layers

    # Base compression from rank
    rank_ratio = head_dim / rank

    # K compression
    if config.k_mode == "identity":
        k_ratio = 1.0
    else:
        k_ratio = rank_ratio
        if config.k_bits == 8:
            k_ratio *= 2
        elif config.k_bits == 4:
            k_ratio *= 4

    # V compression
    if config.v_mode == "identity":
        v_ratio = 1.0
    else:
        v_ratio = rank_ratio
        if config.v_bits == 8:
            v_ratio *= 2
        elif config.v_bits == 4:
            v_ratio *= 4

    # Combined K+V compression (harmonic mean)
    combined_ratio = 2.0 / (1.0 / k_ratio + 1.0 / v_ratio)

    # Adjust for layer coverage
    effective_ratio = 1.0 + (combined_ratio - 1.0) * layer_ratio

    return effective_ratio


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
    config: MixedModeConfig,
    calib_path: str,
    num_layers: int,
    device: str,
    eval_text: str,
) -> Tuple[float, Dict]:
    """Evaluate a single configuration."""
    layer_mask = config.get_layer_mask(num_layers)

    try:
        k_comp, v_comp, metadata = create_mixed_mode_compressors(
            calib_path,
            k_mode=config.k_mode,
            v_mode=config.v_mode,
            k_bits=config.k_bits,
            v_bits=config.v_bits,
            layer_mask=layer_mask,
            device=torch.device(device),
            dtype=torch.float16,
        )

        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
        ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)

        del cache, k_comp, v_comp
        torch.cuda.empty_cache()

        return ppl, metadata

    except Exception as e:
        return None, {"error": str(e)}


def auto_tune_mixed_mode(
    model_name: str,
    calib_path: str,
    target_ppl_delta: float = 0.03,
    device: str = "cuda",
    output_preset: str = None,
) -> Dict:
    """
    Auto-tune mixed-mode compression configuration.

    Searches for the best compression under the PPL budget.
    """
    print("Mixed-Mode Auto-Tuner")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Calibration: {calib_path}")
    print(f"Target ΔPPL: {target_ppl_delta * 100:.1f}%")
    print("=" * 70)

    # Load model config
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads

    # Get rank from calibration
    calib_data = torch.load(calib_path, map_location="cpu")
    rank = calib_data["rank"]

    print(f"\nModel info:")
    print(f"  Layers: {num_layers}")
    print(f"  Head dim: {head_dim}")
    print(f"  Calibration rank: {rank}")

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
        "Deep learning models use multiple layers to extract hierarchical features. "
    ) * 10

    # Baseline PPL
    print(f"\nMeasuring baseline PPL...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_text, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Generate search space
    search_space = generate_search_space()
    print(f"\nSearch space: {len(search_space)} configurations")

    # Evaluate all configurations
    results = []

    print(f"\nEvaluating configurations...")
    print("-" * 70)

    for i, cfg in enumerate(search_space):
        compression = calculate_compression_ratio(cfg, head_dim, rank, num_layers)

        print(
            f"[{i+1}/{len(search_space)}] {cfg.name} (~{compression:.2f}x)...", end=" "
        )

        ppl, metadata = evaluate_config(
            model, tokenizer, cfg, calib_path, num_layers, device, eval_text
        )

        if ppl is not None:
            delta = (ppl - baseline_ppl) / baseline_ppl
            status = "PASS" if delta <= target_ppl_delta else "FAIL"

            results.append(
                {
                    "config": cfg.__dict__,
                    "compression": compression,
                    "ppl": ppl,
                    "delta": delta,
                    "pass": delta <= target_ppl_delta,
                }
            )

            print(f"PPL={ppl:.4f} ({delta*100:+.2f}%) [{status}]")
        else:
            print(f"Error: {metadata.get('error', 'unknown')}")

    # Sort by compression (descending)
    results.sort(key=lambda x: -x["compression"])

    # Find best passing configuration
    passing = [r for r in results if r["pass"]]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if passing:
        best = max(passing, key=lambda x: x["compression"])
        print(
            f"\nBest configuration meeting target ({target_ppl_delta*100:.1f}% ΔPPL):"
        )
        print(f"  Name: {best['config']['name']}")
        print(f"  Compression: {best['compression']:.2f}x")
        print(f"  PPL: {best['ppl']:.4f} ({best['delta']*100:+.2f}%)")
        print(
            f"  K mode: {best['config']['k_mode']} (bits: {best['config']['k_bits']})"
        )
        print(
            f"  V mode: {best['config']['v_mode']} (bits: {best['config']['v_bits']})"
        )
        print(f"  Layer strategy: {best['config']['layer_strategy']}")
    else:
        print(f"\nNo configuration met the {target_ppl_delta*100:.1f}% target.")
        best = min(results, key=lambda x: x["delta"])
        print(f"Closest: {best['config']['name']} ({best['delta']*100:+.2f}%)")

    # Top 10 by compression
    print(f"\n{'Config':<45} {'Compress':<10} {'ΔPPL':<10} {'Status'}")
    print("-" * 75)
    for r in results[:10]:
        name = r["config"]["name"][:44]
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{name:<45} {r['compression']:.2f}x{'':<5} {r['delta']*100:+.2f}%{'':<5} {status}"
        )

    # Save preset for best passing config
    if passing and output_preset is None:
        model_short = model_name.replace("/", "-").lower()
        output_preset = f"kv_preset_{model_short}_mixed_v12.json"

    if passing:
        preset = {
            "model": model_name,
            "version": "v12_mixed",
            "rank": rank,
            "k_mode": best["config"]["k_mode"],
            "v_mode": best["config"]["v_mode"],
            "k_bits": best["config"]["k_bits"],
            "v_bits": best["config"]["v_bits"],
            "layer_strategy": best["config"]["layer_strategy"],
            "compression": best["compression"],
            "baseline_ppl": baseline_ppl,
            "compressed_ppl": best["ppl"],
            "ppl_delta": best["delta"],
            "calibration_file": calib_path,
            "head_dim": head_dim,
            "num_layers": num_layers,
        }

        with open(output_preset, "w") as f:
            json.dump(preset, f, indent=2)
        print(f"\nPreset saved to: {output_preset}")

    return {
        "model": model_name,
        "baseline_ppl": baseline_ppl,
        "target_ppl_delta": target_ppl_delta,
        "results": results,
        "best": best if passing else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Mixed-mode auto-tuner")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to tune",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Calibration file path",
    )
    parser.add_argument(
        "--target-ppl-delta",
        type=float,
        default=0.03,
        help="Maximum allowed PPL degradation (default: 0.03 = 3%%)",
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
        default=None,
        help="Output preset file path",
    )
    args = parser.parse_args()

    # Find calibration file
    calib_path = args.calib
    if calib_path is None:
        import glob

        model_short = args.model.replace("/", "-").lower()
        matches = glob.glob(f"kv_lowrank_calib_{model_short}_r*.pt")
        if matches:
            calib_path = matches[0]
        else:
            print(f"Error: No calibration file found for {args.model}")
            return

    auto_tune_mixed_mode(
        model_name=args.model,
        calib_path=calib_path,
        target_ppl_delta=args.target_ppl_delta,
        device=args.device,
        output_preset=args.output,
    )


if __name__ == "__main__":
    main()
