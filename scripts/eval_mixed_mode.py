#!/usr/bin/env python3
"""
Evaluate mixed-mode KV compression strategies.

Tests different K/V compression configurations to find optimal balance.

Usage:
    python scripts/eval_mixed_mode.py --model Qwen/Qwen2.5-7B
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    create_mixed_mode_compressors,
    load_calibrated_compressors,
)


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


def evaluate_mixed_mode(
    model_name: str,
    calib_path: str,
    device: str = "cuda",
    target_ppl_delta: float = 0.03,
) -> List[Dict]:
    """
    Evaluate different mixed-mode configurations.

    Returns list of results sorted by compression ratio.
    """
    print(f"Mixed-Mode KV Compression Evaluation")
    print(f"  Model: {model_name}")
    print(f"  Calibration: {calib_path}")
    print(f"  Target PPL delta: {target_ppl_delta * 100:.1f}%")
    print("=" * 70)

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers

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

    # Configurations to test
    configs = [
        # (name, k_mode, v_mode, k_bits, v_bits)
        ("V-only FP16", "identity", "full", None, None),
        ("V-only int8", "identity", "full", None, 8),
        ("K+V FP16", "full", "full", None, None),
        ("K+V int8", "full", "full", 8, 8),
        ("K FP16, V int8", "full", "full", None, 8),
        ("K int8, V FP16", "full", "full", 8, None),
    ]

    # Also test layer-selective (compress only middle layers)
    layer_masks = {
        "all": [True] * num_layers,
        "middle_only": [False] * 4 + [True] * (num_layers - 8) + [False] * 4,
        "even_layers": [i % 2 == 0 for i in range(num_layers)],
    }

    results = []

    print(f"\nEvaluating configurations...")
    print("-" * 70)

    for name, k_mode, v_mode, k_bits, v_bits in configs:
        for mask_name, layer_mask in layer_masks.items():
            config_name = f"{name} ({mask_name})"
            print(f"  Testing: {config_name}...")

            try:
                k_comp, v_comp, metadata = create_mixed_mode_compressors(
                    calib_path,
                    k_mode=k_mode,
                    v_mode=v_mode,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    layer_mask=layer_mask,
                    device=torch.device(device),
                    dtype=torch.float16,
                )

                cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
                ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
                delta = (ppl - baseline_ppl) / baseline_ppl

                status = "PASS" if delta <= target_ppl_delta else "FAIL"
                print(
                    f"    Compression: {metadata['total_compression']:.2f}x | "
                    f"PPL: {ppl:.4f} ({delta*100:+.2f}%) [{status}]"
                )

                results.append(
                    {
                        "name": config_name,
                        "k_mode": k_mode,
                        "v_mode": v_mode,
                        "k_bits": k_bits,
                        "v_bits": v_bits,
                        "layer_mask": mask_name,
                        "k_compression": metadata["k_compression"],
                        "v_compression": metadata["v_compression"],
                        "total_compression": metadata["total_compression"],
                        "ppl": ppl,
                        "delta": delta,
                        "pass": delta <= target_ppl_delta,
                    }
                )

                del cache, k_comp, v_comp
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error: {e}")

    # Summary
    print(f"\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Sort by compression ratio
    results.sort(key=lambda x: -x["total_compression"])

    print(f"{'Config':<35} {'Compress':<10} {'PPL':<10} {'Delta':<10} {'Status'}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['name']:<35} {r['total_compression']:.2f}x{'':<5} "
            f"{r['ppl']:<10.4f} {r['delta']*100:+.2f}%{'':<5} {status}"
        )

    # Find best passing configuration
    passing = [r for r in results if r["pass"]]
    if passing:
        best = max(passing, key=lambda x: x["total_compression"])
        print(
            f"\nBest configuration meeting target ({target_ppl_delta*100:.1f}% delta):"
        )
        print(f"  {best['name']}")
        print(f"  Compression: {best['total_compression']:.2f}x")
        print(f"  PPL delta: {best['delta']*100:+.2f}%")
    else:
        print(f"\nNo configuration met the {target_ppl_delta*100:.1f}% target.")
        best = min(results, key=lambda x: x["delta"])
        print(f"Closest: {best['name']} ({best['delta']*100:+.2f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate mixed-mode KV compression")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Calibration file path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--target-ppl-delta",
        type=float,
        default=0.03,
        help="Target PPL delta (default: 0.03 = 3%%)",
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
        help="Output JSON file",
    )
    args = parser.parse_args()

    # Auto-detect calibration file
    calib_path = args.calib
    if calib_path is None:
        model_short = args.model.replace("/", "-").lower()
        # Try common patterns
        for pattern in [f"kv_lowrank_calib_{model_short}_r*.pt"]:
            import glob

            matches = glob.glob(pattern)
            if matches:
                calib_path = matches[0]
                break

    if calib_path is None:
        print(f"Error: No calibration file found for {args.model}")
        print("Run calibration first with scripts/calibrate_kv_lowrank.py")
        return

    results = evaluate_mixed_mode(
        model_name=args.model,
        calib_path=calib_path,
        device=args.device,
        target_ppl_delta=args.target_ppl_delta,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
