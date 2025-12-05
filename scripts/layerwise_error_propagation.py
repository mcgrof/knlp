#!/usr/bin/env python3
"""
Layer-wise Error Propagation Analysis.

Analyzes KV compression sensitivity per layer to identify:
- Robust layers (safe to compress)
- Fragile layers (avoid compression)

This informs mixed-mode compression strategies.

Usage:
    python scripts/layerwise_error_propagation.py --model Qwen/Qwen2.5-7B
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
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


def analyze_single_layer_ablation(
    model,
    tokenizer,
    calib_path: str,
    num_layers: int,
    device: str,
    eval_text: str,
    baseline_ppl: float,
    quantize_bits: int = 8,
) -> Dict[str, List[float]]:
    """
    Ablate compression on one layer at a time.

    For each layer, compress only that layer and measure PPL impact.
    """
    results = {
        "k_only": [],  # Compress K only at layer i
        "v_only": [],  # Compress V only at layer i
        "kv_both": [],  # Compress K and V at layer i
    }

    # Load base compressors
    k_comp_base, v_comp_base, metadata = load_calibrated_compressors(
        calib_path,
        device=torch.device(device),
        dtype=torch.float16,
        quantize_bits=quantize_bits,
    )

    print(f"\nSingle-layer ablation ({num_layers} layers):")

    for layer_idx in range(num_layers):
        print(f"  Layer {layer_idx}...", end=" ", flush=True)

        # K-only compression at this layer
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        k_comp[layer_idx] = k_comp_base[layer_idx]

        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
        ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
        k_delta = (ppl - baseline_ppl) / baseline_ppl
        results["k_only"].append(k_delta)

        # V-only compression at this layer
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp[layer_idx] = v_comp_base[layer_idx]

        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
        ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
        v_delta = (ppl - baseline_ppl) / baseline_ppl
        results["v_only"].append(v_delta)

        # K+V compression at this layer
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        k_comp[layer_idx] = k_comp_base[layer_idx]
        v_comp[layer_idx] = v_comp_base[layer_idx]

        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
        ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
        kv_delta = (ppl - baseline_ppl) / baseline_ppl
        results["kv_both"].append(kv_delta)

        print(f"K:{k_delta*100:+.2f}% V:{v_delta*100:+.2f}% KV:{kv_delta*100:+.2f}%")

    return results


def analyze_cumulative_layers(
    model,
    tokenizer,
    calib_path: str,
    num_layers: int,
    device: str,
    eval_text: str,
    baseline_ppl: float,
    quantize_bits: int = 8,
) -> Dict[str, List[float]]:
    """
    Analyze cumulative compression from first/last layers.

    Compress layers 0..i or layers i..N and measure PPL impact.
    """
    results = {
        "first_n": [],  # Compress layers 0..i
        "last_n": [],  # Compress layers (N-i)..N
    }

    k_comp_base, v_comp_base, metadata = load_calibrated_compressors(
        calib_path,
        device=torch.device(device),
        dtype=torch.float16,
        quantize_bits=quantize_bits,
    )

    print(f"\nCumulative layer analysis:")

    for n in range(num_layers + 1):
        # Compress first n layers (V-only)
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        for i in range(n):
            v_comp[i] = v_comp_base[i]

        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
        ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
        first_delta = (ppl - baseline_ppl) / baseline_ppl
        results["first_n"].append(first_delta)

        # Compress last n layers (V-only)
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        for i in range(num_layers - n, num_layers):
            v_comp[i] = v_comp_base[i]

        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
        ppl = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
        last_delta = (ppl - baseline_ppl) / baseline_ppl
        results["last_n"].append(last_delta)

        print(
            f"  n={n}: first_{n}:{first_delta*100:+.2f}% last_{n}:{last_delta*100:+.2f}%"
        )

    return results


def plot_sensitivity_heatmap(
    single_layer_results: Dict[str, List[float]],
    model_name: str,
    output_dir: str,
):
    """Plot layer sensitivity heatmap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_layers = len(single_layer_results["k_only"])

    # Create heatmap data
    data = np.array(
        [
            [x * 100 for x in single_layer_results["k_only"]],
            [x * 100 for x in single_layer_results["v_only"]],
            [x * 100 for x in single_layer_results["kv_both"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(max(12, num_layers * 0.5), 4))

    sns.heatmap(
        data,
        ax=ax,
        cmap="RdYlGn_r",  # Red = bad, Green = good
        center=0,
        annot=True if num_layers <= 16 else False,
        fmt=".1f",
        xticklabels=[str(i) for i in range(num_layers)],
        yticklabels=["K only", "V only", "K+V"],
        cbar_kws={"label": "ΔPPL (%)"},
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Compression Target", fontsize=12)
    ax.set_title(f"Layer Sensitivity Heatmap: {model_name}", fontsize=14)

    plt.tight_layout()
    model_short = model_name.split("/")[-1]
    plt.savefig(f"{output_dir}/sensitivity_heatmap_{model_short}.png", dpi=300)
    plt.close()


def plot_cumulative_analysis(
    cumulative_results: Dict[str, List[float]],
    model_name: str,
    output_dir: str,
):
    """Plot cumulative layer compression analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_layers = len(cumulative_results["first_n"]) - 1

    fig, ax = plt.subplots(figsize=(10, 6))

    x = list(range(num_layers + 1))
    ax.plot(
        x,
        [d * 100 for d in cumulative_results["first_n"]],
        "b-o",
        label="First N layers",
    )
    ax.plot(
        x, [d * 100 for d in cumulative_results["last_n"]], "r-o", label="Last N layers"
    )

    ax.set_xlabel("Number of Layers Compressed", fontsize=12)
    ax.set_ylabel("ΔPPL (%)", fontsize=12)
    ax.set_title(f"Cumulative Layer Compression: {model_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    model_short = model_name.split("/")[-1]
    plt.savefig(f"{output_dir}/cumulative_layers_{model_short}.png", dpi=300)
    plt.close()


def identify_layer_groups(
    single_layer_results: Dict[str, List[float]],
    threshold: float = 0.01,
) -> Dict[str, List[int]]:
    """
    Identify robust and fragile layers based on sensitivity.

    Returns dict with:
    - robust_k: Layers where K compression is safe
    - robust_v: Layers where V compression is safe
    - fragile_k: Layers where K compression is risky
    - fragile_v: Layers where V compression is risky
    """
    num_layers = len(single_layer_results["k_only"])

    groups = {
        "robust_k": [],
        "robust_v": [],
        "fragile_k": [],
        "fragile_v": [],
    }

    for i in range(num_layers):
        k_delta = single_layer_results["k_only"][i]
        v_delta = single_layer_results["v_only"][i]

        if k_delta <= threshold:
            groups["robust_k"].append(i)
        else:
            groups["fragile_k"].append(i)

        if v_delta <= threshold:
            groups["robust_v"].append(i)
        else:
            groups["fragile_v"].append(i)

    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise error propagation analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to analyze",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Calibration file path",
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
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/layer_sensitivity",
        help="Plot output directory",
    )
    args = parser.parse_args()

    print("Layer-wise Error Propagation Analysis")
    print("=" * 70)
    print(f"Model: {args.model}")

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

    print(f"Calibration: {calib_path}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    config = AutoConfig.from_pretrained(args.model)
    num_layers = config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    # Evaluation text
    eval_text = (
        "Machine learning has transformed how we approach complex problems. "
        "Neural networks can learn patterns from data without explicit programming. "
    ) * 10

    # Baseline PPL
    print(f"\nMeasuring baseline PPL...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_text, args.device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Single-layer ablation
    single_layer = analyze_single_layer_ablation(
        model, tokenizer, calib_path, num_layers, args.device, eval_text, baseline_ppl
    )

    # Cumulative analysis
    cumulative = analyze_cumulative_layers(
        model, tokenizer, calib_path, num_layers, args.device, eval_text, baseline_ppl
    )

    # Identify layer groups
    groups = identify_layer_groups(single_layer, threshold=0.01)

    # Results
    results = {
        "model": args.model,
        "num_layers": num_layers,
        "baseline_ppl": baseline_ppl,
        "single_layer": single_layer,
        "cumulative": cumulative,
        "layer_groups": groups,
    }

    # Save results
    if args.output is None:
        model_short = args.model.replace("/", "-").lower()
        args.output = f"results/layer_sensitivity_{model_short}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Generate plots
    plot_sensitivity_heatmap(single_layer, args.model, args.plot_dir)
    plot_cumulative_analysis(cumulative, args.model, args.plot_dir)
    print(f"Plots saved to: {args.plot_dir}/")

    # Summary
    print("\n" + "=" * 70)
    print("LAYER SENSITIVITY SUMMARY")
    print("=" * 70)

    print(f"\nRobust layers (ΔPPL ≤ 1% when compressed):")
    print(f"  K: {groups['robust_k']}")
    print(f"  V: {groups['robust_v']}")

    print(f"\nFragile layers (ΔPPL > 1% when compressed):")
    print(f"  K: {groups['fragile_k']}")
    print(f"  V: {groups['fragile_v']}")

    # Find most/least sensitive layers
    k_sensitivities = single_layer["k_only"]
    v_sensitivities = single_layer["v_only"]

    most_sensitive_k = sorted(range(num_layers), key=lambda i: -k_sensitivities[i])[:3]
    most_sensitive_v = sorted(range(num_layers), key=lambda i: -v_sensitivities[i])[:3]

    print(f"\nMost sensitive K layers: {most_sensitive_k}")
    print(f"Most sensitive V layers: {most_sensitive_v}")


if __name__ == "__main__":
    main()
