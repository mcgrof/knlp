#!/usr/bin/env python3
"""
Stability Evaluation with Error Bars.

Runs multiple evaluations with different seeds to establish variance bounds.

Metrics:
- PPL (perplexity)
- Tokens/sec (throughput)
- Needle-test accuracy (long-context retrieval)

Usage:
    python scripts/eval_stability.py --model Qwen/Qwen2.5-0.5B --num-runs 3
    python scripts/eval_stability.py --model Qwen/Qwen2.5-7B \
        --preset kv_preset_qwen-qwen2.5-7b_v9.json --num-runs 3
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sample_text(seed: int = 42) -> str:
    """Get evaluation text (WikiText-2 style sample)."""
    # Use a fixed sample for consistent evaluation
    texts = [
        "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to fields including speech recognition.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.",
    ]
    rng = random.Random(seed)
    # Repeat and shuffle for more content
    extended = texts * 20
    rng.shuffle(extended)
    return " ".join(extended)


@torch.no_grad()
def measure_ppl_and_speed(
    model,
    tokenizer,
    text: str,
    device: str,
    cache: Optional[CompressedDynamicCache] = None,
    max_tokens: int = 1024,
) -> Tuple[float, float]:
    """
    Measure perplexity and tokens/sec.

    Returns:
        (perplexity, tokens_per_second)
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    input_ids = inputs.input_ids.to(device)
    seq_len = input_ids.shape[1]

    # Warmup
    if cache is not None:
        _ = model(input_ids[:, :32], past_key_values=cache, use_cache=True)
        cache = CompressedDynamicCache(
            cache.k_compressors, cache.v_compressors, cache.num_layers
        )

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    if cache is not None:
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
    else:
        outputs = model(input_ids, labels=input_ids)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - start

    # Compute PPL
    if hasattr(outputs, "loss") and outputs.loss is not None:
        ppl = torch.exp(outputs.loss).item()
    else:
        # Manual computation
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        ppl = torch.exp(loss).item()

    tokens_per_sec = seq_len / elapsed
    return ppl, tokens_per_sec


# Needle test components (simplified from eval_long_context_needle.py)
NEEDLE_TEMPLATES = [
    {
        "needle": "The secret code for the vault is {value}.",
        "question": "What is the secret code for the vault?",
        "values": ["7392", "4851", "2069", "8476", "3215"],
    },
    {
        "needle": "The capital of the fictional country Zephyria is {value}.",
        "question": "What is the capital of Zephyria?",
        "values": ["Windholm", "Crystalburg", "Starview", "Moondale", "Sunridge"],
    },
]

FILLER_PARAGRAPHS = [
    "Machine learning models have become increasingly sophisticated. "
    "Deep neural networks can now perform tasks that were once thought impossible.",
    "The history of computing spans several decades of rapid innovation. "
    "From room-sized mainframes to pocket-sized smartphones, technology has evolved.",
    "Climate scientists continue to study the effects of global warming. "
    "Rising sea levels and extreme weather events are concerning trends.",
    "The field of medicine has seen remarkable advances in recent decades. "
    "Gene therapy and personalized medicine offer new hope for patients.",
]


def run_needle_test(
    model,
    tokenizer,
    device: str,
    cache_factory,
    context_length: int = 512,
    num_trials: int = 3,
    seed: int = 42,
) -> float:
    """
    Run needle-in-haystack test.

    Returns:
        Success rate (0.0 to 1.0)
    """
    rng = random.Random(seed)
    successes = 0

    for trial in range(num_trials):
        # Select needle
        template = rng.choice(NEEDLE_TEMPLATES)
        value = rng.choice(template["values"])
        needle = template["needle"].format(value=value)
        question = template["question"]

        # Generate haystack
        haystack = ""
        while len(tokenizer.encode(haystack)) < context_length - 100:
            haystack += rng.choice(FILLER_PARAGRAPHS) + "\n\n"

        # Insert needle at random position
        pos = rng.randint(0, len(haystack) // 2)
        prompt = haystack[:pos] + needle + "\n\n" + haystack[pos:]
        prompt += f"\n\nQuestion: {question}\nAnswer:"

        # Truncate to context length
        tokens = tokenizer.encode(prompt)
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            prompt = tokenizer.decode(tokens)

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)

        cache = cache_factory() if cache_factory else None

        with torch.no_grad():
            if cache is not None:
                outputs = model(input_ids, past_key_values=cache, use_cache=True)
                past = outputs.past_key_values
                generated_ids = input_ids

                for _ in range(20):
                    outputs = model(
                        generated_ids[:, -1:], past_key_values=past, use_cache=True
                    )
                    past = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            else:
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        # Check if answer is correct
        if value.lower() in response.lower():
            successes += 1

    return successes / num_trials


def create_cache_factory(
    preset_path: Optional[str],
    num_layers: int,
    device: str,
):
    """Create a cache factory function."""
    if preset_path is None:
        return None

    with open(preset_path) as f:
        preset = json.load(f)

    k_comp, v_comp, metadata = load_calibrated_compressors(
        preset["calibration_file"],
        device=torch.device(device),
        dtype=torch.float16,
        quantize_bits=preset["bits"] if preset["bits"] < 16 else None,
    )

    # Apply target filter
    if preset["target"] == "v":
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
    elif preset["target"] == "k":
        v_comp = [IdentityCompressor() for _ in range(num_layers)]

    def factory():
        return CompressedDynamicCache(k_comp, v_comp, num_layers)

    return factory


def run_stability_evaluation(
    model_name: str,
    preset_path: Optional[str],
    num_runs: int = 3,
    device: str = "cuda",
    context_length: int = 512,
    output_dir: str = "plots/stability",
) -> Dict:
    """
    Run stability evaluation with multiple seeds.

    Returns:
        Results dict with mean and std for each metric
    """
    print("=" * 70)
    print("Stability Evaluation")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Preset: {preset_path or 'Baseline'}")
    print(f"Num runs: {num_runs}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Create cache factory
    cache_factory = create_cache_factory(preset_path, num_layers, device)

    # Run evaluations
    seeds = [42 + i * 1000 for i in range(num_runs)]

    baseline_ppls = []
    baseline_speeds = []
    baseline_needles = []

    compressed_ppls = []
    compressed_speeds = []
    compressed_needles = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{num_runs} (seed={seed}) ---")
        set_seed(seed)

        text = get_sample_text(seed)

        # Baseline evaluation
        print("  Baseline...")
        ppl, speed = measure_ppl_and_speed(
            model, tokenizer, text, device, cache=None, max_tokens=context_length
        )
        baseline_ppls.append(ppl)
        baseline_speeds.append(speed)
        print(f"    PPL: {ppl:.4f}, Speed: {speed:.0f} tok/s")

        needle_acc = run_needle_test(
            model, tokenizer, device, None, context_length, num_trials=5, seed=seed
        )
        baseline_needles.append(needle_acc)
        print(f"    Needle: {needle_acc:.0%}")

        # Compressed evaluation (if preset provided)
        if cache_factory is not None:
            print("  Compressed...")
            cache = cache_factory()
            ppl, speed = measure_ppl_and_speed(
                model, tokenizer, text, device, cache=cache, max_tokens=context_length
            )
            compressed_ppls.append(ppl)
            compressed_speeds.append(speed)
            print(f"    PPL: {ppl:.4f}, Speed: {speed:.0f} tok/s")

            needle_acc = run_needle_test(
                model, tokenizer, device, cache_factory, context_length,
                num_trials=5, seed=seed
            )
            compressed_needles.append(needle_acc)
            print(f"    Needle: {needle_acc:.0%}")

            del cache
            torch.cuda.empty_cache()

    # Compute statistics
    results = {
        "model": model_name,
        "preset": preset_path,
        "num_runs": num_runs,
        "seeds": seeds,
        "context_length": context_length,
        "baseline": {
            "ppl": {"mean": np.mean(baseline_ppls), "std": np.std(baseline_ppls)},
            "tokens_per_sec": {
                "mean": np.mean(baseline_speeds),
                "std": np.std(baseline_speeds),
            },
            "needle_accuracy": {
                "mean": np.mean(baseline_needles),
                "std": np.std(baseline_needles),
            },
            "raw": {
                "ppl": baseline_ppls,
                "tokens_per_sec": baseline_speeds,
                "needle_accuracy": baseline_needles,
            },
        },
    }

    if cache_factory is not None:
        results["compressed"] = {
            "ppl": {"mean": np.mean(compressed_ppls), "std": np.std(compressed_ppls)},
            "tokens_per_sec": {
                "mean": np.mean(compressed_speeds),
                "std": np.std(compressed_speeds),
            },
            "needle_accuracy": {
                "mean": np.mean(compressed_needles),
                "std": np.std(compressed_needles),
            },
            "raw": {
                "ppl": compressed_ppls,
                "tokens_per_sec": compressed_speeds,
                "needle_accuracy": compressed_needles,
            },
        }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std)")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Baseline':<25} {'Compressed':<25}")
    print("-" * 70)

    b = results["baseline"]
    print(
        f"{'PPL':<20} {b['ppl']['mean']:.4f} +/- {b['ppl']['std']:.4f}"
    )
    print(
        f"{'Tokens/sec':<20} {b['tokens_per_sec']['mean']:.0f} +/- {b['tokens_per_sec']['std']:.0f}"
    )
    print(
        f"{'Needle Acc':<20} {b['needle_accuracy']['mean']:.0%} +/- {b['needle_accuracy']['std']:.0%}"
    )

    if "compressed" in results:
        c = results["compressed"]
        print(f"\n{'Compressed:'}")
        print(
            f"{'PPL':<20} {c['ppl']['mean']:.4f} +/- {c['ppl']['std']:.4f}"
        )
        print(
            f"{'Tokens/sec':<20} {c['tokens_per_sec']['mean']:.0f} +/- {c['tokens_per_sec']['std']:.0f}"
        )
        print(
            f"{'Needle Acc':<20} {c['needle_accuracy']['mean']:.0%} +/- {c['needle_accuracy']['std']:.0%}"
        )

        # Deltas
        ppl_delta = (c["ppl"]["mean"] - b["ppl"]["mean"]) / b["ppl"]["mean"] * 100
        speed_delta = (
            (c["tokens_per_sec"]["mean"] - b["tokens_per_sec"]["mean"])
            / b["tokens_per_sec"]["mean"]
            * 100
        )
        print(f"\n{'Deltas:'}")
        print(f"  PPL delta: {ppl_delta:+.2f}%")
        print(f"  Speed delta: {speed_delta:+.2f}%")

    return results


def generate_plots(results: Dict, output_dir: str):
    """Generate stability plots with error bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    model_short = results["model"].split("/")[-1]
    has_compressed = "compressed" in results

    # Plot 1: PPL comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    configs = ["Baseline"]
    means = [results["baseline"]["ppl"]["mean"]]
    stds = [results["baseline"]["ppl"]["std"]]
    colors = ["steelblue"]

    if has_compressed:
        configs.append("Compressed (v9)")
        means.append(results["compressed"]["ppl"]["mean"])
        stds.append(results["compressed"]["ppl"]["std"])
        colors.append("coral")

    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    ax.set_ylabel("Perplexity")
    ax.set_title(f"PPL Stability: {model_short}")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ppl_stability_{model_short}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ppl_stability_{model_short}.png")

    # Plot 2: Tokens/sec comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    means = [results["baseline"]["tokens_per_sec"]["mean"]]
    stds = [results["baseline"]["tokens_per_sec"]["std"]]

    if has_compressed:
        means.append(results["compressed"]["tokens_per_sec"]["mean"])
        stds.append(results["compressed"]["tokens_per_sec"]["std"])

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    ax.set_ylabel("Tokens/sec")
    ax.set_title(f"Throughput Stability: {model_short}")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 10,
            f"{mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/speed_stability_{model_short}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/speed_stability_{model_short}.png")

    # Plot 3: Needle accuracy
    fig, ax = plt.subplots(figsize=(8, 6))

    means = [results["baseline"]["needle_accuracy"]["mean"] * 100]
    stds = [results["baseline"]["needle_accuracy"]["std"] * 100]

    if has_compressed:
        means.append(results["compressed"]["needle_accuracy"]["mean"] * 100)
        stds.append(results["compressed"]["needle_accuracy"]["std"] * 100)

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Needle Test Stability: {model_short}")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 110)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 2,
            f"{mean:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/needle_stability_{model_short}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/needle_stability_{model_short}.png")


def main():
    parser = argparse.ArgumentParser(description="Stability evaluation with error bars")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Compression preset JSON file",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs with different seeds",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/stability",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    results = run_stability_evaluation(
        model_name=args.model,
        preset_path=args.preset,
        num_runs=args.num_runs,
        device=args.device,
        context_length=args.context_length,
        output_dir=args.output_dir,
    )

    # Generate plots
    generate_plots(results, args.output_dir)

    # Save JSON results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
