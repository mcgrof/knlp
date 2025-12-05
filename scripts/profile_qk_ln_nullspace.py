#!/usr/bin/env python3
"""
Profile Q/K LN nullspace FLOP reduction in terms of actual tokens/sec.

Tests generation speed with different LN nullspace configurations:
- baseline: No LN nullspace
- v_only: V-only LN nullspace (v15)
- k_only: K-only LN nullspace
- k_v: Both K and V LN nullspace

Measures wall-clock time and tokens/sec across context lengths.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    LayerNormNullspaceCompressor,
    QKLNNullspaceCompressor,
)


# Test prompts of varying complexity
TEST_PROMPTS = {
    "short": "The capital of France is",
    "medium": (
        "Machine learning is a branch of artificial intelligence that focuses on "
        "building systems that can learn from data. The field has seen tremendous "
        "growth in recent years, with applications ranging from"
    ),
    "long": (
        "Transformer models have revolutionized natural language processing since "
        "their introduction in 2017. The key innovation is the self-attention "
        "mechanism, which allows the model to weigh the importance of different "
        "parts of the input when producing each output. Unlike recurrent neural "
        "networks, transformers can process all positions in parallel during "
        "training, leading to significant speedups. The architecture consists of "
        "an encoder-decoder structure, though many modern models use only the "
        "decoder (like GPT) or only the encoder (like BERT). Key-value caching "
        "is essential for efficient autoregressive generation, as it avoids "
        "recomputing attention for previously generated tokens. This paper explores"
    ),
}


def create_compressors(
    mode: str,
    num_layers: int,
    head_dim: int,
    device: str = "cuda",
) -> Tuple[List, List]:
    """Create K and V compressors based on mode."""
    k_compressors = []
    v_compressors = []

    for _ in range(num_layers):
        if mode == "baseline":
            k_compressors.append(IdentityCompressor())
            v_compressors.append(IdentityCompressor())
        elif mode == "v_only":
            k_compressors.append(IdentityCompressor())
            v_compressors.append(
                LayerNormNullspaceCompressor(head_dim, device=torch.device(device))
            )
        elif mode == "k_only":
            k_compressors.append(
                QKLNNullspaceCompressor(head_dim, device=torch.device(device))
            )
            v_compressors.append(IdentityCompressor())
        elif mode == "k_v":
            k_compressors.append(
                QKLNNullspaceCompressor(head_dim, device=torch.device(device))
            )
            v_compressors.append(
                LayerNormNullspaceCompressor(head_dim, device=torch.device(device))
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return k_compressors, v_compressors


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    k_compressors: List,
    v_compressors: List,
    num_layers: int,
    num_runs: int = 3,
    warmup_runs: int = 1,
    device: str = "cuda",
) -> Dict:
    """Benchmark generation speed."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    times = []

    for run in range(warmup_runs + num_runs):
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if run >= warmup_runs:
            times.append(elapsed)

        # Clear cache
        del cache
        torch.cuda.empty_cache()

    generated_tokens = outputs.shape[1] - prompt_len
    avg_time = sum(times) / len(times)
    tokens_per_sec = generated_tokens / avg_time

    return {
        "prompt_len": prompt_len,
        "generated_tokens": generated_tokens,
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "times": times,
    }


def extend_prompt_to_length(tokenizer, base_prompt: str, target_tokens: int) -> str:
    """Extend prompt to approximately target token count."""
    tokens = tokenizer(base_prompt, return_tensors="pt").input_ids
    current_len = tokens.shape[1]

    if current_len >= target_tokens:
        # Truncate
        tokens = tokens[:, :target_tokens]
        return tokenizer.decode(tokens[0], skip_special_tokens=True)

    # Repeat and extend
    filler = " This is additional context to extend the prompt length."
    result = base_prompt

    while True:
        result += filler
        tokens = tokenizer(result, return_tensors="pt").input_ids
        if tokens.shape[1] >= target_tokens:
            tokens = tokens[:, :target_tokens]
            return tokenizer.decode(tokens[0], skip_special_tokens=True)


def run_profile(
    model_name: str,
    modes: List[str],
    context_lengths: List[int],
    max_new_tokens: int = 128,
    num_runs: int = 3,
    device: str = "cuda",
) -> List[Dict]:
    """Run full profiling suite."""
    print(f"\n{'='*70}")
    print(f"PROFILING: {model_name}")
    print(f"{'='*70}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads

    print(f"Layers: {num_layers}, Head dim: {head_dim}")
    print(f"Context lengths: {context_lengths}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Runs per config: {num_runs}")

    results = []
    base_prompt = TEST_PROMPTS["long"]

    for ctx_len in context_lengths:
        print(f"\n--- Context Length: {ctx_len} ---")

        # Extend prompt to target length
        prompt = extend_prompt_to_length(tokenizer, base_prompt, ctx_len)
        actual_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        print(f"Actual prompt tokens: {actual_len}")

        mode_results = {}

        for mode in modes:
            print(f"  {mode}: ", end="", flush=True)

            k_comp, v_comp = create_compressors(mode, num_layers, head_dim, device)

            try:
                result = benchmark_generation(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    k_comp,
                    v_comp,
                    num_layers,
                    num_runs,
                    warmup_runs=1,
                    device=device,
                )
                print(f"{result['tokens_per_sec']:.1f} tok/s")
                mode_results[mode] = result
            except Exception as e:
                print(f"ERROR: {e}")
                mode_results[mode] = None

        # Calculate speedups vs baseline
        if mode_results.get("baseline") and mode_results["baseline"]:
            baseline_speed = mode_results["baseline"]["tokens_per_sec"]
            for mode in modes:
                if mode != "baseline" and mode_results.get(mode):
                    speedup = mode_results[mode]["tokens_per_sec"] / baseline_speed
                    mode_results[mode]["speedup_vs_baseline"] = speedup

        results.append(
            {
                "model": model_name,
                "context_length": ctx_len,
                "actual_prompt_len": actual_len,
                "max_new_tokens": max_new_tokens,
                "modes": mode_results,
            }
        )

    return results


def print_summary(results: List[Dict]):
    """Print results summary."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n| Model | Context | Mode | Tokens/s | Speedup |")
    print("|-------|---------|------|----------|---------|")

    for r in results:
        model_short = r["model"].split("/")[-1]
        ctx = r["context_length"]

        for mode, data in r["modes"].items():
            if data:
                tps = data["tokens_per_sec"]
                speedup = data.get("speedup_vs_baseline", 1.0)
                speedup_str = f"{speedup:.3f}x" if mode != "baseline" else "-"
                print(f"| {model_short} | {ctx} | {mode} | {tps:.1f} | {speedup_str} |")


def plot_results(results: List[Dict], output_dir: Path):
    """Generate speed comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by model
    models = set(r["model"] for r in results)

    for model in models:
        model_results = [r for r in results if r["model"] == model]
        model_short = model.split("/")[-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        contexts = [r["context_length"] for r in model_results]
        modes = ["baseline", "v_only", "k_only", "k_v"]
        mode_labels = {
            "baseline": "Baseline",
            "v_only": "V-only LN nullspace",
            "k_only": "K-only LN nullspace",
            "k_v": "K+V LN nullspace",
        }
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for mode, color in zip(modes, colors):
            speeds = []
            valid_contexts = []
            for r in model_results:
                if r["modes"].get(mode) and r["modes"][mode]:
                    speeds.append(r["modes"][mode]["tokens_per_sec"])
                    valid_contexts.append(r["context_length"])
            if speeds:
                ax.plot(
                    valid_contexts,
                    speeds,
                    "o-",
                    label=mode_labels[mode],
                    color=color,
                    markersize=8,
                )

        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Tokens per Second")
        ax.set_title(f"{model_short}: Generation Speed vs Context Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_short}_qk_ln_speed.png", dpi=150)
        plt.close()

        print(f"Plot saved: {output_dir}/{model_short}_qk_ln_speed.png")

        # Also plot speedup
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode, color in zip(modes[1:], colors[1:]):  # Skip baseline
            speedups = []
            valid_contexts = []
            for r in model_results:
                if r["modes"].get(mode) and r["modes"][mode]:
                    su = r["modes"][mode].get("speedup_vs_baseline", 1.0)
                    speedups.append(su)
                    valid_contexts.append(r["context_length"])
            if speedups:
                ax.plot(
                    valid_contexts,
                    speedups,
                    "o-",
                    label=mode_labels[mode],
                    color=color,
                    markersize=8,
                )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Speedup vs Baseline")
        ax.set_title(f"{model_short}: LN Nullspace Speedup vs Context Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_short}_qk_ln_speedup.png", dpi=150)
        plt.close()

        print(f"Plot saved: {output_dir}/{model_short}_qk_ln_speedup.png")


def main():
    parser = argparse.ArgumentParser(description="Profile Q/K LN nullspace speed")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Model to profile",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/qk_ln_nullspace_speed",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    modes = ["baseline", "v_only", "k_only", "k_v"]

    results = run_profile(
        args.model,
        modes,
        args.context_lengths,
        args.max_new_tokens,
        args.num_runs,
        args.device,
    )

    print_summary(results)

    output_dir = Path(args.output_dir)
    plot_results(results, output_dir)


if __name__ == "__main__":
    main()
