#!/usr/bin/env python3
"""
Benchmark sequential I/O patterns during model inference.

Measures actual I/O metrics during inference to compare baseline vs
optimized model layouts. Outputs JSON suitable for W&B logging.

Metrics captured:
- Page faults (minor/major)
- I/O bytes read
- Sequential run statistics
- Inference latency breakdown

IMPORTANT: Enforces CPU-only execution for I/O measurement accuracy.
"""

import os
import sys
import json
import time
import argparse
import resource
import mmap
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from contextlib import contextmanager

# Force CPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

import torch

assert not torch.cuda.is_available(), "CUDA should be disabled!"
DEVICE = torch.device("cpu")
print(f"[CPU ENFORCED] Running on: {DEVICE}")


@dataclass
class IOMetrics:
    """I/O metrics for a single inference run."""

    # Page faults
    minor_faults: int = 0  # Pages found in cache
    major_faults: int = 0  # Pages read from disk

    # /proc/self/io metrics (Linux)
    read_bytes: int = 0
    read_chars: int = 0

    # Timing (seconds)
    load_time: float = 0.0
    first_inference_time: float = 0.0
    subsequent_inference_time: float = 0.0
    total_time: float = 0.0

    # Derived metrics
    page_fault_rate: float = 0.0  # faults per second
    throughput_mb_s: float = 0.0  # MB/s read


@dataclass
class BenchmarkResult:
    """Complete benchmark result for W&B logging."""

    # Identifiers
    model_name: str = ""
    model_path: str = ""
    model_variant: str = "baseline"  # "baseline" or "optimized"

    # Model info
    model_size_bytes: int = 0
    num_tensors: int = 0
    format: str = ""  # "safetensors" or "gguf"

    # Test parameters
    num_warmup: int = 0
    num_iterations: int = 0
    sequence_length: int = 0
    batch_size: int = 0

    # Metrics
    metrics: IOMetrics = field(default_factory=IOMetrics)

    # Per-iteration data (for detailed analysis)
    iteration_metrics: List[Dict] = field(default_factory=list)


def read_proc_io() -> Dict[str, int]:
    """Read I/O statistics from /proc/self/io (Linux only)."""
    try:
        with open("/proc/self/io", "r") as f:
            stats = {}
            for line in f:
                key, value = line.strip().split(": ")
                stats[key] = int(value)
            return stats
    except (FileNotFoundError, PermissionError):
        return {}


def get_resource_usage() -> Dict[str, int]:
    """Get resource usage including page faults."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "max_rss_kb": usage.ru_maxrss,
    }


@contextmanager
def measure_io():
    """Context manager to measure I/O during a block."""
    # Before
    usage_before = get_resource_usage()
    io_before = read_proc_io()
    time_before = time.perf_counter()

    yield

    # After
    time_after = time.perf_counter()
    usage_after = get_resource_usage()
    io_after = read_proc_io()

    # Calculate deltas
    result = {
        "minor_faults": usage_after["minor_faults"] - usage_before["minor_faults"],
        "major_faults": usage_after["major_faults"] - usage_before["major_faults"],
        "elapsed_s": time_after - time_before,
    }

    if io_before and io_after:
        result["read_bytes"] = io_after.get("read_bytes", 0) - io_before.get(
            "read_bytes", 0
        )
        result["read_chars"] = io_after.get("rchar", 0) - io_before.get("rchar", 0)

    # Store in a way the caller can access
    measure_io.last_result = result


def drop_caches():
    """
    Attempt to drop filesystem caches (requires root or sudo).

    This ensures we measure cold-start I/O, not cached reads.
    """
    try:
        # Try to drop caches
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        print("Dropped filesystem caches")
        return True
    except (PermissionError, FileNotFoundError):
        print(
            "Warning: Could not drop caches (need root). Results may include cached reads."
        )
        return False


def load_model_for_benchmark(model_name: str, force_reload: bool = False):
    """
    Load model for benchmarking with controlled caching.

    Args:
        model_name: HuggingFace model name or local path
        force_reload: If True, try to drop caches first

    Returns:
        model, tokenizer, load_metrics
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
    )

    if force_reload:
        drop_caches()

    with measure_io():
        if "gpt2" in model_name.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        model = model.to(DEVICE)
        model.eval()

    load_metrics = measure_io.last_result
    return model, tokenizer, load_metrics


def run_inference(
    model,
    tokenizer,
    sequence_length: int = 64,
    batch_size: int = 1,
    num_tokens_generate: int = 10,
) -> Dict:
    """
    Run inference and measure I/O.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        sequence_length: Input sequence length
        batch_size: Batch size
        num_tokens_generate: Number of tokens to generate

    Returns:
        Metrics dictionary
    """
    # Create input
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, sequence_length), device=DEVICE
    )

    with measure_io():
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids)

            # Optional: generate a few tokens
            if num_tokens_generate > 0:
                for _ in range(num_tokens_generate):
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    outputs = model(input_ids)

    return measure_io.last_result


def benchmark_model(
    model_name: str,
    model_variant: str = "baseline",
    num_warmup: int = 2,
    num_iterations: int = 10,
    sequence_length: int = 64,
    batch_size: int = 1,
    num_tokens_generate: int = 10,
    drop_caches_before: bool = False,
) -> BenchmarkResult:
    """
    Run complete benchmark on a model.

    Args:
        model_name: HuggingFace model name or path
        model_variant: "baseline" or "optimized"
        num_warmup: Warmup iterations (not measured)
        num_iterations: Measured iterations
        sequence_length: Input sequence length
        batch_size: Batch size
        num_tokens_generate: Tokens to generate per iteration
        drop_caches_before: Try to drop caches before loading

    Returns:
        BenchmarkResult suitable for W&B
    """
    result = BenchmarkResult(
        model_name=model_name,
        model_variant=model_variant,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} ({model_variant})")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer, load_metrics = load_model_for_benchmark(
        model_name, force_reload=drop_caches_before
    )

    result.metrics.load_time = load_metrics["elapsed_s"]
    result.model_size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    result.num_tensors = len(list(model.parameters()))

    print(f"  Load time: {load_metrics['elapsed_s']:.2f}s")
    print(f"  Minor faults: {load_metrics['minor_faults']}")
    print(f"  Major faults: {load_metrics['major_faults']}")

    # Warmup
    print(f"\nWarmup ({num_warmup} iterations)...")
    for i in range(num_warmup):
        _ = run_inference(
            model, tokenizer, sequence_length, batch_size, num_tokens_generate
        )

    # Measured iterations
    print(f"\nMeasured iterations ({num_iterations})...")
    iteration_metrics = []
    total_minor = 0
    total_major = 0
    total_time = 0.0
    total_read_bytes = 0

    for i in range(num_iterations):
        metrics = run_inference(
            model, tokenizer, sequence_length, batch_size, num_tokens_generate
        )

        iteration_metrics.append(
            {
                "iteration": i,
                "minor_faults": metrics["minor_faults"],
                "major_faults": metrics["major_faults"],
                "elapsed_s": metrics["elapsed_s"],
                "read_bytes": metrics.get("read_bytes", 0),
            }
        )

        total_minor += metrics["minor_faults"]
        total_major += metrics["major_faults"]
        total_time += metrics["elapsed_s"]
        total_read_bytes += metrics.get("read_bytes", 0)

        if i == 0:
            result.metrics.first_inference_time = metrics["elapsed_s"]
            print(f"  First inference: {metrics['elapsed_s']*1000:.1f}ms")

    # Calculate averages
    result.metrics.minor_faults = total_minor
    result.metrics.major_faults = total_major
    result.metrics.total_time = total_time
    result.metrics.read_bytes = total_read_bytes

    if num_iterations > 1:
        subsequent_time = sum(m["elapsed_s"] for m in iteration_metrics[1:])
        result.metrics.subsequent_inference_time = subsequent_time / (
            num_iterations - 1
        )

    if total_time > 0:
        result.metrics.page_fault_rate = (total_minor + total_major) / total_time
        result.metrics.throughput_mb_s = (total_read_bytes / 1e6) / total_time

    result.iteration_metrics = iteration_metrics

    # Print summary
    print(f"\n--- Summary ---")
    print(
        f"Total page faults: {total_minor + total_major} (minor: {total_minor}, major: {total_major})"
    )
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg iteration: {total_time/num_iterations*1000:.1f}ms")
    if result.metrics.subsequent_inference_time > 0:
        print(f"Avg subsequent: {result.metrics.subsequent_inference_time*1000:.1f}ms")
    print(f"Page fault rate: {result.metrics.page_fault_rate:.1f}/s")

    return result


def compare_models(
    baseline_name: str,
    optimized_path: Optional[str] = None,
    **kwargs,
) -> Dict:
    """
    Compare baseline vs optimized model.

    Args:
        baseline_name: HuggingFace model name for baseline
        optimized_path: Path to optimized model (or None to skip)
        **kwargs: Passed to benchmark_model

    Returns:
        Comparison results for W&B
    """
    results = {}

    # Benchmark baseline
    baseline_result = benchmark_model(baseline_name, model_variant="baseline", **kwargs)
    results["baseline"] = asdict(baseline_result)

    # Benchmark optimized if provided
    if optimized_path:
        optimized_result = benchmark_model(
            optimized_path, model_variant="optimized", **kwargs
        )
        results["optimized"] = asdict(optimized_result)

        # Calculate improvement
        bl = baseline_result.metrics
        opt = optimized_result.metrics

        results["comparison"] = {
            "load_time_speedup": (
                bl.load_time / opt.load_time if opt.load_time > 0 else 0
            ),
            "first_inference_speedup": (
                bl.first_inference_time / opt.first_inference_time
                if opt.first_inference_time > 0
                else 0
            ),
            "page_fault_reduction": (
                (
                    bl.minor_faults
                    + bl.major_faults
                    - opt.minor_faults
                    - opt.major_faults
                )
                / (bl.minor_faults + bl.major_faults)
                * 100
                if (bl.minor_faults + bl.major_faults) > 0
                else 0
            ),
            "throughput_improvement": (
                (opt.throughput_mb_s - bl.throughput_mb_s) / bl.throughput_mb_s * 100
                if bl.throughput_mb_s > 0
                else 0
            ),
        }

    return results


def format_for_wandb(results: Dict) -> Dict:
    """
    Format results for W&B logging.

    Returns flat dictionary suitable for wandb.log().
    """
    wandb_metrics = {}

    for variant in ["baseline", "optimized"]:
        if variant not in results:
            continue

        prefix = f"io/{variant}"
        metrics = results[variant]["metrics"]

        wandb_metrics[f"{prefix}/load_time_s"] = metrics["load_time"]
        wandb_metrics[f"{prefix}/first_inference_ms"] = (
            metrics["first_inference_time"] * 1000
        )
        wandb_metrics[f"{prefix}/minor_faults"] = metrics["minor_faults"]
        wandb_metrics[f"{prefix}/major_faults"] = metrics["major_faults"]
        wandb_metrics[f"{prefix}/page_fault_rate"] = metrics["page_fault_rate"]
        wandb_metrics[f"{prefix}/throughput_mb_s"] = metrics["throughput_mb_s"]

        if metrics["subsequent_inference_time"] > 0:
            wandb_metrics[f"{prefix}/subsequent_inference_ms"] = (
                metrics["subsequent_inference_time"] * 1000
            )

    if "comparison" in results:
        for key, value in results["comparison"].items():
            wandb_metrics[f"io/comparison/{key}"] = value

    return wandb_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequential I/O patterns during model inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name (baseline)",
    )
    parser.add_argument(
        "--optimized",
        type=str,
        default=None,
        help="Path to optimized model for comparison",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of measured iterations",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Input sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=10,
        help="Tokens to generate per iteration",
    )
    parser.add_argument(
        "--drop-caches",
        action="store_true",
        help="Try to drop caches before loading (requires root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="io_benchmark_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--wandb-format",
        action="store_true",
        help="Also output W&B-formatted metrics",
    )

    args = parser.parse_args()

    # Run benchmark
    results = compare_models(
        baseline_name=args.model,
        optimized_path=args.optimized,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        num_tokens_generate=args.generate,
        drop_caches_before=args.drop_caches,
    )

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Output W&B format
    if args.wandb_format:
        wandb_metrics = format_for_wandb(results)
        wandb_output = args.output.replace(".json", "_wandb.json")
        print(f"Saving W&B metrics to {wandb_output}")
        with open(wandb_output, "w") as f:
            json.dump(wandb_metrics, f, indent=2)

        print("\n--- W&B Metrics ---")
        for key, value in sorted(wandb_metrics.items()):
            print(
                f"  {key}: {value:.4f}"
                if isinstance(value, float)
                else f"  {key}: {value}"
            )


if __name__ == "__main__":
    main()
