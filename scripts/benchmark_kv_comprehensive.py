#!/usr/bin/env python3
"""
Comprehensive KV Compression Benchmark on B200.

Tests:
1. Quality: lm-eval tasks (hellaswag, arc_easy, winogrande, piqa)
2. Memory: Actual KV cache size at various sequence lengths
3. Throughput: Tokens per second for generation

Usage:
    python scripts/benchmark_kv_comprehensive.py \
        --model Qwen/Qwen2.5-7B \
        --rank 120 \
        --calibration key_results/kv_calib_qwen7b_r120.pt
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    CalibratedCompressor,
    IdentityCompressor,
)


def get_gpu_memory_mb():
    """Get current GPU memory used in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_calibration(calib_path: str, num_layers: int, device: str = "cuda"):
    """Load calibration and create V-only compressors."""
    calib = torch.load(calib_path, map_location=device)

    k_compressors = [IdentityCompressor() for _ in range(num_layers)]
    v_compressors = []

    for layer_data in calib["layers"]:
        V_U = layer_data["V"]["U"].to(device).to(torch.float16)
        V_mean = layer_data["V"]["mean"].to(device).to(torch.float16)
        v_compressors.append(CalibratedCompressor(V_U, V_mean))

    return k_compressors, v_compressors


def measure_cache_memory(
    model,
    tokenizer,
    seq_lengths: list,
    k_compressors=None,
    v_compressors=None,
    device: str = "cuda",
):
    """Measure actual KV cache memory at different sequence lengths."""
    results = []
    num_layers = model.config.num_hidden_layers

    for seq_len in seq_lengths:
        # Generate input of exact length
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (1, seq_len), device=device
        )

        # Baseline: standard cache
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            baseline_cache = outputs.past_key_values

        # Calculate baseline memory (sum of all K,V tensors)
        baseline_bytes = 0
        for layer_kv in baseline_cache:
            k, v = layer_kv
            baseline_bytes += k.numel() * k.element_size()
            baseline_bytes += v.numel() * v.element_size()
        baseline_mb = baseline_bytes / (1024 * 1024)

        del outputs, baseline_cache
        gc.collect()
        torch.cuda.empty_cache()

        # Compressed cache
        if k_compressors and v_compressors:
            compressed_cache = CompressedDynamicCache(
                k_compressors, v_compressors, num_layers
            )

            with torch.no_grad():
                outputs = model(input_ids, use_cache=True, past_key_values=compressed_cache)

            stats = compressed_cache.get_memory_stats()
            compressed_mb = stats["total_mb"]

            del outputs, compressed_cache
            gc.collect()
            torch.cuda.empty_cache()
        else:
            compressed_mb = baseline_mb

        savings_pct = (1 - compressed_mb / baseline_mb) * 100 if baseline_mb > 0 else 0

        results.append({
            "seq_len": seq_len,
            "baseline_mb": baseline_mb,
            "compressed_mb": compressed_mb,
            "savings_pct": savings_pct,
        })

        print(f"  seq_len={seq_len}: baseline={baseline_mb:.2f}MB, "
              f"compressed={compressed_mb:.2f}MB, savings={savings_pct:.1f}%")

    return results


def measure_throughput(
    model,
    tokenizer,
    k_compressors=None,
    v_compressors=None,
    prompt_len: int = 512,
    gen_len: int = 128,
    warmup_runs: int = 2,
    test_runs: int = 5,
    device: str = "cuda",
):
    """Measure generation throughput."""
    num_layers = model.config.num_hidden_layers

    # Create prompt
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, prompt_len), device=device)

    def run_generation(use_compression: bool):
        if use_compression and k_compressors and v_compressors:
            cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)
        else:
            cache = None

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=gen_len,
                past_key_values=cache,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return outputs.shape[1] - prompt_len

    results = {}

    for mode, use_comp in [("baseline", False), ("compressed", True)]:
        if mode == "compressed" and not k_compressors:
            continue

        # Warmup
        for _ in range(warmup_runs):
            run_generation(use_comp)

        torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(test_runs):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            start = time.perf_counter()
            tokens = run_generation(use_comp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        avg_time = sum(times) / len(times)
        tokens_per_sec = gen_len / avg_time

        results[mode] = {
            "tokens_generated": gen_len,
            "avg_time_sec": avg_time,
            "tokens_per_sec": tokens_per_sec,
        }

        print(f"  {mode}: {tokens_per_sec:.1f} tok/s (avg {avg_time:.3f}s)")

    if "baseline" in results and "compressed" in results:
        overhead = (
            (results["compressed"]["avg_time_sec"] / results["baseline"]["avg_time_sec"] - 1)
            * 100
        )
        results["overhead_pct"] = overhead
        print(f"  Compression overhead: {overhead:+.1f}%")

    return results


def run_lm_eval(
    model_name: str,
    k_compressors=None,
    v_compressors=None,
    num_layers: int = None,
    tasks: list = None,
    num_fewshot: int = 0,
    limit: int = 100,
    device: str = "cuda",
):
    """Run lm-eval benchmarks."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm-eval not installed, skipping quality benchmarks")
        return None

    if tasks is None:
        tasks = ["hellaswag", "arc_easy", "winogrande", "piqa"]

    results = {}

    for mode in ["baseline", "compressed"]:
        if mode == "compressed" and not k_compressors:
            continue

        print(f"\n  Running lm-eval ({mode})...")

        # Create model wrapper
        model = HFLM(
            pretrained=model_name,
            device=device,
            dtype="float16",
            trust_remote_code=True,
        )

        # If compressed mode, wrap the generate function
        if mode == "compressed" and k_compressors and v_compressors:
            original_generate = model._model.generate

            def compressed_generate(*args, **kwargs):
                cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)
                kwargs["past_key_values"] = cache
                return original_generate(*args, **kwargs)

            model._model.generate = compressed_generate

        # Run evaluation
        eval_results = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=1,
        )

        mode_results = {}
        for task in tasks:
            if task in eval_results["results"]:
                task_result = eval_results["results"][task]
                # Get main metric (usually acc or acc_norm)
                if "acc_norm,none" in task_result:
                    mode_results[task] = task_result["acc_norm,none"]
                elif "acc,none" in task_result:
                    mode_results[task] = task_result["acc,none"]

        results[mode] = mode_results

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        for task, acc in mode_results.items():
            print(f"    {task}: {acc*100:.1f}%")

    # Calculate deltas
    if "baseline" in results and "compressed" in results:
        results["delta"] = {}
        for task in results["baseline"]:
            if task in results["compressed"]:
                delta = (results["compressed"][task] - results["baseline"][task]) * 100
                results["delta"][task] = delta
                print(f"    {task} delta: {delta:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive KV compression benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--rank", type=int, default=120)
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--skip-lm-eval", action="store_true", help="Skip lm-eval quality tests")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory tests")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput tests")
    parser.add_argument("--lm-eval-limit", type=int, default=100, help="Samples per lm-eval task")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Comprehensive KV Compression Benchmark")
    print(f"Model: {args.model}")
    print(f"Rank: {args.rank}")
    print(f"Calibration: {args.calibration or 'None (baseline only)'}")
    print(f"=" * 60)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"GPU: {gpu_name} x{gpu_count}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers")

    # Load calibration if provided
    k_compressors, v_compressors = None, None
    if args.calibration and Path(args.calibration).exists():
        print(f"\nLoading calibration from {args.calibration}...")
        k_compressors, v_compressors = load_calibration(
            args.calibration, num_layers, args.device
        )
        print(f"  V-only compression with rank={args.rank}")

    results = {
        "model": args.model,
        "rank": args.rank,
        "calibration": args.calibration,
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    # Memory benchmark
    if not args.skip_memory:
        print(f"\n--- Memory Benchmark ---")
        seq_lengths = [256, 512, 1024, 2048]
        memory_results = measure_cache_memory(
            model, tokenizer, seq_lengths,
            k_compressors, v_compressors, args.device
        )
        results["memory"] = memory_results

    # Throughput benchmark
    if not args.skip_throughput:
        print(f"\n--- Throughput Benchmark ---")
        throughput_results = measure_throughput(
            model, tokenizer,
            k_compressors, v_compressors,
            prompt_len=512, gen_len=128,
            device=args.device
        )
        results["throughput"] = throughput_results

    # Cleanup model before lm-eval (uses its own model loading)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Quality benchmark
    if not args.skip_lm_eval:
        print(f"\n--- Quality Benchmark (lm-eval) ---")
        quality_results = run_lm_eval(
            args.model,
            k_compressors, v_compressors, num_layers,
            tasks=["hellaswag", "arc_easy", "winogrande", "piqa"],
            limit=args.lm_eval_limit,
            device=args.device,
        )
        if quality_results:
            results["quality"] = quality_results

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if "memory" in results:
        avg_savings = sum(r["savings_pct"] for r in results["memory"]) / len(results["memory"])
        print(f"Memory savings: {avg_savings:.1f}% average across sequence lengths")

    if "throughput" in results and "overhead_pct" in results["throughput"]:
        print(f"Throughput overhead: {results['throughput']['overhead_pct']:+.1f}%")

    if "quality" in results and "delta" in results["quality"]:
        avg_delta = sum(results["quality"]["delta"].values()) / len(results["quality"]["delta"])
        print(f"Quality impact: {avg_delta:+.1f}% average across tasks")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("key_results") / f"comprehensive_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
