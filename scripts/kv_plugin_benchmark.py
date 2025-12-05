#!/usr/bin/env python3
"""
KV Plugin v3 Benchmark Script

Benchmarks memory, speed, and quality for various compression presets.

Usage:
    python scripts/kv_plugin_benchmark.py --model gpt2 --preset balanced
    python scripts/kv_plugin_benchmark.py --model gpt2 --all-presets
    python scripts/kv_plugin_benchmark.py --model Qwen/Qwen2.5-0.5B --preset aggressive
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.kv_plugin import KVPlugin, KVPluginConfig


def measure_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 1024,
) -> float:
    """Compute perplexity on text."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss

    return torch.exp(neg_log_likelihood).item()


def measure_memory(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context_length: int = 1024,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure peak memory during forward pass."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create dummy input
    input_ids = torch.randint(0, 1000, (1, context_length), device=device)

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)

    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    return {
        "peak_memory_mb": peak_memory,
        "context_length": context_length,
    }


def measure_kv_cache_size(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context_length: int = 1024,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure KV cache tensor sizes directly."""
    # Create input
    input_ids = torch.randint(0, 1000, (1, context_length), device=device)

    # Forward pass with cache
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    # Get past_key_values
    past_kv = outputs.past_key_values
    if past_kv is None:
        return {"kv_cache_mb": 0, "context_length": context_length}

    # Sum up all cache tensor sizes
    total_bytes = 0
    for layer_kv in past_kv:
        for tensor in layer_kv:
            if tensor is not None:
                total_bytes += tensor.numel() * tensor.element_size()

    kv_cache_mb = total_bytes / (1024**2)

    return {
        "kv_cache_mb": kv_cache_mb,
        "context_length": context_length,
        "num_layers": len(past_kv),
    }


def measure_throughput(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    num_trials: int = 5,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure generation throughput."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    # Measure
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_generated = max_new_tokens

    return {
        "avg_time_s": avg_time,
        "tokens_per_second": tokens_generated / avg_time,
        "num_trials": num_trials,
    }


def measure_ttft(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_length: int = 512,
    num_trials: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure time-to-first-token (prefill time)."""
    input_ids = torch.randint(0, 1000, (1, prompt_length), device=device)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids)

    # Measure
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input_ids, use_cache=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "avg_ttft_ms": (sum(times) / len(times)) * 1000,
        "std_ttft_ms": (
            (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
        )
        * 1000,
        "prompt_length": prompt_length,
        "num_trials": num_trials,
    }


def benchmark_preset(
    model_name: str,
    preset: str,
    calibration_text: str,
    eval_text: str,
    device: str = "cuda",
    skip_calibration: bool = False,
    quant_bits: Optional[int] = None,
    quant_target: str = "v",
    quant_per_channel: bool = True,
    quant_storage: bool = False,
) -> Dict[str, Any]:
    """Benchmark a single preset configuration."""
    # Check if this is an orthogonal preset
    is_orthogonal = preset.startswith("orthogonal")

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} with preset '{preset}'")
    if is_orthogonal and skip_calibration:
        print("  (zero-calibration mode)")
    if quant_bits is not None:
        mode = "int8 storage" if quant_storage else "fake"
        print(f"  (with {quant_bits}-bit quantization on {quant_target}, {mode})")
    print(f"{'='*60}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    results = {
        "model": model_name,
        "preset": preset,
        "preset_description": KVPlugin.PRESETS.get(preset, {}).get("description", ""),
        "zero_calibration": is_orthogonal and skip_calibration,
        "quant_bits": quant_bits,
        "quant_target": quant_target if quant_bits else None,
    }

    plugin = None
    if preset != "none":
        # Create plugin with optional quantization override
        quant_overrides = {}
        if quant_bits is not None:
            quant_overrides = {
                "quant_bits": quant_bits,
                "quant_target": quant_target,
                "quant_per_channel": quant_per_channel,
                "quant_storage": quant_storage,
            }
        print(f"Creating plugin with preset '{preset}'...")
        plugin = KVPlugin.from_preset(preset, model, device=device, **quant_overrides)

        # Calibrate (skip for orthogonal if requested)
        if skip_calibration and is_orthogonal:
            print("  Skipping calibration (zero-calibration test)")
            results["calibration"] = "skipped"
        else:
            print("Calibrating compressors...")
            cal_tokens = tokenizer(
                calibration_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).input_ids.to(device)

            try:
                plugin.calibrate([cal_tokens])
                results["calibration"] = "success"
            except Exception as e:
                results["calibration"] = f"failed: {e}"
                print(f"  Warning: Calibration failed: {e}")

        # Patch model to use compressed attention
        print("Patching model with compression...")
        try:
            plugin.patch_model()
            results["patched"] = True
        except Exception as e:
            results["patched"] = f"failed: {e}"
            print(f"  Warning: Patching failed: {e}")

        # Get compression stats
        results["compression"] = {
            "d_compressed": plugin.config.d_compressed,
            "d_model": plugin.config.d_model,
            "n_layers": plugin.config.n_layers,
            "ratio": plugin.config.d_model / plugin.config.d_compressed,
        }

    # Measure perplexity
    print("Measuring perplexity...")
    try:
        ppl = measure_perplexity(model, tokenizer, eval_text, device)
        results["perplexity"] = ppl
        print(f"  PPL: {ppl:.2f}")
    except Exception as e:
        results["perplexity"] = f"error: {e}"
        print(f"  PPL error: {e}")

    # Measure memory
    print("Measuring memory...")
    try:
        mem = measure_memory(model, tokenizer, context_length=1024, device=device)
        results["memory"] = mem
        print(f"  Peak memory: {mem['peak_memory_mb']:.1f} MB")
    except Exception as e:
        results["memory"] = f"error: {e}"
        print(f"  Memory error: {e}")

    # Measure KV cache size
    print("Measuring KV cache size...")
    try:
        # For baseline (no plugin), measure HF past_key_values
        kv = measure_kv_cache_size(model, tokenizer, context_length=1024, device=device)

        # For plugin presets, compute theoretical compressed cache size
        if plugin is not None:
            # Compute theoretical compression based on config
            # d_model is full head dimension, d_compressed is target
            d_model = plugin.config.d_model
            d_compressed = plugin.config.d_compressed
            compression_ratio = d_model / d_compressed if d_compressed > 0 else 1.0
            compressed_cache_mb = kv["kv_cache_mb"] / compression_ratio
            kv["compressed_cache_mb"] = compressed_cache_mb
            kv["compression_ratio"] = compression_ratio
            print(f"  KV cache (full): {kv['kv_cache_mb']:.2f} MB")
            print(
                f"  KV cache (compressed): {kv['compressed_cache_mb']:.2f} MB ({compression_ratio:.1f}x savings)"
            )
        else:
            print(f"  KV cache: {kv['kv_cache_mb']:.2f} MB")

        results["kv_cache"] = kv
    except Exception as e:
        results["kv_cache"] = f"error: {e}"
        print(f"  KV cache error: {e}")

    # Measure TTFT
    print("Measuring TTFT...")
    try:
        ttft = measure_ttft(model, tokenizer, prompt_length=512, device=device)
        results["ttft"] = ttft
        print(f"  TTFT: {ttft['avg_ttft_ms']:.2f} ms")
    except Exception as e:
        results["ttft"] = f"error: {e}"
        print(f"  TTFT error: {e}")

    # Measure throughput
    print("Measuring throughput...")
    try:
        tp = measure_throughput(model, tokenizer, device=device)
        results["throughput"] = tp
        print(f"  Throughput: {tp['tokens_per_second']:.1f} tok/s")
    except Exception as e:
        results["throughput"] = f"error: {e}"
        print(f"  Throughput error: {e}")

    # Cleanup
    del model
    if plugin is not None:
        del plugin
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KV Plugin v3 compression presets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=list(KVPlugin.PRESETS.keys()),
        help="Compression preset to benchmark",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Benchmark all presets",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="File with calibration text (default: use sample text)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration (for testing zero-calibration orthogonal presets)",
    )
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="Use WikiText-2 for validation PPL (more accurate, requires download)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="kv-plugin-v3-benchmark",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="Apply fake quantization to latent (4 or 8 bit)",
    )
    parser.add_argument(
        "--quant-target",
        type=str,
        default="v",
        choices=["v", "kv"],
        help="Quantization target: 'v' (V only) or 'kv' (both K and V)",
    )
    parser.add_argument(
        "--quant-per-channel",
        action="store_true",
        default=True,
        help="Use per-channel quantization (default: True)",
    )
    parser.add_argument(
        "--quant-global",
        action="store_true",
        help="Use global quantization scaling (overrides --quant-per-channel)",
    )
    parser.add_argument(
        "--quant-storage",
        action="store_true",
        help="Use real int8 storage instead of fake quantization (actual memory savings)",
    )
    args = parser.parse_args()

    # Handle quant-global flag
    if args.quant_global:
        args.quant_per_channel = False

    # Initialize W&B if requested
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            run_name = f"kv_v3_{args.model.split('/')[-1]}"
            if args.quant_bits:
                run_name += f"_q{args.quant_bits}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    "model": args.model,
                    "presets": (
                        list(KVPlugin.PRESETS.keys())
                        if args.all_presets
                        else [args.preset]
                    ),
                    "device": args.device,
                    "skip_calibration": args.skip_calibration,
                    "wikitext": args.wikitext,
                    "quant_bits": args.quant_bits,
                    "quant_target": args.quant_target if args.quant_bits else None,
                    "quant_per_channel": (
                        args.quant_per_channel if args.quant_bits else None
                    ),
                },
                name=run_name,
            )
            print(f"W&B initialized: {wandb.run.url}")
        except ImportError:
            print("Warning: wandb not installed, logging disabled")
        except Exception as e:
            print(f"Warning: Failed to init W&B: {e}")

    # Sample texts for calibration and evaluation
    calibration_text = (
        """
    The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input.
    Key-value caching enables efficient autoregressive generation by storing
    intermediate states. Compression techniques can reduce memory requirements
    while maintaining quality. Principal component analysis identifies the most
    important directions in the activation space. Energy-based gating selects
    channels based on their variance or information content.
    """
        * 10
    )

    eval_text = (
        """
    Machine learning models have become increasingly powerful over the past decade.
    Large language models can generate coherent text, answer questions, and assist
    with various tasks. The computational requirements of these models continue to
    grow, driving research into efficiency improvements. Memory bandwidth is often
    the bottleneck for inference, making cache compression particularly valuable.
    """
        * 5
    )

    if args.calibration_file:
        calibration_text = Path(args.calibration_file).read_text()

    # Determine presets to benchmark
    presets = list(KVPlugin.PRESETS.keys()) if args.all_presets else [args.preset]

    # Load WikiText-2 if requested
    if args.wikitext:
        print("Loading WikiText-2 for evaluation...")
        try:
            from datasets import load_dataset

            wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            eval_text = "\n\n".join(wikitext["text"][:100])  # First 100 paragraphs
            print(f"  Loaded {len(eval_text)} chars from WikiText-2 test set")
        except Exception as e:
            print(f"  Warning: Could not load WikiText-2: {e}")
            print("  Using default eval text")

    # Run benchmarks
    all_results = []
    for idx, preset in enumerate(presets):
        results = benchmark_preset(
            args.model,
            preset,
            calibration_text,
            eval_text,
            args.device,
            skip_calibration=args.skip_calibration,
            quant_bits=args.quant_bits,
            quant_target=args.quant_target,
            quant_per_channel=args.quant_per_channel,
            quant_storage=args.quant_storage,
        )
        all_results.append(results)

        # Log to W&B
        if wandb_run is not None:
            try:
                import wandb

                log_data = {"preset": preset, "step": idx}
                if results.get("quant_bits"):
                    log_data["quant_bits"] = results["quant_bits"]
                    log_data["quant_target"] = results.get("quant_target", "v")
                if isinstance(results.get("perplexity"), float):
                    log_data["perplexity"] = results["perplexity"]
                if isinstance(results.get("memory"), dict):
                    log_data["peak_memory_mb"] = results["memory"].get(
                        "peak_memory_mb", 0
                    )
                if isinstance(results.get("kv_cache"), dict):
                    log_data["kv_cache_mb"] = results["kv_cache"].get("kv_cache_mb", 0)
                    log_data["kv_compressed_mb"] = results["kv_cache"].get(
                        "compressed_cache_mb", 0
                    )
                    log_data["compression_ratio"] = results["kv_cache"].get(
                        "compression_ratio", 1.0
                    )
                if isinstance(results.get("ttft"), dict):
                    log_data["ttft_ms"] = results["ttft"].get("avg_ttft_ms", 0)
                if isinstance(results.get("throughput"), dict):
                    log_data["throughput_tok_s"] = results["throughput"].get(
                        "tokens_per_second", 0
                    )
                wandb.log(log_data, step=idx)
            except Exception as e:
                print(f"  Warning: Failed to log to W&B: {e}")

    # Print summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(
        f"{'Preset':<20} {'Calib':>6} {'PPL':>8} {'KV(full)':>9} {'KV(comp)':>9} {'Ratio':>6} {'TTFT':>8} {'Tok/s':>7}"
    )
    print("-" * 95)

    # Get baseline PPL and KV cache for delta calculation
    baseline_ppl = None
    baseline_kv = None
    for r in all_results:
        if r.get("preset") == "none":
            if isinstance(r.get("perplexity"), float):
                baseline_ppl = r["perplexity"]
            kv = r.get("kv_cache", {})
            if isinstance(kv, dict):
                baseline_kv = kv.get("kv_cache_mb")
            break

    for r in all_results:
        ppl = r.get("perplexity", "N/A")
        ppl_delta = ""
        if isinstance(ppl, float):
            if baseline_ppl and r.get("preset") != "none":
                delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
                ppl_delta = f" (+{delta:.1f}%)" if delta > 0 else f" ({delta:.1f}%)"
            ppl = f"{ppl:.2f}"

        kv = r.get("kv_cache", {})
        if isinstance(kv, dict):
            kv_full = kv.get("kv_cache_mb", 0)
            kv_full_str = f"{kv_full:.2f}"
            # Use compressed cache if available, else same as full
            kv_comp = kv.get("compressed_cache_mb", kv_full)
            kv_comp_str = f"{kv_comp:.2f}" if kv_comp > 0 else "-"
            ratio = kv.get("compression_ratio", 1.0)
            ratio_str = f"{ratio:.1f}x" if ratio > 1.0 else "-"
        else:
            kv_full_str = "N/A"
            kv_comp_str = "N/A"
            ratio_str = "-"

        ttft = r.get("ttft", {})
        if isinstance(ttft, dict):
            ttft = f"{ttft.get('avg_ttft_ms', 'N/A'):.1f}"
        else:
            ttft = "N/A"

        tp = r.get("throughput", {})
        if isinstance(tp, dict):
            tp = f"{tp.get('tokens_per_second', 'N/A'):.0f}"
        else:
            tp = "N/A"

        # Calibration status
        calib = "no" if r.get("zero_calibration") else "yes"
        if r.get("preset") == "none":
            calib = "-"

        preset_name = r["preset"]
        if len(preset_name) > 18:
            preset_name = preset_name[:15] + "..."

        print(
            f"{preset_name:<20} {calib:>6} {ppl:>8} {kv_full_str:>9} {kv_comp_str:>9} {ratio_str:>6} {ttft:>8} {tp:>7}"
        )

    # Print PPL delta summary if baseline available
    if baseline_ppl:
        print(f"\nBaseline PPL: {baseline_ppl:.2f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    # Log summary and finish W&B
    if wandb_run is not None:
        try:
            import wandb

            # Calculate summary metrics
            best_throughput = 0
            best_ratio = 1.0
            for r in all_results:
                if isinstance(r.get("throughput"), dict):
                    tp = r["throughput"].get("tokens_per_second", 0)
                    if tp > best_throughput:
                        best_throughput = tp
                if isinstance(r.get("kv_cache"), dict):
                    ratio = r["kv_cache"].get("compression_ratio", 1.0)
                    if ratio > best_ratio:
                        best_ratio = ratio

            wandb.run.summary["best_throughput_tok_s"] = best_throughput
            wandb.run.summary["best_compression_ratio"] = best_ratio
            wandb.run.summary["num_presets_tested"] = len(all_results)
            if baseline_ppl:
                wandb.run.summary["baseline_perplexity"] = baseline_ppl
            wandb.finish()
            print("W&B run finished")
        except Exception as e:
            print(f"Warning: Failed to finish W&B: {e}")


if __name__ == "__main__":
    main()
