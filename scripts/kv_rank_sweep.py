#!/usr/bin/env python3
"""
KV Plugin Rank Sweep - Find the compression failure boundary.

Sweeps d_compressed from high to low to find where PPL begins to degrade.

Usage:
    python scripts/kv_rank_sweep.py --model Qwen/Qwen2.5-0.5B
    python scripts/kv_rank_sweep.py --model Qwen/Qwen2.5-0.5B --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss

    return torch.exp(neg_log_likelihood).item()


def measure_throughput(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    num_trials: int = 3,
    device: str = "cuda",
) -> float:
    """Measure generation throughput."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return max_new_tokens / (sum(times) / len(times))


def run_single_config(
    model_name: str,
    d_compressed: int,
    quant_bits: Optional[int],
    quant_target: str,
    eval_text: str,
    cal_text: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run a single configuration and return metrics."""
    print(
        f"  rank={d_compressed}, bits={quant_bits or 16}, target={quant_target}...",
        end=" ",
        flush=True,
    )

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Create plugin with custom rank
    overrides = {"d_compressed": d_compressed}
    if quant_bits is not None:
        overrides["quant_bits"] = quant_bits
        overrides["quant_target"] = quant_target
        overrides["quant_storage"] = True

    plugin = KVPlugin.from_preset("balanced", model, **overrides)

    # Calibrate
    cal_tokens = tokenizer(
        cal_text, return_tensors="pt", truncation=True, max_length=2048
    ).input_ids.to(device)
    try:
        plugin.calibrate([cal_tokens])
    except Exception as e:
        print(f"calibration failed: {e}")
        del model, plugin
        torch.cuda.empty_cache()
        return {"error": str(e)}

    # Patch
    try:
        plugin.patch_model()
    except Exception as e:
        print(f"patching failed: {e}")
        del model, plugin
        torch.cuda.empty_cache()
        return {"error": str(e)}

    # Measure PPL
    try:
        ppl = measure_perplexity(model, tokenizer, eval_text, device)
    except Exception as e:
        ppl = float("nan")
        print(f"PPL error: {e}")

    # Measure throughput
    try:
        toks = measure_throughput(model, tokenizer, device=device)
    except Exception as e:
        toks = float("nan")

    # Compute compression ratio
    d_model = plugin.config.d_model
    ratio = d_model / d_compressed

    print(f"PPL={ppl:.3f}, ratio={ratio:.1f}x, tok/s={toks:.1f}")

    result = {
        "d_compressed": d_compressed,
        "quant_bits": quant_bits or 16,
        "quant_target": quant_target,
        "ppl": ppl,
        "compression_ratio": ratio,
        "tokens_per_sec": toks,
        "d_model": d_model,
    }

    del model, plugin
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Rank sweep to find failure boundary")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to test",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="128,96,64,48,32,24,16,12,8",
        help="Comma-separated list of ranks to test",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default="16,8,4",
        help="Comma-separated list of bit widths (16=fp16)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="v,kv",
        help="Comma-separated quantization targets",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer ranks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    bits_list = [int(b) for b in args.bits.split(",")]
    targets = args.targets.split(",")

    if args.quick:
        ranks = [64, 32, 16, 8]
        bits_list = [8, 4]
        targets = ["v"]

    # Sample texts
    cal_text = (
        """
    The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input.
    Key-value caching enables efficient autoregressive generation.
    """
        * 10
    )

    eval_text = (
        """
    Machine learning models have become increasingly powerful over the past decade.
    Large language models can generate coherent text, answer questions, and assist
    with various tasks. The computational requirements continue to grow.
    """
        * 5
    )

    print(f"Model: {args.model}")
    print(f"Ranks: {ranks}")
    print(f"Bits: {bits_list}")
    print(f"Targets: {targets}")
    print("=" * 60)

    # First, get baseline
    print("\nBaseline (no compression):")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    baseline_ppl = measure_perplexity(model, tokenizer, eval_text, args.device)
    baseline_toks = measure_throughput(model, tokenizer, device=args.device)
    print(f"  Baseline PPL: {baseline_ppl:.3f}, tok/s: {baseline_toks:.1f}")
    del model
    torch.cuda.empty_cache()

    results = []
    results.append(
        {
            "d_compressed": None,
            "quant_bits": 16,
            "quant_target": "none",
            "ppl": baseline_ppl,
            "compression_ratio": 1.0,
            "tokens_per_sec": baseline_toks,
            "is_baseline": True,
        }
    )

    # Run sweep
    for rank in ranks:
        print(f"\n--- Rank {rank} ---")
        for bits in bits_list:
            for target in targets:
                # Skip fp16 with quantization target
                if bits == 16 and target != "v":
                    continue

                qbits = None if bits == 16 else bits
                result = run_single_config(
                    args.model,
                    rank,
                    qbits,
                    target,
                    eval_text,
                    cal_text,
                    args.device,
                )
                if "error" not in result:
                    results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Rank':<8} {'Bits':<6} {'Target':<8} {'PPL':<10} {'Delta':<10} {'Ratio':<8} {'Tok/s':<8}"
    )
    print("-" * 80)

    for r in results:
        rank = r.get("d_compressed", "-")
        bits = r["quant_bits"]
        target = r["quant_target"]
        ppl = r["ppl"]
        ratio = r["compression_ratio"]
        toks = r["tokens_per_sec"]

        if r.get("is_baseline"):
            delta = "baseline"
        else:
            delta = f"+{((ppl - baseline_ppl) / baseline_ppl) * 100:.2f}%"

        print(
            f"{str(rank):<8} {bits:<6} {target:<8} {ppl:<10.3f} {delta:<10} {ratio:<8.1f}x {toks:<8.1f}"
        )

    # Find failure point
    print("\n" + "=" * 80)
    print("FAILURE BOUNDARY ANALYSIS")
    print("=" * 80)

    threshold_1pct = baseline_ppl * 1.01
    threshold_5pct = baseline_ppl * 1.05
    threshold_10pct = baseline_ppl * 1.10

    for threshold, name in [
        (threshold_1pct, "1% degradation"),
        (threshold_5pct, "5% degradation"),
        (threshold_10pct, "10% degradation"),
    ]:
        failed = [
            r for r in results if r["ppl"] > threshold and not r.get("is_baseline")
        ]
        if failed:
            worst = max(failed, key=lambda x: x["compression_ratio"])
            print(
                f"{name}: first failure at rank={worst['d_compressed']}, "
                f"ratio={worst['compression_ratio']:.1f}x, "
                f"bits={worst['quant_bits']}, target={worst['quant_target']}"
            )
        else:
            print(f"{name}: NO FAILURE FOUND (all configs within threshold)")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "baseline_ppl": baseline_ppl,
                    "baseline_toks": baseline_toks,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
