#!/usr/bin/env python3
"""
Benchmark KV compression with int8 quantization on B200.

Tests low-rank + int8 quantization for additional memory savings.

Usage:
    python scripts/benchmark_kv_quantized.py \
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    QuantizedCalibratedCompressor,
    CalibratedCompressor,
    IdentityCompressor,
)


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


def load_quantized_compressors(calib_path: str, num_layers: int, bits: int = 8, device: str = "cuda"):
    """Load calibration with int8/int4 quantization."""
    calib = torch.load(calib_path, map_location=device)

    k_compressors = [IdentityCompressor() for _ in range(num_layers)]
    v_compressors = []

    for layer_data in calib["layers"]:
        V_U = layer_data["V"]["U"].to(device).to(torch.float16)
        V_mean = layer_data["V"]["mean"].to(device).to(torch.float16)
        v_compressors.append(QuantizedCalibratedCompressor(V_U, V_mean, bits=bits))

    return k_compressors, v_compressors


def load_fp16_compressors(calib_path: str, num_layers: int, device: str = "cuda"):
    """Load calibration without quantization (FP16)."""
    calib = torch.load(calib_path, map_location=device)

    k_compressors = [IdentityCompressor() for _ in range(num_layers)]
    v_compressors = []

    for layer_data in calib["layers"]:
        V_U = layer_data["V"]["U"].to(device).to(torch.float16)
        V_mean = layer_data["V"]["mean"].to(device).to(torch.float16)
        v_compressors.append(CalibratedCompressor(V_U, V_mean))

    return k_compressors, v_compressors


def measure_perplexity(model, tokenizer, cache=None, num_samples=50, device="cuda"):
    """Measure perplexity on WikiText-2."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:num_samples]

    total_loss = 0
    total_tokens = 0

    num_layers = model.config.num_hidden_layers

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        # Reset cache if using compressed
        if cache is not None:
            cache.reset()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids, past_key_values=cache)
            loss = outputs.loss

        total_loss += loss.item() * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def measure_memory(model, tokenizer, k_compressors, v_compressors, seq_len=1024, device="cuda"):
    """Measure cache memory at specific sequence length."""
    num_layers = model.config.num_hidden_layers
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)

    cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, past_key_values=cache)

    stats = cache.get_memory_stats()
    return stats["total_mb"]


def measure_throughput(model, tokenizer, k_compressors, v_compressors,
                       prompt_len=512, gen_len=128, runs=3, device="cuda"):
    """Measure generation throughput."""
    num_layers = model.config.num_hidden_layers
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, prompt_len), device=device)

    times = []
    for _ in range(runs):
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=gen_len,
                past_key_values=cache,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return gen_len / avg_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized KV compression")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--rank", type=int, default=120)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Quantized KV Compression Benchmark")
    print(f"Model: {args.model}")
    print(f"Rank: {args.rank}")
    print(f"=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    num_layers = model.config.num_hidden_layers

    results = {
        "model": args.model,
        "rank": args.rank,
        "timestamp": datetime.now().isoformat(),
    }

    # Test configurations
    configs = [
        ("baseline", None, None),
        ("fp16_lowrank", "fp16", None),
        ("int8_lowrank", "int8", 8),
    ]

    print(f"\n--- Perplexity Benchmark ---")
    ppl_results = {}

    for name, mode, bits in configs:
        print(f"  Testing {name}...")

        if mode is None:
            # Baseline - no compression
            ppl = measure_perplexity(model, tokenizer, cache=None,
                                     num_samples=args.num_samples, device=args.device)
        elif mode == "fp16":
            k_comp, v_comp = load_fp16_compressors(args.calibration, num_layers, args.device)
            cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
            ppl = measure_perplexity(model, tokenizer, cache=cache,
                                     num_samples=args.num_samples, device=args.device)
        else:  # int8
            k_comp, v_comp = load_quantized_compressors(args.calibration, num_layers, bits, args.device)
            cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
            ppl = measure_perplexity(model, tokenizer, cache=cache,
                                     num_samples=args.num_samples, device=args.device)

        ppl_results[name] = ppl
        print(f"    {name}: PPL = {ppl:.4f}")

    # Calculate deltas
    baseline_ppl = ppl_results["baseline"]
    for name in ppl_results:
        if name != "baseline":
            delta = (ppl_results[name] - baseline_ppl) / baseline_ppl * 100
            print(f"    {name} delta: {delta:+.1f}%")

    results["perplexity"] = ppl_results

    print(f"\n--- Memory Benchmark (seq_len=1024) ---")
    memory_results = {}

    for name, mode, bits in configs[1:]:  # Skip baseline
        if mode == "fp16":
            k_comp, v_comp = load_fp16_compressors(args.calibration, num_layers, args.device)
        else:
            k_comp, v_comp = load_quantized_compressors(args.calibration, num_layers, bits, args.device)

        mem = measure_memory(model, tokenizer, k_comp, v_comp, seq_len=1024, device=args.device)
        memory_results[name] = mem
        print(f"    {name}: {mem:.2f} MB")

    results["memory"] = memory_results

    print(f"\n--- Throughput Benchmark ---")
    throughput_results = {}

    for name, mode, bits in configs:
        if mode is None:
            k_comp = [IdentityCompressor() for _ in range(num_layers)]
            v_comp = [IdentityCompressor() for _ in range(num_layers)]
        elif mode == "fp16":
            k_comp, v_comp = load_fp16_compressors(args.calibration, num_layers, args.device)
        else:
            k_comp, v_comp = load_quantized_compressors(args.calibration, num_layers, bits, args.device)

        tps = measure_throughput(model, tokenizer, k_comp, v_comp, device=args.device)
        throughput_results[name] = tps
        print(f"    {name}: {tps:.1f} tok/s")

    baseline_tps = throughput_results["baseline"]
    for name in throughput_results:
        if name != "baseline":
            delta = (throughput_results[name] - baseline_tps) / baseline_tps * 100
            print(f"    {name} overhead: {delta:+.1f}%")

    results["throughput"] = throughput_results

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Baseline PPL: {ppl_results['baseline']:.4f}")
    print(f"FP16 low-rank PPL: {ppl_results['fp16_lowrank']:.4f} ({(ppl_results['fp16_lowrank']/baseline_ppl - 1)*100:+.1f}%)")
    print(f"INT8 low-rank PPL: {ppl_results['int8_lowrank']:.4f} ({(ppl_results['int8_lowrank']/baseline_ppl - 1)*100:+.1f}%)")
    print(f"\nFP16 memory: {memory_results['fp16_lowrank']:.2f} MB")
    print(f"INT8 memory: {memory_results['int8_lowrank']:.2f} MB")
    print(f"\nFP16 throughput: {throughput_results['fp16_lowrank']:.1f} tok/s ({(throughput_results['fp16_lowrank']/baseline_tps - 1)*100:+.1f}%)")
    print(f"INT8 throughput: {throughput_results['int8_lowrank']:.1f} tok/s ({(throughput_results['int8_lowrank']/baseline_tps - 1)*100:+.1f}%)")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("key_results") / f"quantized_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
