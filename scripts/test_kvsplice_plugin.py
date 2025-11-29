#!/usr/bin/env python3
"""
Quick test of KVSplice plugin to verify memory savings.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, "/data/knlp/scripts")
from kvsplice_plugin import enable_kvsplice


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_generation(model, tokenizer, prompt="Hello, my name is", max_new_tokens=50):
    """Test generation and measure memory."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.reset_peak_memory_stats()
    mem_before = get_memory_mb()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    mem_after = get_memory_mb()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "text": generated_text,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "mem_peak": mem_peak,
        "cache_size": mem_after - mem_before,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("KVSplice Plugin Test")
    print("=" * 80)

    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    print(f"Model loaded. Memory: {get_memory_mb():.2f} MB")

    # Test baseline
    print("\n" + "=" * 80)
    print("TEST 1: Baseline (no KVSplice)")
    print("=" * 80)

    result_baseline = test_generation(model, tokenizer)

    print(f"\nGenerated: {result_baseline['text'][:100]}...")
    print(f"\nMemory stats:")
    print(f"  Before: {result_baseline['mem_before']:.2f} MB")
    print(f"  After:  {result_baseline['mem_after']:.2f} MB")
    print(f"  Peak:   {result_baseline['mem_peak']:.2f} MB")
    print(f"  Cache:  {result_baseline['cache_size']:.2f} MB")

    # Clear cache
    del model
    torch.cuda.empty_cache()

    # Test with KVSplice
    print("\n" + "=" * 80)
    print("TEST 2: With Calibrated KVSplice (50% compression)")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float16
    ).to(device)
    tokenizer2 = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # Calibrate KVSplice on actual K/V data
    from kvsplice_plugin import calibrate_kvsplice

    calibrated = calibrate_kvsplice(
        model, tokenizer2, compression_ratio=0.5, calibration_samples=500
    )

    # Enable KVSplice with calibrated weights
    enable_kvsplice(model, compression_ratio=0.5, calibrated_kvsplice=calibrated)

    result_kvsplice = test_generation(model, tokenizer)

    print(f"\nGenerated: {result_kvsplice['text'][:100]}...")
    print(f"\nMemory stats:")
    print(f"  Before: {result_kvsplice['mem_before']:.2f} MB")
    print(f"  After:  {result_kvsplice['mem_after']:.2f} MB")
    print(f"  Peak:   {result_kvsplice['mem_peak']:.2f} MB")
    print(f"  Cache:  {result_kvsplice['cache_size']:.2f} MB")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    cache_reduction = (
        (result_baseline["cache_size"] - result_kvsplice["cache_size"])
        / result_baseline["cache_size"]
        * 100
    )

    print(f"\nCache size:")
    print(f"  Baseline:  {result_baseline['cache_size']:.2f} MB")
    print(f"  KVSplice:  {result_kvsplice['cache_size']:.2f} MB")
    print(f"  Reduction: {cache_reduction:.1f}%")

    if cache_reduction > 30:
        print(f"\n✓ SUCCESS: Significant memory savings!")
    elif cache_reduction > 0:
        print(f"\n⚠ PARTIAL: Some savings but less than expected")
    else:
        print(f"\n✗ FAILURE: No memory savings (double-caching bug still present)")


if __name__ == "__main__":
    main()
