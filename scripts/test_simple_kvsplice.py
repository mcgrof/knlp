#!/usr/bin/env python3
"""
Test simplified KVSplice plugin with two-phase calibration.

Phase 1: Reconstruction loss (learned PCA)
Phase 2: Task loss (optional, not implemented yet)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, "/data/knlp/scripts")
from kvsplice_simple_plugin import calibrate_kv_compressor, enable_kvsplice_simple


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
    print("Simplified KVSplice Plugin Test")
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
    print("TEST 1: Baseline (no compression)")
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

    # Test with learned compressors
    print("\n" + "=" * 80)
    print("TEST 2: With Learned KV Compressors (Phase 1)")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # Phase 1: Calibrate with reconstruction loss
    print("\nPhase 1: Training compressors with reconstruction loss...")
    compressors = calibrate_kv_compressor(
        model,
        tokenizer,
        compression_ratio=0.5,
        calibration_samples=1000,  # Small for quick test
        calibration_steps=300,
        learning_rate=1e-4,  # Lower LR for stability
        per_layer=False,  # Shared compressor first
    )

    # Enable compression
    enable_kvsplice_simple(model, compressors, per_layer=False)

    result_phase1 = test_generation(model, tokenizer)

    print(f"\nPhase 1 Generated: {result_phase1['text'][:100]}...")
    print(f"\nPhase 1 Memory stats:")
    print(f"  Before: {result_phase1['mem_before']:.2f} MB")
    print(f"  After:  {result_phase1['mem_after']:.2f} MB")
    print(f"  Cache:  {result_phase1['cache_size']:.2f} MB")

    # Phase 2: Refine with task loss
    print("\n" + "=" * 80)
    print("Phase 2: Task-Aware Fine-Tuning")
    print("=" * 80)

    from kvsplice_simple_plugin import refine_with_task_loss

    compressors = refine_with_task_loss(
        model,
        tokenizer,
        compressors,
        per_layer=False,
        learning_rate=1e-5,
        refinement_steps=200,  # Quick test
        gradient_accumulation=4,
    )

    # Test after Phase 2
    print("\nTesting generation after Phase 2 refinement...")
    result_phase2 = test_generation(model, tokenizer)

    print(f"\nPhase 2 Generated: {result_phase2['text'][:100]}...")
    print(f"\nPhase 2 Memory stats:")
    print(f"  Before: {result_phase2['mem_before']:.2f} MB")
    print(f"  After:  {result_phase2['mem_after']:.2f} MB")
    print(f"  Cache:  {result_phase2['cache_size']:.2f} MB")

    result_compressed = result_phase2  # Use Phase 2 result for final comparison

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    cache_reduction = (
        (result_baseline["cache_size"] - result_compressed["cache_size"])
        / result_baseline["cache_size"]
        * 100
    )

    print(f"\nCache size:")
    print(f"  Baseline:   {result_baseline['cache_size']:.2f} MB")
    print(f"  Compressed: {result_compressed['cache_size']:.2f} MB")
    print(f"  Reduction:  {cache_reduction:.1f}%")

    print(f"\nGeneration quality:")
    print(f"  Baseline: {result_baseline['text'][:80]}...")
    print(f"  Phase 1:  {result_phase1['text'][:80]}...")
    print(f"  Phase 2:  {result_phase2['text'][:80]}...")

    # Check if outputs are similar (first 20 tokens)
    baseline_tokens = tokenizer.encode(result_baseline["text"])[:20]
    phase1_tokens = tokenizer.encode(result_phase1["text"])[:20]
    phase2_tokens = tokenizer.encode(result_phase2["text"])[:20]

    # Count matching tokens
    phase1_matches = sum(a == b for a, b in zip(baseline_tokens, phase1_tokens))
    phase2_matches = sum(a == b for a, b in zip(baseline_tokens, phase2_tokens))

    phase1_pct = phase1_matches / len(baseline_tokens) * 100
    phase2_pct = phase2_matches / len(baseline_tokens) * 100

    print(f"\nToken-level agreement (first 20 tokens):")
    print(f"  Phase 1: {phase1_pct:.1f}%")
    print(f"  Phase 2: {phase2_pct:.1f}%")
    print(f"  Improvement: {phase2_pct - phase1_pct:+.1f}%")

    if cache_reduction > 30 and phase2_pct > 70:
        print(f"\n✓ SUCCESS: Good memory savings with reasonable quality!")
    elif cache_reduction > 30 and phase2_pct > phase1_pct:
        print(f"\n✓ PROGRESS: Memory savings + Phase 2 improved quality")
    elif cache_reduction > 30:
        print(f"\n⚠ PARTIAL: Memory savings achieved but quality still degraded")
    else:
        print(f"\n✗ FAILURE: Insufficient memory savings")


if __name__ == "__main__":
    main()
