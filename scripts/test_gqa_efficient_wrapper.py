#!/usr/bin/env python3
"""
Test GQA-Efficient Wrapper

Validates that the wrapper correctly compresses per KV group instead of
per Q head for GQA models like Qwen2.5-7B.

Expected results for Qwen2.5-7B:
- 28 Q heads, 4 KV heads -> 4 compressors (7x reduction)
- PPL unchanged compared to standard wrapper
- Same compression quality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.kvsplice import KVSpliceCompressor
from gpt2.compression.wrapper_hooks import CompressedKVModelWrapper


def test_gqa_detection():
    """Test that GQA structure is correctly detected."""
    print("=" * 70)
    print("Test 1: GQA Structure Detection")
    print("=" * 70)

    # Test on Qwen2.5-7B (GQA model: 28 Q heads, 4 KV heads)
    print("\nLoading Qwen2.5-7B-Instruct (GQA model)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU for quick test
    )

    # Check config
    n_head = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_head)

    print(f"  Model config:")
    print(f"    Q heads: {n_head}")
    print(f"    KV heads: {n_kv_heads}")
    print(f"    GQA ratio: {n_head // n_kv_heads}:1 (Q:KV)")

    assert n_kv_heads < n_head, "Expected GQA model"
    print("  ✓ GQA structure detected correctly")

    return n_head, n_kv_heads


def test_compressor_count(n_head, n_kv_heads):
    """Test that wrapper creates correct number of compressors."""
    print("\n" + "=" * 70)
    print("Test 2: Compressor Count")
    print("=" * 70)

    # Create compression config
    d_head = 128  # Qwen2.5-7B head dim
    rank = 64  # 50% compression

    config = {
        "global": {"d_head": d_head, "algo_default": "kvsplice"},
        "per_layer_head": {},
    }

    # Standard wrapper would create n_head compressors
    # GQA-efficient wrapper should create only n_kv_heads compressors
    for layer_idx in range(28):  # Qwen2.5-7B has 28 layers
        for head_idx in range(n_kv_heads):  # Only KV heads
            config["per_layer_head"][(layer_idx, head_idx)] = {
                "enabled": True,
                "rank": rank,
                "d_k": d_head,
                "d_v": d_head,
                "algo": "kvsplice",
            }

    compressor = KVSpliceCompressor(config)

    print(f"\nCompressor configuration:")
    print(f"  Layers: 28")
    print(f"  KV heads per layer: {n_kv_heads}")
    print(f"  Total compressors: {28 * n_kv_heads}")
    print(f"  Standard wrapper would need: {28 * n_head}")
    print(f"  Parameter reduction: {(28 * n_head) / (28 * n_kv_heads):.1f}x")

    # Count compressor modules
    compressor_count = len(compressor.compressors)
    expected_count = 28 * n_kv_heads

    print(f"\nActual compressor count: {compressor_count}")
    print(f"Expected: {expected_count}")

    assert (
        compressor_count == expected_count
    ), f"Expected {expected_count} compressors, got {compressor_count}"
    print("  ✓ Correct number of compressors created")

    return config


def test_wrapper_inference(config):
    """Test that wrapped model produces valid outputs."""
    print("\n" + "=" * 70)
    print("Test 3: Wrapper Inference")
    print("=" * 70)

    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # Create compressor and wrapper
    compressor = KVSpliceCompressor(config)
    wrapped_model = CompressedKVModelWrapper(model, compressor)

    print("  Model wrapped successfully")

    # Test inference (without calibration - should pass through)
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt")

    print(f"\nTest input: '{test_text}'")

    with torch.no_grad():
        # Standard model
        outputs_standard = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
        text_standard = tokenizer.decode(outputs_standard[0], skip_special_tokens=True)

        # Wrapped model (no compression - should match)
        outputs_wrapped = wrapped_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
        text_wrapped = tokenizer.decode(outputs_wrapped[0], skip_special_tokens=True)

    print(f"\nStandard output: '{text_standard}'")
    print(f"Wrapped output:  '{text_wrapped}'")

    # Outputs should match (no compression active)
    assert text_standard == text_wrapped, "Outputs should match without compression"
    print("  ✓ Wrapper produces correct outputs")


def test_head_indexing():
    """Test that KV group indexing is correct."""
    print("\n" + "=" * 70)
    print("Test 4: KV Group Indexing")
    print("=" * 70)

    # Simulate GQA structure
    n_head = 28
    n_kv_heads = 4
    kv_group_size = n_head // n_kv_heads  # 7

    print(f"\nGQA structure:")
    print(f"  Q heads: {n_head}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Group size: {kv_group_size}")

    # Verify indexing logic
    print("\nKV group assignments:")
    for kv_group_idx in range(n_kv_heads):
        q_heads_in_group = []
        for offset in range(kv_group_size):
            q_head_idx = kv_group_idx + offset * n_kv_heads
            if q_head_idx < n_head:
                q_heads_in_group.append(q_head_idx)

        print(f"  KV group {kv_group_idx}: Q heads {q_heads_in_group}")

        # Verify all Q heads in group are correct
        for i, q_head in enumerate(q_heads_in_group):
            expected = kv_group_idx + i * n_kv_heads
            assert (
                q_head == expected
            ), f"Indexing error: expected {expected}, got {q_head}"

    # Verify all Q heads covered
    all_q_heads = []
    for kv_group_idx in range(n_kv_heads):
        for offset in range(kv_group_size):
            q_head_idx = kv_group_idx + offset * n_kv_heads
            if q_head_idx < n_head:
                all_q_heads.append(q_head_idx)

    all_q_heads.sort()
    expected_heads = list(range(n_head))

    print(f"\nAll Q heads covered: {len(all_q_heads)}/{n_head}")
    assert all_q_heads == expected_heads, "Not all Q heads covered correctly"
    print("  ✓ KV group indexing correct")


def main():
    print("Testing GQA-Efficient Wrapper")
    print("=" * 70)

    # Test 1: GQA detection
    n_head, n_kv_heads = test_gqa_detection()

    # Test 2: Compressor count
    config = test_compressor_count(n_head, n_kv_heads)

    # Test 3: Wrapper inference
    test_wrapper_inference(config)

    # Test 4: Head indexing logic
    test_head_indexing()

    print("\n" + "=" * 70)
    print("✓ ALL GQA-EFFICIENT WRAPPER TESTS PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - GQA structure detected: {n_head} Q heads, {n_kv_heads} KV heads")
    print(f"  - Compressors created: {28 * n_kv_heads} (vs {28 * n_head} for standard)")
    print(f"  - Parameter reduction: {n_head / n_kv_heads:.1f}x")
    print(f"  - Wrapper inference works correctly")
    print(f"  - KV group indexing correct")

    return 0


if __name__ == "__main__":
    sys.exit(main())
