"""
Integration tests for KV Plugin with HuggingFace.

These tests verify that the compressed cache is actually on the hot path
during inference, not bypassed by HF's DynamicCache.

Tests:
1. Destroy-cache test: If we corrupt the cache, outputs should be garbage
2. Identity-compressor test: Identity compression should match baseline exactly
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    ZeroCompressor,
    RandomCompressor,
)


# Use small model for fast tests
TEST_MODEL = "Qwen/Qwen2.5-0.5B"
TEST_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all tests."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return model, tokenizer


def get_baseline_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    """Get logits without any cache modification."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
    return outputs.logits


def get_logits_with_cache(
    model, tokenizer, prompt: str, cache: CompressedDynamicCache
) -> torch.Tensor:
    """Get logits using custom cache."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
    return outputs.logits


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between flattened tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute max absolute difference."""
    return (a.float() - b.float()).abs().max().item()


class TestDestroyCache:
    """
    Test that corrupting the cache causes output to change dramatically.

    If this test fails (outputs similar with corrupted cache), it means
    the cache is being bypassed.
    """

    def test_zero_cache_changes_output(self, model_and_tokenizer):
        """
        With zero compressor, outputs should be very different from baseline.

        If cos_sim > 0.9, the cache is being bypassed.
        """
        model, tokenizer = model_and_tokenizer
        num_layers = model.config.num_hidden_layers

        # Baseline without cache
        baseline_logits = get_baseline_logits(model, tokenizer, TEST_PROMPT)

        # Create zero-compressor cache
        k_compressors = [ZeroCompressor() for _ in range(num_layers)]
        v_compressors = [ZeroCompressor() for _ in range(num_layers)]
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        # Get logits with corrupted cache
        corrupted_logits = get_logits_with_cache(model, tokenizer, TEST_PROMPT, cache)

        # Compute similarity
        cos_sim = cosine_similarity_flat(baseline_logits, corrupted_logits)

        print(f"\nZero-cache test:")
        print(f"  Baseline logits shape: {baseline_logits.shape}")
        print(f"  Corrupted logits shape: {corrupted_logits.shape}")
        print(f"  Cosine similarity: {cos_sim:.4f}")

        # If cache is on the path, corrupted should be very different
        # Allow some tolerance since first token might be similar
        assert cos_sim < 0.9, (
            f"Cache bypass detected! Cosine similarity {cos_sim:.4f} > 0.9 "
            "means the cache is not affecting inference."
        )

    def test_random_cache_changes_output(self, model_and_tokenizer):
        """
        With random compressor, outputs should be garbage.
        """
        model, tokenizer = model_and_tokenizer
        num_layers = model.config.num_hidden_layers

        baseline_logits = get_baseline_logits(model, tokenizer, TEST_PROMPT)

        k_compressors = [RandomCompressor() for _ in range(num_layers)]
        v_compressors = [RandomCompressor() for _ in range(num_layers)]
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        corrupted_logits = get_logits_with_cache(model, tokenizer, TEST_PROMPT, cache)

        cos_sim = cosine_similarity_flat(baseline_logits, corrupted_logits)

        print(f"\nRandom-cache test:")
        print(f"  Cosine similarity: {cos_sim:.4f}")

        assert (
            cos_sim < 0.9
        ), f"Cache bypass detected! Cosine similarity {cos_sim:.4f} > 0.9"


class TestIdentityCompressor:
    """
    Test that identity compressor produces same output as baseline.

    If this test fails, the cache implementation has bugs.
    """

    def test_identity_matches_baseline(self, model_and_tokenizer):
        """
        Identity compressor should produce near-identical logits.
        """
        model, tokenizer = model_and_tokenizer
        num_layers = model.config.num_hidden_layers

        baseline_logits = get_baseline_logits(model, tokenizer, TEST_PROMPT)

        k_compressors = [IdentityCompressor() for _ in range(num_layers)]
        v_compressors = [IdentityCompressor() for _ in range(num_layers)]
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        identity_logits = get_logits_with_cache(model, tokenizer, TEST_PROMPT, cache)

        cos_sim = cosine_similarity_flat(baseline_logits, identity_logits)
        max_diff = max_abs_diff(baseline_logits, identity_logits)

        print(f"\nIdentity-compressor test:")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max abs difference: {max_diff:.6f}")

        # Should be nearly identical (allowing for fp16 noise)
        assert cos_sim > 0.999, (
            f"Identity compressor should match baseline! "
            f"Cosine similarity {cos_sim:.4f} < 0.999"
        )
        assert max_diff < 0.1, f"Identity compressor max diff {max_diff:.4f} > 0.1"


class TestCacheOnHotPath:
    """
    Combined test to verify cache is definitely on the hot path.
    """

    def test_cache_affects_generation(self, model_and_tokenizer):
        """
        Test that the cache affects multi-token generation.
        """
        model, tokenizer = model_and_tokenizer
        num_layers = model.config.num_hidden_layers
        input_ids = tokenizer(TEST_PROMPT, return_tensors="pt").input_ids.to(
            model.device
        )

        # Baseline generation
        with torch.no_grad():
            baseline_output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

        # Generation with zero cache
        k_compressors = [ZeroCompressor() for _ in range(num_layers)]
        v_compressors = [ZeroCompressor() for _ in range(num_layers)]
        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        with torch.no_grad():
            corrupted_output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
            )

        corrupted_text = tokenizer.decode(corrupted_output[0], skip_special_tokens=True)

        print(f"\nGeneration test:")
        print(f"  Baseline: {baseline_text}")
        print(f"  Corrupted: {corrupted_text}")

        # Texts should be different if cache is on path
        assert (
            baseline_text != corrupted_text
        ), "Cache bypass detected! Baseline and corrupted outputs are identical."


if __name__ == "__main__":
    # Run tests directly
    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
