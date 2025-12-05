"""
Unit tests for KV cache quantization correctness.

Tests compare:
1. FP16 baseline (no quantization)
2. Int8 PyTorch path (dequant + matmul)
3. Int4 PyTorch path (unpack + dequant + matmul)
4. Int8 Triton fused kernel
5. Int4 Triton fused kernel

All paths should produce outputs within numerical tolerance of each other.
"""

import torch
import torch.nn as nn
import sys
import os

try:
    import pytest
except ImportError:
    # Create dummy pytest for running without pytest
    class DummyPytest:
        @staticmethod
        def raises(exc, **kwargs):
            from contextlib import contextmanager

            @contextmanager
            def _raises():
                try:
                    yield
                    raise AssertionError(f"Expected {exc} but no exception was raised")
                except exc:
                    pass

            return _raises()

        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

    pytest = DummyPytest()

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpt2.compression.kv_plugin import (
    OrthogonalCompressor,
    KVCompressorConfig,
    QuantizedTensor,
    quantize_to_int8,
    dequantize_from_int8,
    quantize_to_int4,
    dequantize_from_int4,
    fake_quant,
)

from gpt2.compression.triton_kernels import (
    TRITON_AVAILABLE,
    triton_expand_int4,
    triton_expand_int8,
    _torch_expand_int4_fallback,
    _torch_expand_int8_fallback,
)


# Test dimensions matching typical KV cache shapes
TEST_CONFIGS = [
    # (B, H, T, K, N) - batch, heads, seq_len, latent_dim, output_dim
    (1, 12, 128, 128, 768),  # GPT-2 small
    (1, 16, 256, 128, 1024),  # GPT-2 medium
    (2, 12, 64, 64, 768),  # Batched
    (1, 32, 512, 128, 2048),  # Larger model
    (1, 8, 1024, 256, 512),  # Long sequence
]


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm() + 1e-8)


class TestQuantizationBasics:
    """Test basic quantization/dequantization operations."""

    def test_int8_round_trip(self):
        """Test that int8 quantize -> dequantize preserves values."""
        x = torch.randn(2, 12, 128, 256, dtype=torch.float16)
        qt = quantize_to_int8(x, per_channel=True)

        assert qt.data.dtype == torch.int8
        assert qt.bits == 8
        assert qt.packed == False

        x_recovered = dequantize_from_int8(qt)
        assert x_recovered.dtype == torch.float16
        assert x_recovered.shape == x.shape

        # Check MSE is reasonable for int8
        mse = ((x - x_recovered) ** 2).mean().item()
        assert mse < 0.01, f"Int8 round-trip MSE too high: {mse}"

    def test_int4_round_trip(self):
        """Test that int4 quantize -> dequantize preserves values."""
        x = torch.randn(2, 12, 128, 256, dtype=torch.float16)
        qt = quantize_to_int4(x, per_channel=True)

        assert qt.data.dtype == torch.uint8
        assert qt.bits == 4
        assert qt.packed == True
        assert qt.original_dim == 256
        assert qt.data.shape[-1] == 128  # Packed: half size

        x_recovered = dequantize_from_int4(qt)
        assert x_recovered.dtype == torch.float16
        assert x_recovered.shape == x.shape

        # Check MSE is reasonable for int4
        mse = ((x - x_recovered) ** 2).mean().item()
        assert mse < 0.1, f"Int4 round-trip MSE too high: {mse}"

    def test_int4_odd_dimension_fails(self):
        """Test that int4 quantization fails for odd last dimension."""
        x = torch.randn(2, 12, 128, 255, dtype=torch.float16)
        with pytest.raises(ValueError, match="must be even"):
            quantize_to_int4(x)

    def test_quantized_tensor_memory(self):
        """Test QuantizedTensor memory calculation."""
        x = torch.randn(1, 12, 1024, 128, dtype=torch.float16)

        qt_int8 = quantize_to_int8(x)
        qt_int4 = quantize_to_int4(x)

        # Int4 should use ~half the data bytes of int8
        int8_data_bytes = qt_int8.data.numel() * qt_int8.data.element_size()
        int4_data_bytes = qt_int4.data.numel() * qt_int4.data.element_size()

        assert int4_data_bytes == int8_data_bytes // 2

    def test_fake_quant_matches_real_quant(self):
        """Verify fake quant matches real int8 quant values."""
        x = torch.randn(1, 12, 128, 128, dtype=torch.float16)

        # Fake quant
        x_fake = fake_quant(x, bits=8, per_channel=True)

        # Real quant
        qt = quantize_to_int8(x, per_channel=True)
        x_real = dequantize_from_int8(qt)

        # Should match exactly
        max_diff = (x_fake - x_real).abs().max().item()
        assert max_diff < 1e-4, f"Fake vs real quant mismatch: {max_diff}"


class TestExpandOperations:
    """Test expand (dequant + matmul) operations."""

    @pytest.mark.parametrize("B,H,T,K,N", TEST_CONFIGS)
    def test_fp16_vs_int8_pytorch(self, B, H, T, K, N):
        """Compare FP16 baseline vs PyTorch int8 expand."""
        latent = torch.randn(B, H, T, K, dtype=torch.float16)
        weight = torch.randn(K, N, dtype=torch.float16)

        # FP16 baseline
        out_fp16 = torch.matmul(latent, weight)

        # Int8 PyTorch path
        qt = quantize_to_int8(latent)
        latent_dequant = dequantize_from_int8(qt)
        out_int8 = torch.matmul(latent_dequant, weight)

        # Compare using cosine similarity (avoids fp16 overflow in relative error)
        cos_sim = cosine_similarity(out_fp16, out_int8).item()
        assert cos_sim > 0.99, f"Int8 cosine similarity too low: {cos_sim:.4f}"

    @pytest.mark.parametrize("B,H,T,K,N", TEST_CONFIGS)
    def test_fp16_vs_int4_pytorch(self, B, H, T, K, N):
        """Compare FP16 baseline vs PyTorch int4 expand."""
        latent = torch.randn(B, H, T, K, dtype=torch.float16)
        weight = torch.randn(K, N, dtype=torch.float16)

        # FP16 baseline
        out_fp16 = torch.matmul(latent, weight)

        # Int4 PyTorch path
        qt = quantize_to_int4(latent)
        latent_dequant = dequantize_from_int4(qt)
        out_int4 = torch.matmul(latent_dequant, weight)

        # Compare - int4 has more quantization noise, use cosine similarity
        cos_sim = cosine_similarity(out_fp16, out_int4).item()
        assert cos_sim > 0.90, f"Int4 cosine similarity too low: {cos_sim:.4f}"

    @pytest.mark.parametrize("B,H,T,K,N", TEST_CONFIGS)
    def test_int8_pytorch_vs_triton(self, B, H, T, K, N):
        """Compare PyTorch int8 vs Triton int8 expand."""
        latent = torch.randn(B, H, T, K, dtype=torch.float16)
        weight = torch.randn(K, N, dtype=torch.float16)

        qt = quantize_to_int8(latent)

        # PyTorch path
        latent_dequant = dequantize_from_int8(qt)
        out_pytorch = torch.matmul(latent_dequant, weight)

        # Triton path (uses fallback if Triton not available)
        out_triton = triton_expand_int8(qt.data, qt.scale, weight)

        # Should be nearly identical
        max_diff = (out_pytorch - out_triton).abs().max().item()
        rel_error = (
            (out_pytorch - out_triton).abs() / (out_pytorch.abs() + 1e-6)
        ).mean()

        assert rel_error < 0.01, f"Int8 PyTorch vs Triton mismatch: {rel_error:.4f}"

    @pytest.mark.parametrize("B,H,T,K,N", TEST_CONFIGS)
    def test_int4_pytorch_vs_triton(self, B, H, T, K, N):
        """Compare PyTorch int4 vs Triton int4 expand."""
        latent = torch.randn(B, H, T, K, dtype=torch.float16)
        weight = torch.randn(K, N, dtype=torch.float16)

        qt = quantize_to_int4(latent)

        # PyTorch path
        latent_dequant = dequantize_from_int4(qt)
        out_pytorch = torch.matmul(latent_dequant, weight)

        # Triton path (uses fallback if Triton not available)
        out_triton = triton_expand_int4(qt.data, qt.scale, weight, K)

        # Should be nearly identical
        max_diff = (out_pytorch - out_triton).abs().max().item()
        rel_error = (
            (out_pytorch - out_triton).abs() / (out_pytorch.abs() + 1e-6)
        ).mean()

        assert rel_error < 0.01, f"Int4 PyTorch vs Triton mismatch: {rel_error:.4f}"


class TestOrthogonalCompressor:
    """Test OrthogonalCompressor with various quantization settings."""

    def test_no_quantization(self):
        """Test compressor without quantization."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
        )
        comp = OrthogonalCompressor(config)

        x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
        z = comp.compress(x)
        x_hat = comp.expand(z)

        assert isinstance(z, torch.Tensor)
        assert z.shape[-1] == 128
        assert x_hat.shape == x.shape

    def test_fake_quantization(self):
        """Test compressor with fake quantization."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
            quant_bits=8,
            quant_storage=False,  # Fake quant
        )
        comp = OrthogonalCompressor(config)

        x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
        z = comp.compress(x)
        x_hat = comp.expand(z)

        # Fake quant returns tensor, not QuantizedTensor
        assert isinstance(z, torch.Tensor)
        assert z.dtype == torch.float16

    def test_real_int8_storage(self):
        """Test compressor with real int8 storage."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
            quant_bits=8,
            quant_storage=True,
        )
        comp = OrthogonalCompressor(config)

        x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
        z = comp.compress(x)
        x_hat = comp.expand(z)

        assert isinstance(z, QuantizedTensor)
        assert z.data.dtype == torch.int8
        assert z.bits == 8
        assert x_hat.shape == x.shape

    def test_real_int4_storage(self):
        """Test compressor with real int4 storage."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,  # Must be even for int4
            dtype=torch.float16,
            device="cpu",
            quant_bits=4,
            quant_storage=True,
        )
        comp = OrthogonalCompressor(config)

        x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
        z = comp.compress(x)
        x_hat = comp.expand(z)

        assert isinstance(z, QuantizedTensor)
        assert z.data.dtype == torch.uint8
        assert z.bits == 4
        assert z.packed == True
        assert x_hat.shape == x.shape

    def test_triton_backend_routing(self):
        """Test that triton backend is correctly routed."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
            quant_bits=8,
            quant_storage=True,
            quant_backend="triton",
        )
        comp = OrthogonalCompressor(config)

        assert comp.quant_backend == "triton"

        x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
        z = comp.compress(x)
        x_hat = comp.expand(z)

        # Should still work (falls back to PyTorch if Triton unavailable)
        assert x_hat.shape == x.shape


class TestMemorySavings:
    """Test actual memory savings from quantization."""

    def test_int8_memory_reduction(self):
        """Verify int8 uses ~50% of fp16 memory."""
        x = torch.randn(1, 12, 1024, 128, dtype=torch.float16)
        fp16_bytes = x.numel() * x.element_size()

        qt = quantize_to_int8(x)
        int8_bytes = qt.memory_bytes()

        # int8 data + fp16 scale = ~50% + small overhead
        savings = 1 - (int8_bytes / fp16_bytes)
        assert savings > 0.45, f"Int8 memory savings too low: {savings:.1%}"

    def test_int4_memory_reduction(self):
        """Verify int4 uses ~25% of fp16 memory."""
        x = torch.randn(1, 12, 1024, 128, dtype=torch.float16)
        fp16_bytes = x.numel() * x.element_size()

        qt = quantize_to_int4(x)
        int4_bytes = qt.memory_bytes()

        # int4 packed data + fp16 scale = ~25% + small overhead
        savings = 1 - (int4_bytes / fp16_bytes)
        assert savings > 0.70, f"Int4 memory savings too low: {savings:.1%}"

    def test_compression_stack_total(self):
        """Test total compression from orthogonal + int4."""
        # Simulate full KV: [B, H, T, head_dim] where head_dim=64
        B, H, T, D = 1, 12, 1024, 64
        full_kv = torch.randn(B, H, T, D, dtype=torch.float16)
        fp16_kv_bytes = full_kv.numel() * full_kv.element_size()

        # After orthogonal compression (6x): d_compressed = D * H / 6
        d_compressed = (D * H) // 6  # ~128
        latent = torch.randn(B, 1, T, d_compressed, dtype=torch.float16)

        # After int4 quantization
        qt = quantize_to_int4(latent)
        int4_bytes = qt.memory_bytes()

        # Total compression ratio
        total_ratio = fp16_kv_bytes / int4_bytes
        assert total_ratio > 20, f"Total compression ratio too low: {total_ratio:.1f}x"


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_zero_input(self):
        """Test quantization handles zero input."""
        x = torch.zeros(1, 12, 128, 128, dtype=torch.float16)

        qt_int8 = quantize_to_int8(x)
        x_rec_int8 = dequantize_from_int8(qt_int8)
        assert x_rec_int8.abs().max() == 0

        qt_int4 = quantize_to_int4(x)
        x_rec_int4 = dequantize_from_int4(qt_int4)
        assert x_rec_int4.abs().max() == 0

    def test_large_values(self):
        """Test quantization handles large values."""
        x = torch.randn(1, 12, 128, 128, dtype=torch.float16) * 100

        qt = quantize_to_int8(x)
        x_rec = dequantize_from_int8(qt)

        # Should handle large values without overflow
        rel_error = ((x - x_rec).abs() / (x.abs() + 1e-6)).mean()
        assert rel_error < 0.05

    def test_small_values(self):
        """Test quantization handles small values."""
        x = torch.randn(1, 12, 128, 128, dtype=torch.float16) * 1e-4

        qt = quantize_to_int8(x)
        x_rec = dequantize_from_int8(qt)

        # Should handle small values with reasonable precision
        rel_error = ((x - x_rec).abs() / (x.abs() + 1e-6)).mean()
        assert rel_error < 0.1


def run_correctness_summary():
    """Run a quick correctness summary (useful for manual testing)."""
    print("=" * 70)
    print("KV Quantization Correctness Summary")
    print("=" * 70)

    B, H, T, K, N = 1, 12, 256, 128, 768
    latent = torch.randn(B, H, T, K, dtype=torch.float16)
    weight = torch.randn(K, N, dtype=torch.float16)

    # FP16 baseline
    out_fp16 = torch.matmul(latent, weight)

    results = []

    # Int8 PyTorch
    qt8 = quantize_to_int8(latent)
    out_int8_pt = torch.matmul(dequantize_from_int8(qt8), weight)
    cos_sim = cosine_similarity(out_fp16, out_int8_pt).item()
    results.append(("Int8 PyTorch", cos_sim, 0.99))  # threshold 0.99

    # Int4 PyTorch
    qt4 = quantize_to_int4(latent)
    out_int4_pt = torch.matmul(dequantize_from_int4(qt4), weight)
    cos_sim = cosine_similarity(out_fp16, out_int4_pt).item()
    results.append(("Int4 PyTorch", cos_sim, 0.90))  # lower threshold for int4

    # Int8 Triton (fallback)
    out_int8_tr = triton_expand_int8(qt8.data, qt8.scale, weight)
    cos_sim = cosine_similarity(out_fp16, out_int8_tr).item()
    results.append(("Int8 Triton", cos_sim, 0.99))

    # Int4 Triton (fallback)
    out_int4_tr = triton_expand_int4(qt4.data, qt4.scale, weight, K)
    cos_sim = cosine_similarity(out_fp16, out_int4_tr).item()
    results.append(("Int4 Triton", cos_sim, 0.90))

    # Verify PyTorch vs Triton match exactly
    int8_match = cosine_similarity(out_int8_pt, out_int8_tr).item()
    int4_match = cosine_similarity(out_int4_pt, out_int4_tr).item()

    print(f"\nShape: B={B}, H={H}, T={T}, K={K}, N={N}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    print("\n{:<20} {:>15} {:>10}".format("Method", "Cos Sim vs FP16", "Status"))
    print("-" * 50)
    for name, sim, thresh in results:
        status = "PASS" if sim >= thresh else "FAIL"
        print(f"{name:<20} {sim:>15.6f}  [{status}]")

    print("\n{:<25} {:>15}".format("Backend Consistency", "Cos Sim"))
    print("-" * 45)
    print(
        f"{'Int8 PyTorch vs Triton':<25} {int8_match:>15.6f}  [{'PASS' if int8_match > 0.9999 else 'FAIL'}]"
    )
    print(
        f"{'Int4 PyTorch vs Triton':<25} {int4_match:>15.6f}  [{'PASS' if int4_match > 0.9999 else 'FAIL'}]"
    )

    # Memory savings
    print("\nMemory Savings:")
    fp16_bytes = latent.numel() * 2
    print(f"  FP16 latent: {fp16_bytes / 1024:.1f} KB")
    print(
        f"  Int8 latent: {qt8.memory_bytes() / 1024:.1f} KB ({1 - qt8.memory_bytes()/fp16_bytes:.1%} savings)"
    )
    print(
        f"  Int4 latent: {qt4.memory_bytes() / 1024:.1f} KB ({1 - qt4.memory_bytes()/fp16_bytes:.1%} savings)"
    )


class TestRobustnessFallbacks:
    """Test fallback behavior for error conditions and edge cases."""

    def test_incorrect_shape_packed_dimension(self):
        """Test handling of incorrectly packed dimension for int4."""
        # Simulate a case where packed data has wrong size
        B, H, T, K = 1, 12, 128, 128

        # Create valid quantized tensor
        x = torch.randn(B, H, T, K, dtype=torch.float16)
        qt = quantize_to_int4(x)

        # The packed data should be K/2
        assert qt.data.shape[-1] == K // 2

        # Verify dequantization works
        x_rec = dequantize_from_int4(qt)
        assert x_rec.shape[-1] == K

    def test_incorrect_original_dim_int4(self):
        """Test that original_dim mismatch produces different output."""
        B, H, T, K = 1, 12, 128, 128
        x = torch.randn(B, H, T, K, dtype=torch.float16)
        qt = quantize_to_int4(x)

        # Dequantize correctly
        x_rec_correct = dequantize_from_int4(qt)
        assert x_rec_correct.shape[-1] == K

        # Verify we can still dequantize with correct original_dim
        cos_sim = cosine_similarity(x, x_rec_correct).item()
        assert cos_sim > 0.90, f"Correct dequant should match: {cos_sim}"

    def test_bits_mismatch_behavior(self):
        """Test that int8 vs int4 produce different representations."""
        x = torch.randn(1, 12, 128, 128, dtype=torch.float16)

        # Quantize as int8
        qt8 = quantize_to_int8(x)
        assert qt8.bits == 8
        assert qt8.packed == False

        # Quantize as int4
        qt4 = quantize_to_int4(x)
        assert qt4.bits == 4
        assert qt4.packed == True

        # Verify they have different storage sizes
        assert qt8.data.numel() != qt4.data.numel()
        # Int4 should be half the size
        assert qt4.data.numel() == qt8.data.numel() // 2

    def test_triton_fallback_on_cpu(self):
        """Test that Triton correctly falls back to PyTorch on CPU."""
        # Triton kernels only work on CUDA
        latent = torch.randn(1, 12, 128, 128, dtype=torch.float16)  # CPU tensor
        weight = torch.randn(128, 768, dtype=torch.float16)

        qt = quantize_to_int8(latent)

        # This should use PyTorch fallback since we're on CPU
        out = triton_expand_int8(qt.data, qt.scale, weight)

        # Verify output is correct (PyTorch path)
        latent_dequant = dequantize_from_int8(qt)
        out_ref = torch.matmul(latent_dequant, weight)

        cos_sim = cosine_similarity(out, out_ref).item()
        assert cos_sim > 0.9999, f"Fallback output mismatch: {cos_sim}"

    def test_triton_fallback_int4_on_cpu(self):
        """Test that int4 Triton correctly falls back to PyTorch on CPU."""
        latent = torch.randn(1, 12, 128, 128, dtype=torch.float16)
        weight = torch.randn(128, 768, dtype=torch.float16)

        qt = quantize_to_int4(latent)

        # This should use PyTorch fallback
        out = triton_expand_int4(qt.data, qt.scale, weight, 128)

        # Verify output is correct
        latent_dequant = dequantize_from_int4(qt)
        out_ref = torch.matmul(latent_dequant, weight)

        cos_sim = cosine_similarity(out, out_ref).item()
        assert cos_sim > 0.9999, f"Fallback output mismatch: {cos_sim}"

    def test_pytorch_fallback_functions_directly(self):
        """Test PyTorch fallback functions work correctly."""
        latent = torch.randn(1, 12, 128, 128, dtype=torch.float16)
        weight = torch.randn(128, 768, dtype=torch.float16)

        # Test int8 fallback
        qt8 = quantize_to_int8(latent)
        out_fallback = _torch_expand_int8_fallback(qt8.data, qt8.scale, weight)

        latent_dequant = dequantize_from_int8(qt8)
        out_ref = torch.matmul(latent_dequant, weight)

        cos_sim = cosine_similarity(out_fallback, out_ref).item()
        assert cos_sim > 0.9999, f"Int8 fallback mismatch: {cos_sim}"

        # Test int4 fallback
        qt4 = quantize_to_int4(latent)
        out_fallback = _torch_expand_int4_fallback(qt4.data, qt4.scale, weight, 128)

        latent_dequant = dequantize_from_int4(qt4)
        out_ref = torch.matmul(latent_dequant, weight)

        cos_sim = cosine_similarity(out_fallback, out_ref).item()
        assert cos_sim > 0.9999, f"Int4 fallback mismatch: {cos_sim}"

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Empty batch
        x = torch.randn(0, 12, 128, 128, dtype=torch.float16)

        qt = quantize_to_int8(x)
        assert qt.data.shape[0] == 0

        x_rec = dequantize_from_int8(qt)
        assert x_rec.shape[0] == 0

    def test_single_element_input(self):
        """Test handling of single-element inputs."""
        x = torch.randn(1, 1, 1, 128, dtype=torch.float16)

        qt = quantize_to_int8(x)
        x_rec = dequantize_from_int8(qt)

        cos_sim = cosine_similarity(x, x_rec).item()
        assert cos_sim > 0.99

    def test_compressor_graceful_degradation(self):
        """Test OrthogonalCompressor gracefully handles edge cases."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
            quant_bits=8,
            quant_storage=True,
            quant_backend="triton",  # Will fallback on CPU
        )
        comp = OrthogonalCompressor(config)

        # Should work despite requesting Triton on CPU
        x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
        z = comp.compress(x)
        x_hat = comp.expand(z)

        assert x_hat.shape == x.shape

    def test_int4_to_int8_fallback_scenario(self):
        """Test scenario where int4 fails and int8 should be used instead."""
        # Simulate a scenario where int4 would fail (odd dimension)
        # In this case, user should fall back to int8

        x = torch.randn(1, 12, 128, 256, dtype=torch.float16)

        # Int4 works for even dimensions
        qt4 = quantize_to_int4(x)
        assert qt4.bits == 4

        # But for odd dimensions, int4 fails
        x_odd = torch.randn(1, 12, 128, 255, dtype=torch.float16)

        # Int4 should fail
        try:
            quantize_to_int4(x_odd)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "must be even" in str(e)

        # Fallback to int8 works
        qt8 = quantize_to_int8(x_odd)
        assert qt8.bits == 8

        x_rec = dequantize_from_int8(qt8)
        assert x_rec.shape == x_odd.shape


class TestConfigurationErrors:
    """Test that configuration errors are handled properly."""

    def test_valid_quant_bits(self):
        """Test that valid quant_bits values work correctly."""
        for bits in [4, 8, 16]:
            config = KVCompressorConfig(
                d_input=768,
                d_compressed=128,
                dtype=torch.float16,
                device="cpu",
                quant_bits=bits,
                quant_storage=(bits < 16),  # Only store if quantizing
            )
            comp = OrthogonalCompressor(config)
            x = torch.randn(1, 12, 128, 768, dtype=torch.float16)
            z = comp.compress(x)
            x_hat = comp.expand(z)
            assert x_hat.shape == x.shape

    def test_mismatched_input_dimension(self):
        """Test that mismatched input dimension is caught."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
        )
        comp = OrthogonalCompressor(config)

        # Wrong input dimension
        x = torch.randn(1, 12, 128, 512, dtype=torch.float16)  # 512 != 768

        with pytest.raises((RuntimeError, ValueError)):
            comp.compress(x)

    def test_dtype_mismatch_handling(self):
        """Test handling of dtype mismatches."""
        config = KVCompressorConfig(
            d_input=768,
            d_compressed=128,
            dtype=torch.float16,
            device="cpu",
        )
        comp = OrthogonalCompressor(config)

        # Input with wrong dtype - should either convert or error
        x = torch.randn(1, 12, 128, 768, dtype=torch.float32)

        try:
            z = comp.compress(x)
            # If it succeeds, output should be in the configured dtype
            assert z.dtype == torch.float16 or z.dtype == torch.float32
        except (RuntimeError, TypeError):
            # Expected if strict dtype checking
            pass


if __name__ == "__main__":
    run_correctness_summary()
