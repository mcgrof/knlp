#!/usr/bin/env python3
"""
OrthogonalCompressor Unit Tests

Tests the OrthogonalCompressor from KV Plugin v3 for:
1. Shape & API correctness
2. Orthonormality/stability of projection matrices
3. Plugin integration with HuggingFace models

Run with: pytest tests/test_orthogonal_compressor.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOrthogonalCompressorUnit:
    """Unit tests for OrthogonalCompressor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for all tests."""
        from gpt2.compression.kv_plugin import (
            KVCompressorConfig,
            OrthogonalCompressor,
        )

        self.KVCompressorConfig = KVCompressorConfig
        self.OrthogonalCompressor = OrthogonalCompressor

    def test_shape_roundtrip(self):
        """Test compress -> expand shape roundtrip."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=32)
        compressor = self.OrthogonalCompressor(config)
        compressor = compressor.cpu().float()  # Ensure CPU and float32 for unit test

        # Test input
        x = torch.randn(2, 8, 128, 64)  # [B, H, T, d_input]

        # Compress
        z = compressor.compress(x)
        assert z.shape == (2, 8, 128, 32), f"Expected (2,8,128,32), got {z.shape}"

        # Expand
        x_hat = compressor.expand(z)
        assert x_hat.shape == x.shape, f"Expected {x.shape}, got {x_hat.shape}"

    def test_orthonormality(self):
        """Test that W.T @ W ≈ I (columns are orthonormal)."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=32)
        compressor = self.OrthogonalCompressor(config)
        compressor = compressor.cpu().float()  # Ensure CPU and float32 for unit test

        # Get compression weight
        W = compressor.compress_proj.weight.T  # [d_input, d_compressed]

        # Check W.T @ W ≈ I
        product = W.T @ W  # [d_compressed, d_compressed]
        identity = torch.eye(32)

        error = (product - identity).abs().max().item()
        assert error < 1e-3, f"W.T @ W not close to identity, max error: {error}"

    def test_expand_is_transpose(self):
        """Test that expand weight is transpose of compress weight."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=32)
        compressor = self.OrthogonalCompressor(config)

        W_compress = compressor.compress_proj.weight  # [d_compressed, d_input]
        W_expand = compressor.expand_proj.weight  # [d_input, d_compressed]

        # expand weight should be transpose of compress weight
        error = (W_expand - W_compress.T).abs().max().item()
        assert error < 1e-6, f"Expand weight not transpose of compress, error: {error}"

    def test_full_rank_identity(self):
        """Test d_compressed == d_input gives approximate identity."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=64)
        compressor = self.OrthogonalCompressor(config)
        compressor = compressor.cpu().float()  # Ensure CPU and float32 for unit test

        x = torch.randn(1, 1, 10, 64)
        z = compressor.compress(x)
        x_hat = compressor.expand(z)

        # Should be very close to identity (loosen tolerance for float32)
        error = (x_hat - x).abs().max().item()
        assert error < 2e-3, f"Full-rank not identity, max error: {error}"

    def test_calibration_improves_reconstruction(self):
        """Test that calibration improves reconstruction MSE."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=32)
        compressor = self.OrthogonalCompressor(config)
        compressor = compressor.cpu().float()  # Ensure CPU and float32 for unit test

        # Create calibration data with structure
        torch.manual_seed(42)
        # Data with dominant directions (first 32 dims have 10x variance)
        base = torch.randn(100, 64)
        base[:, :32] *= 10
        calibration_data = base  # [100, 64] - calibrate expects [N, d_input]

        # Measure MSE before calibration
        test_data = base[:20]
        z_before = compressor.compress(test_data)
        recon_before = compressor.expand(z_before)
        mse_before = ((test_data - recon_before) ** 2).mean().item()

        # Calibrate
        compressor.calibrate(calibration_data)

        # Measure MSE after calibration
        z_after = compressor.compress(test_data)
        recon_after = compressor.expand(z_after)
        mse_after = ((test_data - recon_after) ** 2).mean().item()

        # After calibration should be better (lower MSE)
        assert (
            mse_after < mse_before
        ), f"Calibration didn't improve MSE: {mse_before:.4f} -> {mse_after:.4f}"

    def test_works_without_calibration(self):
        """Test compressor works without calibration (zero-calibration mode)."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=32)
        compressor = self.OrthogonalCompressor(config)
        compressor = compressor.cpu().float()  # Ensure CPU and float32 for unit test

        # Should be marked as calibrated by default
        assert compressor.calibrated is True

        # Should work without calibration
        x = torch.randn(1, 1, 10, 64)
        z = compressor.compress(x)
        x_hat = compressor.expand(z)

        # Output should be finite
        assert torch.isfinite(x_hat).all(), "Output contains NaN/Inf"

    def test_different_dtypes(self):
        """Test compressor works with different dtypes."""
        config = self.KVCompressorConfig(d_input=64, d_compressed=32)

        for dtype in [torch.float32, torch.float16]:
            compressor = self.OrthogonalCompressor(config)
            compressor = compressor.cpu()  # Ensure CPU for unit test
            compressor = compressor.to(dtype)

            x = torch.randn(1, 1, 10, 64, dtype=dtype)
            z = compressor.compress(x)
            x_hat = compressor.expand(z)

            assert z.dtype == dtype, f"Compressed dtype mismatch for {dtype}"
            assert x_hat.dtype == dtype, f"Expanded dtype mismatch for {dtype}"
            assert torch.isfinite(x_hat).all(), f"Output not finite for {dtype}"


class TestOrthogonalPluginIntegration:
    """Integration tests for orthogonal preset with KVPlugin."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for all tests."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from gpt2.compression.kv_plugin import KVPlugin

        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.KVPlugin = KVPlugin

    def test_orthogonal_preset_exists(self):
        """Test orthogonal presets are registered."""
        assert "orthogonal" in self.KVPlugin.PRESETS
        assert "orthogonal_aggressive" in self.KVPlugin.PRESETS

    def test_create_orthogonal_plugin(self):
        """Test creating plugin with orthogonal preset."""
        model = self.AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        plugin = self.KVPlugin.from_preset("orthogonal", model)

        assert plugin is not None
        assert plugin.config.compressor_type == "orthogonal"
        assert len(plugin.k_compressors) == 12  # GPT-2 has 12 layers
        assert len(plugin.v_compressors) == 12

    @torch.no_grad()
    def test_gpt2_forward_with_orthogonal(self):
        """Test GPT-2 forward pass with orthogonal preset."""
        model_name = "openai-community/gpt2"
        tokenizer = self.AutoTokenizer.from_pretrained(model_name)
        model = self.AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        # Create plugin and patch model
        plugin = self.KVPlugin.from_preset("orthogonal", model)
        plugin.patch_model()

        # Forward pass with cache
        inputs = tokenizer("Hello orthogonal KV!", return_tensors="pt")
        out = model(**inputs, use_cache=True)

        logits = out.logits if hasattr(out, "logits") else out["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[1] == inputs["input_ids"].shape[1]
        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"

    @torch.no_grad()
    def test_gpt2_generate_with_orthogonal(self):
        """Test GPT-2 generation with orthogonal preset."""
        model_name = "openai-community/gpt2"
        tokenizer = self.AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self.AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        # Create plugin and patch model
        plugin = self.KVPlugin.from_preset("orthogonal", model)
        plugin.patch_model()

        # Generate
        inputs = tokenizer("The quick brown", return_tensors="pt")
        gen = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )

        assert gen.shape[0] == 1
        assert gen.shape[1] > inputs["input_ids"].shape[1]

        # Decode and check it's reasonable text
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        assert len(text) > len("The quick brown")

    @torch.no_grad()
    def test_orthogonal_without_calibration(self):
        """Test orthogonal preset works without explicit calibration."""
        model_name = "openai-community/gpt2"
        tokenizer = self.AutoTokenizer.from_pretrained(model_name)
        model = self.AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        # Create plugin - don't calibrate
        plugin = self.KVPlugin.from_preset("orthogonal", model)

        # All compressors should be marked calibrated (zero-calibration mode)
        for compressor in plugin.k_compressors:
            assert compressor.calibrated is True
        for compressor in plugin.v_compressors:
            assert compressor.calibrated is True

        # Patch and run
        plugin.patch_model()
        inputs = tokenizer("Testing zero-cal", return_tensors="pt")
        out = model(**inputs, use_cache=True)

        assert torch.isfinite(out.logits).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
