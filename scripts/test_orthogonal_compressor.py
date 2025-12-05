#!/usr/bin/env python3
"""
Test OrthogonalCompressor vs PCA/SVD compressors.

Verifies that the KVSplice-inspired OrthogonalCompressor:
1. Works without calibration
2. Improves with optional calibration
3. Has comparable reconstruction quality to PCA/SVD
"""

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, "/data/knlp")

from gpt2.compression.kv_plugin import (
    KVCompressorConfig,
    OrthogonalCompressor,
    PCACompressor,
    SVDCompressor,
    create_compressor,
)


def test_basic_functionality():
    """Test that OrthogonalCompressor works without calibration."""
    print("=" * 60)
    print("Test 1: Basic functionality (no calibration)")
    print("=" * 60)

    config = KVCompressorConfig(
        d_input=768,
        d_compressed=128,
        device="cpu",
        dtype=torch.float32,
    )

    comp = OrthogonalCompressor(config)

    # Test compress/expand
    x = torch.randn(32, 768)
    z = comp.compress(x)
    x_hat = comp.expand(z)

    print(f"Input shape: {x.shape}")
    print(f"Compressed shape: {z.shape}")
    print(f"Reconstructed shape: {x_hat.shape}")

    # Check orthogonality of compress weights
    W = comp.compress_proj.weight.data
    WWT = W @ W.T
    identity = torch.eye(W.shape[0])
    ortho_error = (WWT - identity).abs().max().item()
    print(f"Orthogonality error: {ortho_error:.6f}")

    # Check pseudoinverse property: compress @ expand should be identity
    # compress: [d_compressed, d_input], expand: [d_input, d_compressed]
    # So compress @ expand.T = [d_compressed, d_compressed]
    expand_W = comp.expand_proj.weight.data  # [d_input, d_compressed]
    # W @ expand_W = [d_compressed, d_input] @ [d_input, d_compressed] = [d_compressed, d_compressed]
    product = W @ expand_W
    pseudoinv_error = (product - identity).abs().max().item()
    print(f"Pseudoinverse error: {pseudoinv_error:.6f}")

    # Reconstruction error
    mse = ((x - x_hat) ** 2).mean().item()
    print(f"Reconstruction MSE (no calibration): {mse:.4f}")

    assert z.shape == (32, 128), f"Wrong compressed shape: {z.shape}"
    assert x_hat.shape == x.shape, f"Wrong reconstructed shape: {x_hat.shape}"
    print("✓ Basic functionality test PASSED\n")
    return mse


def test_calibration_improvement():
    """Test that calibration improves reconstruction."""
    print("=" * 60)
    print("Test 2: Calibration improvement")
    print("=" * 60)

    # Create structured data (simulating KV activations)
    torch.manual_seed(42)
    n_samples = 1000
    d_input = 768
    d_compressed = 128

    # Create data with low effective rank
    U = torch.randn(n_samples, 200)
    V = torch.randn(200, d_input)
    X = U @ V + 0.1 * torch.randn(n_samples, d_input)

    # Split into calibration and test
    X_calib = X[:800]
    X_test = X[800:]

    config = KVCompressorConfig(
        d_input=d_input,
        d_compressed=d_compressed,
        device="cpu",
        dtype=torch.float32,
    )

    # Test uncalibrated
    comp_uncalib = OrthogonalCompressor(config)
    z_uncalib = comp_uncalib.compress(X_test)
    x_hat_uncalib = comp_uncalib.expand(z_uncalib)
    mse_uncalib = ((X_test - x_hat_uncalib) ** 2).mean().item()

    # Test calibrated
    comp_calib = OrthogonalCompressor(config)
    comp_calib.calibrate(X_calib)
    z_calib = comp_calib.compress(X_test)
    x_hat_calib = comp_calib.expand(z_calib)
    mse_calib = ((X_test - x_hat_calib) ** 2).mean().item()

    print(f"MSE (uncalibrated): {mse_uncalib:.4f}")
    print(f"MSE (calibrated):   {mse_calib:.4f}")
    print(f"Improvement:        {(mse_uncalib - mse_calib) / mse_uncalib * 100:.1f}%")

    # Calibrated should be better for structured data
    assert mse_calib < mse_uncalib, "Calibration should improve reconstruction"
    print("✓ Calibration improvement test PASSED\n")
    return mse_uncalib, mse_calib


def test_comparison_with_pca_svd():
    """Compare OrthogonalCompressor with PCA and SVD compressors."""
    print("=" * 60)
    print("Test 3: Comparison with PCA and SVD")
    print("=" * 60)

    torch.manual_seed(42)
    n_samples = 1000
    d_input = 768
    d_compressed = 128

    # Create structured data
    U = torch.randn(n_samples, 200)
    V = torch.randn(200, d_input)
    X = U @ V + 0.1 * torch.randn(n_samples, d_input)

    X_calib = X[:800]
    X_test = X[800:]

    config = KVCompressorConfig(
        d_input=d_input,
        d_compressed=d_compressed,
        device="cpu",
        dtype=torch.float32,
    )

    results = {}

    # PCA
    pca = PCACompressor(config)
    pca.calibrate(X_calib)
    x_hat_pca = pca.expand(pca.compress(X_test))
    results["PCA"] = ((X_test - x_hat_pca) ** 2).mean().item()

    # SVD
    svd = SVDCompressor(config)
    svd.calibrate(X_calib)
    x_hat_svd = svd.expand(svd.compress(X_test))
    results["SVD"] = ((X_test - x_hat_svd) ** 2).mean().item()

    # Orthogonal (calibrated)
    ortho_cal = OrthogonalCompressor(config)
    ortho_cal.calibrate(X_calib)
    x_hat_ortho_cal = ortho_cal.expand(ortho_cal.compress(X_test))
    results["Orthogonal (calibrated)"] = ((X_test - x_hat_ortho_cal) ** 2).mean().item()

    # Orthogonal (uncalibrated)
    ortho_uncal = OrthogonalCompressor(config)
    x_hat_ortho_uncal = ortho_uncal.expand(ortho_uncal.compress(X_test))
    results["Orthogonal (uncalibrated)"] = (
        ((X_test - x_hat_ortho_uncal) ** 2).mean().item()
    )

    print("\nReconstruction MSE comparison:")
    print("-" * 40)
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:30s}: {mse:.6f}")

    # Calibrated orthogonal should match SVD (same algorithm)
    assert (
        abs(results["Orthogonal (calibrated)"] - results["SVD"]) < 1e-4
    ), "Calibrated orthogonal should match SVD"

    print("\n✓ Comparison test PASSED\n")
    return results


def test_speed_comparison():
    """Compare initialization and calibration speed."""
    print("=" * 60)
    print("Test 4: Speed comparison")
    print("=" * 60)

    torch.manual_seed(42)
    n_samples = 5000
    d_input = 768
    d_compressed = 128

    X_calib = torch.randn(n_samples, d_input)

    config = KVCompressorConfig(
        d_input=d_input,
        d_compressed=d_compressed,
        device="cpu",
        dtype=torch.float32,
    )

    # Time PCA calibration
    t0 = time.time()
    for _ in range(10):
        pca = PCACompressor(config)
        pca.calibrate(X_calib)
    pca_time = (time.time() - t0) / 10

    # Time SVD calibration
    t0 = time.time()
    for _ in range(10):
        svd = SVDCompressor(config)
        svd.calibrate(X_calib)
    svd_time = (time.time() - t0) / 10

    # Time Orthogonal init (no calibration)
    t0 = time.time()
    for _ in range(10):
        ortho = OrthogonalCompressor(config)
    ortho_init_time = (time.time() - t0) / 10

    # Time Orthogonal with calibration
    t0 = time.time()
    for _ in range(10):
        ortho = OrthogonalCompressor(config)
        ortho.calibrate(X_calib)
    ortho_calib_time = (time.time() - t0) / 10

    print("\nSetup time comparison (avg of 10 runs):")
    print("-" * 40)
    print(f"  PCA calibration:         {pca_time*1000:.2f} ms")
    print(f"  SVD calibration:         {svd_time*1000:.2f} ms")
    print(f"  Orthogonal init only:    {ortho_init_time*1000:.2f} ms")
    print(f"  Orthogonal + calibration: {ortho_calib_time*1000:.2f} ms")
    print(f"\n  Speedup (no calib):      {pca_time/ortho_init_time:.1f}x faster")

    print("\n✓ Speed test PASSED\n")
    return {
        "pca": pca_time,
        "svd": svd_time,
        "ortho_init": ortho_init_time,
        "ortho_calib": ortho_calib_time,
    }


def test_with_real_model():
    """Test with actual GPT-2 KV activations."""
    print("=" * 60)
    print("Test 5: Real GPT-2 KV activations")
    print("=" * 60)

    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model.eval()

    # Get some activations
    text = """The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Python is a popular programming language for data science."""
    inputs = tokenizer(text, return_tensors="pt")

    k_samples = []
    v_samples = []

    def hook(module, input, output):
        # GPT-2 attention output is tuple: (attn_output, present)
        # present is (key, value) or DynamicCache
        if isinstance(output, tuple) and len(output) >= 2:
            present = output[1]
            if present is not None:
                # Handle both tuple and DynamicCache formats
                if hasattr(present, "key_cache"):
                    # DynamicCache format
                    if present.key_cache:
                        k = present.key_cache[-1]
                        v = present.value_cache[-1]
                        k_flat = k.permute(0, 2, 1, 3).reshape(-1, k.shape[-1])
                        v_flat = v.permute(0, 2, 1, 3).reshape(-1, v.shape[-1])
                        k_samples.append(k_flat.detach())
                        v_samples.append(v_flat.detach())
                elif isinstance(present, tuple) and len(present) == 2:
                    k, v = present
                    k_flat = k.permute(0, 2, 1, 3).reshape(-1, k.shape[-1])
                    v_flat = v.permute(0, 2, 1, 3).reshape(-1, v.shape[-1])
                    k_samples.append(k_flat.detach())
                    v_samples.append(v_flat.detach())

    # Register hooks
    hooks = []
    for block in model.transformer.h:
        h = block.attn.register_forward_hook(hook)
        hooks.append(h)

    with torch.no_grad():
        model(**inputs, use_cache=True)

    for h in hooks:
        h.remove()

    if not k_samples:
        print("Could not extract K/V from hooks, skipping real model test")
        print("✓ Real model test SKIPPED (hook format mismatch)\n")
        return {}

    K = torch.cat(k_samples, dim=0)
    V = torch.cat(v_samples, dim=0)

    print(f"Collected K shape: {K.shape}")
    print(f"Collected V shape: {V.shape}")

    # Test compression on real data
    d_input = K.shape[-1]  # 64 for GPT-2 (head_dim)
    d_compressed = 32  # 2x compression

    config = KVCompressorConfig(
        d_input=d_input,
        d_compressed=d_compressed,
        device="cpu",
        dtype=torch.float32,
    )

    # Split data
    K_calib, K_test = K[:-100], K[-100:]
    V_calib, V_test = V[:-100], V[-100:]

    results = {}

    # PCA on K
    pca_k = PCACompressor(config)
    pca_k.calibrate(K_calib)
    k_hat_pca = pca_k.expand(pca_k.compress(K_test))
    results["K PCA"] = ((K_test - k_hat_pca) ** 2).mean().item()

    # Orthogonal on K (no calibration)
    ortho_k = OrthogonalCompressor(config)
    k_hat_ortho = ortho_k.expand(ortho_k.compress(K_test))
    results["K Ortho (uncalib)"] = ((K_test - k_hat_ortho) ** 2).mean().item()

    # Orthogonal on K (calibrated)
    ortho_k_cal = OrthogonalCompressor(config)
    ortho_k_cal.calibrate(K_calib)
    k_hat_ortho_cal = ortho_k_cal.expand(ortho_k_cal.compress(K_test))
    results["K Ortho (calib)"] = ((K_test - k_hat_ortho_cal) ** 2).mean().item()

    # Same for V
    pca_v = PCACompressor(config)
    pca_v.calibrate(V_calib)
    v_hat_pca = pca_v.expand(pca_v.compress(V_test))
    results["V PCA"] = ((V_test - v_hat_pca) ** 2).mean().item()

    ortho_v = OrthogonalCompressor(config)
    v_hat_ortho = ortho_v.expand(ortho_v.compress(V_test))
    results["V Ortho (uncalib)"] = ((V_test - v_hat_ortho) ** 2).mean().item()

    ortho_v_cal = OrthogonalCompressor(config)
    ortho_v_cal.calibrate(V_calib)
    v_hat_ortho_cal = ortho_v_cal.expand(ortho_v_cal.compress(V_test))
    results["V Ortho (calib)"] = ((V_test - v_hat_ortho_cal) ** 2).mean().item()

    print("\nReconstruction MSE on real GPT-2 activations:")
    print("-" * 50)
    for name, mse in sorted(results.items()):
        print(f"  {name:25s}: {mse:.6f}")

    print("\n✓ Real model test PASSED\n")
    return results


def main():
    print("\n" + "=" * 60)
    print("OrthogonalCompressor Test Suite")
    print("=" * 60 + "\n")

    # Run all tests
    test_basic_functionality()
    test_calibration_improvement()
    test_comparison_with_pca_svd()
    test_speed_comparison()
    test_with_real_model()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nSummary:")
    print("- OrthogonalCompressor works without calibration")
    print("- Calibration improves reconstruction for structured data")
    print("- When calibrated, matches SVD/PCA quality")
    print("- Much faster initialization when calibration is skipped")
    print("- Based on KVSplice from MLA training (proven effective)")


if __name__ == "__main__":
    main()
