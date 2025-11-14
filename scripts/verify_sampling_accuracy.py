#!/usr/bin/env python3
"""Verify that sampling produces acceptable thresholds."""

import torch
import numpy as np

def test_sampling_accuracy(size=50_000_000, sparsity=0.5, num_trials=10):
    """Test how close sampling threshold is to exact kthvalue."""

    device = torch.device("cuda:0")

    print(f"Testing sampling accuracy ({size/1e6:.1f}M elements, {num_trials} trials)")
    print("=" * 60)

    errors = []

    for trial in range(num_trials):
        # Random importance scores
        data = torch.rand(size, device=device, dtype=torch.bfloat16)
        k = int(sparsity * size)

        # Exact threshold
        exact_threshold = torch.kthvalue(data.float(), k).values.item()

        # Approximate threshold (2% sample)
        sample_size = max(1, int(size * 0.02))
        idx = torch.randint(0, size, (sample_size,), device=device)
        sample = data[idx].float()
        k_sample = max(1, int(k * (sample_size / size)))
        approx_threshold = torch.kthvalue(sample, k_sample).values.item()

        # Compute error
        rel_error = abs(approx_threshold - exact_threshold) / exact_threshold
        errors.append(rel_error)

        print(f"Trial {trial + 1}: exact={exact_threshold:.6f}, "
              f"approx={approx_threshold:.6f}, error={rel_error*100:.3f}%")

        del data, sample, idx

    print("=" * 60)
    print(f"Mean relative error: {np.mean(errors)*100:.3f}%")
    print(f"Max relative error:  {np.max(errors)*100:.3f}%")
    print(f"Std relative error:  {np.std(errors)*100:.3f}%")
    print("=" * 60)

    # Test sparsity difference
    print("\nTesting actual sparsity achieved:")
    data = torch.rand(size, device=device, dtype=torch.bfloat16)
    k = int(sparsity * size)

    # Exact
    exact_threshold = torch.kthvalue(data.float(), k).values
    exact_sparsity = (data.float() <= exact_threshold).float().mean().item()

    # Approximate
    sample_size = max(1, int(size * 0.02))
    idx = torch.randint(0, size, (sample_size,), device=device)
    sample = data[idx].float()
    k_sample = max(1, int(k * (sample_size / size)))
    approx_threshold = torch.kthvalue(sample, k_sample).values
    approx_sparsity = (data.float() <= approx_threshold).float().mean().item()

    print(f"Target sparsity: {sparsity:.1%}")
    print(f"Exact sparsity:  {exact_sparsity:.4%}")
    print(f"Approx sparsity: {approx_sparsity:.4%}")
    print(f"Difference:      {abs(approx_sparsity - exact_sparsity)*100:.3f} percentage points")
    print("=" * 60)

    if abs(approx_sparsity - exact_sparsity) < 0.01:  # Within 1 percentage point
        print("✅ Sampling is accurate enough for pruning!")
    else:
        print("⚠️  Sampling may introduce noticeable sparsity drift")

if __name__ == "__main__":
    test_sampling_accuracy()
