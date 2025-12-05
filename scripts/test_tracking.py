#!/usr/bin/env python3
"""Test script for verifying experiment tracking integrations."""

import subprocess
import sys
import time

def test_trackio():
    """Test Trackio integration."""
    print("Testing Trackio integration...")

    # Check if trackio is installed
    try:
        import trackio
        print("✓ Trackio is installed")
    except ImportError:
        print("✗ Trackio not installed. Install with: pip install trackio")
        return False

    # Run a minimal training with trackio
    cmd = [
        sys.executable, "gpt2/train.py",
        "--dataset", "shakespeare",
        "--max-iters", "10",
        "--eval-interval", "5",
        "--log-interval", "5",
        "--tracker", "trackio",
        "--tracker-project", "test-project",
        "--tracker-run-name", "test-run",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Trackio test completed successfully")
        print("  Run 'trackio show' to view the dashboard")
        return True
    else:
        print(f"✗ Trackio test failed with return code {result.returncode}")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        return False

def test_wandb():
    """Test WandB integration."""
    print("\nTesting WandB integration...")

    # Check if wandb is installed
    try:
        import wandb
        print("✓ WandB is installed")
    except ImportError:
        print("✗ WandB not installed. Install with: pip install wandb")
        return False

    # Run a minimal training with wandb (offline mode to avoid login)
    import os
    os.environ["WANDB_MODE"] = "offline"

    cmd = [
        sys.executable, "gpt2/train.py",
        "--dataset", "shakespeare",
        "--max-iters", "10",
        "--eval-interval", "5",
        "--log-interval", "5",
        "--tracker", "wandb",
        "--tracker-project", "test-project",
        "--tracker-run-name", "test-run",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ WandB test completed successfully")
        print("  Note: Running in offline mode. Use 'wandb sync' to upload runs.")
        return True
    else:
        print(f"✗ WandB test failed with return code {result.returncode}")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        return False

def test_none():
    """Test with no tracking (baseline)."""
    print("\nTesting without tracking (baseline)...")

    cmd = [
        sys.executable, "gpt2/train.py",
        "--dataset", "shakespeare",
        "--max-iters", "10",
        "--eval-interval", "5",
        "--log-interval", "5",
        "--tracker", "none",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Baseline test completed successfully")
        return True
    else:
        print(f"✗ Baseline test failed with return code {result.returncode}")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Experiment Tracking Integrations")
    print("=" * 60)

    # Test baseline first
    baseline_ok = test_none()

    if not baseline_ok:
        print("\n⚠️  Baseline test failed. Please fix training script issues first.")
        sys.exit(1)

    # Test trackio
    trackio_ok = test_trackio()

    # Test wandb
    wandb_ok = test_wandb()

    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Baseline: {'✓' if baseline_ok else '✗'}")
    print(f"  Trackio:  {'✓' if trackio_ok else '✗'}")
    print(f"  WandB:    {'✓' if wandb_ok else '✗'}")
    print("=" * 60)
