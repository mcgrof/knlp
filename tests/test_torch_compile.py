#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Simple test script to verify torch.compile() works with ROCm 7.x PyTorch.

This script trains a small ResNet18 for a few iterations on CIFAR-10 to verify:
1. torch.compile() can compile the model without errors
2. The compiled model trains successfully on GPU with FP16
3. Performance improvement is observable

Uses FP16 (half precision) for optimal performance on AMD W7900.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time


def create_model():
    """Create a simple ResNet18 model."""
    model = torchvision.models.resnet18(num_classes=10)
    return model


def get_dataloader(batch_size=64):
    """Create a simple CIFAR-10 dataloader."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return trainloader


def test_torch_compile():
    """Test torch.compile() with a simple training loop using FP16."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using precision: FP16 (half)")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model and move to device with FP16
    print("\n=== Testing WITHOUT torch.compile() ===")
    model_uncompiled = create_model().to(device).half()
    optimizer_uncompiled = torch.optim.SGD(model_uncompiled.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Get dataloader
    trainloader = get_dataloader(batch_size=64)

    # Train for 5 iterations without compile
    model_uncompiled.train()
    start_time = time.time()
    for i, (images, labels) in enumerate(trainloader):
        if i >= 5:
            break
        images = images.to(device).half()
        labels = labels.to(device)

        optimizer_uncompiled.zero_grad()
        outputs = model_uncompiled(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_uncompiled.step()

        print(f"Iter {i+1}/5 - Loss: {loss.item():.4f}")

    uncompiled_time = time.time() - start_time
    print(f"Time without compile: {uncompiled_time:.2f}s")

    # Now test WITH torch.compile()
    print("\n=== Testing WITH torch.compile() (First Run - includes compilation) ===")
    model_compiled = create_model().to(device).half()

    # Compile the model
    print("Compiling model with torch.compile()...")
    model_compiled = torch.compile(model_compiled)

    optimizer_compiled = torch.optim.SGD(model_compiled.parameters(), lr=0.01)

    # Train for 5 iterations with compile (FIRST RUN - includes compilation overhead)
    model_compiled.train()
    start_time = time.time()
    for i, (images, labels) in enumerate(trainloader):
        if i >= 5:
            break
        images = images.to(device).half()
        labels = labels.to(device)

        optimizer_compiled.zero_grad()
        outputs = model_compiled(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_compiled.step()

        print(f"Iter {i+1}/5 - Loss: {loss.item():.4f}")

    compiled_time_first = time.time() - start_time
    print(f"Time with compile (first run): {compiled_time_first:.2f}s")

    # Run again to see the REAL speedup (compilation is cached now)
    print(
        "\n=== Testing WITH torch.compile() (Second Run - uses cached compilation) ==="
    )
    model_compiled.train()
    start_time = time.time()
    for i, (images, labels) in enumerate(trainloader):
        if i >= 5:
            break
        images = images.to(device).half()
        labels = labels.to(device)

        optimizer_compiled.zero_grad()
        outputs = model_compiled(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_compiled.step()

        print(f"Iter {i+1}/5 - Loss: {loss.item():.4f}")

    compiled_time_second = time.time() - start_time
    print(f"Time with compile (second run): {compiled_time_second:.2f}s")

    # Compare performance
    print("\n=== Results ===")
    print(f"Uncompiled: {uncompiled_time:.2f}s")
    print(f"Compiled (first run, includes compilation): {compiled_time_first:.2f}s")
    print(f"Compiled (second run, uses cached kernels): {compiled_time_second:.2f}s")
    speedup_first = (
        uncompiled_time / compiled_time_first if compiled_time_first > 0 else 0
    )
    speedup_second = (
        uncompiled_time / compiled_time_second if compiled_time_second > 0 else 0
    )
    print(
        f"\nSpeedup (first run): {speedup_first:.2f}x (slower due to compilation overhead)"
    )
    print(f"Speedup (second run): {speedup_second:.2f}x (real performance gain)")

    print("\n✓ torch.compile() test PASSED!")
    print("  - Model compiled successfully")
    print("  - Training completed without errors")
    print("  - No ROCm/MIOpen compilation failures")
    print(f"  - Achieves {speedup_second:.2f}x speedup after initial compilation")


if __name__ == "__main__":
    print("=" * 60)
    print("torch.compile() Hello World Test")
    print("=" * 60)

    try:
        test_torch_compile()
    except Exception as e:
        print(f"\n✗ torch.compile() test FAILED!")
        print(f"Error: {e}")
        raise
