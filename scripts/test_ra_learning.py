#!/usr/bin/env python3
"""
Microbenchmark to verify GPT2_RA_Learned alternation actually learns.

Tests whether alternation_logits receive gradients and can be optimized.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.model import GPTConfig
from ra import GPT2_RA_Learned


def test_gradient_flow():
    """Test 1: Verify gradients flow to alternation_logits."""
    print("=" * 70)
    print("TEST 1: Gradient Flow")
    print("=" * 70)

    config = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=1000,
        dropout=0.0,
    )

    model = GPT2_RA_Learned(config)
    model.train()

    # Random input
    x = torch.randint(0, 1000, (2, 32))
    targets = torch.randint(0, 1000, (2, 32))

    # Forward pass
    logits, loss = model(x, targets)

    # Check if alternation_logits requires grad
    print(f"alternation_logits requires_grad: {model.alternation_logits.requires_grad}")
    print(f"Initial alternation_logits: {model.alternation_logits.detach().numpy()}")

    # Backward pass
    loss.backward()

    # Check gradients
    if model.alternation_logits.grad is not None:
        print(f"✓ Gradients exist: {model.alternation_logits.grad.numpy()}")
        print(f"  Gradient norm: {model.alternation_logits.grad.norm().item():.6f}")
    else:
        print("✗ NO GRADIENTS - alternation_logits.grad is None")
        return False

    return True


def test_learning_with_penalty():
    """Test 2: Can alternation_logits learn to minimize a penalty?"""
    print("\n" + "=" * 70)
    print("TEST 2: Learning with Direct Penalty")
    print("=" * 70)
    print("Goal: Learn to prefer reciprocal attention (push logits positive)")
    print()

    config = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=1000,
        dropout=0.0,
    )

    model = GPT2_RA_Learned(config)
    model.train()

    # Optimizer ONLY for alternation_logits
    optimizer = torch.optim.Adam([model.alternation_logits], lr=0.1)

    # Random data
    x = torch.randint(0, 1000, (4, 32))
    targets = torch.randint(0, 1000, (4, 32))

    print(f"Initial logits: {model.alternation_logits.detach().numpy()}")

    # Train for 50 steps with penalty favoring reciprocal (positive logits)
    for step in range(50):
        optimizer.zero_grad()

        # Forward pass
        logits, ce_loss = model(x, targets)

        # Penalty: want high probability of reciprocal (sigmoid -> 1)
        # Loss = -sum(sigmoid(logits)) encourages positive logits
        p_recip = torch.sigmoid(model.alternation_logits)
        penalty = -p_recip.sum()  # Want to maximize sum, so minimize negative

        # Total loss
        loss = ce_loss + penalty

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"Step {step:2d}: logits={model.alternation_logits.detach().numpy()}, "
                f"p_recip={p_recip.detach().numpy()}, penalty={penalty.item():.3f}"
            )

    final_logits = model.alternation_logits.detach().numpy()
    final_p = torch.sigmoid(model.alternation_logits).detach().numpy()

    print(f"\nFinal logits: {final_logits}")
    print(f"Final p_recip: {final_p}")

    # Check if logits increased (learned to prefer reciprocal)
    if (final_logits > 1.0).all():
        print("✓ SUCCESS: Logits learned to increase (prefer reciprocal)")
        return True
    else:
        print("✗ FAILURE: Logits did not learn")
        return False


def test_learning_opposite_penalty():
    """Test 3: Can alternation_logits learn the opposite direction?"""
    print("\n" + "=" * 70)
    print("TEST 3: Learning Opposite Direction")
    print("=" * 70)
    print("Goal: Learn to prefer standard attention (push logits negative)")
    print()

    config = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=1000,
        dropout=0.0,
    )

    model = GPT2_RA_Learned(config)
    model.train()

    # Optimizer ONLY for alternation_logits
    optimizer = torch.optim.Adam([model.alternation_logits], lr=0.1)

    # Random data
    x = torch.randint(0, 1000, (4, 32))
    targets = torch.randint(0, 1000, (4, 32))

    print(f"Initial logits: {model.alternation_logits.detach().numpy()}")

    # Train for 50 steps with penalty favoring standard (negative logits)
    for step in range(50):
        optimizer.zero_grad()

        # Forward pass
        logits, ce_loss = model(x, targets)

        # Penalty: want low probability of reciprocal (sigmoid -> 0)
        # Loss = sum(sigmoid(logits)) encourages negative logits
        p_recip = torch.sigmoid(model.alternation_logits)
        penalty = p_recip.sum()  # Want to minimize sum

        # Total loss
        loss = ce_loss + penalty

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"Step {step:2d}: logits={model.alternation_logits.detach().numpy()}, "
                f"p_recip={p_recip.detach().numpy()}, penalty={penalty.item():.3f}"
            )

    final_logits = model.alternation_logits.detach().numpy()
    final_p = torch.sigmoid(model.alternation_logits).detach().numpy()

    print(f"\nFinal logits: {final_logits}")
    print(f"Final p_recip: {final_p}")

    # Check if logits decreased (learned to prefer standard)
    if (final_logits < -1.0).all():
        print("✓ SUCCESS: Logits learned to decrease (prefer standard)")
        return True
    else:
        print("✗ FAILURE: Logits did not learn")
        return False


def test_layer_specific_learning():
    """Test 4: Can different layers learn different preferences?"""
    print("\n" + "=" * 70)
    print("TEST 4: Layer-Specific Learning")
    print("=" * 70)
    print("Goal: First 2 layers standard, last 2 layers reciprocal")
    print()

    config = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=1000,
        dropout=0.0,
    )

    model = GPT2_RA_Learned(config)
    model.train()

    # Optimizer ONLY for alternation_logits
    optimizer = torch.optim.Adam([model.alternation_logits], lr=0.1)

    # Random data
    x = torch.randint(0, 1000, (4, 32))
    targets = torch.randint(0, 1000, (4, 32))

    print(f"Initial logits: {model.alternation_logits.detach().numpy()}")

    # Train for 100 steps with layer-specific penalty
    for step in range(100):
        optimizer.zero_grad()

        # Forward pass
        logits, ce_loss = model(x, targets)

        # Penalty: layers 0,1 should be standard (low sigmoid)
        #          layers 2,3 should be reciprocal (high sigmoid)
        p_recip = torch.sigmoid(model.alternation_logits)

        penalty = (
            p_recip[0]
            + p_recip[1]  # Want these low (standard)
            - p_recip[2]
            - p_recip[3]  # Want these high (reciprocal)
        )

        # Total loss
        loss = ce_loss + penalty

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"Step {step:3d}: logits={model.alternation_logits.detach().numpy()}, "
                f"p_recip={p_recip.detach().numpy()}"
            )

    final_logits = model.alternation_logits.detach().numpy()
    final_p = torch.sigmoid(model.alternation_logits).detach().numpy()

    print(f"\nFinal logits: {final_logits}")
    print(f"Final p_recip: {final_p}")

    # Check if pattern learned correctly
    early_standard = (final_p[0] < 0.3) and (final_p[1] < 0.3)
    late_reciprocal = (final_p[2] > 0.7) and (final_p[3] > 0.7)

    if early_standard and late_reciprocal:
        print("✓ SUCCESS: Learned layer-specific pattern correctly")
        return True
    else:
        print("✗ FAILURE: Did not learn correct pattern")
        print(f"  Early layers standard? {early_standard}")
        print(f"  Late layers reciprocal? {late_reciprocal}")
        return False


def test_branching_breaks_gradients():
    """Test 5: Demonstrate that Python if-statement breaks gradients."""
    print("\n" + "=" * 70)
    print("TEST 5: Why Branching Breaks Gradients")
    print("=" * 70)
    print()

    # Simulate the alternation mechanism
    logit = torch.tensor([0.0], requires_grad=True)

    print("Scenario 1: Differentiable blend (both paths computed)")
    print("-" * 70)

    # Differentiable approach
    p = torch.sigmoid(logit)
    use_recip = (p > 0.5).float()
    use_recip = use_recip - p.detach() + p  # Straight-through

    # Compute BOTH paths
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    # Blend results (differentiable)
    result = use_recip * a + (1 - use_recip) * b
    loss = result.sum()

    loss.backward()
    print(f"Logit gradient: {logit.grad}")
    print(f"✓ Gradients flow through blend operation")

    print("\nScenario 2: Python if-statement (only one path)")
    print("-" * 70)

    # Reset
    logit = torch.tensor([0.0], requires_grad=True)
    p = torch.sigmoid(logit)
    use_recip = (p > 0.5).float()
    use_recip = use_recip - p.detach() + p  # Straight-through

    # Python if-statement (THIS IS WHAT GPT2_RA_Learned DOES)
    if use_recip.item() > 0.5:
        result = a.sum()  # Reciprocal path
    else:
        result = b.sum()  # Standard path

    # Try to backward
    try:
        result.backward()
        print(f"Logit gradient: {logit.grad}")
        if logit.grad is None or logit.grad.abs().sum() == 0:
            print(f"✗ No gradients - computation graph is disconnected")
    except RuntimeError as e:
        print(f"✗ Backward failed: {e}")
        print(f"✗ Tensor disconnected from input - no grad_fn")

    print("\nConclusion:")
    print("GPT2_RA_Learned uses Python if-statement (line 1329)")
    print("This disconnects alternation_logits from CE loss")
    print("Gradients only flow when explicit penalty is added")

    return True


def main():
    print("\n" + "=" * 70)
    print("GPT2_RA_Learned Alternation Learning Test Suite")
    print("=" * 70)
    print()

    results = []

    # Test 1: Basic gradient flow
    results.append(("Gradient Flow from CE Loss", test_gradient_flow()))

    # Test 2: Learn to prefer reciprocal
    results.append(("Learn Reciprocal Preference", test_learning_with_penalty()))

    # Test 3: Learn to prefer standard
    results.append(("Learn Standard Preference", test_learning_opposite_penalty()))

    # Test 4: Layer-specific learning
    results.append(("Layer-Specific Learning", test_layer_specific_learning()))

    # Test 5: Demonstrate branching issue
    results.append(("Branching Gradient Analysis", test_branching_breaks_gradients()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("✓ All tests passed - alternation learning works!")
    else:
        print("✗ Some tests failed - alternation learning is broken!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
