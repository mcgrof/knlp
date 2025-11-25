#!/usr/bin/env python3
"""
Quick timing comparison of V0 (baseline) vs V1 (RA).

Shows forward, backward, and total iteration times.
"""

import os
import sys
import time
import torch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from gpt2.model import GPT2, GPTConfig


def time_step(step_name: str, num_iters: int = 50):
    """Time forward+backward for a specific step."""
    print(f"\n{'='*80}")
    print(f"Timing {step_name}")
    print(f"{'='*80}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    batch_size = 8
    block_size = 1024

    # Create model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    model = GPT(config)
    model = model.to(device)
    model.train()

    # Patch with RA if V1
    if step_name == "V1":
        from ra_patch import patch_gpt2_with_ra_v5

        model = patch_gpt2_with_ra_v5(model, R=4, use_self_restart=False)
        print("✓ Patched with RA (R=4)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

    # Dummy data
    x = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)

    # Warmup
    print("Warming up (10 iterations)...")
    for _ in range(10):
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timing
    print(f"Timing {num_iters} iterations...")
    forward_times = []
    backward_times = []
    total_times = []

    for i in range(num_iters):
        t_start = time.perf_counter()

        # Forward
        t_fwd_start = time.perf_counter()
        logits, loss = model(x, y)
        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        forward_times.append((t_fwd_end - t_fwd_start) * 1000)

        # Backward
        t_bwd_start = time.perf_counter()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()
        backward_times.append((t_bwd_end - t_bwd_start) * 1000)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        total_times.append((t_end - t_start) * 1000)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_iters} iterations")

    # Statistics (skip first 5 as warmup)
    fwd_avg = sum(forward_times[5:]) / len(forward_times[5:])
    bwd_avg = sum(backward_times[5:]) / len(backward_times[5:])
    total_avg = sum(total_times[5:]) / len(total_times[5:])
    opt_avg = total_avg - fwd_avg - bwd_avg

    print(f"\n{'='*80}")
    print(f"Results for {step_name} (averaged over {len(forward_times[5:])} iters)")
    print(f"{'='*80}")
    print(f"  Forward:   {fwd_avg:7.2f} ms")
    print(f"  Backward:  {bwd_avg:7.2f} ms")
    print(f"  Optimizer: {opt_avg:7.2f} ms")
    print(f"  Total:     {total_avg:7.2f} ms")
    print(f"{'='*80}")

    return {
        "forward": fwd_avg,
        "backward": bwd_avg,
        "optimizer": opt_avg,
        "total": total_avg,
    }


def main():
    print("\n" + "=" * 80)
    print("V0 vs V1 Timing Comparison")
    print("=" * 80)

    # Time V0 (baseline)
    v0_times = time_step("V0", num_iters=50)

    # Time V1 (RA)
    v1_times = time_step("V1", num_iters=50)

    # Comparison
    print(f"\n{'='*80}")
    print("V1 vs V0 Comparison")
    print(f"{'='*80}")
    print(f"                V0         V1      Difference    Change")
    print(f"{'='*80}")

    for phase in ["forward", "backward", "optimizer", "total"]:
        v0 = v0_times[phase]
        v1 = v1_times[phase]
        diff = v1 - v0
        pct = ((v1 / v0) - 1) * 100
        sign = "+" if diff > 0 else ""
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(
            f"  {phase.capitalize():10s}  {v0:7.2f}ms  {v1:7.2f}ms  {sign}{diff:7.2f}ms  {sign}{pct:+6.2f}% {arrow}"
        )

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
