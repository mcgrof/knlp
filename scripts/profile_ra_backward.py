#!/usr/bin/env python3
"""
Profile V0 (baseline) vs V1 (RA) to identify backward pass overhead.

Usage:
    python3 scripts/profile_ra_backward.py --step V0
    python3 scripts/profile_ra_backward.py --step V1

    # Compare results:
    tensorboard --logdir=./profiling_results
"""

import os
import sys
import argparse
import torch
import torch.profiler

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from gpt2.model import GPT2, GPTConfig


def profile_step(step_name: str, num_iters: int = 20):
    """Profile forward+backward for a specific ablation step."""
    print(f"Profiling step {step_name}...")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 4  # Smaller batch for faster profiling
    block_size = 512  # Smaller context for faster profiling

    # Create model based on step
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )
    print(f"Model config: batch={batch_size}, block_size={block_size}")

    model = GPT2(config)
    model = model.to(device)
    model.train()

    # Patch with RA if V1
    if step_name == "V1":
        from ra_patch import patch_gpt2_with_ra_v5

        model = patch_gpt2_with_ra_v5(
            model,
            R=4,
            use_self_restart=False,
        )
        print("Patched with RA (R=4)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

    # Get dummy data
    x = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Profile
    output_dir = f"./profiling_results/{step_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Profiling {num_iters} iterations...")
    print(f"Results will be saved to: {output_dir}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,  # Skip first 2 iterations
            warmup=2,  # Warmup for 2 iterations
            active=6,  # Profile 6 iterations
            repeat=1,  # Only do this once
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for step in range(num_iters):
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            prof.step()

    print(f"✓ Profiling complete for {step_name}")
    print(f"  View results: tensorboard --logdir={output_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"Summary for {step_name}")
    print("=" * 80)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Profile RA backward pass")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["V0", "V1"],
        help="Ablation step to profile (V0=baseline, V1=RA)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of iterations to profile (default: 20)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, profiling on CPU")
        print("Results will be less informative without GPU timing")

    profile_step(args.step, args.iters)

    print("\n" + "=" * 80)
    print("To compare V0 vs V1:")
    print("  1. Run: python3 scripts/profile_ra_backward.py --step V0")
    print("  2. Run: python3 scripts/profile_ra_backward.py --step V1")
    print("  3. View: tensorboard --logdir=./profiling_results")
    print("  4. Navigate to 'TRACE' tab to see timeline comparison")
    print("=" * 80)


if __name__ == "__main__":
    main()
