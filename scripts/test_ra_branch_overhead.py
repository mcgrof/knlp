#!/usr/bin/env python3
"""
Test if GPT2_RA slowdown is due to branching overhead.

Creates two versions from same checkpoint:
  1. Original: Runtime branching with sigmoid
  2. Binarized: Pre-committed decisions, no branching

Compares inference speed to isolate branching cost.

Usage:
    python scripts/test_ra_branch_overhead.py [checkpoint.pt]
    python scripts/test_ra_branch_overhead.py  # Uses random init if no checkpoint
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.model import GPTConfig
from ra import GPT2_RA


class GPT2_RA_NoBranch(GPT2_RA):
    """
    GPT2_RA modified to eliminate all runtime branching.

    Inherits from GPT2_RA but overrides attention forward to use
    pre-baked decisions instead of runtime sigmoid + branching.
    """

    def __init__(self, config):
        super().__init__(config)
        # Will store per-layer SDPA argument order after binarization
        self.sdpa_arg_order = None

    def binarize_alternation(self):
        """
        Commit to hard decisions based on learned alternation logits.
        Creates lookup table for zero-overhead inference.
        """
        with torch.no_grad():
            # Get decisions from learned logits
            probs = torch.sigmoid(self.alternation_logits).cpu()
            decisions = (probs > 0.5).tolist()

            # Store argument order: (q_idx, k_idx) where 0=q, 1=k from stack
            # Standard: (0, 1) means stack[0]=q, stack[1]=k -> SDPA(q, k, v)
            # Reciprocal: (1, 0) means stack[1]=k, stack[0]=q -> SDPA(k, q, v)
            self.sdpa_arg_order = [
                (1, 0) if use_recip else (0, 1)
                for use_recip in decisions
            ]

            num_recip = sum(decisions)
            print(f"\nBinarized alternation decisions:")
            print(f"  Reciprocal: {num_recip}/{len(decisions)} layers")
            print(f"  Standard:   {len(decisions) - num_recip}/{len(decisions)} layers")
            print(f"  Per layer: {['R' if d else 'S' for d in decisions]}")

    def forward(self, idx, targets=None):
        """Override to use binarized attention."""
        B, T = idx.size()

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks with NO-BRANCH attention
        for layer_idx, block in enumerate(self.blocks):
            # Pre-norm
            x_norm = block.ln_1(x)

            # NO-BRANCH ATTENTION
            attn_out = self._forward_attention_nobranch(x_norm, layer_idx)

            # Residual
            x = x + attn_out

            # MLP
            x = x + block.mlp(block.ln_2(x))

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    def _forward_attention_nobranch(self, x, layer_idx):
        """
        Attention forward with ZERO branching overhead.

        Uses pre-baked sdpa_arg_order to select q/k arguments via indexing.
        No sigmoid evaluation, no conditionals.
        """
        block = self.blocks[layer_idx]
        attn = block.attn

        B, T, C = x.size()

        # QKV projection
        q, k, v = attn.c_attn(x).split(attn.config.n_embd, dim=2)
        q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)  # [B, H, T, D]
        v = v.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)  # [B, H, T, D]

        # NO BRANCHING: Use indexing to select arguments
        if self.sdpa_arg_order is None:
            raise RuntimeError("Must call binarize_alternation() before inference")

        q_idx, k_idx = self.sdpa_arg_order[layer_idx]

        # Stack and select (no conditional)
        qk_stack = torch.stack([q, k], dim=0)  # [2, B, H, T, D]
        q_arg = qk_stack[q_idx]  # Either q or k
        k_arg = qk_stack[k_idx]  # Either k or q

        # Single SDPA call (no branching)
        y = F.scaled_dot_product_attention(
            q_arg,
            k_arg,
            v,
            dropout_p=attn.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = attn.resid_dropout(attn.c_proj(y))

        return y


def benchmark_model(model, device, num_tokens=100, num_runs=5, warmup=2):
    """
    Benchmark inference speed.

    Args:
        model: Model to benchmark
        device: Device to run on
        num_tokens: Tokens to generate per run
        num_runs: Number of runs to average
        warmup: Warmup runs (discarded)

    Returns:
        Average tokens/second
    """
    model.eval()
    batch_size = 1
    prompt_len = 10

    # Create prompt
    prompt = torch.randint(0, 50257, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            current_seq = prompt.clone()
            for _ in range(min(num_tokens, 20)):
                logits, _ = model(current_seq)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                current_seq = torch.cat([current_seq, next_token], dim=1)
                if current_seq.size(1) > model.config.block_size:
                    current_seq = current_seq[:, -model.config.block_size:]

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for run in range(num_runs):
        current_seq = prompt.clone()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_tokens):
                logits, _ = model(current_seq)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                current_seq = torch.cat([current_seq, next_token], dim=1)
                if current_seq.size(1) > model.config.block_size:
                    current_seq = current_seq[:, -model.config.block_size:]

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        tok_per_sec = num_tokens / elapsed
        print(f"  Run {run+1}/{num_runs}: {elapsed:.3f}s ({tok_per_sec:.1f} tok/s)")

    # Statistics
    avg_time = sum(times) / len(times)
    avg_tok_per_sec = num_tokens / avg_time

    return avg_tok_per_sec


def main():
    print("=" * 80)
    print("GPT2_RA Branching Overhead Test")
    print("=" * 80)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load or create model
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Create config from checkpoint or use default
        config = GPTConfig.from_name("gpt2")
        config.block_size = 1024

        # Load model
        model_original = GPT2_RA(config).to(device)
        model_original.load_state_dict(state_dict, strict=False)
        print(f"Loaded model: {model_original.get_num_params() / 1e6:.2f}M parameters")
    else:
        print("\nCreating random initialized model (no checkpoint provided)")
        config = GPTConfig.from_name("gpt2")
        config.block_size = 1024
        model_original = GPT2_RA(config).to(device)
        print(f"Model: {model_original.get_num_params() / 1e6:.2f}M parameters")

    # Show learned alternation
    print("\nLearned alternation probabilities:")
    with torch.no_grad():
        probs = torch.sigmoid(model_original.alternation_logits).cpu()
        decisions = (probs > 0.5).tolist()
        print(f"  Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
        print(f"  Decisions (>0.5): {['Reciprocal' if d else 'Standard' for d in decisions]}")

    # Create no-branch version
    print("\nCreating no-branch variant...")
    model_nobranch = GPT2_RA_NoBranch(config).to(device)
    model_nobranch.load_state_dict(model_original.state_dict(), strict=False)
    model_nobranch.binarize_alternation()

    # Test parameters
    num_tokens = 100
    num_runs = 5

    print("\n" + "=" * 80)
    print("BENCHMARK: Original (with branching)")
    print("=" * 80)
    speed_original = benchmark_model(model_original, device, num_tokens, num_runs)

    print("\n" + "=" * 80)
    print("BENCHMARK: No-Branch (binarized)")
    print("=" * 80)
    speed_nobranch = benchmark_model(model_nobranch, device, num_tokens, num_runs)

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nOriginal (branching):    {speed_original:.1f} tok/s")
    print(f"No-Branch (binarized):   {speed_nobranch:.1f} tok/s")

    speedup = speed_nobranch / speed_original
    overhead_pct = (speedup - 1) * 100

    print(f"\nSpeedup: {speedup:.3f}x")
    if speedup > 1.0:
        print(f"Branching overhead: {overhead_pct:.1f}%")
        print("\n✓ HYPOTHESIS CONFIRMED: Branching causes slowdown")
    else:
        print(f"\n✗ HYPOTHESIS REJECTED: Branching not the bottleneck")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)
    if speedup > 1.05:
        print("Significant speedup from eliminating branching.")
        print("The runtime sigmoid + if/else overhead is measurable.")
        print("Production models should use binarized inference.")
    elif speedup > 1.01:
        print("Small speedup from eliminating branching.")
        print("Branching has minor overhead but not the main bottleneck.")
    else:
        print("No speedup from eliminating branching.")
        print("The slowdown must come from something else:")
        print("  - Transpose operations themselves")
        print("  - Different kernel selection in SDPA")
        print("  - Memory access patterns")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
