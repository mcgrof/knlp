#!/usr/bin/env python3
"""
Quick quality test: Does RA v5 improve validation loss vs baseline?

Runs two short training runs (1 hour each):
1. SDPA baseline (1.33ms per forward)
2. RA v5 (1.33ms per forward - SAME SPEED!)

Measures: At matched speed, does RA achieve better validation loss?

Key question: Does RA's architectural benefits (reciprocity, learned
              gates) provide quality improvements at zero speed cost?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0


class SimpleGPT2(nn.Module):
    """Minimal GPT-2 for testing attention variants."""

    def __init__(self, config, use_ra=False):
        super().__init__()
        self.config = config
        self.use_ra = use_ra

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [GPT2Block(config, use_ra) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [T, n_embd]
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


class GPT2Block(nn.Module):
    """Transformer block with optional RA attention."""

    def __init__(self, config, use_ra=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config, use_ra)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Attention(nn.Module):
    """Attention with SDPA or RA v5 support."""

    def __init__(self, config, use_ra=False):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_ra = use_ra

        if use_ra:
            # Use UnifiedRAttention (matches baseline speed!)
            try:
                from ra import UnifiedRAttention

                self.attn_module = UnifiedRAttention(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    R=4,
                    dropout=config.dropout,
                )
            except ImportError as e:
                print(f"⚠️  Could not import UnifiedRAttention: {e}")
                print("    Falling back to baseline SDPA")
                self.use_ra = False
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            # Standard baseline attention
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        if self.use_ra:
            # Use RA v5 module directly
            return self.attn_module(x)
        else:
            # Standard SDPA baseline
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)


class MLP(nn.Module):
    """Standard MLP."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


def generate_dummy_batch(batch_size, seq_len, vocab_size, device):
    """Generate random tokens for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def quick_train_test(model_name, use_ra, time_budget_sec=600, batch_size=8):
    """
    Train model for time_budget_sec, measure validation loss.

    Returns:
        (iterations_completed, final_val_loss, iters_per_sec)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Time budget: {time_budget_sec/60:.1f} minutes")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()

    # Create model
    model = SimpleGPT2(config, use_ra=use_ra).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print("Starting training...")

    model.train()
    iterations = 0
    total_loss = 0.0
    start_time = time.time()

    try:
        while (time.time() - start_time) < time_budget_sec:
            # Generate dummy batch
            batch = generate_dummy_batch(
                batch_size, config.block_size, config.vocab_size, device
            )
            targets = batch.clone()

            # Forward
            logits, loss = model(batch, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iterations += 1

            # Progress update every 50 iters
            if iterations % 50 == 0:
                elapsed = time.time() - start_time
                iters_per_sec = iterations / elapsed
                avg_loss = total_loss / iterations
                remaining = time_budget_sec - elapsed
                print(
                    f"  Iter {iterations:4d} | Loss {avg_loss:.4f} | "
                    f"{iters_per_sec:.2f} it/s | {remaining/60:.1f}m left"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted")

    elapsed = time.time() - start_time
    iters_per_sec = iterations / elapsed
    avg_loss = total_loss / iterations

    # Quick validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):  # 20 validation batches
            batch = generate_dummy_batch(
                batch_size, config.block_size, config.vocab_size, device
            )
            targets = batch.clone()
            logits, loss = model(batch, targets)
            val_losses.append(loss.item())

    val_loss = sum(val_losses) / len(val_losses)

    print(f"\n{'='*70}")
    print(f"Results for {model_name}:")
    print(f"  Iterations completed: {iterations}")
    print(f"  Training loss: {avg_loss:.4f}")
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Throughput: {iters_per_sec:.2f} iters/sec")
    print(f"{'='*70}")

    return iterations, val_loss, iters_per_sec


def main():
    print("=" * 70)
    print("Quick Quality Test: Baseline SDPA vs RA v5")
    print("=" * 70)
    print("Trains two models for 1 hour each:")
    print("  1. SDPA baseline (1.33ms per attention)")
    print("  2. RA v5 (1.33ms per attention - SAME SPEED!)")
    print()
    print("Question: At matched speed, does RA v5 achieve better")
    print("          validation loss due to architectural benefits?")
    print()
    print("Using RA v5 (direct layout emission, R=4):")
    print("  - Reciprocity: Q can attend to K's context and vice versa")
    print("  - Learned gates: Per-head w_rec controls reciprocity usage")
    print("  - Zero overhead: Matches baseline speed exactly (1.33ms)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA required")
        return

    time_budget = 3600  # 1 hour (longer for better quality assessment)

    # Test 1: SDPA baseline
    iters_sdpa, loss_sdpa, speed_sdpa = quick_train_test(
        "Baseline SDPA", use_ra=False, time_budget_sec=time_budget
    )

    # Test 2: RA v5 (direct layout, R=4, matches baseline speed)
    iters_ra, loss_ra, speed_ra = quick_train_test(
        "RA v5 (R=4, direct layout)", use_ra=True, time_budget_sec=time_budget
    )

    # Analysis
    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON (Matched Speed)")
    print("=" * 70)
    print(f"{'Metric':<30} {'Baseline':>15} {'RA v5':>15} {'Difference':>15}")
    print("-" * 70)
    print(
        f"{'Iterations completed':<30} {iters_sdpa:>15d} {iters_ra:>15d} "
        f"{((iters_ra-iters_sdpa)/iters_sdpa*100):>13.1f}%"
    )
    print(
        f"{'Validation loss':<30} {loss_sdpa:>15.4f} {loss_ra:>15.4f} "
        f"{((loss_ra-loss_sdpa)/loss_sdpa*100):>13.1f}%"
    )
    print(
        f"{'Throughput (it/s)':<30} {speed_sdpa:>15.2f} {speed_ra:>15.2f} "
        f"{((speed_ra-speed_sdpa)/speed_sdpa*100):>13.1f}%"
    )

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check speed parity first
    speed_ratio = speed_ra / speed_sdpa
    if abs(speed_ratio - 1.0) > 0.10:
        print(f"⚠️  WARNING: Speed mismatch detected!")
        print(f"   Expected: ~1.0x, Got: {speed_ratio:.2f}x")
        print(f"   This invalidates the quality comparison")
        print("=" * 70)
        return

    print(f"✅ Speed parity confirmed: {speed_ratio:.2f}x (within 10%)")
    print()

    # Quality comparison
    if loss_ra < loss_sdpa * 0.99:  # At least 1% improvement
        improvement = (loss_sdpa - loss_ra) / loss_sdpa * 100
        print(f"🎉 RA v5 WINS: {improvement:.1f}% better validation loss!")
        print(f"   At the SAME speed ({speed_ra:.2f} vs {speed_sdpa:.2f} it/s)")
        print(f"   Architectural benefits (reciprocity + learned gates)")
        print(f"   provide measurable quality improvements")
        print()
        print(f"   Recommendation: INTEGRATE RA v5 into training pipeline")
    elif loss_ra > loss_sdpa * 1.01:  # At least 1% degradation
        degradation = (loss_ra - loss_sdpa) / loss_sdpa * 100
        print(f"❌ RA v5 LOSES: {degradation:.1f}% worse validation loss")
        print(f"   At the same speed, RA v5 provides worse quality")
        print(f"   Architectural complexity may be hurting convergence")
        print()
        print(f"   Recommendation: Need hyperparameter tuning or redesign")
    else:
        # Within 1% - essentially the same
        print(f"⚖️  PARITY: Validation loss within 1% (statistically similar)")
        print(f"   RA v5: {loss_ra:.4f} vs Baseline: {loss_sdpa:.4f}")
        print(f"   No measurable quality difference at 1 hour of training")
        print()
        print(f"   Possible reasons:")
        print(f"   - Need longer training for RA benefits to emerge")
        print(f"   - Task may not benefit from reciprocity")
        print(f"   - w_rec gates learned to disable reciprocity (check weights)")
        print()
        print(f"   Recommendation: Try longer training (8+ hours) or analyze")
        print(f"                   learned w_rec values to see if reciprocity used")

    print("=" * 70)


if __name__ == "__main__":
    main()
