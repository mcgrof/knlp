#!/usr/bin/env python3
"""
Quick quality test: Does RA actually improve validation loss?

Runs two short training runs (10 minutes each):
1. SDPA baseline (fast)
2. RA Triton (2.45x slower)

Measures: How much validation loss does each achieve in 10 minutes?

Key question: Does RA's potential quality benefit offset its speed penalty?
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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GPT2Block(config, use_ra) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
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
    """Attention with SDPA or RA support."""

    def __init__(self, config, use_ra=False):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_ra = use_ra

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if use_ra:
            # RA-specific parameters
            self.d_bias = nn.Parameter(torch.zeros(config.n_head, config.block_size))
            self.w_std = nn.Parameter(torch.ones(config.n_head) * 0.5)
            self.w_rec = nn.Parameter(torch.ones(config.n_head) * 0.3)
            self.w_disc = nn.Parameter(torch.ones(config.n_head) * 0.2)

    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_ra:
            # Use RA Triton kernel
            try:
                from triton_ra_attention import triton_ra_attention
                d_bias = self.d_bias[:, :T].unsqueeze(0).expand(B, -1, -1)  # [B, H, T]
                w_std = self.w_std.unsqueeze(0).expand(B, -1)  # [B, H]
                w_rec = self.w_rec.unsqueeze(0).expand(B, -1)
                w_disc = self.w_disc.unsqueeze(0).expand(B, -1)

                y = triton_ra_attention(q, k, v, d_bias, w_std, w_rec, w_disc)
            except:
                # Fallback to SDPA if Triton fails
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Standard SDPA
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


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
            batch = generate_dummy_batch(batch_size, config.block_size, config.vocab_size, device)
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
                print(f"  Iter {iterations:4d} | Loss {avg_loss:.4f} | "
                      f"{iters_per_sec:.2f} it/s | {remaining/60:.1f}m left")

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
            batch = generate_dummy_batch(batch_size, config.block_size, config.vocab_size, device)
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
    print("="*70)
    print("Quick Quality Test: SDPA vs RA")
    print("="*70)
    print("Trains two models for 10 minutes each:")
    print("  1. SDPA baseline (fast)")
    print("  2. RA Triton (2.45x slower)")
    print()
    print("Question: Does RA achieve better validation loss")
    print("          despite completing fewer iterations?")
    print("="*70)

    if not torch.cuda.is_available():
        print("❌ CUDA required")
        return

    time_budget = 600  # 10 minutes

    # Test 1: SDPA baseline
    iters_sdpa, loss_sdpa, speed_sdpa = quick_train_test(
        "SDPA baseline",
        use_ra=False,
        time_budget_sec=time_budget
    )

    # Test 2: RA Triton
    iters_ra, loss_ra, speed_ra = quick_train_test(
        "RA Triton",
        use_ra=True,
        time_budget_sec=time_budget
    )

    # Analysis
    print("\n\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Metric':<30} {'SDPA':>15} {'RA Triton':>15} {'RA vs SDPA':>15}")
    print("-"*70)
    print(f"{'Iterations completed':<30} {iters_sdpa:>15d} {iters_ra:>15d} "
          f"{iters_ra/iters_sdpa:>14.2f}x")
    print(f"{'Validation loss':<30} {loss_sdpa:>15.4f} {loss_ra:>15.4f} "
          f"{loss_ra/loss_sdpa:>14.3f}x")
    print(f"{'Throughput (it/s)':<30} {speed_sdpa:>15.2f} {speed_ra:>15.2f} "
          f"{speed_ra/speed_sdpa:>14.2f}x")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if loss_ra < loss_sdpa:
        improvement = (loss_sdpa - loss_ra) / loss_sdpa * 100
        print(f"✅ RA WINS: {improvement:.1f}% better validation loss!")
        print(f"   Despite {(1 - iters_ra/iters_sdpa)*100:.1f}% fewer iterations")
        print(f"   RA provides better quality per unit time")
    else:
        degradation = (loss_ra - loss_sdpa) / loss_sdpa * 100
        print(f"❌ RA LOSES: {degradation:.1f}% worse validation loss")
        print(f"   RA is both slower AND lower quality")
        print(f"   Not worth using in current form")

    print("="*70)


if __name__ == "__main__":
    main()
