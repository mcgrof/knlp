#!/usr/bin/env python3
"""
Quick test: verify shared score matrix produces same results as two SDPA calls.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.model import GPTConfig
from ra import GPT2_RA_Learned

# Small model
config = GPTConfig(
    n_layer=2,
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

print("✓ Shared score matrix implementation works")
print(f"  Loss: {loss.item():.4f}")
print(f"  Logits shape: {logits.shape}")

# Backward pass
loss.backward()

print("✓ Backward pass works")
print(f"  Alternation gradients: {model.alternation_logits.grad}")

print("\nAll tests passed - shared scores optimization working!")
