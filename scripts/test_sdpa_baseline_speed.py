#!/usr/bin/env python3
"""
Quick benchmark: SDPA vs open-coded attention speed on A10G.
Measures actual iteration time to estimate what L0 COULD be with SDPA.
"""
import torch
import torch.nn.functional as F
import time

device = "cuda"
B, H, T, D = 8, 12, 1024, 64  # GPT-2 124M dimensions, batch=8

Q = torch.randn(B, H, T, D, device=device)
K = torch.randn(B, H, T, D, device=device)
V = torch.randn(B, H, T, D, device=device)

# Warmup
for _ in range(10):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Benchmark SDPA
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out1 = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
torch.cuda.synchronize()
sdpa_time = (time.time() - start) / 100 * 1000

# Open-coded attention
def opencoded_attn(Q, K, V):
    S = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    S = S.masked_fill(~mask, float('-inf'))
    attn = F.softmax(S, dim=-1)
    return torch.matmul(attn, V)

# Warmup
for _ in range(10):
    _ = opencoded_attn(Q, K, V)

# Benchmark open-coded
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out2 = opencoded_attn(Q, K, V)
torch.cuda.synchronize()
opencoded_time = (time.time() - start) / 100 * 1000

print("=" * 70)
print(f"Attention Performance on A10G (batch=8, T=1024):")
print(f"  SDPA (fused):      {sdpa_time:.2f} ms/iter")
print(f"  Open-coded:        {opencoded_time:.2f} ms/iter")
print(f"  Overhead:          {(opencoded_time/sdpa_time - 1)*100:.1f}%")
print()
print(f"Extrapolating to full GPT-2 training iteration:")
print(f"  Your L0 actual:    4535 ms/iter (open-coded)")
print(f"  Estimated w/SDPA:  {4535 * sdpa_time / opencoded_time:.0f} ms/iter")
print(f"  Lost speedup:      {(4535 - 4535 * sdpa_time / opencoded_time):.0f} ms/iter")
print("=" * 70)
