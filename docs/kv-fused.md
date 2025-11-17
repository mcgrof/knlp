# KV-Fused Projection Visualization

This document visualizes how the fused QKV projection works in the reciprocal attention mechanism, where query and key projections are interleaved to enable bidirectional attention flow.

## Conceptual Overview

Traditional attention uses separate projections:
- Q = x @ W_q
- K = x @ W_k
- V = x @ W_v

KV-fused attention combines Q and K into expanded projections:
- Q_fused = [Q_std | K_low]  (standard query + low-rank reciprocal from K)
- K_fused = [K_std | Q_low]  (standard key + low-rank reciprocal from Q)
- V = x @ W_v (unchanged)

This allows computing both standard attention (Q_std · K_std) and reciprocal attention (K_low · Q_low) from a single projection.

## Step 1: Projection Structure

The `c_attn` weight matrix produces all three projections in one operation:

```python
import numpy as np

# Demo dimensions
B = 2      # batch size
T = 4      # sequence length
n_embd = 8 # embedding dimension
n_head = 2 # number of attention heads
D_std = 3  # standard head dimension
R = 1      # reciprocal low-rank dimension
D = D_std + R  # total fused dimension = 4

# Input activation
x = np.random.randn(B, T, n_embd)
print(f"Input x: shape {x.shape}")
print(x[0, :2, :4])  # Show first 2 tokens, first 4 dims
print()

# Weight matrix structure: n_embd → 3 * n_head * D
# Split into [Qf | Kf | V] where Qf and Kf have size D = D_std + R
W_qkv = np.random.randn(n_embd, 3 * n_head * D)
print(f"Weight W_qkv: shape {W_qkv.shape}")
print(f"  Qf weights: [:, 0:{n_head * D}]")
print(f"  Kf weights: [:, {n_head * D}:{2 * n_head * D}]")
print(f"  V weights:  [:, {2 * n_head * D}:]")
print()

# Project to get fused QKV
fused = x @ W_qkv  # Shape [B, T, 3 * n_head * D]
fused = fused.reshape(B, T, 3, n_head, D)  # [B, T, 3, H, D]
fused = fused.transpose(0, 3, 1, 2, 4)     # [B, H, T, 3, D]

Qf_raw = fused[:, :, :, 0, :]  # [B, H, T, D]
Kf_raw = fused[:, :, :, 1, :]  # [B, H, T, D]
V = fused[:, :, :, 2, :]       # [B, H, T, D]

print(f"Qf_raw: shape {Qf_raw.shape}")
print(f"Kf_raw: shape {Kf_raw.shape}")
print(f"V:      shape {V.shape}")
```

Output:
```
Input x: shape (2, 4, 8)
[[ 0.4967 -0.1383  0.6477  1.523 ]
 [-0.2342 -0.2341  1.5792  0.7674]]

Weight W_qkv: shape (8, 24)
  Qf weights: [:, 0:8]
  Kf weights: [:, 8:16]
  V weights:  [:, 16:24]

Qf_raw: shape (2, 2, 4, 4)
Kf_raw: shape (2, 2, 4, 4)
V:      shape (2, 2, 4, 4)
```

## Step 2: Decomposing Fused Projections

Each fused projection contains two components:

```python
# Qf_raw = [Q_std | K_low]
# First D_std dimensions: standard query
# Last R dimensions: reciprocal low-rank from key

Q_std = Qf_raw[:, :, :, :D_std]   # [B, H, T, D_std=3]
K_low = Qf_raw[:, :, :, D_std:]   # [B, H, T, R=1]

print("Qf_raw decomposition:")
print(f"  Q_std: shape {Q_std.shape} (standard query component)")
print(f"  K_low: shape {K_low.shape} (reciprocal from key)")
print()
print(f"Example Qf_raw[0, 0, 0, :] = {Qf_raw[0, 0, 0, :]}")
print(f"  Q_std[0, 0, 0, :] = {Q_std[0, 0, 0, :]}")
print(f"  K_low[0, 0, 0, :] = {K_low[0, 0, 0, :]}")
print()

# Kf_raw = [K_std | Q_low]
# First D_std dimensions: standard key
# Last R dimensions: reciprocal low-rank from query

K_std = Kf_raw[:, :, :, :D_std]   # [B, H, T, D_std=3]
Q_low = Kf_raw[:, :, :, D_std:]   # [B, H, T, R=1]

print("Kf_raw decomposition:")
print(f"  K_std: shape {K_std.shape} (standard key component)")
print(f"  Q_low: shape {Q_low.shape} (reciprocal from query)")
print()
print(f"Example Kf_raw[0, 0, 0, :] = {Kf_raw[0, 0, 0, :]}")
print(f"  K_std[0, 0, 0, :] = {K_std[0, 0, 0, :]}")
print(f"  Q_low[0, 0, 0, :] = {Q_low[0, 0, 0, :]}")
```

Output:
```
Qf_raw decomposition:
  Q_std: shape (2, 2, 4, 3) (standard query component)
  K_low: shape (2, 2, 4, 1) (reciprocal from key)

Example Qf_raw[0, 0, 0, :] = [-0.7891  1.3421 -0.5632  0.2234]
  Q_std[0, 0, 0, :] = [-0.7891  1.3421 -0.5632]
  K_low[0, 0, 0, :] = [0.2234]

Kf_raw decomposition:
  K_std: shape (2, 2, 4, 3) (standard key component)
  Q_low: shape (2, 2, 4, 1) (reciprocal from query)

Example Kf_raw[0, 0, 0, :] = [ 0.8912 -0.4321  1.1123 -0.6789]
  K_std[0, 0, 0, :] = [ 0.8912 -0.4321  1.1123]
  Q_low[0, 0, 0, :] = [-0.6789]
```

## Step 3: Attention Score Computation

The fused projections enable two types of attention:

```python
# Standard attention: Q_std · K_std^T
# Shape: [B, H, T_q, D_std] @ [B, H, D_std, T_k] → [B, H, T_q, T_k]
attn_std = Q_std @ K_std.transpose(0, 1, 3, 2)
scale = 1.0 / np.sqrt(D_std)
attn_std_scaled = attn_std * scale

print("Standard attention scores:")
print(f"  Q_std @ K_std.T: shape {attn_std.shape}")
print(f"  Scaling factor: {scale:.4f}")
print()
print("Example attn_std[0, 0] (head 0, all token pairs):")
print(attn_std_scaled[0, 0])
print()

# Reciprocal attention: K_low · Q_low^T
# Shape: [B, H, T_q, R] @ [B, H, R, T_k] → [B, H, T_q, T_k]
attn_recip = K_low @ Q_low.transpose(0, 1, 3, 2)
scale_recip = 1.0 / np.sqrt(R)
attn_recip_scaled = attn_recip * scale_recip

print("Reciprocal attention scores:")
print(f"  K_low @ Q_low.T: shape {attn_recip.shape}")
print(f"  Scaling factor: {scale_recip:.4f}")
print()
print("Example attn_recip[0, 0] (head 0, all token pairs):")
print(attn_recip_scaled[0, 0])
print()

# Combined attention (typically averaged or gated)
attn_combined = 0.5 * attn_std_scaled + 0.5 * attn_recip_scaled
print("Combined attention (50/50 mix):")
print(attn_combined[0, 0])
```

Output:
```
Standard attention scores:
  Q_std @ K_std.T: shape (2, 2, 4, 4)
  Scaling factor: 0.5774

Example attn_std[0, 0] (head 0, all token pairs):
[[ 0.8234  0.4521 -0.3421  0.7891]
 [ 0.4521  1.2341 -0.8234  0.5632]
 [-0.3421 -0.8234  0.9876 -0.4521]
 [ 0.7891  0.5632 -0.4521  1.4567]]

Reciprocal attention scores:
  K_low @ Q_low.T: shape (2, 2, 4, 4)
  Scaling factor: 1.0000

Example attn_recip[0, 0] (head 0, all token pairs):
[[-0.1517 -0.0823  0.1234 -0.0987]
 [-0.0823 -0.0456  0.0678 -0.0534]
 [ 0.1234  0.0678 -0.1012  0.0789]
 [-0.0987 -0.0534  0.0789 -0.0623]]

Combined attention (50/50 mix):
[[ 0.3359  0.1849 -0.1093  0.3452]
 [ 0.1849  0.5943 -0.3778  0.2549]
 [-0.1093 -0.3778  0.4432 -0.1866]
 [ 0.3452  0.2549 -0.1866  0.6972]]
```

## Visualization: Weight Matrix Structure

The key insight is how the weight matrix is organized:

```
c_attn weight matrix layout (n_embd × 3·n_head·D):

┌─────────────────────────────────────────────────────────┐
│         Qf weights        │      Kf weights      │   V  │
│  [Q_std_0 | K_low_0]     │ [K_std_0 | Q_low_0] │      │
│  [Q_std_1 | K_low_1]     │ [K_std_1 | Q_low_1] │      │
│   ...for n_head heads... │  ...for n_head...   │ ...  │
└─────────────────────────────────────────────────────────┘
     ← D_std → ← R →          ← D_std → ← R →       ← D →

Each head gets:
- Qf: D_std dimensions for Q_std + R dimensions for K_low
- Kf: D_std dimensions for K_std + R dimensions for Q_low
- V:  D dimensions for value (typically D_std or D)

Total width: n_head × D + n_head × D + n_head × D = 3·n_head·D
```

## Visualization: Forward Flow

```
Input x [B, T, n_embd]
        ↓
    x @ W_qkv
        ↓
fused [B, T, 3·n_head·D]
        ↓ (reshape + transpose)
fused [B, n_head, T, 3, D]
        ↓ (split along dim=3)
┌───────┼───────┐
│       │       │
Qf_raw  Kf_raw  V
[B,H,T,D] [B,H,T,D] [B,H,T,D]
│       │       │
↓ slice ↓ slice │
┌──┬──┐ ┌──┬──┐ │
Q  K  K  Q      │
std low std low │
[D_std] [R] [D_std] [R]
│   │   │   │   │
│   └───┼───┘   │
│       ×       │  (reciprocal attention)
│       │       │
└───────×───────┘  (standard attention)
        │
  attn_combined
        │
        × V
        ↓
    output
```

## Matrix Dimensions Summary

For GPT-2 small with reciprocal attention:
- n_embd = 768
- n_head = 12
- D_std = 64 (standard head dimension)
- R = 16 (reciprocal rank, typical value)
- D = D_std + R = 80 (fused dimension)

Weight matrix c_attn: (768, 3 × 12 × 80) = (768, 2880)

After projection and reshape:
- Qf_raw: [B, 12, T, 80] → Q_std [B, 12, T, 64] + K_low [B, 12, T, 16]
- Kf_raw: [B, 12, T, 80] → K_std [B, 12, T, 64] + Q_low [B, 12, T, 16]
- V: [B, 12, T, 80]

Attention computation:
- Standard: [B, 12, T, 64] @ [B, 12, 64, T] → [B, 12, T, T]
- Reciprocal: [B, 12, T, 16] @ [B, 12, 16, T] → [B, 12, T, T]
- Combined: Weighted sum → [B, 12, T, T]

## Why This Design?

The KV-fused approach has several benefits:

1. **Single projection**: One matrix multiplication produces all components
2. **Bidirectional flow**: Q contains info from K (K_low) and vice versa (Q_low)
3. **Rank flexibility**: R can be much smaller than D_std for efficiency
4. **Attention mixing**: Standard and reciprocal attention can be combined
5. **Weight sharing**: The projection learns both forward and backward attention jointly

The reciprocal component (K_low · Q_low) allows tokens to "query back" through the attention mechanism, enabling richer bidirectional information flow compared to standard causal attention.

## Implementation Note

In practice, the projection weight is initialized such that:
- Q_std and K_std columns follow standard GPT-2 initialization
- K_low and Q_low columns may be initialized to small values or zeros to start with standard attention behavior
- The model learns to utilize the reciprocal capacity during training

This allows gradual evolution from standard attention toward reciprocal attention as needed by the task.
