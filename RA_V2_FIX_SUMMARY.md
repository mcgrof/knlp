# RA Ultimate v2: Performance Fix Summary

## Problem Identified

The original `ra_ultimate.py` had a critical performance regression:

**Issue**: Building massive T×T bias tensors
```python
# SLOW - allocates 201MB per forward pass!
bias = torch.zeros(B, n_heads, T, T, device=device, dtype=dtype)
bias.masked_fill_(~causal_mask, float('-inf'))
```

For B=8, H=12, T=1024:
- Single bias tensor: 8 × 12 × 1024 × 1024 × 2 bytes = **201 MB**
- We created TWO (baseline + RA packs): **402 MB total**
- Plus expensive `masked_fill_()` operations

**Result**: 4.65ms (3.86x slower than baseline 1.21ms)

## Root Cause

We misunderstood ChatGPT's `eval.py` "fused bias" optimization. The goal
wasn't to merge causal + discoverability into a T×T tensor, but to use
Flash Attention's native causal support and add ONLY discoverability as
a column bias.

## Solution: Column Bias Only

**Key insight from ChatGPT**: Never build T×T masks!

### Before (Slow)
```python
# Allocates [B, H, T, T] - HUGE!
bias = torch.zeros(B, H, T, T, device=device, dtype=dtype)
bias.masked_fill_(~causal_mask, float('-inf'))
bias = bias + discoverability  # Add column bias to full mask

out = F.scaled_dot_product_attention(
    Q, K, V, attn_mask=bias, is_causal=False  # Already in bias
)
```

### After (Fast)
```python
# Column bias only: [B, H, 1, T] -> [B*H, 1, T]
d = d_bias[:, :T].unsqueeze(0).unsqueeze(-2)  # [1, H, 1, T]
d = d - d.mean(dim=-1, keepdim=True)  # Zero-mean
d = d * w_disc.view(1, -1, 1, 1)  # Scale
col_bias = d.expand(B, H, 1, T).reshape(B * H, 1, T).contiguous()

# Reshape to [B*H, T, D]
q = Q.transpose(1, 2).reshape(B * H, T, D)
k = K.transpose(1, 2).reshape(B * H, T, D)
v = V.transpose(1, 2).reshape(B * H, T, D)

# Let Flash handle causal mask
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=col_bias, is_causal=True, dropout_p=0.0
    )

out = out.reshape(B, T, H, D).transpose(1, 2).contiguous()
```

### Memory Savings

**Before**:
- Bias size: B × H × T × T = 8 × 12 × 1024 × 1024 = 100,663,296 elements
- In BF16: 201 MB per tensor
- Total: ~400 MB (2 tensors)

**After**:
- Column bias size: B × H × 1 × T = 8 × 12 × 1 × 1024 = 98,304 elements
- In BF16: 0.19 MB per tensor
- Total: ~0.4 MB (2 tensors)

**Savings**: 999x less memory allocated! (400 MB → 0.4 MB)

## Additional Optimizations

1. **New SDPA API**: Uses `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` instead
   of deprecated `torch.backends.cuda.sdp_kernel()`

2. **Proper tensor shapes**: SDPA expects `[B*H, T, D]` not `[B, H, T, D]`
   for optimal Flash Attention path

3. **Native causal handling**: `is_causal=True` lets Flash handle causal
   mask internally (more efficient than manual masking)

## Expected Performance

Based on our previous ra_same_flop.py results (2.71ms @ R=16):

**Conservative estimate** (R=8, better optimizations):
- 100% RA heads: ~2.2-2.4ms (1.8-2.0x vs 1.21ms baseline)
- 50% selective: ~1.7-1.9ms (1.4-1.6x)
- 25% selective: ~1.4-1.6ms (1.15-1.3x)

**Optimistic estimate** (if column bias optimization works as expected):
- 100% RA heads: ~1.8-2.0ms (1.5-1.65x)
- 50% selective: ~1.5-1.6ms (1.24-1.32x)
- 25% selective: ~1.3-1.4ms (1.07-1.16x)

## How to Test

On GPU machine:

```bash
python3 ra_ultimate_v2.py
```

Expected output:
```
======================================================================
ULTIMATE RA v2 Benchmark (Fixed Bias Handling)
======================================================================
Optimizations:
  - Same-FLOP (D_std=56, R=8)
  - Shared W matrix
  - Head-selective routing
  - BF16 + Flash backend
  - Efficient bias (expand instead of allocate)

1. Baseline SDPA (optimized)...
   1.21 ms/iter

2. Ultimate RA v2 (100% RA heads, threshold=0.0)...
   ~2.0 ms/iter (~1.65x)  # Expected

2. Ultimate RA v2 (~50% RA heads, threshold=0.25)...
   ~1.6 ms/iter (~1.32x)  # Expected

2. Ultimate RA v2 (~25% RA heads, threshold=0.35)...
   ~1.4 ms/iter (~1.16x)  # Expected

======================================================================
VERDICT
======================================================================
Good! Within 15% of baseline
   Best: ~1.4ms vs 1.21ms baseline
   ~16% overhead
======================================================================
```

## Comparison to Previous Results

| Implementation | Performance | vs Baseline |
|----------------|-------------|-------------|
| Open-coded (original) | 9.13ms | 4.6x |
| Triton kernel | 4.86ms | 2.43x |
| SDPA folded (low-rank) | 3.78ms | 1.89x |
| Same-FLOP (ra_same_flop.py) | 2.71ms | 1.35x |
| **Ultimate v1 (BROKEN)** | **4.65ms** | **3.86x** ← regression! |
| **Ultimate v2 (FIXED)** | **~1.4-2.0ms** | **~1.16-1.65x** ← expected |

## Why Column Bias Works

Flash Attention internally:
1. Tiles the computation in blocks (e.g., 128×128)
2. Computes softmax within each block
3. For causal mask, only processes lower triangle blocks

When you pass:
- `attn_mask=[B*H, 1, T]`: Broadcast across rows (cheap!)
- `is_causal=True`: Flash skips upper triangle (free!)

When you build `[B*H, T, T]`:
- Can't use Flash's causal optimization
- Must allocate and fill entire mask
- Extra memory traffic

## Code Quality

The fixed implementation:
- Zero T×T tensor allocations
- Uses latest PyTorch SDPA API
- Matches Flash Attention's expected input format
- Maintains all previous optimizations (same-FLOP, head-selective, shared W)

## Next Steps

1. **Run benchmark** on GPU: `python3 ra_ultimate_v2.py`
2. **Verify performance** matches expectations (~1.4-2.0ms)
3. **If successful**: Replace ra_ultimate.py with ra_ultimate_v2.py
4. **Quality test**: Run `quick_quality_test.py` with v2 implementation
5. **Integration**: Update train_ra_mla.py to use UltimateRAv2

## Files

- `ra_ultimate_v2.py`: Fixed implementation (ready to test)
- `ra_ultimate.py`: Broken version (for reference)
- `ra_same_flop.py`: Previous working version (2.71ms baseline)

## Credit

Fix based on ChatGPT's advice:
> Never build a T×T mask. Let Flash handle causal via is_causal=True.
> Pass discoverability as a column bias only with shape [B*H, 1, T].

This eliminated the 999x memory allocation overhead and should restore
expected performance.
