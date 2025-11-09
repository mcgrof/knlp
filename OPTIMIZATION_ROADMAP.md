# RA Optimization Roadmap: From 2.71ms to Baseline

## Current State
- Same-FLOP RA: 2.71ms (1.35x vs 2.00ms baseline)
- Gap to close: 0.71ms (35% overhead)

## Phase 1: Quick Wins (Target: 2.2ms, 1.1x) - 1 day

### 1.1: Hygiene + Precision (5-10% gain)
```python
# Force BF16 everywhere
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Force Flash Attention backend
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_mem_efficient=True,
        enable_math=False
    ):
        out = ra_same_flop(Q, K, V, ...)

# Ensure contiguous memory layout
Q = Q.contiguous()
K = K.contiguous()
V = V.contiguous()
```

**Expected**: 2.71ms → 2.50ms

### 1.2: Optimal R Selection (10-15% gain)
Test R ∈ {4, 8, 12, 16} and find best speed/quality tradeoff.

**Current hypothesis**:
- R=16: 2.71ms (tested)
- R=12: ~2.55ms (expect)
- R=8: ~2.45ms (expect)
- R=4: ~2.20ms (expect, but quality risk)

**Action**: Run benchmark_same_flop_ratios.py

**Expected best**: R=8 at ~2.45ms with acceptable quality

### 1.3: Shared W Matrix (5-10% gain)
Stop creating new W for each forward pass. Share across heads per layer.

```python
class RASameFlop(nn.Module):
    def __init__(self, D=64, R=8):
        super().__init__()
        # One W per layer, shared across all heads
        self.W_recip = nn.Parameter(torch.randn(D, R) * 0.02)

    def forward(self, Q, K, V):
        # Compute once per layer
        QW = torch.matmul(Q, self.W_recip)  # [B, H, T, R]
        KW = torch.matmul(K, self.W_recip)  # [B, H, T, R]

        # Reuse for all heads
        # ...
```

**Expected**: 2.45ms → 2.20ms

**Phase 1 Target**: 2.20ms (1.1x overhead) ✅

## Phase 2: Head-Selective Routing (Target: 2.0ms, 1.0x) - 2 days

### 2.1: Implement Learnable Head Gating
```python
class RASameFlop(nn.Module):
    def __init__(self, n_heads=12, D=64, R=8):
        super().__init__()
        # Per-head reciprocity gate (learnable)
        self.w_rec = nn.Parameter(torch.ones(n_heads) * 0.3)

    def forward(self, Q, K, V):
        # Separate heads by w_rec threshold
        threshold = 0.1
        ra_heads = (self.w_rec > threshold).cpu()

        if ra_heads.sum() == 0:
            # All heads use baseline SDPA
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Split and pack
        Q_baseline = Q[:, ~ra_heads]  # No RA
        Q_ra = Q[:, ra_heads]         # Use RA

        # Two SDPA calls
        out = torch.zeros_like(Q)
        out[:, ~ra_heads] = sdpa(Q_baseline, K_baseline, V_baseline)
        out[:, ra_heads] = ra_same_flop(Q_ra, K_ra, V_ra, R=8)

        return out
```

### 2.2: Train with Sparsity Regularization
```python
# Encourage heads to specialize (some use RA, some don't)
loss = task_loss + lambda_sparse * torch.mean(self.w_rec)  # L1 penalty
```

**Expected**: If 50-75% of heads learn w_rec < 0.1:
- 50% selective: 2.20ms × 0.5 + 2.00ms × 0.5 = **2.10ms**
- 75% selective: 2.20ms × 0.25 + 2.00ms × 0.75 = **2.05ms**

**Phase 2 Target**: 2.0-2.1ms (matches baseline!) ✅

## Phase 3: Advanced Optimizations (Target: 1.8ms, 0.9x) - 1 week

### 3.1: INT8 Quantization on Reciprocal Channel
Keep D_std in BF16, quantize R-channel to INT8:
```python
# D_std part: BF16 (keep precision)
Q_std = Q[..., :D_std].to(torch.bfloat16)

# R part: INT8 (reciprocal can tolerate lower precision)
QW_fp = torch.matmul(Q, W)
QW_int8 = quantize_int8(QW_fp)  # [B, H, T, R] in INT8

# Combine
Q_aug = [Q_std (BF16), QW_int8 (INT8)]
```

**Expected gain**: 10-20% on RA heads → 2.05ms × 0.9 = **1.85ms**

### 3.2: Token-Level Selective RA (Optional)
```python
# Lightweight scorer: which tokens need reciprocity?
importance = sigmoid(linear(hidden_states.mean(-1)))  # [B, T]
use_ra_tokens = importance > 0.8  # Top 20% tokens

# Two attention paths
fast_tokens = ~use_ra_tokens
slow_tokens = use_ra_tokens

# Blend results
```

**Expected gain**: 10% → 1.85ms × 0.9 = **1.67ms**

**Phase 3 Target**: 1.8ms (10% faster than baseline!) ✅

## Phase 4: CTR Kernel (Target: 1.5-1.6ms) - 2 weeks

Custom Triton kernel with causal-transpose-reuse:
```python
# For each tile (I, J) where I >= J (causal):
# Load Q[I], K[J], Q[J], K[I] once
# Compute:
#   - q_i · k_j (standard)
#   - q_j · k_i (reciprocal)
# Single online softmax per row
# Bandwidth ≈ baseline
```

**Expected**: 1.5-1.6ms (20-25% faster than baseline)

**Effort**: High (custom kernel development)
**Priority**: Low (only if phases 1-3 insufficient)

## Quality Validation Plan

For each optimization phase:

1. **Quick sanity check** (5 minutes)
   ```bash
   # Train on dummy data for 100 iterations
   python3 quick_quality_test.py --steps 100
   ```

2. **Short ablation** (30 minutes)
   ```bash
   # Train on real data for limited steps
   python3 train.py --max-steps 1000
   ```

3. **Full validation** (if phase looks promising)
   ```bash
   # Run full 1-hour comparison
   python3 quick_quality_test.py
   ```

## Decision Tree

```
Start: 2.71ms (1.35x)
  ↓
Apply Phase 1 (hygiene + R=8 + shared W)
  ↓
→ 2.20ms (1.1x)
  ↓
Quality check: Does R=8 maintain val_loss?
  ├─ YES → Proceed to Phase 2
  └─ NO  → Use R=12 (2.35ms), still good
     ↓
Apply Phase 2 (head-selective)
  ↓
→ 2.0-2.1ms (1.0-1.05x) ← TARGET MET!
  ↓
Quality check: Do selective heads help?
  ├─ YES → SUCCESS, ship it
  └─ NO  → Evaluate if 2.2ms (Phase 1) is acceptable
     ↓
If want even faster:
Apply Phase 3 (INT8 + token-selective)
  ↓
→ 1.8ms (0.9x) ← BEATS BASELINE!
```

## Timeline

- **Week 1**: Phases 1-2 (reach baseline speed)
- **Week 2**: Quality validation, tune hyperparameters
- **Week 3**: Phase 3 if needed (beat baseline)
- **Week 4**: Integration, documentation

## Success Criteria

**Minimum viable**: 2.0-2.2ms (1.0-1.1x) with quality maintained
**Stretch goal**: 1.8ms (0.9x) while improving quality
**Moonshot**: 1.5ms (0.75x) with CTR kernel

## Next Immediate Actions

1. **Run benchmark_same_flop_ratios.py** to find optimal R
2. **Implement hygiene fixes** (BF16, Flash backend forcing)
3. **Implement shared W matrix** reuse
4. **Measure**: Should reach ~2.2ms after these 3 steps

Expected time: 4-6 hours to implement + test Tier 1 optimizations.
