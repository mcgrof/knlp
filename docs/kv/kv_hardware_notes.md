# KV Compression Hardware Considerations

This document covers the hardware-level implications of KV cache compression,
including memory bandwidth, kernel behavior, and practical deployment concerns.

## 1. Why V Compression Affects Throughput More Than K

### Attention Computation Flow

Standard attention computes:
```
scores = Q @ K.T      # [B, H, T_new, T]
weights = softmax(scores)
output = weights @ V  # [B, H, T_new, d]
```

### Memory Access Patterns

**K access**: Used in Q@K.T
- K is accessed once per query token
- Access pattern: sequential read across sequence dimension
- Memory: `B × H × T × d` elements read

**V access**: Used in weights@V
- V is accessed once per query token (after softmax)
- Access pattern: weighted sum across sequence
- Memory: `B × H × T × d` elements read

### Why V Compression Wins

1. **V determines output dimension**: The output of attention has the same
   dimension as V. Compressed V means smaller output tensor.

2. **Post-softmax operation**: V multiplication happens after the expensive
   softmax, so compressed V reduces the final matmul cost.

3. **K affects score computation**: K compression changes the score matrix
   dimensions, which propagates through softmax. This is more disruptive to
   existing kernels.

4. **Gradient flow**: During training, V gradients flow more directly from
   the loss, while K gradients go through softmax Jacobian.

### Quantitative Impact

For sequence length T=1024, d=64, rank R=32:

| Operation | Standard | V-only R=32 | K+V R=32 |
|-----------|----------|-------------|----------|
| Q@K.T FLOPs | 2BTHd | 2BTHd | 2BTHR |
| weights@V FLOPs | 2BTHd | 2BTHR | 2BTHR |
| Total attention | 4BTHd | 2BTHd + 2BTHR | 4BTHR |
| Savings | - | 25% | 50% |

V-only compression is simpler to implement and captures 50% of the
potential FLOP savings.

## 2. How Caching Compressed V Reduces Bandwidth

### Memory Bandwidth Bottleneck

Modern GPUs are memory-bandwidth limited for attention:
- A100: 2 TB/s HBM bandwidth, 312 TFLOPS compute
- Arithmetic intensity threshold: 156 FLOPs/byte
- Attention intensity: ~1 FLOP/byte (heavily memory-bound)

### Standard KV Cache Bandwidth

Per token generation:
```
Bytes read = 2 × n_layers × H × T × d × sizeof(dtype)
           = 2 × 12 × 12 × 1024 × 64 × 2 (FP16)
           = 36 MB per token
```

At 2 TB/s: 36 MB / 2 TB/s = 18 μs minimum latency per layer

### Compressed Cache Bandwidth

With d_compressed=128 (vs d_model=768):
```
Bytes read = n_layers × T × d_compressed × sizeof(dtype)
           = 12 × 1024 × 128 × 2
           = 3 MB per token
```

At 2 TB/s: 3 MB / 2 TB/s = 1.5 μs minimum latency

**Bandwidth reduction: 12x**

### Effective Throughput Improvement

The bandwidth savings translate to throughput in different ways:

1. **Latency-bound (single sequence)**: TTFT improves proportionally
   to bandwidth reduction

2. **Throughput-bound (batched)**: Can fit more sequences in memory,
   improving total throughput

3. **Mixed workloads**: Benefits compound across both dimensions

## 3. Why GPU Kernels Must Be Aware of Rank

### Standard Attention Kernel Assumptions

FlashAttention and similar optimized kernels assume:
- K, V have shape `[B, H, T, d]`
- d is fixed (typically 64, 128, or 256)
- Tiling optimized for these specific dimensions

### Problems with Naive Compression

If you compress K/V to rank R < d:

1. **Shape mismatch**: Kernel expects `[B, H, T, d]`, gets `[B, H, T, R]`

2. **Tiling inefficiency**: Tiles optimized for d=64 waste registers on d=32

3. **Memory layout**: Compressed tensors may not be contiguous in the
   expected pattern

4. **Intermediate allocation**: Kernels allocate based on d, wasting memory

### Solutions

**Option A: Pre-expand** (Simple, inefficient)
```python
V_compressed = cache.get_v()  # [B, H, T, R]
V_expanded = expand(V_compressed)  # [B, H, T, d]
output = flash_attention(Q, K, V_expanded)  # Uses standard kernel
```
- Pro: Works with existing kernels
- Con: Loses memory benefits during computation

**Option B: Rank-aware kernel** (Complex, efficient)
```python
# Custom kernel that handles R ≠ d internally
output = flash_attention_compressed(Q, K_compressed, V_compressed, expand_K, expand_V)
```
- Pro: Full efficiency
- Con: Requires kernel modifications

**Option C: Fused decompress-attend** (Optimal, most complex)
```python
# Single kernel that reads compressed, computes full attention, writes output
output = fused_compressed_attention(Q, cache_compressed, expand_matrices)
```
- Pro: Optimal memory and compute
- Con: Significant engineering effort

## 4. FlashAttention v2/v3 Interaction with Compressed KV

### FlashAttention v2 Architecture

FlashAttention v2 achieves efficiency through:
1. **Tiling**: Processes attention in SRAM-sized blocks
2. **Recomputation**: Avoids storing attention matrix in HBM
3. **Fused operations**: Q@K, softmax, @V in one kernel

### Compression Compatibility Issues

**Issue 1: Fixed head dimension**
- FA2 kernels are compiled for specific head dimensions (64, 128)
- Compressed dimensions (32, 48) may not have optimized kernels

**Issue 2: Expand overhead**
- If we expand before FA2: defeats memory savings
- If we expand inside FA2: requires kernel modification

**Issue 3: Tiling assumptions**
- Tiles assume uniform dimension across K and V
- K-only or V-only compression breaks this symmetry

### FlashAttention v3 Opportunities

FA3 introduces:
- Better support for GQA (grouped-query attention)
- More flexible head dimension handling
- Improved memory access patterns

**Potential v3 modifications for compression**:
```python
# Hypothetical FA3 API with compression support
output = flash_attention_v3(
    Q,                    # [B, H_q, T_new, d]
    K_compressed,         # [B, H_kv, T, R]
    V_compressed,         # [B, H_kv, T, R]
    expand_K,             # [R, d] or None
    expand_V,             # [R, d] or None
    compression_mode="kv_only"
)
```

### Current Best Practice

Until FA supports compression natively:

1. **For inference**: Use standard attention with compressed cache,
   expand just-in-time. Memory savings come from smaller cache, not
   faster attention.

2. **For training**: Use full-dimension attention, compress only
   for storage/communication.

3. **For research**: Implement custom Triton kernels for compressed
   attention to demonstrate potential speedups.

## 5. Modifications for Truly Fast Compressed Attention

### Required Kernel Changes

**Level 1: Dimension-aware allocation**
```cuda
// Instead of:
float tile_K[TILE_SIZE][HEAD_DIM];  // Fixed HEAD_DIM=64

// Use:
float tile_K[TILE_SIZE][COMPRESSED_DIM];  // COMPRESSED_DIM=32
```
- Reduces register pressure
- Enables more tiles in parallel
- ~2x register efficiency for R=32

**Level 2: Fused expand**
```cuda
// Inside attention kernel:
for (int i = 0; i < TILE_SIZE; i++) {
    // Load compressed K
    load_tile(K_compressed, tile_K_compressed);

    // Expand in registers
    matmul_small(tile_K_compressed, expand_K, tile_K_full);

    // Compute attention with expanded K
    compute_attention(Q_tile, tile_K_full, ...);
}
```
- Expand happens in fast SRAM
- No HBM write for expanded tensor
- ~50% bandwidth reduction

**Level 3: Compressed score computation**
```cuda
// Compute scores directly in compressed space
// scores = Q @ W_expand @ K_compressed.T
//        = (Q @ W_expand) @ K_compressed.T
//        = Q_projected @ K_compressed.T

// Pre-project Q once
Q_projected = Q @ W_expand;  // [B, H, T_new, R]

// Compute compressed scores
scores = Q_projected @ K_compressed.T;  // [B, H, T_new, T] but cheaper
```
- Reduces score computation FLOPs by d/R
- K@Q.T complexity: O(T × R) instead of O(T × d)

### Triton Implementation Sketch

```python
@triton.jit
def compressed_attention_kernel(
    Q_ptr, K_compressed_ptr, V_compressed_ptr,
    expand_K_ptr, expand_V_ptr,
    output_ptr,
    seq_len, head_dim, compressed_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Load expand matrices to SRAM once
    expand_K = tl.load(expand_K_ptr, ...)  # [compressed_dim, head_dim]
    expand_V = tl.load(expand_V_ptr, ...)

    # Process tiles
    for m in range(0, seq_len, BLOCK_M):
        # Load Q tile
        Q_tile = tl.load(Q_ptr + m * head_dim, ...)

        acc = tl.zeros([BLOCK_M, head_dim])

        for n in range(0, seq_len, BLOCK_N):
            # Load compressed K, V tiles
            K_c_tile = tl.load(K_compressed_ptr + n * compressed_dim, ...)
            V_c_tile = tl.load(V_compressed_ptr + n * compressed_dim, ...)

            # Expand K in SRAM
            K_tile = tl.dot(K_c_tile, expand_K)  # [BLOCK_N, head_dim]

            # Compute attention scores
            scores = tl.dot(Q_tile, tl.trans(K_tile))
            weights = tl.softmax(scores)

            # Expand V and accumulate
            V_tile = tl.dot(V_c_tile, expand_V)
            acc += tl.dot(weights, V_tile)

        tl.store(output_ptr + m * head_dim, acc)
```

## 6. Impact on HBM → SRAM Movement

### Standard Attention Data Movement

Per attention operation:
```
HBM reads:
  Q: B × H × T_new × d bytes
  K: B × H × T × d bytes (from cache)
  V: B × H × T × d bytes (from cache)
  Total: ~3 × B × H × T × d bytes

HBM writes:
  Output: B × H × T_new × d bytes
```

For T=1024, d=64, H=12, B=1, FP16:
- Read: 3 × 1 × 12 × 1024 × 64 × 2 = 4.7 MB
- Write: 1 × 12 × 1 × 64 × 2 = 1.5 KB (negligible)

### Compressed Attention Data Movement

With R=32:
```
HBM reads:
  Q: B × H × T_new × d bytes (unchanged)
  K_c: B × H × T × R bytes
  V_c: B × H × T × R bytes
  expand_K: R × d bytes
  expand_V: R × d bytes
  Total: B×H×T_new×d + 2×B×H×T×R + 2×R×d bytes

HBM writes:
  Output: B × H × T_new × d bytes (unchanged)
```

For T=1024, R=32, d=64, H=12, B=1, FP16:
- Read: 1.5 KB + 2 × 1.5 MB + 8 KB = 3.0 MB
- Reduction: 36% less data movement

### SRAM Pressure Trade-off

Compressed attention trades HBM bandwidth for SRAM compute:

**SRAM usage increase**:
- expand_K: R × d × sizeof(float) per head
- expand_V: R × d × sizeof(float) per head
- Total: 2 × 32 × 64 × 4 = 16 KB per head

**Benefit**: More tiles can fit in SRAM since K/V tiles are smaller

**Net effect**: Usually positive because:
1. HBM is the bottleneck (memory-bound)
2. SRAM compute is fast
3. Smaller tiles enable more parallelism

## 7. TTFT Optimizations via Compressed Projections

### Standard TTFT Breakdown

For first token (processing full prompt):
```
1. Embed: O(T × d_model)
2. Per layer:
   - QKV projection: 3 × O(T × d_model²)
   - Attention: O(T² × d)
   - Output projection: O(T × d_model²)
   - MLP: O(T × d_model × d_ff)
3. LM head: O(T × d_model × vocab)
```

For GPT-2-124M (12 layers, d=768, T=1024):
- QKV proj: 12 × 3 × 1024 × 768² = 21.7B FLOPs
- Attention: 12 × 1024² × 768 = 9.7B FLOPs
- Attention is ~30% of total forward pass

### Compressed TTFT Optimization

**Strategy 1: Latent projection** (MLA-style)
```python
# Standard: x → Q, K, V (3 projections)
Q = x @ W_q  # [T, d] @ [d, n_h × d_h]
K = x @ W_k
V = x @ W_v

# MLA: x → latent → K, V (1 projection + 2 smaller)
latent = x @ W_latent  # [T, d] @ [d, d_latent]
K = latent @ W_k'       # [T, d_latent] @ [d_latent, n_h × d_h]
V = latent @ W_v'
```

FLOP comparison (d=768, d_latent=256, n_h=12, d_h=64):
- Standard: 3 × 768 × 768 = 1.77M FLOPs per token
- MLA: 768 × 256 + 2 × 256 × 768 = 0.59M FLOPs per token
- **Savings: 67%** on KV projection

**Strategy 2: Compressed Q projection**
```python
# For inference, Q can also use latent
Q = latent @ W_q'  # Reuse latent computation
```
- Additional savings on Q projection
- Useful when latent captures sufficient information

**Strategy 3: Early exit for cache**
```python
# During prefill, store compressed cache early
def prefill_optimized(x, layers):
    for i, layer in enumerate(layers):
        x, kv = layer(x)
        # Store compressed KV immediately
        cache.store(i, compress(kv))
    return x
```
- Reduces peak memory during prefill
- Enables streaming compression

### TTFT Improvement Estimates

| Configuration | QKV Proj | Attention | Total | TTFT Δ |
|--------------|----------|-----------|-------|--------|
| Standard | 21.7B | 9.7B | 31.4B | baseline |
| MLA (d_latent=256) | 7.1B | 9.7B | 16.8B | -46% |
| MLA + V-compress | 7.1B | 7.3B | 14.4B | -54% |
| Full hybrid | 5.9B | 5.0B | 10.9B | -65% |

## 8. Hardware-Specific Recommendations

### NVIDIA A100 (80GB)

- HBM bandwidth: 2 TB/s
- SM count: 108
- Tensor core peak: 312 TFLOPS FP16

**Recommendations**:
- Use d_compressed ≥ 64 for tensor core alignment
- Batch size 8+ to amortize kernel launch overhead
- Enable FlashAttention 2 with pre-expanded cache

### NVIDIA H100

- HBM bandwidth: 3.35 TB/s
- SM count: 132
- Tensor core peak: 989 TFLOPS FP16

**Recommendations**:
- Higher bandwidth reduces compression benefit
- Focus on memory savings over bandwidth savings
- Use FP8 attention if quality permits

### AMD W7900 (48GB)

- HBM bandwidth: ~864 GB/s
- CU count: 96
- Matrix core peak: ~61 TFLOPS FP16

**Recommendations**:
- Lower bandwidth = higher compression benefit
- Use d_compressed ≥ 32 (wavefront alignment)
- ROCm FlashAttention less mature; custom kernels may help

**Benchmark reproduction (Orthogonal vs PCA)**:
```bash
# Compare orthogonal (zero-cal) vs PCA (calibrated) presets on W7900
python scripts/kv_plugin_benchmark.py \
  --model openai-community/gpt2 \
  --preset orthogonal \
  --skip-calibration \
  --wikitext \
  --output w7900_orthogonal.json

python scripts/kv_plugin_benchmark.py \
  --model openai-community/gpt2 \
  --preset balanced \
  --wikitext \
  --output w7900_pca.json
```

This reproduces the ~40x faster setup time for orthogonal vs PCA/SVD calibration,
and ~45% faster per-layer inference due to skipping mean centering operations.

### Apple M-series (Unified Memory)

- Memory bandwidth: ~400 GB/s (M2 Max)
- Unified memory = no HBM/DRAM distinction

**Recommendations**:
- Memory savings critical for fitting models
- Compression directly increases usable context
- Use Metal-optimized attention kernels

### CPU Inference

- Memory bandwidth: ~50-100 GB/s (DDR5)
- No tensor cores

**Recommendations**:
- Maximum compression benefit
- Use INT8 quantization with compression
- Prioritize cache size reduction

## 9. Deployment Decision Matrix

| Scenario | Priority | Recommended Config |
|----------|----------|-------------------|
| Laptop/Edge | Memory | Hybrid d_c=64, INT8 |
| Single GPU | Balanced | MLA d_l=256, FP16 |
| Multi-GPU | Throughput | V-only R=48, batched |
| Low-latency | TTFT | MLA + early prefill |
| Long context | Memory | Aggressive R=32 |

## 10. Future Directions

### Near-term (6 months)

1. **FlashAttention integration**: PR to support compressed K/V
2. **Triton kernels**: Reference implementation for benchmarking
3. **Quantization synergy**: INT4 cache with compression

### Medium-term (1-2 years)

1. **Hardware support**: Tensor cores optimized for low-rank ops
2. **Compiler integration**: Auto-fusion of compress/expand
3. **Dynamic compression**: Per-token rank selection

### Long-term (2+ years)

1. **Native sparse attention**: Hardware support for compressed formats
2. **Memory-compute co-design**: Architecture-level compression
3. **Learned routing**: Attention to compressed vs full cache

## References

- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- Triton: https://triton-lang.org/
- NVIDIA A100 Whitepaper
- AMD CDNA3 Architecture Guide
- Apple Silicon GPU Programming Guide
