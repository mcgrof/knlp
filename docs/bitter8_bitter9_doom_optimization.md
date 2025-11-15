# Bitter8/Bitter9: Doom-Style GPU Memory Optimization

## Background

After implementing the double-pass elimination in bitter7 (which showed
1.8x CPU speedup but minimal GPU improvement), we need a different
approach to reduce GPU memory bandwidth pressure.

GPU profiling showed bitter7 with high memory access time (18.25%) and
low compute utilization (42%), indicating a memory-bandwidth bottleneck
rather than a computation bottleneck.

## The Doom Hack: Fast Inverse Square Root

The famous Quake III "Doom hack" used a bit-level trick to compute fast
inverse square roots for lighting calculations. While we can't use the
exact C bit manipulation in PyTorch, we can leverage PyTorch's optimized
`rsqrt()` function which implements fast inverse square root in hardware.

### Mathematical Foundation

Bitter7 computes: `importance = |w| × (|v| + ε)^0.25`

The 4th root can be computed as:
- Standard: `x^0.25 = (x^0.5)^0.5 = sqrt(sqrt(x))`
- Fast rsqrt: `rsqrt(x) = x^(-0.5) = 1/sqrt(x)`
- Double rsqrt: `rsqrt(rsqrt(x)) = (x^(-0.5))^(-0.5) = x^0.25` ✓

So we can replace `pow(0.25)` with `rsqrt(rsqrt(x))`.

### Bitter8: FP16 + Fast Inverse Sqrt

Bitter8 combines two memory optimizations:

1. **FP16 precision**: Reduces memory bandwidth by 2x
2. **Fast rsqrt**: Hardware-accelerated inverse square root

```python
# Bitter7 (baseline):
variance_importance = (torch.abs(v) + 1e-8) ** 0.25
importance = torch.abs(w) * variance_importance

# Bitter8 (optimized):
w_fp16 = w.to(torch.float16)
v_fp16 = v.to(torch.float16)
v_abs = torch.abs(v_fp16) + 1e-8
fourth_root = torch.rsqrt(torch.rsqrt(v_abs))  # Doom-style!
importance = torch.abs(w_fp16) * fourth_root
importance = importance.float()  # back to fp32 for threshold
```

**Expected benefits:**
- 2x memory bandwidth reduction (FP16 vs FP32)
- Faster 4th root computation (rsqrt is hardware-optimized)
- Same pruning quality as bitter7
- 30-50% speedup in pruning time

### Bitter9: Bitter8 + torch.compile

Bitter9 adds graph-level optimization on top of bitter8:

```python
@torch.compile(mode="reduce-overhead", fullgraph=True)
def _compute_bitter9_importance_compiled(w, v):
    """Compiled version with kernel fusion."""
    w_fp16 = w.to(torch.float16)
    v_fp16 = v.to(torch.float16)
    v_abs = torch.abs(v_fp16) + 1e-8
    fourth_root = torch.rsqrt(torch.rsqrt(v_abs))
    importance = torch.abs(w_fp16) * fourth_root
    return importance.float()
```

torch.compile provides additional optimizations:
- **Kernel fusion**: Combines multiple operations into single kernels
- **Memory layout optimization**: Reduces intermediate tensor allocations
- **Graph-level optimization**: Optimizes entire computation graph

**Expected benefits over bitter8:**
- Additional 20-40% speedup from kernel fusion
- Reduced memory fragmentation
- Optimized memory access patterns

**Total expected benefits over bitter7:**
- 50-70% total speedup
- Lower GPU memory access time
- Higher compute utilization

## Implementation Details

### Why FP16 is Safe for Importance Computation

Pruning doesn't require high precision - we only need approximate
ranking of parameters. FP16 provides:
- 10-bit mantissa (vs 23-bit in FP32)
- Same exponent range as FP32
- Sufficient precision for relative importance ranking

The threshold comparison is done in FP32 after converting back, ensuring
mask decisions are stable.

### Why rsqrt is Faster than pow

Modern GPUs have dedicated hardware for:
- `rsqrt()`: Single instruction (hardware inverse sqrt)
- `pow(x, 0.25)`: Multiple instructions (exp, log, mul)

Using `rsqrt(rsqrt(x))` is 2-3x faster than `pow(x, 0.25)` on most GPUs.

### torch.compile Kernel Fusion

Without compilation, bitter8 creates intermediate tensors:
```
temp1 = w.to(torch.float16)      # allocation
temp2 = v.to(torch.float16)      # allocation
temp3 = torch.abs(temp2)         # allocation
temp4 = temp3 + 1e-8             # allocation
temp5 = torch.rsqrt(temp4)       # allocation
temp6 = torch.rsqrt(temp5)       # allocation
...
```

With torch.compile, these are fused into minimal kernels that
reuse memory and optimize access patterns.

## Expected GPU Metrics

Based on the optimizations, we expect bitter8/bitter9 to achieve:

**Memory Access Time:**
- Bitter7: 18.25% (baseline)
- Bitter8: ~14-15% (2x bandwidth reduction from FP16)
- Bitter9: ~12-13% (additional fusion benefits)

**Compute Utilization:**
- Bitter7: 42% (baseline)
- Bitter8: ~50-55% (less time waiting on memory)
- Bitter9: ~55-60% (optimized kernels)

## Validation Plan

1. **CPU benchmark**: Verify rsqrt is faster than pow
2. **Quality check**: Ensure FP16 doesn't degrade pruning quality
3. **GPU profiling**: Measure actual memory bandwidth reduction
4. **Training run**: Full 10K iteration validation

## References

- Fast Inverse Square Root: https://en.wikipedia.org/wiki/Fast_inverse_square_root
- PyTorch torch.rsqrt: https://pytorch.org/docs/stable/generated/torch.rsqrt.html
- PyTorch torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
