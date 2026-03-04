# BPA v29: Fused INT4 FlashAttention on AMD W7900

## 1. Problem Statement

BPA v28 identified a gap in the systems story: the dequant-then-
attention path (INT4 KV cache -> dequant to FP16 -> SDPA) adds
intermediate memory traffic that a fused kernel could eliminate.
On NVIDIA H100, this was not a bottleneck because decode is
compute-bound for 7B models. The question for v29 is whether
the AMD Radeon Pro W7900, with different memory bandwidth and
compute characteristics, would benefit from a fused INT4 attention
kernel.

The experiment is a strict A/B test with one variable: the
attention kernel implementation.

**Pipeline A** (control): INT4 packed KV -> dequant to FP16 ->
expand GQA heads -> torch SDPA.

**Pipeline B** (experimental): INT4 packed KV -> Triton kernel
that unpacks in registers, applies scales, and computes attention
without ever materializing FP16 KV tensors in global memory.

## 2. Kernel Design Overview

The fused Triton kernel (`_fused_int4_attn_kernel`) handles
single-token decode (M=1) for the Qwen2.5-0.5B attention
configuration (14 query heads, 2 KV heads via GQA, head_dim=64,
group_size=32).

Key design decisions:

- **INT4 packing**: 2 values per uint8 byte. Low nibble = even
  indices, high nibble = odd indices. Packed shape:
  `[B, n_kv, T, hd//2]`.

- **Register-resident unpack**: Bitwise AND/shift to extract
  nibbles, cast to float32, apply per-group scale. No intermediate
  FP16 tensors written to memory.

- **Group scaling via broadcast mask**: Instead of 2D tensor
  slicing (unsupported in Triton 3.2.0 on ROCm), a broadcast
  mask selects between group 0 and group 1 scales per packed
  byte position. This avoids all slice operations.

- **Even/odd Q decomposition**: Q is loaded as separate even-
  and odd-indexed vectors matching the packed INT4 layout. The
  dot product QK^T is computed as:
  `sum(q_even * k_low) + sum(q_odd * k_high)`.

- **Online softmax**: Standard FlashAttention-style running max
  and sum with rescaling, enabling streaming over arbitrary
  sequence lengths.

- **Output accumulation**: Separate even/odd accumulators avoid
  scatter-gather into a single HEAD_DIM vector inside the loop.
  Final interleaved store at the end.

- **GQA expansion**: Each program instance maps to one
  (batch, query_head) pair. The kv_head_id is computed as
  `query_head // n_rep` to share KV across grouped heads.

## 3. Benchmark Methodology

**Hardware**: AMD Radeon Pro W7900 (48 GB VRAM, ROCm 6.2.4)

**Software**: PyTorch 2.6.0+rocm6.2.4, Triton 3.2.0

**Model config**: n_heads=14, n_kv_heads=2, head_dim=64,
group_size=32 (Qwen2.5-0.5B attention dimensions)

**Grid**: L in {2048, 4096, 8192, 16384}, B in {1, 4, 8}
(12 configurations)

**Protocol**:
1. Generate random FP16 KV tensors
2. Quantize and pack to INT4
3. Warmup (5 runs)
4. Timed runs (5 repeats)
5. Record mean, std, min, max latency
6. Measure peak GPU memory allocation

**Correctness validation**: Pipeline B output compared against
Pipeline A for B in {1, 2}, T in {64, 256, 1024}. Maximum
absolute error < 0.001 across all configurations, confirming
numerical equivalence within FP16 precision.

## 4. Results Table

| B | L | A (ms) | B (ms) | Speedup | Lat% | Thru% | Peak A (MB) | Peak B (MB) |
|---|------|--------|--------|---------|------|-------|-------------|-------------|
| 1 | 2048 | 0.272 | 0.116 | 2.34x | +57% | +134% | 8.7 | 0.3 |
| 4 | 2048 | 0.364 | 0.120 | 3.05x | +67% | +205% | 34.7 | 1.2 |
| 8 | 2048 | 0.522 | 0.132 | 3.96x | +75% | +296% | 69.5 | 2.4 |
| 1 | 4096 | 0.418 | 0.180 | 2.33x | +57% | +133% | 17.4 | 0.6 |
| 4 | 4096 | 0.662 | 0.202 | 3.27x | +69% | +227% | 69.5 | 2.4 |
| 8 | 4096 | 1.126 | 0.224 | 5.04x | +80% | +404% | 139.0 | 4.7 |
| 1 | 8192 | 0.737 | 0.349 | 2.11x | +53% | +111% | 34.7 | 1.2 |
| 4 | 8192 | 1.340 | 0.339 | 3.95x | +75% | +295% | 139.0 | 4.7 |
| 8 | 8192 | 2.149 | 0.408 | 5.27x | +81% | +427% | 277.9 | 9.5 |
| 1 | 16384 | 1.334 | 0.606 | 2.20x | +55% | +120% | 69.5 | 2.4 |
| 4 | 16384 | 2.535 | 0.637 | 3.98x | +75% | +298% | 277.9 | 9.5 |
| 8 | 16384 | 3.869 | 0.709 | 5.46x | +82% | +446% | 555.8 | 18.9 |

Key observations:

- **Speedup range**: 2.11x to 5.46x across all configurations.
- **Batch scaling**: Speedup increases with batch size. At B=1,
  speedup is 2.1-2.3x. At B=8, speedup reaches 4.0-5.5x.
- **Length scaling**: Speedup is relatively stable across context
  lengths for a given batch size, with slight increase at longer
  sequences.
- **Pipeline A is memory-bandwidth bound**: Latency scales linearly
  with B*L, consistent with the intermediate FP16 materialization
  dominating execution time.
- **Pipeline B is compute-bound**: The fused kernel keeps data in
  registers and the latency growth with B*L is much flatter.

## 5. Memory Accounting Comparison

Per-token memory traffic per KV layer:

| Component | Bytes/token |
|-----------|-------------|
| Dense FP16 KV (2 heads x 64 dims x 2 bytes x K+V) | 512 |
| INT4 packed (2 heads x 32 bytes x K+V) | 128 |
| INT4 scales (2 heads x 2 groups x 2 bytes x K+V) | 16 |
| INT4 total (packed + scales) | 144 |

Pipeline memory traffic per decode token (all T tokens):

| Pipeline | Read (bytes/tok) | Notes |
|----------|------------------|-------|
| A: dequant+SDPA | 144 + 512 + 512 = 1168 | Read INT4, write FP16, read FP16 for SDPA |
| B: fused | 144 | Read INT4 only, unpack in registers |

**Traffic reduction**: 87.7% (from 1168 to 144 bytes/token).

Peak GPU memory allocation comparison confirms the theoretical
reduction. Pipeline A must allocate the expanded FP16 KV tensors
(with GQA expansion to 14 heads), while Pipeline B only reads
the packed INT4 data with no intermediate allocations:

| Config | Peak A | Peak B | Reduction |
|--------|--------|--------|-----------|
| B=8, L=16K | 555.8 MB | 18.9 MB | 29.4x |
| B=4, L=8K | 139.0 MB | 4.7 MB | 29.6x |
| B=1, L=2K | 8.7 MB | 0.3 MB | 29.0x |

The ~29x memory reduction is consistent with eliminating the
GQA-expanded FP16 intermediate (14/2 = 7x head expansion, plus
FP16 vs INT4 = ~3.56x compression, gives 7 * 3.56 = ~25x
theoretical).

## 6. Decision Conclusion

**All three decision criteria are met:**

| Criterion | Threshold | Measured | Verdict |
|-----------|-----------|----------|---------|
| Latency improvement | >= 5% | 53-82% | **PASS** |
| Throughput improvement | >= 10% | 111-446% | **PASS** |
| KV memory traffic reduction | >= 20% | 87.7% | **PASS** |

**Conclusion: A fused INT4 FlashAttention Triton kernel IS worth
implementing on AMD W7900.**

The W7900 is memory-bandwidth-limited for the dequant-then-
attention path. The intermediate FP16 KV materialization (plus
GQA head expansion) dominates execution time. The fused kernel
eliminates this bottleneck entirely by keeping unpacked values
in registers.

This is the opposite finding from v28 on H100, where the decode
path was compute-bound and the dequant overhead was in the noise
(0.05-0.27%). The W7900 has lower memory bandwidth relative to
its compute throughput, making the memory traffic reduction from
the fused kernel translate directly into latency gains.

**Implications for production**: The 2-5x speedup applies to the
attention-only component of decode. In an end-to-end model
serving scenario, the attention time fraction determines the
realized benefit. For memory-bandwidth-bound serving (batch
decode, long context), this kernel could be a significant
optimization. For compute-bound regimes (short context, small
batch), the benefit would be proportionally smaller.

## Files

| File | Description |
|------|-------------|
| `artifacts/v29_flash/kernel_bench.json` | Full benchmark results |
| `artifacts/v29_flash/bench_results.csv` | CSV results table |
| `artifacts/v29_flash/latency_vs_L.png` | Latency vs context length plot |
| `artifacts/v29_flash/throughput_vs_batch.png` | Throughput vs batch size plot |
| `artifacts/v29_flash/kv_bytes_per_token.png` | KV memory traffic comparison |
| `scripts/v29_flash_bench.py` | Benchmark script (both pipelines) |
