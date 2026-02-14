# BPA v15 Branch Tree

## Outcome: BRANCH A+D Hybrid

rope_complex (M1-B) is the only structural method to pass, achieving
FULL PASS at 1% tolerance. But its practical advantage over INT8 is
marginal (0.550 vs 0.564 kv_bytes_ratio). This is a hybrid of
Branch A (M1 wins) and Branch D (only INT8 wins for practical use).

## Branch A: rope_complex wins structurally

The complex-plane grouping exploits RoPE magnitude as a low-rank
target. Next steps:

### A1: Push rank lower
- Current: rank_frac=0.5 (rank 32/64) gives FULL PASS
- Test rank_frac in {0.25, 0.125} to find break point
- If rank 16 passes: kv_bytes_ratio drops to ~0.4
- If rank 8 passes: kv_bytes_ratio drops to ~0.3
- Expected: rank 16 may work (magnitude has ~19 significant
  components from v14b analysis), rank 8 likely fails

### A2: Combine with INT8 V storage
- Current implementation dequantizes INT8 V back to float for cache
- True INT8 storage would save an additional 2x on V
- K: low-rank magnitude (rank/64 of float) + exact phase (float)
- V: INT8 (1 byte per element)
- This could push kv_bytes_ratio to 0.35-0.4

### A3: Adaptive rank per layer
- Phase 4 shows uniform sensitivity across layers for INT8/INT4
- But rope_complex might benefit from per-layer rank allocation
- Calibrate optimal rank per layer based on magnitude spectrum
- Use more rank in sensitive layers (0, 8), less in tolerant ones

### A4: Complex-plane grouping for V
- V is full-rank in raw space, but has it been tested in complex
  space (treating RoPE pairs)?
- V may have compressible structure in polar coordinates too
- If V magnitude is low-rank: compound compression gains

## Branch D: INT8 is the pragmatic winner

INT8 achieves identical quality (PASS@1%) with simpler implementation
and no calibration overhead. Next steps:

### D1: Native INT8 KV cache
- Current implementation quantize+dequantizes (simulated)
- Implement native INT8 storage in KV cache for real memory savings
- Requires custom attention kernel (INT8 @ FP16 GEMM)
- AMD ROCm supports INT8 GEMM via hipBLASLt

### D2: Mixed INT8/INT4 with layer schedule
- Phase 4 shows 23/24 layers tolerate per-layer INT4
- But cumulative INT4 fails across all layers
- Try: INT4 for 12 most tolerant layers, INT8 for 12 sensitive
- Could achieve ~0.4 kv_bytes_ratio if cumulative error stays <3%

### D3: INT6 or asymmetric quantization
- INT4 is too aggressive, INT8 is easy
- Test 6-bit quantization (simulated) or asymmetric INT8
- Might find the sweet spot between 4 and 8 bits

## Branch B: Logit-space methods (FAIL)

Nystrom and logit sketch both failed to achieve meaningful
compression. Not recommended for further investment.

### B1: If revisited
- Would need learned landmark selection (not uniform)
- Would need trained sketch parameters (not random projection)
- Logit-space methods fundamentally fight the curse of
  dimensionality: approximating O(L^2) attention with O(mL)
  landmarks requires m to be large enough that savings vanish

## Branch C: KVSplice variants (FAIL)

All M4-M6 variants failed. The 4x segment compression is too
aggressive. Not recommended for further investment.

### C1: If revisited
- Try 2x segment compression (pairs instead of quads)
- Train for much longer (50K+ steps)
- Use curriculum (start at 2x, tighten to 4x)
- Per-layer rank allocation based on Phase 4 sensitivity

## Branch E: Model generality

v15 was tested on Qwen2.5-0.5B only. The complex-plane insight
depends on RoPE, so it applies to any RoPE-based model (Llama,
Mistral, etc.). Key questions:

### E1: Larger models
- Qwen2.5-7B, 14B, 72B have more KV heads and larger head_dim
- More KV heads means more total K/V memory to compress
- Larger head_dim might have even lower effective rank

### E2: Non-RoPE models
- GPT-4, etc. may use learned positional embeddings
- Complex-plane grouping is specific to RoPE
- Would need a different structural decomposition

## Priority Ranking

1. **A1** (push rank lower) - highest potential, low effort
2. **D2** (mixed INT8/INT4) - pragmatic, uses Phase 4 data
3. **A2** (INT8 V storage) - engineering win, no research risk
4. **D1** (native INT8 cache) - production deployment
5. **A3** (adaptive rank per layer) - moderate potential
6. Everything else: low priority or not recommended
