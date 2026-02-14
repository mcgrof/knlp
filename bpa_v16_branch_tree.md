# BPA v16 Branch Tree

## Outcome: CASE A — Hybrid wins (precision-dominant path)

Mixed-precision S2 (18xINT4 + 6xINT8) achieves kv_ratio=0.333 with
FULL PASS@1%. Hybrid_r24_S2 (structural K + precision V) achieves
0.363. Both break the 0.5 wall decisively. Precision (sensitivity-
guided INT8/INT4 allocation) dominates structure (rope_complex SVD)
for raw storage reduction.

## Priority Branches

### A1: Automated bit allocation via gradient sensitivity [HIGH]
v16 used static layer sensitivity from v15 Phase 4 (single-pass INT4
perturbation). A learned allocator could optimize the INT8/INT4
boundary per-layer using training-time gradient magnitudes or Fisher
information. Target: find whether 4 INT8 layers suffice (currently 6)
or whether INT3/INT2 is viable for tolerant layers.

### A2: Extend to larger models [HIGH]
v16 results are on Qwen2.5-0.5B (24 layers, 2 KV heads). Validate
on Qwen2.5-7B and Llama-3-8B to confirm sensitivity patterns
generalize. Larger models may have more redundant layers, enabling
even more aggressive INT4 allocation.

### A3: Bandwidth-bound latency validation [MEDIUM]
v16 Phase 4 showed no latency improvement because the W7900 at L=16K
is compute-bound. Test at L=65K+ or with batch>1 to find the
bandwidth-bound regime where 66% memory reduction translates to
actual throughput gains.

### A4: Native INT4 kernel integration [MEDIUM]
Current implementation quantizes then dequantizes (cast-back). Real
INT4 kernels (e.g., GPTQ-style or AWQ marlin kernels) would store
KV in packed INT4 and fuse dequantization into attention. This would
unlock actual memory savings during inference.

### A5: Fine-grained rank allocation [LOW]
v16 used uniform rank=24 across all layers for rope_complex. Layer
sensitivity data suggests some layers could tolerate lower rank (e.g.,
rank=16 for tolerant layers, rank=32 for sensitive ones). The storage
benefit is small since INT4 V dominates, but could improve quality
margins.

## Deprecated Branches

### D1: Push rope_complex below rank=16
The rank cliff at 24 is sharp and the quality degradation at rank=16
is irreversible without retraining. The magnitude manifold's
dimensionality is an intrinsic property of the model.

### D2: Learned KVSplice compression
v15 Phase 3 showed all learned segment compressors fail catastrophically.
Mixed precision achieves better results without training.

### D3: Logit-space or latent-space compression
MLA/splice methods fail due to lossy random projections (v13-v14b).
Direct quantization respects the native representation.

## v17 Recommendation

Focus on A1 (automated bit allocation) and A2 (larger model
validation). The sensitivity-guided mixed-precision approach is
simple, effective, and requires no training. The key open question
is whether the layer sensitivity patterns transfer across model
families and scales, or whether per-model calibration is always
required.
