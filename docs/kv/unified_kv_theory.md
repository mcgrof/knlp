# Unified KV Compression Theory

This document synthesizes three independent research threads into a unified
framework for understanding and designing KV cache compression:

1. **EGG-style energy gating** (v36)
2. **MLA latent-space compression** (DeepSeek-style)
3. **KVSplice learned projections** (architecture-level)

## 1. The Fundamental Insight: KV Behaves as Low-Rank

All three approaches exploit the same underlying observation: **Key-Value
activations in trained transformers empirically exhibit low-rank structure**.
This is not an architectural guarantee but a learned property - models appear
to use far fewer effective dimensions than available. See the rationale in
https://github.com/mcgrof/attention-low-rank for detailed analysis.

### Empirical Evidence

**From EGG energy probing (GPT-2 124M)**:
- Top-32 dimensions capture ~64% of V energy (out of 64 total)
- ~54 dimensions needed for 90% energy concentration
- Energy distribution follows power-law decay across channels

**From KVSplice training experiments**:
- 50% compression (256→128) adds only 0.5-1.4% quality loss
- 90% compression (256→26) can *improve* perplexity (acts as regularizer)
- Compression tightens spectral decay: 79.9% energy@32 vs baseline 63.3%

**From MLA architecture analysis**:
- Shared latent space (768→256) loses minimal information
- Latent channels behave like learned low-rank mixtures
- 6x compression with negligible quality impact

### Mathematical Foundation

The K/V matrices can be approximated via SVD:

```
KV = U · Σ · Vᵀ
```

Where:
- `U` contains the left singular vectors (token patterns)
- `Σ` contains singular values (energy per mode)
- `Vᵀ` contains the right singular vectors (channel patterns)

The rapid decay of singular values (`Σ`) explains why low-rank
approximations work: most information is concentrated in the top-r modes.

## 2. Three Approaches to Compression

### 2.1 EGG: Post-Hoc Channel Selection

**When**: After model training, before/during inference
**How**: Select top-k channels by energy score
**Memory**: O(r) per token instead of O(d)

```
V_compressed = gather(V, top_k_indices)  # [B,H,T,k]
```

**Strengths**:
- No retraining required
- Works on any pretrained model
- Zero-cost calibration (just energy computation)

**Weaknesses**:
- Hard selection may miss channel interactions
- Doesn't reshape the learned representation
- Static selection per head/layer

**Best for**: Drop-in compression of pretrained models

### 2.2 MLA: Architecture-Level Bottleneck

**When**: During architecture design and training
**How**: Force KV through low-dimensional latent space
**Memory**: O(d_latent) instead of O(n_heads × d_head)

```
latent = to_latent(x)           # [B,T,256]
K, V = from_latent(latent)      # Expand on-the-fly
cache = latent                   # 6x compression
```

**Strengths**:
- Model learns to use latent space efficiently
- Shared representation across heads
- End-to-end trainable

**Weaknesses**:
- Requires training from scratch (or fine-tuning)
- Fixed compression ratio once trained
- Cannot adapt to inference-time constraints

**Best for**: New models where training compute is available

### 2.3 KVSplice: Learned Low-Rank Projection

**When**: During training, applied to MLA latent
**How**: Learn orthogonal projection matrices
**Memory**: O(d_compressed) where d_compressed < d_latent

```
Z = W_compress(latent)          # [B,T,128]
latent_hat = W_expand(Z)        # Reconstruct
cache = Z                        # 12x total compression
```

**Strengths**:
- Trained end-to-end with model
- Acts as regularizer (can improve quality)
- Stacks with MLA for multiplicative compression

**Weaknesses**:
- Requires training
- Fixed projection after training

**Best for**: Maximum compression with quality guarantees

### 2.4 Orthogonal: Analytic KVSplice

**When**: Deploy time (zero-calibration) or with optional calibration
**How**: Random orthonormal basis, transpose for expansion
**Memory**: O(d_compressed) where d_compressed < d_input

```
W_compress: [d_input, d_compressed], orthogonal initialization
W_expand:   [d_compressed, d_input] = W_compress.T

Z = x @ W_compress           # Compress to low-rank
x_hat = Z @ W_expand         # Expand back (W_expand = W_compress.T)
cache = Z                     # Store compressed
```

**Strengths**:
- No calibration required (orthogonal init provides valid projection)
- 40x faster setup than PCA/SVD (no eigendecomposition)
- 45% faster inference than PCA (no mean centering)
- With calibration, matches SVD/PCA quality
- Same compression ratio as PCA/SVD methods

**Weaknesses**:
- Without calibration, higher reconstruction error than data-adapted PCA
- Fixed basis (not data-dependent without calibration)

**Best for**: Zero-calibration deployment, rapid iteration, edge devices

### Why Orthogonal = Analytic KVSplice

The `OrthogonalCompressor` in KV Plugin v3 is the **analytic version** of
KVSplice's learned latent projection. The connection:

1. **KVSplice (trained)**: Learns `W_compress` and `W_expand` via gradient
   descent, minimizing reconstruction loss over training data

2. **Orthogonal (analytic)**: Uses orthogonal initialization for `W_compress`,
   sets `W_expand = W_compress.T` by construction

The mathematical relationship:
```
KVSplice optimal:  W_expand* = W_compress† (pseudo-inverse)
Orthogonal:        W_expand = W_compress.T  (exact inverse for orthonormal W)
```

When `W_compress` is orthonormal, `W_compress.T @ W_compress = I`, making the
compress-expand operation a perfect projection onto a subspace. With optional
calibration, the orthogonal compressor finds the subspace that maximizes
variance retention (equivalent to PCA principal subspace).

**Calibration trade-off**:
- Zero-calibration: Use random orthonormal basis. Fast deploy, ~2-3x higher
  reconstruction MSE than calibrated SVD
- With calibration: Find SVD/PCA subspace. Matches calibrated methods exactly

This makes `orthogonal` presets ideal for:
- **Ship now, tune later**: Deploy with zero calibration, add calibration
  when data becomes available
- **Fast prototyping**: Quick iteration without calibration overhead
- **Edge deployment**: No eigendecomposition compute required

## 3. Unification: The Low-Rank Compression Spectrum

All three approaches express the same mathematical operation:

```
Original:     KV ∈ ℝ^{d}
Compressed:   Z  ∈ ℝ^{r}    where r << d
Reconstruct:  KV_hat = f(Z)
```

The key differences are **when** and **how** the projection is computed:

| Approach | Projection | When Computed | Learns From |
|----------|------------|---------------|-------------|
| EGG TopK | Index gather | Calibration (once) | Energy statistics |
| EGG Soft | Learned gate | Training | Gradient descent |
| MLA | Fixed linear | Architecture | End-to-end training |
| KVSplice | Learned linear | Training | Reconstruction loss |
| Orthogonal | Fixed linear | Initialization | Random orthonormal |

### The Compression Hierarchy

These approaches can be **composed** for multiplicative savings:

```
Standard GPT-2:
  K,V = [B, H, T, d_head]           # 768 dims total
  Cache memory: O(H × T × d_head)

MLA alone (6x):
  latent = [B, T, d_latent]          # 256 dims
  Cache memory: O(T × d_latent)

MLA + KVSplice (12x):
  Z = [B, T, d_compressed]           # 128 dims
  Cache memory: O(T × d_compressed)

MLA + KVSplice + EGG (18-24x):
  Z_gated = [B, T, r]                # 64-85 dims
  Cache memory: O(T × r)
```

## 4. Hybrid Gate+Latent Operator

The unified theory suggests a **hybrid operator** that combines:
- MLA-style latent projection (architecture-level)
- Energy-based gating (post-hoc selection)
- Learned reconstruction (end-to-end training)

### Architecture

```python
class HybridKVCompressor:
    """Combines latent projection with energy gating."""

    def __init__(self, d_in, d_latent, d_compressed):
        # MLA-style latent projection
        self.to_latent = nn.Linear(d_in, d_latent)

        # Energy-based gate (learned or calibrated)
        self.gate = EnergyGate(d_latent, d_compressed)

        # Reconstruction
        self.from_latent = nn.Linear(d_latent, d_in)

    def compress(self, x):
        latent = self.to_latent(x)       # d_in → d_latent
        gated = self.gate(latent)         # d_latent → d_compressed
        return gated                       # Cache this

    def decompress(self, z):
        latent = self.gate.expand(z)      # d_compressed → d_latent
        return self.from_latent(latent)   # d_latent → d_in
```

### Why This Works

1. **Latent projection** (MLA) captures cross-head structure
2. **Energy gating** (EGG) selects important latent channels
3. **Learned reconstruction** (KVSplice) minimizes information loss

The composition is more powerful than any single approach because:
- MLA provides a learned, shared representation
- EGG identifies which latent dimensions matter
- KVSplice learns to project and reconstruct optimally

## 5. Design Principles for KV Plugin v2.5

Based on the unified theory, KV Plugin v2.5 should:

### Principle 1: Layer-Adaptive Compression

Different layers have different energy concentration. Early layers often
need more dimensions; later layers can be compressed more aggressively.

```python
# Per-layer optimal rank from energy probe
layer_ranks = calibrate_per_layer_ranks(model, target_ppl_delta=0.02)
```

### Principle 2: Compose Rather Than Replace

Stack compression operators for multiplicative savings:
- Base: MLA latent (6x)
- Add: KVSplice projection (2x more = 12x total)
- Add: Energy gating (1.5x more = 18x total)

### Principle 3: Calibrate Then Fix

Compute expensive statistics (energy, SVD) once during calibration.
Use static gates/projections at inference for zero overhead.

```python
# Calibration phase (expensive, once)
energies = collect_kv_energies(model, calibration_data)
gates = create_static_gates(energies, target_ranks)

# Inference phase (cheap, every forward)
V_compressed = V[..., gates.indices]  # Just index gather
```

### Principle 4: Trade-off Curve Awareness

Provide a clear decision tree based on memory budget:

| Memory Budget | Approach | Expected PPL Impact |
|---------------|----------|---------------------|
| ≤ 25% | Hybrid EGG+Latent k=16 | ~15-25% |
| ≤ 50% | Hybrid EGG+Latent k=32 | ~5-10% |
| ≤ 75% | KVSplice alone | ~0.5-2% |
| ≤ 87.5% | MLA alone | ~0% |

## 6. Mathematical Appendix

### Energy-Optimal Rank Selection

Given energy scores `E[d]` for each dimension, the optimal rank `r` for
target energy retention `τ` is:

```
r* = argmin_r { r : Σ_{i=1}^{r} E[sorted[i]] ≥ τ × Σ_d E[d] }
```

For GPT-2 with τ=0.90, this gives r* ≈ 54 (out of 64).

### Composition of Compressions

If compression A achieves ratio `ρ_A` and B achieves `ρ_B`, the
composed compression achieves:

```
ρ_total = ρ_A × ρ_B
```

Example:
- MLA: ρ_A = 768/256 = 3x
- KVSplice 50%: ρ_B = 256/128 = 2x
- Total: ρ_total = 3 × 2 = 6x (vs standard attention)

Wait, MLA compresses 1536→256 (6x), so:
- MLA: ρ_A = 1536/256 = 6x
- KVSplice 50%: ρ_B = 256/128 = 2x
- Total: ρ_total = 6 × 2 = 12x

### Reconstruction Error Bounds

For optimal rank-r approximation via SVD:

```
||KV - KV_hat||_F ≤ Σ_{i=r+1}^{d} σ_i
```

This bounds the worst-case reconstruction error. EGG energy gating
provides a data-dependent approximation that can be tighter for
specific input distributions.

## 7. Research Directions

### Open Questions

1. **Per-head vs per-layer gating**: Should each head have its own gate,
   or can we use a shared gate per layer?

2. **Dynamic gating**: Can we adapt compression ratio per-token based
   on content complexity?

3. **K vs V asymmetry**: Is K more compressible than V, or vice versa?
   Initial evidence suggests V is more compressible.

4. **Attention-aware selection**: Can we use attention patterns to
   identify which K/V dimensions will be attended to?

### Future Work

- **v38**: Integration with MLA+KVSplice training
- **v39**: Hardware-optimized kernels for compressed attention
- **v40**: Neural architecture search over compression configurations

## References

- EGGROLL Paper: arXiv:2511.16652
- DeepSeek-V2/V3: MLA architecture
- xKV Paper: Low-rank KV compression validation
- EGG mechanisms: `docs/kv/egg_mechanisms.md`
- EGG energy probe: `scripts/egg_kv_energy_probe.py`
- KVSplice implementation: `gpt2/mla.py`
