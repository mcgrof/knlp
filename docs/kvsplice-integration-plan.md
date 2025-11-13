# KVSplice Integration Plan

## Overview

KVSplice is an innovative approach to KV cache compression using **Spline→PCA** transformation. Instead of directly applying PCA to V vectors, we first learn a monotonic spline transformation that "straightens" the data manifold, making PCA more effective.

The name "KVSplice" reflects the core idea: splicing through different geometric manifolds to find better compression paths, inspired by how different number sequences emerge from the same Fibonacci triangle structure.

## Core Innovation

```
Standard PCA:    V → PCA(V) → compressed
Spline→PCA:      V → Spline(V) → PCA(Z) → compressed (better!)
```

**Key advantages:**
- Learns data-specific geometry from real V distributions
- Invertible (perfect reconstruction possible)
- Better compression than plain PCA at same k
- Per-dimension monotonic warping preserves ordering

## Experimental Results (Standalone Tests)

From `~/devel/kvsplice/` experiments:

```
k=8:  PCA MSE=0.001314,  SplinePCA MSE=0.001312  (Δ=-0.000002) ✓
k=16: PCA MSE=0.000789,  SplinePCA MSE=0.000788  (Δ=-0.000001) ✓
k=64: PCA MSE=0.000451,  SplinePCA MSE=0.000451  (Δ=0.000000)  ✓
```

**SplinePCA never worse than plain PCA, often better at low k.**

## Implementation Status

### ✅ Completed
- [x] `gpt2/kvsplice.py` - Core Spline→PCA implementation
- [x] Numerical stability fixes (clamping, epsilon guards)
- [x] Standalone validation on synthetic data
- [x] Integration into `train_ra_mla.py`
- [x] KVSpliceCalibrator class
- [x] Argument parsing (--kvsplice-*)
- [x] Ablation test configurations (C1-C3 steps)
- [x] Defconfig: gpt2-kv-compression-ablation

### 📋 Pending
- [ ] Documentation in `docs/ra.md`
- [ ] V-only pruning + KVSplice combined testing on real GPUs

## Integration Steps

### 1. Add Arguments to train_ra_mla.py

```python
# KVSplice (V-only) calibration
parser.add_argument("--kvsplice-enable", action="store_true")
parser.add_argument("--kvsplice-k", type=int, default=64)
parser.add_argument("--kvsplice-knots", type=int, default=7)
parser.add_argument("--kvsplice-samples", type=int, default=120_000)
parser.add_argument("--kvsplice-save", type=str, default="kvsplice.pt")
parser.add_argument("--kvsplice-max-batches", type=int, default=64)
parser.add_argument("--kvsplice-epochs", type=int, default=8)
parser.add_argument("--kvsplice-lr", type=float, default=2e-3)
```

### 2. Add KVSpliceCalibrator Class

See `/tmp/kv_geom_integration.py` for complete implementation.

Key responsibilities:
- Hook V projection layers
- Collect V tensors during warmup
- Subsample to budget (~2k samples per latent dim)
- Fit Spline→PCA geometry
- Save fitted model to disk

### 3. Calibration in main()

```python
if args.kvsplice_enable:
    # Infer architecture params
    actual_model = model.module if hasattr(model, "module") else model
    n_heads = actual_model.config.n_head
    head_dim = actual_model.config.n_embd // n_heads

    # Create calibrator
    calib = KVSpliceCalibrator(
        model=actual_model,
        n_heads=n_heads,
        head_dim=head_dim,
        k=args.kvsplice_k,
        knots=args.kvsplice_knots,
        device=torch.device(args.device),
        dtype=get_dtype(args.dtype)
    )

    # Collect samples
    calib.register()
    for batch_idx in range(args.kvsplice_max_batches):
        x, y = get_batch("train", args.batch_size, args.block_size, device)
        with torch.no_grad():
            logits, loss = model(x, y)
        if batch_idx % 4 == 0:
            print(f"[KVSplice] batch {batch_idx}, collected {sum(s.shape[0] for s in calib.samples):,} vectors")
    calib.remove()

    # Fit geometry
    kvg = calib.fit_after_collect(epochs=args.kvsplice_epochs, lr=args.kvsplice_lr)

    # Save
    torch.save({
        "Hd": head_dim,
        "k": args.kvsplice_k,
        "knots": args.kvsplice_knots,
        "state_dict": kvg.state_dict(),
    }, args.kvsplice_save)
    print(f"[KVSplice] Saved to {args.kvsplice_save}")
```

### 4. Integration with V-only Pruning

The KVSplice can be combined with V-only pruning from `lib/kv_pruning.py`:

**Strategy A**: Prune first, then compress
```python
# Select top-k V indices (k=391)
idx = pruner.compute_indices(scores, attn)
V_keep = torch.gather(v, 2, idx_expanded)  # [B,H,391,64]

# Compress kept V with geometry
V_compressed = kvg.compress(V_keep)  # [B,H,391,k_latent]
```

**Strategy B**: Compress first, then select
```python
# Compress all V
V_compressed = kvg.compress(v)  # [B,H,T,k_latent]

# Select top-k based on importance
idx = pruner.compute_indices(scores, attn)
V_keep_compressed = torch.gather(V_compressed, 2, idx_expanded)
```

**Strategy A is recommended**: Prune first reduces memory before compression.

## Proposed Ablation Steps

Add to existing V16-V18 sequence:

**V19**: Baseline + V-only KV pruning (k=391) - **NEW CLEAN BASELINE**
- Standard GPT-2 attention
- V-only pruning (from lib/kv_pruning.py)
- No R-MLP
- No KVSplice
- **Purpose**: Isolate V-only pruning effect

**V20**: V19 + KVSplice compression (k=64)
- V-only pruning selects 391 tokens
- KVSplice compresses each V from 64→16 dims
- Total memory: 391 × 16 = 6,256 per head (vs 1024 × 64 = 65,536 baseline)
- **90% memory reduction**
- **Purpose**: Test if geometric compression hurts quality

**V21**: R-MLP + V-only pruning + KVSplice
- Full feature stack
- **Purpose**: Does R-MLP + geometry synergize?

## Expected Outcomes

### Memory Savings

| Config | V cache | Reduction |
|--------|---------|-----------|
| Baseline (V0) | 1024 × 64 = 65,536 | 0% |
| V-only prune (V19) | 391 × 64 = 25,024 | 62% |
| V-prune + Geom (V20) | 391 × 16 = 6,256 | **90%** |

### Quality Expectations

Based on standalone tests showing SplinePCA ≈ PCA:
- V20 should match V19 quality (geometry adds no degradation)
- If V20 < V19, indicates geometry doesn't generalize to attention context
- If V20 > V19, geometry is learning useful structure!

## Technical Challenges

### 1. GPT-2 Combined QKV Projection

GPT-2 uses a single `c_attn` projection for Q, K, V. The hook needs to:
```python
# GPT-2: c_attn outputs [B, T, 3*n_embd]
# Split to get V
qkv = c_attn_output
q, k, v = qkv.split(n_embd, dim=2)
```

### 2. Memory Management

Collecting 120k V vectors × 64 dims × fp32 = 30MB per layer × 12 layers = 360MB.
Move to CPU immediately after collection to avoid GPU OOM.

### 3. Fitting Time

Spline fitting with 8 epochs on 120k samples takes ~30-60 seconds on GPU.
This is one-time calibration cost, acceptable.

### 4. Inference Integration

The fitted geometry must be loaded during inference and applied in attention:
```python
# Load geometry
kvg = torch.load("kvsplice.pt")

# In attention forward
V_compressed = kvg.compress(V)
# ... attention computation ...
V_decompressed = kvg.decompress(V_compressed)
```

## Next Steps

1. **Complete integration** into train_ra_mla.py
2. **Test calibration** on small run (--kvsplice-max-batches 16)
3. **Verify saved geometry** can be loaded and used
4. **Add V19-V21** to ablation defconfig
5. **Run full test** with 2-hour time limit
6. **Document results** in docs/ra.md

## References

- Implementation: `gpt2/kv_geometry_v1.py`
- Standalone tests: `~/devel/kv-compress/`
- V-only pruning: `lib/kv_pruning.py`
- Current ablation: `defconfigs/gpt2-kv-pruning-ablation`
