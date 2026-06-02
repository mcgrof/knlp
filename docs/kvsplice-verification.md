# KVSplice verification

KVSplice is a learned KV cache compression layer that achieves 12x
total compression (6x from MLA + 2x from KVSplice). Before claiming
compression ratios or memory savings, verify both training quality
and inference memory reduction.

### Training Verification

When evaluating KVSplice training results:

1. **Compare across GPU types**: Run ablation on multiple GPUs (W7900,
   A100, H100) to verify consistency and detect hardware-specific
   issues

2. **Check transform parameter learning**: Extract scale/shift values
   from checkpoints to verify the learned monotonic transform is
   actually training (not stuck at initialization)
   ```bash
   python scripts/extract_kvsplice_params.py \
     --checkpoint path/to/checkpoint.pt
   ```

3. **Monitor KVSplice metrics in W&B**: Verify that scale_mean,
   scale_std, shift_mean, shift_std are logged during training. If
   missing, check architecture detection in
   `_compute_kvsplice_param_metrics()`

4. **Verify compression ratio setting**: Confirm CONFIG_MLA_COMPRESSION_RATIO
   is set correctly in defconfig and matches W&B config. Default is
   0.5 (2x compression on top of MLA)

5. **Quality degradation tolerance**: KVSplice should add only
   0.5-1.4% quality loss compared to MLA alone. Larger degradation
   indicates a bug

### Inference Verification

Before publishing inference memory savings claims:

1. **Run direct cache measurement**: Use
   `scripts/verify_kvsplice_memory.py` to measure actual cache tensor
   sizes across sequence lengths
   ```bash
   python scripts/verify_kvsplice_memory.py
   ```

2. **Verify cache tensor shapes**: Inspect returned cache objects to
   confirm dimensions:
   - MLA: `[B, T, d_latent]` where d_latent=256
   - KVSplice: `[B, T, d_compressed]` where d_compressed=128 (ratio=0.5)

3. **Check compression ratio accuracy**: Memory savings should match
   theoretical predictions within 5%:
   - Expected savings: `compression_ratio * 100%`
   - Example: ratio=0.5 should give 50% cache reduction vs MLA

4. **Test multiple sequence lengths**: Verify compression holds across
   256, 512, and 1024 token sequences. Savings should scale linearly

5. **Calculate production throughput**: Estimate how many parallel
   sequences fit in GPU memory with compressed cache vs standard
   cache. Include model weights in calculation

### Transform Parameter Analysis

KVSplice uses a learned monotonic transform before low-rank
projection. To verify it's learning:

1. **Extract parameters from checkpoint**:
   ```bash
   python scripts/extract_kvsplice_params.py \
     --checkpoint test_matrix_results_*/checkpoint.pt
   ```

2. **Check for variance across dimensions**: If all scale values are
   identical and all shift values are zero, parameters are not
   learning

3. **Initial values to expect**:
   - Scale: softplus(1.0) ≈ 1.3133 (initialization)
   - Shift: 0.0 (initialization)
   - After training: should show variance across 256 dimensions

4. **Pruning candidates**: Dimensions with scale < 0.1 after training
   are low-importance and candidates for pruning

5. **LayerNorm impact**: If transform parameters don't learn, try
   adding LayerNorm to latent space to stabilize gradients

### Known Issues

**Transform parameters not learning**: Current experiments show
KVSplice transform parameters remain at initialization values (scale
≈ 1.3133, shift = 0.0) even after 1000+ iterations. This means
KVSplice is working purely via low-rank projection (compress/expand
layers), not the learned transform. This may be optimal if the
compress/expand layers can learn the mapping directly.

**Architecture detection for metrics**: Early versions failed to log
KVSplice metrics because code only checked for `raw_model.transformer`
(standard GPT-2) but MLA uses `raw_model.blocks`. Fixed in commit
that added dual architecture detection.

**Memory measurement pitfalls**: Don't measure cache memory by running
full forward passes (passing all previous tokens). This defeats the
purpose of caching. Instead, extract cache objects from blocks with
`use_cache=True` and measure tensor sizes directly.

### Verification Scripts

- `scripts/verify_kvsplice_memory.py`: Measure cache tensor sizes
- `scripts/extract_kvsplice_params.py`: Extract learned transform
  parameters
- `scripts/compare_kvsplice_gpus.py`: Compare results across GPU types
- `scripts/plot_kvsplice_inference_memory.py`: Generate visualization
  plots

### Documentation Updates

After verification, update documentation with plots and results:

1. **Add inference verification section** to `docs/kvsplice.md`:
   - Include cache memory comparison plots
   - Show compression breakdown visualization
   - Document cache tensor shapes
   - Provide memory savings table

2. **Update GPU comparison summary** in
   `docs/kvsplice/gpu-comparison-summary.md`:
   - Add inference verification results
   - Compare theoretical vs actual compression
   - Document production implications

3. **Generate publication-quality plots** (300 DPI):
   ```bash
   python scripts/plot_kvsplice_inference_memory.py
   ```

See `docs/kvsplice.md` for complete inference verification results
with plots showing 50% cache reduction (12 MB → 6 MB at 1024 tokens)
and 83.3% total reduction vs standard GPT-2 (36 MB → 6 MB).

