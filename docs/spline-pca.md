# Spline-PCA Tokenization Roadmap

This document outlines the development roadmap for spline-based PCA tokenization with hierarchical memory tiering, progressing from LeNet5/MNIST validation to ResNet50/ImageNet deployment.

## Vision

Traditional neural networks store all weights in uniform memory, treating all parameters equally. This ignores natural sparsity in both:
1. **Spatial domain**: Not all weight dimensions are equally important (PCA)
2. **Temporal domain**: Not all weights update at the same frequency (splines)

By exploiting these hierarchies, we can tier memory intelligently:
- **Hot tier (HBM/L1)**: Frequently accessed, high-variance components
- **Warm tier (CPU RAM)**: Moderately accessed, mid-variance components
- **Cold tier (SSD)**: Rarely accessed, low-variance components

This enables training much larger models on existing hardware without sacrificing quality.

## Motivation

### The Memory Wall Problem

Modern large models face a fundamental bottleneck:
```
GPT-4:     ~1.8T parameters × 2 bytes = 3.6TB
H100 HBM:  80GB per GPU
Ratio:     45:1 (need 45 GPUs just to hold weights)
```

Current solutions:
- **Model parallelism**: Shard across many GPUs (expensive, complex)
- **Offloading**: Move everything to CPU/disk (slow, uniform treatment)
- **Quantization**: Reduce precision (quality loss, still uniform)

### Our Approach: Semantic Tiering

Instead of treating all weights uniformly, exploit natural hierarchies:

**PCA (Spatial Hierarchy)**:
- Top 10% of principal components explain 90% of variance
- Keep these in fast memory, offload the rest
- Graceful degradation: 40% components = 99% quality at 2.5x less memory

**Splines (Temporal Hierarchy)**:
- Some weights update frequently (active learning)
- Others stabilize quickly (converged features)
- Tier by update frequency, not just variance

**Result**: 50-80% memory reduction with <5% quality loss

## Implementation Stages

### Stage 1: Foundation (Completed ✅)

**Hierarchical Tiering Infrastructure**:
- `lib/tiering.py`: Adam state analysis, emulated/real offloading
- Tier assignment by optimizer state momentum/variance
- Emulated mode: fake delays for hardware evaluation
- Real mode: actual CPU/disk offloading
- Benchmark harness for measuring impact

**Status**: Functional, tested on GPT-2

### Stage 2: LeNet5 Validation (Current)

**Goal**: Validate tiering on simple model before adding complexity

#### Phase 2.1: Baseline Tiering
```bash
make defconfig-lenet5-adam-tier-emulated
make DEVICE=cpu
```

**Validate**:
- Tiering infrastructure works on CPU
- tier_hints.json generated correctly
- Understand which layers are hot vs cold
- Measure emulated overhead on CPU

**Expected Results**:
- Early conv layers: Hot (feature extraction, high gradients)
- Later FC layers: Cold (stable features, low variance)
- Training time: 50 minutes on modern CPU
- Accuracy: ~99.2% (MNIST baseline)

**Success Criteria**:
- Tiering completes without errors
- Tier distribution makes intuitive sense
- Emulated overhead <10% on CPU

#### Phase 2.2: PCA Tokenizer
```bash
# Enable PCA tokenizer in defconfig
CONFIG_LENET5_ENABLE_TOKENIZER=y
CONFIG_LENET5_TOKENIZER_PCA=y
CONFIG_LENET5_PCA_COMPONENTS=64
```

**Validate**:
- PCA compression works (784→64 dims)
- Accuracy within 1% of baseline
- Component variance hierarchy exists
- Tier by variance explained

**Expected Results**:
- Compression ratio: 12x fewer input dimensions
- Top 16 components: Capture ~80% variance (digit shape)
- Next 32 components: Capture ~15% variance (stroke details)
- Last 16 components: Capture ~5% variance (noise/style)

**Success Criteria**:
- Match baseline accuracy (±1%)
- Clear variance hierarchy in PCA components
- Top components visually interpretable

#### Phase 2.3: Spline Trajectories
```bash
# Enable spline-PCA tokenizer
CONFIG_LENET5_TOKENIZER_SPLINE_PCA=y
CONFIG_LENET5_SPLINE_CONTROL_POINTS=8
```

**Validate**:
- Track component evolution during training
- Fit cubic splines to trajectories
- Identify stable vs active components
- Tier by temporal variance

**Expected Results**:
- Some components stabilize quickly (early learning)
- Others continue adapting (fine-tuning)
- Splines compress trajectories with <5% error
- Temporal tiering differs from variance tiering

**Success Criteria**:
- Spline fitting succeeds
- Trajectory variance measurable
- Temporal tiers differ from spatial tiers
- Combined tiering outperforms either alone

### Stage 3: Integration and Tuning

#### Phase 3.1: Multi-Modal Tiering

Combine three signals for tier assignment:
1. **Spatial**: PCA variance explained
2. **Temporal**: Spline trajectory variance
3. **Optimizer**: Adam state momentum/variance

**Scoring Function**:
```python
score = α × variance_explained +
        β × trajectory_variance +
        γ × adam_state_magnitude
```

Tune α, β, γ to maximize memory reduction while minimizing quality loss.

#### Phase 3.2: Threshold Optimization

Current thresholds are conservative:
- HBM: 30% (top components)
- CPU: 50% (middle components)
- SSD: 20% (bottom components)

Experiment with more aggressive settings:
- HBM: 10-20% (maximize offloading)
- CPU: 60-70% (balance latency/capacity)
- SSD: 20-30% (cold storage)

Measure accuracy vs memory tradeoff curve.

#### Phase 3.3: Dynamic Tiering

Instead of static tier assignment, adapt during training:
- Monitor component access patterns in real-time
- Move components between tiers based on usage
- Prefetch predicted hot components
- Evict cold components proactively

**Benefits**:
- Adapts to training phase (early: exploration, late: fine-tuning)
- Responds to learning rate schedule
- Handles non-stationary access patterns

### Stage 4: ResNet50 Deployment

**Challenges**:
1. **Scale**: 25M parameters vs 60K (400x larger)
2. **Complexity**: 50 layers with residual connections
3. **Input size**: 224×224×3 vs 28×28 (150K dims vs 784)
4. **Training time**: 12 hours vs 50 minutes (14x longer)

#### Phase 4.1: PCA for ImageNet

**Input Compression**:
- Original: 224×224×3 = 150,528 dimensions
- Target PCA: 150K → 1K-2K components
- Compression ratio: 75-150x

**Challenges**:
- Fitting PCA on ImageNet (1.2M images)
- Memory for covariance matrix (150K×150K)
- Computational cost of SVD

**Solutions**:
- Incremental PCA (batch fitting)
- Random projections for initialization
- Sparse PCA for structured compression

#### Phase 4.2: Hierarchical Tiering

ResNet50 has natural hierarchy:
- **Early layers**: Low-level features (edges, textures) - stable
- **Middle layers**: Mid-level features (shapes, parts) - moderate
- **Late layers**: High-level features (objects, semantics) - active

Combined with PCA hierarchy:
- **Spatial**: PCA component variance
- **Temporal**: Layer update frequency
- **Architectural**: Early vs late layers

**Expected Tier Distribution**:
- HBM (10-20%): Late-layer high-variance PCA components
- CPU (60-70%): Middle-layer moderate-variance components
- SSD (20-30%): Early-layer low-variance components

#### Phase 4.3: Validation on GPU

Move from CPU to GPU for production deployment:
- CPU validation: RAM vs SSD (20x gap, easy to measure)
- GPU deployment: HBM vs RAM (5x gap, real production)

**Metrics**:
- Accuracy: Match baseline (±0.5%)
- Training time: <20% overhead from tiering
- Memory reduction: 50-80% GPU memory saved
- Throughput: Maintain >80% of baseline tokens/sec

### Stage 5: Advanced Optimizations

#### Gradient-Aware Tiering

Current approach uses optimizer states (momentum/variance). Extend to gradient analysis:
- **Large gradients**: Active learning, keep hot
- **Small gradients**: Converged, can offload
- **Gradient variance**: Stability indicator

#### Activation Offloading

So far we only tier weights. Extend to activations:
- **Batch size scaling**: Activations grow linearly with batch
- **Checkpoint activations**: Recompute instead of store
- **Tier activations**: Offload to CPU during forward pass

Combined weight + activation tiering enables even larger models.

#### Learned Tier Assignment

Replace hand-tuned thresholds with learned policy:
- **Input**: Component statistics (variance, gradients, access frequency)
- **Output**: Tier assignment (HBM/CPU/SSD)
- **Training**: Reinforcement learning to minimize (latency + memory cost)

**Benefits**:
- Adapts to specific model/dataset/hardware
- Discovers non-obvious tiering strategies
- Continuously improves with experience

## Success Criteria Summary

### Minimum Viable (Proof of Concept)
- **LeNet5**: PCA tokenizer matches baseline accuracy (±1%)
- **Splines**: Compress trajectories with <10% error
- **Tiering**: 30% memory reduction, <10% latency overhead

### Strong Result (Ready for Production)
- **LeNet5**: PCA+splines beat baseline by 0.1% (regularization effect)
- **ResNet50**: Match ImageNet accuracy (±0.5%)
- **Tiering**: 50% memory reduction, <20% latency overhead

### Transformative (Research Impact)
- **ResNet50**: Beat baseline by 0.5%+ (compression as regularization)
- **Tiering**: 70%+ memory reduction, <10% latency overhead
- **Scaling**: Enable 2-3x larger models on same hardware

## Key Insights

### Why PCA?

**Spatial sparsity**: Not all dimensions are equally important
- Top 10% components explain 90% variance
- Natural importance hierarchy for tiering
- Compression can act as regularization

### Why Splines?

**Temporal sparsity**: Not all weights update equally
- Some converge quickly (stable features)
- Others adapt continuously (active learning)
- Spline compression enables online learning

### Why LeNet5 First?

**Fast iteration cycle**: 5-10 min experiments vs 12-hour runs
- Debug quickly on simple model
- Understand failure modes
- Build intuition for scaling

**CPU validation**: Bigger tiering impact than GPU
- RAM vs SSD: 20x latency gap
- HBM vs RAM: 5x latency gap
- Easier to measure and tune

### Why This Could Work

**Transformers have poor locality** (current limitation):
- Every token attends to everything (global attention)
- All layers accessed every iteration
- Traditional tiering provides minimal benefit

**But images have structure**:
- Early layers extract local features (edges, textures)
- These stabilize quickly and are candidates for offloading
- PCA exploits spatial structure
- Splines exploit temporal structure

**Combined with RA/R-MLP** (future work):
- Reciprocal architectures have better locality
- Reciprocal pathways delayed until stabilization
- Natural tiering: standard pathway hot, reciprocal cold

## Future Directions

### Beyond Images: Video and 3D

Video adds temporal dimension:
- **Spatial PCA**: Compress each frame
- **Temporal PCA**: Compress across frames
- **Splines**: Natural representation for motion

3D data (medical imaging, molecular dynamics):
- **Volumetric PCA**: Compress 3D volumes
- **Multi-scale hierarchy**: Coarse to fine resolution
- **Anatomical priors**: Leverage structure

### Integration with RA/R-MLP

Your reciprocal architectures have natural tiering points:
- **Standard pathway**: Always active, keep hot
- **Reciprocal pathway**: Delayed activation, can start cold
- **Gates**: Tier by gate activation frequency

Spline-PCA could represent gate trajectories:
- Track when gates open/close
- Fit splines to activation patterns
- Predict future gate states

### Hardware Co-Design

With emulated tiering, we can evaluate future hardware:
- **CXL memory**: 2μs latency, 400 GB/s bandwidth
- **Fabric-attached memory**: 5μs latency, 200 GB/s
- **Persistent memory**: 10μs latency, 100 GB/s

Inform hardware purchasing decisions before spending $50K.

## References

### Related Work

**Memory Optimization**:
- DeepSpeed ZeRO: Offload everything uniformly
- HuggingFace Accelerate: Device map for inference
- FlashAttention: Kernel fusion for memory efficiency

**Dimensionality Reduction**:
- PCA: Classical linear projection
- Autoencoders: Learned non-linear compression
- Random projections: Fast approximate compression

**Temporal Compression**:
- Checkpoint gradients: Recompute instead of store
- Reversible networks: Reconstruct activations on demand
- Spline interpolation: Compress time-series data

### Novel Contributions

**Adam state-based tiering**: Use optimizer states to infer importance
- Momentum/variance as proxy for update frequency
- No extra computation, leverage existing state

**PCA-spline combination**: Spatial + temporal hierarchy
- PCA: Which dimensions matter
- Splines: Which dimensions are active
- Combined: Semantic importance scoring

**Graceful degradation**: Trade memory for quality smoothly
- Not binary (keep/discard)
- Continuous spectrum of compression ratios
- User-tunable threshold parameters

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy (for PCA and splines)
- Modern CPU (for LeNet5 validation)

### Quick Start

```bash
# Phase 1: Validate tiering infrastructure
make defconfig-lenet5-adam-tier-emulated
make DEVICE=cpu

# Phase 2: Add PCA tokenizer (edit defconfig)
CONFIG_LENET5_ENABLE_TOKENIZER=y
CONFIG_LENET5_TOKENIZER_PCA=y
make DEVICE=cpu

# Phase 3: Add spline trajectories (edit defconfig)
CONFIG_LENET5_TOKENIZER_SPLINE_PCA=y
make DEVICE=cpu

# Analyze results
python3 scripts/analyze_tier_assignments.py tier_hints_lenet5.json
```

### Documentation

- `docs/hierarchical-tiering.md`: Tiering infrastructure overview
- `lib/tiering.py`: Tier analyzer implementations
- `lib/tokenizers.py`: PCA and spline tokenizers
- `lenet5/model.py`: Model variants with tokenizer support

## Conclusion

Spline-PCA tokenization with hierarchical tiering represents a fundamentally different approach to the memory wall problem. Instead of treating all weights uniformly, we exploit natural sparsity in both spatial (PCA) and temporal (splines) domains to create semantic importance hierarchies.

By starting with LeNet5 validation and progressing systematically to ResNet50 deployment, we minimize risk while building intuition. The fast iteration cycle on CPU enables rapid prototyping without GPU contention.

If successful, this approach could enable training 2-3x larger models on existing hardware, or equivalently reduce hardware costs 2-3x for current model sizes. The emulated tiering infrastructure allows evaluating future hardware (CXL, fabric memory) before making purchasing decisions.

This is a long-term research direction with near-term validation milestones. The building blocks are in place - now it's time to validate the vision.
