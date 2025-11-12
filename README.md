# knlp: Kernel Developer ❤️ Machine Learning

**Transformer architecture research from a kernel developer's perspective**

> **🎯 Latest: Reciprocal MLP Breakthrough (Nov 2025)**: R-MLP achieves **12% validation loss improvement** (3.75 → 3.32) over baseline GPT-2 by learning to use bidirectional information flow between attention and MLP layers. Reciprocal weights learned positive values (mean=+0.13), demonstrating the model discovered attention signals enhance MLP computation.

> **🚀 Reciprocal Attention (RA)**: Production-ready bidirectional attention mechanism achieves **2.17% speedup** over baseline with zero extra FLOPs or memory. Single-SDPA-call folded architecture validated on GPT-2 124M.

> **📦 KVSplice Compression**: Geometric KV cache compression using Spline→PCA learns data-specific manifold transformations. Targets 90% memory reduction (391 tokens × 16 dims) while maintaining quality. See [docs/ra.md](docs/ra.md) for technical details.

> **⚡ Legacy Pruning Results**: AdamWPrune achieves 20% training speedup and 8.2% GPU memory reduction on GPT-2, plus 74.56% accuracy on ResNet-50 at 50% sparsity.

knlp explores efficient transformer architectures through reciprocal mechanisms, geometric compression, and state-based optimization - bringing kernel development rigor to ML research.

## Key Results

### Memory Efficiency Across Models

| Model | Parameters | Dataset | Sparsity | GPU Memory | Accuracy/Perplexity | Efficiency |
|-------|------------|---------|----------|------------|---------------------|------------|
| **GPT-2** | **124M** | **FineWebEdu** | **50%** | **25311 MiB** | **49.99 ppl** | **8.2% GPU memory savings** |
| ResNet-50 | 25.6M | CIFAR-100 | 50% | 12602.5 MiB | 74.56% | 6.06/100MiB |
| ResNet-18 | 11.2M | CIFAR-10 | 70% | 1489.2 MiB | 90.66% | 6.09/100MiB |
| LeNet-5 | 61.7K | MNIST | 70% | 434.5 MiB* | 98.9% | 22.74/100MiB |

*CUDA/PyTorch baseline overhead (~450 MiB) dominates for small models

## GPT-2 Transformer Results (124M Parameters)

### Latest: Reciprocal Architecture Results (Nov 2025)

**R-MLP (Reciprocal MLP) achieves 12% improvement over baseline**
by learning bidirectional information flow between attention and
MLP layers.

| Architecture | Val Loss | Improvement | Key Innovation |
|--------------|----------|-------------|----------------|
| Baseline GPT-2 | 3.75 | - | Standard transformer |
| RA only (V16) | 3.38 | 10% | Folded reciprocal attention |
| RA + frozen R-MLP (V17) | 3.38 | 10% | No reciprocal learning |
| **RA + R-MLP (V18)** | **3.32** | **12%** | **Learned reciprocal weights (+0.13 mean)** |

**Key finding**: R-MLP learned **positive** reciprocal weights,
proving the model discovered attention information from previous
layers enhances MLP computation. See
[docs/ra.md](docs/ra.md#kvsplice-geometric-kv-cache-compression)
for architecture details.

**Training**: 2 hours per step, ~2065 iterations, 4× NVIDIA A10G
(24GB). Minimal overhead (~4ms/iter). Production-ready stability.

---

### Legacy: The Bitter Lesson Confirmed (Pruning Research)

Our GPT-2 pruning experiments validated Rich Sutton's Bitter
Lesson: **simpler algorithms leveraging computation outperform
complex, engineered approaches**.

**Test Configuration:**
- Model: GPT-2 (124M parameters)
- Dataset: FineWebEdu
- Target Sparsity: 50%
- Training: 10,000 iterations (13,000 for bitter3/4*)

### Performance Results

| Optimizer | Algorithm | Final Perplexity | Iterations | Training Time | Memory |
|-----------|-----------|------------------|------------|---------------|--------|
| **AdamWSPAM** | **Magnitude** | **42.82** (baseline) | 10,000 | 8.3 hours | 5.03x weights |
| AdamWPrune | Bitter3* | **43.11** | 13,000* | 10.4 hours | 3.03x weights |
| AdamWPrune | Bitter4* | **44.88** | 13,000* | 10.3 hours | 3.03x weights |
| AdamWPrune | Bitter2† | 46.07 | 12,100† | 8.3 hours | 3.03x weights |
| AdamWPrune | Bitter1 | 49.99 | 10,000 | ~6.9 hours | 3.03x weights |
| AdamWPrune | Bitter0 | 51.51 | 10,000 | ~6.9 hours | 3.03x weights |

*Bitter3/4 use 30% more iterations (13,000) with gradient-magnitude pruning
†Bitter2 uses 21% more iterations (12,100) for scale-aware magnitude pruning

### AdamWPrune Bitter Variants Explained

The "bitter" naming follows Rich Sutton's Bitter Lesson: simpler methods with more computation often outperform complex, clever algorithms.

#### Bitter0 (Original AdamWPrune)
- **Formula**: `momentum_score × stability_score` where:
  - `momentum_score = |exp_avg| / (sqrt(|exp_avg_sq|) + eps)`
  - `stability_score = 1 / (variance + eps)`
- **Philosophy**: Complex hybrid approach using Adam's momentum and stability signals
- **Intuition**: Prune weights with low momentum relative to their second moment and high variance
- **Result**: 51.51 perplexity - poorest performance despite algorithmic complexity

#### Bitter1 (Pure Magnitude)
- **Formula**: `|w|` (absolute weight value)
- **Philosophy**: Simplest possible approach - prune small weights
- **Intuition**: Small weights contribute less to the network's function
- **Memory**: Uses boolean masks (8x memory reduction vs float masks)
- **Result**: 49.99 perplexity - better than complex bitter0

#### Bitter2 (Scale-Aware Magnitude)
- **Formula**: `|w|` with 21% more iterations
- **Philosophy**: Same as bitter1 but leverages saved computational resources
- **Intuition**: Use the efficiency gains from faster pruning to train longer
- **Result**: 46.07 perplexity - significant improvement from extra training

#### Bitter3 (Gradient-Magnitude with Adam States)
- **Formula**: `|w| × sqrt(|exp_avg_grad| + ε)` where `exp_avg_grad` is Adam's first moment (with abs() for safety)
- **Philosophy**: Combine weight importance with gradient activity using optimizer state
- **Intuition**: Important weights are both large AND actively being updated
- **Key Innovation**: Leverages Adam's exponential moving average (state["exp_avg"]) for stable gradient signal
- **State Usage**: Uses Adam's first moment only (simpler than bitter0's dual-moment approach)
- **Implementation**: Takes absolute value before sqrt to handle negative gradients correctly
- **Schedule**: Cubic sparsity ramping (gentler early pruning)
- **Result**: **43.11 perplexity** - nearly matches AdamWSPAM baseline (42.82)

#### Bitter4 (Gradient-Magnitude + Layer-Adaptive with Adam States)
- **Formula**: `|w| × sqrt(|exp_avg_grad| + ε)` with varying sparsity per layer (using Adam's state["exp_avg"] with abs())
- **Philosophy**: Same as bitter3 but recognizes layers have different redundancy
- **State Usage**: Same as bitter3 - Adam's first moment for gradient activity
- **Implementation**: Same abs() before sqrt as bitter3 to handle negative gradients
- **Layer Distribution**:
  - Early layers: 0.7× target sparsity (preserve feature extraction)
  - Late layers: 1.3× target sparsity (more task-specific redundancy)
- **Result**: **44.88 perplexity** - layer-adaptive underperformed uniform sparsity for GPT-2

### Key Insights

1. **Bitter Lesson Validated**: Gradient-magnitude (bitter3) achieves near state-of-the-art with simple formula
2. **Surprising Finding**: Layer-adaptive sparsity (bitter4) underperformed uniform sparsity by 1.77 perplexity
3. **Memory Efficiency**: 40% reduction in theoretical overhead (5.03x → 3.03x weights), 8.2% actual GPU savings
4. **Training Efficiency**: 20% faster per-iteration than AdamWSPAM baseline
5. **Breakthrough**: Bitter3 proves simple `|w| × sqrt(|grad_ema|)` nearly matches SOTA with better efficiency

### Final Rankings

1. **Best Overall**: AdamWSPAM (42.82 ppl) - when quality is paramount
2. **Best AdamWPrune**: Bitter3 (43.11 ppl) - best efficiency/quality trade-off
3. **Unexpected**: Bitter4 (44.88 ppl) - layer-adaptive complexity didn't help transformers
4. **Clear Pattern**: Simpler methods (bitter3 > bitter1) outperform complex ones (bitter0, bitter4)

### Visual Evidence

#### Perplexity Comparison
![GPT-2 Perplexity Comparison](images/gpt2/gpt2_perplexity_comparison.png)
*All AdamWPrune variants achieve significant speedup with modest perplexity trade-offs*

#### The Bitter Lesson Confirmed
![The Bitter Lesson](images/gpt2/gpt2_bitter_lesson.png)
*Simpler algorithms (bitter1) outperform complex hybrid approaches (bitter0), confirming Sutton's principle*

#### GPU Memory Consumption Analysis
![GPU Memory Analysis](images/gpt2/gpt2_gpu_memory_analysis.png)
*AdamWPrune variants consistently use 8.2% less GPU memory while achieving 17-20% faster training*

**Key GPU Memory Observations:**
- **AdamWSPAM baseline**: 27.2 GiB (56.1% of 48GB AMD W7900)
- **All AdamWPrune variants**: 25.0 GiB (51.5%) - **8.2% memory reduction**
- **Training speed**: Bitter0/1 complete in ~6.9 hours vs baseline's 8.3 hours
- **Bitter2 exception**: Intentionally trains longer (8.3 hours) for better quality

The memory savings directly translate to the ability to:
- Train with **9% larger batch sizes** on the same hardware
- Run **multiple experiments** in parallel with saved memory
- **Deploy on smaller GPUs** that couldn't fit the baseline

#### Perplexity Matching Projections
![Perplexity Projections](images/gpt2/gpt2_perplexity_projections.png)
*Projected iterations needed to match baseline perplexity - even with 2x training, memory savings persist*

**Feasibility Analysis** (assuming 0.75 ppl improvement per 1000 iterations):
- **Bitter2**: Would need **+64% more iterations** (16,433 total) to match baseline
- **Bitter1**: Would need **+96% more iterations** (19,560 total)
- **Bitter0**: Would need **+116% more iterations** (21,587 total)

**Critical Trade-off**: Even if we trained AdamWPrune variants 2x longer to match perplexity:
- Still **8.2% GPU memory savings** throughout training
- But lose the **20% speed advantage**
- Makes sense only when memory is the primary constraint

→ See [GPT-2 detailed analysis](docs/gpt2.md) for complete findings and more visualizations

### Implementation Notes (November 2024)

**Pruning Schedule**: All methods now use cubic sparsity schedule `s(t) = s_final × ((t-warmup)/(T-warmup))^3` following [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635). This enables fair comparison between methods.

**Bitter Variants**: Multiple pruning importance metrics were evaluated. bitter7 (variance-based: `|w| × v^0.25` where v is Adam's exp_avg_sq) showed most promise and is the focus of ongoing work.

## ResNet CNN Results

### Optimizer Performance vs Model Size

#### Critical Finding: Optimal Optimizer Changes with Model Scale

Our testing reveals that **the best-performing optimizer depends on model size**:

**ResNet-18 (11.2M parameters, CIFAR-10):**
- **Winner: AdamW** (90.30% accuracy)
- Adam: 89.85% (-0.45%)
- AdamWSPAM: 89.75% (-0.55%)
- AdamWAdv: 89.42% (-0.88%)
- SGD: 89.22% (-1.08%)

**ResNet-50 (25.6M parameters, CIFAR-100) - Latest Results (September 2025):**

**🏆 AdamWPrune with AdamWSpam Base Sets New Record:**

| Sparsity | AdamWPrune (AdamWSpam base) | AdamWSpam (best) | AdamWPrune (AdamW base) | Improvement |
|----------|------------------------------|------------------|--------------------------|-------------|
| **50%** | **74.56%** 🥇 | 74.11% (Magnitude) | 74.54% | +0.45% |
| **70%** | **73.89%** 🥇 | 73.11% (Magnitude) | 72.30% | +0.78% |
| **90%** | **72.84%** 🥇 | 72.63% (Magnitude) | 73.26% | +0.21% |
| Baseline | **72.60%** | 71.86% | N/A | +0.74% |

**Breakthrough Configuration**: `CONFIG_ADAMWPRUNE_BASE_ADAMWSPAM=y` with SPAM theta=50.0

**Key Achievements**:
- **Universal superiority**: AdamWPrune outperforms AdamWSpam at ALL sparsity levels
- **50% sweet spot**: 74.56% accuracy - **1.96% better than baseline** (pruning improves accuracy!)
- **Memory efficiency**: 12602.5 MB with 6.06% accuracy per GB
- **Stability**: Lower variance (0.23% std) in final epochs

**Key Insight**: As model complexity increases from ResNet-18 to ResNet-50, AdamWSPAM's spike-aware momentum adaptation becomes more beneficial than AdamW's simpler decoupled weight decay. This suggests that larger models with more complex loss landscapes benefit from SPAM's gradient spike detection and momentum reset mechanisms.

### GPU Memory Analysis

#### ResNet-18 Production-Scale Results (September 2025)

**Key Findings with AdamW Base:**
- **Identical performance without pruning**: AdamW (90.30%) vs AdamWPrune (90.28%) at ~1307 MB
- **Best accuracy at 50% sparsity**: Both movement and state pruning achieve 90.69%
- **Pruning method comparison**: Movement > State > Magnitude for accuracy retention
- **Memory overhead**: Magnitude pruning adds ~93 MB, movement/state add ~167-195 MB

| Configuration | Optimizer | Pruning Method | Sparsity | GPU Memory (Actual) | Accuracy |
|--------------|-----------|----------------|----------|---------------------|----------|
| **AdamW Baseline** | AdamW | None | 0% | **1307.6 MiB** | 90.30% |
| **AdamWPrune Baseline** | AdamWPrune | None | 0% | **1307.4 MiB** | 90.28% |
| AdamW Movement | AdamW | Movement | 50% | 1475.6 MiB | **90.69%** |
| AdamWPrune State | AdamWPrune | State | 50% | 1474.6 MiB | **90.69%** |
| AdamW Movement | AdamW | Movement | 70% | 1474.6 MiB | 89.68% |
| AdamWPrune State | AdamWPrune | State | 70% | 1503.0 MiB | 89.37% |
| AdamW Movement | AdamW | Movement | 90% | 1475.5 MiB | 89.10% |
| AdamWPrune State | AdamWPrune | State | 90% | 1502.9 MiB | 88.65% |
| AdamW Magnitude | AdamW | Magnitude | 50% | 1400.0 MiB | 88.97% |
| AdamW Magnitude | AdamW | Magnitude | 70% | 1399.9 MiB | 88.44% |
| AdamW Magnitude | AdamW | Magnitude | 90% | 1398.9 MiB | 86.85% |

**Key Achievements**:
- **Proper weight decay**: AdamW base with parameter groups (no decay on bias/BatchNorm)
- **Tied best accuracy**: State and movement pruning both achieve 90.69% at 50% sparsity
- **Consistent memory usage**: AdamWPrune uses similar memory to AdamW across configurations

### Visual Evidence of AdamWPrune Performance

#### All Methods Comparison
![All Methods Comparison](images/resnet18/all_methods_comparison.png)
*Comprehensive comparison showing all pruning methods including AdamWPrune achieving competitive accuracy*

![Memory and Accuracy Comparison](images/resnet18/all_methods_memory_accuracy.png)
*Memory and accuracy comparison across all methods - state and movement pruning achieve similar results*

#### GPU Memory Analysis
![GPU Memory Comparison](images/resnet18/gpu_memory_comparison.png)
*Comprehensive GPU memory usage comparison across all tested configurations*

![GPU Memory Timeline](images/resnet18/gpu_memory_timeline.png)
*Real-time GPU memory usage during training phases*

![Memory vs Accuracy Scatter](images/resnet18/memory_vs_accuracy_scatter.png)
*Memory-accuracy trade-off: State and movement pruning achieve similar memory usage (~1489 MB) with comparable accuracy*

→ See [ResNet-18 detailed findings](docs/resnet18.md) for detailed analysis

### ResNet-50 Visual Evidence

#### Performance Evolution Across Optimizers
![AdamWPrune Accuracy Evolution](images/resnet50/adamwprune_accuracy_evolution.png)
*AdamWPrune showing superior performance at 50% sparsity (74.68%) with stable training*

![SGD vs AdamWPrune](images/resnet50/sgd_model_comparison.png)
*SGD excels at 70% sparsity while AdamWPrune dominates at 50%*

#### GPU Memory Efficiency
![GPU Memory Comparison](images/resnet50/gpu_memory_comparison.png)
*AdamWPrune achieves lowest memory usage (12602.5 MB) across all configurations*

![Memory vs Accuracy Scatter](images/resnet50/memory_vs_accuracy_scatter.png)
*Clear winner: AdamWPrune achieves best accuracy-memory trade-off*

![Training Memory Comparison](images/resnet50/training_memory_comparison.png)
*Detailed memory analysis showing AdamWPrune's consistent efficiency*

→ See [ResNet-50 detailed findings](docs/resnet50.md) for complete analysis

#### LeNet-5 Comprehensive Analysis
![LeNet-5 Memory Analysis](images/lenet5/training_memory_comparison.png)
*Comprehensive 6-panel analysis showing AdamWPrune's memory efficiency patterns*

#### Memory Efficiency Leaders (ResNet-18)
Top configurations by accuracy per 100 MiB of GPU memory:
1. **AdamW** (no pruning): 6.91 efficiency score - Best overall
2. **AdamWPrune** (no pruning): 6.91 efficiency score
3. **AdamW** (magnitude_50): 6.35 efficiency score
4. **AdamW** (magnitude_70): 6.32 efficiency score
5. **AdamWPrune** (state_50): 6.15 efficiency score
6. **AdamW** (movement_50): 6.15 efficiency score

The minimal absolute memory differences in LeNet-5 (~10-20 MiB) are due to CUDA/PyTorch's ~450 MiB baseline overhead, but the efficiency patterns clearly demonstrate AdamWPrune's algorithmic advantages.

## How AdamWPrune Works

Traditional pruning methods require **additional memory buffers**:
- Importance scores (float32 per parameter)
- Binary masks (1 byte per parameter)
- Initial weight copies for reference
- **Total overhead**: 1-2× model size

**AdamWPrune's innovation**: Reuses existing AdamW optimizer states:
- Based on AdamW for proper decoupled weight decay
- `exp_avg` (momentum) → tracks weight importance
- `exp_avg_sq` (variance) → provides stability signals
- Only adds boolean mask when pruning active (1 byte/param)
- **Proven minimal overhead**: When pruning disabled, 1307.4 MB (vs AdamW's 1307.6 MB)
- **Pruning overhead**: State pruning adds ~167-195 MB, comparable to movement pruning
- **Result**: Achieves 90.69% accuracy at 50% sparsity, tied with movement pruning

### Why AdamW as Base?

AdamWPrune is built on AdamW rather than Adam for critical reasons:
1. **Decoupled weight decay**: AdamW properly decouples L2 regularization from gradient-based updates
2. **Parameter groups**: Excludes bias and BatchNorm parameters from weight decay (critical for performance)
3. **Better baseline**: AdamW outperforms Adam (90.30% vs older 90.31% with improper implementation)
4. **Industry standard**: AdamW is the de facto standard for transformer and modern architectures

### Model Checkpointing

Industry best practices recommend saving model checkpoints at peak accuracy, not just at training completion. This is critical because:
- **Peak ≠ Final**: Models often achieve best accuracy mid-training (e.g., AdamWPrune: 74.68% at epoch 63, final only 70.56%)
- **Overfitting protection**: Later epochs may degrade performance
- **Deployment ready**: Best checkpoints are production-ready models

📚 **[Checkpoint Best Practices Guide](docs/checkpoint-best-practices.md)** - Learn why and how to implement proper checkpointing strategies based on our experimental findings.

## Detailed Findings

- **[State-Based Pruning Deep Dive](docs/adding_state_pruning.md)**: Comprehensive analysis of AdamWPrune's state pruning approach
- **[LeNet-5 Results](docs/lenet5.md)**: Proof of concept on MNIST
- **[ResNet-18 Results](docs/resnet18.md)**: Production-scale validation on CIFAR-10
- **[ResNet-50 Results](docs/resnet50.md)**: ImageNet-scale demonstration of superior memory efficiency
- **[GPT-2 Results](docs/gpt2.md)**: Transformer validation confirming bitter lesson with 20% speedup
- **[Key Test Results Archive](https://github.com/mcgrof/knlp-key-results)**: Complete test matrix results with all graphs and metrics
  - [R-MLP Validation (Nov 2025)](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20251111_170325/FINDINGS.md): **12% improvement** over baseline GPT-2, reciprocal weights learned positive
  - [ResNet-50 AdamWSpam Base Results](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20250913_200218/ANALYSIS.md): **74.56% at 50% sparsity** - state-of-the-art
  - [ResNet-50 CIFAR-100 Extended Results](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20250908_190856/summary_report.txt): 74.54% at 50% sparsity with AdamW base
  - [ResNet-50 CIFAR-100 Initial Results](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20250908_121537/summary_report.txt): 72.38% at 70% sparsity with lowest GPU memory
  - [ResNet-18 CIFAR-10 Results](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20250903_180836/report.md): 90.66% accuracy with lowest memory usage
  - [GPT-2 Bitter Lesson Test Results](https://github.com/mcgrof/knlp-key-results/tree/master/key_results/test_matrix_results_20250923_010926/): Confirms bitter lesson - simpler algorithms outperform complex ones

## Transformer Model Findings (GPT-2)

### The Bitter Lesson Confirmed

Our GPT-2 experiments validate Rich Sutton's Bitter Lesson in neural network pruning: **simpler algorithms leveraging computation outperform complex, engineered approaches**.

**Test Configuration:**
- Model: GPT-2 (124M parameters)
- Dataset: FineWebEdu
- Target Sparsity: 50%
- Training: 10,000 iterations (12,100 for bitter2)

### Performance Results

| Optimizer | Algorithm | Final Perplexity | Iterations | Training Time | Memory |
|-----------|-----------|------------------|------------|---------------|--------|
| **AdamWSPAM** | **Magnitude** | **42.82** (best) | 10,000 | Baseline | 5.03x weights |
| AdamWPrune | Bitter2* | 46.07 | 12,100* | Baseline* | 3.03x weights |
| AdamWPrune | Bitter1 | 49.99 | 10,000 | ~20% faster | 3.03x weights |
| AdamWPrune | Bitter0 | 51.51 | 10,000 | ~20% faster | 3.03x weights |

*Bitter2 uses 21% more iterations (12,100) to explore scale-aware pruning, resulting in similar wall-clock time despite faster per-iteration speed.

### Key Transformer Insights

1. **Bitter Lesson Validated**: Simpler pruning algorithms (bitter1/2) outperformed the complex hybrid approach (bitter0)
   - Bitter2 achieves 46.07 perplexity with simple scale-aware magnitude
   - Bitter0's complex momentum-stability hybrid performs worst at 51.51

2. **Training Efficiency**: Bitter0/1 achieve ~20% speedup; bitter2 trades speed for quality with extended training

3. **Memory Efficiency**: 40% reduction in theoretical overhead, 8.2% actual GPU memory savings
   - Traditional approach: 5.03x model weights (Adam states + movement scores)
   - AdamWPrune: 3.03x model weights (Adam states + boolean mask only)

4. **Clear Trade-offs**: Speed and memory benefits come with 7-20% perplexity increase
   - Acceptable for memory-constrained environments
   - Valuable for large-scale training where 20% speedup is critical

### Practical Implications

**When to use AdamWPrune on transformers:**
- Memory-constrained training environments
- Large-scale experiments where 20% speedup matters
- Research exploring efficient training methods

**When to use traditional pruning:**
- Production models requiring absolute best perplexity
- Small models where memory isn't a constraint

## R&D Projects

This project has evolved into a research platform for rapid prototyping and validation of training optimization ideas:

- **[AdamWPrune](docs/adding_state_pruning.md)**: Proven state-based pruning (documented above)
- **[Reciprocal Attention (RA) and CoGA](docs/ra.md)**: Experimental attention mechanism R&D - see dedicated documentation

The modular infrastructure enables fast iteration on new optimization techniques with automated testing and validation.

## Pruning Method Insights

### Movement Pruning: Designed for Fine-tuning, Not Training from Scratch

Movement pruning, introduced by Sanh et al. (2020) in "Movement Pruning: Adaptive Sparsity by Fine-Tuning", was specifically designed for **fine-tuning pre-trained models**, not training from scratch. This distinction is critical:

**Key characteristics:**
- **Fine-tuning context**: Movement pruning identifies weights moving toward zero during adaptation to downstream tasks
- **Pre-trained models**: Works best with models that have already learned meaningful representations
- **Architecture differences**: Transformers and CNNs exhibit different pruning behaviors
  - Transformers: Many redundant parameters naturally move toward zero, leading to aggressive pruning
  - CNNs: More structured weight patterns maintain their magnitudes during training

**Training from scratch limitations:**
- Random initial weights lack meaningful movement patterns
- Weights moving toward zero early in training may still be important later
- Can lead to aggressive over-pruning beyond target sparsity levels
- Less stable than magnitude-based methods for random initialization

**Practical implications for GPT-2 comparisons:**
Given that movement pruning is optimized for fine-tuning scenarios, our GPT-2 training-from-scratch experiments focus primarily on comparing **magnitude pruning vs. AdamWPrune's state-based pruning**, which are both designed for training from random initialization.

## Features

- **Multi-Model Support**: Extensible architecture supporting LeNet-5, ResNet-18, ResNet-50, and GPT-2
- **GPU Optimization**: Optimized for modern GPUs with detailed monitoring
- **Vendor-Agnostic GPU Monitoring**: Uses [gputop.py](https://github.com/mcgrof/gputop) for consistent memory tracking across NVIDIA/AMD/Intel GPUs
- **TrackIO Integration**: Visualize training metrics with [TrackIO](https://github.com/mcgrof/trackio/tree/20250921-trackio-view) for beautiful GPU utilization graphs
- **Multiple Pruning Methods**:
  - **Magnitude pruning**: Conservative, suitable for training from scratch
  - **Movement pruning**: Best for fine-tuning pre-trained models
  - **State-based pruning (AdamWPrune)**: Novel approach using optimizer states
- **Kconfig System**: Linux kernel-style configuration for experiment management
- **Test Matrix**: Automated testing across optimizer and pruning combinations
- **Comprehensive Visualization**: Memory timeline, efficiency analysis, and trade-off plots

## Quick Start

### Configure with Experiment Tracking

```bash
# Basic configuration (no tracking)
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4

# Enable local tracking with Trackio
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=trackio

# Enable cloud tracking with WandB (requires wandb login)
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=wandb

# Enable both for comparison
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=wandb,trackio

# Run training
make
```

**Auto-generated project names**: `{model}-{5char-checksum}` (e.g., `gpt2-a3f2c`, `resnet50-7b9d1`)
- Consistent across runs from same directory
- No collisions between machines/directories
- No manual configuration needed

See [docs/experiment-tracking.md](docs/experiment-tracking.md) for detailed tracking configuration.

### Test ResNet-18 with AdamWPrune

```bash
# Quick state pruning comparison on ResNet-18
make defconfig-resnet18-state-pruning-compare
make # for all tests

# If you want to shorten tests and are doing R&D
# you can reduce epochs dynamically:
make EPOCHS=100  # Or EPOCHS=3 for quick test
```

### Test LeNet-5 (Original Model)

```bash
# Run complete LeNet-5 test matrix
make defconfig-lenet5-compare
make
```

### Interactive Configuration

```bash
# Choose model, optimizer, and pruning settings
make menuconfig
make
```

## Installation

### Core Dependencies
```bash
pip install torch torchvision numpy matplotlib
```

### Optional: Experiment Tracking
```bash
# Local tracking (no authentication needed)
pip install trackio

# Cloud tracking (requires account)
pip install wandb
wandb login  # One-time setup

# Or install both
pip install trackio wandb
```

### Testing Tracker Integration
```bash
# Test WandB with fake metrics
make wandb-test

# Test Trackio with fake metrics
make trackio-test
```

## Model-Specific Configurations

### ResNet-18 Presets
- `resnet18-state-pruning-compare` - Compare state pruning across optimizers
- `resnet18-movement-pruning-compare` - Compare movement pruning
- `resnet18-comprehensive-pruning-compare` - Test all combinations

### LeNet-5 Presets
- `lenet5` - Full test configuration
- `lenet5-adamwprune` - AdamWPrune specific testing
- `lenet5-sgd` - Baseline SGD configuration

## Advanced Usage

### Continuing Interrupted Test Runs

If your test matrix is interrupted (system crash, power failure, etc.), you can continue from where you left off:

```bash
# Continue the most recent interrupted test matrix
make continue
```

See [Continuation Documentation](docs/continue.md) for detailed information on resuming interrupted experiments.

### Reproduce All Results

```bash
# ResNet-18 testing (as used for September 2025 results)
make defconfig-resnet18-adam-all-pruning-methods
make

# Generate all visualizations
make update-graphs
```

### Custom Experiments

```bash
# Direct training with specific settings
cd resnet18
python train.py --optimizer adamwprune --pruning-method state --target-sparsity 0.7
```

## Optimizer Variants

- **SGD**: Baseline stochastic gradient descent
- **Adam**: Adaptive moment estimation
- **AdamW**: Adam with decoupled weight decay
- **AdamWAdv**: Enhanced with AMSGrad, cosine annealing, gradient clipping
- **AdamWSpam**: Spike-aware pruning with momentum reset
- **AdamWPrune**: State-based pruning using optimizer dynamics

📚 **[Understanding Adam Optimizers: A Complete Guide](docs/adam-optimizers.md)** - Learn about the evolution from Adam to AdamW and modern variants, with practical guidelines for choosing the right optimizer for your model.

## Movement Pruning

Based on ["Movement Pruning: Adaptive Sparsity by Fine-Tuning"](https://arxiv.org/abs/2005.07683) by Sanh et al. (2020). Tracks weight movement patterns to determine importance.

## References

- Movement Pruning: Victor Sanh, Thomas Wolf, Alexander M. Rush (2020). ["Movement Pruning: Adaptive Sparsity by Fine-Tuning" PDF](https://arxiv.org/abs/2005.07683) & ["Audio summary"](https://open.spotify.com/episode/0Vrw2FiL44wlxxU4QA2zxt?si=rP3Ifc8JT1-iQJuEklCL2g)
- SPAM: Tuan Nguyen, Tam Nguyen, Vinh Nguyen, Hoang Dang, Dung D. Le, Anh Tran (2024). ["SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training" PDF](https://arxiv.org/abs/2409.07321) & ["Audio summary"](https://open.spotify.com/episode/7vKFYxrH1o137zl9MfcKAz?si=oVMoHS61QD6Jjm3XYOTDNQ)
- Gradient Problems in RNNs: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013). ["On the difficulty of training recurrent neural networks" PDF](https://arxiv.org/abs/1211.5063) & ["Audio summary"](https://open.spotify.com/episode/0okbpKt5U4jmiYwqhVks1S?si=QeGK8t2MT5iYzcj5VE9dMw)
- Adam: Diederik P. Kingma, Jimmy Ba (2014). ["Adam: A Method for Stochastic Optimization" PDF](https://arxiv.org/abs/1412.6980) & ["Audio summary"](https://open.spotify.com/episode/6GIPqEzRvwHvRMYYI3M4Ar?si=hMWeNH9PR-O48or43EN2iQ)
- AdamW: Ilya Loshchilov, Frank Hutter (2019). ["Decoupled Weight Decay Regularization" PDF](https://arxiv.org/abs/1711.05101) & ["Audio summary"](https://open.spotify.com/episode/0s5ywoHyIS1dTTT2cLxPpV?si=h335wbgGQ0m94FsBtX-SxQ)
- Adafactor: Noam Shazeer, Mitchell Stern (2018). ["Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" PDF](https://arxiv.org/abs/1804.04235) & ["Audio summary"](https://open.spotify.com/episode/46DNk6Mkfk4r6xikZPzYT1?si=UUkAQyQEQai-rQypL_lqgA)

## Citation

If you use this work, please cite:

```bibtex
@misc{AdamWPrune2025,
  title        = {AdamWPrune: Multi-Model State-based Pruning},
  author       = {Luis Chamberlain},
  year         = {2025},
  howpublished = {\url{https://github.com/mcgrof/knlp}},
  note         = {State-based pruning across LeNet-5 and ResNet-18}
}
```

## License

All AdamWPrune code except scripts/kconfig is MIT licensed. The scripts/kconfig directory is GPLv2. The project as a whole is GPLv2. AI models generated by this project can be licensed as you choose.

See LICENSE for details.

## RATIO: Hardware-Aware Golden Ratio Architecture

**NEW**: RATIO combines inference-optimal architecture (golden ratio 1:2.5) with GPU-aligned tensor dimensions for maximum hardware efficiency.

### Key Innovations
- **Golden Ratio Enforcement**: Attention:MLP = 1:2.5 (inference-optimal vs GPT-2's 1:2.0)
- **GPU-Aligned Dimensions**: All tensors are multiples of 64 for tensor core efficiency (following Karpathy's nanoGPT vocab padding principle)
- **Structure-Aware Pruning**: New bitter7-9 variants that preserve golden ratio during pruning
- **MLA Compression**: 6× smaller KV cache (latent_dim=128)

### Pruning Variants for RATIO

**bitter7 (Conservative Variance)**: `|w| * (exp_avg_sq^0.25 + eps)`
- Uses Adam's second moment (beta2=0.999) which accumulates slowly
- Fourth root = maximum conservatism
- Best for: Finding parameters with long-term low activity

**bitter8 (RATIO Structure-Aware)**: `structural_weight * |w| * sqrt(|exp_avg| + eps)`  
- Attention params weighted 2.5× (scarce, 1/2.5 of capacity)
- MLP params weighted 1.0× (abundant, 2.5/2.5)
- Coupling params weighted 3.0× (critical bidirectional flow)
- Best for: RATIO models where ratio must be preserved

**bitter9 (RATIO Conservative - Recommended)**: `structural_weight * |w| * (exp_avg_sq^0.25 + eps)`
- Most conservative: combines structure + variance signals
- Best for: Final deployment with maximum confidence

### Configuration
```bash
make defconfig-gpt2-ratio-ablation && make
```

See `docs/ra.md` and `docs/ratio-pruning-variants.md` for technical details.

