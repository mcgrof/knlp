# Mechanistic Interpretability Infrastructure

This document describes the mechanistic interpretability (mechint)
infrastructure for post-training analysis of trained models. The framework
provides tools to discover sparse circuits, analyze feature importance, and
understand model internals through automated ablation and optimization.

## Design Philosophy

The mechint infrastructure is designed as a modular, extensible framework
supporting multiple analysis methods. Each method freezes trained model
weights and optimizes learnable masks or probes to identify minimal circuits
driving specific behaviors or metrics.

Key principles:

- **Post-training analysis**: Works with existing checkpoints, no retraining
- **Frozen model weights**: Only masks/probes are optimized
- **Multiple checkpoint support**: Analyze ablation runs with glob patterns
- **Extensible architecture**: Easy to add new analysis methods
- **W&B integration**: Interactive visualization of discovered circuits
- **CLI-driven workflow**: Kconfig + Makefile for reproducibility

## Architecture Overview

```
lib/mechint/              # Core mechanistic interpretability library
├── __init__.py          # Module exports
├── kv_circuit.py        # KV feature-circuit analysis (example)
├── visualize.py         # W&B visualization and plotting
└── [future methods]     # Neuron analysis, attention heads, etc.

scripts/
└── run_mechint_analysis.py  # CLI runner for all analysis methods

gpt2/Kconfig.mechint     # Configuration menu for mechint options

gpt2/defconfigs/
└── gpt2-*-mechint       # Example configurations
```

## Analysis Methods

### KV Feature-Circuit Analysis (Implemented)

Discovers sparse circuits in attention key-value projections using binary
channel masking. Based on "Scaling Sparse Feature Circuit Finding to Gemma 9B"
methodology.

**How it works:**

1. Load trained checkpoint and freeze all model weights
2. Inject learnable binary masks over attention K/V channels
3. Optimize masks to maintain target metric while maximizing sparsity
4. Use straight-through estimators (STE) for gradient flow through discrete masks
5. Apply continuous sparsification schedule (cubic/linear/exponential)
6. Anneal temperature for smooth convergence to binary masks
7. Save learned masks and generate visualizations

**Output:**

- Per-layer channel importance heatmaps
- Sparsity vs loss curves during optimization
- Faithfulness plots (pruned vs unpruned performance)
- Circuit summary reports in markdown
- W&B logs for interactive exploration

**Configuration options** (see `gpt2/Kconfig.mechint`):

- `KNLP_MECHINT_KV_TARGET_SPARSITY`: Fraction to prune (e.g., 0.95 = 95%)
- `KNLP_MECHINT_KV_STEPS`: Optimization iterations (500 recommended)
- `KNLP_MECHINT_KV_LR`: Learning rate for mask optimization
- `KNLP_MECHINT_KV_TEMP_SCHEDULE`: Temperature annealing (linear/exponential/cosine)
- `KNLP_MECHINT_KV_SPARSITY_SCHEDULE`: Sparsity ramp (cubic/linear/exponential)
- `KNLP_MECHINT_KV_L1_LAMBDA`: L1 regularization strength
- `KNLP_MECHINT_KV_HARD_MASKS`: Binary vs soft masks

### Future Analysis Methods (Planned)

The infrastructure is designed to support additional methods:

- **Neuron importance analysis**: Identify critical MLP neurons
- **Attention head pruning**: Find minimal attention head subsets
- **Layer-wise relevance**: Track information flow across layers
- **Activation patching**: Circuit discovery via targeted interventions
- **Causal tracing**: Path analysis for specific model behaviors

Each method follows the same pattern: freeze weights, optimize probes/masks,
preserve target metrics, maximize interpretability.

## Configuration System

All mechint analysis is configured through Kconfig. Enable the master switch
and choose an analysis mode:

```kconfig
CONFIG_KNLP_MECHINT=y

# Choose one:
CONFIG_KNLP_MECHINT_KV_POSTTRAIN=y      # Run new analysis
CONFIG_KNLP_MECHINT_KV_FROM_RUNS=y     # Visualize existing results
```

### Checkpoint Discovery

The infrastructure supports multiple checkpoint sources:

**Single checkpoint:**
```kconfig
CONFIG_KNLP_MECHINT_KV_CHECKPOINT="./output/final_model.pt"
```

**Glob pattern (ablation runs):**
```kconfig
CONFIG_KNLP_MECHINT_KV_CHECKPOINT="./output/final_model_step*.pt"
```
Automatically discovers and analyzes stepV0, stepV1, etc.

**Directory:**
```kconfig
CONFIG_KNLP_MECHINT_KV_CHECKPOINT="./checkpoints"
```
Analyzes all .pt files in directory.

### MODELS CLI Override

The `MODELS=` command-line override provides dynamic checkpoint selection
without editing defconfigs. The system intelligently adapts based on directory
contents:

**Analysis results directory** (contains `final_masks.pt`):
```bash
make defconfig-gpt2-kv-tying-w7900-ablation-mechint MODELS=mechint_analysis_kv
# → Enables FROM_RUNS visualization-only mode
```

**Checkpoint directory** (contains .pt files):
```bash
make defconfig-gpt2-kv-tying-w7900-ablation-mechint MODELS=./other_output
# → Updates checkpoint pattern to ./other_output/final_model_step*.pt
```

**Explicit checkpoint path:**
```bash
make defconfig-gpt2-kv-tying-w7900-ablation-mechint MODELS=./custom.pt
# → Sets explicit checkpoint path
```

This mirrors the existing `BASELINE=` override for baseline run comparison.

## Workflow Examples

### Basic Single-Checkpoint Analysis

Train a model, then analyze the final checkpoint:

```bash
# 1. Train model
make defconfig-gpt2-vanilla-baseline
make

# 2. Run KV circuit analysis
make defconfig-gpt2-mechint-kv-analysis
make mechint  # (when integrated)
```

Results saved to `mechint_analysis_kv/`:
- `final_masks.pt`: Learned binary masks
- `kv_channel_importance.png`: Heatmaps per layer
- `sparsity_curves.png`: Optimization progress
- `circuit_analysis_report.md`: Summary statistics

### Ablation Run Analysis

Train multiple model variants, analyze all checkpoints:

```bash
# 1. Train ablation run (saves stepV0.pt, stepV1.pt automatically)
make defconfig-gpt2-kv-tying-w7900-ablation
make

# 2. Analyze all ablation steps
make defconfig-gpt2-kv-tying-w7900-ablation-mechint
make mechint
```

The defconfig uses glob pattern `./output/final_model_step*.pt` to
automatically discover all ablation checkpoints. Each step gets independent
circuit analysis, enabling comparison of KV importance across architectural
variants.

### Cross-Comparison Workflow

Compare circuits from different training runs:

```bash
# Train run A
make defconfig-gpt2-baseline
make
mv ./output ./output_baseline

# Train run B
make defconfig-gpt2-kv-tying
make
mv ./output ./output_kv_tying

# Analyze baseline
make defconfig-gpt2-mechint-kv-analysis MODELS=./output_baseline
make mechint

# Analyze KV tying
make defconfig-gpt2-mechint-kv-analysis MODELS=./output_kv_tying
make mechint

# Compare results in W&B
```

Each run creates separate analysis directories. W&B project tracks all runs
for side-by-side comparison of discovered circuits.

### Visualization-Only Mode

Re-visualize existing analysis without rerunning optimization:

```bash
make defconfig-gpt2-mechint-kv-analysis MODELS=mechint_analysis_kv
make mechint
```

When MODELS points to a directory containing `final_masks.pt`, automatically
enables `FROM_RUNS` mode. Regenerates plots and W&B logs from saved masks.

## Integration with Training Workflow

### Automatic Checkpoint Saving

All trainers now save step-specific checkpoints for ablation runs:

- Vanilla trainer: `final_model_stepV0.pt`, `final_model_stepV1.pt`, etc.
- RA trainer: `final_model_stepV0.pt`, `final_model_stepV3.pt`, etc.
- Best checkpoints: `best_model_stepV0.pt`, `best_model_stepV1.pt`, etc.
- Metrics: `training_metrics_stepV0.json`, etc.

This enables full ablation study analysis without manual checkpoint management.

### Default Output Locations

Training uses `./output/` by default (configurable via `--output-dir`).
Mechint analysis defaults to analyzing `./output/final_model_step*.pt`.

For W7900-specific runs, use dedicated output directories:

```bash
# Training
make defconfig-gpt2-kv-tying-w7900-ablation
# Saves to: ./output/final_model_stepV*.pt

# Analysis
make defconfig-gpt2-kv-tying-w7900-ablation-mechint
# Reads from: ./output/final_model_stepV*.pt
# Saves to: mechint_analysis_kv_w7900/
```

## W&B Integration

All analysis methods support W&B logging for interactive exploration:

```kconfig
CONFIG_KNLP_MECHINT_VISUALIZE_WANDB=y
CONFIG_WANDB_PROJECT="gpt2-mechint-kv-circuits"
```

**Logged artifacts:**

- Optimization curves (loss, sparsity, temperature)
- Per-layer heatmaps (channel importance visualization)
- Summary tables (sparsity, mean importance, channels kept)
- Learned masks as artifacts (downloadable for later use)
- Faithfulness metrics (degradation vs sparsity trade-off)

**Querying results:**

Use W&B API for programmatic analysis:

```python
import wandb
api = wandb.Api()
runs = api.runs("gpt2-mechint-kv-circuits")

for run in runs:
    sparsity = run.summary.get("mechint/sparsity")
    loss = run.summary.get("mechint/loss")
    print(f"{run.name}: {sparsity:.1%} sparsity, {loss:.4f} loss")
```

## Implementation Details

### Straight-Through Estimator (STE)

Binary masks require discrete {0,1} values, but optimization needs gradients.
STE solves this:

```python
# Forward: discrete binary mask
probs = sigmoid(logits / temperature)
hard_mask = (probs > 0.5).float()

# Backward: gradient flows through soft probabilities
mask = hard_mask + (probs - probs.detach())
```

Forward pass uses crisp binary decisions. Backward pass uses relaxed sigmoid
gradients. Temperature annealing gradually sharpens the sigmoid.

### Continuous Sparsification

Instead of step-function thresholding, smoothly increase sparsity target:

```python
# Cubic schedule (gentle start, aggressive end)
progress = step / total_steps
current_sparsity = target_sparsity * (progress ** 3)

# At step 250/500 with target=0.95:
# progress = 0.5
# current_sparsity = 0.95 * 0.125 = 0.119 (11.9%)

# At step 450/500:
# progress = 0.9
# current_sparsity = 0.95 * 0.729 = 0.693 (69.3%)
```

Allows model to adapt gradually rather than shock pruning.

### Temperature Annealing

High temperature = soft probabilities (gradients everywhere).
Low temperature = sharp decisions (binary-like).

```python
# Linear schedule: T = 1.0 → 0.1
temp = start_temp - (start_temp - end_temp) * (step / total_steps)

# Early (step=100/500, T=0.82):
sigmoid(2.0 / 0.82) = 0.917  # soft
sigmoid(-2.0 / 0.82) = 0.083

# Late (step=450/500, T=0.19):
sigmoid(2.0 / 0.19) = 0.9999  # hard
sigmoid(-2.0 / 0.19) = 0.0001
```

Smooth transition from exploration to exploitation.

## Extending the Framework

To add a new analysis method:

### 1. Create Analysis Module

Add `lib/mechint/your_method.py`:

```python
class YourMethodAnalyzer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        # Inject your probes/masks

    def run_analysis(self, train_loader, val_loader):
        # Optimize your probes
        # Return results dict
        pass

    def save_results(self):
        # Save learned parameters
        pass
```

### 2. Add Kconfig Options

Edit `gpt2/Kconfig.mechint`:

```kconfig
config KNLP_MECHINT_YOUR_METHOD
    bool "Your method description"
    help
      Detailed explanation of what this method does.
```

### 3. Update Runner Script

Edit `scripts/run_mechint_analysis.py`:

```python
if config.KNLP_MECHINT_YOUR_METHOD:
    from lib.mechint.your_method import YourMethodAnalyzer
    analyzer = YourMethodAnalyzer(model, config, device)
    results = analyzer.run_analysis(train_loader, val_loader)
```

### 4. Add Visualization

Extend `lib/mechint/visualize.py` with method-specific plots.

### 5. Create Example Defconfig

Add `gpt2/defconfigs/gpt2-your-method-mechint` showing recommended settings.

## Performance Considerations

### Memory Usage

Analysis requires:
- Model parameters (loaded in eval mode)
- Learnable masks/probes (small overhead)
- Batch of data for forward passes
- Gradient buffers (only for masks, not model)

**Optimization:**
- Use smaller batch size than training (8 vs 32)
- Disable training-specific features (dropout, augmentation)
- Enable mixed precision for faster forward passes
- Use `torch.no_grad()` for metric computation

### Computational Cost

KV circuit analysis on GPT-2 124M:
- 500 optimization steps × ~5 seconds/step = ~40 minutes (W7900)
- Scales with model size, not dataset size (uses small eval set)
- GPU utilization: 60-80% (forward-bound, not memory-bound)

### Batch Size Tuning

Larger batches = more stable gradients, slower iteration:
- 4: Fast iteration, noisy gradients
- 8: Recommended balance
- 16: Slower but smoother convergence

Test with dry-run mode to find optimal batch size for your GPU.

## Troubleshooting

### Analysis diverges (loss increases)

- **Cause:** Learning rate too high or sparsity ramping too fast
- **Fix:** Reduce `KV_LR` to 0.001 or use gentler schedule (cubic → linear)

### Binary masks don't sharpen

- **Cause:** Temperature not annealing or insufficient steps
- **Fix:** Check `TEMP_SCHEDULE` is "linear:1.0:0.1", increase `KV_STEPS`

### OOM during analysis

- **Cause:** Batch size too large or model doesn't fit
- **Fix:** Reduce `BATCH_SIZE` to 4, ensure mixed precision enabled

### Checkpoints not discovered

- **Cause:** Glob pattern doesn't match filenames
- **Fix:** Check actual checkpoint names, adjust pattern or use MODELS=

### W&B not logging

- **Cause:** Not authenticated or project doesn't exist
- **Fix:** Run `wandb login`, check project name matches config

## References

**Papers:**
- "Scaling Sparse Feature Circuit Finding to Gemma 9B" (inspiration for KV method)
- "Movement Pruning" (gradient-based importance)
- "Lottery Ticket Hypothesis" (sparse subnetwork discovery)

**Implementation:**
- `lib/mechint/kv_circuit.py`: KV feature-circuit analysis
- `lib/mechint/visualize.py`: Visualization and W&B logging
- `scripts/run_mechint_analysis.py`: CLI runner
- `gpt2/Kconfig.mechint`: Configuration options

**Example defconfigs:**
- `gpt2/defconfigs/gpt2-mechint-kv-analysis`: Generic KV analysis
- `gpt2/defconfigs/gpt2-kv-tying-w7900-ablation-mechint`: W7900-specific ablation
