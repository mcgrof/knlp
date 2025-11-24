# Quick Start Guide

## Kernel-Style Workflow

### Step 1: Choose a Defconfig

```bash
# List available defconfigs
make list-defconfigs

# Load a defconfig (kernel-style: make defconfig-<name>)
make defconfig-gpt2-kv-tying-w7900-ablation

# Or use interactive menu (kernel-style: make menuconfig)
make menuconfig
```

### Step 2: Run Experiment

```bash
# Standard workflow
make  # Runs training or test matrix based on config

# With CLI overrides (like kernel KBUILD_*)
make TRACKER=wandb  # Enable W&B tracking
make TIME=3600      # Override max training time
make MODELS=./checkpoints  # Use custom checkpoint dir
```

### Step 2b: Skip Baseline with Reference Run (Optional)

When running ablation studies, skip expensive baseline re-runs by referencing
a previous baseline:

```bash
# Load ablation defconfig with baseline reference
make defconfig-gpt2-r-mlp-prune BASELINE=mcgrof/old-project/abc123

# Run with baseline reference (skips M0, runs M1-M3)
make BASELINE=mcgrof/old-project/abc123
```

**Note:** BASELINE must be specified on both commands to ensure `config.py`
is regenerated with the baseline reference.

**What happens:**

1. Baseline run automatically copied to current project
2. Baseline step (M0, V0, L0, S0, R0, C0) filtered from test matrix
3. Only non-baseline steps executed (M1, M2, M3, etc.)

**Benefits:**

- Save hours on expensive baseline re-runs
- Compare experiments across different projects
- Use known-good baseline from previous work

**Format:** `entity/project/run_id` (from W&B run URL)

See [scripts/copy_wandb_run.py](../scripts/copy_wandb_run.py) for manual run
copying.

### Step 3: Analyze Results

```bash
# Generate visualizations
make update-graphs

# Run mechanistic interpretability analysis
make defconfig-gpt2-kv-tying-w7900-ablation-mechint MODELS=./output
make mechint
```

**Auto-generated project names**: `{model}-{5char-checksum}` (e.g.,
`gpt2-a3f2c`, `resnet50-7b9d1`)

- Consistent across runs from same directory
- No collisions between machines/directories
- No manual configuration needed

See [experiment-tracking.md](experiment-tracking.md) for detailed tracking
configuration.

## Model-Specific Examples

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
