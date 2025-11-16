# Experiment Tracking

This project supports multiple experiment tracking backends for logging
training metrics, GPU utilization, and system statistics. This enables
reproducible experiments and easy comparison across runs.

## Supported Trackers

### WandB (Weights & Biases)

WandB provides cloud-based experiment tracking with rich visualization
and collaboration features.

**Setup:**
```bash
pip install wandb
wandb login
```

**Enable in training:**
```bash
python gpt2/train.py --tracker wandb --tracker-project my-project
```

### Trackio

Trackio is a local-first experiment tracking system that stores
metrics in your home directory cache.

**Setup:**
```bash
pip install trackio
```

**Enable in training:**
```bash
python gpt2/train.py --tracker trackio --tracker-project my-project
```

**View dashboard:**
```bash
trackio show --project "my-project"
```

### Using Multiple Trackers

You can enable both trackers simultaneously:

```bash
python gpt2/train.py --tracker wandb,trackio --tracker-project my-project
```

## Configuration via Defconfig

Trackers can be configured in defconfig files using Kconfig syntax:

```bash
CONFIG_TRACKER="wandb,trackio"
CONFIG_TRACKER_PROJECT="gpt2-experiments"
CONFIG_TRACKER_RUN_NAME="baseline-adamw"
```

## Logged Metrics

### Training Metrics

- `train_loss`: Training loss (averaged over log_interval steps)
- `train_perplexity`: Training perplexity exp(loss)
- `val_loss`: Validation loss
- `val_perplexity`: Validation perplexity
- `learning_rate`: Current learning rate
- `sparsity`: Model sparsity (for pruning experiments)
- `final/best_val_perplexity`: Best validation perplexity (logged once
  at end)
- `final/best_val_loss`: Best validation loss (logged once at end)

### GPU Metrics

Aggregate GPU metrics (averaged across all GPUs):

- `gpu/compute_util_avg`: Average GPU compute utilization (%)
- `gpu/memory_util_avg`: Average GPU memory bandwidth utilization (%)
- `gpu/memory_used_avg_gb`: Average memory used per GPU (GB)
- `gpu/memory_used_total_gb`: Total memory used across all GPUs (GB)
- `gpu/memory_total_gb`: Total GPU memory available (GB)

Per-GPU metrics (logged by WandB agent automatically):

- `system.gpu.{i}.memory`: GPU memory allocated (MB)
- `system.gpu.{i}.memoryAllocated`: GPU memory allocated (%)
- `system.gpu.{i}.gpu`: GPU compute utilization (%)
- `system.gpu.{i}.temp`: GPU temperature (°C)
- `system.gpu.{i}.powerWatts`: GPU power consumption (W)

## WandB API Helper Scripts

The `scripts/` directory contains utilities for querying WandB data
programmatically. These require the wandb Python package to be
installed.

### Prerequisites

Install wandb and authenticate:

```bash
# Using system Python with micromamba environment
source ~/bin/wl700-ml  # Activates w7900-ml micromamba env
wandb login
```

Or install in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install wandb pandas
wandb login
```

### scripts/inspect_wandb_keys.py

Discover what metrics are available in a run. Useful when you're not
sure what keys WandB logged.

**Usage:**
```bash
python scripts/inspect_wandb_keys.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**Example output:**
```
Run: gpt2_adamwprune_bitter9_state_50 (v2h4vzbc)
State: finished
URL: https://wandb.ai/mcgrof-citizen/gpt2-bitter9-compiled-b200x4/runs/v2h4vzbc
--------------------------------------------------------------------------------

Total unique keys: 87

GPU-related keys:
  - gpu/compute_util_avg
  - gpu/memory_used_avg_gb
  - gpu/memory_util_avg
  - system.gpu.0.gpu
  - system.gpu.0.memory
  ...

Training metrics keys:
  - learning_rate
  - sparsity
  - train_loss
  - train_perplexity
  - val_loss
  - val_perplexity
  ...
```

### scripts/query_wandb_gpu.py

Query GPU metrics from training history. Uses the metrics logged
during training (custom metrics from our trainer).

**Usage:**
```bash
python scripts/query_wandb_gpu.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**Example output:**
```
Run: gpt2_adamwprune_bitter9_state_50 (v2h4vzbc)
State: finished
URL: https://wandb.ai/mcgrof-citizen/gpt2-bitter9-compiled-b200x4/runs/v2h4vzbc
--------------------------------------------------------------------------------

GPU 0:
  Memory Used:
    Current:  8543.2 MB (8.34 GB)
    Average:  8234.5 MB (8.04 GB)
    Maximum:  8678.1 MB (8.48 GB)
  Memory Allocated (%):
    Current:  35.6%
    Average:  34.3%
    Maximum:  36.2%
  GPU Utilization (%):
    Current:  78.2%
    Average:  72.5%
    Maximum:  94.3%

...

TOTAL ACROSS ALL 4 GPUs:
  Current Total Memory:  33.4 GB
  Average Total Memory:  32.2 GB
  Maximum Total Memory:  33.9 GB
```

### scripts/query_wandb_gpu_full.py

Query GPU metrics from WandB's system event stream. This provides
more detailed system-level metrics including power and temperature.

**Usage:**
```bash
python scripts/query_wandb_gpu_full.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**Example output:**
```
Run: gpt2_adamwprune_bitter9_state_50 (v2h4vzbc)
State: finished
URL: https://wandb.ai/mcgrof-citizen/gpt2-bitter9-compiled-b200x4/runs/v2h4vzbc
--------------------------------------------------------------------------------

Detected 4 active GPUs

GPU 0:
  Memory Allocated:
    Current:  8.34 GB (34.7%)
    Average:  8.04 GB (33.5%)
    Maximum:  8.48 GB (35.3%)
  GPU Utilization:
    Current:  78.2%
    Average:  72.5%
    Maximum:  94.3%
  Power:
    Current:  245.3 W
    Average:  238.7 W
    Maximum:  265.0 W
  Temperature:
    Current:  72°C
    Average:  68°C
    Maximum:  76°C

...

TOTAL ACROSS ALL 4 GPUs:
  Current Total Memory:  33.4 GB
  Average Total Memory:  32.2 GB
  Maximum Total Memory:  33.9 GB
```

## Querying Latest Run

All scripts support querying the latest run by omitting `--run-name`
and `--run-id`:

```bash
# Query most recent run in project
python scripts/query_wandb_gpu.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4
```

## Comparing Multiple Runs

To compare GPU metrics across multiple runs (e.g., baseline vs
optimized), create a custom Python script using the WandB API:

```python
#!/usr/bin/env python3
import wandb

api = wandb.Api()

# Define runs to compare
runs_to_compare = [
    "gpt2_adamwspam_magnitude_50",  # baseline
    "gpt2_adamwprune_bitter8_state_50",
    "gpt2_adamwprune_bitter9_state_50",
]

project = "mcgrof-citizen/gpt2-bitter9-compiled-b200x4"

for run_name in runs_to_compare:
    runs = api.runs(project, filters={"config.run_name": run_name})

    if not runs:
        print(f"Run not found: {run_name}")
        continue

    run = runs[0]

    # Get GPU metrics from history
    history = run.history(
        keys=["gpu/memory_util_avg", "gpu/compute_util_avg"],
        samples=1000
    )

    if not history.empty:
        mem_util = history["gpu/memory_util_avg"].mean()
        compute_util = history["gpu/compute_util_avg"].mean()

        print(f"{run_name}:")
        print(f"  GPU Memory Util: {mem_util:.2f}%")
        print(f"  GPU Compute Util: {compute_util:.2f}%")
        print()
```

## Environment Setup Script

For convenience, source the environment setup script before running
WandB queries:

```bash
source ~/bin/wl700-ml  # Activates micromamba w7900-ml environment
python scripts/query_wandb_gpu.py --project my-project
```

This script activates the `w7900-ml` micromamba environment which
includes wandb and other ML dependencies.

## Troubleshooting

### "ModuleNotFoundError: No module named 'wandb'"

Install wandb in your Python environment:

```bash
pip install wandb pandas
```

Or use the micromamba environment:

```bash
source ~/bin/wl700-ml
```

### "No GPU metrics found in this run"

Check that GPU monitoring was enabled during training. Ensure
pynvml is installed:

```bash
pip install pynvml
```

GPU metrics are logged automatically if pynvml is available and GPUs
are detected.

### WandB Authentication

If you get authentication errors, log in to WandB:

```bash
wandb login
```

Enter your API key when prompted. The key is stored in
`~/.netrc` for future use.

## Best Practices

### Project Naming

Use descriptive project names that indicate the experiment domain:

- `gpt2-bitter8-vs-bitter9-b200x4`: Comparing optimizations on
  specific hardware
- `gpt2-r-mlp-prune`: Reciprocal MLP pruning experiments
- `gpt2-ratio-ablation`: RATIO architecture ablation study

### Run Naming

Use systematic run names that encode configuration:

```
{model}_{optimizer}_{variant}_{method}_{sparsity}
```

Examples:
- `gpt2_adamwprune_bitter9_state_50`
- `gpt2_adamwspam_magnitude_50`
- `gpt2_r-mlp_v2_baseline`

This makes filtering and comparison straightforward.

### Tags and Groups

Use WandB tags and groups for organizing related experiments:

```bash
python gpt2/train.py \
  --tracker wandb \
  --tracker-project gpt2-ablation \
  --tracker-run-name step-05 \
  --tracker-tags "ablation,ra-ct,step-05" \
  --tracker-group "ratio-ablation"
```

### Final Summary Metrics

Important metrics should be logged once at the end of training under
the `final/` namespace:

```python
self.log_metrics({
    "final/best_val_perplexity": self.best_perplexity,
    "final/best_val_loss": self.best_val_loss,
})
```

This prevents duplicate time-series plots in WandB while preserving
summary statistics.
