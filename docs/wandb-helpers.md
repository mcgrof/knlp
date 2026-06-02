# WandB helper scripts

When analyzing experiment results or comparing GPU performance across
runs, use the W&B query scripts in the scripts/ directory. These
require the micromamba environment.

### Environment Setup

Before running any W&B query scripts:

```bash
source ~/bin/wl700-ml  # Activates w7900-ml micromamba environment
```

This provides wandb, pandas, and other dependencies needed for
querying experiment data.

### Available Scripts

**scripts/inspect_wandb_keys.py**: Discover available metrics in a run

Usage for inspecting what data is available:
```bash
python scripts/inspect_wandb_keys.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**scripts/query_wandb_gpu.py**: Query GPU metrics from training history

Usage for checking GPU memory and compute utilization:
```bash
python scripts/query_wandb_gpu.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**scripts/query_wandb_gpu_full.py**: Query detailed GPU metrics from
system events

Usage for detailed system metrics including power and temperature:
```bash
python scripts/query_wandb_gpu_full.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**scripts/plot_torch_compile_impact.py**: Generate publication-quality
visualizations comparing GPU performance across runs

This is a reusable visualization script that queries W&B and
generates four graphs showing performance comparisons. Used to
prove torch.compile() was the bottleneck.

Usage:
```bash
source ~/bin/wl700-ml
python scripts/plot_torch_compile_impact.py
```

The script is hardcoded to query
`mcgrof-citizen/gpt2-bitter8-nocompile-w7900` but can be easily
adapted for other projects by editing the `project` variable in
`main()`.

Generated graphs (300 DPI, publication quality):
- `torch_compile_comparison.png`: Side-by-side memory and compute
  comparison
- `torch_compile_grouped.png`: All runs in grouped bar chart with
  color coding
- `torch_compile_before_after.png`: Dramatic before/after horizontal
  bars with annotations
- `bitter8_vs_baseline.png`: Spotlight showing minimal overhead of
  state-based pruning

The script demonstrates the pattern for:
1. Querying W&B API for multiple runs
2. Extracting system.gpu.* metrics from event stream
3. Computing averages across runs
4. Creating matplotlib visualizations with annotations
5. Using color coding (red=bad, green=good) for clarity

When to use this script:
- After GPU profiling reveals performance differences
- To prove bottleneck hypotheses with visual evidence
- To compare optimization variants systematically
- To generate graphs for documentation or papers

Customization tips:
- Edit `project` variable to query different W&B project
- Modify `fetch_wandb_data()` to extract different metrics
- Update graph functions to change visual style
- Add new graph types by creating new functions following existing
  patterns

### Comparing Runs

To compare GPU performance across multiple runs (baseline vs
optimizations), write a custom Python script using the W&B API.
See docs/tracker.md for detailed examples.

Pattern for comparing runs:
```python
import wandb

api = wandb.Api()
project = "mcgrof-citizen/gpt2-bitter9-compiled-b200x4"

run_names = ["baseline", "bitter8", "bitter9"]

for name in run_names:
    runs = api.runs(project, filters={"config.run_name": name})
    if runs:
        run = runs[0]
        history = run.history(
            keys=["gpu/memory_util_avg", "gpu/compute_util_avg"],
            samples=1000
        )
        if not history.empty:
            print(f"{name}:")
            print(f"  Memory: {history['gpu/memory_util_avg'].mean():.2f}%")
            print(f"  Compute: {history['gpu/compute_util_avg'].mean():.2f}%")
```

### Key Metrics to Check

When analyzing GPU performance issues:

- `gpu/memory_util_avg`: Memory bandwidth utilization (%)
- `gpu/compute_util_avg`: Compute utilization (%)
- `gpu/memory_used_avg_gb`: Average memory per GPU (GB)

Low memory utilization (<20%) indicates memory bandwidth bottleneck.
Low compute utilization (<50%) indicates compute bottleneck.
Compare optimization runs to baseline to verify improvements.

