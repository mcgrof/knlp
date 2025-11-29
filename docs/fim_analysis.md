# FIM (Fisher Information Matrix) Analysis

Unified workflow for extracting, analyzing, and visualizing Fisher Information Matrix metrics from W&B experiments.

## Quick Start

```bash
# 1. Extract and analyze FIM metrics from W&B
source ~/envs/w7900-ml/bin/activate
python scripts/analyze_fim_metrics.py \
  --entity mcgrof-citizen \
  --project gpt2-ra-v2-h100 \
  --output-dir test_matrix_results \
  --output-summary fim.txt

# 2. View human-readable summary
cat test_matrix_results/fim.txt
```

## What Gets Generated

### 1. JSON Export (`fim_metrics_<project>.json`)
Raw FIM metrics extracted from W&B API:
- Run metadata (name, ID, URL, state)
- Summary metrics (final values)
- Optional history (time series)

### 2. CSV Tall-Table (`fim_summary_<project>.csv`)
Standardized format: `run_name, layer, head, metric, value`

Example:
```csv
run_name,layer,head,metric,value
baseline,global,all,trace_global_mean,0.9165
baseline,layer0,head0,cond,4028828.5
baseline,layer0,head0,eigmax,0.0403
```

### 3. Human-Readable Summary (`fim.txt`)
Per-run analysis with interpretations:
- Global curvature scale (trace, eigmax)
- Anisotropy (condition number)
- Low-rank structure (energy_r8, energy_r16, effective rank)
- Cross-run comparisons
- Hotspot detection (extreme metric values)
- Interpretation guide

## Understanding FIM Metrics

### Trace (Sum of Eigenvalues)
**What it means:** Total curvature / total sensitivity of parameter block

**High trace hotspots:**
- Head/layer doing disproportionate representational work
- Critical feature extractor
- **DO NOT** compress or prune these heads
- Good for: adaptive precision, dual-tier KV cache

**Example:** `trace = 0.975` in layer11/head5 → critical head

### Eigmax (Largest Eigenvalue)
**What it means:** Maximum curvature direction / highest sensitivity

**High eigmax hotspots:**
- Extremely sensitive along one specific direction
- Handles rare/high-stakes token interactions
- **Best for:** KVSplice latent compression, top-eigenvector projection
- Usually rank-1 or rank-2 reconstruction recovers most behavior

**Example:** `eigmax = 0.28` in layer6/head8 → specialized head

### Condition Number (eigmax / eigmin)
**What it means:** Anisotropy / ill-conditioning

**High condition number:**
- Highly anisotropic (few key directions matter a LOT)
- Natural low-rank structure
- **Best compression targets** (can safely compress KV cache, projections)
- Prune low-energy directions with near-zero loss

**Interpretation:**
- `cond < 1e3`: Well-conditioned / isotropic
- `cond ~ 1e5`: Moderately anisotropic
- `cond ~ 1e7`: Strong anisotropy (some very stiff directions)
- `cond > 1e7`: Extremely ill-conditioned (few stiff directions dominate)

**Example:** `cond = 2.7e7` in layer6/head8 → excellent compression target

### Energy Metrics (energy_r8, energy_r16)
**What it means:** Fraction of total energy in top k eigenvalues

**Interpretation:**
- `energy_r8 = 0.95`: Almost all energy in top 8 modes → very low-rank
- `energy_r8 = 0.90`: Most energy in top 8 modes → low effective rank
- `energy_r8 = 0.80`: Substantial low-rank structure
- `energy_r8 < 0.80`: Energy more spread across spectrum

**Effective rank:** Smallest k where energy_rk ≥ 0.9

## Brilliant W&B Visualization

### Single-Number Insights (Scalars)

```python
from scripts.fim_wandb_viz import log_fim_to_wandb

log_fim_to_wandb(
    entity="mcgrof-citizen",
    project="gpt2-ra-v2-h100",
    run_id="abc123",
    df=fim_dataframe,
    global_stats=global_stats,
    low_rank_stats=low_rank_stats,
    summary_text=summary_text,
)
```

This logs:
- `fim/compression_potential` (0-100): High cond + high energy_r8 = very compressible
- `fim/efficiency_score` (0-100): Lower trace = more efficient
- `fim/effective_rank`: How many eigenvalues capture 90% energy
- `fim/trace_global_mean`: Overall curvature scale
- `fim/cond_global_mean`: Overall anisotropy

### Compression Heatmap (W&B Table)

Layer×head grid color-coded by compression potential:
- 🟢 **Green** (`high_potential`): High compression targets
  - High condition number + high low-rank structure
  - Safe to compress KV cache, projections, MLP
- 🔴 **Red** (`critical`): DO NOT compress
  - High trace (> 0.95)
  - Critical feature extractors
- 🟡 **Yellow** (`moderate`): Compression with caution

**Usage in W&B UI:**
1. Go to run page
2. Find "fim/compression_heatmap" table
3. Filter by `category == "high_potential"` to find compression targets
4. Filter by `category == "critical"` to find heads to protect

### Scatter Plot: Condition vs Trace

**Axes:**
- X: Condition number (log scale)
- Y: Trace
- Color: Layer depth
- Size: Effective rank

**Quadrants:**
- **Top-right:** Important but compressible (high trace, high cond)
- **Top-left:** Critical, not compressible (high trace, low cond)
- **Bottom-right:** Compressible, not critical (low trace, high cond) ← BEST TARGETS
- **Bottom-left:** Boring (low trace, low cond)

### HTML Panel

Full FIM interpretation with:
- Per-run summaries
- Cross-run comparisons
- Hotspot report
- Interpretation guide

Access in W&B: `fim/interpretation` artifact

## Compression Strategy Based on FIM

### Step 1: Identify Compression Targets
```bash
# Generate FIM analysis
python scripts/analyze_fim_metrics.py --entity X --project Y --output-dir results

# Find high compression potential heads
grep "high_potential" results/fim_summary_*.csv
```

### Step 2: Protect Critical Heads
```bash
# Find critical heads (DO NOT compress)
grep "critical" results/fim_summary_*.csv
```

### Step 3: Apply Tiered Compression

**High compression potential heads:**
- KV cache: 70-90% compression (low-rank projection)
- Projections: Quantize to FP8/FP4
- MLP: Aggressive pruning of low-energy directions

**Critical heads:**
- KV cache: Minimal compression (maybe 30-50%)
- Projections: Keep FP16/BF16
- MLP: No pruning

**Moderate heads:**
- KV cache: 50-70% compression
- Adaptive precision based on token difficulty

## Example Insights from RA Experiment

From `gpt2-ra-v2-h100` analysis:

### Cross-Run Comparison
```
RA-zero-gate: trace=0.912, cond=7.98e6  (more anisotropic)
RA-one-gate:  trace=0.948, cond=4.68e6  (less anisotropic, "fattier")
Baseline:     trace=0.917, cond=7.49e6
```

**Interpretation:**
- RA-zero-gate is most efficient (lowest trace)
- RA-zero-gate is most compressible (highest cond)
- RA-one-gate made model "fattier" (highest trace)

### Hotspots
```
Top condition number: layer6/head8 (cond=2.7e7)
  → Excellent KVSplice target
  → Can aggressively compress with minimal loss

Top trace: layer11/head5 (trace=0.975)
  → Critical head, DO NOT prune
  → Consider adaptive precision or dual-tier cache
```

## Integration with Test Matrix

Add FIM analysis to test matrix runs:

```python
# In gpt2/trainers/vanilla.py or base.py

def run_fim_analysis_and_log(self):
    """Run FIM analysis after training and log to W&B."""
    import subprocess

    # Generate FIM analysis
    subprocess.run([
        "python", "scripts/analyze_fim_metrics.py",
        "--entity", self.wandb_entity,
        "--project", self.wandb_project,
        "--output-dir", self.output_dir,
        "--output-summary", "fim.txt"
    ])

    # Read and parse results
    from scripts.fim_wandb_viz import log_fim_to_wandb
    import pandas as pd

    df = pd.read_csv(f"{self.output_dir}/fim_summary_{self.wandb_project}.csv")
    # ... compute stats ...

    log_fim_to_wandb(
        entity=self.wandb_entity,
        project=self.wandb_project,
        run_id=self.wandb_run_id,
        df=df,
        global_stats=global_stats,
        low_rank_stats=low_rank_stats,
        summary_text=summary_text,
    )
```

## References

- **Condition number:** eigmax / eigmin (anisotropy measure)
- **Trace:** Sum of eigenvalues (total curvature)
- **Effective rank:** Min k such that energy_rk ≥ threshold
- **Energy:** Σ(top k eigenvalues) / Σ(all eigenvalues)

## Future Enhancements

Potential additions:
1. **Temporal FIM tracking:** Track how metrics evolve during training
2. **Per-token FIM:** Analyze which tokens trigger high-curvature regions
3. **FIM-guided pruning:** Auto-generate pruning masks from FIM analysis
4. **Compression ratio estimator:** Predict achievable compression based on FIM
5. **W&B Report templates:** Auto-generate W&B reports with FIM insights
