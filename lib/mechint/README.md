# Mechanistic Interpretability Analysis Tools

Post-training analysis tools for discovering sparse feature circuits in transformer models via KV channel masking.

## Quick Start

### Clone and Setup

```bash
git clone <your-repo-url>
cd knlp
pip install torch numpy matplotlib wandb
```

### Analyze Pre-Trained Models

If you have trained model checkpoints (e.g., `model_v0.pt`, `model_v1.pt`):

```bash
# Analyze a single model
python3 scripts/run_mechint_analysis.py \
  --checkpoint /path/to/model.pt \
  --dataset finewebedu \
  --output-dir mechint_results \
  --steps 500 \
  --target-sparsity 0.95

# Analyze multiple variants and compare
python3 scripts/run_mechint_analysis.py --checkpoint model_v0.pt --dataset finewebedu
python3 scripts/run_mechint_analysis.py --checkpoint model_v1.pt --dataset finewebedu

# Compare variants (V0 as baseline)
python3 scripts/compare_mechint_variants.py \
  --v0 mechint_analysis_finewebedu_model_v0 \
  --v1 mechint_analysis_finewebedu_model_v1 \
  --output comparison_results
```

## What You Get

### Per-Model Analysis

Each analysis run produces:

1. **KV Channel Importance Heatmaps**: Visualize which Key/Value channels matter per layer
2. **Sparsity Curves**: Track optimization progress (loss, sparsity, temperature)
3. **Circuit Analysis Report**: Markdown summary with metrics and per-layer breakdown
4. **Learned Masks**: Binary masks identifying important channels (`final_masks.pt`)

Output structure:
```
mechint_analysis_<dataset>_<model>/
├── kv_channel_importance.png       # Combined heatmap (all layers)
├── transformer.h.0.attn_importance.png  # Per-layer heatmaps
├── transformer.h.1.attn_importance.png
├── ...
├── sparsity_curves.png             # Optimization progress
├── circuit_analysis_report.md      # Text summary
└── final_masks.pt                  # Learned importance masks
```

### Variant Comparison

When comparing two variants (e.g., baseline vs KV-tying):

1. **Overall Summary**: Side-by-side loss degradation and sparsity comparison
2. **Delta Heatmap**: Per-layer sparsity differences from baseline (color-coded)
3. **Key Findings**: Automated analysis of differences
   - Which variant performs better
   - Sparsity pattern differences by layer group (early/middle/late)
   - Top 3 layers with biggest differences
4. **Detailed Comparisons**: Per-layer sparsity, importance, and channels kept
5. **Interactive W&B Table**: Sortable comparison with delta columns

Output structure:
```
comparison_results/
├── overall_summary.png          # Metrics comparison bar chart
├── delta_heatmap.png           # V1 - V0 sparsity per layer
├── key_findings.png            # Automated textual analysis
├── key_findings.txt            # Text version of findings
├── sparsity_comparison.png     # Per-layer sparsity bars
├── importance_comparison.png   # Per-layer mean importance
├── channels_comparison.png     # Per-layer channels kept
└── variant_comparison.md       # Markdown report with delta table
```

## Configuration Options

### Analysis Parameters

- `--target-sparsity`: Target fraction of channels to prune (default: 0.95)
- `--steps`: Optimization steps for mask learning (default: 500)
- `--lr`: Learning rate for mask optimization (default: 0.01)
- `--dataset`: Dataset name (default: finewebedu)

### W&B Integration

Results automatically log to Weights & Biases:

```bash
# Logs to CONFIG_TRACKER_PROJECT from your .config
python3 scripts/run_mechint_analysis.py --checkpoint model.pt

# Override project name
python3 scripts/compare_mechint_variants.py \
  --v0 dir_v0 --v1 dir_v1 \
  --project my-mechint-project

# Disable W&B logging
python3 scripts/run_mechint_analysis.py --checkpoint model.pt --no-wandb
```

## Example: Analyzing Multiple Checkpoints

```bash
#!/bin/bash
# Analyze all models in a directory

MODELS_DIR="/path/to/checkpoints"
OUTPUT_BASE="mechint_analysis"

# Analyze each model
for model in $MODELS_DIR/*.pt; do
    name=$(basename $model .pt)
    python3 scripts/run_mechint_analysis.py \
        --checkpoint $model \
        --dataset finewebedu \
        --output-dir ${OUTPUT_BASE}_${name}
done

# Compare first two variants
V0_DIR=$(ls -d ${OUTPUT_BASE}_* | head -1)
V1_DIR=$(ls -d ${OUTPUT_BASE}_* | head -2 | tail -1)

python3 scripts/compare_mechint_variants.py \
    --v0 $V0_DIR \
    --v1 $V1_DIR \
    --output comparison_v0_v1
```

## Interpretation Guide

### Key Findings Example

```
KEY FINDINGS
============================================================

• V1 has WORSE loss degradation: -18.59% vs -19.31%

Sparsity Pattern Differences:
  Early layers (0-3):  V1 is -2.5% less sparse
  Middle layers (4-7): V1 is -0.2% less sparse
  Late layers (8-11):  V1 is +4.9% more sparse

Biggest Differences:
  h.6: +8.6%
  h.5: +8.3%
  h.7: +7.7%
```

**Interpretation**:
- V1 variant performs slightly worse (less loss reduction from pruning)
- V1 retains more channels in early layers (less aggressive pruning)
- V1 is more aggressive in late layers (higher sparsity)
- Layers 5-7 show the most architectural difference

### Sparsity Metrics

- **Target Sparsity**: Goal sparsity level (e.g., 95% = keep only 5% of channels)
- **Achieved Sparsity**: Actual sparsity after optimization
- **Loss Degradation**: Percentage change in loss after pruning (negative = improvement)
- **Faithfulness**: How well the pruned model preserves original behavior

## Advanced Usage

### Visualization Only

Re-create visualizations from existing masks:

```bash
python3 scripts/run_mechint_analysis.py \
  --visualize-only /path/to/mechint_analysis_dir
```

### Custom Comparison

Compare more than 2 variants or use different baseline:

```python
from lib.mechint import compare_variants

variant_dirs = {
    "Baseline": "mechint_analysis_v0",
    "KV-Tying": "mechint_analysis_v1",
    "K=V^T": "mechint_analysis_v2",
}

compare_variants(
    variant_dirs=variant_dirs,
    output_dir="multi_variant_comparison",
    project_name="my-mechint-project",
    use_wandb=True,
)
```

## Citation

Based on techniques from:
- "Scaling Sparse Feature Circuit Finding to Gemma 9B"
- Mechanistic interpretability research on sparse circuit discovery

## Troubleshooting

**ImportError**: Install dependencies with `pip install torch numpy matplotlib wandb`

**OOM Errors**: Reduce batch size in dummy data generation or use smaller models

**No models found**: Check glob pattern matches your checkpoint naming (default: `./output/final_model_step*.pt`)

**W&B login required**: Run `wandb login` or use `--no-wandb` flag
