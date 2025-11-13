# GPT-2 Training Usage Guide

## Quick Start

The GPT-2 training system now has a unified entry point with modular trainers.

### Vanilla GPT-2 Training

Standard GPT-2 without architectural modifications:

```bash
python gpt2/train.py \
  --architecture vanilla \
  --dataset finewebedu \
  --batch-size 8 \
  --gradient-accumulation 8 \
  --max-iters 10000 \
  --optimizer adamwspam
```

### Unified RA Single Run

Train a single V-series ablation step:

```bash
python gpt2/train.py \
  --architecture unified-ra \
  --ra-step V1 \
  --dataset finewebedu \
  --batch-size 8 \
  --max-time 7200
```

### Ablation Study

Run multiple steps sequentially:

```bash
python gpt2/train.py \
  --architecture unified-ra \
  --ablation-mode \
  --ablation-steps V0,V1,V3,V7,V9 \
  --dataset finewebedu \
  --max-time 7200
```

## Architecture Options

### `--architecture vanilla`

**Trainer**: `gpt2/trainers/vanilla.py` (`VanillaGPT2Trainer`)

**Features**:
- Standard GPT-2 model
- Multiple optimizers (AdamW, AdamWSPAM, AdamWPrune)
- Pruning support (magnitude, movement, state-based)
- Bitter variants (bitter2-bitter9) for AdamWPrune

**Use for**:
- Baseline comparisons
- Pruning experiments
- Standard GPT-2 training

### `--architecture unified-ra`

**Trainer**: `gpt2/trainers/unified_ra.py` (`UnifiedRATrainer`)

**Features**:
- Unified RA (folded layout, zero overhead)
- R-MLP support
- V-series ablation steps (V0-V10+)
- Gate analysis
- Self-restart mechanism

**Use for**:
- RA ablation studies
- R-MLP experiments
- Architecture research

## V-Series Ablation Steps

### Core Steps

- **V0**: Baseline GPT-2 (control)
- **V1**: Unified RA (R=4) ← **Primary innovation**
- **V2**: Unified RA + Self-Restart
- **V3**: Unified RA + R-MLP (basic)
- **V7**: Unified RA (R=8, higher capacity)
- **V9**: Unified RA (R=2, minimal capacity)

### R-MLP Variants

- **V4**: Unified RA + R-MLP + Mixer
- **V5**: Unified RA + R-MLP + Gates
- **V6**: Unified RA + R-MLP + All mechanisms

### Parameter Sweeps

- **V8**: Unified RA (R=8) + Self-Restart
- **V10**: Unified RA + Self-Restart + 6x MLP

## Common Options

### Training Duration

```bash
# Time-based (recommended for ablations)
--max-time 7200  # 2 hours in seconds

# Iteration-based (traditional)
--max-iters 10000
```

### Batch Size

```bash
--batch-size 8
--gradient-accumulation 8  # Effective batch = 8 * 8 = 64
```

### Optimizer

```bash
--optimizer adamwspam  # Recommended for RA experiments
--learning-rate 6e-4
--warmup-steps 200
```

### Experiment Tracking

```bash
--tracker wandb
--tracker-project my-gpt2-experiments
--tracker-run-name v1-baseline
```

### Checkpointing

```bash
--save-checkpoint
--checkpoint-interval 1000
--output-dir ./checkpoints
```

## Examples

### Quick Sanity Check (10 minutes)

```bash
python gpt2/train.py \
  --architecture unified-ra \
  --ra-step V1 \
  --max-time 600 \
  --dataset finewebedu
```

### Production RA Ablation (2 hours per step)

```bash
python gpt2/train.py \
  --architecture unified-ra \
  --ablation-mode \
  --ablation-steps V0,V1,V3,V7,V9 \
  --max-time 7200 \
  --dataset finewebedu \
  --tracker wandb \
  --save-checkpoint
```

### Vanilla Pruning Experiment

```bash
python gpt2/train.py \
  --architecture vanilla \
  --pruning-method state \
  --target-sparsity 0.5 \
  --optimizer adamwprune \
  --adamwprune-variant bitter7 \
  --max-iters 13000
```

### DDP Training (4 GPUs)

```bash
torchrun --nproc_per_node=4 gpt2/train.py \
  --architecture unified-ra \
  --ra-step V1 \
  --batch-size 8 \
  --gradient-accumulation 2
```

## Directory Structure

```
gpt2/
├── train.py                 # Unified dispatcher (main entry point)
├── trainers/
│   ├── __init__.py
│   ├── base.py              # BaseGPT2Trainer (common code)
│   ├── vanilla.py           # VanillaGPT2Trainer
│   ├── unified_ra.py        # UnifiedRATrainer
│   └── ablation.py          # AblationCoordinator
├── model.py                 # GPT-2 model
├── ra_v5_patch.py           # RA patching utilities
└── old/                     # Legacy code (archived)
    ├── train_ra_mla.py      # Old monolithic trainer
    └── README.md            # Migration guide
```

## Troubleshooting

### Import Errors

If you see "ModuleNotFoundError: No module named 'gpt2'":

```bash
# Run from project root, not gpt2/ subdirectory
cd /path/to/project
python gpt2/train.py ...
```

### OOM Errors

Reduce batch size or gradient accumulation:

```bash
--batch-size 4 --gradient-accumulation 16  # Same effective batch
```

### Slow Training

Enable compilation:

```bash
--compile  # Uses torch.compile() for speedup
```

## Migration from Old Code

### If you have old defconfigs

Old defconfigs in `defconfigs/old/` still work:

```bash
make defconfig-old-gpt2-ratio-ablation
make  # Uses gpt2/old/train_ra_mla.py
```

### If you want to migrate to new trainers

Replace manual script invocation with:

```bash
# Old
python gpt2/train_ra_mla.py --ra-mla-ablation-step V1 ...

# New
python gpt2/train.py --architecture unified-ra --ra-step V1 ...
```

## Getting Help

- **Documentation**: See `gpt2/REFACTORING.md` for architecture details
- **Legacy code**: See `gpt2/old/README.md` for old experiments
- **RA evolution**: See `docs/ra.md` for generation timeline
