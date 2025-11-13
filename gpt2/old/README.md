# Legacy GPT-2 Training Code

This directory contains deprecated training scripts and architecture implementations kept for reproducibility of past experiments.

## Contents

### Legacy Trainers (DEPRECATED)

**train_ra_mla.py** - Original monolithic trainer
- Handles both MLA-based RA (Gen 2) and Unified RA (V-series)
- Used by defconfigs in `defconfigs/old/`
- **Status**: Deprecated, use new modular trainers instead
- **Replacement**: `gpt2/trainers/unified_ra.py` for V-series ablations

**train_vanilla_original.py** - Original vanilla trainer
- Standard GPT-2 training with pruning support
- **Status**: Deprecated
- **Replacement**: `gpt2/trainers/vanilla.py`

### Legacy Architectures

**ra_mla_gpt2.py** - Generation 2 RA (MLA-based)
- DeepSeek MLA latent compression
- Local reciprocal band (ra_window=64)
- Reciprocal MLP mechanisms (3 types)
- **Status**: Legacy, kept for reproducibility
- **Defconfigs**: `defconfigs/old/gpt2-ratio-ablation*`

**ra_lens_gpt2.py** - Lens-gated RA (deprecated)
- Experimental lens-based gating
- **Status**: Deprecated, slower than Unified RA
- **Defconfigs**: `defconfigs/old/gpt2-lens-ablation`

**ra_mla_flash_pytorch.py** - FlashAttention variant
- Custom FlashAttention implementation for MLA
- **Status**: Experimental

## Migration Guide

### For New Experiments

**DON'T** use code in this directory. Use the new modular trainers:

```bash
# Vanilla GPT-2
python gpt2/train.py --architecture vanilla

# Unified RA (V-series)
python gpt2/train.py --architecture unified-ra --ra-step V1

# Ablation study
python gpt2/train.py --architecture unified-ra --ablation-mode \
  --ablation-steps V0,V1,V3,V7,V9
```

### For Reproducing Old Experiments

Old defconfigs in `defconfigs/old/` still reference these scripts:

```bash
# Legacy MLA-based RA experiments
make defconfig-old-gpt2-ratio-ablation
make  # Uses gpt2/old/train_ra_mla.py
```

## Evolution Timeline

1. **Generation 1** (Removed): Explicit S.T mixing
   - Status: Removed in commit c2bbed3
   - Problem: ~2× computational cost

2. **Generation 2** (Legacy): MLA-based RA
   - Location: `gpt2/old/ra_mla_gpt2.py`
   - Status: Kept for reproducibility
   - Used by: `defconfigs/old/gpt2-ratio-ablation*`

3. **Generation 3** (Current): Unified RA
   - Location: `ra.py` (project root)
   - Status: Production-ready
   - Trainers: `gpt2/trainers/unified_ra.py`
   - Defconfigs: `defconfigs/gpt2-unified-ra-ablation`

## Why Keep This?

Scientific reproducibility. All past experiments referenced in papers/reports
can still be reproduced using the exact same code that generated the results.

**New work should use the modular trainers in `gpt2/trainers/`.**
