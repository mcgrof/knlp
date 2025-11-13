# GPT-2 Training Refactoring Plan

## Goal

Unify `train.py` and `train_ra_mla.py` into a modular architecture that:
- Eliminates code duplication
- Clarifies which trainer handles which experiments
- Makes it easy to add new GPT-2 variants
- Preserves backward compatibility with existing defconfigs

## Current State (Before Refactoring)

```
gpt2/
├── train.py (1213 lines)
│   └── Vanilla GPT-2 + pruning evaluation
└── train_ra_mla.py (3318 lines)
    ├── Everything from train.py PLUS
    ├── Legacy RA (Gen 2: MLA-based, steps 0-18)
    └── Unified RA (Gen 3, V-series V0-V19)
```

**Problems**:
- 80% code duplication between files
- Confusing naming (train_ra_mla handles both old and new RA)
- Bug fixes need to be applied twice
- Unclear entry point for users

## Target State (After Refactoring)

```
gpt2/
├── train.py                    # Unified entry point (dispatcher)
├── trainers/
│   ├── __init__.py
│   ├── base.py                 # BaseGPT2Trainer (common functionality)
│   ├── vanilla.py              # VanillaGPT2Trainer (standard GPT-2)
│   ├── unified_ra.py           # UnifiedRATrainer (V-series)
│   └── ablation.py             # AblationCoordinator
├── model.py                    # GPT-2 model (unchanged)
├── ra.py                       # Unified RA architecture
├── ra_v5_patch.py              # Patching utilities
└── old/
    ├── train_ra_mla.py         # Legacy trainer (for defconfigs/old/)
    ├── ra_mla_gpt2.py          # Legacy RA implementation
    └── ra_lens_gpt2.py         # Deprecated lens architecture
```

## Migration Phases

### Phase 1: Structure Creation ✅ DONE

- [x] Create `gpt2/trainers/` module
- [x] Create `BaseGPT2Trainer` with common functionality
- [x] Create skeleton for `VanillaGPT2Trainer`
- [x] Create skeleton for `UnifiedRATrainer`
- [x] Create skeleton for `AblationCoordinator`

### Phase 2: Extract Common Code (TODO)

From `train.py` and `train_ra_mla.py`, extract to `BaseGPT2Trainer`:

- [ ] `get_batch()` - Data loading
- [ ] `get_lr()` - Learning rate scheduling
- [ ] `estimate_loss()` - Validation
- [ ] `setup_device()` - Device/dtype setup
- [ ] `setup_ddp()` - Distributed training
- [ ] `save_checkpoint()` - Checkpoint management
- [ ] `load_checkpoint()` - Checkpoint loading

### Phase 3: Implement VanillaGPT2Trainer (TODO)

Extract from `train.py`:

- [ ] Model initialization (standard GPT-2)
- [ ] Optimizer setup (AdamW, AdamWSPAM, AdamWPrune)
- [ ] Training loop
- [ ] Data preparation (shakespeare, finewebedu)
- [ ] Pruning evaluation logic

### Phase 4: Implement UnifiedRATrainer (TODO)

Extract V-series from `train_ra_mla.py`:

- [ ] V-series step configuration (V0-V19)
- [ ] Unified RA patching (via `ra.py`)
- [ ] R-MLP support
- [ ] KV pruning variants
- [ ] Gate analysis functions
- [ ] Delayed activation logic

### Phase 5: Create Unified train.py (TODO)

New `train.py` that dispatches to appropriate trainer:

```python
# Simplified example
if args.ablation_mode:
    from trainers import AblationCoordinator
    coordinator = AblationCoordinator(args, config, args.ablation_steps)
    coordinator.run()
elif args.architecture == 'unified-ra':
    from trainers import UnifiedRATrainer
    trainer = UnifiedRATrainer(args, config)
    trainer.train()
else:  # vanilla
    from trainers import VanillaGPT2Trainer
    trainer = VanillaGPT2Trainer(args, config)
    trainer.train()
```

### Phase 6: Legacy Support (TODO)

- [ ] Move `train_ra_mla.py` to `gpt2/old/`
- [ ] Update `defconfigs/old/*` to use `gpt2/old/train_ra_mla.py`
- [ ] Add deprecation notice to old trainer
- [ ] Ensure old experiments remain reproducible

### Phase 7: Testing (TODO)

- [ ] Test vanilla trainer (baseline GPT-2)
- [ ] Test Unified RA trainer (V0, V1, V3)
- [ ] Test ablation mode (full V-series)
- [ ] Test DDP training
- [ ] Test checkpoint save/load
- [ ] Run dry-run validation (`make check`)

## Usage After Refactoring

### Vanilla GPT-2 Training

```bash
make defconfig-gpt2-vanilla-baseline
make
# Or directly:
python gpt2/train.py --architecture vanilla
```

### Unified RA Single Run

```bash
make defconfig-gpt2-unified-ra-v1
make
# Or directly:
python gpt2/train.py --architecture unified-ra --ra-step V1
```

### Unified RA Ablation Study

```bash
make defconfig-gpt2-unified-ra-ablation
make
# Or directly:
python gpt2/train.py --architecture unified-ra --ablation-mode \
  --ablation-steps V0,V1,V3,V7,V9
```

### Legacy RA (for reproducibility)

```bash
make defconfig-old-gpt2-ratio-ablation
make  # Uses gpt2/old/train_ra_mla.py
```

## Benefits

1. **Code Reuse**: Common code in `BaseGPT2Trainer` (~70% reduction)
2. **Clarity**: Each trainer has a clear purpose
3. **Maintainability**: Bug fixes in one place
4. **Extensibility**: Easy to add new GPT-2 variants
5. **Testing**: Can test trainers independently
6. **Documentation**: Code organization matches concepts

## Backward Compatibility

- Old defconfigs in `defconfigs/old/` continue to work
- Legacy trainer in `gpt2/old/train_ra_mla.py` preserved
- New defconfigs use new modular trainers
- All experiments remain reproducible

## Timeline

This is a gradual refactoring. Each phase can be completed independently:

- **Phase 1-2**: ~1-2 days (structure + common code)
- **Phase 3**: ~1 day (vanilla trainer)
- **Phase 4**: ~2 days (Unified RA trainer)
- **Phase 5-6**: ~1 day (dispatcher + legacy)
- **Phase 7**: ~1 day (testing)

**Total**: ~6-8 days of focused work

## Current Status

- ✅ Phase 1 complete (structure created)
- 🔄 Phase 2-7 TODO (implementation)

## Next Steps

1. Extract common code to `BaseGPT2Trainer`
2. Implement `VanillaGPT2Trainer`
3. Implement `UnifiedRATrainer`
4. Create unified `train.py` dispatcher
5. Test thoroughly
6. Move old trainer to `gpt2/old/`
