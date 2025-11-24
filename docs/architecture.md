# Kernel-Inspired Architecture

knlp adopts Linux kernel development practices to enable collaborative ML
research with rigorous validation and reproducible experiments.

## Configuration Management (Kconfig)

- **Hierarchical menus**: `make menuconfig` for interactive configuration
- **Defconfig presets**: Hardware-specific configurations (W7900, A10G, A100)
- **Dependency tracking**: Automatic validation of configuration combinations
- **Documentation integration**: Help text links to detailed docs

## Build System (Makefile)

- **Unified interface**: `make defconfig-<name>; make` for all experiments
- **Parallel execution**: Multi-GPU training with DDP support
- **CLI overrides**: `MODELS=path`, `BASELINE=run_id`, `TIME=3600`
- **Reproducible builds**: Deterministic configuration → experiment mapping

## Testing Infrastructure

- **Test matrices**: Automated cross-product testing (optimizer × pruning ×
  sparsity)
- **Dry-run validation**: Catch architecture bugs before GPU training
- **Continuation support**: Resume interrupted experiments (`make continue`)
- **Result archiving**: Automated preservation of key results

## Multi-Model Support

- **Extensible architecture**: LeNet-5, ResNet-18, ResNet-50, GPT-2
- **Vendor-agnostic monitoring**: [gputop.py](https://github.com/mcgrof/gputop)
  for NVIDIA/AMD/Intel
- **TrackIO integration**:
  [TrackIO](https://github.com/mcgrof/trackio/tree/20250921-trackio-view) for
  GPU utilization visualization
- **Multiple pruning methods**: Magnitude, movement, state-based (AdamWPrune)

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

If your test matrix is interrupted (system crash, power failure, etc.), you
can continue from where you left off:

```bash
# Continue the most recent interrupted test matrix
make continue
```

See [continue.md](continue.md) for detailed information on resuming
interrupted experiments.

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

## Design Philosophy

The kernel-inspired workflow enables collaborative ML research with the rigor
and reproducibility of systems programming. This methodology draws
inspiration from:

- **Linux kernel development**: Kconfig, defconfigs, Makefile patterns,
  rigorous testing
- **Andrej Karpathy's nanoGPT**: Clean implementation style, educational focus
- **Community contributors**: Ablation study ideas, architectural suggestions,
  validation testing
