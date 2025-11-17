# Hyperparameter Auto-Detection

## Overview

The AUTO hyperparameter system automatically selects optimal `batch_size` and `gradient_accumulation` based on GPU hardware, making defconfigs portable across different systems without manual tuning.

## How It Works

When `CONFIG_HYPER_PARAM_AUTO=y` is set, the training script:

1. **Detects GPU hardware** at runtime:
   - GPU type (H100, B200, W7900, A10G, etc.)
   - Total GPU memory
   - **Free GPU memory** (accounts for other processes)
   - GPU count (for multi-GPU setups)
   - torch.compile() support (auto-detected or manual)

2. **Selects batch_size** based on free memory:
   - Uses 80% of free memory as safety margin
   - Applies model-specific memory scaling factors
   - Checks torch.compile() status (compile allows larger batches)

3. **Computes gradient_accumulation**:
   - Calculates to reach target effective batch size
   - `effective_batch = batch_size × gradient_accumulation × num_gpus`

## Model-Specific Memory Scaling

Different models have different memory requirements per sample. The system applies scaling factors:

| Model | Scale Factor | Reason |
|-------|--------------|--------|
| GPT-2 | 1.0x | Baseline (large memory: attention matrices, activations) |
| ResNet-50 | 1.5x | Medium memory (conv activations, image batches) |
| ResNet-18 | 2.0x | Smaller than ResNet-50, can fit 2x larger batches |
| LeNet-5 | 4.0x | Tiny model on small images, can fit 4x larger batches |

This means on the same GPU, LeNet-5 will use 4x larger batches than GPT-2.

## GPU Memory Heuristics

Based on **free** GPU memory (with 80% safety margin):

| Free Memory | Batch Size (compile ON) | Batch Size (compile OFF) |
|-------------|-------------------------|--------------------------|
| 128GB+ | 256 | 128 |
| 64GB+ | 128 | 64 |
| 32GB+ | 32 | 16 |
| 16GB+ | 16 | 8 |
| 8GB+ | 8 | 4 |
| 4GB+ | 4 | 2 |
| <4GB | 2 | 1 |

These are then multiplied by the model's scale factor.

## CPU Mode

When no GPU is available, the system uses CPU-optimized batch sizes:

**Base CPU batch size:** 16 (for GPT-2 baseline)
**Scaled by model factor:**
- GPT-2: 16 × 1.0 = 16
- ResNet-50: 16 × 1.5 = 24
- ResNet-18: 16 × 2.0 = 32
- LeNet-5: 16 × 4.0 = 64

CPUs typically have much more RAM than GPU VRAM (32-256GB vs 16-80GB), so larger batch sizes are feasible. The model scale factor ensures tiny models like LeNet-5 can use appropriately large batches on CPU.

### Example: ResNet-18 on H100 (70GB Free)

1. H100 has ~70GB free memory → usable = 70 × 0.8 = 56GB
2. 56GB fits in "32GB+" tier → base batch_size = 32 (compile ON)
3. ResNet-18 scale factor = 2.0x → actual batch_size = 32 × 2 = 64
4. Target effective batch = 512
5. gradient_accumulation = 512 / 64 = 8
6. **Result: batch=64, grad_acc=8, effective=512**

## Configuration

### Enable AUTO Mode (Recommended)

In your defconfig:

```kconfig
# Hyperparameter auto-detection
CONFIG_HYPER_PARAM_AUTO=y
CONFIG_TARGET_EFFECTIVE_BATCH=512  # For ResNet/LeNet
CONFIG_COMPILE_AUTO=y              # Auto-detect torch.compile() support
```

### Manual Override

To manually control hyperparameters:

```kconfig
CONFIG_HYPER_PARAM_MANUAL=y
CONFIG_BATCH_SIZE=128
# Set gradient accumulation in model-specific config
```

## Target Effective Batch Size

Choose based on your model and memory constraints:

| Model Type | Recommended Target | Notes |
|------------|-------------------|-------|
| LeNet-5 | 512 | Small model, moderate target |
| ResNet-18 | 512 | Medium model, balanced |
| ResNet-50 | 256 | Larger model, conservative |
| GPT-2 124M | 1024 | Transformer baseline |
| GPT-2 Large | 512-1024 | Adjust based on context length |

## torch.compile() Auto-Detection

When `CONFIG_COMPILE_AUTO=y`, the system:

**Enables torch.compile() for:**
- NVIDIA GPUs (H100, B200, A100, A10G, etc.)
- Most AMD GPUs

**Disables torch.compile() for:**
- AMD W7900 (known ROCm issues)
- AMD MI210 (stability issues)
- CPU (not supported)

### Manual Override

```kconfig
CONFIG_COMPILE_MANUAL=y
CONFIG_COMPILE_MODEL=y  # Force enable
```

Or force disable:

```kconfig
CONFIG_COMPILE_MANUAL=y
CONFIG_COMPILE_MODEL=n  # Force disable
```

## Runtime Output

When training starts, you'll see:

```
Hyperparams: AUTO mode - GPU: NVIDIA H100 (80.0GB total, 72.3GB free) × 1,
model=resnet18, compile=ON → batch=64, grad_acc=8 (effective=512, target=512)
Compile: Enabled (GPU 'NVIDIA H100' has good torch.compile support)
```

This confirms:
- Detected GPU
- Selected batch_size and gradient_accumulation
- Actual vs target effective batch
- torch.compile() status

## Benefits

1. **Portable defconfigs**: Same config works on H100, W7900, A10G, or CPU
2. **Optimal utilization**: Automatically uses larger batches on bigger GPUs
3. **Fair comparisons**: Maintains constant effective batch across hardware
4. **No OOMs**: Conservative memory margins prevent out-of-memory errors
5. **GPU-aware**: Accounts for other processes using GPU memory

## Implementation

The auto-detection logic is in `lib/hyperparams.py`:
- `get_gpu_info()`: Queries GPU hardware
- `auto_detect_hyperparams()`: Computes batch_size and gradient_accumulation
- `auto_detect_compile()`: Determines torch.compile() support
- `apply_hyperparams()`: Applies to config object

Called in training scripts:
```python
from lib.hyperparams import apply_hyperparams
apply_hyperparams(config, verbose=True, model_type="resnet18")
```

## Troubleshooting

**OOM errors despite AUTO mode:**
- Reduce `CONFIG_TARGET_EFFECTIVE_BATCH`
- Check if other processes are using GPU memory
- Try `CONFIG_COMPILE_MANUAL=y` and `CONFIG_COMPILE_MODEL=n`

**Batch size too small:**
- Check free GPU memory (other processes may be using it)
- Verify the target effective batch isn't too low
- Consider MANUAL mode for explicit control

**torch.compile() issues:**
- Use `CONFIG_COMPILE_MANUAL=y` to disable
- Report GPU model for blacklist consideration
