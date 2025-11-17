# Experiment Tracking

AdamWPrune supports flexible experiment tracking with both Trackio (local) and Weights & Biases (cloud), which can run independently or simultaneously for comparison.

## Quick Start

### Command-Line Usage (Recommended)

```bash
# No tracking (default)
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4

# Enable only Trackio (local, no authentication needed)
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=trackio

# Enable only WandB (requires wandb login)
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=wandb

# Enable both for comparison
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=both
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=wandb,trackio
```

### Smart Project Names

Project names are auto-generated as `{model}-{checksum}` where:
- `model`: Selected model (e.g., `gpt2`, `resnet50`, `lenet5`)
- `checksum`: 5-character hash of hostname + IP + directory

Examples: `gpt2-a3f2c`, `resnet50-7b9d1`

This ensures:
- Consistent naming across runs
- No collisions between different machines/directories
- Readable, meaningful project names
- No manual configuration needed

## Installation

### Trackio (Lightweight, Local-First)
```bash
pip install trackio
```

**No login required!** Trackio stores everything locally in SQLite.

### WandB (Full-Featured, Cloud-Based)
```bash
pip install wandb
wandb login  # One-time authentication
```

## Configuration Options

### Kconfig Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_ENABLE_TRACKIO` | Enable Trackio tracking | `n` |
| `CONFIG_ENABLE_WANDB` | Enable WandB tracking | `n` |
| `CONFIG_TRACKER_PROJECT` | Project name (auto-generated) | `{model}-{checksum}` |
| `CONFIG_TRACKER_RUN_NAME` | Run name (optional) | `{model}_{optimizer}_{timestamp}` |
| `CONFIG_WANDB_ENTITY` | WandB team/username | `""` (uses default) |
| `CONFIG_WANDB_OFFLINE` | Run WandB offline | `n` |
| `CONFIG_TRACKIO_PORT` | Trackio dashboard port | `7860` |

### Menu Configuration (Optional)

For fine-tuned control, use `make menuconfig`:
1. Navigate to "Experiment Tracking"
2. Enable/disable individual trackers
3. Configure project names and settings

## Usage Examples

### Training with Trackers

#### Using Trackio Only
```bash
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam \
    --tracker trackio \
    --tracker-project my-experiments \
    --tracker-run-name experiment-1
```

After training, view results with:
```bash
trackio show
```

#### Using WandB Only
```bash
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam \
    --tracker wandb \
    --tracker-project my-experiments \
    --tracker-run-name experiment-1
```

For offline mode (no login required):
```bash
export WANDB_MODE=offline
python gpt2/train.py --tracker wandb ...
```

Later sync offline runs:
```bash
wandb sync
```

#### Using Both Trackers Simultaneously

```bash
# Configure for both
make defconfig DEFCONFIG=gpt2-finewebedu-a10gx4 TRACKER=both
make

# Or run directly
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam
# Both trackers will log to the same auto-generated project name
```

This allows you to:
- Compare local vs cloud tracking
- Have local backup while using cloud features
- Test both systems simultaneously

#### No Tracking (Default)
```bash
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam \
    --tracker none
```

## Tracked Metrics

Both trackers log the following metrics:
- **Training Loss**: Loss at each logging interval
- **Validation Loss**: Loss at each evaluation interval
- **Validation Perplexity**: Perplexity at each evaluation interval
- **Learning Rate**: Current learning rate
- **Sparsity**: Current model sparsity (for pruning experiments)
- **Final Metrics**: Best validation loss, total training time

### Validation Points: What Actually Happens

At each validation point (every `eval_interval` iterations), the training process:

1. **Pauses Training**: Model switches to eval mode (disables dropout, etc.)

2. **Runs Inference on Validation Data**:
   - Processes `eval_samples` random batches from held-out validation set
   - Default: 200 batches × 64 sequences × 1024 tokens = **13.1M validation tokens**
   - No gradient computation (forward pass only)

3. **Computes Validation Loss**:
   - Averages loss across all 200 validation batches
   - Example: `val_loss = 10.96`

4. **Calculates Perplexity**:
   ```python
   val_perplexity = exp(min(val_loss, 20))
   ```
   - Perplexity = e^(average_validation_loss)
   - Capped at e^20 to prevent overflow on very high losses
   - Example: `exp(10.96) = 57550.31`

5. **Logs to Trackers**:
   - `val_loss` and `val_perplexity` sent to W&B and Trackio
   - These are the data points you see in charts

6. **Saves Best Checkpoint**:
   - If current `val_loss` is best so far, saves model weights to disk

7. **Resumes Training**: Model switches back to training mode

**Important**: W&B requires **2+ data points** to render line charts. With only 1 validation
point (initial evaluation at iter 0), metrics appear as bar charts. Configure
`eval_interval` to ensure multiple validation points within training time budget.

## Testing Integration

### Test Individual Trackers

```bash
# Test WandB integration with fake data
make wandb-test

# Test Trackio integration with fake data
make trackio-test
```

These tests:
- Generate realistic fake training metrics
- Submit to respective tracking systems
- Verify configuration is working
- Don't require actual model training

## Comparison

| Feature | Trackio | WandB |
|---------|---------|-------|
| Installation Size | Lightweight | Heavy |
| Cloud Dependency | No (local SQLite) | Yes (cloud storage) |
| Dashboard | Local web UI | Cloud web UI |
| Team Collaboration | Limited | Full support |
| Cost | Free | Free tier + paid plans |
| Offline Mode | Always offline | Requires configuration |
| Advanced Features | Basic | Extensive (sweeps, reports, etc.) |

## Recommendations

- **Use Trackio** when:
  - You want simple, local experiment tracking
  - You're concerned about data privacy
  - You don't need cloud collaboration features
  - You want minimal dependencies

- **Use WandB** when:
  - You need team collaboration
  - You want advanced features (hyperparameter sweeps, reports)
  - You need cloud backup of experiments
  - You want integration with other ML tools

## Multi-GPU Monitoring Integration

### 4x A10G GPU Tracking

The experiment tracking system automatically integrates with multi-GPU monitoring for AWS g5.12xlarge instances:

```bash
# Enable both experiment tracking and GPU monitoring
make defconfig DEFCONFIG=gpt2/defconfigs/gpt2-finewebedu-a10gx4 TRACKER=wandb,trackio
make test_matrix GPU_MONITOR=y
```

### Multi-GPU Metrics Logged

**Aggregate Metrics:**
- `gpu_total_memory_used` - Combined memory across all 4 A10G GPUs
- `gpu_total_memory_percent` - Overall memory utilization
- `gpu_average_utilization` - Mean GPU utilization across devices
- `gpu_total_power` - Combined power consumption
- `gpu_max_temperature` - Highest temperature across all GPUs

**Load Balance Metrics:**
- `gpu_load_balance_cv` - Coefficient of variation (lower = better balance)
- `gpu_load_balance_quality` - Categorical rating (Excellent/Good/Fair/Poor)

**Per-GPU Metrics:**
- `gpu_{0,1,2,3}_memory_used` - Individual GPU memory usage
- `gpu_{0,1,2,3}_utilization` - Individual GPU utilization
- `gpu_{0,1,2,3}_temperature` - Individual GPU temperatures
- `gpu_{0,1,2,3}_power` - Individual GPU power consumption

### Generated Visualizations

Multi-GPU analysis automatically creates additional graphs:
- **Per-GPU Memory Breakdown** - 2x2 grid showing each A10G GPU
- **GPU Load Balance Analysis** - Distribution and balance metrics
- **Timeline Comparisons** - Multi-GPU memory usage over time

### Quality Thresholds

Load balance quality classification:
- **Excellent (CV < 5%)**: Very well balanced across GPUs
- **Good (CV 5-10%)**: Well balanced distribution
- **Fair (CV 10-20%)**: Acceptable but could improve
- **Poor (CV ≥ 20%)**: Imbalanced - investigate DDP configuration

### Best Practices for 4x A10G

1. **Monitor Load Balance**: Target CV < 5% for optimal performance
2. **Track Total Memory**: Sum across all GPUs for true memory usage
3. **Per-GPU Analysis**: Identify individual GPU bottlenecks
4. **Power Monitoring**: Track total power consumption (4x A10G can draw ~600W)
5. **Temperature Management**: Monitor max temperature across devices

For detailed multi-GPU monitoring documentation, see [Multi-GPU Monitoring Guide](multi-gpu-monitoring.md).

## For Developers

### Adding Tracker Support to Your Training Script

```python
# Check if trackers are enabled in config
import sys
sys.path.append('.')  # Add project root
from config import config

# Initialize trackers based on config
trackio_enabled = getattr(config, 'ENABLE_TRACKIO', False)
wandb_enabled = getattr(config, 'ENABLE_WANDB', False)

# Get auto-generated project name
project = getattr(config, 'TRACKER_PROJECT', 'my-project')

# Initialize Trackio
if trackio_enabled:
    import trackio
    trackio.init(project=project, config=training_config)

# Initialize WandB
if wandb_enabled:
    import wandb
    wandb_project = getattr(config, 'WANDB_PROJECT', project)
    wandb.init(project=wandb_project, config=training_config)

# Log metrics to both
def log_metrics(metrics):
    if trackio_enabled:
        trackio.log(metrics)
    if wandb_enabled:
        wandb.log(metrics)
```

### CLI Variable Support

The Kconfig system supports CLI variables:

```bash
# Scripts check for CLI variables
scripts/check-cli-set-var.sh TRACKER  # Returns 'y' if set
scripts/check-tracker-enabled.sh wandb $TRACKER  # Checks if wandb is in TRACKER
```

### Project Name Generation

The `scripts/generate-project-name.sh` script creates consistent project names:

```bash
./scripts/generate-project-name.sh gpt2 tracking
# Output: gpt2-a3f2c
```

Checksum is derived from:
- Hostname (short form)
- IP address (first interface)
- Current directory name
- MD5 hash (first 5 characters)

## Troubleshooting

### Trackio Issues
- If dashboard doesn't open: `trackio show --port 8080`
- Database location: `~/.trackio/trackio.db`

### WandB Issues
- Login issues: `wandb login` or use offline mode
- Sync offline runs: `wandb sync ./wandb/offline-run-*`
