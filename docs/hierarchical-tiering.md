# Hierarchical Memory Tiering

Hierarchical memory tiering places model weights across different memory tiers (GPU HBM, CPU RAM, SSD) based on usage patterns inferred from Adam optimizer states.

## Motivation

Modern AI accelerators face memory bottlenecks. Hierarchical memory systems (CXL, fabric-attached memory) can expand capacity, but we need strategies to decide what stays in fast memory vs what gets offloaded.

This infrastructure allows testing tier placement strategies without specialized hardware by:
1. **Emulated mode**: Inject realistic delays based on tier specs (for algorithm development)
2. **Real mode**: Actually offload weights to CPU/disk (for memory reduction validation)

## How It Works

### 1. Adam State Analysis

The analyzer examines Adam optimizer states (momentum `exp_avg`, variance `exp_avg_sq`) to infer weight update frequency:

- **High momentum/variance** → frequently updated → keep in HBM
- **Low momentum/variance** → stable weights → offload to CPU/SSD

### 2. Tier Assignment

Modules are scored by average Adam state magnitude and assigned to tiers:
- Top 30%: HBM (fastest, ~800 GB/s)
- Next 50%: CPU RAM (medium, ~150 GB/s over PCIe)
- Bottom 20%: SSD (slowest, ~10 GB/s NVMe)

### 3. Enforcement

**Emulated mode**: Forward hooks inject realistic latency based on tier and tensor size, but weights stay on GPU.

**Real mode**: Pre-forward hooks move weights from offload tier to GPU, post-forward hooks evict them back.

## Configuration

Enable via Kconfig:

```bash
make menuconfig
# Navigate to: Hierarchical Memory Tiering (Experimental)
#   [*] Enable hierarchical memory tiering
#     Tiering strategy: Adam state-based tiering
#     Tiering mode: Emulated tiering (fake delays) or Real offloading
#   [*] Generate tier hints JSON after training
#       Tier hints JSON output path: tier_hints.json
#   HBM tier threshold: 0.3
#   CPU tier threshold: 0.5
#   [ ] Run inference benchmark after training
```

Or set directly in defconfig:

```
CONFIG_ENABLE_HIERARCHICAL_TIERING=y
CONFIG_TIERING_ADAM_STATE=y
CONFIG_TIERING_EMULATED=y  # or CONFIG_TIERING_REAL_OFFLOAD=y
CONFIG_TIERING_GENERATE_JSON=y
CONFIG_TIERING_JSON_OUTPUT="tier_hints.json"
CONFIG_TIERING_HBM_THRESHOLD="0.3"
CONFIG_TIERING_CPU_THRESHOLD="0.5"
```

## Workflow

### Step 1: Train and Generate Tier Hints

```bash
make defconfig-gpt2-vanilla-baseline
# Edit .config to enable tiering
make menuconfig  # Enable hierarchical tiering, set thresholds
make

# After training, tier_hints.json is generated:
# {
#   "transformer.h.0.attn": "HBM",
#   "transformer.h.0.mlp": "HBM",
#   "transformer.h.5.attn": "CPU",
#   ...
# }
```

### Step 2: Benchmark Inference Impact

```bash
# Baseline (no tiering)
python3 scripts/benchmark_tiered_inference.py \
  --model openai-community/gpt2 \
  --checkpoint checkpoints/model.pt \
  --tier-hints tier_hints.json \
  --mode emulated \
  --baseline \
  --batch-size 1 \
  --seq-length 128 \
  --num-iterations 100

# Output:
# Baseline Results:
#   Mean latency: 12.45 ms
#   Throughput: 10285 tokens/s
#
# Tiered Results:
#   Mean latency: 13.21 ms
#   Throughput: 9694 tokens/s
#
# Impact:
#   Latency overhead: +6.1%
#   Throughput degradation: -5.7%
```

### Step 3: Adjust Thresholds

If impact is too high, adjust thresholds to keep more in HBM:

```
CONFIG_TIERING_HBM_THRESHOLD="0.5"  # 50% in HBM instead of 30%
CONFIG_TIERING_CPU_THRESHOLD="0.7"  # 70% in CPU+HBM, 30% in SSD
```

Retrain and re-benchmark.

## Tier Specifications

Current values model realistic hardware:

| Tier | Setup (μs) | Bandwidth (GB/s) | Example Hardware |
|------|------------|------------------|------------------|
| HBM  | 1          | 800              | A100/H100/W7900  |
| CPU  | 5          | 150              | DDR5 + PCIe 4.0  |
| SSD  | 30         | 10               | NVMe Gen4        |

These can be adjusted in `lib/tiering.py` for different systems.

## Real Offloading

Real offloading actually moves weights to CPU/disk to reduce GPU memory usage:

```
CONFIG_TIERING_REAL_OFFLOAD=y
```

**Benefits**:
- Reduces GPU memory consumption
- Enables larger models on smaller GPUs

**Tradeoffs**:
- Real transfer latency (not just emulated)
- May impact throughput if too aggressive
- Requires careful threshold tuning

**When to use**:
- Memory-constrained scenarios
- Validating that tiering actually reduces memory
- Comparing emulated vs real latency

## Use Cases

### 1. Algorithm Development (Emulated)

Test different tier assignment strategies without hardware:

```python
from lib.tiering import AdamStateTierAnalyzer

analyzer = AdamStateTierAnalyzer(hbm_threshold=0.4, cpu_threshold=0.6)
tier_assignments = analyzer.analyze_optimizer_states(optimizer, model)
```

### 2. Memory Reduction (Real)

Fit larger models on smaller GPUs:

```
CONFIG_TIERING_REAL_OFFLOAD=y
CONFIG_TIERING_HBM_THRESHOLD="0.2"  # Keep only 20% on GPU
```

### 3. Hardware Evaluation (Emulated)

Predict performance on future hardware (CXL, fabric-attached memory):

```python
# Modify lib/tiering.py:
TIER_CXL = TierSpec(name="CXL", setup_us=2.0, bandwidth_gb_s=400.0)
```

## Implementation Details

### Emulated Tiering Hook

```python
def _tier_latency_hook(self, module, inputs, output):
    tier_name = getattr(module, "_tier_name", None)
    if tier_name == "HBM":
        return  # No delay

    tier_spec = self.tier_specs[tier_name]
    n_bytes = output.numel() * output.element_size()
    delay_s = tier_spec.latency_for_bytes(n_bytes)
    time.sleep(delay_s)  # Block CPU to simulate wall-clock impact
```

### Real Offloading Hook

```python
def pre_forward_hook(mod, inputs):
    mod.to("cuda")  # Move to GPU before forward

def post_forward_hook(mod, inputs, output):
    mod.to("cpu")  # Evict to CPU after forward
```

## Future Enhancements

- **Gradient-aware tiering**: Also consider gradient magnitudes
- **Dynamic tiering**: Adjust tiers during training based on observed access patterns
- **Multi-GPU tiering**: Shard across GPUs + CPU + SSD
- **Activation offloading**: Also tier activations, not just weights
- **Learned tier assignment**: Use RL to find optimal placement

## References

- HuggingFace Accelerate: https://huggingface.co/blog/accelerate-large-models
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
