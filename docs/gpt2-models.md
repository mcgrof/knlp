# GPT-2 Model R&D Framework

A scalable architecture for rapidly prototyping and evaluating GPT-2 model variants with minimal friction.

## Overview

This framework enables efficient R&D iteration on GPT-2 architectural variants by using **convention-based auto-discovery**. Add a new model, and it automatically appears in benchmarks, comparisons, and training workflows without manual registry updates.

### Design Principles

1. **Convention over Configuration**: Follow naming patterns, get automatic integration
2. **Hierarchical Naming**: Model names reveal feature composition at a glance
3. **Zero Registration**: No manual MODEL_REGISTRY or import list updates
4. **Rapid Testing**: One command to compare inference speed across all variants
5. **Configuration Integration**: Kconfig system for reproducible experiments

## Model Naming Convention

### Hierarchical Pattern

Models follow a strict naming hierarchy showing incremental feature additions:

```
GPT2                    # Baseline GPT-2 (standard transformer)
GPT2_RA                 # + Reciprocal Attention
GPT2_MLA                # + Multi-head Latent Attention (DeepSeek)
GPT2_MLA_RA             # + MLA + RA combined
GPT2_MLA_KV             # + MLA + KV cache compression
GPT2_MLA_RA_KV          # + MLA + RA + KV compression
GPT2_MLA_KV2            # + MLA with 2-latent variant
GPT2_MLA_KV2M           # + KV2 + MLP compression
GPT2_MLA_RA_KVM         # + RA + MLA + KV + MLP compression
```

### Feature Abbreviations

| Abbreviation | Feature | Description |
|--------------|---------|-------------|
| `_RA` | Reciprocal Attention | Learned alternation between Q@K.T and K@Q.T |
| `_MLA` | Multi-head Latent Attention | DeepSeek-style cache compression via shared latent |
| `_KV` | KV Compression | KVSplice cache compression technique |
| `_KV2` | 2-Latent MLA | Separate latent spaces for K and V |
| `M` suffix | MLP Compression | MLPSplice layer compression |

### Why This Matters

Clear hierarchical naming enables:
- **Instant understanding**: `GPT2_MLA_RA_KV` tells you exactly what features are active
- **Easy comparison**: Compare `GPT2` ’ `GPT2_RA` ’ `GPT2_MLA_RA` to isolate feature impact
- **Scalable research**: Add new features without renaming existing models

## Adding a New Model

### The Five Conventions

Follow these conventions for automatic discovery and integration:

1. **Name starts with `GPT2`** (enforces naming pattern)
2. **Inherits from `nn.Module`** (standard PyTorch model)
3. **Implements `get_num_params()`** (parameter counting interface)
4. **Config parameter name** (`config: GPTConfig` or `cfg: RA_MLA_Config`)
5. **Defined in correct module** (`ra.py` for MLA-based, `gpt2/model.py` for baseline variants)

That's it. No registration, no imports, no manual updates.

### Example: Adding GPT2_MLA

Here's a complete example of adding Multi-head Latent Attention (DeepSeek-style) as a new model variant.

#### Step 1: Define the Model Class

Create the model in `ra.py`:

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from gpt2.model import GPTConfig, Block

@dataclass
class RA_MLA_Config:
    """Configuration for MLA-based models."""
    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64
    d_latent: int = 256        # Latent dimension for cache compression
    n_layers: int = 12
    vocab_size: int = 50304
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True


class MLA_Attention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2 style).

    Compresses KV cache by projecting to shared latent space (d_latent),
    then expanding per-head. Reduces memory from O(n_heads * head_dim)
    to O(d_latent) for cached states.
    """

    def __init__(self, cfg: RA_MLA_Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.d_latent = cfg.d_latent

        # Query: standard per-head projection
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.bias)

        # KV: compress to shared latent, then expand per-head
        self.kv_compress = nn.Linear(cfg.d_model, cfg.d_latent, bias=cfg.bias)
        self.k_expand = nn.Linear(cfg.d_latent, cfg.n_heads * cfg.head_dim, bias=cfg.bias)
        self.v_expand = nn.Linear(cfg.d_latent, cfg.n_heads * cfg.head_dim, bias=cfg.bias)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.bias)

        self.dropout = nn.Dropout(cfg.dropout)

        # Cache for inference
        self.register_buffer("kv_latent_cache", None, persistent=False)

    def forward(self, x, use_cache=False):
        B, T, C = x.shape

        # Compute Q
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        # Compress to latent space
        kv_latent = self.kv_compress(x)  # (B, T, d_latent) - CACHE THIS

        # Expand to per-head K and V
        k = self.k_expand(kv_latent)  # (B, T, n_heads * head_dim)
        v = self.v_expand(kv_latent)  # (B, T, n_heads * head_dim)

        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out, kv_latent  # Return latent for cache


class MLA_Block(nn.Module):
    """Transformer block with MLA attention."""

    def __init__(self, cfg: RA_MLA_Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        # Attention with residual
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x


class GPT2_MLA(nn.Module):
    """
    GPT-2 with Multi-head Latent Attention.

    Replaces standard attention with MLA for reduced KV cache memory.
    Cache size: O(n_layers * seq_len * d_latent) vs O(n_layers * n_heads * seq_len * head_dim)

    For typical config (12 layers, 12 heads, head_dim=64, d_latent=256):
    - Standard cache: 12 * 12 * T * 64 = 9,216 * T elements
    - MLA cache: 12 * T * 256 = 3,072 * T elements
    - Reduction: 66.7% smaller
    """

    def __init__(self, cfg: RA_MLA_Config):
        super().__init__()
        self.cfg = cfg

        # Token + position embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # Transformer blocks with MLA
        self.blocks = nn.ModuleList([MLA_Block(cfg) for _ in range(cfg.n_layers)])

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.pos_emb(pos)  # (T, d_model)

        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """Count parameters (exclude embeddings if requested)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params
```

#### Step 2: Test Auto-Discovery

The model is now automatically discoverable:

```bash
./scripts/compare_inference.py --list
```

Output:
```
Supported model architectures:
--------------------------------------------------------------------------------
   1. GPT2                      (config: gpt)
   2. GPT2_MLA                  (config: mla)   NEW MODEL
   3. GPT2_MLA_KV               (config: mla)
   ...
--------------------------------------------------------------------------------
```

#### Step 3: Add Kconfig Options

Add configuration options to `Kconfig`:

```kconfig
menu "MLA Configuration"
    depends on KNLP_RA_MLA

config KNLP_MLA_LATENT_DIM
    int "MLA latent dimension"
    default 256
    help
      Latent dimension for Multi-head Latent Attention cache compression.

      Smaller values reduce memory but may hurt quality:
      - 256: Standard (66% cache reduction for 12-head model)
      - 128: Aggressive (83% cache reduction)
      - 512: Conservative (33% cache reduction)

config KNLP_MLA_ENABLE_ROPE
    bool "Enable RoPE positional encoding in MLA"
    default y
    help
      Apply RoPE (Rotary Position Embedding) to K queries in MLA.
      Adds positional information to compressed cache.

endmenu
```

#### Step 4: Create Defconfig

Create `defconfigs/gpt2-mla` for reproducible experiments:

```kconfig
# GPT2-MLA: Multi-head Latent Attention baseline
# Tests cache compression with DeepSeek-style MLA

CONFIG_KNLP_MODEL="GPT2_MLA"

# Dataset
CONFIG_KNLP_DATASET_NAME="finewebedu"

# Model architecture
CONFIG_KNLP_RA_MLA=y
CONFIG_KNLP_MLA_LATENT_DIM=256
CONFIG_KNLP_MLA_ENABLE_ROPE=y

# Training hyperparameters
CONFIG_KNLP_OPTIMIZER="adamw"
CONFIG_KNLP_LEARNING_RATE="6e-4"
CONFIG_KNLP_BATCH_SIZE=16
CONFIG_KNLP_GRADIENT_ACCUMULATION=4
CONFIG_KNLP_MAX_ITERS=10000

# Compile and efficiency
CONFIG_KNLP_COMPILE_MODEL=y
CONFIG_KNLP_MIXED_PRECISION=y

# Tracking
CONFIG_KNLP_WANDB=y
CONFIG_TRACKER_PROJECT="gpt2-mla-baseline"
CONFIG_TRACKER_RUN_NAME="mla_latent256"
```

#### Step 5: Run Experiments

```bash
# Load configuration
make defconfig-gpt2-mla

# Train the model
make

# Compare inference speed vs baseline
./scripts/compare_inference.py --test-models="GPT2,GPT2_MLA" --num-tokens 100
```

### Example Output

```
================================================================================
INFERENCE SPEED COMPARISON
================================================================================

Model Type                Parameters      Throughput           vs Baseline
--------------------------------------------------------------------------------
GPT2                       123.69M            248.0 tok/s     (baseline)
GPT2_MLA                   117.36M            187.3 tok/s     1.32x slower
================================================================================

Cache Memory Usage (seq_len=1024):
  GPT2:       442.37 MB  (full KV cache)
  GPT2_MLA:   147.46 MB  (66.7% reduction)
```

## Testing Models with compare_inference.py

### Quick Start

```bash
# List all available models
./scripts/compare_inference.py --list

# Test single model with random weights
./scripts/compare_inference.py --test-models="GPT2_MLA" --num-tokens 100

# Compare multiple models
./scripts/compare_inference.py --test-models="GPT2,GPT2_MLA,GPT2_MLA_RA" \
    --num-tokens 100 --num-runs 3

# Load from checkpoint
./scripts/compare_inference.py path/to/checkpoint.pt --num-tokens 100
```

### Common Workflows

#### 1. Sanity Check New Model

After adding a new model, verify it works:

```bash
# Quick inference test (small tokens, single run)
./scripts/compare_inference.py --test-models="GPT2_MLA" \
    --num-tokens 10 --num-runs 1
```

Expected output:
- Model instantiates without errors
- Forward pass completes
- Token generation works

#### 2. Measure Inference Speed

Compare throughput across variants:

```bash
# Longer sequence, multiple runs for stability
./scripts/compare_inference.py \
    --test-models="GPT2,GPT2_MLA,GPT2_MLA_RA" \
    --num-tokens 100 \
    --num-runs 5
```

Metrics provided:
- Tokens per second (tok/s)
- Slowdown vs baseline
- Parameter count

#### 3. Checkpoint Comparison

Compare trained models:

```bash
./scripts/compare_inference.py \
    results/gpt2_baseline/final.pt \
    results/gpt2_mla/final.pt
```

Shows:
- Inference speed differences
- Memory usage patterns
- Whether training affected speed (it shouldn't - speed is architectural)

#### 4. Memory Profiling

For detailed cache size measurements:

```bash
./scripts/measure_kv_cache.py
```

Tests all architectures at multiple sequence lengths and reports:
- Theoretical cache size
- Measured GPU memory usage
- Reduction vs baseline

### Understanding Results

#### Speed vs Memory Tradeoff

```
Model           Speed       Cache Size    Use Case
GPT2            248 tok/s   442 MB       Baseline reference
GPT2_MLA        187 tok/s   147 MB       Long context generation
GPT2_MLA_KV     156 tok/s    98 MB       Extreme memory constraint
```

**Interpretation**:
- MLA variants trade speed for memory (extra projections)
- Choose based on bottleneck: compute-bound ’ use GPT2, memory-bound ’ use MLA
- Slowdown is architectural overhead, not implementation issue

#### Parameter Count Differences

MLA models have fewer parameters due to latent compression:

```python
# Standard attention per layer:
Q: d_model ’ n_heads * head_dim    # 768 ’ 768
K: d_model ’ n_heads * head_dim    # 768 ’ 768
V: d_model ’ n_heads * head_dim    # 768 ’ 768
# Total: 3 * (768 * 768) = 1,769,472 params

# MLA attention per layer:
Q: d_model ’ n_heads * head_dim    # 768 ’ 768
KV_compress: d_model ’ d_latent    # 768 ’ 256
K_expand: d_latent ’ n_heads * head_dim  # 256 ’ 768
V_expand: d_latent ’ n_heads * head_dim  # 256 ’ 768
# Total: 768*768 + 768*256 + 2*256*768 = 1,179,648 params
# Reduction: 33% fewer parameters
```

## Configuration System Integration

### Kconfig Hierarchy

```
KNLP_GPT2                    # Enable GPT-2 training
   KNLP_RA                  # Enable Reciprocal Attention
      KNLP_RA_RECIPROCAL_RATIO
      KNLP_RA_LEARNABLE
   KNLP_RA_MLA              # Enable MLA variants
       KNLP_MLA_LATENT_DIM
       KNLP_MLA_ENABLE_ROPE
       KNLP_MLA_KV_COMPRESSION
           KNLP_KV_COMPRESSION_RATIO
```

### Creating Defconfigs

Defconfigs provide one-command experiment reproducibility:

```bash
# Create new defconfig
cat > defconfigs/gpt2-mla-ablation <<'EOF'
# MLA ablation study: test latent dimensions

CONFIG_TEST_MATRIX_MODE=y
CONFIG_RA_MLA_ABLATION_MODE=y
CONFIG_RA_MLA_ABLATION_STEPS="B MLA RAMLA RAMLAKV"

CONFIG_KNLP_DATASET_NAME="finewebedu"
CONFIG_KNLP_MAX_TIME="2h"
CONFIG_KNLP_BATCH_SIZE=16

CONFIG_KNLP_WANDB=y
CONFIG_TRACKER_PROJECT="mla-ablation"
EOF

# Use it
make defconfig-gpt2-mla-ablation
make
```

### Defconfig Best Practices

1. **Name clearly**: `gpt2-{feature}-{purpose}`
   - `gpt2-mla-baseline`: Single MLA training run
   - `gpt2-mla-ablation`: Test multiple MLA variants
   - `gpt2-ra-profile`: Profile RA overhead

2. **Document purpose**: Add header comments explaining the experiment

3. **Set tracking**: Always configure W&B project/run names for reproducibility

4. **Use time limits**: Prefer `CONFIG_KNLP_MAX_TIME` over `CONFIG_KNLP_MAX_ITERS` for fair comparisons

## Advanced Topics

### Adding Hybrid Models

Combine multiple features by composing mechanisms:

```python
class GPT2_MLA_RA(nn.Module):
    """MLA attention + RA alternation."""

    def __init__(self, cfg: RA_MLA_Config):
        super().__init__()
        # Use MLA_Attention but wrap with RA alternation logic
        self.blocks = nn.ModuleList([
            MLA_RA_Block(cfg) for _ in range(cfg.n_layers)
        ])
```

Follow the naming convention: features in alphabetical order after base (MLA before RA).

### Custom Config Classes

For models needing different hyperparameters:

```python
@dataclass
class CustomConfig:
    """Config for experimental models."""
    d_model: int = 768
    custom_param: float = 0.5  # New hyperparameter
    ...

class GPT2_Experimental(nn.Module):
    def __init__(self, cfg: CustomConfig):  # Custom config type
        ...
```

Auto-discovery will detect this via signature inspection.

### Ablation Studies

Test multiple variants systematically:

```python
# In gpt2/trainers/ramla.py, add to parse_step():
arch_map = {
    "NEWARCH": "new_architecture",
    ...
}
```

Then configure ablation:
```kconfig
CONFIG_RA_MLA_ABLATION_STEPS="B NEWARCH MLA"
```

Run matrix:
```bash
make defconfig-gpt2-ablation
make  # Tests all steps automatically
```

## Troubleshooting

### Model Not Discovered

**Symptom**: New model doesn't appear in `--list`

**Checklist**:
- [ ] Name starts with `GPT2`?
- [ ] Inherits from `nn.Module`?
- [ ] Has `get_num_params()` method?
- [ ] First parameter is `config` or `cfg`?
- [ ] Defined in `ra.py` or `gpt2/model.py`?

**Debug**:
```python
# Add to discover_gpt2_models() for diagnostics
print(f"Checking {name}: {obj}")
```

### Inference Test Fails

**Symptom**: Dimension mismatch or shape errors

**Common causes**:
1. **Embedding size mismatch**: Ensure `d_model` matches across components
2. **Head dimension**: Verify `n_heads * head_dim == d_model`
3. **Vocab size**: Use multiple of 64 for efficiency (e.g., 50304 not 50257)

**Debug**:
```bash
# Minimal test
./scripts/compare_inference.py --test-models="GPT2_MLA" \
    --num-tokens 5 --num-runs 1
```

### Kconfig Not Loading

**Symptom**: Settings don't apply after `make defconfig-X`

**Check**:
```bash
# View loaded config
cat .config | grep KNLP_MLA

# Verify syntax (no spaces around =)
grep " = " defconfigs/gpt2-mla  # Should be empty

# Regenerate
make clean
make defconfig-gpt2-mla
python3 scripts/config.py  # Check config.py output
```

## References

### Key Files

- `gpt2/model.py`: Baseline GPT-2 implementation
- `ra.py`: All RA and MLA model variants
- `scripts/compare_inference.py`: Automated benchmarking
- `scripts/measure_kv_cache.py`: Memory profiling
- `Kconfig`: Configuration options
- `defconfigs/`: Experiment presets

### Related Documentation

- [CLAUDE.md](../CLAUDE.md): Development guidelines and conventions
- [docs/tracker.md](tracker.md): W&B integration and experiment tracking
- [docs/gpt2.md](gpt2.md): GPT-2 experiments with AdamWPrune

### Papers

- **Reciprocal Attention**: Novel technique (internal research)
- **Multi-head Latent Attention**: [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- **GPT-2**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

**Ready to scale**: Add models, run experiments, iterate fast.
