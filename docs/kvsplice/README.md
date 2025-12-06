# KVSplice: Learned KV Cache Compression

## Overview

KVSplice achieves KV cache compression through learned low-rank projection
on top of MLA's latent representation. This compresses the MLA latent space
(256 dims) to smaller dimensions (128 dims at 50% compression) using
orthogonal linear transformations.

**Compression stack**:
- Standard GPT-2: 1536 dims (full K+V per head)
- MLA: 256 dims (6x compression via shared latent)
- MLA + KVSplice: 128 dims (12x total compression)

## Architecture

### Multi-head Latent Attention (MLA)

MLA from DeepSeek-V2/V3 achieves 6x KV cache compression by storing a shared
latent representation instead of full per-head K/V matrices:

```python
# Standard attention: Store full K, V for each head
k = w_k(x)  # [B, n_heads, T, d_head] - 768 dims total
v = w_v(x)  # [B, n_heads, T, d_head] - 768 dims total
cache = (k, v)  # 1536 dims cached

# MLA: Store compressed latent, generate K, V on-the-fly
latent = to_latent(x)  # [B, T, 256] - shared across heads
k, v = from_latent(latent)  # Expand to per-head K, V on demand
cache = latent  # 256 dims cached (6x compression)
```

### KVSplice Layer

KVSplice adds learned low-rank projection to further compress the MLA latent:

```python
class MLA_KVSplice(nn.Module):
    """Learned KV cache compression via low-rank projection."""

    def __init__(self, d_latent: int, compression_ratio: float = 0.5):
        super().__init__()
        d_compressed = int(d_latent * compression_ratio)

        # Low-rank projection with orthogonal initialization
        self.compress = nn.Linear(d_latent, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_latent, bias=False)

        nn.init.orthogonal_(self.compress.weight)
        with torch.no_grad():
            self.expand.weight.copy_(self.compress.weight.T)
```

The implementation uses latent-only SDPA (Scaled Dot-Product Attention) where
the compressed latent is stored and expanded on-demand during attention.

## B200x4 Results (FineWebEdu)

**Hardware**: 4x NVIDIA B200 (191.5GB total VRAM)

**Training configuration**:
- Dataset: FineWebEdu (web text)
- Model: GPT-2 124M
- Training time: 1 hour per variant
- Batch size: 76 per GPU
- Effective batch: 1216 (76 x 4 GPUs x 4 grad_acc)
- Optimizer: AdamWSPAM

### Quality Comparison

| Architecture | Val Loss | Val PPL | Iterations | Speed |
|--------------|----------|---------|------------|-------|
| MLA (6x) | **5.54** | **255** | 2350 | 367ms/iter |
| MLA+KVSplice (12x) | 5.69 | 296 | 3000 | 285ms/iter |

**Quality degradation**: +2.7% val_loss, +16% val_perplexity

### Training Speed

KVSplice trains **22% faster** than MLA alone (285ms vs 367ms per iteration)
due to smaller latent dimensions. In the same training time:
- MLA: 2350 iterations
- KVSplice: 3000 iterations (28% more)

Despite more iterations, KVSplice shows quality degradation, indicating the
compression bottleneck affects representational capacity. Longer training
runs may help close this gap.

### Memory Savings (Inference)

| Seq Length | Standard GPT-2 | MLA (6x) | KVSplice (12x) | Savings |
|------------|----------------|----------|----------------|---------|
| 256 tokens | 9 MB | 3 MB | 1.5 MB | 83% |
| 512 tokens | 18 MB | 6 MB | 3 MB | 83% |
| 1024 tokens | 36 MB | 12 MB | 6 MB | 83% |

For production inference with 24GB GPU memory:
- Standard GPT-2: ~650 parallel sequences
- MLA: ~1950 sequences (3x throughput)
- KVSplice: ~3900 sequences (6x throughput)

## Implementation

**Code**: `gpt2/mla.py`

**Key classes**:
- `MLA_KVSplice`: Learned compression layer
- `MLA_Flash`: Base MLA attention with flash attention
- `MLABlock`: MLA transformer block
- `MLAKVBlock`: MLA + KVSplice transformer block
- `GPT2_MLA`: GPT-2 with MLA (6x compression)
- `GPT2_MLA_KV`: GPT-2 with MLA + KVSplice (12x compression)

**Running the ablation**:
```bash
make defconfig-gpt2-kvsplice-ablation
make
```

**Configuration** (`defconfigs/gpt2-kvsplice-ablation`):
```
CONFIG_ENABLE_MLA=y
CONFIG_MLA_VARIANT="mla,mla_kv"
CONFIG_MLA_D_LATENT=256
CONFIG_MLA_COMPRESSION_RATIO="0.5"
```

## When to Use

**Use MLA+KVSplice when**:
- KV cache memory is the inference bottleneck
- High-concurrency serving (many parallel sequences)
- Memory-constrained deployment (8-24GB GPUs)
- Willing to accept ~16% perplexity degradation

**Use MLA only when**:
- Quality is critical (no degradation acceptable)
- Already have 6x compression benefit
- Sufficient GPU memory for target concurrency

**Use standard GPT-2 when**:
- Training efficiency matters (MLA trains slower)
- No inference deployment planned

## Future Work

- Longer training runs to assess convergence gap
- Higher compression ratios (70%, 90%)
- Compare with xKV and other compression methods
- Production inference latency benchmarks

## References

- DeepSeek-V2/V3 papers (original MLA architecture)
- xKV paper (low-rank KV compression validation)
- Implementation: `gpt2/mla.py`
