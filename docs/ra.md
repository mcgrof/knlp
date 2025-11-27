# Double Attention: Evaluating Reciprocal Patterns

## Research Question

**What does computing attention twice per layer add to model quality?**

Standard Transformer attention computes `softmax(Q@K.T)@V` once per layer.
Double attention computes two attention operations per layer:

1. **Forward attention**: `softmax(Q@K.T)@V` (standard)
2. **Reciprocal attention**: `softmax(K@Q.T)@V` (roles swapped)

The outputs can be combined via addition, concatenation, gating, or other
mechanisms.

## Motivation

Original RA R&D explored double attention but the FLOPS cost was prohibitive.
This research evaluates the actual value of double attention by accepting the
2x FLOPS cost and scaling down model size to maintain fair comparison.

**Goal**: Understand if computing both Q@K.T and K@Q.T per layer provides
meaningful quality improvements that justify the computational cost.

## Cost-Quality Tradeoff

Computing attention twice doubles the FLOPs per attention layer. To maintain
fair comparisons, we scale down the model size:

| Configuration | Layers | d_model | n_heads | Attn Ops/Layer | Total Attn Ops |
|---------------|--------|---------|---------|----------------|----------------|
| GPT-2 124M Baseline | 12 | 768 | 12 | 1 | 12 |
| **GPT-2 Tiny Double** | 6 | 512 | 8 | 2 | 12 |

Both configurations perform approximately the same number of total attention
operations, allowing direct quality comparison.

## Experiment Design

### Models to Compare

1. **Baseline GPT-2 Tiny**: 6 layers, d_model=512, n_heads=8, single attention
2. **Double Attention Tiny**: 6 layers, d_model=512, n_heads=8, double attention

### Hyperparameters

```python
# Model architecture
n_layers = 6
d_model = 512
n_heads = 8
head_dim = 64  # d_model / n_heads
block_size = 1024
vocab_size = 50257

# Training
learning_rate = 6e-4
batch_size = 32  # Auto-tuned by GPU
gradient_accumulation = 8  # Auto-tuned
warmup_steps = 2000
max_iters = 10000
max_time = 14400  # 4 hours

# Dataset
dataset = "tinystories"
```

### Deterministic Seeding

All experiments use **fixed random seeds** for reproducibility:

```python
import torch
import numpy as np
import random

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may hurt performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Each configuration runs with **5 different seeds** (42, 43, 44, 45, 46) to
measure statistical significance.

### Metrics

**Training metrics**:
- Training loss curve
- Validation loss/perplexity
- Tokens/second throughput
- GPU memory usage

**Quality metrics**:
- Final validation perplexity
- LM-Eval benchmark scores (HellaSwag, ARC-Easy, Winogrande)

**Statistical analysis**:
- Mean and std dev across 5 seeds
- T-test for significance (p < 0.05)

## Double Attention Implementation

### Architecture

```python
class DoubleAttention(nn.Module):
    """
    Compute both forward (Q@K.T) and reciprocal (K@Q.T) attention.

    Standard: softmax(Q@K.T)@V
    Reciprocal: softmax(K@Q.T)@V
    Combined: forward + reciprocal (or other fusion)
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Single QKV projection (shared for both attention ops)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Forward attention: Q@K.T
        y_forward = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        # Reciprocal attention: K@Q.T
        y_reciprocal = F.scaled_dot_product_attention(
            k, q, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        # Combine (simple addition for now)
        y = y_forward + y_reciprocal

        # Merge heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)

        return y
```

### Combination Strategies

Future experiments can test different ways to combine forward and reciprocal:

1. **Addition**: `y = y_forward + y_reciprocal`
2. **Concatenation**: `y = concat(y_forward, y_reciprocal)` with projection
3. **Learned gating**: `y = alpha * y_forward + (1-alpha) * y_reciprocal`
4. **Attention over attention**: Use another attention to mix the two

Initial experiments use simple addition.

## Results

*Experiments pending*

## References

- MLA compression research: `docs/kvsplice.md`
- Fisher Information Matrix analysis: `docs/FIM.md`
