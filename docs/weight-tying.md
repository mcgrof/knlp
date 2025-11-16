# Weight Tying in Neural Networks

**Parameter Reduction Through Strategic Sharing**

This document explains weight tying as a general parameter reduction technique, covering multiple forms of weight sharing including input-output embedding tying, KV attention tying, and connections to reciprocal architectures.

---

## Table of Contents

1. [Overview](#overview)
2. [The Core Principle](#the-core-principle)
3. [Input-Output Embedding Tying](#input-output-embedding-tying)
4. [KV Attention Tying](#kv-attention-tying)
5. [Weight Tying in Reciprocal Architectures](#weight-tying-in-reciprocal-architectures)
6. [When to Use Weight Tying](#when-to-use-weight-tying)
7. [Implementation Patterns](#implementation-patterns)
8. [References](#references)

---

## Overview

**Weight tying** is a parameter reduction technique that shares the same weight tensor between different layers or projections in a neural network. By strategically identifying components that can use identical transformations, we reduce model size without sacrificing representational capacity.

**Key benefits:**
- ✅ **Fewer parameters**: Reduces memory footprint and training cost
- ✅ **Regularization**: Shared weights constrain the optimization space
- ✅ **Inductive bias**: Encourages symmetric or reciprocal relationships
- ✅ **Efficient inference**: Smaller models load faster and use less GPU memory

**Common forms:**
1. **Input-Output Embedding Tying**: Share `wte` (input embedding) with `lm_head` (output projection)
2. **KV Attention Tying**: Use value projection for both keys and values (`K = V`)
3. **Cross-Layer Tying**: Share projections across transformer blocks
4. **Reciprocal Tying**: Share weights between reciprocal pathways (see [RA documentation](ra.md))

---

## The Core Principle

### What is Weight Tying?

Weight tying means **multiple operations use the exact same parameter tensor**, not just copies with the same initial values.

**Without tying** (separate parameters):
```python
class SeparateParams(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Total params: vocab_size × d_model × 2

    def forward(self, x):
        embedded = self.input_embed(x)
        # ... transformer blocks ...
        logits = self.output_proj(hidden_states)
        return logits
```

**With tying** (shared parameters):
```python
class TiedParams(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie the weights: same tensor, not a copy!
        self.output_proj.weight = self.input_embed.weight
        # Total params: vocab_size × d_model × 1  (50% reduction!)

    def forward(self, x):
        embedded = self.input_embed(x)
        # ... transformer blocks ...
        logits = self.output_proj(hidden_states)
        return logits
```

**Critical difference:**
```python
# After tying:
assert self.output_proj.weight is self.input_embed.weight  # Same object!
assert id(self.output_proj.weight) == id(self.input_embed.weight)

# Gradient updates affect BOTH uses:
self.output_proj.weight.grad  # Accumulates gradients from both paths
```

### Energy Conservation Through Tying

Weight tying creates **implicit constraints** on the optimization:

```python
# Forward pass with tied weights W:
input_embedding = W @ token_ids      # Use 1: input embedding
output_logits = hidden @ W^T         # Use 2: output projection (transposed)

# Backward pass:
W.grad = grad_from_input + grad_from_output  # Gradients combined!
```

This coupling means:
- **Input path gradients** affect how outputs are projected
- **Output path gradients** affect how inputs are embedded
- Model must find weights that work well for **both** tasks

---

## Input-Output Embedding Tying

### The Classic Language Modeling Tie

**Used in:** GPT-2, GPT-3, BERT, most transformer language models

**Hypothesis**: The mapping from token IDs to embeddings (`wte`) should be similar to the mapping from hidden states back to token logits (`lm_head`).

### Mathematical View

```python
# Input embedding (token → vector):
e = wte[token_id]  # Lookup: vocab_size × d_model

# Output projection (vector → logits):
logits = hidden @ lm_head^T  # MatMul: (d_model) @ (vocab_size × d_model)^T
```

When tied: `lm_head.weight = wte.weight`

```python
logits = hidden @ wte^T  # Hidden state compared against ALL token embeddings
logits[i] = hidden · wte[i]  # Dot product = similarity to token i's embedding
```

**Interpretation**: Output logits measure **cosine similarity** (scaled) between the hidden state and each token's embedding.

### Implementation in GPT-2

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50304  # Padded to multiple of 64
    n_embd: int = 768        # Embedding dimension
    # ... other config ...

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Input embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position embedding
            # ... attention blocks ...
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights: lm_head uses wte's weight matrix
        self.transformer.wte.weight = self.lm_head.weight

        # Parameter count:
        # WITHOUT tying: vocab_size × n_embd × 2 = 50304 × 768 × 2 = 77M params
        # WITH tying:    vocab_size × n_embd × 1 = 50304 × 768     = 38.5M params
        # Reduction: 50%!
```

### Gradient Flow with Tying

```python
# Forward pass:
tok_emb = self.transformer.wte(idx)         # Use wte for input
# ... 12 transformer blocks ...
logits = self.lm_head(hidden_states)        # Use wte^T for output

# Backward pass:
# Gradient from lm_head:
grad_output = dL/d(lm_head.weight)  # Shape: (vocab_size, n_embd)

# Gradient from wte:
grad_input = dL/d(wte.weight)       # Shape: (vocab_size, n_embd)

# Since they're tied:
wte.weight.grad = grad_input + grad_output  # Combined gradient!
```

**Effect**: The embedding matrix is updated to work well for both **encoding inputs** and **decoding outputs**.

### Parameter Count Comparison

For GPT-2 124M:

| Component | Without Tying | With Tying | Reduction |
|-----------|--------------|------------|-----------|
| Input embedding (wte) | 38.6M | 38.6M | - |
| Output projection (lm_head) | 38.6M | **0M** (shared) | 100% |
| **Total embedding params** | **77.2M** | **38.6M** | **50%** |
| Attention + MLP params | 85M | 85M | - |
| **Total model params** | **162M** | **124M** | **23%** |

Weight tying accounts for almost **all** the parameter reduction from 162M → 124M in "GPT-2 124M"!

---

## KV Attention Tying

### Simplified Attention with K = V

**Used in:** Experimental architectures, memory-efficient transformers

**Hypothesis**: Keys and values in self-attention can share the same projection, reducing parameters by ~33% in the attention mechanism.

For detailed explanation of KV tying implementation and ablation results, see **[KV Tying Documentation](kv-tying.md)**.

### Quick Overview

**Standard attention** (3 separate projections):
```python
Q = x @ W_q  # Query projection:  n_embd → n_embd
K = x @ W_k  # Key projection:    n_embd → n_embd
V = x @ W_v  # Value projection:  n_embd → n_embd

attn_weights = softmax(Q @ K^T / sqrt(d_k))
output = attn_weights @ V
```

**KV-tied attention** (2 projections, K = V):
```python
Q = x @ W_q  # Query projection:  n_embd → n_embd
V = x @ W_v  # Value projection:  n_embd → n_embd
K = V        # Tie K to V (no separate projection!)

attn_weights = softmax(Q @ V^T / sqrt(d_k))
output = attn_weights @ V
```

**Parameter reduction:**
```
Standard: 3 × (n_embd × n_embd) parameters
KV-tied:  2 × (n_embd × n_embd) parameters
Reduction: 33% of QKV parameters
```

**Trade-off**: Saves parameters but constrains the attention mechanism. See [kv-tying.md](kv-tying.md) for empirical evaluation.

---

## Weight Tying in Reciprocal Architectures

### Connection to Reciprocal Attention (RA)

Weight tying appears in a subtle form in [Reciprocal Attention](ra.md), where reciprocal components can optionally share weights with standard components.

**R-MLP weight tying** (experimental, tested in ablation V2):
```python
class ReciprocalMLP(nn.Module):
    def __init__(self, config, use_weight_tying=False):
        self.c_fc_std = nn.Linear(n_embd, d_ff_std, bias=False)  # Standard pathway

        if use_weight_tying:
            # Tie reciprocal pathway to attention output projection
            # Reuses learned representations from attention
            self.c_fc_rec = attention_module.c_proj  # Shared weights!
        else:
            # Independent reciprocal pathway
            self.c_fc_rec = nn.Linear(n_embd, r_ff, bias=False)
```

**Research question** (V2 vs V5 ablation):
- **V2**: R-MLP with weight tying enabled (ties reciprocal MLP to attention projection)
- **V5**: R-MLP with independent weights (no tying)
- **Goal**: Determine if explicit coupling helps or if independent learning is better

For full details on R-MLP ablation study, see **[RA Documentation](ra.md)**.

### Geometric Initialization vs Weight Tying

RA uses **geometric gate initialization** instead of heavy reliance on weight tying:

```python
# Instead of tying weights, RA scales contributions geometrically:
w_std_init = D_std / D  # e.g., 60/64 = 0.9375 for standard pathway
w_rec_init = R / D      # e.g., 4/64  = 0.0625 for reciprocal pathway

# This eliminates need for warmup delays and enables stable training from step 0
```

This is a **different philosophy** than weight tying:
- **Weight tying**: Forces parameters to be identical
- **Geometric init**: Scales contributions based on dimensional capacity

See [RA geometric initialization](ra.md#the-solution-geometric-initialization) for details.

---

## When to Use Weight Tying

### ✅ Good Use Cases

**1. Input-Output Symmetry**
- Language modeling (input tokens ↔ output tokens)
- Autoencoding (compress ↔ decompress)
- Embedding inversion (encode ↔ decode)

**2. Memory-Constrained Environments**
- Edge deployment (limited GPU/NPU memory)
- Large vocabulary models (embeddings dominate parameter count)
- Multi-task models (share representations across tasks)

**3. Regularization Through Constraints**
- Prevent overfitting on small datasets
- Encourage reciprocal symmetry
- Reduce effective parameter count for generalization

### ❌ When to Avoid Weight Tying

**1. Asymmetric Tasks**
```python
# Bad: Encoder-decoder with different purposes
encoder_embed = nn.Embedding(src_vocab, d_model)  # Source language
decoder_embed = nn.Embedding(tgt_vocab, d_model)  # Target language
# Don't tie these! Different vocabularies, different purposes
```

**2. Sufficient Parameter Budget**
- Large models with billions of parameters (tying saves <1%)
- Research prototypes where expressiveness > efficiency
- Domains where overfitting isn't a concern

**3. Empirically Harmful**
- Always validate with ablation studies!
- Some architectures perform worse with tying (see [kv-tying.md](kv-tying.md) ablation results)
- If tying degrades quality, the regularization may be too strong

---

## Implementation Patterns

### Pattern 1: Direct Weight Assignment

**Most common** - used in PyTorch standard libraries:

```python
class ModelWithTying(nn.Module):
    def __init__(self, config):
        self.input_layer = nn.Embedding(vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        # Direct assignment: output_layer.weight points to input_layer.weight
        self.output_layer.weight = self.input_layer.weight

        # Now they share the same tensor:
        assert self.output_layer.weight is self.input_layer.weight  # True!
```

**Important**: This must happen **after** both layers are created, **before** optimizer initialization.

### Pattern 2: Conditional Tying via Config

**Flexible** - allows enabling/disabling via configuration:

```python
@dataclass
class Config:
    use_weight_tying: bool = True
    vocab_size: int = 50000
    d_model: int = 768

class ConfigurableModel(nn.Module):
    def __init__(self, config):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.use_weight_tying:
            self.lm_head.weight = self.wte.weight
            print("Weight tying enabled: lm_head shares wte weights")
        else:
            print("Weight tying disabled: lm_head has independent weights")
```

This pattern enables A/B testing: train both configurations and compare.

### Pattern 3: Optimizer-Aware Parameter Filtering

**Critical** - prevent optimizer from tracking tied parameters twice:

```python
def configure_optimizers(model, weight_decay):
    # Separate params into decay and no_decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Skip lm_head.weight if it's tied (already counted as wte.weight)
            if name == 'lm_head.weight' and model.config.use_weight_tying:
                # When tied, lm_head.weight IS wte.weight (same object)
                # Don't add it again!
                continue

            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ])
    return optimizer
```

**Why this matters:**
```python
# Without filtering:
len(list(model.parameters()))  # Counts tied weight TWICE
sum(p.numel() for p in model.parameters())  # Inflated parameter count!

# With filtering:
len([p for n, p in model.named_parameters() if ...])  # Counts each weight ONCE
```

### Pattern 4: Transposed Usage

**For embeddings** - output projection uses transpose:

```python
class TransposedTying(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.embed.weight

    def forward(self, token_ids, hidden_states):
        # Input path: lookup embedding
        embedded = self.embed(token_ids)  # (B, T, d_model)
        # Uses: embed.weight[token_ids]

        # ... transformer blocks ...

        # Output path: project to logits
        logits = self.proj(hidden_states)  # (B, T, vocab_size)
        # Uses: hidden_states @ embed.weight^T

        return logits
```

**Key insight**: Embedding lookup and linear projection are **transposes** of each other when weight-tied:
```python
embed(token):  weight[token]        # Row selection
proj(hidden):  hidden @ weight^T    # Matrix multiply with transpose
```

---

## Implementation in Our Codebase

### GPT-2 Input-Output Tying

**File**: `gpt2/model.py`

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: lm_head shares wte's embedding matrix
        self.transformer.wte.weight = self.lm_head.weight

        # Apply weight initialization (tied weight gets initialized once)
        self.apply(self._init_weights)
```

### KV Tying in Attention

**File**: `gpt2/model.py`

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # When KV tying enabled: only create Q and V projections (K = V)
        qkv_dim = 2 * config.n_embd if config.kv_tying else 3 * config.n_embd
        self.c_attn = nn.Linear(config.n_embd, qkv_dim, bias=config.bias)
        self.kv_tying = config.kv_tying

    def forward(self, x):
        if self.kv_tying:
            # Split into Q and V only
            q, v = self.c_attn(x).split(self.n_embd, dim=2)
            # Reshape for multi-head attention
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = v  # Tie K to V!
        else:
            # Standard: split into Q, K, V
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            # ... reshape ...

        # Attention computation proceeds normally
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y
```

For complete KV tying details, see **[kv-tying.md](kv-tying.md)**.

---

## References

### Related Documentation

- **[KV Tying](kv-tying.md)**: Detailed implementation and ablation results for attention KV tying
- **[Reciprocal Attention](ra.md)**: Geometric initialization and reciprocal pathway design
- **[Mechanistic Interpretability](mechint.md)**: Post-training analysis of tied vs untied models

### Academic Papers

**Weight tying in language models:**
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) - Original proposal
- [Tying Word Vectors and Word Classifiers](https://arxiv.org/abs/1611.01462) - Theoretical analysis
- [Press & Wolf (2017)](https://arxiv.org/abs/1608.05859) - Empirical validation on LSTMs

**Transformer weight sharing:**
- [Universal Transformers](https://arxiv.org/abs/1807.03819) - Layer-wise weight sharing
- [ALBERT](https://arxiv.org/abs/1909.11942) - Cross-layer parameter sharing
- [DeBERTa](https://arxiv.org/abs/2006.03654) - Disentangled attention without KV tying

**Parameter efficiency:**
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-rank adaptation as alternative to tying
- [Adapter Layers](https://arxiv.org/abs/1902.00751) - Task-specific parameters without full fine-tuning

### Code References

**Our implementation:**
- `gpt2/model.py`: Input-output tying and KV tying in `CausalSelfAttention`
- `ra.py`: Reciprocal MLP weight tying (experimental, ablation V2)
- `gpt2/Kconfig`: Config options for `GPT2_WEIGHT_TYING` and `GPT2_KV_TYING`

**Defconfigs:**
- `gpt2/defconfigs/gpt2-vanilla-baseline`: Standard tying (input-output only)
- `gpt2/defconfigs/gpt2-kv-tying-w7900-ablation`: KV tying ablation study (V0 vs V1)
- `gpt2/defconfigs/gpt2-ra-ablation`: R-MLP tying ablation (V2 vs V5)

---

## Summary

**Weight tying is a powerful parameter reduction technique** that shares tensors between layers performing related functions:

| Tying Type | Layers Shared | Param Reduction | Use Case |
|------------|--------------|-----------------|----------|
| **Input-Output** | wte ↔ lm_head | ~50% of embeddings | Language modeling |
| **KV Attention** | K ↔ V | ~33% of QKV | Memory-constrained attention |
| **Cross-Layer** | Blocks[i] ↔ Blocks[j] | ~1/n_layers | Universal transformers |
| **Reciprocal MLP** | R-MLP ↔ Attn proj | Variable | Experimental (see [RA](ra.md)) |

**Key principles:**
1. ✅ **Same tensor, not copies**: Tied parameters share memory and gradients
2. ✅ **Symmetric tasks**: Best when forward/backward paths are conceptually similar
3. ✅ **Regularization effect**: Constrains optimization to find shared representations
4. ✅ **Validate empirically**: Always run ablations to confirm tying helps your architecture

**For specific implementations:**
- Input-output tying: See implementation section above
- KV tying: See **[kv-tying.md](kv-tying.md)**
- Reciprocal architectures: See **[ra.md](ra.md)**
