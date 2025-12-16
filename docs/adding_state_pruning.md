# Adding State-Based Pruning to Any Optimizer

This document demonstrates how to add state-based pruning to any optimizer
with minimal code changes, and shares experimental findings on ResNet-18
and GPT-2.

## Why State-Based Pruning Works: The FIM Connection

A fundamental discovery explains the effectiveness of state-based pruning:
**Adam's exp_avg_sq approximates the Fisher Information Matrix diagonal**.

```
FIM_diag(θ) = E[(∂L/∂θ)²] = E[g²]
Adam exp_avg_sq = β₂ · exp_avg_sq + (1-β₂) · g² ≈ E[g²]
```

This equivalence, validated by [Squisher (2025)](https://arxiv.org/abs/2507.18807),
means that bitter7's importance score directly leverages Fisher Information:

```python
# bitter7: Uses FIM diagonal for pruning
importance = |w| × (exp_avg_sq + ε)^0.25  # exp_avg_sq ≈ FIM diagonal
```

- **High exp_avg_sq** = high FIM = parameter is sensitive to perturbation
- **Low exp_avg_sq** = low FIM = parameter is stable, safe to prune

This is why state-based pruning outperforms magnitude pruning: it uses
accumulated gradient statistics (FIM) that Adam has already computed,
rather than ignoring gradient history entirely.

See [docs/hierarchical-tiering.md](hierarchical-tiering.md) for how this
unifies our compression, pruning, and tiering research.

## GPT-2 Results (B200x4)

State-based pruning achieves significant improvements on GPT-2 124M with
FineWebEdu dataset on NVIDIA B200x4 GPUs:

![AdamWPrune B200x4 Results](../images/adamwprune_fair_comparison.png)
*State-based pruning outperforms magnitude baseline with identical
hyperparameters. bitter7 achieves 37.28 PPL (15.6% better) than
movement pruning baseline (44.15 PPL). All runs with torch.compile.*

| Variant | PPL | vs Baseline | Iterations |
|---------|-----|-------------|------------|
| Movement Pruning | 44.15 | - | 5,000 |
| bitter8 | 40.94 | **-7.3%** | 2,500 |
| **bitter7** | **37.28** | **-15.6%** | 7,000 |

All runs use torch.compile with identical hyperparameters (batch 128,
grad_acc 8, lr 0.0006, effective batch 1024).

See [docs/pruning.md](pruning.md) for full GPT-2 pruning research and
[docs/adamwprune_variants.md](adamwprune_variants.md) for bitter variant
details.

## ResNet-18 Results (CIFAR-10)

Testing on ResNet-18 with CIFAR-10 validates the approach on vision models:

### Optimizer Performance (No Pruning)
- **AdamW**: 90.30% accuracy
- **AdamWPrune (AdamW base)**: 90.28% accuracy
- **Memory usage**: Both ~1307 MB

### Pruning Performance at 50% Sparsity
- **AdamW + Movement Pruning**: 90.69% accuracy
- **AdamWPrune + State Pruning**: 90.69% accuracy (tied)
- **AdamW + Magnitude Pruning**: 88.97% accuracy

### Pruning Performance at 70% Sparsity
- **AdamW + Movement Pruning**: 89.68% accuracy
- **AdamWPrune + State Pruning**: 89.37% accuracy
- **AdamW + Magnitude Pruning**: 88.44% accuracy

## Minimal Code Required

Adding state-based pruning requires ~50 lines of code:

### 1. State Initialization (~15 lines)

```python
adamprune_state = {
    "pruning_enabled": enable_pruning,
    "target_sparsity": 0.7,
    "warmup_steps": 100,
    "pruning_frequency": 50,
    "ramp_end_epoch": 75,
    "step_count": 0,
    "masks": {},
    "pruning_strategy": "bitter7",  # Uses exp_avg_sq (FIM diagonal)
}

if adamprune_state["pruning_enabled"]:
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            mask = torch.ones_like(module.weight.data, dtype=torch.bool)
            module.register_buffer("adamprune_mask", mask)
            adamprune_state["masks"][module] = module.adamprune_mask
```

### 2. Gradient Masking (~10 lines)

```python
def apply_adamprune_masking(optimizer, adamprune_state):
    if adamprune_state is None or not adamprune_state["pruning_enabled"]:
        return

    for module, mask in adamprune_state["masks"].items():
        if module.weight.grad is not None:
            module.weight.grad.mul_(mask.float())
```

### 3. Mask Updates Using FIM Diagonal (~25 lines)

```python
def update_adamprune_masks(optimizer, adamprune_state, epoch):
    if adamprune_state is None or not adamprune_state["pruning_enabled"]:
        return

    adamprune_state["step_count"] += 1

    if (adamprune_state["step_count"] < adamprune_state["warmup_steps"] or
        adamprune_state["step_count"] % adamprune_state["pruning_frequency"] != 0):
        return

    progress = min(epoch / adamprune_state["ramp_end_epoch"], 1.0)
    current_sparsity = adamprune_state["target_sparsity"] * progress

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue

            state = optimizer.state[p]
            if "exp_avg_sq" in state:
                # exp_avg_sq ≈ FIM diagonal (Squisher paper)
                # Fourth root dampens extreme values for stable pruning
                importance = p.abs() * (state["exp_avg_sq"].abs() + 1e-8) ** 0.25

                for module, mask in adamprune_state["masks"].items():
                    if module.weight is p:
                        threshold = torch.quantile(importance.flatten(), current_sparsity)
                        mask.data = importance > threshold
                        break
```

## Integration Example

```diff
def train_epoch(model, train_loader, criterion, optimizer, device, adamprune_state=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

+       apply_adamprune_masking(optimizer, adamprune_state)

        optimizer.step()

+       if batch_idx % 100 == 0:
+           update_adamprune_masks(optimizer, adamprune_state, epoch)
```

## Comparison Summary

| Method | Sparsity | Code | Memory | ResNet Acc | GPT-2 PPL |
|--------|----------|------|--------|------------|-----------|
| Magnitude | 50% | ~30 lines | +93 MB | 88.97% | - |
| Movement | 50% | ~100 lines | +168 MB | 90.69% | 44.15 |
| **State (bitter7)** | **50%** | **~50 lines** | **+168 MB** | **90.69%** | **37.28** |

State-based pruning matches movement pruning on ResNet-18 while achieving
15.6% better perplexity on GPT-2, with half the code complexity.

## References

- [Squisher Paper](https://arxiv.org/abs/2507.18807) - FIM diagonal ≈ Adam exp_avg_sq
- [GPT-2 Pruning Results](pruning.md) - Full B200x4 experiments
- [Bitter Variants](adamwprune_variants.md) - bitter7, bitter8 details
- [Hierarchical Tiering](hierarchical-tiering.md) - Unified FIM framework
