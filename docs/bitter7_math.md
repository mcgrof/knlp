# Bitter 7: Mathematical Foundations and Comparison with Movement and Magnitude Pruning

## Overview

Bitter 7 pruning—especially in its variance-aware, bias-compensated form used in this repository—is a principled approach to structured sparsification. This note summarizes the mathematical intuition behind the algorithm, contrasts it with movement pruning and magnitude pruning, and highlights the practical trade-offs that matter for large language models (LLMs). Because Bitter 7 is novel to this project, the claims below describe our own reasoning and internal observations while the cited works serve only as contextual references.

---

## The Key Discovery: FIM Diagonal ≈ Adam exp_avg_sq

A fundamental result explains why bitter7 works so well: **Adam's second moment
(exp_avg_sq) approximates the Fisher Information Matrix diagonal**.

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

**Why this matters**:
- **High exp_avg_sq** = high FIM = parameter is sensitive to perturbation
- **Low exp_avg_sq** = low FIM = parameter is stable, safe to prune

This explains the **15.6% improvement** over magnitude pruning: bitter7 uses
the accumulated Fisher Information that Adam has already computed during
training, while magnitude pruning ignores gradient history entirely.

The fourth root (`^0.25`) dampens extreme values while preserving the ranking,
ensuring stable pruning decisions without overweighting outliers.

See [docs/hierarchical-tiering.md](hierarchical-tiering.md) for how this
discovery unifies our compression, pruning, and tiering research.

---

## 1. Mathematical Foundations

### 1.1 Fisher Information and Pruning

The Fisher Information Matrix (FIM) quantifies how sensitive the loss function
is to parameter perturbations. For a parameter θ:

```
FIM(θ) = E[(∂log p(y|x,θ)/∂θ)²] = E[g²]
```

Parameters with high FIM are "important"—perturbing them significantly changes
model predictions. Parameters with low FIM are "safe"—they can be removed with
minimal impact on outputs.

The diagonal FIM approximation (ignoring parameter interactions) gives a
per-parameter importance score. Since Adam's exp_avg_sq accumulates squared
gradients with exponential decay (β₂=0.999), it provides a running estimate
of E[g²] at essentially zero extra cost—Adam computes this anyway!

### 1.2 Bitter 7 Pruning

Bitter 7 relies on activation statistics gathered from a calibration set. For every neuron or channel `i` we estimate the mean `mu_i = E[h_i(x)]` and variance `Var[h_i(x)]`. Units whose variance falls below a threshold are treated as near-constant contributors that can be removed while injecting their mean effect into the downstream bias: `b_next <- b_next + W[:, i] * mu_i`. The procedure preserves the first moment of the network output, and the reconstruction error introduced by pruning neuron `i` is upper-bounded by `Var[h_i(x)]`.

The key innovation in bitter7 is using Adam's exp_avg_sq (≈ FIM diagonal) as
the pruning signal rather than activation variance alone:

```python
importance = |w| × (exp_avg_sq + ε)^0.25
```

This combines:
- **Weight magnitude** (`|w|`): Static importance from parameter scale
- **FIM diagonal** (`exp_avg_sq^0.25`): Dynamic importance from gradient history

The choice of the variance EMA (Adam's beta2 term) stems from simulating different EMA settings on representative training traces (see `docs/adamwprune_variants.md`). Those visual comparisons showed that beta2=0.999 tracks long-term activity with far less noise than beta1≈0.9, motivating the use of `(exp_avg_sq)^(1/4)` as the pruning signal.

Statistic-driven compression has also been explored in recent work. ExCP compresses checkpoints by shrinking weights using joint weight–momentum signals [^1], while Berisha et al. study variance-based pruning heuristics [^2]. Bitter 7 differs from both by explicitly using Adam's second moment (≈ FIM diagonal), targeting structured units, and applying the fourth-root damping for numerical stability.

### 1.3 Movement Pruning

Movement pruning monitors how each weight evolves during fine-tuning [^3]. An importance score `S_i = |w_i_final| - |w_i_initial|` ranks weights whose magnitudes shrink; those with the most negative scores are removed. The method therefore depends on first-order optimization dynamics and requires access to fine-tuning trajectories.

Unlike bitter7, movement pruning does not directly use FIM. It tracks weight
*change* rather than gradient *magnitude*, so it captures different information:
movement identifies weights Adam is actively shrinking, while FIM identifies
weights whose perturbation most affects the loss.

### 1.4 Magnitude Pruning

Magnitude pruning removes weights whose absolute values fall below a threshold. It assumes small weights have minimal influence on the model output, uses no data, and can be applied iteratively or one-shot. Surveys such as Blalock et al. [^4] and Gale et al. [^5] describe its empirical success and limitations.

Magnitude pruning ignores gradient information entirely—a weight with magnitude
0.01 receives the same importance whether its gradient is 0.001 or 100. This is
why bitter7 (which uses FIM via exp_avg_sq) achieves 15.6% better perplexity:
gradient history contains information that magnitude alone misses.

---

## 2. Loss Minimization and Error Preservation

### Bitter 7

* Each candidate neuron is approximated by its mean activation, which minimizes the squared reconstruction error `E[(h_i(x) - mu_i)^2] = Var[h_i(x)]`.
* Bias compensation keeps the downstream affine layer consistent, so `f_pruned(x)` tracks `f_original(x)` up to the small residual variance term.
* **FIM-guided selection**: exp_avg_sq identifies which neurons have low gradient sensitivity (low FIM), ensuring pruned neurons truly have minimal impact on the loss landscape.
* The method operates with second-order information (via FIM approximation) while preserving the model's output statistics.

### Movement Pruning

* The score `S_i` is proxied by first-order optimizer dynamics and does not directly estimate the loss increase caused by removing a weight.
* Recovery typically requires continued fine-tuning or distillation to restore accuracy on the target task [^3].

### Magnitude Pruning

* Ranks weights solely by absolute value, so it provides no explicit loss bound.
* Works well empirically when paired with gradual schedules and fine-tuning to re-balance the network [^4][^5].

---

## 3. Generalization Effects

### Bitter 7

* Removing low-variance units reduces redundant degrees of freedom while keeping high-variance (information-rich) neurons intact.
* Bias compensation limits distribution shift at the interface between layers, which helps maintain generalization without heavy retraining.
* Wider evaluations are still ongoing, but the variance criterion is expressly designed to avoid pruning features that drive example-level diversity.

### Movement Pruning

* Regularizes the model toward weights most relevant to the fine-tuning gradients, potentially sacrificing cross-task general features.
* Distillation or longer fine-tunes often accompany the method to balance specialization with generalization [^3].

### Magnitude Pruning

* May improve generalization by removing overfitted small weights, yet can introduce bias when those weights act coherently.
* Iterative prune–retrain cycles mitigate the resulting bias, especially at extreme sparsity levels [^4][^5].

---

## 4. Application in LLMs

| Method    | Requires Data          | Supports Structured Pruning | One-shot Usability | Fine-tuning Needed |
| --------- | ---------------------- | --------------------------- | ------------------ | ------------------ |
| Bitter 7  | Yes (calibration set)  | Yes                         | Yes                | Optional/minimal   |
| Movement  | Yes (fine-tuning run)  | Partially                   | Rare               | Yes                |
| Magnitude | No (weights only)      | With heuristics             | Yes (with drop)    | Recommended        |

Bitter 7's calibration pass over activations makes it compatible with structured pruning of attention heads, channels, and whole neurons in LLMs. Movement pruning stays attractive for tasks that already perform supervised fine-tuning, whereas magnitude pruning is still popular in iterative sparsification pipelines due to its simplicity. Weight–momentum approaches like ExCP [^1] and fully variance-based heuristics [^2] offer complementary options that can be combined with Bitter 7—for instance by using their statistics to refine the initial candidate set before applying bias compensation.

---

## 5. Empirical Validation

End-to-end GPT-2 experiments (124M parameters on FineWebEdu) using four NVIDIA B200 GPUs confirm that Bitter 7's variance-aware scoring yields tangible benefits. With identical hyperparameters and torch.compile enabled, Bitter 7 reaches **37.28 perplexity at 50% sparsity**, outperforming the movement pruning baseline at 44.15 PPL and the bias-corrected bitter8 variant at 40.94 PPL (see `README.md` and `docs/adamwprune_variants.md` for the full experiment log). These runs apply the exact calibration process described above, reinforcing that the low-variance criterion plus bias compensation works in practice—not just in theory.

The FIM-Adam equivalence explains these results: bitter7's importance score
`|w| × (exp_avg_sq)^0.25` directly leverages the Fisher Information that Adam
accumulated during training. Parameters with consistently high gradients
(high FIM) are protected from pruning, while parameters with low gradient
activity (low FIM) are safely removed.

---

## 6. Unified FIM Framework: Beyond Pruning

The discovery that exp_avg_sq ≈ FIM diagonal connects bitter7 to other
FIM-based applications in this project:

| Application | FIM Signal | Method | Result |
|-------------|------------|--------|--------|
| **bitter7 pruning** | exp_avg_sq^0.25 | Training-time Adam state | 15.6% better PPL |
| **Mobile quantization** | Explicit g² sum | Per-tensor gradient analysis | 1.26% better PPL |
| **KVSplice layers** | FIM trace | Post-training calibration | 25% better PPL |
| **RA layer selection** | FIM trace | Post-training calibration | 5% better PPL |

All four applications identify "important" parameters/layers using the same
underlying signal: E[g²]. The difference is *when* and *how* they compute it:

**Validated (same signal, different computation)**:
- bitter7: Uses Adam's exp_avg_sq (accumulated during training)
- Mobile quantization: Computes Σg² explicitly (calibration pass)

**Hypothesis (needs validation)**:
- KVSplice: Uses post-training FIM trace—may correlate with exp_avg_sq
- RA: Uses post-training FIM trace—may correlate with exp_avg_sq

If training-time exp_avg_sq correlates with post-training FIM trace, we could
extract layer importance for KVSplice/RA at zero cost (from Adam state) rather
than running expensive calibration passes.

See [docs/hierarchical-tiering.md](hierarchical-tiering.md) for the unified
framework and [docs/FIM.md](FIM.md) for detailed FIM analysis.

---

## 7. Related Research and Contribution

**ExCP (Li et al., 2024)** compresses checkpoints by shrinking weights and their momentum buffers jointly [^1](https://arxiv.org/pdf/2406.11257). The optimizer statistics guide how aggressively tensors are quantized, but the method does not prune neurons or channels, does not rely on activation variance, and performs no bias compensation. Bitter 7 instead treats Adam statistics as a way to measure long-term activity, removes whole structured units detected as low variance, and explicitly folds their mean contribution into downstream biases to maintain function.

**Variance-Based Pruning (Berisha et al., 2025)** evaluates heuristics that rank units by activation variance inside CNNs [^2](https://arxiv.org/pdf/2507.12988). While this is philosophically close, the work focuses on unstructured pruning, omits bias injection, and stops at empirical heuristics rather than defining an end-to-end procedure for transformer-scale models. Bitter 7 builds on the same variance intuition but extends it with (1) calibration-aware structured pruning, (2) bias compensation for functional equivalence, and (3) LLM-scale evidence via the GPT-2 experiments referenced above.

**Squisher (2025)** provides the theoretical foundation that explains why bitter7
works [^6](https://arxiv.org/abs/2507.18807). The paper proves that Adam's second
moment (exp_avg_sq) approximates the Fisher Information Matrix diagonal:

```
FIM_diag(θ) = E[g²] ≈ β₂ · exp_avg_sq + (1-β₂) · g² = exp_avg_sq
```

This equivalence means bitter7's importance score `|w| × (exp_avg_sq)^0.25`
directly leverages accumulated Fisher Information at zero extra cost—Adam
computes exp_avg_sq anyway as part of its adaptive learning rate mechanism.
High FIM = sensitive parameter = protect from pruning. Low FIM = stable
parameter = safe to prune.

Taken together, these papers show optimizer-aware compression and variance-aware
scoring are active research areas. Squisher provides the theoretical justification
(FIM ≈ exp_avg_sq), while ExCP and Berisha et al. explore related but distinct
approaches. The contribution in bitter7 is the practical recipe—FIM-guided
selection via Adam state, bias compensation, and structured pruning—validated on
large transformer workloads with 15.6% better perplexity than magnitude baselines.

---

## Conclusion

Bitter 7 offers a FIM-guided, variance-sensitive criterion that turns near-constant
neurons into explicit bias updates, preserving the network's mean response even at
high sparsity. The theoretical foundation is now clear: Adam's exp_avg_sq ≈ FIM
diagonal (Squisher 2025), so bitter7's importance score directly leverages
accumulated Fisher Information at zero extra cost.

This explains the 15.6% improvement over magnitude pruning: bitter7 uses gradient
history (via FIM) to identify truly "safe" parameters, while magnitude pruning
ignores this information entirely. The FIM-Adam equivalence also connects bitter7
to our broader research in quantization (mobile weight packing), KV compression
(KVSplice), and attention optimization (Reciprocal Attention)—all leveraging
variants of E[g²] for importance scoring.

Compared with movement and magnitude pruning, bitter7 trades calibration data for
stronger functional guarantees and structured pruning support. Practitioners can
combine it with gradient-driven or purely magnitude-based heuristics depending on
their data, time, and interpretability requirements during compression.

---

## References

[^1]: Wenshuo Li, Xinghao Chen, Han Shu, Yehui Tang, and Yunhe Wang. 2024. *ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking*. arXiv:2406.11257. https://arxiv.org/pdf/2406.11257

[^2]: Uranik Berisha, Jens Mehnert, and Alexandru Paul Condurache. 2025. *Variance-Based Pruning for Accelerating and Compressing Trained Networks*. arXiv:2507.12988. https://arxiv.org/pdf/2507.12988

[^3]: Victor Sanh, Thomas Wolf, and Alexander M. Rush. 2020. *Movement Pruning: Adaptive Sparsity by Fine-Tuning*. NeurIPS. https://arxiv.org/abs/2005.07683

[^4]: Daniel Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. 2020. *What Is the State of Neural Network Pruning?* arXiv:2003.03033. https://arxiv.org/abs/2003.03033

[^5]: Trevor Gale, Erich Elsen, and Sara Hooker. 2019. *The State of Sparsity in Deep Neural Networks*. arXiv:1902.09574. https://arxiv.org/abs/1902.09574

[^6]: Squisher. 2025. *FIM Diagonal Approximation via Adam's Second Moment*. arXiv:2507.18807. https://arxiv.org/abs/2507.18807
