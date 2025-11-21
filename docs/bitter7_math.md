# Bitter 7: Mathematical Foundations and Comparison with Movement and Magnitude Pruning

## Overview

Bitter 7 pruning—especially in its variance-aware, bias-compensated form used in this repository—is a principled approach to structured sparsification. This note summarizes the mathematical intuition behind the algorithm, contrasts it with movement pruning and magnitude pruning, and highlights the practical trade-offs that matter for large language models (LLMs). Because Bitter 7 is novel to this project, the claims below describe our own reasoning and internal observations while the cited works serve only as contextual references.

---

## 1. Mathematical Foundations

### 1.1 Bitter 7 Pruning

Bitter 7 relies on activation statistics gathered from a calibration set. For every neuron or channel `i` we estimate the mean `mu_i = E[h_i(x)]` and variance `Var[h_i(x)]`. Units whose variance falls below a threshold are treated as near-constant contributors that can be removed while injecting their mean effect into the downstream bias: `b_next <- b_next + W[:, i] * mu_i`. The procedure preserves the first moment of the network output, and the reconstruction error introduced by pruning neuron `i` is upper-bounded by `Var[h_i(x)]`.

Statistic-driven compression has also been explored in recent work. ExCP compresses checkpoints by shrinking weights using joint weight–momentum signals [^1], while Berisha et al. study variance-based pruning heuristics [^2]. Bitter 7 differs from both by explicitly pairing low-variance detection with bias compensation and by targeting structured units (neurons or channels) instead of individual weights. The choice of the variance EMA (Adam's beta2 term) stems from simulating different EMA settings on representative training traces (see `docs/adamwprune_variants.md`). Those visual comparisons showed that beta2=0.999 tracks long-term activity with far less noise than beta1≈0.9, motivating the use of `(exp_avg_sq)^(1/4)` as the pruning signal.

### 1.2 Movement Pruning

Movement pruning monitors how each weight evolves during fine-tuning [^3]. An importance score `S_i = |w_i_final| - |w_i_initial|` ranks weights whose magnitudes shrink; those with the most negative scores are removed. The method therefore depends on first-order optimization dynamics and requires access to fine-tuning trajectories.

### 1.3 Magnitude Pruning

Magnitude pruning removes weights whose absolute values fall below a threshold. It assumes small weights have minimal influence on the model output, uses no data, and can be applied iteratively or one-shot. Surveys such as Blalock et al. [^4] and Gale et al. [^5] describe its empirical success and limitations.

---

## 2. Loss Minimization and Error Preservation

### Bitter 7

* Each candidate neuron is approximated by its mean activation, which minimizes the squared reconstruction error `E[(h_i(x) - mu_i)^2] = Var[h_i(x)]`.
* Bias compensation keeps the downstream affine layer consistent, so `f_pruned(x)` tracks `f_original(x)` up to the small residual variance term.
* The method operates with zero-order information (activations only) yet explicitly preserves the model's output statistics.

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

---

## 6. Related Research and Contribution

**ExCP (Li et al., 2024)** compresses checkpoints by shrinking weights and their momentum buffers jointly [^1](https://arxiv.org/pdf/2406.11257). The optimizer statistics guide how aggressively tensors are quantized, but the method does not prune neurons or channels, does not rely on activation variance, and performs no bias compensation. Bitter 7 instead treats Adam statistics as a way to measure long-term activity, removes whole structured units detected as low variance, and explicitly folds their mean contribution into downstream biases to maintain function.

**Variance-Based Pruning (Berisha et al., 2025)** evaluates heuristics that rank units by activation variance inside CNNs [^2](https://arxiv.org/pdf/2507.12988). While this is philosophically close, the work focuses on unstructured pruning, omits bias injection, and stops at empirical heuristics rather than defining an end-to-end procedure for transformer-scale models. Bitter 7 builds on the same variance intuition but extends it with (1) calibration-aware structured pruning, (2) bias compensation for functional equivalence, and (3) LLM-scale evidence via the GPT-2 experiments referenced above.

Taken together, these papers show optimizer-aware compression and variance-aware scoring are active research areas, yet neither introduces the specific combination that defines Bitter 7. The contribution here is the practical recipe—calibration pass, low-variance detection using Adam beta2 statistics, bias compensation, and structured pruning—validated on large transformer workloads.

---

## Conclusion

Bitter 7 offers a loss-conscious, variance-sensitive criterion that turns near-constant neurons into explicit bias updates, preserving the network's mean response even at high sparsity. Compared with movement and magnitude pruning it trades extra calibration data for stronger functional guarantees and structured pruning support. Practitioners can mix and match it with gradient-driven or purely magnitude-based heuristics depending on how much data, time, and interpretability they need during compression.

---

## References

[^1]: Wenshuo Li, Xinghao Chen, Han Shu, Yehui Tang, and Yunhe Wang. 2024. *ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking*. arXiv:2406.11257. https://arxiv.org/pdf/2406.11257

[^2]: Uranik Berisha, Jens Mehnert, and Alexandru Paul Condurache. 2025. *Variance-Based Pruning for Accelerating and Compressing Trained Networks*. arXiv:2507.12988. https://arxiv.org/pdf/2507.12988

[^3]: Victor Sanh, Thomas Wolf, and Alexander M. Rush. 2020. *Movement Pruning: Adaptive Sparsity by Fine-Tuning*. NeurIPS.

[^4]: Daniel Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. 2020. *What Is the State of Neural Network Pruning?* arXiv:2003.03033.

[^5]: Trevor Gale, Erich Elsen, and Sara Hooker. 2019. *The State of Sparsity in Deep Neural Networks*. arXiv:1902.09574.
