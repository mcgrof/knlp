# BPA v27: Submission Readiness Assessment

## What Is Publishable Now

The core empirical finding — that k* is bounded by a small constant
across the tested depth range — is solid and novel. The specific
contributions:

1. **Systematic per-layer INT4 ablation** across 5 models and 3
   architectures (Qwen2, Mistral, Llama). This is the first work
   (to our knowledge) to measure per-layer INT4 sensitivity
   exhaustively and determine the minimum protection count.

2. **Two distinct mechanisms** for bounded k*: sink dominance
   (Qwen) and uniform robustness (Mistral, Llama-2). This
   dual-mechanism story is stronger than either alone, now
   confirmed across 3 architecture families.

3. **Accumulation theory** with calibrated parameters. The error
   propagation framework from v24 provides mechanistic grounding.

4. **Sub-0.30 kv_ratio** at PASS_3% across all tested models.
   Practical relevance for serving systems.

## What Is Strong Enough for arXiv Immediately

The following could be posted to arXiv with minimal additional work:

- Main result table (k* across 4 models)
- Sensitivity distribution plots (sink vs uniform)
- k* vs D plot
- Accumulation theory summary
- Clear limitations section

This would be positioned as an **empirical study** with systems
implications, not as a theoretical contribution.

## What Is Missing for a Stronger Conference Submission

### Critical gaps (ordered by impact)

1. **No fused INT4 attention kernel** (blocks MLSys submission)
   - Without end-to-end latency improvement, the systems story is
     incomplete. kv_ratio reduction is necessary but not sufficient
     for a systems paper.
   - Estimated effort: 2-4 weeks to integrate FlashInfer INT4 or
     write a custom CUDA kernel.
   - Impact: Would elevate from "interesting measurement" to
     "deployable technique."

2. **Limited model scale** (weakens NeurIPS/ICML angle)
   - Three architecture families (Qwen2, Mistral, Llama) now
     covered, including MHA (Llama-2-7b with 32 KV heads).
   - No models above 7.6B. Adding a 13B+ model (D>=40) would
     extend D range and strengthen the O(1) argument.
   - Llama-2-7b limited to L<=4K (short context window).

3. **No downstream task evaluation** (weakens ML venue angle)
   - All quality measurement is PPL on wikitext. Reviewers at
     NeurIPS/ICML will ask about MMLU, HumanEval, etc.
   - Estimated effort: 1-2 days to add lm-eval-harness integration.

4. **Oracle ranking is expensive** (practical limitation)
   - D forward passes per model is acceptable for calibration but
     the paper should discuss transfer ranking (0.5B ranking applied
     to 7B) and proxy signals as future directions.

### Minor gaps

- Only g=32 tested; other group sizes may shift k*
- Greedy decoding only
- Fixed seed count (3); more seeds would tighten confidence

## Venue Fit Assessment

### arXiv-first (Recommended for immediate posting)

**Fit: Strong.** The result is clean, novel, and well-supported by
data. arXiv gives priority without peer review overhead. This
establishes the result while additional experiments are run.

Positioning: "Empirical study of per-layer KV quantization
sensitivity in transformer LLMs."

### ACL/EMNLP Findings

**Fit: Moderate.** The NLP angle is thin — this is more of a
systems/efficiency paper. The PPL-only evaluation is a weakness
for NLP venues. The finding about attention sinks aligns with
existing NLP work (StreamingLLM).

Strengths: Clear writing, systematic experiments, negative results
documented.
Weaknesses: No downstream tasks, limited to KV quantization.

### MLSys

**Fit: Moderate-to-Strong (with fused kernel).**

Without fused kernel: **Weak.** MLSys reviewers want end-to-end
speedups. kv_ratio reduction alone is a partial contribution.

With fused kernel: **Strong.** Sub-0.30 kv_ratio with <3% quality
loss and measured latency improvement would be a compelling MLSys
contribution. The dual-mechanism story (sink vs uniform) adds
novelty beyond pure engineering.

### NeurIPS/ICML Workshop

**Fit: Strong.** The right length and depth for a workshop paper.
Could be submitted to the Efficient ML workshop or similar.

### NeurIPS/ICML Main Conference

**Fit: Weak without additional work.** The theoretical contribution
(accumulation theory) is not rigorous enough for a theory paper. The
empirical contribution needs more architectures and downstream
evaluation. The systems contribution needs fused kernels.

## COMPLETED: Third Architecture (Llama-2-7b)

**Llama-2-7b-hf oracle + k-sweep completed on H100.**

Results:
- k*(3%) = 0 (all-INT4 passes with max_delta=1.01%)
- k*(1%) = 2 (need 2 protected layers for 1% tolerance)
- Uniform robustness pattern (no dominant sink layer)
- Layer 0 NOT most sensitive (delta=0.76%, ranked 22nd)
- Most sensitive: layer 3 (1.18%), then layer 25 (1.05%)
- kv_ratio = 0.2812 at k*=0

This substantially strengthens the O(1) story with a third
architecture family and an MHA model (32 KV heads).

## Single Most Credible Additional Experiment

**Run a 13B+ model (D >= 40) to extend the D range.**

Candidates:
- NousResearch/Llama-2-13b-hf (D=40, MHA, 40 heads, L<=4K)
- Qwen2.5-14B (D=48, if accessible)

If k* <= 3 at D=40+, the O(1) argument becomes substantially
stronger across D = {24, 28, 32, 40+}.

## Honest Assessment

The BPA arc is a solid empirical contribution. The O(1) observation
is novel and practically relevant. The dual-mechanism story is
interesting and now confirmed across 3 architecture families.
The accumulation theory provides useful intuition even if it is
not a formal proof.

The main weakness is the gap between "measured kv_ratio" and
"demonstrated end-to-end speedup." This is a known limitation that
is clearly documented but will cost points at systems venues.

The path to a strong paper is:
1. Post arXiv now (establishes priority)
2. Add fused kernel + larger D model over 4-6 weeks
3. Submit to MLSys or an appropriate workshop with full results

The result is real. The science is tight. The execution is careful.
What it lacks is the final engineering step to close the
theory-to-practice gap.
