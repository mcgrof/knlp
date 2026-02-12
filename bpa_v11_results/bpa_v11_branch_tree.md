# BPA v11 Branch Tree

## Observed Outcomes

### What worked
- **Matched-quality PASS at all 4 L values** (512, 1024, 2048, 4096)
  at both 1% and 3% tolerance. BPA can find configs that preserve
  dense PPL.
- **Latency improvements** of 2-5% across all L at batch=1 with
  tuned configs. Consistent across seeds for latency (though PPL
  variance is high).
- **RoPE position tracking fix** was critical — without it, any
  eviction destroys quality on RoPE models.

### What did not work
- **Far budget does not spike**: k_far stays at 3.0-3.6 with max=4
  across all stress modes (easy and hard). The pressure signal
  does not distinguish hard retrieval-dependent spans from easy
  local-context spans. The 2x spike criterion is not met.
- **Conservative configs dominate**: At L>=1024, BPA must keep
  95-99% of tokens to match quality. The model genuinely uses
  most of its context.
- **Static sparse fails at L>=1024**: A fixed local window +
  random far budget is too rigid. BPA's adaptive controller
  helps but only marginally.
- **Bandwidth is not the bottleneck**: Even at batch=4, KV cache
  is only 48-52MB at L=4096, tiny vs 48GB GPU memory. Latency
  wins don't increase with batch size.

## Branch Analysis

### BRANCH 2: Matched-quality PASS but latency flat
**This is where we are.** BPA passes quality but latency gains
are modest (2-5%) and don't scale with batch size.

Root cause: Qwen2.5-0.5B at these context lengths is compute-
bound, not memory-bandwidth-bound. The KV cache is small
relative to model weights and attention compute.

Next steps for v12:
1. **Profile attention kernel time** vs gate time vs MLP time to
   confirm compute-bound diagnosis.
2. **Test on a larger model** (3B-7B) or much longer contexts
   (16K-32K) where KV cache becomes a real fraction of GPU
   memory and bandwidth.
3. **Consider fused KV gather/scatter** — the 2-5% win comes
   from smaller attention matmul, not memory bandwidth. A fused
   kernel that selects and attends in one pass could amplify
   this.

### BRANCH 4: Far budget does not spike
**This is also where we are.** The entropy+residual pressure
signal cannot distinguish hard (far-retrieval) spans from easy
(local-context) spans.

Root cause: The pressure signal (0.7*entropy + 0.3*residual_norm)
reflects overall prediction difficulty, not retrieval need. A
token can be "hard" because of vocabulary rarity without needing
far context, or "easy" while silently depending on far context.

Next steps for v12:
1. **Add attention-based retrieval signal**: Instead of entropy,
   track how much attention weight falls on far tokens. If the
   model's own attention points far, that's the ground truth for
   far retrieval need.
2. **Two-stage scoring**: Cheap entropy check first, then
   attention-weight-based refinement only when entropy is high.
3. **Supervised controller**: Label tokens as "needs far context"
   using attention patterns from a dense reference run, then
   train a tiny MLP to predict the label from cheap features.

### BRANCH 5 (not triggered): Memory peaks
Not an issue at this scale. Qwen2.5-0.5B uses <8GB at L=4096
batch=4. Would become relevant at 7B+ models or 32K+ context.

## Decision: v12 Direction

The most impactful next step is **testing on a regime where KV
cache actually dominates**. Options ranked by expected impact:

1. **Longer context**: Use the same Qwen2.5-0.5B but test at
   L=8192, 16384, 32768 (all within max_ctx=32768). KV cache
   at L=32768 is ~384MB per sequence, much more relevant.
2. **Bigger model**: A 3B-7B model has ~10x the KV cache per
   token. At L=4096, this could be ~500MB of KV per sequence.
3. **Attention-based retrieval signal**: Fix the far budget
   controller by using the model's own attention as the oracle
   for what tokens matter.

Priority: Option 1 first (cheapest to test, reuses existing
infrastructure). If wins appear at long context, then pursue
option 3 to make the controller actually adaptive.
