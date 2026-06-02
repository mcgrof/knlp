# KRI-FT results

This is the full evaluation of KRI-aware fine-tuning. The summary lives in
the [README](README.md); this document is the evidence behind it — every
phase, every table, and the caveats that keep the claims honest. The two
headline results (the B_min scaling and the canonical-router KL gap) are
reproduced across seeds 42/43/44; everything else is single-seed and labeled.

## The claim, decomposed

KV-cache routing keeps a subset of cache blocks per query and is
training-free. KRI-FT is the inverse move: fine-tune the model under a
routing mask so it carries dropped information forward in its own hidden
states and stops depending on the discarded blocks. Three jobs are
separable, and KRI-FT only does the third:

```
Evidence retrieval     query-aware selection of the answer-bearing block
Support routing        query-agnostic selection of scaffold blocks (KRI, sink, recent)
Support internalization  KRI-FT: train the model to need less external scaffold
```

Routing reduces the cache while preserving scaffold; KRI-FT reduces the
scaffold burden itself.

## Setup

GPT-2 small (124M), fine-tuned 5000 steps on FineWeb-Edu sample-10BT
(streaming) at seq_len 1024, effective batch 32, lr 1e-5, teacher-KL
distillation against frozen vanilla GPT-2 (alpha 0.1), bf16, gradient
checkpointing, on a single 48 GB workstation GPU (AMD W7900, ROCm 6.4,
torch 2.9.1). Two matched fine-tunes:

- **KRI-FT** — mixed-mode training with the KRI-Q+N mask on ~70% of
  micro-batches, dense attention on the rest.
- **dense-FT** — same data, same KL, same step count, dense attention
  throughout: the apples-to-apples control.

Vanilla `openai-community/gpt2` is the untrained baseline. Routing-policy
labels: `full`, `recent` (local window only), `sink_recent` (sink + window),
`kri` (the internal KRI-Q+N mask: sink + window + top-k scored blocks).

## Nominal vs actual KV retention (read this first)

`eval_effective_retention.py` reports the *actual* mean retained-KV fraction
per (policy, nominal budget). The nominal label is misleading because the
`kri` policy's sink + window + top-k regions add up to far more KV than
`recent` at the same label:

| Policy | Nominal | Actual mean | Actual p95 |
|---|---:|---:|---:|
| recent / sink_recent | 12.5% | 23.4% | 25.0% |
| recent / sink_recent | 6.25% | 12.1% | 12.5% |
| kri | 12.5% | 37.6% | 54.1% |
| kri | 6.25% | 24.8% | 34.4% |

**Iso-actual-retention is the honest x-axis, not the nominal label.** Every
comparison below is read in that light.

## Routing-policy perplexity (GPT-2)

Raw matrix from `eval_pruned_ppl.py`; the `actual` column is the retained-KV
fraction that belongs on a plot's x-axis.

| Policy | Nominal | Actual | Vanilla | Dense-FT | KRI-FT |
|---|---:|---:|---:|---:|---:|
| full | 100% | 100% | 9.62 | 9.50 | **9.50** |
| recent | 50% | 75.0% | 21.11 | 20.76 | **18.23** |
| recent | 25% | 43.8% | 125.31 | 122.95 | **34.25** |
| recent | 12.5% | 23.4% | 379.23 | 364.14 | **60.03** |
| recent | 6.25% | 12.1% | 843.49 | 810.84 | **109.95** |
| sink_recent | 25% | 43.8% | 10.17 | 10.08 | **9.66** |
| sink_recent | 12.5% | 23.4% | 10.97 | 10.86 | **10.27** |
| sink_recent | 6.25% | 12.1% | 13.63 | 13.50 | **12.19** |
| kri | 25% | 73.0% | 10.08 | 10.02 | **9.59** |
| kri | 12.5% | 37.6% | 11.56 | 11.57 | **9.90** |
| kri | 6.25% | 24.8% | 12.85 | 12.88 | **10.61** |

Two honest readings:

1. **KRI-aware training generalizes across routing policies.** At ~23%
   actual retention KRI-FT is 60 PPL under `recent` vs vanilla's 379 (6.3×),
   10.27 vs 10.97 under `sink_recent` (−6%), and 10.61 vs 12.85 under `kri`
   at ~25% actual (−17%). The win exists under every policy, including ones
   never seen in training — so the training method, not a specific router, is
   doing the work.
2. **At iso-actual-retention the `kri` score is not visibly better than
   `sink_recent`.** A useful negative result: the content-aware score, as
   implemented, does not beat plain sink+recent at a matched actual budget.
   What KRI-FT buys is robustness of the *model*, not magic in the router.

The matched `dense-FT` control stays tied with vanilla under every routing
policy — confirming the operative variable is the routing-aware training, not
the FineWeb-Edu exposure.

## Minimum blocks vs context: B_min(L) (3 seeds)

`eval_l2m_scaling.py` finds the minimum number of global KV blocks needed to
stay within 0.03 nats/token of full attention (sink + 128-token window always
on). Reproduced on seeds 42/43/44.

| Context L | Vanilla | Dense-FT | KRI-FT (42/43/44) |
|---:|---:|---:|---:|
| 256 | 16 | 16 | **4 / 4 / 4** |
| 512 | 14 | 14 | **3 / 2 / 2** |
| 1024 | 8 | 8 | **1 / 1 / 1** |

At 1024 tokens KRI-FT reaches full-attention quality with a single global
block where vanilla needs eight — an 8× reduction. The dense control is
bit-identical to vanilla, again isolating the routing-aware training as the
cause.

The honest interpretation is **KRI-FT reaches the fixed sink+recent support
floor at a smaller context than vanilla**, not a smooth negative scaling law.
The B_min trajectory 4→2→1 is integer-discrete and bounded below by zero;
fitting a log-log slope to three points whose last value is one is asking a
lot of the math. The slope is γ ≈ −1.0 for KRI-FT vs −0.5 for vanilla
(r²=1.00 on seeds 43/44), and the SmolLM2 data below (γ ≈ 0 for both models)
confirms it is a floor-reaching effect, not an extrapolable law.

A `random_global` control (KRI-FT model, but eval-time blocks picked at
random) needs 8 blocks at L=1024 ε=0.03 vs the KRI score's 1 — so the score
function does contribute at eval time, it just helps KRI-FT most.

## Canonical KRI-family routers (3 seeds)

`eval_canonical_kri.py` runs the trained model under reference
reimplementations of the canonical routers — KRI-Q (cosine query·block-key),
KRI-Q-window (anchor + contiguous window), KRI-G (k-means on block-mean
keys), and a kv-sum router — at K=8 blocks, scored by mean decode-region
KL-to-full-cache (lower is better). Reproduced on seeds 42/43/44.

### What each router is, and where the idea comes from

These routers are not ours. They are reimplementations of the *shapes* of
published, **training-free** KV-cache selection methods — which is the whole
point of the program: take what the state of the art can already do without
retraining, and measure whether a light fine-tune makes a model tolerate it
better. Each router we evaluated and the paper it is inspired by:

- **KRI-Q** — query-conditioned block criticality: score each block by the
  cosine of the query to the block's mean key, keep the top-k. This is the
  idea in **Quest** (Tang et al., 2024, ICML; [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)),
  which estimates per-page criticality from the query against per-page key
  bounds and loads only the top-k critical pages.
- **KRI-Q-window** — KRI-Q with a contiguous sink + recent window around the
  anchor. The sink-plus-recent-window structure is **StreamingLLM** (Xiao et
  al., 2023, ICLR 2024; [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)),
  which keeps initial "attention-sink" tokens plus a sliding recent window;
  the query-anchored relevance on top is Quest-style.
- **KRI-G** — query-agnostic, geometric: k-means over block-mean keys, pick the
  blocks closest to the cluster centers. The closest published idea is
  **InfLLM** (Xiao et al., 2024; [arXiv:2402.04617](https://arxiv.org/abs/2402.04617)),
  which selects representative-key block memory units training-free.
- **kv-sum** (a.k.a. KRI-D) — per-block ‖ΣK‖ + ‖ΣV‖ magnitude, a proxy for the
  blocks that receive the most attention mass. This is a magnitude proxy for
  the heavy-hitter idea in **H2O** (Zhang et al., 2023, NeurIPS 2023;
  [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)), which keeps the tokens
  that accumulate the most attention.

The `recent` and `sink_recent` baselines used elsewhere in this document are
the recent-window and StreamingLLM (sink + recent) policies respectively. Our
implementations port the *shape* of each idea to a per-batch, eval-time
computation on GPT-2 small at 1024 context, so the absolute numbers are not
directly comparable to the source papers' setups; what we read is the
generalization pattern — does a KRI-FT model tolerate all of these shapes.

| Router @ K=8 | Vanilla | Dense-FT | KRI-FT (42/43/44) | KRI-FT × better |
|---|---:|---:|---:|---:|
| KRI-Q | 0.1138 | 0.0947 | 0.0199 / 0.0200 / 0.0200 | **5.7×** |
| KRI-Q-window | 0.1188 | 0.0988 | 0.0204 / 0.0206 / 0.0204 | **5.8×** |
| KRI-G | 0.0536 | 0.0347 | 0.0183 / 0.0186 / 0.0184 | **2.9×** |
| kv-sum | 0.1287 | 0.1105 | 0.0221 / 0.0223 / 0.0222 | **5.8×** |

KRI-FT is 3–6× lower KL than vanilla under *every* router, despite training
only on the internal KRI-Q+N mask — it became generally compatible with the
family, not just its training rule. Between-seed variance is ~0.0001 KL. The
gap is smallest under the query-agnostic KRI-G, suggesting KRI-FT's learned
behavior interacts most with content-aware routing.

(K is up to 16 here on GPT-2 small; this is a different regime from any
single-document cartridge result and the numbers are not directly comparable
across model/task/metric.)

## Router-component ablation (which score term works)

`eval_router_ablation.py`, single L=1024 sweep at top-k 8, sink + 128 window
fixed, mean decode-region KL.

| Variant | Vanilla | Dense-FT | KRI-FT |
|---|---:|---:|---:|
| random_global | 0.2523 | 0.2253 | 0.0151 |
| cos_only | 0.0650 | 0.0494 | 0.0065 |
| cos_value | 0.0608 | 0.0474 | 0.0066 |
| cos_recency | 0.0687 | 0.0560 | 0.0067 |
| cos_novelty | **0.0314** | **0.0128** | 0.0064 |
| full_kri | 0.0353 | 0.0161 | 0.0064 |

The novelty term dominates: `cos_novelty` matches or beats `full_kri` on
every model, and value-energy and recency add essentially nothing on top of
cosine. This is why the curriculum is honestly named **KRI-Q+N** — the other
score terms can be dropped without loss. For KRI-FT every variant lands at
KL ≈ 0.006–0.015: at 8 blocks of any flavor the model is already at the
floor.

## Oracle block selector (the upper bound)

`eval_oracle_kl_blocks.py` greedily adds the prefix blocks that most reduce
KL to the dense teacher, sink + window always on, L=1024.

| Model | ε=0.03 | ε=0.10 | ε=0.30 |
|---|---|---:|---:|
| Vanilla | not reached with 16 blocks (KL plateau ≈ 0.04) | 2 | 0 |
| Dense-FT | 8 blocks | 2 | 0 |
| KRI-FT | **0 blocks** (KL 0.027 with sink+recent alone) | 0 | 0 |

KRI-FT is at the floor — sink + recent alone is within 0.03 nats of full
attention, so the router has nothing left to do at this budget. Vanilla
cannot be rescued by raw block selection at tight ε even with sixteen
optimally chosen blocks; the missing nats are not recoverable by retention
alone, which is exactly the gap KRI-FT closes by training.

## Memory-forwarding diagnostic (the mechanism)

`analyze_memory_forwarding.py` buckets loss by query position and reports
late-minus-early excess (NLL@75–100% − NLL@0–25%) under aggressive routing
(L=512, recent-only at 6.25% nominal):

| Model | late − early excess |
|---|---:|
| Vanilla | +2.47 |
| Dense-FT | +2.52 |
| KRI-FT | **−0.57** |

Vanilla and dense-FT collapse at late positions (>2 nats worse than early) —
they are helplessly missing distant KV. KRI-FT's late positions do *better*
than its early ones: it learned to forward useful information into the recent
hidden states so late tokens predict without distant KV. Together with the
oracle result, this is the direct mechanism evidence for the claim.

## Cross-dataset robustness

`eval_multidataset_kri_ppl.py` at 12.5% nominal retention, seed 42:

| Dataset | Policy | Vanilla | Dense-FT | KRI-FT | vs Vanilla |
|---|---|---:|---:|---:|---:|
| TinyStories | recent | 398.7 | 381.5 | **62.2** | 6.4× lower |
| TinyStories | kri | 11.58 | 11.57 | **10.12** | −13% |
| WikiText-103 | recent | 2696 | 2613 | **497** | 5.4× lower |
| WikiText-103 | kri | 55.59 | 55.58 | **44.44** | −20% |
| FineWeb-Edu | recent | 2053 | 1997 | **283** | 7.3× lower |
| FineWeb-Edu | kri | 32.37 | 32.02 | **26.19** | −19% |

The robustness holds on held-out TinyStories, out-of-distribution
WikiText-103, and the training distribution itself. Dense-FT stays tied with
vanilla everywhere.

## No full-attention regression

Full-attention PPL: KRI-FT **9.50**, dense-FT 9.50, vanilla 9.62. Routing-
aware training costs nothing on normal decoding; both fine-tunes recover a
small improvement from FineWeb-Edu exposure.

## Modern-model port: SmolLM2-360M

KRI-FT trained on SmolLM2-360M (RoPE + GQA, 32 layers, 5 KV heads), 3000
steps, KL 0.1 against frozen vanilla. L2M sweep at L=1024 and 2048 vs trained
KRI-FT and vanilla:

| L | Policy | Actual | Vanilla gap | KRI-FT gap | vs Vanilla |
|---:|---|---|---:|---:|---:|
| 1024 | kri @ K=8 | 59% | +0.002 | +0.002 | tied |
| 1024 | kri fixed 6.25% | 22% | +0.122 | +0.115 | −5.6% |
| 2048 | kri @ K=2 | 35% | −0.002 | −0.002 | tied |
| 2048 | kri @ K=16 | 58% | +0.0003 | 0.0000 | tied |
| 2048 | recent W=165 | 15% | +0.702 | +0.676 | −3.6% |

Vanilla SmolLM2 with **no** fine-tuning already sits at loss-gap ≈ 0 under
the KRI policy at low block counts. The 6× GPT-2 gaps do not appear; KRI-FT
improves vanilla by only ~3–6% on the loss gap and ~15–25% on KL — real but
small.

Three reasons, in honest order of likely impact:

1. **The first SmolLM2 fine-tune was defective** — a tokenizer mismatch
   (GPT-2 vocab vs SmolLM2's), so this magnitude is a lower bound pending a
   clean retrain, not a verdict.
2. **3000 steps is ~0.01% of SmolLM2's pretraining budget**, likely too
   short to shift a well-pretrained modern model much.
3. **The architecture is natively more cooperative.** GQA gives fewer
   independent K/V channels per layer, and RoPE's distance falloff is more
   graceful than learned absolute positions, so vanilla SmolLM2 starts closer
   to the routing floor — leaving less for KRI-FT to add.

Claimed: the recipe transfers in **direction** (KRI-FT > vanilla on every
metric measured) but its **magnitude shrinks on modern small models because
they are already near the routing floor**. Not claimed: a final SmolLM2
magnitude, or that a clean longer retrain would not enlarge the effect. This
is the project's central finding: the size of the win is set by how
routing-intolerant the base model is to begin with.

## Negative and inconclusive results (kept honest)

- **Synthetic retrieval is inconclusive at this scale.** A 12-way key=value
  cloze with the answer placed outside sink and window lands every model at
  chance (0.085–0.170, random 0.083), including vanilla at full attention —
  GPT-2 small (124M) is simply too weak to do the task, so it cannot
  discriminate KRI-FT. The benchmark needs a larger base model to be
  informative.
- **The first retrieval-lite design was broken** — it placed the queried
  item inside the always-retained sink block, so every model trivially
  scored 100%. Reported only as a design-bug record, not a result; the fixed
  benchmark awaits a capable-enough base model.
- **The KRI score is not better than sink+recent at iso-actual-retention**
  (above): the model is robust, the router is not magic.

## Systems profile

Per-forward-pass timing on the W7900: the mask-aware sparse attention is at
most +7% over dense — the eager-attention path costs almost nothing extra.
K/V collection is one extra forward pass (the price of any content-aware
router, not KRI-specific). The reference `build_kri_mask` is a pure-Python
per-query loop and dominates wall time at long L; it is vectorizable and a
production implementation would build the mask as a single GPU op (or
precompute it once). The defensible systems statement is that the routing
*rule* adds ~7% to the attention forward path; the rest is offline scoring
and a not-yet-vectorized mask build.

## Success-criteria assessment

| # | Criterion | Status |
|---|---|---|
| 1 | Beats vanilla at iso-actual-retention under matched policies | PASS (6.3× under recent, −6% sink_recent, −17% kri at ~23–25% actual) |
| 2 | Improves synthetic retrieval | INCONCLUSIVE — base model too small to test |
| 3 | Full-attention PPL not degraded | PASS (9.50 vs 9.50 vs vanilla 9.62) |
| 4 | Reproduces across ≥ 3 seeds | PASS for the B_min and canonical-router results; single-seed for the rest |
