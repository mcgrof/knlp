# Cartridges at Scale: A LongHealth Reproduction

Cartridges at Scale (CAS, [arXiv:2606.04557](https://arxiv.org/abs/2606.04557))
extends the pre-trained KV-cache idea from one document to a whole collection:
train one cartridge per document, keep many resident at once, and serve any
subset without re-prefilling. Its central finding is a training-rule effect —
independently trained cartridges lose most of their accuracy when co-loaded, and
a change in how they are trained (mixed-visibility joint training) rescues that
accuracy back to near the uncompressed ceiling.

This page documents an independent reproduction on a frozen Qwen3-8B over the
LongHealth medical question-answering benchmark. The qualitative result and the
baselines reproduce cleanly. A single isolated cartridge reaches 0.65 at the
paper's training recipe (about 0.50 at the small-batch recipe most of this page
was measured with) where the paper reports 0.736; what follows is what we
measured, what that gap is not, and where the remaining distance lives.
Everything here is reproducible from the
harness in
[`research/cartridges_cas/`](https://github.com/mcgrof/knlp/tree/main/research/cartridges_cas).

## Table of contents

- [What a cartridge is, and what CAS adds](#what-cas-adds)
- [What reproduces](#what-reproduces)
- [The cartridge path is lossless](#cartridge-path-lossless)
- [Collapse and rescue](#collapse-rescue)
- [The single-cartridge gap](#single-cartridge-gap)
- [What our code does](#what-our-code-does)
- [Deltas against the public Cartridges implementation](#deltas)
- [Where the next gains come from](#next-gains)
- [Reproduce](#reproduce)
- [Serving and upstream enablement](#serving)

<a id="what-cas-adds"></a>
## What a cartridge is, and what CAS adds

A cartridge is a trained KV cache. Train it once per document offline, inject it
into the model's KV cache at serving time, and skip prefill. The base technique
([Cartridges](https://arxiv.org/abs/2508.17032)) trains that prefix by
self-study context distillation: a teacher answers synthetic questions about the
document, and the cartridge is trained to match the teacher's next-token
distribution. One document, one cartridge, one prefix at serving time.

CAS keeps that per-document cartridge and adds four things:

1. **Scale to a collection.** One cartridge per document, many resident on the
   GPU simultaneously, any subset composed at request time.
2. **A budget manager.** During training, keep a fixed number of cartridges on
   the GPU and rotate the active set every few steps so a large collection trains
   without exceeding memory.
3. **Mixed-visibility joint training.** Independently trained cartridges
   interfere when placed in one cache. CAS co-trains them with a per-sample
   visibility rule so they learn to coexist.
4. **Collapse and rescue** as the headline: isolated cartridges collapse toward
   the no-context floor when co-loaded; the joint rule prevents it.

The reproduction target is that story — split a collection into per-document
cartridges, observe the collapse when they are combined, and recover it with the
training-rule change — measured on Qwen3-8B and LongHealth.

<a id="what-reproduces"></a>
## What reproduces

All numbers below use one evaluation protocol: the LongHealth multiple-choice
task with option-text prompting, thinking enabled, temperature 0.6, a 2048-token
generation budget, fuzzy matching of the generated answer against the five
options, and a mean over at least three runs. The cartridge path and the
in-context baselines sample identically (top-k 20, top-p 0.95), so the paths are
directly comparable.

| Measurement | Ours | Paper |
|---|---|---|
| No-context baseline (question only) | 0.39 | 0.375 |
| Full document in context | 0.855 | 0.874 |
| Full-document KV loaded through the cartridge path | 0.86 | — |
| Single isolated cartridge, paper recipe | 0.65 | 0.736 |
| Single isolated cartridge, small-batch recipe | 0.50 | 0.736 |

The two ends of the range reproduce to within about two points: the model answers
LongHealth at 0.39 with no record and at 0.855 with the full record in context.
The compression path — a trained cartridge standing in for the record — is the
piece that does not yet match.

<a id="cartridge-path-lossless"></a>
## The cartridge path is lossless

Before attributing the gap to training, we ruled out the execution path. The
cartridge runs through a custom FlexAttention Qwen3 forward with a trainable KV
cache, not through the same code as the in-context baselines, so the 0.855
full-document anchor does not by itself prove the cartridge path is faithful.

The control: prefill the entire ~12K-token record normally, capture its exact
key/value tensors, write them into the cartridge format unchanged, and evaluate
that untrained "cartridge" through the cartridge path. It reaches **0.86**,
matching ordinary full-document inference (0.855) to within a point, with per-
patient scores tracking tightly. The cartridge path therefore has no positional,
RoPE, serialization, or loading loss. The ceiling for a perfect cartridge is
0.86, and the entire distance from 0.50 to 0.86 is a compression-and-training
question, not an execution one.

<a id="collapse-rescue"></a>
## Collapse and rescue

The qualitative CAS result reproduces. With five per-document cartridges:

| Arm | Alone (oracle) | Co-loaded | Co-load delta |
|---|---|---|---|
| Isolated cartridges | 0.58 | 0.38 | **−0.20** (collapse to the no-context floor) |
| Mixed-visibility joint | 0.44 | 0.46 | **+0.02** (no collapse) |

An isolated cartridge that scores 0.58 alone drops to 0.38 — the no-context floor
— when the other four cartridges share its cache. The mixed-visibility cartridge
starts lower alone (0.44 — in this reproduction a jointly trained cartridge pays
for coexistence when evaluated by itself) but does not degrade when co-loaded; its
co-load delta flips sign. That sign flip is the rescue. It mirrors the paper's
larger-N result, where isolated accuracy collapses from 0.736 toward 0.26 under
co-load while the joint rule holds it near 0.78. One divergence to note: the
paper's joint training raises the alone score (near 0.79, above its 0.736
isolated) where ours lowers it, the same single-cartridge quality shortfall
measured in the next section.

The mechanism is worth stating precisely, because a naive joint trainer produces
the opposite of a rescue. If every cartridge is physically resident during
training and only masked, the request tokens always sit at the same cache offset,
and the cartridges learn only the co-loaded geometry — they then fail when served
alone. A faithful implementation assembles each training example's cache
per-sample, so that most examples present the target cartridge alone (solo
geometry) and a minority present it alongside sampled distractors. Getting that
per-sample assembly right, and matching the per-cartridge training budget, is what
turns the inverted result into the sign flip above.

<a id="single-cartridge-gap"></a>
## The single-cartridge gap

A single isolated cartridge reaches about 0.50 under the strict scorer (0.58 at
its best), against the paper's 0.736. This is the open item, and it is a specific
one. It is not the evaluation protocol: the no-context and full-document anchors
land on the paper's values, and raising the generation budget to 8192 tokens does
not move the cartridge score. It is not the sampler: the cartridge path and the
baselines are matched. It is not the execution path: the lossless control above
reaches 0.86. It is not training length: cartridges trained for 80 epochs reach a
distillation loss of 0.017 and score the same as ones trained for a fraction of
that. It is not raw data volume, cartridge capacity, or the initialization.

The load-bearing observation is a gap between the training objective and free
generation. A cartridge trains to a distillation loss of about 0.035 — it matches
the teacher's next-token distribution well under teacher forcing — yet, decoding
on its own, it produces the correct answer only about 0.4–0.5 of the time. This
holds even when the cartridge is trained directly on paraphrases of the exact
questions it is later asked: it is no more accurate on those trained questions
(0.38) than on held-out ones (0.50). Minimizing the teacher-forced distillation
loss is not, by itself, producing a cartridge that free-generates the right
answer. The distance from 0.50 to the paper's 0.736 lives in that objective-to-
generation transfer, not in which questions the synthetic corpus asks.

We confirmed the last point directly. Reshaping the synthetic training
distribution does not close the gap:

| Synthesis strategy | Single-cartridge accuracy |
|---|---|
| Generic self-study | 0.50 |
| Forced hard question forms (negation, temporal, binding, comparison) | 0.48 |
| Hard-negative entity binding (same-record distractors, five-option) | 0.40 |
| Paraphrases of the benchmark's own questions (held-out) | 0.50 |

Each of these trains to the same low distillation loss and free-generates at the
same 0.4–0.5. A document-utility audit confirms the corpus is genuinely
document-dependent — the answers require the record, they are not generic clinical
prose — so the flatness is not a filler-data artifact.

What does move it is the optimizer regime. All of the numbers above come from a
small-batch regime: an effective batch of about 8, a peak learning rate of 0.02,
a few hundred steps. The paper trains at a global batch of 128 with a linear
schedule peaking at 0.1 for 80 epochs, with the cartridge sized to one twentieth
of the document (632 tokens for patient_02) and initialized from the document's
first tokens. Training the same patient_02 cartridge on the same data at that
recipe lifts it from 0.55 to **0.65 ± 0.05** (the same cartridge evaluated three
times), against the paper's 0.736. That is the only single change that has moved
this number, and it retires two earlier readings: "learning rate ruled out" and
"training length ruled out" were both taken at batch 8, where a peak of 0.1
diverges; at batch 128 it is the right rate. The distillation loss reaches
0.015 at this recipe, about where the small-batch recipe already got to at 80
epochs (0.017) without any score gain, so the lift does not come from a lower
fit; it comes from the optimization path that batch 128 at a peak of 0.1
takes to it. The remaining 0.09 is left to data scale — the paper synthesizes
roughly 40k self-study conversations per cartridge, this reproduction about
4.4k — and to the objective-to-generation transfer. A confirmation on four more patients at
the same recipe is in progress.

<a id="what-our-code-does"></a>
## What our code does

The harness wraps the public Cartridges package and drives it from a small set of
scripts. The pieces that matter for a reproduction:

**The cartridge path.** A FlexAttention Qwen3 forward with a trainable KV cache.
For the collection, all N cartridges live in one cache; each is hidden or revealed
per training sample by a reveal-vector block mask. A cartridge's identity is its
cache slot: cartridge id equals `kv_idx // KV_TOKENS`.

**The per-sample visibility mask.** The library's block mask attends iff the key
is a global sink or the key's sequence id equals the query's — it can express
visible-to-all or visible-to-one, but not a per-sample *random* reveal. The
faithful joint rule needs the latter, so the mask is replaced with a full-length
reveal-vector lookup: for each key position, look up whether its cartridge is
revealed for this sample.

**Two trainers.** The isolated trainer trains one cartridge with the target always
visible. The joint trainer holds all N cartridges in one cache and, per sample,
reveals the target alone 75% of the time and reveals the target plus a random
handful of distractor cartridges the other 25%; gradients flow only into the
revealed cartridges. This is the mixed-visibility rule that produces the rescue.
Both use the exact distillation loss — the negative teacher log-probability under
the cartridge, over the teacher's top-20 tokens per position.

**The synthesis pipeline.** A question-generator model reads a note and produces
diverse questions; an answerer model answers each with thinking enabled and
returns its top-20 token log-probabilities as the distillation targets. The
generated corpus is stored in the same on-disk format the trainer consumes.

**The evaluation harness.** The LongHealth multiple-choice protocol with the
option-text prompt, plus a matched-sampler decoder so the cartridge and the
baselines sample identically, plus the cache-path parity control that produces the
0.86 lossless number.

The reproduction values that a reader needs in one place:

| Quantity | Value |
|---|---|
| Base model | Qwen3-8B, frozen |
| Document | LongHealth patient record, ~12K tokens |
| Cartridge size | document tokens ÷ 20 ≈ 611 KV entries |
| Cartridge on disk | ~90 MB, bf16 (Qwen3-8B, GQA, 611 tokens) |
| Frozen prefix | one attention-sink token, load-bearing |
| Distillation targets | teacher top-20 token log-probabilities per position |
| Training loss reached | 0.015 (paper recipe); ~0.035, 0.017 at 80 epochs (small batch) |
| Mixed-visibility rate | target alone 75%, target + sampled distractors 25% |
| Serving TTFT | 757 ms → 81 ms median (677 ms/query saved) |
| Decode | ~35 tok/s, both paths |

<a id="deltas"></a>
## Deltas against the public Cartridges implementation

Three alignments to the paper matter for anyone reproducing from the public code,
because the stock trainer differs from the paper's description in ways that a
reproduction must reconcile.

**Distillation loss reduction.** The distillation loss is defined per token: sum
the negative teacher log-probability over the top-20 classes, then average over
tokens. Averaging over the sparse *entries* instead of over tokens under-scales
the gradient by roughly the top-k factor, which puts the effective learning rate
far below the paper's. The harness reduces per token.

**Learning-rate warmup.** The warmup must be applied before the first optimizer
step; applying it after means the first update fires at peak learning rate rather
than at the warmup floor.

**Packing window.** The teacher answers can be long. Packing to a short sequence
length truncates a meaningful fraction of teacher answers mid-thought, which
trains cartridges to begin reasoning without concluding. The packing window is
enlarged so complete teacher trajectories survive into training.

Two further deltas are execution details rather than the paper's recipe. The
public package writes FlexAttention as a raw kernel because RDNA3 cannot compile
it; on CUDA sm≥80 the compiled path restores about a 16× training speedup. And the
teacher top-k flattening keeps all K log-probabilities when a confident teacher's
cumulative mass never reaches the retention threshold, rather than silently
collapsing to a hard top-1 label.

None of these three alignments closes the single-cartridge gap on their own: with
all of them applied the cartridge trains cleanly to a distillation loss of 0.03
and still scores about 0.50. They are corrections a faithful reproduction needs,
not the missing ingredient.

<a id="next-gains"></a>
## Where the next gains come from

The gap is an objective-to-generation transfer problem, so the levers are the
parts of the paper's method that shape that transfer and are least specified in
the public materials:

- **The exact self-study distribution.** A larger, cross-family question-generator;
  sampling notes proportional to their length; generating diverse questions in
  batches per note rather than one at a time. The paper's ablations put the
  question-generation strategy among the largest quality levers, and it is the one
  our reconstruction is least sure of.
- **The faithful optimizer regime — done, and worth 0.10.** The paper trains at
  batch 128 with a linear learning-rate schedule peaking near 0.1; at batch 8 or
  16 that peak diverges on this stack, at batch 128 it lifts a single cartridge
  from 0.55 to 0.65. This is now the frozen baseline recipe, documented with the
  trainer in the
  [harness README](https://github.com/mcgrof/knlp/tree/main/research/cartridges_cas).
- **A possible free-generation term.** A cartridge that minimizes teacher-forced
  distillation loss but free-generates poorly points at an objective that covers
  positions or a consistency signal our reconstruction omits.

The reproduction has ruled out the cheap explanations, so the remaining distance
is worth spending real training on the items above, in that order.

A separate question is cost rather than quality. Every cartridge starts from the
same kind of state, the KV of its own document's first tokens, and nine hours of
training on one H100 moves it from there. If part of that motion is the same for
every document, it can be fitted once on trained cartridges and applied to the
start of every new one. The harness README describes the study and its tools:
an audit of where trained cartridges move relative to their starts, a
leave-one-document-out fit of the shared part, and held-out loss curves that ask
whether a corrected start reaches the baseline's loss levels earlier than the
run-to-run floor.

<a id="reproduce"></a>
## Reproduce

The harness lives in
[`research/cartridges_cas/`](https://github.com/mcgrof/knlp/tree/main/research/cartridges_cas).
It wraps the pinned public Cartridges package, applies the alignments above, and
runs the phases in order: synthesize the self-study corpora, train isolated
cartridges, run the combine-at-inference collapse evaluation, then train the
mixed-visibility cartridges and run the rescue evaluation. The evaluation harness
reports the no-context, full-document, and per-cartridge numbers with the matched
sampler.

Two settings decide whether the numbers mean anything, and both are easy to get
wrong:

- **Cartridge reconstruction.** A checkpoint stores trainable and frozen tensors.
  For an isolated cartridge the frozen part is the single attention-sink token —
  load it, because dropping it leaves a degenerate control prefix. For a joint
  cartridge the frozen part is the distractor cartridges — load the trainable part
  only, because baking the distractors in turns each oracle into a collapse.
- **The evaluation protocol.** Use the option-text prompt with thinking enabled.
  Before trusting any cartridge score, confirm the no-context baseline lands near
  0.39 and the full-document baseline near 0.855; if they do not, the protocol is
  wrong and every cartridge number is uninterpretable.

<a id="serving"></a>
## Serving and upstream enablement

The serving side is already built. A cartridge injects as a `KVConnectorBase_V1`
plugin into vLLM with no core modifications, and on this record the time-to-first-
token drops from 757 ms (re-prefilling the ~12K-token document every query) to
81 ms (loading the 611-token KV prefix), a 677 ms median saving per query at equal
decode throughput. The connector, the multi-cartridge router, GPU residency and
eviction, and an asymmetric-KV storage codec have all been implemented and tested
against vLLM and LMCache.

CAS makes that plumbing broadly useful. Retrieval-grounded serving does not have
to re-prefill retrieved documents on every request: a document collection can be
compiled once into cartridges and composed at request time, which is exactly the
multi-cartridge residency and routing the connector already handles. The next step
is a first-class serving path for it — cartridge storage, per-request composition,
and an asymmetric-KV offload tier.

## Related

- [Cartridges: pre-trained KV caches](https://mcgrof.github.io/knlp/cartridges_visualization.html)
  — the base technique, the prefill speedup measurements, and the vLLM connector.
- [Cartridges at Scale](https://arxiv.org/abs/2606.04557) — the paper this
  reproduces.
- [Learned Prefix Caches](https://mcgrof.github.io/knlp/learned_prefix_caches.html)
  — the family this belongs to: Cartridges, ReasonCACHE, and prefix induction as
  one shared learned-prefix object.
- [Cartridge economics calculator](https://kvcache.io/cartridge-economics.html)
  — build cost, serving cost, and quality accounting for cartridge deployments.
