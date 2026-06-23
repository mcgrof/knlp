# KRI-FT: training a model to tolerate KV-cache routing

Long-context decoding is bottlenecked by KV-cache memory traffic — every
generated token reads the whole cache, so fewer KV bytes per token means
higher arithmetic intensity and more decode throughput before a server
saturates memory bandwidth. KV-cache *routing* (StreamingLLM, H2O, SnapKV,
Quest, and the [KRI](kri.html) family) attacks this by keeping only a subset
of cache blocks per query. These routers are training-free: they run on an
off-the-shelf model and pick which blocks attention may see. The catch is
that the model was never trained to lose those blocks, so aggressive routing
degrades it.

KRI-FT asks the inverse question. Instead of building a better router, it
*fine-tunes the model* under a routing mask so the model learns to carry the
dropped information forward in its own hidden states. The router stays
training-free and unchanged; the model becomes the thing that tolerates it.
A KRI-FT checkpoint exports to a standard HuggingFace model with no custom
inference code.

## Status

Parked after a negative scaling result for the original recipe; fundamental
viability remains open pending one oracle-frontier experiment at 7B/8B.

The current evidence supports a narrow claim: the GPT-2 / SmolLM2 KRI-FT
recipe did not scale cleanly past 3B with the LoRA rank, adapted projections,
hard-mask training, and unrecalibrated KRI-D selector that line used. The
evidence does not support a universal "KRI-FT cannot help large models" —
recent literature cuts the other way:

- UNIQUE applies sparsity-aware LoRA on an 8B model and matches or slightly
  exceeds its dense baseline under fixed sparse attention.
- DejaVu demonstrates predictable contextual sparsity at OPT-175B scale.
- Sub-token routing retains 99.1% at 7B and 99.8% at 72B under aggressive
  KV budgets — routing tolerance can improve with scale.
- Diaz finds keys comparatively stable while values absorb most learned
  compression, which makes a K-heavy KRI-FT objective suspect.

The right next step is one decisive 7B/8B experiment with a soft-mask
training objective, the deployed K=16 mask, recalibrated KRI-D after FT, and
a value-side LoRA condition. See the Future R&D outlets table below.

## The GPT-2 finding (the small-model regime)

**KRI-FT helps models that lack native routing tolerance, and modern small
models already mostly have it.**

On GPT-2 small (124M) the win is large and reproduces across three seeds:

- **Up to 8× fewer global KV blocks** for the same quality — one block at
  1024-token context where vanilla needs eight, to stay within 0.03
  nats/token of full attention.
- **3–6× lower KL-to-full-cache** under every canonical KRI router (KRI-Q,
  KRI-Q-window, KRI-G, kv-sum), despite training on only one routing rule.
- **6.3× lower perplexity** than vanilla under naive recent-only routing at
  matched actual retention, holding across TinyStories, WikiText-103, and
  FineWeb-Edu.
- **No full-attention cost** (perplexity 9.50 vs 9.50 for the matched dense
  control, 9.62 vanilla).

The mechanism: under aggressive routing, vanilla degrades by more than two
nats at late token positions while KRI-FT's late positions do *better* than
its early ones. The model learned to forward long-range information into its
recent hidden states. A matched dense-trained control (same data, same steps,
dense attention) stays tied with vanilla under routing, which isolates the
routing-aware training as the cause rather than the extra data.

## The SmolLM2 limit (the small-modern-model regime)

Ported to SmolLM2-360M with RoPE and grouped-query attention, the recipe
transfers in direction but the magnitude collapses to a few percent. Vanilla
SmolLM2 is *already* near the routing floor (loss gap ≈ 0 under KRI routing
at low block counts), so there is little left for KRI-FT to internalize.
Modern small models appear to ship with much of the routing tolerance KRI-FT
installs into GPT-2.

That observation does not generalize to 7B+. At 7B, the dense-attention
quality is high enough and the per-block contribution structure rich enough
that routing tolerance is not free; the question is whether a different
training recipe (soft mask, value-side LoRA, recalibrated selector) can move
the oracle frontier.

## Future R&D outlets

Read this table as the queue, not a roadmap. Each row is a concrete next
experiment; status is one of *required*, *open*, *low probability*, *defer*.

| direction                              | status         | smallest decisive test                                                          |
|----------------------------------------|----------------|---------------------------------------------------------------------------------|
| Oracle routability frontier at 7B/8B   | **required**   | Base vs rank-64 QKVO under oracle K=16: does the frontier move?                  |
| LoRA rank scaling                      | open           | Rank 16 / 64 / 128, normalized by model width                                   |
| K-side vs value-side adaptation        | high priority  | QK vs VO vs QKVO factorial at 7B                                                |
| Hard mask vs soft-mask training        | high priority  | Same run with sigmoid boundary mask                                             |
| Selector recalibration                 | **required**   | Old pre-FT KRI-D prior vs recomputed KRI-D after FT                             |
| End-to-end routed serving              | open           | Run trained checkpoint through the [routing branch](routing.html) at K=16       |
| Budget robustness                      | open           | Train K=16; evaluate at K=8, 16, 32                                             |
| Cross-task generalization              | open           | Train on one long-context corpus, evaluate on two unseen                        |
| xa25 complementarity                   | open           | KRI-FT alone vs +xa25 at matched budget                                         |
| K16/V8 asymmetric KV interaction       | secondary      | Full precision vs asymmetric KV after FT                                        |
| Cross-model prior transfer             | low probability| Rank-correlation of block scores across model sizes                             |
| Linear-attention models                | defer          | No KV-block routing analogue; skip until a concrete state-pruning operator exists |

The **kill criterion**: if the oracle frontier at 7B/8B does not move by at
least 3 absolute points (matched task, matched mask, two seeds), kill the
KRI-FT line for that scale. If the oracle moves but recomputed KRI-D does
not, the training vehicle is alive and the selector is the failure.

## Reproduce

The code, full results, and every caveat are in the
[`gpt2-kri-ft/`](https://github.com/mcgrof/knlp/tree/main/gpt2-kri-ft)
directory. It runs on a single GPU with plain `matmul`+`softmax` attention
(no FlashAttention/Triton dependency) and exports a standard HuggingFace
checkpoint.

```bash
bash scripts/run_smoke.sh          # device check, tests, 200-step smoke
bash scripts/run_train_kri_gpt2.sh # full fine-tune from openai-community/gpt2
bash scripts/run_eval.sh           # the routing eval matrix
```

See the [interactive visualization](kri_ft_visualization.html) for the
routing-robustness curves, the block-budget reduction, the cross-router KL
gap, the memory-forwarding mechanism, and the GPT-2-vs-SmolLM2 narrowing.

## Related

- [KRI](kri.html) — the training-free routing-prior family that KRI-FT is
  designed to compose with.
- [routing](routing.html) — the serving substrate; KRI-FT checkpoints serve
  through it without custom inference.
- [SPF](spf.html) — separate scheduler-side prefetch experiment, parked.
