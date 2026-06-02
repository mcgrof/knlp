# gpt2-kri-ft — KRI-aware fine-tuning for KV-cache routing tolerance

KRI-FT is a training recipe. It fine-tunes a language model under a
content-aware KV-cache routing mask so that the resulting checkpoint keeps
its quality when a serving stack later throws away most of the KV cache at
inference. The router itself is training-free and unchanged; what we train
is the model, so that it stops depending on KV blocks a router would drop.

This repository is the self-contained R&D evaluation of that recipe: the
code to reproduce it, and an honest read of where it helps and where it does
not. The short version of the finding is at the top because it is the point
of the whole project:

> **KRI-FT helps models that lack native routing tolerance, and modern small
> models already mostly have it.** On GPT-2 small the win is large and
> reproducible — up to 8× fewer retained KV blocks for the same quality,
> 3–6× lower KL-to-full-cache under every routing rule we tested, no
> full-attention regression, across three seeds. On a modern small model
> (SmolLM2-360M, RoPE + grouped-query attention) the same recipe moves the
> needle only ~3–6%, because the untrained model is already close to the
> routing floor. KRI-FT is therefore a conditional tool, not a universal win,
> and its value is predicted by how routing-tolerant the base model already
> is.

## Table of contents

- [What KRI-FT is](#what-kri-ft-is)
- [Why it might matter](#why-it-might-matter)
- [The training curriculum (KRI-Q+N)](#the-training-curriculum-kri-qn)
- [What we found](#what-we-found)
- [The honest limit: modern models](#the-honest-limit-modern-models)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Scope and what is not claimed](#scope-and-what-is-not-claimed)

## What KRI-FT is

Long-context autoregressive decode is bottlenecked by KV-cache memory
traffic: every generated token reads the whole cache, so the fewer KV bytes
per token, the higher the arithmetic intensity and the more a server can
decode before it saturates memory bandwidth. KV-cache *routing* attacks this
by keeping only a subset of cache blocks per query — StreamingLLM (sink +
recent), H2O, SnapKV, Quest, and the KRI family are all routers in this
sense. They are training-free: they run on an off-the-shelf model and choose
which blocks attention may see.

The problem is that an off-the-shelf model was never trained to lose those
blocks, so aggressive routing degrades it. KRI-FT asks the inverse question:
if we *fine-tune* the model under the routing mask, does it learn to carry
the dropped information forward in its own hidden states, and thus tolerate
the router at inference?

KRI (Key Routing Index) is an existing family of training-free, content-aware
block-level routers — geometric (KRI-G, k-means on block-mean keys),
query-conditioned (KRI-Q / KRI-Q-window, cosine of the query to block-mean
keys), and others. KRI-FT is **not** a new router. It is the training-time
curriculum that makes a model cooperate with that family. A trained
checkpoint produced by this code is what we call a KRI-FT model, and it
exports back to a standard HuggingFace checkpoint with no custom inference
code.

## Why it might matter

If a model can be made to keep its quality at, say, one global KV block plus
a sink and a local window where it used to need eight, that is a direct
reduction in KV bytes per token — the exact lever that governs decode
throughput at long context and high concurrency. The systems value of
KRI-FT, where it applies, is that the byte reduction comes from the model
having internalized the information, not from a lossy cache approximation
the router has to gamble on.

## Program framing: what can we adopt without full retraining?

The goal of this work is not to invent a router. It is to look at the current
state of the art in KV-cache routing and ask **which ideas can be adopted
without paying for a full pretraining run.** The routers themselves
(Quest-style query criticality, StreamingLLM sinks, H2O heavy-hitters, InfLLM
representative blocks — attributed in [`RESULTS.md`](RESULTS.md)) are already
training-free, which is exactly why they are attractive. Where training-free is
not enough — where a model degrades under routing because it was never taught
to lose those blocks — the sensible compromise we are willing to evaluate is a
**light, parameter-efficient fine-tune (PEFT)** rather than a from-scratch
retrain. KRI-FT is that compromise: a short fine-tune that buys routing
tolerance, measured honestly against what the base model already had.

## The training curriculum (KRI-Q+N)

The mask operates on KV *blocks* (default 16 tokens). Block 0 is always kept
(the sink). For each query past a sampled prefill split, attention sees the
local window `[t−W, t]`, the sink, and the top-k prefix blocks chosen by a
score. The score combines the cosine of the query to the block's key
centroid (the canonical KRI-Q content-aware term) with a novelty term
against already-selected blocks. A router-component ablation (below) showed
the novelty term dominates and the value-energy and recency terms add
essentially nothing, so the curriculum is honestly named **KRI-Q+N**.

Training uses the same mask as inference. In the default `mixed` mode a
fraction of micro-batches use the routing mask and the rest use dense
attention, which keeps the model from drifting away from normal LM behavior;
an optional dense-teacher KL term keeps the student close to the frozen base.
The implementation is in `src/kri_mask.py` and `src/train_kri.py`.

## What we found

All GPT-2 numbers below are on GPT-2 small (124M), fine-tuned 5000 steps on
FineWeb-Edu against a frozen-vanilla KL teacher; `dense-FT` is the matched
control (identical data, KL, and step count, dense attention throughout) and
`vanilla` is untouched GPT-2. The two headline results are reproduced across
seeds 42/43/44; the rest are single-seed and labeled as such in
[`RESULTS.md`](RESULTS.md).

**Routing robustness generalizes across routers.** At a matched *actual*
retained-KV fraction, KRI-FT beats vanilla under every routing policy,
including ones it never trained against. Under `recent`-only at ~23% actual
retention KRI-FT is 60 PPL versus vanilla's 379 (6.3× lower); the win holds
on TinyStories, WikiText-103, and FineWeb-Edu alike. The matched dense-FT
control stays tied with vanilla under routing — so the operative variable is
the routing-aware training, not the extra data exposure.

**Up to 8× fewer global blocks for the same quality.** Sweeping the minimum
number of global KV blocks needed to stay within 0.03 nats/token of full
attention: KRI-FT needs 4 / 3 / 1 blocks at context 256 / 512 / 1024, where
vanilla and dense-FT need 16 / 14 / 8. At 1024 tokens that is an 8×
reduction. The honest reading is that KRI-FT reaches the fixed sink+recent
support floor at a *smaller* context than vanilla does — not an unbounded
scaling law (three points, the last one literally one block).

**3–6× lower KL under canonical KRI routers.** Run under reimplemented
canonical KRI-Q, KRI-Q-window, KRI-G, and a kv-sum router at K=8 blocks,
KRI-FT's KL-to-full-cache is 3–6× lower than vanilla's, despite being trained
only on the internal KRI-Q+N mask. It became generally compatible with the
router family, not just its training rule. This is the result that survives
the seed bar most cleanly (the gap is multiplicative; between-seed variance
is ~0.0001 KL).

**The mechanism is memory forwarding.** Bucketing loss by query position
under aggressive routing, vanilla and dense-FT degrade by >2 nats at late
positions (they are helplessly missing distant KV), while KRI-FT's late
positions do *better* than its early ones (excess −0.57 vs vanilla +2.47).
The model learned to push useful long-range information into its recent
hidden states. An oracle block-selection check agrees: KRI-FT is already
within 0.03 nats of full attention with zero global blocks, while vanilla
cannot reach that threshold even with sixteen optimally chosen blocks.

**No full-attention cost.** Full-attention perplexity is 9.50 for KRI-FT,
identical to the dense control and slightly better than vanilla's 9.62. The
routing-aware training does not tax normal decoding.

## The honest limit: modern models

Ported to SmolLM2-360M (RoPE + grouped-query attention, a modern Llama-style
architecture), the recipe transfers in *direction* — KRI-FT beats vanilla on
every metric — but the *magnitude* collapses to ~3–6% on the loss gap and
~15–25% on KL, nothing like the 6×/8× on GPT-2. The reason is the finding,
not a footnote: vanilla SmolLM2 is *already* near the routing floor at the
contexts we tested (loss gap ≈ 0 under the KRI policy at low block counts),
so there is little room left for KRI-FT's support-internalization to add.
Modern small models appear to ship with much of the routing tolerance that
KRI-FT installs into GPT-2.

This first SmolLM2 pass also carried a defective fine-tune (a tokenizer
mismatch, and training worth only ~0.01% of the model's pretraining budget),
so the small effect is a lower bound, not a verdict. But the structural point
stands and is the thesis of the project: **the size of the KRI-FT win is set
by how routing-intolerant the base model is to begin with.** That makes
KRI-FT a tool to reach for when a base model degrades badly under routing,
and a tool to skip when it does not — a decision a cheap routing-tolerance
probe can make before any training.

## Quick start

The code runs on a single GPU; it was developed on an AMD Radeon Pro W7900
(ROCm 6.4, PyTorch 2.9.1) using only `torch.matmul` + `softmax` attention, so
it has no CUDA-only dependencies (no FlashAttention/Triton/xFormers) and the
exported model is a plain HuggingFace checkpoint.

A few-minute smoke run (device check, unit tests, 200 training steps, a small
eval):

```bash
bash scripts/run_smoke.sh
```

The full fine-tune from `openai-community/gpt2`, then the routing eval matrix:

```bash
bash scripts/run_train_kri_gpt2.sh
bash scripts/run_eval.sh
```

Dependencies are in `requirements.txt` (PyTorch, transformers, datasets).

## Repository layout

```
README.md                  this file
RESULTS.md                 the full honest results, every phase + caveats
requirements.txt
src/
  model_gpt2_kri.py        GPT-2 small with [B,H,Tq,Tk] mask support
  model_smollm2_kri.py     SmolLM2 / Llama-family loader (the modern-model port)
  kri_mask.py              training-free KRI router + mask builder
  data.py                  tokenized packed-sequence streams (HF datasets)
  train_kri.py             training loop: dense / kri / mixed, optional KL
  eval_pruned_ppl.py       KV-routed perplexity across policies + budgets
  eval_effective_retention.py  actual retained-KV fraction (the honest x-axis)
  eval_synthetic_retrieval.py  key=value lookup scored by logprob
  eval_l2m_scaling.py      minimum-blocks-vs-context (B_min) sweep
  analyze_l2m_gamma.py     gamma fit over the B_min curve
  eval_router_ablation.py  which score terms matter (-> KRI-Q+N)
  eval_multidataset_kri_ppl.py  cross-dataset robustness
  eval_oracle_kl_blocks.py greedy oracle block selector (the upper bound)
  eval_canonical_kri.py    KRI-FT under canonical KRI-family routers
  canonical_kri.py         reference KRI-Q / KRI-Q-window / KRI-G / kv-sum
  probe_attention_entropy.py   native routing-tolerance probe
  analyze_memory_forwarding.py late-vs-early position diagnostic
  export_hf_gpt2.py        export back to a HuggingFace GPT-2 checkpoint
  utils.py                 seeding, device, dtype, logging
tests/
  test_weight_loading.py        custom dense forward == HF GPT-2
  test_kri_mask_causal.py       mask is strictly causal + budget-capped
  test_export_equivalence.py    round-trip custom <-> HF equivalence
scripts/
  run_smoke.sh / run_train_kri_gpt2.sh / run_eval.sh
```

## Scope and what is not claimed

This is GPT-2 small (124M, learned absolute positions, 1024-token cap) and a
360M modern model, on TinyStories / WikiText-103 / FineWeb-Edu, scored by
KV-routed perplexity and KL-to-full-cache. We do **not** claim a new router,
a long-context (>1024 on GPT-2) scaling law, a settled SmolLM2 magnitude, or
a downstream-task result — the synthetic-retrieval probe is too hard for a
124M model and lands at chance, and that is reported as such. The defensible
claim is the conditional one in the thesis above, backed by the
cross-router, cross-dataset, multi-seed evidence on GPT-2 and the honest
shrinkage on a modern base.

## Collaboration and commit protocol

This work follows the Multi-AI Collaboration Protocol (MACP) and its
Claude↔Codex MCP-receipt extension, documented at
[linux-kdevops/agents](https://github.com/linux-kdevops/agents). Commits
carry `Generated-by` / `Signed-off-by` trailers (and, when an MCP agent was
consulted, the `MCP-*` receipt and `Collab-*` plan-grading trailers) so that
who-did-what and the real token cost are reconstructable from git history
alone.
