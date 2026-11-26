# Certified LM-head decode: an exact argmax from about half the output head

Every token a language model generates ends with the same enormous multiply. The
hidden state runs against the output embedding — a matrix with one row per
vocabulary token, well over a hundred thousand rows — to score every possible next
token, and the model takes the argmax. At batch-1 decode that single read *is* the
latency: for Qwen2.5-7B it is a 1.09 GB sweep of memory, redone every step, and
almost all of it is wasted because only a handful of tokens are ever plausible.

This page documents a lossless way to skip part of that read. The decoder returns
the **exact** token the dense head would return — the same argmax, bit for bit, not
an approximation. It gets there not by *predicting* which rows matter, which has a
hard ceiling shown below, but by *certifying* the answer: a cheap upper bound proves
which rows could possibly win, the rest are never read, and the result is
mathematically guaranteed to equal the dense argmax.

## Correction, November 2026: the byte saving on held-out text

The exactness result on this page stands. The byte and latency numbers in the
sections below do not describe a deployed decoder, and this section replaces them.

The original replay built the rank-1280 shadow basis from the same hidden states it
then scored: 600 to 1,500 positions, fewer than or close to the rank. A basis fitted
to so few states contains them almost exactly, so the residual term of the bound was
near zero and the certificate closed after reading about 2% to 7% of the rows. A
deployed shadow is built once, offline, and then meets hidden states it has not
seen. The held-out test builds the basis from 65,408 WikiText-2 train positions and
scores 8,176 WikiText-2 test positions. The in-sample column repeats the original
method on the first 1,500 test positions.

| model | shadow rank | head bytes read, held-out | head bytes read, in-sample | exact |
|---|---:|---:|---:|---:|
| Qwen2.5-7B (d=3584) | 1279 | 117.8% | 24.7% | 1.000 |
| Qwen2.5-7B | 2688 | 130.1% | 42.6% | 1.000 |
| Qwen2.5-7B | 3226 | 97.0% | 51.1% | 1.000 |
| Qwen2.5-7B | 3405 | 70.6% | 54.0% | 1.000 |
| Qwen2.5-7B | 3584 (full) | **56.0%** | 56.9% | 1.000 |
| Llama-3.2-3B (d=3072) | 1097 | 117.5% | 24.3% | 1.000 |
| Llama-3.2-3B | 2765 | 61.9% | 46.3% | 1.000 |
| Llama-3.2-3B | 2918 | 52.4% | 48.8% | 1.000 |
| Llama-3.2-3B | 3072 (full) | **51.4%** | 51.4% | 1.000 |

"Head bytes read" is the int8 shadow plus the exact rows fetched, as a fraction of
the dense 16-bit head. At the published rank the held-out certificate opens every
block, so it reads the whole head plus the shadow and costs more than the dense
head. The in-sample column reproduces the published 24.6%. Hidden states are not
low-rank enough for a rank-1280 basis to transfer: the saving appears only when the
basis keeps nearly every direction.

At full rank the method becomes an int8 screen of the whole head followed by an
exact fetch of the few blocks that can still hold the argmax (5.9% of rows on
Qwen2.5-7B, 1.3% on Llama-3.2-3B). That decoder is exact and reads 51% to 56% of the
dense head on text it has not seen. This is the deployable result. The 5× traffic
reduction and the 1.97× and 1.85× latency figures below were measured with the
in-sample shadow.

### Latency on held-out hidden states

On hidden states it has not seen, the full-rank certified head runs at 0.99 to 1.28
times the speed of the dense head.
[`scripts/fdec_cert_latency.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cert_latency.py)
builds the shadow from WikiText-2 train hidden states and times 1,000 WikiText-2
test positions, one token at a time, against the dense bf16 head on the same GPU.
The timed variant is the one-shot fused fetch: it opens a fixed share of the
vocabulary blocks in bound order, checks the certificate once, and falls back to the
dense head for a token that does not certify.

| model | GPU | dense head, mean | certified head, mean | speed, mean | speed, median | p95, dense → certified | blocks fetched | certified | head bytes read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | RTX 4090 | 1.166 ms | 1.087 ms | 1.07× | 1.23× | 1.17 → 2.11 ms | 12.5% | 88.0% | 73.1% |
| Llama-3.2-3B | RTX 4090 | 0.867 ms | 0.678 ms | 1.28× | 1.30× | 0.87 → 0.67 ms | 6.25% | 98.6% | 57.6% |
| Qwen3-8B | Radeon Pro W7900 | 2.002 ms | 1.978 ms | 1.01× | 1.16× | 2.01 → 3.68 ms | 12.5% | 87.5% | 73.5% |
| Qwen2.5-0.5B | RTX 4090 | 0.311 ms | 0.315 ms | 0.99× | 1.04× | 0.31 → 0.58 ms | 12.5% | 95.0% | 67.1% |

The table shows the fastest of three fetch budgets for each model. "Head bytes read"
counts the whole head for a token that fell back. Such a token pays for the screen,
the fetch and the dense product, so the 95th percentile is about 1.8 times the dense
head on three of the four models. The block-by-block loop reads the fewest bytes,
51% to 56% of the head, but its mean speed is 0.72 to 1.16 times the dense head
because each block costs a kernel launch.

The script repeats the in-sample setup on the same GPU. There the fused variant is
1.56 to 1.78 times faster in the mean on the three larger models, which reproduces
the size of the figures below. The difference between the two results is the shadow
and not the kernel.

Every variant is scored against the argmax of float32 logits. The loop certified
every token and returned that argmax for every token on all four models. The dense
bf16 head rounds its logits to 8 significant bits, which makes ties among the top
tokens: it returned the float32 argmax on 94.8% to 98.6% of the tokens. A bf16 dense
head is therefore not an exact reference, and an optimized path can score below
1.000 against it without an error in the certificate.

You can reproduce the table with
[`scripts/fdec_cert_heldout.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cert_heldout.py).
The certificate also stays exact when the model runs with int8 weights and an
asymmetric K16/V8 KV cache; see [fdec](fdec.md).

## Contents

- [Correction, November 2026](#correction-november-2026-the-byte-saving-on-held-out-text)
- [The output head is a wasteful read](#the-output-head-is-a-wasteful-read)
- [The structure is real, and it grows with scale](#the-structure-is-real-and-it-grows-with-scale)
- [Prediction hits a wall](#prediction-hits-a-wall)
- [The trick: certify, don't predict](#the-trick-certify-dont-predict)
- [Results: bytes, latency, and an honest Pareto switch](#results-bytes-latency-and-an-honest-pareto-switch)
- [The kernel](#the-kernel)
- [Where it stands](#where-it-stands)

## The output head is a wasteful read

The LM head computes `logits = W_U · h`, where `W_U` is the `[V, d]` output
embedding (V ≈ 152k rows for Qwen2.5-7B, d = 3584) and `h` is the decode hidden
state. The model takes the argmax. At batch-1 this is memory-bound: one token per
step, dominated by streaming all of `W_U` from HBM. The arithmetic is trivial; the
bytes are the cost. So the question is sharp and purely about bandwidth — find
`argmax_v (W_U[v] · h)` without reading all of `W_U`, and never be wrong.

## The structure is real, and it grows with scale

A cheap oracle test settles whether the opportunity exists at all. Partition the
vocabulary into 256 contiguous **blocks** of neighbouring token ids — directly
addressable with a single integer divide, portable across model families with no
per-model table. Then ask: if an oracle picked the best few blocks *perfectly*,
would the true next token still be inside them?

It would, overwhelmingly, and the coverage climbs with model scale. At a fixed
12.5% fetch budget — the best 32 of 256 blocks — the true next token is covered:

| model | vocab | dense ppl | covered @ 12.5% fetch | @ 25% fetch |
|---|---:|---:|---:|---:|
| Pythia-410M | 50K | 22.3 | 0.915 | 0.958 |
| Pythia-6.9B | 50K | 12.6 | 0.942 | 0.973 |
| Qwen2.5-0.5B | 152K | 16.6 | 0.949 | 0.980 |
| Llama-3.2-3B | 128K | 10.1 | 0.961 | 0.984 |
| Qwen2.5-7B | 152K | 8.5 | **0.970** | 0.989 |

Two same-tokenizer ladders separate scale from architecture: Pythia 410M→6.9B
lifts coverage 0.915→0.942, and Qwen 0.5B→7B lifts it 0.949→0.970. This is scale,
not a model quirk — larger models concentrate the plausible next token into fewer
neighbourhoods. The opportunity is real, and it improves as models grow.

## Prediction hits a wall

The obvious move is to train a router: a small head that reads the hidden state and
predicts which blocks to fetch. We tried it thoroughly. It runs into a wall worth
stating plainly, because the wall is what makes certification necessary rather than
merely elegant.

A learned router never reaches the oracle ceiling. The gap between what a router
predicts and what the oracle could pick is wide — roughly 0.77 predicted against
0.95-plus available — and, crucially, it is **not a router-size problem**. Making
the router bigger does not close it. Pushing harder with parameter-efficient
fine-tuning (LoRA adapters, base weights frozen) to train the hidden states
themselves to expose block identity does not close it either: an unsupervised
concentration penalty did nothing, and a supervised co-trained router moved the
deployable gap only from 0.188 to 0.140 while a fresh router on the adapted
representation barely budged (0.188 to 0.182) — so the small gain came from
co-adaptation, not from a genuinely more routable representation, and it was
confounded by ordinary domain adaptation on top.

The conclusion is consistent across every attempt: the predictor-to-oracle gap is a
representational property of the base model, not something a small adapter
relocates. The hidden state simply is not linearly separable by block to the degree
the oracle implies. Any predict-then-fetch scheme is therefore both lossy — it can
miss the true token — and capped — it cannot be trained to stop missing. Walk
around it entirely.

## The trick: certify, don't predict

The idea in one line: instead of guessing which rows could be the argmax, compute a
cheap, *provable upper bound* on every row's logit, and rule rows out. A row you can
prove cannot win is a row you never read.

Build a low-rank shadow of the head, once per model, offline:

- Fit an orthonormal basis `B` of shape `[d, r]` with `r ≈ 0.36·d` (1280 for the
  7B) by running PCA on a small sample of real decode hidden states. This choice is
  decisive: a basis fit to where hidden states actually live makes the part of `h`
  it cannot represent tiny.
- Build the **shadow head** `aq = quantize_int8(W_U · B)`, of shape `[V, r]` with a
  per-column scale. Because `r ≪ d` and int8 is half of bf16, the shadow is about
  18% of the dense head's bytes — a cheap screen you can afford to read in full.
- Record `delta[v] = ‖W_U[v] − B Bᵀ W_U[v]‖`, the norm of each row's component
  *outside* the basis — exactly what the shadow cannot see — plus the small slack
  from int8 rounding.

Now the elegant part. For a decode hidden state `h`, project it into the basis,
`q = Bᵀ h`, and measure the residual `ρ = ‖h − B q‖` — how much of `h` lives outside
the shadow's reach. Then for **every** vocabulary row `v`, this quantity is a
guaranteed upper bound on its true logit:

```
U_v  =  aq[v] · (scale ⊙ q)   +   ρ · delta[v]   +   slack·‖q‖
        └── cheap int8 GEMV ──┘   └─ Cauchy–Schwarz on the ─┘
            (the shadow score)       unseen component
```

The middle term is the whole trick. The shadow score captures the part of the logit
*inside* the basis exactly. Whatever lives *outside* the basis can contribute at
most `ρ · delta[v]`, by Cauchy–Schwarz, because `ρ` bounds the out-of-basis part of
`h` and `delta[v]` bounds the out-of-basis part of the row. Add the rounding slack
and `U_v` is a true ceiling: the real logit can never exceed it. And when `h` lives
mostly in the basis — which the PCA fit ensures — `ρ` is tiny, so the bound is
*tight*: it ranks the rows almost as well as the real logits, at a fraction of the
bytes.

The decode is then a short, self-terminating search:

1. Reduce the per-row bounds to a per-block bound `U_b = max_{v ∈ block b} U_v` —
   one number per block.
2. **Open** blocks in descending `U_b`. Opening a block reads its true rows from
   `W_U` and computes their real logits — the only place the dense head is ever
   touched. Track the best real logit so far, `m`.
3. **Stop the moment you can prove you are done.** When `m` exceeds the largest
   `U_b` among all still-unopened blocks, no unopened block can hold a logit above
   its own ceiling, which already sits below a value you have reached. The dense
   argmax is provably in the opened set, and it equals `m`. Return it.

The result is lossless by construction. If the bound were ever too loose for an
early stop, the search just opens more blocks — in the worst case all of them,
recovering the exact dense head. There is no approximation knob and no path to a
wrong answer; the only thing that varies is *how many* blocks you read before the
certificate closes. On Qwen2.5-7B that is, on average, about 2% of the vocabulary —
so a token costs the 18% shadow read plus a ~2% exact fetch and comes out
bit-identical to the dense head every time.

## Results: bytes, latency, and an honest Pareto switch

The numbers in this section come from the in-sample shadow described in the
correction above. They show what the kernel does when the certificate closes early;
they are not held-out results.

The byte reduction is a property of the bound and the model, identical on any GPU.
With the int8 shadow and an exact residual, across Qwen2.5 scales (600 real hidden
states, argmax match 1.000 at every size):

| model | shadow bytes | mean fetched | total head bytes | vs dense |
|---|---:|---:|---:|---:|
| 0.5B | 18.0% | 15.8% | 33.8% | 3.0× |
| 1.5B | 18.3% | 4.0% | 22.3% | 4.5× |
| 7B | 17.9% | 2.2% | **20.1%** | **5.0×** |
| 14B | 18.0% | 2.4% | 20.4% | 4.9× |

Scale helps twice over: larger, more low-rank hidden states make the true argmax
separate further from the field, so the certificate closes after fewer blocks — the
fetched fraction collapses from 15.8% at 0.5B to ~2.4% at 14B. Total output-head
traffic settles near a fifth of a dense bf16 head, losslessly.

Bytes become latency only once the data-dependent fetch is a real kernel. With a
fused Triton gather and a sync-free driver, batch-1 decode (W7900, dense head
~1.68 ms):

| model | dense | certified (mean) | mean speedup | lossless |
|---|---:|---:|---:|---:|
| 0.5B | 0.42 ms | 0.73 ms | 0.58× (loses) | 1.000 |
| 1.5B | 0.72 ms | 0.67 ms | 1.08× | 1.000 |
| 7B | 1.68 ms | 0.85 ms | **1.97×** | 1.000 |
| 14B | 2.41 ms | 1.37 ms | 1.76× | 0.998 |

The win grows with model size, strongest exactly where it matters — the big models
with the largest heads. The 0.5B loses: its `d=896` head is not low-rank enough for
a tight bound, and its dense read is already cheap. On an H100, whose dense head is
only ~0.37 ms, the eager path drowns in launch overhead; capture the entire batch-1
certified path into a single CUDA-graph replay and the win returns (7B 1.85×, 14B
1.92× on the graph).

Frame it as a **Pareto switch**, not a universal replacement. At batch-1 the
certified head wins on both GPUs. At low batch it keeps winning on a bandwidth-bound
GPU (1.24–1.41× to batch 32 on the W7900) but reaches only parity on the H100, whose
dense batched GEMM is already at its kernel floor and whose tensor cores starve on a
small-batch, data-dependent fetch. So: certified head below a measured per-model,
per-GPU batch threshold, dense head above it, with the byte reduction durable in
every regime. And one dose of realism on the full model — the output head is only
~7% of per-token decode traffic at 7B, so the head-only win dilutes to roughly +6%
on the full step. It is a real double-digit full-model gain only for small models
with large vocabularies, short context, and small batch.

## The kernel

The deployment path is a Triton kernel. Stage 1 computes the int8 shadow bound and
reduces it to per-block maxima entirely on-device. Stage 2 opens blocks through a
fused gather-GEMV that reads `W_U` rows directly through gathered pointers — no row
copy, no dtype cast — and certifies without ever materialising the full logit
vector. On the single-token path it matches a reference bound to 2.9e-6, the worst
bound violation is −2.73 (well under zero, so the bound holds), the argmax match is
1.0000, and it runs at 1.66× the dense GEMV while reading 25.3% of the bytes.

Two numerical rules are the difference between a win and a wash. Compute the
residual directly as `‖h − B q‖`, never as `sqrt(‖h‖² − ‖q‖²)` — subtracting two
near-equal numbers manufactures a false residual and a useless bound. And time the
whole certified path end to end, not an isolated kernel, or a microbenchmark will
report a win the full pipeline does not have.

## Where it stands

The certified decode is the shipped result on this line: lossless, hardware-
independent in its byte savings, a clear batch-1 and low-batch win on
bandwidth-bound hardware, an H100 batch-1 win through a CUDA graph, deployed as a
measured Pareto switch against a strong dense baseline. The routing detour around it
is the negative that justifies it — the structure the oracle sees is real, but no
learned predictor, sized up or fine-tuned, can exploit it without loss, so
certifying the answer beats guessing it. The one open lever is an H100-batch
kernel-efficiency question (saturating the shadow and fetch kernels at small batch),
which we judged not worth the spend while the H100 dense head is already at its
floor.

The kernel lives in
[`lm_head/kernel/`](https://github.com/mcgrof/knlp/tree/main/lm_head/kernel)
(`certdecode_kernel.py`, with `gen_artifact.py` to build the offline shadow). The
interactive version of this page is
[lm_head_decode_visualization.html](lm_head_decode_visualization.html).
