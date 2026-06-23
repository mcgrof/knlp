# Certified LM-head decode: read a quarter of the output head, losslessly

Every token a language model generates ends with the same enormous multiply. The
hidden state runs against the output embedding — a matrix with one row per
vocabulary token, well over a hundred thousand rows — to score every possible next
token, and the model takes the argmax. At batch-1 decode that single read *is* the
latency: for Qwen2.5-7B it is a 1.09 GB sweep of memory, redone every step, and
almost all of it is wasted because only a handful of tokens are ever plausible.

This page documents a lossless way to skip most of that read. The decoder returns
the **exact** token the dense head would return — the same argmax, bit for bit, not
an approximation — while touching about a quarter of the output head's bytes. It
gets there not by *predicting* which rows matter, which has a hard ceiling shown
below, but by *certifying* the answer: a cheap low-rank upper bound proves which
rows could possibly win, the rest are never read, and the result is mathematically
guaranteed to equal the dense argmax. The payoff is a 1.97× batch-1 decode-latency
win on a bandwidth-bound GPU, 1.85× on an H100 through a CUDA graph, and a ~5×
reduction in output-head traffic everywhere — with the answer provably unchanged.

## Contents

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
