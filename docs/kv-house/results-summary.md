# KV-House milestone 1 results

Milestone 1 asked whether fixed blocks of consecutive KV-cache
tokens carry cross-token (temporal) redundancy that an orthogonal
transform along the token axis can turn into compression wins at
matched bytes (the plan and lanes are in
[experiment-plan.md](experiment-plan.md), the prior art in
[prior-art.md](prior-art.md)). The answer is a clean split, and it
ends the orthogonal-transform wedge: **the temporal structure is
real and strong, but exploiting it by orthogonally mixing tokens
before quantization makes attention fidelity worse, not better.
The pre-registered recommendation is NO_GO.**

## Setup

One NVIDIA A100-SXM4-40GB pod ran the whole milestone in about 80
minutes. Models: Qwen/Qwen2.5-0.5B-Instruct (contexts 512/1024/2048,
four samples per class) and Qwen/Qwen2.5-7B-Instruct (contexts
1024/4096/8192, two samples per class), torch 2.7.0, transformers
5.15.1, bfloat16, `attn_implementation="sdpa"`. Text classes:
WikiText-103 prose, source code, and random-token controls, each
sample a disjoint slice of one per-class token stream. Per layer the
capture records post-RoPE K and V, strided post-RoPE Q probes, and
pre-RoPE K. Code at knlp commits `dad45968` (package),
`cb2cc706` (lanes), `85bb324d` (docs); thresholds were fixed in
`experiments/kv_house/run_milestone1.py` before any data existed.

## The seven questions

| Question | 0.5B | 7B |
|---|---|---|
| Q1 temporal structure vs controls | PASS | PASS |
| Q2 DC mode concentrates block energy | PASS (73-76%) | PASS (66-73%) |
| Q4 structure survives RoPE | PASS (3.2x) | PASS (2.3x) |
| Q5 best transform <= 0.8x identity attention error at uniform 4-bit | PASS (0.798, oracle) | FAIL (0.844, oracle) |
| Q6 transformed-attention identity | PASS (max err 2.5e-10) | PASS |
| Q7 sealed-block prefix contract + PIA | PASS | PASS |
| Recommendation | GO_WITH_MODIFIED_HYPOTHESIS | **NO_GO** |

Q1 is unambiguous with the corrected controls: on WikiText the
top-quarter singular-energy gap between real contiguous blocks and
scattered pseudo-blocks is 0.08-0.20 depending on tensor and block
size, roughly twice the random-token control, in 8/9 cells at both
scales. Two genuinely surprising positives: post-RoPE K carries
*more* temporal locality than pre-RoPE K (2.3-3.2x the locality
gap — the rotary rotation of nearby positions is similar, and the
shared K structure dominates), and a single DC mode holds two
thirds to three quarters of block energy.

## Why NO_GO despite real structure

The 7B attention-damage ranking at the scored uniform 4-bit
allocation (joint K+V replacement, model's own queries, bytes
include per-block transform metadata):

| Lane | attn-out rel err | bytes/token/side |
|---|---|---|
| pca_oracle (per-block KLT) | 0.075 | 104.5 |
| anchor_delta (CacheGen-style) | 0.087 | 67.2 |
| identity | 0.089 | 67.2 |
| poweriter_householder | 0.109 | 69.2 |
| dc_householder | 0.153 | 67.2 |
| corpus_pca | 0.154 | 67.2 |
| hadamard | 0.156 | 67.2 |
| dct2 | 0.161 | 67.2 |
| haar | 0.164 | 67.2 |
| random_householder | 0.168 | 67.2 |

Three facts close the case. First, every fixed orthogonal temporal
transform is roughly 1.7-1.9x *worse* than doing nothing, at
identical bytes — and front-loading bits onto leading modes makes
it worse still (identity 0.171 vs dc_householder 0.460 in the
unscored front-loaded family). This reproduces and generalizes the
negative result KVLinC reported for token-mixing Hadamard rotation:
it holds for DC-Householder, DCT-II, Haar, Hadamard, and random
reflectors alike, at two model scales, under both uniform and
per-mode bit ladders. Second, the per-block KLT oracle — the
ceiling — extracts only 16-20% attention-error reduction over
identity while shipping a per-block basis that costs +56% bytes,
so at true matched bytes even the oracle loses. The deployable
corpus KLT transfers none of the oracle's edge (0.154, tied with
the fixed bases): the useful basis is block-specific. Third, the
mechanism is visible in the data: K reconstruction error is
essentially flat across all lanes (0.192-0.197; orthogonal
transforms preserve quantization MSE) while attention error spreads
2.2x, and the lane with the *worst* reconstruction error
(anchor-delta, 0.223) has the second-*best* attention fidelity.
Attention does not punish the size of quantization noise; it
punishes noise that has been coherently mixed across tokens.
Per-token-local error (identity, anchor-delta) is benign; the same
error rotated across a block's tokens is not. Ranking codecs by
K/V MSE would have inverted every conclusion here.

The one lane that consistently edges identity at equal bytes is
CacheGen-style anchor-delta — which exploits the same temporal
redundancy while keeping quantization noise token-local. That
territory is occupied (CacheGen, DeltaKV), so it does not support
a novelty claim; it supports using CacheGen-style coding.

## What survives

The sealed-block machinery is model-validated and stands on its
own: deterministic, suffix-independent, randomly-accessible
artifacts; Prefix Integrity Analysis PASS with classification
SAFE_ONLY_WITH_CUSTOM_CONNECTOR at zero danger score; the
transformed-attention identity holds at 1e-10 on real caches. The
temporal-structure measurements (Q1/Q2/Q4), including the
RoPE-amplifies-K-locality observation and the two-null control
design (scattered pseudo-blocks for spectral metrics, within-block
shuffles for order metrics — a within-block permutation provably
cannot move a singular value), are reusable beyond this program.

## Decision

NO_GO for sequence-axis orthogonal transform coding of KV blocks.
The milestone-2 ideas this gate was protecting (heavy-hitter
temporal bases, attention-aware per-mode allocation, WY
representations) are not pursued: they all presuppose that
concentrating energy into modes helps after quantization, and the
measured mechanism — token-mixed quantization noise is what
attention punishes — cuts against the entire family, with the
per-block-basis metadata cost as a second independent blocker.
Raw measurements: `small/big_structure.csv`, `small/big_ablation.jsonl`,
and both verdict JSONs ship with the run artifacts.
