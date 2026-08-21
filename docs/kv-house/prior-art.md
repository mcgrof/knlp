# KV-House prior art

This document maps the published work closest to KV-House so that its
novelty claims rest on verified reading rather than optimism. KV-House
is a storage codec for transformer KV caches: it applies orthogonal
transforms — Householder products, DCT, Haar, or KLT — along the
*temporal* (token/sequence) axis of sealed blocks of B contiguous
tokens, quantizes the resulting modes heterogeneously (more bits for
high-energy modes, fewer for the rest), and retains every token. Sealed
blocks are immutable once written, depend only on their own tokens
(never on the query or any suffix), and decode independently of one
another, which makes them compatible with hash-keyed prefix caching and
paged random access. The review below establishes which of these
ingredients are already occupied in the literature, which combination
appears to be unexplored, and which claims still need verification
before publication.

Two axis terms recur throughout. The *feature axis* (also channel or
head-dim axis) runs within a single token's key or value vector; the
*sequence axis* (or token axis) runs across tokens. A *mode* is one
coefficient of the temporal transform over a block — the DC mode of a
DCT block is the mean of the block's B token vectors, higher modes
capture progressively faster variation across the block.

Contents:

- [Method comparison](#method-comparison)
- [Closest neighbors](#closest-neighbors)
- [Novelty boundary](#novelty-boundary)

## Reading the table

*Query dep.* records whether the compressed artifact depends on the
query or decode trajectory. *Prefix reusable* asks whether the artifact
for a prefix is identical under any suffix — the property prefix caches
need. *Random access* asks whether a token or block can be used without
decoding others. *Reconstruction* records whether full-precision KV
must be materialized before attention (none / partial / full).

## Method comparison

| Method | Axis | Operation | Query dep. | Drops tokens | Prefix reusable | Random access | Reconstruction | RoPE | Per-layer/head adaptivity |
|---|---|---|---|---|---|---|---|---|---|
| [H2O](https://arxiv.org/abs/2306.14048) | token (selection) | eviction: heavy hitters + recent window | current-query | yes | no | yes (raw, scattered) | none | none; original positions kept | per-head scores, uniform budget |
| [Scissorhands](https://arxiv.org/abs/2305.17118) | token (selection) | eviction: pivotal-token retention | current-query | yes | no | yes (raw) | none (dequant if 4-bit combo) | none | per-head tracking, fixed budget |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | token (positional) | eviction: sinks + rolling window | none | yes | qualified (stream-position-bound) | yes | none | cache-relative re-indexing (central) | uniform everywhere |
| [Keyformer](https://arxiv.org/abs/2403.09054) | token (selection) | eviction: Gumbel-regularized scoring | current-query | yes | no | yes | none | PE-agnostic; no re-indexing | per-layer scores, uniform budget |
| [Q-Hitter](https://proceedings.mlsys.org/paper_files/paper/2024/hash/bbb7506579431a85861a05fff048d3e1-Abstract-Conference.html) | token select + per-token quant | eviction + low-bit quantization | current-query | yes | no | yes | partial (per-token dequant) | none | per-layer selection rule |
| [SnapKV](https://arxiv.org/abs/2404.14469) | token | eviction: observation-window top-k + pooling | current-query | yes | no | yes | none | post-RoPE keys unchanged | per-head selection, uniform budget |
| [PyramidKV](https://arxiv.org/abs/2406.02069) | layer budget + token | eviction: pyramid layer budgets | current-query | yes | no | yes | none | positions unchanged | per-layer budgets |
| [Ada-KV](https://arxiv.org/abs/2407.11550) | head budget + token | eviction: bound-guided head budgets | current-query | yes | no | yes | none | not discussed | per-head budgets in-layer |
| [HeadKV](https://arxiv.org/abs/2410.19258) | head budget + token | eviction: offline head profiling | current-query (tokens); offline (heads) | yes | no | yes | none | not discussed | global per-head budgets |
| [FastGen](https://arxiv.org/abs/2310.01801) | token, head-policy steered | eviction: hybrid per-head policies | prefix-local profiling | yes | qualified | yes | post-RoPE untouched | none | per-head policy choice |
| [DuoAttention](https://arxiv.org/abs/2410.10819) | head split + token | eviction: retrieval vs streaming heads | none (offline gate) | yes | qualified (windows slide) | yes | none | post-RoPE; sink lineage | binary per-head |
| [RazorAttention](https://arxiv.org/abs/2407.15891) | head split + token; mean pool | eviction + mean compensation token | none (offline) | yes | qualified (running mean mutates) | yes for kept raw KV | none | ALiBi analytic; RoPE heads empirical | static per-head split |
| [Quest](https://arxiv.org/abs/2406.10774) | token pages (16) | selection: top-K critical pages, no eviction | current-query (selection only) | no | yes | yes (self-contained pages) | none | post-RoPE metadata | dense first 2 layers; per-head kernels |
| [KVzip](https://arxiv.org/abs/2505.23416) | token (per-head pairs) | eviction by context-reconstruction score | prefix-local (query-agnostic) | yes | yes | yes | none (scoring is one-time) | post-RoPE positions kept | non-uniform per-head/layer |
| [KIVI](https://arxiv.org/abs/2402.02750) | none (quant grouping) | 2-bit asymmetric quantization | none | no | yes | yes (per token group) | none (fused dequant) | post-RoPE as stored | uniform (G=32, R=128) |
| [RotateKV](https://arxiv.org/abs/2501.16383) | feature (FWHT + reorder) | rotation + 2-bit quantization | none | no | yes | yes (group-wise) | full (inverse rotation) | pre-RoPE grouped-head rotation | per-layer reorder; grouped heads |
| [QuaRot](https://arxiv.org/abs/2404.00456) | feature (Hadamard) | rotation + 4-bit quantization | none | no | yes | yes | partial (in-kernel dequant) | post-RoPE Hadamard | none |
| [int4 on Apple Silicon](https://arxiv.org/abs/2605.05699) | feature (SRFT + rotation + scale) | rotation + int4 quantization | none | no | yes | yes (per group) | full (fused, ~25 ns/vec) | not addressed (post-RoPE) | per-layer, per-channel; K/V separate |
| [Eigen Attention](https://arxiv.org/abs/2408.05646) | feature | PCA/SVD low-rank | none | no | yes | yes | partial (keys up-projected for RoPE) | up-project before rotation | per-layer rank |
| [KVTC](https://arxiv.org/abs/2511.01815) | feature (global PCA) | transform coding + DP bits + DEFLATE | none | no | qualified (moving window boundary) | no (monolithic DEFLATE stream) | full | pre-RoPE (undone, reapplied) | global basis; per-component bits |
| [AATC](https://arxiv.org/abs/2608.14191) | feature (whitening SVD) | transform coding + reverse water-filling | none (offline calib) | no | qualified (recent-window state) | yes (per token) | full | latent pre-rotation; rotate after reconstruct | per-layer head-groups; one global budget |
| [CacheGen](https://arxiv.org/abs/2310.07240) | token (anchor + delta) | delta coding + tiered quant + arithmetic coding | none | no | yes | chunk-level (~1.5K tokens); anchor within groups | full | not addressed (position-bound) | 3 layer tiers; per-chunk levels |
| [DeltaKV](https://arxiv.org/abs/2602.08005) | token (long-range residuals) + learned proj | residual vs retrieved references | none (compress); query-driven decompress | no | yes | qualified (needs reference pool) | partial (mask-selected) | pre-RoPE, position-invariant | hybrid full-attention layers |
| [Probabilistic Language Tries](https://arxiv.org/abs/2604.15356) | token (predictive delta) | model-predictive delta + trie dedup | none | no | yes | no (strictly sequential decode) | full + extra model compute | positions carry zero entropy | per-position surprisal bits |
| [Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) | token (evaluation study) | analysis of 5 eviction policies | varies by policy | yes (studied methods) | mostly no | yes | none | not addressed | as in studied policies |
| [ContiguousKV](https://arxiv.org/abs/2601.13631) | none (chunk-granular I/O) | selective SSD loading + prefetch | current-query (selection only) | no (storage keeps all) | yes (raw artifact) | yes (chunk = I/O unit) | none | not discussed | layer Periods share indices |

The table splits into four families. Eviction and selection methods
(H2O through KVzip) compress by choosing tokens; they are KV-House's
accuracy-versus-bytes baselines, not its neighbors, because they
violate the every-token-retained contract. Feature-axis quantizers and
rotations (KIVI through Eigen Attention) act within each token's
vector and leave the sequence axis untouched; they compose with
KV-House rather than colliding with it. Feature-axis transform coders
(KVTC, AATC) share KV-House's information-theoretic framing but
decorrelate the wrong axis. Sequence-axis coders (CacheGen, DeltaKV,
the trie paper) exploit the same cross-token redundancy KV-House
targets, but by prediction rather than orthogonal energy compaction —
these are the genuine collisions and get detailed treatment below.

## Closest neighbors

### AATC: the token-independence idealization, stated and conceded

[AATC](https://arxiv.org/abs/2608.14191) is the sharpest theoretical
foil for KV-House. It proves that under a white-noise quantization
model the expected attention-output distortion factorizes into a token
factor times a channel factor, then spends its entire optimization
budget on the channel factor: per-layer whitening transforms plus
attention-aware reverse water-filling over channels, reaching
near-lossless quality at ~5.8x. The token factor is idealized away by
its Assumption 1 (quantization errors "mutually independent across
tokens"), and Remark 2 concedes the point directly: "cross-token
independence is an idealization: neighboring KV representations are
correlated, so per-token errors are not strictly independent.
Exploiting this dependence, however, would require a computationally
intractable token varying bit allocation." KV-House's answer is that a
fixed orthogonal transform over a sealed block exploits exactly that
dependence *without* per-token bit allocation — the correlation moves
into the mode spectrum, and the bit allocation runs over B modes, once,
offline. AATC's attention-aware channel weighting is complementary and
could stack on KV-House's per-mode coefficients.

### CacheGen: sequence-axis coding by anchor and delta

[CacheGen](https://arxiv.org/abs/2310.07240) (SIGCOMM 2024) is the
canonical prior art for coding stored KV across contiguous tokens. It
splits the context into groups of ten contiguous tokens, encodes the
first token of each group independently as an anchor, and records
per-token deltas against that anchor, motivated by measured token-wise
locality (delta variance 2.4-2.9x lower than raw values). Chunks of
~1.5K tokens are independently decodable and reusable across requests —
sealed-artifact semantics KV-House shares. The difference is the
coding class: anchor+delta is first-order predictive (DPCM-style)
coding that captures cross-token redundancy only relative to one
reference token, with uniform bins per layer tier. An orthogonal block
transform concentrates the same redundancy into an energy-ranked mode
spectrum, admits per-mode bit allocation, and decodes any block by a
single inverse rotation with no anchor dependency chain. CacheGen is
therefore both the closest systems precedent and the head-to-head
baseline a transform coder must beat at matched bitrate.

### Householder on the wrong axis: the int4 Apple Silicon ablation

[When Quantization Is Free](https://arxiv.org/abs/2605.05699) is the
nearest-sounding prior art by name — it ablates learned Householder
rotations for KV-cache quantization and finds a product of k = d/2 =
128 reflectors "effectively lossless" at 4 bits with d = 256. But the
paper's own construction places every transform (SRFT, the Cayley
rotation, the Householder variant, the per-coordinate scale) inside a
single token's d-dimensional vector: the reflectors act on channel
coordinates, the sequence dimension is explicitly preserved untouched.
The Householder overlap with KV-House is nominal — same operator
family, orthogonal axis. Two of its findings still transfer: Householder
products are a practical, learnable parameterization of orthogonal
transforms in a fused kernel, and a fixed random transform base can act
as a regularizer over fully learned rotations. Both observations apply
to KV-House's temporal transforms, and the feature-axis pipeline
composes with a sequence-axis block transform rather than competing
with it.

### Quest: sealed pages without coding

[Quest](https://arxiv.org/abs/2406.10774) organizes the cache into
fixed 16-token pages carrying channelwise min/max key metadata computed
only from each page's own tokens, then selects the top-K critical pages
per decode step by bounding attention scores against the live query.
Nothing is evicted and nothing is coded — Quest cuts bandwidth, not
bytes. Its relevance is structural: the pages are exactly KV-House's
sealed-block geometry (fixed size, self-contained, suffix-independent,
fetched by index), demonstrated at production quality with dedicated
kernels. The two compose naturally: keep Quest-style pre-transform
summary metadata per sealed block for selection, then decode only the
selected KV-House blocks — a workload that exploits precisely the block
random access KV-House guarantees and that a monolithic-stream codec
like KVTC cannot serve.

### Pitfalls of KV cache compression: the cost of dropping tokens

[The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231)
is motivating evidence rather than a competitor. Evaluating five
eviction policies on Llama3.1-8B and Qwen2.5-14B, it shows that
token-dropping compression silently breaks instruction following —
specific instructions degrade fastest and can be ignored entirely,
worst in multi-instruction settings — and that system-prompt
confidentiality behavior fails in ways governed by the compression
method, instruction ordering, and eviction bias. This is the citable
case for KV-House's every-token-retained contract: a codec that
degrades gracefully in reconstruction error cannot delete an
instruction, whereas an eviction policy can and does. It also frames
the fair comparison: eviction methods remain the baselines to beat at
matched byte budgets, with instruction-following benchmarks included
alongside perplexity.

### ContiguousKV: the block is the I/O unit

[ContiguousKV](https://arxiv.org/abs/2601.13631) accelerates re-prefill
from SSD-resident prefix caches by aligning the semantic pruning unit
with the storage I/O unit: a fixed contiguous chunk of ~16 tokens is
simultaneously the importance-scoring granule and the atomic read, so
fetching one important chunk causes zero read amplification. No bytes
are transformed or quantized — the contribution is scheduling. Its
granularity-alignment thesis is independent support for KV-House's
sealed fixed-B blocks: the same argument that makes a contiguous token
chunk the right unit of selective load makes it the right unit of
transform coding, and a KV-House compressed block can serve directly as
the ContiguousChunk payload, stacking byte reduction under its
prefetch pipeline. Its observation that important-chunk indices are
highly similar across adjacent layers may also inform per-layer
treatment of block transforms.

### Sequential KV tries: predictive coding of the same redundancy

[Sequential KV Cache Compression via Probabilistic Language Tries](https://arxiv.org/abs/2604.15356)
is the closest conceptual neighbor in print. It claims the same
territory — sequence-axis coding of KV with every token retained and
prefix-stable artifacts — via model-predictive delta coding: store the
quantized residual between each actual KV vector and the model's
expected KV under its own next-token distribution, with bits allocated
by token surprisal. Its information-theoretic claim, that per-token KV
entropy is bounded by token surprisal rather than per-component
magnitude, is the strongest published theoretical support for the
existence of the cross-token redundancy KV-House exploits. The
differences are decisive on the systems side: decoding is strictly
sequential (position i requires re-predicting from all prior
positions), reconstruction needs extra forward-pass-like model compute,
and the paper is theory-only — its headline ratios are Shannon-limit
bounds, not measurements. Predictive coding and transform coding are
the two classical routes to the same redundancy; KV-House takes the
transform route precisely because it preserves random-accessible
blocks and cheap linear decode.

## Novelty boundary

### Already-known ideas

Feature-axis orthogonal transforms before KV quantization are settled
territory: [QuaRot](https://arxiv.org/abs/2404.00456),
[RotateKV](https://arxiv.org/abs/2501.16383),
[OSCAR](https://arxiv.org/abs/2605.17757),
[KVLinC](https://arxiv.org/abs/2510.05373),
[NOVA-KV](https://arxiv.org/abs/2608.04074),
[MatryoshkaKV](https://arxiv.org/abs/2410.14731), the
[Apple Silicon int4 work](https://arxiv.org/abs/2605.05699), and
[Codec-Gauge](https://arxiv.org/abs/2607.20538) all rotate or project
within each token's vector, several stating explicitly that they never
mix across tokens. Transform coding as a framing for KV storage —
orthogonal decorrelation, bit allocation, entropy coding — is likewise
occupied on the feature axis by [KVTC](https://arxiv.org/abs/2511.01815)
and [AATC](https://arxiv.org/abs/2608.14191), including heterogeneous
bit allocation over transform components (DP over principal components;
reverse water-filling over channels).

Sequence-axis coding of KV exists in two established forms. Predictive:
[CacheGen](https://arxiv.org/abs/2310.07240)'s anchor+delta groups,
[DeltaKV](https://arxiv.org/abs/2602.08005)'s long-range retrieval
residuals, and the [trie paper](https://arxiv.org/abs/2604.15356)'s
model-predictive deltas. Lossy-spectral:
[FreqKV](https://arxiv.org/abs/2505.00570) applies a DCT along the
sequence dimension and truncates high-frequency components with
iterative recompression and light fine-tuning, and
[FAEDKV](https://arxiv.org/abs/2507.20030) keeps a frequency subset of
an infinite-window DFT — both shrink or entangle the token inventory
rather than preserving it. Token-axis DCT also occurs *incidentally*
inside video-codec approaches:
[VcLLM/LLM.265](https://arxiv.org/abs/2407.00467) runs HEVC
intra-prediction plus integer-DCT transform coding over KV tensors
chunked into frames (the 2D DCT blocks span the token axis), and the
[GPU-native video codec KV-reuse system](https://arxiv.org/abs/2602.09725)
maps tokens to the frame axis but runs lossless H.265, explicitly
skipping DCT and quantization.

Finally, several structural ingredients of KV-House are individually
established: fixed self-contained token blocks with per-block metadata
(Quest pages, ContiguousKV chunks, CacheGen chunks); a rank-1
cross-token mean as a compression primitive
([RazorAttention](https://arxiv.org/abs/2407.15891)'s compensation
token is exactly a DC-mode-only code of a dropped span); cross-token
shared quantization statistics ([KIVI](https://arxiv.org/abs/2402.02750)
shares scales over 32-token groups); and the query-independent,
prefix-reusable artifact contract itself
([KVzip](https://arxiv.org/abs/2505.23416) holds it via eviction,
CacheGen and KVTC via coding).

### Combinations that appear unexplored

No published work was found that applies an exactly orthogonal
transform (Householder, DCT, Haar, or KLT) along the token axis of
sealed fixed-size KV blocks while preserving every token invertibly.
FreqKV, the only deliberate sequence-axis orthogonal transform on a
decoder KV cache, truncates the spectrum and mutates earlier cache
state; VcLLM's token-axis DCT is incidental to codec frame chunking and
applied to intra-prediction residuals; KVLinC evaluated token-mixing
Hadamard pre-multiplication and rejected it. The specific conjunction —
deliberate temporal orthogonal transform, all tokens recoverable,
sealed immutable blocks — appears unoccupied.

Heterogeneous per-mode quantization of *sequence-axis* transform
coefficients also appears unexplored. Bit allocation over transform
components exists only on the feature axis (KVTC's per-component DP,
AATC's channel water-filling) or as coarse layer tiers (CacheGen). The
combination of temporal energy compaction with a per-mode bit ladder —
the core rate-distortion move of image and audio codecs, applied across
tokens — has no published instance.

Third, no lossy cross-token transform has been combined with
prefix-cache sealed-block semantics (immutable, suffix-independent,
randomly accessible by block). CacheGen's chunks come closest but use
delta coding at streaming granularity;
[FibQuant](https://arxiv.org/abs/2605.11478) argues explicitly that
random access forces per-token fixed-address codes and therefore
*avoids* token mixing — evidence the combination is recognized as hard
and left unoccupied rather than already taken. Relatedly, the
composition of feature-axis rotation (per token) with a sequence-axis
block transform (across the block's tokens) is unexplored, though each
half exists separately and nothing prevents stacking them.

### Claims requiring more literature verification

The narrow claim "no one has applied a token-axis orthogonal transform
to KV caches" is falsified and must not be used. KVLinC's Observation 1
ablated Hadamard pre-multiplication of K and V — a genuine token-axis
orthogonal rotation — and found it "almost always yields worse
performance" because mixing tokens before quantization amplifies noise.
KV-House therefore carries a burden beyond novelty: it must show why a
block-local transform with per-mode heterogeneous bit allocation
succeeds where whole-sequence Hadamard mixing with uniform quantization
failed. That is an empirical question the ablation grid has to answer
head-on, and the write-up must cite and engage the negative result.

The distinction between deliberate and incidental token-axis transform
coding is thin as a defense against VcLLM/LLM.265. A reviewer can
reasonably read HEVC's integer DCT over KV frame chunks as prior art
for the transform-coding core, even though the paper attributes the
benefit to outlier smoothing and has no block-alignment, prefix-cache,
or per-mode-allocation design. The safe position is to claim the
sealed-block contract plus exact orthogonality plus mode-wise
allocation as the contribution, not the bare existence of a token-axis
transform, and to include VcLLM as a measured baseline if feasible.

DeltaKV's locality measurement needs direct engagement: it reports that
over 60% of each token's most similar KV neighbors lie more than 16
positions away, and reads this as evidence that effective cross-token
compression must use global retrieval. If true at KV-House's block
sizes, it bounds how much redundancy a local B-token transform can
concentrate. This is less a literature question than a measurement to
reproduce — per-block energy-compaction curves against B directly test
it — but the claim must be addressed rather than ignored.

Residual search risk remains at the edges: unindexed workshop papers,
preprints newer than this survey, and the patent literature (not
searched at all) could still collide. Two corroborating signals cap
the risk but do not eliminate it — FreqKV's own claim (as of early
2026) to be the first frequency-domain KV compression for decoder-only
models, and FibQuant's argument that token mixing conflicts with
random access — both consistent with the sealed-block orthogonal
combination being open. A patent search and a final pass over mid-2026
preprints should precede any public first-claim language.
