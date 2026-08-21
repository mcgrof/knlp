# KV-House experiment plan

This document defines the KV-House research program and its first
milestone gate. KV-House asks one question the existing KV-cache
compression literature leaves open (see
[prior-art.md](prior-art.md)): do fixed blocks of B consecutive KV
tokens carry enough cross-token (temporal) redundancy that an
orthogonal transform along the token axis — a Householder
reflector, DCT, Haar, or a KLT basis — concentrates it into a few
modes that can be quantized heterogeneously, at equal or better
attention fidelity per byte than coding tokens independently? The
program is ablation-driven: Householder transforms are a candidate,
not a conclusion, and the plan is built to kill the idea cheaply if
the redundancy is not there.

Two constraints are architectural, not aspirational. Every logical
token is retained (eviction methods break instruction following in
ways aggregate benchmarks miss). And sealed blocks are ordinary
prefix-cache objects: block j depends only on tokens up to its own
end, is immutable once sealed, is byte-identical under any suffix,
and decodes without touching any other block. The repository's
Prefix Integrity Analysis harness
([routing/prefix_integrity/](../../routing/prefix_integrity/README.md))
polices this contract; a candidate that fails ordinary prefix reuse
is not promoted regardless of its rate-distortion curve.

## Terms

A *block* is B consecutive tokens of one layer/head K or V stream,
shaped [B, d]. A *mode* is one row of the transformed block
Y = R @ X, where R is a [B, B] orthogonal matrix acting along the
token axis. An *allocation* assigns a bit width to each mode; the
transform itself is lossless, and all compression comes from the
allocation. Byte accounting always charges payload, per-mode
scales, per-block transform metadata, and headers.

## Code layout

The package is `kv_house/` (transforms, per-mode quantizer,
temporal statistics, attention-damage metrics, the sealed-block
codec, and a PIA adapter), with Gate-0 CPU tests under
`tests/kv_house/` and the measurement lanes under
`experiments/kv_house/` (capture, temporal structure, transform
ablation, milestone verdicts). The structural precedent is the
`symkv/` regime, with the transformed axis moved from heads to the
sequence.

## Transform lanes

| Lane | Role |
|---|---|
| identity | no temporal transform; the reference |
| anchor_delta | CacheGen-style anchor plus residuals; the prior-art baseline |
| dc_householder | one reflector mapping the block mean to mode 0 |
| random_householder | placebo reflector; must not help |
| hadamard | flat token rotation with no energy compaction; the KVLinC-rejected control |
| dct2 / haar | fixed signal-processing bases |
| pca_oracle | per-block KLT; the ceiling, charged for its basis metadata |
| corpus_pca | calibration-fitted KLT; the deployable adaptive basis |
| poweriter_householder | one data-driven reflector from the leading temporal direction |

The KVLinC result (token-mixing Hadamard with uniform bits hurts)
is treated as a standing prediction: if per-mode allocation does
not separate the energy-compacting lanes from the flat Hadamard
control, the temporal-transform hypothesis is in trouble.

## Hypotheses

| ID | Hypothesis | Kill test |
|---|---|---|
| H0 | KV has exploitable temporal, not just channel, redundancy | token-shuffled controls inside each block |
| H1 | DC Householder concentrates neighboring-token energy | mode-energy spectra before vs after |
| H2 | The concentration survives RoPE enough to matter for K | post-RoPE K vs pre-RoPE K diagnostic |
| H3 | Orthogonal transforms beat anchor-delta at equal bytes | CacheGen-style baseline lane |
| H4 | Householder is useful, not just "any transform works" | DCT, Haar, random-orthogonal lanes |
| H5 | Temporal KLT shows real headroom | per-block PCA oracle |
| H6 | Heavy-hitter-weighted modes preserve attention better than DC | milestone 2 |
| H7 | Prefix-internal statistics suffice; no future-query scores | milestone 2 |
| H8 | Head type predicts compression tolerance | milestone 2 |
| H9 | Per-head/layer budgets dominate uniform budgets | milestone 2 |
| H10 | Low-energy modes tolerate 2-4 bits while top modes keep 8-16 | precision sweep |
| H11 | Temporal transform composes with feature-axis quantization | milestone 2 factorial |
| H12 | Sealed blocks remain normal prefix-cache objects | PIA determinism/reuse/partial-tail tests |
| H13 | Transformed-domain attention avoids full reconstruction | identity unit tests now; kernels milestone 3 |
| H14 | Error stays bounded over long decode | milestone 2 generation-drift tests |
| H15 | Block granularity aligns with storage pages | read-geometry tests, milestone 2 |

## Milestone 1

Block sizes 8, 16, 32. A small instruct model plus a 7B-class
model, on coherent text (WikiText-103), structured text (source
code), and random-token controls, contexts 512-2048 (small) and
1024-8192 (7B). No GPU kernels. The seven questions and their
pre-registered thresholds live in
`experiments/kv_house/run_milestone1.py`; thresholds were fixed
before any data was collected and are not adjusted afterward. The
milestone ends in one of GO, GO_WITH_MODIFIED_HYPOTHESIS, or
NO_GO:

- NO_GO if temporal structure fails its shuffled control, or if
  even the per-block KLT oracle cannot beat identity by 20% on
  attention-output error at matched bytes — the temporal
  hypothesis would be dead, and that is a result.
- GO_WITH_MODIFIED_HYPOTHESIS if the oracle shows headroom but no
  fixed or cheaply-adaptive basis captures it — the program would
  pivot to adaptive-basis discovery.
- GO if a deployable lane (fixed basis or corpus KLT or one
  data-driven reflector) captures a material fraction of the
  oracle's gain.

Ranking uses attention-level damage, not raw reconstruction error:
each candidate block reconstruction is swapped into an otherwise
full-precision cache and probed with the model's own captured
queries (logit error restricted to the block, attention KL, top-1
and top-4 agreement, attention-output relative error), K-side,
V-side, and jointly.

## Later milestones

Milestone 2 adds heavy-hitter-derived temporal bases (as smooth
importance modes and small orthogonalized subspaces, never
eviction), attention-aware per-mode allocation in the AATC spirit,
head-type structure, end-to-end generation quality including
instruction-retention and generation-horizon drift, and the full
PIA codec run. Milestone 3, only if justified, covers
transformed-domain attention performance and multi-reflector
(compact WY) representations. Compact WY is gated on multiple
reflectors earning their place; one reflector needs only rank-one
updates.
