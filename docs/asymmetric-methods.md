# Asymmetric KV: methods and results

The supported result is an 8K/batch-32 Qwen3-8B in-process throughput improvement with
FP16 keys and FP8 values. It holds for the two tested weight formats under
their matched controls. A separate dense 1K/batch-32 in-process test does not meet the confirmed
gain criteria. Actual AWQ HTTP quality fails its preservation gate. This page
preserves these outcomes and the conditions behind
quality, memory and throughput claims.

The separate [dense 1K and 8K HTTP studies](asymmetric-serving-methods.html)
report client-visible latency, throughput, failures and model qualification.

[Current guide](asymmetric-quantization.html) · [Install](asymmetric-install.md) ·
[Historical results and failure record](asymmetric-history.html)

## Precision and scaling reference

The completed two-model reference compares native K16/V16 with K16/V8 (A),
K8/V16 (R) and K8/V8 (S), each with fixed (F) and token/head (D) scales.
[Complete results and aggregate data](asymmetric-results.html#decision-reference)
separate reference quality, physical backing and actual-worker support.

Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` uses
FP16 weights and compute. Qwen/Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` uses BF16. Queries remain
native in every arm. Compressed components persist as E4M3FN codes and
FP32 scales; each layer temporarily reconstructs native attention operands.
No stored codes are requantized. Fixed scales use native cache-write operands
after normalization/RoPE, over 32 independent documents at 1K and 8K,
including 128 decode steps. Scale construction clamps the FP32 absolute
maximum at 1e-12, multiplies by the rounded FP32 reciprocal of 448, divides
operands in FP32, clips to ±448 and rounds E4M3FN to nearest/even.
Dynamic scaling applies this rule per fresh token/head vector.

Calibration, development and confirmation documents are disjoint. Confirmation
uses 64 paired documents at each context with 128 scored targets each, 200
retrieval instances per context and all 1,319 GSM8K test items. GSM8K is an
exposed standard benchmark. The original last-numeric-answer-label parser,
2,048-token cap and truncation semantics are retained. Correct labeled answers
at the cap count; errors, missing parses and truncations remain in the records.

Two certificate families each allocate alpha .025 across all 12 model/recipe
combinations (.025/12 per certificate). Every certificate requires native math
accuracy at least 50%, retrieval gross-loss upper bounds ≤5% at both lengths
and relative mean NLL upper bounds ≤1% at both lengths. Preservation adds a
math gross-loss upper bound ≤2%. Aggregate math instead requires the paired
net accuracy lower bound ≥−1 percentage point. Gross bounds are exact binomial;
net bounds invert Tango's matched-pair score with approximate coverage.
For binary endpoints, gross loss is the number of native-correct to
candidate-wrong transitions divided by all paired questions. Net change
uses gains minus losses over that same denominator.
NLL uses 100,000 paired document-bootstrap draws, seed 20261020, linear quantiles.
No certificate slot was removed or margin changed after results.

Direct A−R and A−S comparisons use eight contrasts across models and scale
policies within each of five endpoint families. Exact McNemar tests cover
binary endpoints; centered paired bootstrap tests cover document NLL. Holm
adjustment is per named endpoint. Descriptive paired 95% intervals differ
from certificate bounds. A nonsignificant difference does not establish
equivalence, and failure to certify does not automatically prove degradation.

The measured reference source is
[`782a4c5cd69b575b4736c14285477e306e176421`](https://github.com/mcgrof/knlp/commit/32952a7914efe546ac50cc085289886843d25430).
One redundant row-integrity comparison between a saved GPU FP32 mean and a
CPU FP64 reduction exceeded an absolute 1e-6 check by 5.47e-8. The documented
[reader correction](https://github.com/mcgrof/knlp/commit/45744906bed133e7f8f7f6a3d1639325b2d487a4)
uses the FP32 summation rounding bound gamma_n = n·u/(1−n·u), n=128 and
u=2^-24, while requiring finite nonnegative exact-FP32 losses and mean.
All 1,792 NLL rows pass. Saved means, samples, statistical calculations,
seeds and gates are unchanged; no GPU sample was rerun.

Physical backing includes FP32 scales and unique shared storage once.
A/R backing and allocation-size multisets match within each policy.
CUDA allocation rounding, temporary layer reconstruction and process peaks
remain separate. Reference execution time cannot rank fused serving kernels.

[The harness README](https://github.com/mcgrof/knlp/blob/b967e9c76f8f7de3baa2d6616ef294ce4f601627/tools/kv/decision/README.md)
contains preparation, source freezing, calibration, independent oracle,
confirmation and CPU evidence-analysis commands. An externally supplied aggregate result
records model/input/source and reader-result hashes. Redraw its figures with
Matplotlib and the public renderer:

```sh
python -m tools.kv.decision.figures \
  --data /path/to/experiment-data/asymmetric-decision-reference-20261022.json \
  --out redrawn-reference
```

This redraw reproduces the aggregate figures; it does not independently
re-audit private per-item captures. Optimized worker quality, graph correctness,
allocator capacity and useful serving performance require separate evidence.

## Actual-worker comparison

The worker comparison uses dense Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, FP16 weights and queries,
one H100 SXM80 and tensor parallel size one. Fixed per-layer/component
K16/V8 and dynamic per-token/head K8/V8 compare against native FP16
FlashInfer and FlashAttention-3. This compares complete deployment recipes:
the two candidates differ in both key precision and scale policy. The
separate seven-arm reference controls those factors within each policy.

The measured [harness](https://github.com/mcgrof/knlp/commit/c9d527d70cb092b396cb618a8479deb1c4af9ddb) is
`8dc2faed5485445342130ffd423911a505da2426`, [vLLM](https://github.com/mcgrof/vllm/commit/e5c8d87fd6bbf9cdedf266d160890599d938abb0) is
`e5c8d87fd6bbf9cdedf266d160890599d938abb0`, and [FlashInfer](https://github.com/mcgrof/flashinfer/commit/800a550f8b736e49f8cfa136b82bfaf6d9d32711) is
`800a550f8b736e49f8cfa136b82bfaf6d9d32711`. The runtime uses Torch
2.11.0+cu129, Transformers 5.5.3, tokenizers 0.22.2, Triton 3.6.0,
NumPy 1.26.4 and SciPy 1.17.1. The worker protocol SHA-256 is
`097d1c75e9cb091f5deca5d8b1853bc851f96d32058af408b435c1f37bf4a94f`.
Loaded shared objects, generated binaries, packages and worker identities
are retained with the raw evidence. No serving invocation ran in this panel.

The dynamic E4M3FN writer uses the reference's FP32 maximum floor of
1e-12, rounded FP32 reciprocal of 448, round-to-nearest FP32 division,
clipping and nearest-even FP8 conversion. Its opt-in mode leaves the
existing writer default unchanged. Actual-worker numerical checks cover
fresh prefill, decode, mixed traffic, changing physical slots and graph
replay. Stored codes and scales are compared with independent oracles.
Numerical qualification is separate from the full task-quality certificate.

Each worker collects 64 paired documents at each of 1K and 8K, 128 scored
targets per document, 200 retrieval instances per context and all 1,319
GSM8K test questions. Quality uses cohorts of 32 and the retained answer
parser and 2,048-token cap. Each quality objective allocates alpha 0.025
across four candidate/control certificates, so each uses alpha 0.00625.
No slot is removed if the symmetric backend has an engineering support gap.
Both native comparisons must pass. The preservation objective uses the
2% math gross-loss upper bound; aggregate math uses the paired net lower
bound of minus one percentage point. Both retain native accuracy at least
50%, both 5% retrieval gross-loss bounds and both 1% relative mean
prediction-loss bounds. Binary gross loss uses all paired questions as its
denominator. The matched-pair score intervals and 100,000 document bootstrap
draws use the reference methods. The serving route is aggregate math and
cannot switch to preservation after results.

The engine permits 96 sequences, 65,536 batched tokens and length 10,240,
with 16-token blocks. Prefix caching, chunked prefill and automatic KV-scale
calculation are disabled. Full decode graphs use capture sizes
1, 8, 32, 44, 45, 59, 60 and 96, with compilation mode zero. Eight server,
two client and two control CPUs are disjoint; the descriptor limit is at
least 8,192. These settings are common across arms.

Memory accounting separates a 17,000-block equal-slot audit from a common
40-GiB persistent-cache budget, including K, V and scales. The engine's
payload request reserves 36 MiB for allocator tails and 55,296 bytes for
scalar allocation allowance. Actual backing allocations determine whether
the budget passes. Scratch, staging, weights, graph pools and inactive
allocator pools remain separate from persistent-cache bytes. The graph
memory figure reported by the runtime is a rounded allocation-change
estimate already included in process memory; it must not be added again.

The common whole-device ceiling is 76 GiB, with a 68-GiB Torch reserved
ceiling. Driver-memory observations target a query start every 20 ms and
must leave no conservative query-bounded gap over 500 ms. Observed peaks
are sampled, not a claim of continuous measurement. Numerical checks have
an explicit memory-ceiling exemption; quality and serving do not. Sampling
coverage remains required during numerical checks. Complete captured
resource and statistical analyses are replayed independently on CPU.

### Serving was gated off

Fixed K16/V8 passes the complete preservation objective against both native
backends. Neither candidate establishes the aggregate-math certificate
against both: their math lower bounds against FlashInfer are -1.1131 and
-1.0630 percentage points, outside the -1-point margin. Both pass that
objective against FA3 alone. All retrieval and prediction-loss gates pass.
These are unestablished certificates, not proof of aggregate degradation.

The frozen serving route required aggregate math against both controls.
No native pilot, candidate timing or ten-block serving confirmation ran.
The objective was not switched after results. Physical cache capacity and
quality collection time therefore establish no new serving-speed benefit.
The earlier completed batch and HTTP performance studies remain separate.

[Worker results and memory](asymmetric-results.html#decision-worker) include
counts, adjusted bounds and exact physical allocations. Machine-readable
results are retained separately in the experiment archive.
[Reproduction commands](https://github.com/mcgrof/knlp/blob/c9d527d70cb092b396cb618a8479deb1c4af9ddb/tools/kv/decision/README.md)
cover source freezing, each actual-worker stage, equal-slot audits and
independent CPU analysis. The dependency's exact tested revision is public
on `2026-10-22-feature-fp8-reference-scales` in the vLLM fork.

## Current 8K results

Each study uses one H100 SXM 80 GB, FP16 compute, TP1, 22 CPU threads,
8,192 input tokens, 32 requests and 128 forced output tokens. Native FP16
KV uses FlashInfer. The candidate uses the same FlashInfer source and the
vLLM mixed fresh-prefill route. The preceding compact implementation is a
separate control. Workload, weights, scales, scheduling and KV byte pool
are matched within each study. EOS is ignored only in forced-output timing.

| Weights | KV pool per arm | Candidate / native | Candidate / preceding compact | Confirmation blocks |
| --- | ---: | ---: | ---: | ---: |
| Packed INT4 AWQ | 48 GiB | 1.034482642 | 1.048813951 | 10 |
| Dense FP16 | 40 GiB | 1.035457685 | 1.047289961 | 10 |

Ratios are medians of within-block complete-request output-token rates.
All ten fresh confirmation blocks favor the candidate in each comparison.
The one-sided exact sign p-value is 0.0009765625; Holm adjustment for the
predeclared two-comparison family gives 0.001953125. All four medians exceed
the frozen 1.01 screening threshold. This combines a positive sign test
with a practical median threshold; it is not a confidence-bound test of
at least 1% gain. Fixed pilot blocks are excluded. No extra confirmation
blocks are added after observing the results.

AWQ and dense block data, qualification receipts and profiles are retained in the experiment archive.

The request rate includes prefill, scheduling and decode. The source change
routes eligible fresh prompts inside a mixed decode/prefill batch through
bounded compact staging. At this shape, all 32 requests use that route;
the preceding implementation uses it for the first eight only. This is
not evidence of a faster decode kernel. The three-block batch-8 engineering
screens are separate, descriptive protection checks, not statistical
noninferiority results or new batch-8 performance claims.

## AWQ HTTP quality

The actual AWQ HTTP worker does not establish quality preservation. It scored
1,195/1,319 strict GSM8K answers correct, versus 1,198 for native FlashInfer
and 1,202 for native FA3. The candidate lost 20 and gained 17 answers against
FlashInfer, and lost 22 and gained 15 against FA3. One-sided exact 95%
loss-probability upper bounds are 2.196% and 2.373%, above the frozen 2% limit.
The denominator is all 1,319 paired questions; this is not a bound on net
accuracy change. Both paired net-change confidence intervals include zero.

This separate test uses the same Qwen3-8B AWQ revision, packed INT4 weights,
FP16 compute, original calibrated scales and a 48 GiB KV pool on H100 80 GB.
HTTP quality concurrency is 32, which is not a fixed GPU batch size.
NLL and retrieval preservation pass at 8K context, and all three arms match
serial in-process outputs for 32 prompts in both reference orders. Full
GSM8K uses natural question lengths and natural completion. Earlier batched
quality was not substituted for actual HTTP quality.

No AWQ HTTP performance confirmation was run. Excluded pilot timings are
not confirmed HTTP gains. The earlier batch-throughput results retain their
own execution scope. [Full report](https://github.com/mcgrof/knlp/blob/99df2dad01d7c7a88acd8404e211500eca73d2c8/docs/asymmetric-awq-http-quality.md) ·
Exact identities, intervals and evidence hashes.

## What “quality passes” means

Negative log likelihood (NLL) measures prediction loss. Each study uses
32 independent document windows and 128 scored cached-decode targets per
window. The one-sided 95% upper bound on candidate-minus-control NLL must
not exceed `ln(1.01) = 0.009950330853168092`. The document is the resampling
unit. Retrieval uses 100 frozen cases and requires at least 80% native
accuracy. GSM8K uses all 1,319 test questions and requires at least 50%
native accuracy.

A gross loss is a case the control answers correctly and the candidate
answers incorrectly. One-sided exact 95% gross-loss upper bounds must be
at most 5% for retrieval and 2% for GSM8K. Gains on other questions do not
cancel losses. Errors, truncations and strict parser failures stay in the
denominator. The fixed chat adaptation disables thinking and uses the
retained answer parser; these are not unmodified leaderboard scores.

| Weights / comparison to native | NLL upper bound | Retrieval, native → candidate | Retrieval losses / upper bound | GSM8K, native → candidate | GSM losses / upper bound |
| --- | ---: | --- | --- | --- | --- |
| AWQ | 0.00145939 | 91/100 → 91/100 | 1 / 4.65598% | 1,199/1,319 → 1,201/1,319 | 17 / 1.92699% |
| Dense FP16 | 0.00063625 | 96/100 → 96/100 | 0 / 2.95130% | 1,177/1,319 → 1,178/1,319 | 12 / 1.46988% |

Both also pass against preceding compact. These are bounded preservation
gates, not zero loss or mathematical equivalence. The production numerical
contract checks routing, storage, scale placement and outputs against the
accepted dispatch on the same quantized operands. The earlier ideal-FP32
objective still has retained diagnostic failures; it is not relabeled as
passing. See the [numerical reference and failure record](asymmetric-history.html#next-tests).
The separate small head-dimension-64 model remains quality-ineligible and
is not an installation option. BF16 and symmetric failures also remain
outside the advertised recipe.

## Memory and compatibility

FP16 K plus FP16 V requires four bytes per key/value element pair. FP16 K
plus E4M3 FP8 V requires three: 25% less persistent payload at equal token
slots. With the same byte pool, that is one third more slots. Dense 40 GiB
pool measurements have 291,264 native and 388,352 asymmetric token slots
per layer, with 42,948,624,384 actual backing bytes in each arm after page
rounding. Backing allocations are deduplicated by device and storage pointer.

The serving workspace includes 256 MiB of K/V staging, 32 MiB of direct
float scratch and 16 KiB of indices. FlashInfer wrapper buffers add further
storage: the retained workspace inventory totals 296.140625 MiB of unique
constant GPU storage and 8 MiB of pinned host memory. Per-plan metadata,
allocator overhead, graph pools, weights and other runtime allocations are
additional. The dense clean observation records 69,533,171,712 bytes of
peak reserved Torch memory and 70,341,623,808 bytes of sampled driver usage.
Instrumentation's temporary oracle workspace is excluded from that window.
Per-allocation receipts are retained in the experiment archive.

This is not a 25% total-HBM saving. No new maximum resident batch, capacity
sweep or service-latency endpoint was measured. Historical 44/58/59 batch
endpoints belong to their original implementations.

The supported geometry is Q32/KV8/D128, NHD layout, page16, FP16 compute,
SM90 and one worker/stream. Automatic prefix caching, chunked prefill,
sliding windows, attention sinks and soft caps are disabled. Each compact
prefill prompt must be whole and fresh, at most 8,192 tokens, with at most
4,096 gathered page occurrences across the eligible suffix. Unsupported
calls retain serving fallback; the installation qualification fails if its
eligible mixed prompts fall back. The wrapper rejects recipe drift before
loading a model. It does not claim that the underlying serving fork rejects
every possible unsupported configuration.

## Dense 1K result

The completed bridge uses dense FP16 Qwen3-8B, one H100 SXM 80 GB, 1,024
input tokens, batch 32, 128 forced outputs and a common 32 GiB KV budget.
Four excluded pilot blocks precede twelve fixed counterbalanced confirmation
blocks. Both native backends use FP16 KV. Native FlashInfer and actual native
FlashAttention version3 (FA3) are distinct controls.

| Current routing compared with | Median ratio | Change | Positive blocks | Holm-adjusted p | Combined screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Preceding compact | 0.998675270 | −0.1325% | 3/12 | 1 | Fail |
| Native FlashInfer | 1.009657397 | +0.9657% | 12/12 | 0.000732421875 | Fail |
| Native FA3 | 0.996497645 | −0.3502% | 1/12 | 1 | Fail |

The multiplicity family has three comparisons. None passes both the
Holm-adjusted 0.05 sign test and the 1.01 median-ratio screen. The positive
FlashInfer sign test misses the 1% screen; rounding must not turn it into
a pass. No equivalence or noninferiority conclusion follows. All production
and frozen task-quality gates pass. At this shorter shape, preceding compact
already admits all 32 whole prompts together, so the added mixed-suffix
eligibility is not needed. Compact and current routing produce identical
profile outputs for all 32 requests.

**Timing-audit limit:** the original strict reader fails. The frozen runner
turns off device observations while retaining host completion observations;
per-step worker CUDA events are empty. A disclosed supplemental CPU reader
checks the unchanged synchronized complete-request elapsed span, all 128
completion steps, exact request identities, 4,096 output tokens, prompt hashes,
scheduling counts and no preemption in every cell. Original and restored
reports match. The separately serialized start/end boundary timestamps were
not saved. This supports the scoped host-endpoint analysis above, not a full
original-protocol pass or per-step CUDA-duration claim. All failures remain
in the retained record.

The experiment archive retains complete 1K result data, including pilot and confirmation blocks.

## Reproduction

The [current installer](asymmetric-install.md) is a bounded route check.
It records its source separately from the original measured harnesses:

| Role | Immutable source |
| --- | --- |
| AWQ measured harness | [f91b3c3f](https://github.com/mcgrof/knlp/commit/f91b3c3f1b59090a40398ebf1a9b4a980c2215a0) |
| Dense measured harness | [a3996c8f](https://github.com/mcgrof/knlp/commit/1d2838e47ba6e98cc0b898af42c574d8940fc5fd) |
| Current vLLM candidate | [276305b2](https://github.com/mcgrof/vllm/commit/276305b29e0d11f95ada503ba76323ab24992e59) |
| Preceding compact vLLM | [26cd429b](https://github.com/mcgrof/vllm/commit/26cd429bac01d07909ed91e353f564d7aaa8d38f) |
| Native vLLM | [c21c0878](https://github.com/mcgrof/vllm/commit/c21c0878210695db7b108cdc285aaf0cf890190a) |
| Shared FlashInfer | [800a550f](https://github.com/mcgrof/flashinfer/commit/800a550f8b736e49f8cfa136b82bfaf6d9d32711) |
| 1K measured harness | [c02c4447](https://github.com/mcgrof/knlp/commit/f05a4b526b506dac58e13d71635a52fd4d8c5e9b) |

Supply the archived frozen inputs outside the checkout, following the
[input format documentation](https://github.com/mcgrof/knlp/tree/main/tools/kv/native/fixtures).
Their README gives dataset licenses and the exact export transformation.
Prompt IDs, token arrays, scoring tasks and effective calibration values are
unchanged. The dense scale export removes machine paths; both original and
export hashes are recorded. A new file wrapper does not constitute new
calibration. Never replace these values with unit scales.

The measured engine configurations remain in `measured-protocol.json` for
each format. The quick start writes a new protocol with the current harness
and exported-file hashes. To run the full original workloads, use the named
harness checkout, the native/compact/current source selection and the
[stage commands](https://github.com/mcgrof/knlp/blob/main/docs/decode-october-2026.md#qualify-then-time-fresh-processes), with
batch 32, length 8192 and the 48/40 GiB pools above. Freeze the adapted protocol
and task hashes before quality scoring. Use the published ten-block arm orders
from each data file for the 8K confirmation; do not use the older seven-block
batch 8 recipe. Derive timing qualification from the actual complete production,
NLL, retrieval and full GSM8K evidence, never from a fabricated success file.

A fresh execution is a replication: timing need not match the old values.
Generating new prompts, new scales, changing compute/backend/capture policy,
or combining an unqualified source pair requires the affected quality gates
again. Installation checks alone cannot qualify such a change.
