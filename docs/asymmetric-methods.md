# Asymmetric KV: methods and results

The supported result is an 8K/batch-32 Qwen3-8B in-process throughput improvement with
FP16 keys and FP8 values. It holds for the two tested weight formats under
their matched controls. A separate dense 1K test does not meet the confirmed
gain criteria. Actual AWQ HTTP quality fails its preservation gate. This page
preserves these outcomes and the conditions behind
quality, memory and throughput claims.

[Current guide](paper_memory_decode.html) · [Install](asymmetric-install.md) ·
[Historical results and failure record](asymmetric-history.html)

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

[AWQ block data and qualification](data/mixed-fresh-prefill-20261017.json) ·
[Dense block data, quality, physical storage and profiles](data/dense-fp16-routing-20261018.json)

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
own execution scope. [Full report](https://github.com/mcgrof/knlp/blob/dc4cce41c1541bac943aa3776f71c2e3bcfd3832/docs/asymmetric-awq-http-quality.md) ·
[Exact identities, intervals and evidence hashes](data/asymmetric-awq-http-quality-20261019.json).

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
See the downloadable dense data for per-allocation receipts.

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

[Complete 1K result data, including pilot and confirmation blocks](data/dense-1k-routing-20261019.json)

## Reproduction

The [current installer](asymmetric-install.md) is a bounded route check.
It records its source separately from the original measured harnesses:

| Role | Immutable source |
| --- | --- |
| AWQ measured harness | [f91b3c3f](https://github.com/mcgrof/knlp/commit/f91b3c3f1b59090a40398ebf1a9b4a980c2215a0) |
| Dense measured harness | [a3996c8f](https://github.com/mcgrof/knlp/commit/a3996c8fa73e4596a99f49fc52a3afa78e4b8c6d) |
| Current vLLM candidate | [276305b2](https://github.com/mcgrof/vllm/commit/276305b29e0d11f95ada503ba76323ab24992e59) |
| Preceding compact vLLM | [26cd429b](https://github.com/mcgrof/vllm/commit/26cd429bac01d07909ed91e353f564d7aaa8d38f) |
| Native vLLM | [c21c0878](https://github.com/mcgrof/vllm/commit/c21c0878210695db7b108cdc285aaf0cf890190a) |
| Shared FlashInfer | [800a550f](https://github.com/mcgrof/flashinfer/commit/800a550f8b736e49f8cfa136b82bfaf6d9d32711) |
| 1K measured harness | [c02c4447](https://github.com/mcgrof/knlp/commit/c02c4447e1c4ded8433a1ff5c6f676df0dd602ad) |

Download the [frozen input artifacts](https://github.com/mcgrof/knlp/tree/main/tools/kv/native/fixtures/current).
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
