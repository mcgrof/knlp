# Dense FP16 routing transfer and the separate D64 model check

Measured October 18, 2026.

Dense Qwen3-8B confirms a complete-request serving gain at 8K input and
batch 32: **3.55% over native FP16 KV** and **4.73% over preceding compact**.
Both comparisons pass the frozen production, task-quality and speed rules.
Ten fresh confirmation blocks give median ratios 1.035457684541 and
1.047289961362. Each comparison wins all ten blocks; exact one-sided
sign p=0.0009765625, or 0.001953125 after Holm correction for two tests.
The three-block batch-8 engineering screen passes at median 1.002069198
against its 0.99 threshold. It is not statistical noninferiority or a new
batch-8 gain.

This transfers the earlier AWQ routing result to dense FP16 weights.
It qualifies the tested model, backend and workload. It does not establish
BF16 support, a new resident-capacity endpoint or a gain for every model.
The separate D64 repair is not part of this speed stack. The earlier
[AWQ result and synthetic D64 qualification](decode-routing-d64-followup-2026.md)
and the historical eight-model table retain their original scopes.

## Exact dense recipe

The model is `Qwen/Qwen3-8B`, revision and tokenizer
`b968826d9c46dd6066d109eabc6255188de91218`. Actual loaded parameters and
compute are FP16. Dense operators are checked affirmatively: 852 CPU
matrix-multiply calls join to 852 positive-duration CUDA kernels.
There is no packed INT4 dispatch in these observations.

| Role | Immutable source |
| --- | --- |
| Native vLLM | `c21c0878210695db7b108cdc285aaf0cf890190a` |
| Preceding compact vLLM | `26cd429bac01d07909ed91e353f564d7aaa8d38f` |
| Fresh-prefill routing vLLM | `276305b29e0d11f95ada503ba76323ab24992e59` |
| FlashInfer in all three arms | `800a550f8b736e49f8cfa136b82bfaf6d9d32711` |
| Model/quality/timing core and profile helpers | `a3996c8fa73e4596a99f49fc52a3afa78e4b8c6d` |

Native uses **FlashInfer with FP16 keys and values**. Both asymmetric arms
store FP16 keys and E4M3 FP8 values. Compact prefill gathers into a bounded
FP16 workspace for native FA3 prefill; decode uses paged mixed FA2.
The new vLLM route admits fresh prefills inside mixed prefill/decode
batches. Its FlashInfer source is unchanged. Use the
[measured routing branch](https://github.com/mcgrof/vllm/tree/2026-10-17-feature-mixed-fresh-prefill)
at the exact candidate above; do not combine it with the separate D64 fix
and call that combined stack qualified.

Every arm uses 8,192 actual input tokens, 32 requests and 128 forced
output tokens, one H100 SXM 80 GB, tensor parallel size one, page size 16,
NHD layout and the same **40 GiB KV byte pool**. There are 22 threads,
at most 96 sequences and 65,536 batched tokens. Automatic prefix caching
and chunked prefill are off. `FULL_DECODE_ONLY` captures
1/8/32/44/45/59/60/96. The B8 screen has a separately frozen 32 GiB pool.
The effective scale SHA-256 is
`267dc6420febfb8fd6201ba60dd0eeae6b16315e93090337a82005fd9bf4f656`.
The frozen input manifest SHA-256 is
`26f3249f56612cba18fc94ddf5d28273804c3426983bc81374052e0ce68c9e2b`.

[All arm rates, orders, ratios, quality bounds, physical tensor layouts
and component measurements](data/dense-fp16-routing-20261018.json)
are available as sanitized JSON. Use the pinned harness's
[`tools/kv/native`](https://github.com/mcgrof/knlp/tree/a3996c8fa73e4596a99f49fc52a3afa78e4b8c6d/tools/kv/native)
commands with these exact source and input identities. Model execution
must use immutable revisions. Later reader fixes do not change this
measured source pin. The measured environment uses Python 3.12.11,
Torch 2.11.0+cu129, CUDA runtime 12.9 and nvcc 12.9.86. The pinned
harness retains the package constraints and setup command. Reproduction
requires the frozen inputs above; preparing different prompts is a new
replication, not an exact replay of these observations.

## Routing and quality

All 360 production checks pass: 36 layers, five actual fresh-admission
cohorts 8/7/7/7/3 and two fixed prompt rotations. The independent oracle
reads raw worker order, query offsets, sequence lengths and block tables
before candidate slicing. Encoded bytes, casts, same-dispatch outputs and
log-normalizers, placement, finiteness and repeatability are checked.
Four fixed mapping/placement faults are detected. Actual 32-row decode
graph inputs, positions and lengths change, with at least 32 calls and
three distinct input hashes per correctness pass. All 32 fresh requests
use compact prefill; preceding compact uses it only for the first eight.

This is equivalence to the accepted operation. The unchanged ideal-FP32
diagnostic still fails 9 of 360 checks for both candidate and trusted
operation. Full-fallback output and log-normalizer diagnostics pass all
360 under their original tolerances. The strict ideal-arithmetic objective
remains failed. These fixed correlated checks are not an error-rate survey.

Fresh candidate NLL on 32 documents at 8K, retrieval on 100 cases at 8K,
and all 1,319 GSM8K questions pass against native and preceding compact
separately. Complete dense 8K/B32 raw controls are reused only after
independent identity and raw-score replay. This verifies source math,
model/tokenizer, scales, prompt IDs, parser and actual B32 scheduler route.
The 1K and AWQ panels cannot substitute for these controls.

| Measure | Versus native | Versus preceding compact | Frozen rule |
| --- | ---: | ---: | --- |
| Paired NLL, upper 95% bound | 0.000636251 | 0 | At most ln(1.01) |
| Retrieval gross-loss upper bound | 2.951% | 2.951% | At most 5%; control accuracy at least 80% |
| GSM8K gross-loss upper bound | 1.470% | 0.227% | At most 2%; control accuracy at least 50% |

Retrieval is 96/100 in all arms. GSM8K is 1178/1319 for candidate,
1177 native and 1178 preceding compact. Candidate has 12 losses and
13 gains against native, with no correctness changes against preceding
compact. New correct answers do not cancel gross losses. No calibration,
prompt, parser, floor or margin changes. GSM quality retains its original
generation policy rather than the throughput test's forced outputs.

## Timing and profiles

Three pilot orders are NAC, ACN, CNA, where N/A/C mean native, preceding
compact and candidate. Their median C/A is 1.048441279 and passes 1.01.
Pilots are excluded from confirmation. The ten fresh confirmation orders
are NAC, ACN, CNA, CAN, ANC, NCA, NAC, ACN, CNA, CAN. Each claimed contrast
requires median at least 1.01 and a one-sided exact sign test with Holm
correction at 0.05. Ties count as nonpositive. Every arm has a full warmup
and synchronized complete-request timer, including prefill and decode.
All ten blocks are retained. There are no selective retries or added
trials. Earlier incomplete attempts are not pooled or used as replacements.

One matched full-request profile per arm follows primary timing, with
one warmup and actual CUDA launch correlation. The following milliseconds
sum operation-level CUDA kernel unions. Gather is included with prefill;
they are not cohort elapsed time or a critical path.

| Fresh / decode requests | Native prefill ms | Preceding gather + prefill ms | Candidate gather + prefill ms |
| --- | ---: | ---: | ---: |
| 8 / 0 | 332.586 | 348.059 | 347.110 |
| 7 / 8 | 295.781 | 501.631 | 308.100 |
| 7 / 15 | 293.401 | 498.722 | 309.020 |
| 7 / 22 | 292.349 | 497.382 | 307.031 |
| 3 / 29 | 125.768 | 204.584 | 131.437 |

Correlated host launch API cost, decode, residual layer work and residual
batch work are reported separately in the JSON. Each arm has 1,452 unmatched
kernels; those are explicitly counted and excluded from duration aggregates.
The profiles show removal of much of the mixed-admission prefill overhead.
The separate request trials establish the serving gain; the profiles do
not isolate its complete critical-path cause.

## Physical memory at a fixed byte pool

All arms have 42,948,624,384 deduplicated K/V backing bytes across 36 layers.
The small difference from 40 GiB is block-rounding slack. Native has 18,204
physical blocks and 291,264 logical KV tokens per layer. Both asymmetric
arms have 24,272 blocks and 388,352 tokens. These are physical slot counts,
not a measured maximum resident batch. Equal-token K16/V8 payload saves
25% before metadata. At the same 40 GiB pool, the pool stays fixed and
has more slots; total GPU memory does not fall by 25%.

| Per-arm component | Native | Preceding compact | Candidate |
| --- | ---: | ---: | ---: |
| K payload bytes | 21,474,312,192 | 28,632,416,256 | 28,632,416,256 |
| V payload bytes | 21,474,312,192 | 14,316,208,128 | 14,316,208,128 |
| Scale backing bytes | 432 | 432 | 432 |
| Compact payload bytes | 0 | 268,435,456 | 268,435,456 |
| Compact float workspace bytes | 0 | 33,554,432 | 33,554,432 |
| Compact index bytes | 0 | 16,384 | 16,384 |
| Allocator peak reserved bytes | 69,231,181,824 | 69,533,171,712 | 69,533,171,712 |
| Driver sampled maximum bytes, 1 Hz | 70,050,119,680 | 70,352,109,568 | 70,352,109,568 |

One shared 256 MiB compact payload remains after oracle release, with no
full FP16 value shadow. The 32 MiB float workspace and 16 KiB index buffer
are separate. A zero compact workspace for native does not mean native
FlashInfer has no internal workspace. Physical shapes, strides, offsets,
aliases and deduplicated backing bytes are in the JSON. Metadata CPU/CUDA
view spans are separate: some pairs share host addresses and must not be
added as independent GPU allocation.

Clean uninstrumented generation has a 64.7578125 GiB allocator peak and
19 driver samples at 500 ms reaching 65.5107422 GiB, within predeclared
68/72 GiB limits. The table's per-arm driver samples are at 1 Hz across
each complete profile command, including initialization and warmup.
Neither sample series is a continuous peak measurement. No capacity sweep
or new resident-batch endpoint was measured.

Independent CPU readers reproduce the qualification from original and
fresh archive-restored raw data. A versioned reader correction replaces
an obsolete 48 GiB AWQ assumption with the already frozen 40 GiB dense
pool check. The original rejection is retained; no measured source,
data, threshold, timer or order changes. Negative cases verify that wrong
pools, incomplete evidence and failed gates cannot promote the result.
The current public helper accepts an explicit `expected["pool_bytes"]`
for new readers; the existing AWQ default remains 48 GiB. GPU execution
remains pinned to the measured source above.

## Separate D64 model outcome

The repaired paged path **unblocks production, but the model remains
quality-ineligible** under the original retrieval rules. This does not
show demonstrated quality harm. It fails to establish preservation, and
the native control itself is below the required accuracy floor.

The model is `Qwen/Qwen2.5-0.5B-Instruct`, revision and tokenizer
`7ae557604adf67be50417f59c2c2f167def9a775`, with 24 layers, 14 query
heads, two KV heads and head dimension 64. It uses the original 1K input,
batch-32, 8 GiB pool, FP16 model/query/key and E4M3 FP8 value recipe.
Original calibration, prompts, parser, NLL rules and model-specific floors
are unchanged. Compact prefill is disabled. The routing change above is
excluded from this separate experiment.

| Role | vLLM | FlashInfer |
| --- | --- | --- |
| Native FP16 K/V, FA3 | `c21c0878210695db7b108cdc285aaf0cf890190a` | `800a550f8b736e49f8cfa136b82bfaf6d9d32711` |
| Original mixed path | `1a4dcc01ab09cd337499c0138c732eb741218b57` | `800a550f8b736e49f8cfa136b82bfaf6d9d32711` |
| Repaired mixed path | `1a4dcc01ab09cd337499c0138c732eb741218b57` | `72b18b4f1eef45a718de640568d7a0144cf3ab89` |

The exact original remapped-page fixture reproduces its failure at maximum
error 0.04670727 and mean 0.01275503. Repaired decode and prefill pass the
unchanged maximum/mean limits 0.005/0.001. Physical bytes, page mapping
[3,0,2], partial tail and finite FP32 reference reconstruction are checked.
Both exhaustive writers cover all 65,536 FP16 patterns across all 24 layers,
including five captured replays and negative padding. Native and repaired
real-model production each complete 32 requests with 128 outputs and
127 changing graph replays with 127 distinct input hashes. Every actually
recorded loaded binary is retained immutably by its content hash and
independently verified. The prior 60-receipt synthetic suite remains
historical evidence with its own source/runtime scope and expected failures.

Fresh NLL on 32 documents passes: its one-sided upper bound is
0.000570135157, below ln(1.01). Fresh retrieval on 100 cases scores
**41/100 native and 43/100 repaired**. The native control is below its
original 50% floor. Candidate has two gross losses and four gains; the
one-sided loss upper bound is **6.16192%**, above the original 5% limit.
Higher aggregate accuracy cannot cancel those losses or fix the control
floor. **GSM8K is gated off: zero questions run.** There is no D64
comparative timing or capacity study. The original GSM margin and 15%
control floor remain unchanged and untested in this panel.

An earlier collector reused mutable JIT-library paths, losing older binary
versions after later compilation. Its strict reader rejection is retained.
Raw quality still replays from each saved answer and target with exact
source/input/parser/statistic identities. A separately frozen
production-only validation captures every loaded binary immediately by
hash. Protocols, model inputs, inference bodies and the actual package
lock match. The negative quality panel is never repeated or replaced.
Original and fresh archive-restored reports agree. The reusable public
`snapshot_attention_libraries` helper also retains binary versions by hash;
numerical receipts now keep these files before further compilation. Its
eight CPU checks cover failed numerical checks and path reuse as well as
corrupt data. GPU execution remains pinned to the source above. This completes the
production evidence without converting the recorded quality failure into
a pass. The earlier failed historical model row retains its original
sources and workload; this is a new repaired-source result.

[Sanitized D64 fixtures, quality bounds and exact source/input identities](data/d64-model-quality-20261018.json)
retain the measured scope. Contiguous/ragged APIs, symmetric FP8 numerical
forms, BF16 and the combined routing-plus-D64 serving stack remain
unqualified. There is no full-model D64 speed claim or upstream PR here.
