# October 17: mixed fresh-prefill routing and the D64 numerical repair

The focused Qwen3-8B-AWQ routing test passes its production and task-quality
gates. Ten fresh confirmation blocks give median throughput gains of
4.8814% over accepted compact and 3.4483% over native.
Both comparisons win all ten blocks: exact one-sided sign p=0.0009765625,
p=0.001953125 after Holm correction for two comparisons. Pilots are separate.
The batch-8 (B8) engineering screen passes at median 0.99888838
against a 0.99 threshold. Its three blocks are not a statistical
noninferiority study. Keep this route opt-in and
scoped to the tested model, backend and recipe.

[Routing and quality](#routing-and-quality) ·
[D64 numerical repair](#d64-is-a-separate-numerical-qualification) ·
[Component profiles](#matched-component-profiles)

## Routing and quality

The new route compacts only the fresh-prefill suffix inside a mixed batch.
The accepted implementation uses compact for the initial all-fresh admission
and falls back on the four mixed admissions. Decode, persistent K16/V8
storage, scales and the one 256 MiB workspace remain unchanged. The study
uses Qwen/Qwen3-8B-AWQ revision 4da05a8e, packed Machete weights, FP16,
and tensor parallel size one,
8,192 actual input tokens, batch 32 (B32) and 128 forced outputs. Page size is 16/NHD,
The byte pool is 48 GiB, with 22 threads, at most 96 sequences and
65,536 batched tokens.
Automatic prefix caching (APC) and chunked prefill are off. FULL_DECODE_ONLY captures 1/8/32/44/45/59/60/96.
This is a separate new source test; it does not rescore the eight-model matrix.

| Role | Exact source |
| --- | --- |
| Native vLLM | `c21c0878210695db7b108cdc285aaf0cf890190a` |
| Accepted compact vLLM | `26cd429bac01d07909ed91e353f564d7aaa8d38f` |
| Fresh-prefill routing vLLM | `276305b29e0d11f95ada503ba76323ab24992e59` |
| FlashInfer for every routing arm | `800a550f8b736e49f8cfa136b82bfaf6d9d32711` |
| Model/quality/timing harness | `f91b3c3f1b59090a40398ebf1a9b4a980c2215a0` |
| Separate profiling collector | `babba7fb5f333f76d0fb6455ac924b5bfae11fd0` |

The [fresh-prefill vLLM branch](https://github.com/mcgrof/vllm/tree/2026-10-17-feature-mixed-fresh-prefill)
retains the exact measured candidate. The separate
[paged D64 FlashInfer branch](https://github.com/mcgrof/flashinfer/tree/2026-10-17-fix-mixed-d64-paged-layout)
retains the numerical repair. The original measured branches keep their
previous tips.

The independent oracle reconstructs pages from raw worker request order,
query offsets, KV lengths and block tables before candidate slicing. All 360
calls pass exact stored-byte/cast, same-dispatch output and log-normalizer,
repeatability, finite-value and output-placement checks. Four injected
mapping/placement faults are detected. All 36 layers cover fresh counts
8/7/7/7/3 with decode counts 0/8/15/22/29, twice with rotated prompts and
changing decode-graph inputs. This qualifies routing equivalence to the
accepted operation. The original ideal-FP32 gate remains failed; unchanged
fixed diagnostics contain 27 output violations among 3,538,944 elements,
maximum absolute error 0.015625, and zero sampled log-normalizer violations.
Those misses are not relabeled passes or evidence of a new routing defect.

Fresh candidate negative log likelihood (NLL) on 32 documents, retrieval
on 100 cases and all 1,319 GSM8K math cases pass against both
controls. Prior control quality reuse is bound to actual source, model,
tokenizer, scale and prompt identities and B32 coverage. NLL upper bounds
are 0.00145939 versus native and 0.00094033 versus accepted, below ln(1.01).
Retrieval is 91/100, with one gross loss versus native and zero versus
accepted; upper bounds 4.656%/2.951% stay within 5%. GSM is 1201/1319,
with 17 losses/19 gains versus native 1199 and zero losses versus accepted 1201.
The upper bounds 1.927%/0.227% stay within 2%. Higher aggregate accuracy is
not the decision rule. Clean allocator peak is 63.2265625 GiB; the largest
of 20 driver observations at 500 ms is 63.9775391 GiB. No new resident-capacity
endpoint is measured, and sampled driver observations are not continuous peaks.

The study retains three counterbalanced pilot blocks, ten new confirmation
blocks and three B8 engineering blocks. Every arm has its own warmup and
synchronized complete-request timer. No selective retry or extra timing
block follows a negative observation. [All sanitized ratios, arm rates,
quality summaries and component measurements](data/mixed-fresh-prefill-20261017.json)
retain the frozen orders. Component profiles run separately after the
primary observations, on one H100 for all three profile arms. Kernel unions
and correlated kernel-launch API time remain separate. Residual layer and
batch work and unmatched-kernel counts are included explicitly; unmatched
kernels are excluded from duration aggregates. These profiles do not prove
a critical path or replace the end-to-end speed tests.

The earlier stopped attempts remain incomplete. Recorder creation, GC-frozen
workspace inspection and required GSM manifest defects were repaired without
changing inference, scoring or thresholds. The original native profile
collector also required a compact module absent from the native source.
Its compatibility repair is collector-only. Verified passed stages are
reused through immutable export identities and independent CPU replay;
original failed attempts are never rescored as complete. The final reader
requires every primary gate and the separately completed three profiles.

## D64 is a separate numerical qualification

D64 means a head dimension of 64.

FlashInfer `72b18b4f1eef45a718de640568d7a0144cf3ab89` repairs the tested paged
FlashAttention-2 (FA2) NHD paths with FP16 queries/keys and FP8 values. It is not used in the
routing timings above. Against original `800a550f`, the fixed D64 fixture
(14 Q heads, 2 KV heads, 33 stored tokens, pages [3,0,2], tail 1, K scale 1, V scale 1/64)
gives maximum/mean errors 0.0467072725/0.0127550298. The repair gives
0.00002778694/0.00000644855. Across candidate checks, the largest maximum
error is 0.00008416176 and the largest mean is 0.00002120968, within unchanged
0.005/0.001 limits; outputs are finite.

The original grid completes 18 controls and the candidate 35 checks, plus
geometry fixtures, for 60 retained numerical receipts. Tests cover declared
D64 length boundaries, remapped pages, page zero, partial tails, decode,
prefill and changing graph replays, with actual D128 and symmetric controls.
CPU replay reconstructs saved physical bytes and references; actual mapped
JIT-library hashes match retained binaries. The unchanged CPU producer test
covers 108 concrete cases. The runtime uses vLLM 26cd429, which differs from
the historical full model row. Contiguous/ragged entrypoints, symmetric FP8
numerical forms and the full Qwen0.5 model remain unqualified. The older
Qwen0.5 failure row retains its original scope. No full-model D64 speed or
capacity claim follows, and no upstream PR is submitted by this closeout.

All owned pods are deleted after stopped-writer export verification and
independent API checks. Raw archives restore byte for byte, and complete
CPU reports agree on original and restored data. Total task use is
2.417941 GPU-hours with an unsettled
$9.77 estimate including disk allowance.

## Matched component profiles

The following milliseconds sum CUDA kernel unions within each operation.
Gather is included with prefill. They describe equivalent fresh work on
each scheduler cohort; they are not critical-path or whole-request times.
The JSON also reports unchanged decode, residual layer/batch work, mapped
and unmatched kernel counts, and correlated host launch API time separately.

| Fresh / decode requests | Native prefill ms | Accepted gather + prefill ms | Candidate gather + prefill ms |
| --- | ---: | ---: | ---: |
| 8 / 0 | 321.442 | 340.263 | 340.126 |
| 7 / 8 | 283.789 | 486.304 | 297.842 |
| 7 / 15 | 285.136 | 488.860 | 299.679 |
| 7 / 22 | 280.709 | 485.813 | 297.427 |
| 3 / 29 | 122.948 | 204.651 | 129.955 |

In these traces, the three seven-fresh cohorts fall from about 486–489 ms
to 297–300 ms for accepted versus candidate gather plus prefill. Native
prefill alone is about 281–285 ms. The initial all-fresh work is unchanged
between the two compact routes. The native end-to-end gain is established
by the separate confirmation trials; this profile does not isolate its
cause. Correlated host launch cost and the remaining device work are
available separately in the data file.
