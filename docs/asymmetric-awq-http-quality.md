# Qwen3-8B AWQ HTTP quality: preservation not established

The actual HTTP-worker test did not pass the frozen GSM8K preservation
rule against either native control. No AWQ HTTP performance confirmation
was run. This narrows the supported execution scope; the earlier 8K,
batch-32 in-process throughput result does not establish a quality-preserving
HTTP serving gain.

The test used Qwen3-8B AWQ on one H100 80 GB SXM, packed INT4 weights,
FP16 compute and a fixed 48 GiB persistent KV pool. The candidate stored
FP16 keys and E4M3 FP8 values. Both native controls stored FP16 keys and
values. All arms used the same HTTP adapter, original frozen model-specific
scales, prompt IDs, scheduler limits and graph policy. Client concurrency
was 32; this is not a claim that every GPU step had batch size 32.

| GSM8K result | K16/V8 | Native FlashInfer | Native FA3 |
| --- | ---: | ---: | ---: |
| Strict correct / 1,319 | 1,195 | 1,198 | 1,202 |
| Native-correct → candidate-wrong | — | 20 | 22 |
| Native-wrong → candidate-correct | — | 17 | 15 |
| One-sided 95% loss-probability upper bound | — | 2.196% | 2.373% |
| Required upper bound | — | <2% | <2% |

The denominator for the transition probability is all 1,319 paired
questions. This gate limits loss of previously correct answers. It is
not a 2% threshold on net accuracy change. The observed net changes were
−0.227 percentage points against FlashInfer and −0.531 points against FA3.
Their paired-question bootstrap 95% intervals were [−1.137, +0.758] and
[−1.440, +0.379] percentage points. Both include zero. The failed
preservation gate does not prove a large or significant net accuracy loss.

The other quality checks passed. At 8K context, both comparisons passed
the NLL margin on 32 held-out documents with 128 cached decode steps
per document. Prefill step zero was excluded. All arms scored 91/100 on
retrieval; each comparison had one lost and one gained answer, with a
4.656% loss upper bound below the 5% limit. All native accuracy floors
passed. GSM8K used natural question lengths and natural completion,
with a 2,048-token limit and the unchanged strict answer parser.

All three arms matched serial in-process generation on 32 prompts in
both reference orders. Earlier batched-versus-serial disagreements were
retained, so the HTTP quality result was measured directly rather than
borrowed from batched execution. The serial parity check does not claim
invariance to every batching pattern.

Excluded pilot timings remain exploratory. They cannot support a new
HTTP throughput or tail-latency claim, and no additional blocks were added
to rescue this quality result. The fixed KV pool also does not measure
a 25% reduction in total GPU memory.

The archived result
contains exact model and source revisions, paired estimates and intervals,
protocol limits, and raw evidence hashes. The independent reader is
[`serving_quality_audit.py`](../tools/kv/native/serving_quality_audit.py),
with [reproduction instructions](asymmetric-serving-study.md).
