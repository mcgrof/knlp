# Recent models: tested capability and current limits

The fused K16/V8 implementation tested here does not provide a qualified serving
benefit across recent architectures. Ministral 3 failed the frozen
quality-preservation task. Qwen3.5-9B and Gemma 4 12B ran native calibration,
but their cache layouts or attention features are outside the asymmetric
implementation. These outcomes have different meanings.

The separate October 20 comparison tests Qwen3.8-27B and Gemma 4 26B-A4B
with physical Transformers reference caches. Qwen passes both K16/V8
and symmetric FP8; neither Gemma candidate clears all prediction-loss
bounds. It establishes no distinctive asymmetric advantage or fused
serving support. See the [completed comparison](asymmetric-quantization.html#recent-models)
for quality, state memory and reproduction.

The [precision/scaling reference](asymmetric-results.html#decision-reference)
adds matched fixed and dynamic policies on Qwen3-8B and Qwen2.5-7B-Instruct.
It identifies model-dependent value from preserving keys and distinguishes
answer preservation from aggregate accuracy. It does not extend optimized
support to any of the architectures below.

| Model | Actual test | Result |
| --- | --- | --- |
| Ministral-3-8B-Instruct-2512 | Physical K16/V8, both native controls, production checks and full task quality on H200 | Preservation not established on the frozen GSM8K task; no performance confirmation |
| Qwen3.5-9B | Native FlashInfer, FP16 compute, 32 calibration captures on H200 | Native capability passes; asymmetric shared-page layout unsupported |
| Gemma-4-12B-it | Native Triton attention, FP16 compute, 32 calibration captures on H200 | Native capability passes with its declared logit mask; asymmetric attention configuration unsupported |

Read the [Ministral quality result](asymmetric-ministral-quality.md) for
both task definitions, paired loss bounds and the parser limitation.
A failed quality gate does not establish that all uses of quantization
harm this model. The native-only rows have no asymmetric task-quality or
performance measurement at all.

Qwen3.5-9B has eight full-attention layers and 24 recurrent layers. The
native runtime shares each physical cache backing between three recurrent
layers and one attention layer. Attention uses 16 query heads, four KV
heads and head dimension 256. Reinterpreting that backing as one whole
key region followed by one whole value region would violate its page
layout. The current guard rejects this configuration. A correct layout
port requires more than changing a head-dimension check.

The fraction of attention layers does not determine a total-memory
saving: recurrent state and attention KV scale differently with sequence
length. No 25% total-memory or one-quarter-of-layers saving is inferred.

Gemma 4 combines sliding and global attention with head dimensions 256
and 512 and soft-capped logits. The selected native FlashInfer backend
rejected this configuration; native Triton ran the full calibration
panel. Initial finite-logit checks encountered two intentional negative
infinities at model-declared suppressed token IDs, 258883 and 258882.
The corrected check requires that exact mask and finite logits everywhere
else. This was not evidence of FP16 numerical instability. A brief native
BF16 diagnostic did not qualify a BF16 asymmetric recipe.

The native capability runs used 16 documents at each of 1K and 8K input,
with 128 forced output tokens per capture. They used eager execution, a
16 GiB cache budget, one GPU, and language-only inputs. They establish
loading, execution and calibration under those settings, not production
quality, graph qualification or throughput. Some recurrent parameters
remain FP32 as required by the native implementation.

The archived record
contains exact checkpoint and harness revisions, runtime policy, measured
geometry, cache backing sizes and raw evidence hashes. Model definitions
are available from [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B),
[Google](https://huggingface.co/google/gemma-4-12B-it) and
[Mistral](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16).
The supported Qwen3 recipe retains its separate
[installation and validation scope](asymmetric-install.md).
