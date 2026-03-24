# Cross-Model KV Sensitivity and Decode Scaling

This document brings together the cross-model side of the BPA and fused-KV
work. The goal is straightforward: document what generalized across diverse
transformer families, document what did not, and connect model-specific KV
sensitivity back to the broader decode-scaling and memory-traffic results.

This work matters for two related reasons. First, it shows that cache traffic
influences transformer decode performance across models, context lengths, batch
sizes, and hardware targets. Second, it shows that aggressive KV compression is
not governed by one universal policy. Some model families tolerate uniform
low-precision KV settings well. Others need asymmetric handling, runtime
calibration, or more conservative key precision. The result is a combined story
about both scaling and model-family sensitivity.

For the decode-side systems characterization, use [Memory-Traffic Saturation in Autoregressive Decode](https://github.com/mcgrof/knlp/blob/main/docs/memory_traffic_saturation_in_autoregressive_decode.md). For fused-kernel implementation and deployment notes, use [Fused KV Quantization](https://github.com/mcgrof/knlp/blob/main/docs/fused_kv_quantization.md). For script provenance and the reason this material is now split into clearer standalone documents, use [Cross-Model KV Sensitivity and Decode Scaling: Lineage and Script Provenance](https://github.com/mcgrof/knlp/blob/main/docs/cross_model_kv_sensitivity_and_decode_scaling_lineage.md).

## Table of Contents

- [What this document covers](#what-this-document-covers)
- [What we set out to test](#what-we-set-out-to-test)
- [What generalized across model families](#what-generalized-across-model-families)
- [What varied by family and model](#what-varied-by-family-and-model)
- [How the cross-model result connects to decode scaling](#how-the-cross-model-result-connects-to-decode-scaling)
- [Current evidence in `knlp`](#current-evidence-in-knlp)
- [How to use these results](#how-to-use-these-results)
- [Open work](#open-work)

## What this document covers

Use this document as the cross-model companion to the standalone memory-traffic
saturation writeup. The memory-traffic document is about the decode regime and
its cross-GPU behavior. This document is about what happens when that same work
is viewed across model families.

The underlying question is whether KV bandwidth scaling and decode-time cache
traffic matter only for a small set of models, or whether the effect survives
across a broader transformer landscape that includes Qwen, Llama, Mistral,
Gemma, Phi, Nemotron-style models, and DeepSeek MLA-related work. The current
answer is that the decode regime does generalize, but the safest compression
policy does not. That is why cross-model sensitivity work and decode-scaling
work need to be read together.

## What we set out to test

The first objective was to test whether cache traffic materially influences
transformer decode performance across models, context lengths, and batch sizes
under large-scale benchmarking. The second objective was to test whether low-
precision KV policies generalize across families or whether model-specific
behavior forces calibration, asymmetry, or bounded protection. The third
objective was to connect those observations back to systems decisions: when is
fused KV compression a real win, when is uniform INT4 safe, and when should a
runtime compatibility check decide the policy instead of a static rule.

## What generalized across model families

The broadest thing that generalized is the decode-side memory-traffic story.
Across the work collected in the paper and the associated `knlp` scripts, cache
traffic repeatedly shows up as a major determinant of decode behavior. This is
why the cross-model work belongs next to the memory-traffic saturation result
rather than being treated as a separate curiosity. The scaling regime is not an
H100-only or Qwen-only artifact.

The second thing that generalized is the usefulness of fused kernel paths over
non-fused ones. The underlying mechanism is stable: compression only helps if
it removes real traffic from the decode path that dominates runtime. That is a
cross-model implementation lesson, not a family-specific anecdote.

The third thing that generalized is that the problem cannot be reduced to a
single architectural scalar. Different families can share the same broad
decode-time memory-traffic regime while still differing substantially in their
KV precision sensitivity. That is why the model-family effort ended up
producing both a scaling result and a calibration result.

## What varied by family and model

The strongest family-level variation shows up in KV sensitivity. Earlier BPA
work documented a sharp split between two empirical patterns. In Qwen-family
models, sensitivity is dominated by a small number of critical layers,
especially the sink-like early layers. In Mistral and Llama-family models,
sensitivity is much more uniform and all-INT4 policies can stay within
tolerance. That earlier bounded-protection story is still important because it
established that not all model families react to KV quantization the same way.

The later fused-KV work extends that picture from a handful of families to a
larger benchmark set. The paper summary in `docs/fused_kv_quantization.md`
states that the runtime ratio classifier identifies sensitive models with 100%
accuracy across 14 models and generalizes to 72B. It also states that values
universally tolerate INT4 while key precision floors are model-dependent and
concentrated in the Qwen family. That is the current clearest public summary of
why cross-model calibration matters.

DeepSeek MLA-related work belongs in the same discussion even though it is not
yet folded into one clean benchmark matrix. Scripts such as
`scripts/analyze_mla_kv_structure.py` and the DeepSeek KVSplice work explore
whether MLA-style latent KV structure changes the usual K/V sensitivity story.
That is exactly the kind of architectural variation this cross-model document
is meant to track.

## How the cross-model result connects to decode scaling

The cross-model sensitivity work and the decode-scaling work answer different
parts of the same problem. The decode-scaling side tells you when KV traffic is
likely to matter: longer contexts, larger batches, and operating regimes where
decode becomes increasingly bandwidth-limited. The cross-model sensitivity side
tells you what kind of KV policy is safe once you are in that regime.

Read together, the combined result is stronger than either side alone. The
memory-traffic saturation work explains why reducing cache traffic can improve
decode. The cross-model sensitivity work explains why the best compression
policy cannot be assumed to be uniform across families. That is why current
practice in the repo combines fused kernels with a calibration or ratio-
classifier step instead of assuming that one low-precision setting works for
every model family.

## Current evidence in `knlp`

The current evidence in `knlp` is spread across several documents and scripts.
This section pulls those pieces together.

The current high-level summary lives in `docs/fused_kv_quantization.md`, which
captures the paper-level claims about 14 models, runtime calibration, family-
specific key sensitivity, and cross-GPU consistency. The earlier family-level
sensitivity evidence lives in `docs/bpa_v27_main_report.md` and
`docs/bpa_v27_submission_readiness.md`, where the sink-dominance versus uniform-
robustness split across Qwen, Mistral, and Llama was first made explicit. The
ratio-classifier implementation path is documented through
`scripts/bpa_h100_exp4_ratio_classifier.py` and
`scripts/marin_w7900_ratio_classifier.py`. DeepSeek MLA analysis is currently
represented by `scripts/analyze_mla_kv_structure.py` and related DeepSeek
plugin/evaluation scripts.

That scattered state is exactly why this document exists. The current repo has
real cross-model results, but until now the narrative has been spread across
older BPA reports, fused-KV notes, paper-facing summaries, and model-family
scripts.

## How to use these results

Use the cross-model result when you need to decide whether a decode optimization
is likely to generalize beyond one family. If the question is purely about
whether decode is bandwidth-limited, start with the memory-traffic saturation
document. If the question is whether a low-precision KV policy is safe for a
specific model family, start here and then follow the calibration links.

This document is also useful when evaluating deployment policy. A static claim
that INT4 or asymmetric KV settings work everywhere is not supported by the
current evidence. The stronger and more useful claim is that the decode regime
generalizes broadly, while the right compression policy can still vary by model
family. That is why runtime calibration remains part of the practical story.

## Open work

The cross-model effort is not finished. The most obvious remaining gap is that
we still do not have one fully unified benchmark matrix that crosses every
family, GPU, batch, and context under one perfectly symmetric protocol. The
current evidence is strong, but it is assembled from a combination of paper-
level benchmarking, earlier family studies, matched-lane decode characterization,
and ratio-classifier work.

There is also open work in extending the family coverage more cleanly. DeepSeek
MLA and other architecture-specific variants should eventually land in the same
public narrative with the same clarity as Qwen, Mistral, and Llama. The same
is true for newer larger families where the question is not just whether decode
is bandwidth-limited, but whether the calibration policy continues to compress
cleanly at scale.
