# Cross-Model KV Sensitivity and Decode Scaling: Lineage and Script Provenance

This document explains where the cross-model KV sensitivity and decode-scaling
story came from, why it is now split into a cleaner standalone document, and
which older reports and scripts it was assembled from.

The reason for the split is the same reason the memory-traffic saturation work
was split out. The result is now larger than the scattered BPA-era files that
first produced it. A new reader should not have to reconstruct the cross-model
story from submission-readiness notes, family-specific reports, fused-kernel
writeups, paper summaries, and calibration scripts that were accumulated over
multiple phases of R&D. At the same time, the lineage still matters. The old
artifacts explain how the questions were asked, why the result evolved the way
it did, and what evidence already existed before the standalone document was
created.

For the main standalone result document, use:
- [Cross-Model KV Sensitivity and Decode Scaling](https://github.com/mcgrof/knlp/blob/main/docs/cross_model_kv_sensitivity_and_decode_scaling.md)

For the decode-side systems characterization, use:
- [Memory-Traffic Saturation in Autoregressive Decode](https://github.com/mcgrof/knlp/blob/main/docs/memory_traffic_saturation_in_autoregressive_decode.md)

## Why split this out now

The cross-model effort now serves more than one purpose. It captures the older
family-level bounded-protection story, it connects the later fused-KV work to a
larger benchmark set, and it explains why runtime calibration is part of the
deployment story instead of an afterthought. That makes it too important to
leave buried inside older BPA notes or inside the fused-KV document alone.

The standalone document is the clean narrative for readers who want the result.
This lineage document keeps the provenance visible so the cleanup does not hide
where the result came from.

## How the work evolved

The older BPA arc first exposed the family-level split in sensitivity. Qwen
models showed sink-dominated behavior in which a small number of layers needed
protection. Mistral and Llama showed much more uniform robustness. Those
results were written up in documents such as `docs/bpa_v27_main_report.md` and
`docs/bpa_v27_submission_readiness.md`. That stage of the work answered the
question of whether bounded protection might exist at all.

The later fused-KV and paper-facing work expanded the problem. Instead of only
asking how many layers need protection, the work also asked whether fused
compression changes real decode performance, whether cache traffic explains the
runtime behavior across models and GPUs, and whether a short runtime
compatibility check can choose the right policy for each model family. That is
where the ratio-classifier scripts and the 14-model paper benchmark enter the
story.

The result today is therefore a combined lineage. One branch came from earlier
family-level sensitivity and bounded-protection experiments. The other branch
came from the later decode-scaling and fused-kernel work. The standalone cross-
model document exists because those branches now need to be read together.

## Why the older artifacts are not enough by themselves

The older artifacts are still valuable, but they are not a good default entry
point. Submission-readiness documents are written for an internal decision
point, not as a durable public overview. Paper summaries flatten details to fit
publication needs. Scripts capture implementation reality but not always the
narrative that explains why a reader should care. If someone tried to reconstruct
the whole cross-model story from those pieces alone, they would either miss
important links between them or overfit to one phase of the work.

That is exactly the situation this standalone document is meant to fix.

## Main provenance sources

The earlier family-level bounded-protection result is documented most clearly
in:
- `docs/bpa_v27_main_report.md`
- `docs/bpa_v27_submission_readiness.md`
- `docs/bpa_v27_paper_outline.md`

The later fused-KV and paper-level summary is documented most clearly in:
- `docs/fused_kv_quantization.md`
- `paper-memory-decode`

The current calibration and compatibility machinery is represented by:
- `scripts/bpa_h100_exp4_ratio_classifier.py`
- `scripts/marin_w7900_ratio_classifier.py`
- `docs/kv_plugin/calibration_guide.md`

The DeepSeek MLA / architectural-extension side is represented by:
- `scripts/analyze_mla_kv_structure.py`
- DeepSeek KVSplice scripts and evaluation helpers

## Relationship to BPA and memory-traffic saturation

This lineage still belongs to the broader BPA story. BPA is what made decode
traffic and KV policy a first-class systems question instead of a loose model-
compression side topic. But the cross-model result now deserves its own clear
home, just as the memory-traffic saturation result did.

The clean ordering is now:
1. BPA provides the background and initial problem framing.
2. Memory-traffic saturation documents the decode-side systems regime.
3. Cross-model KV sensitivity and decode scaling documents what generalized
   across families and what still requires model-specific policy.
4. Fused KV quantization documents the strongest concrete intervention path.

Keeping those pieces separate makes each one clearer, but keeping this lineage
document preserves the chain that connects them.
