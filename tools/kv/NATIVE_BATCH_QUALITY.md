# Quality at actual decode concurrency

Short answers can finish before every submitted request has completed its
prompt. A batch of 64 retrieval questions therefore does not by itself
check cached decoding with 64 active requests. This supplementary evaluator
uses the same frozen language-model windows as native qualification and
keeps every request alive through 128 scored cached-decode predictions.

`native_batch_quality.py` observes the pinned V2 runner's request mapping.
It records the original FP32 target loss and top prediction, then forces
the frozen next token. Attention, cache writing, allocation and scheduling
run unchanged. A single-request bridge must reproduce every native target
loss and top prediction exactly before a batched result is accepted. The
observer is removed afterward; serving measurements use fresh processes.

Use the complete source pair and inputs expected by `tools.kv.native`.
Supply the primary native quality directory and its complete native/GSM8K
qualification. Freeze `batch-quality-protocol.json` with the evaluator's
`module_sha256`, the primary `core_protocol_sha256`, and source revisions
before its scored measurements. Set the following environment explicitly:

```sh
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
python -m tools.kv.native_batch_quality --inputs inputs \
    --quality results/quality --output results/batch32 \
    --arm k16_v8 --batch 32
```

The default uses a 48 GiB explicit cache budget and checks both 2K and 8K
windows. For an automatic pool under the primary total-memory envelope, use
`--utilization` and `--length 8192` at the claimed maximum active batch.
The primary core package hashes stay fixed; this separate evaluator records
its own source hash and protocol alongside them.

All 32 document windows contribute at each length. Batches larger than 32
repeat those documents to exercise the requested concurrency. Duplicate
requests are averaged within each original document and do not increase
the statistical sample size. Each cohort must sustain at least 32 decode
steps at the full requested concurrency with zero preemption. Scores use
the existing one-percent perplexity margin, paired both with the primary
baseline and with the same arm's single-request reference. Failures retain
raw per-request targets, losses, predictions and scheduler observations.

This certifies the stated fixed-target loss regression. Arithmetic accuracy,
retrieval accuracy and unrelated workloads require their own evidence.
