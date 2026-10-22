# Precision placement and scale-policy reference

This harness compares native K/V with asymmetric K16/V8, reverse K8/V16
and symmetric K8/V8. Each compressed format has fixed per-layer/component
scales and dynamic per-token/head scales. Q and attention arithmetic remain
native. Actual E4M3FN tensors store compressed components; each attention
layer temporarily reconstructs them. These timings do not measure a fused
serving backend.

The scoped checkpoints are Qwen3-8B FP16 at
`b968826d9c46dd6066d109eabc6255188de91218` and Qwen2.5-7B-Instruct BF16 at
`a09a35458c702b33eeacc393d103063234e8bc28`. Other architectures are rejected.
The earlier `tools/kv/recent` implementation and results are unchanged.

Use Python 3.12.11, Torch 2.11.0 with CUDA 12.9, Transformers 5.10.4,
Accelerate 1.10.1, NumPy 1.26.4, SciPy 1.17.1 and pytest 8.4.2.
Input preparation additionally uses datasets 4.8.4. Install the CUDA Torch
wheel from the official PyTorch index and the remaining packages from PyPI.

```sh
python -m pytest -q tests/test_decision_cache.py tests/test_decision_statistics.py
python -m tools.kv.decision.prepare --model qwen3 --out inputs/qwen3
python -m tools.kv.decision.prepare --model qwen25 --out inputs/qwen25
python -m tools.kv.decision.freeze --out source.tar.gz
```

Run from an extracted source archive, which includes `SOURCE.json` and
checks every source file before execution. Download the exact checkpoint
revision with `huggingface_hub.snapshot_download`. Create an official weight
manifest from `HfApi.model_info(..., files_metadata=True)` with `model`,
`revision` and `files` mapping safetensor names to `bytes` and LFS `sha256`.
The runner verifies every downloaded weight file against that manifest.
Freeze source, weight, input and protocol identities before model outputs.

```sh
python -m tools.kv.decision.run --inputs inputs.json.gz --model-path model \
  --weights official-weights.json --out calibration --stage calibration
python -m tools.kv.decision.oracle --calibration calibration
python -m tools.kv.decision.run --inputs inputs.json.gz --model-path model \
  --weights official-weights.json --out development --stage development \
  --scales calibration/scales.json --batch 8
python -m tools.kv.decision.run --inputs inputs.json.gz --model-path model \
  --weights official-weights.json --out confirmation --stage confirmation \
  --scales calibration/scales.json --development development --batch 8
```

Calibration observes native cache-write operands after normalization/RoPE
on 32 independent documents at 1K and 8K, including 128 forced greedy decode
steps. Each fixed component scale is FP32 `max(maxabs, 1e-12)/448` over all
those operands. A second native replay audits clipping and error for both
scale policies on every calibration operand. Saved activation fragments
also undergo independent E4M3FN codebook and reconstruction comparisons
on CUDA when available (otherwise CPU). The recorded device is explicit.
Dynamic scales apply the same equation independently to each fresh
head/token vector. Stored codes are never requantized. Conversion clips
to ±448, uses nearest/even E4M3FN rounding and reconstructs by FP32
multiplication followed by the native dtype cast.

Input preparation excludes the first 128 eligible Wikipedia articles used
by the earlier reference, then draws 32 calibration, 16 development and
64 confirmation documents from a seeded new pool. Document IDs are disjoint
between stages. Within each stage the same documents supply the two context
lengths; inference remains per document within each named endpoint.
Retrieval instances have independent generators and unique cluster IDs.
Development uses 32 retrieval instances/context and 64 GSM8K training items.
Confirmation uses 200 retrieval instances/context and all 1,319 GSM8K test
items. GSM8K is an exposed standard benchmark. The previous reference's
answer-label parser and 2,048-token limit remain unchanged; saved truncation
and parser-failure fields are retained with that scoring rule.

Each candidate receives two separate certificates: answer preservation and
aggregate-math noninferiority. The two families each have alpha .025 across
12 candidate/model combinations. Every certificate requires native accuracy
at least 50%, retrieval gross-loss upper bounds at most 5% at both lengths,
and relative mean NLL upper bounds at most 1% at both lengths. Preservation
also requires GSM gross-loss upper at most 2%. Aggregate-math instead requires
a paired net GSM accuracy lower bound at least −1 percentage point.
Gross bounds are exact binomial. Net bounds invert Tango's efficient score
with approximate coverage. NLL uses 100,000 paired document bootstrap draws,
seed 20261020 and linear quantiles. Report absolute NLL deltas separately.

Direct A-minus-R and A-minus-S tests form eight contrasts per named endpoint
family across the two models and policies. Binary outcomes use exact McNemar
p-values and descriptive paired score intervals; NLL uses a centered paired
bootstrap with descriptive and Bonferroni-eight intervals. Holm adjustment
is applied across all eight contrasts per endpoint, without pooling different
endpoints into a general superiority claim. Failed or incomplete panels
cannot be replaced or enlarged to obtain a pass.

CPU analysis expects `inputs/{qwen3,qwen25}/inputs.json.gz` and
`{qwen3,qwen25}/artifact-mirror/confirmation/` below the supplied study root. A `PLAN.json` `executions` mapping may
name retained attempt directories explicitly; no latest-result selection occurs:

```sh
python -m tools.kv.decision.analyze --study study --out analysis.json
```

Audit retained evidence independently before interpreting the certificates:

```sh
python -m tools.kv.decision.evidence --study study --output evidence.json
```

This checks frozen input/source/weight identities, complete item inventories,
original EOS/truncation flags, and text decoded from retained token IDs. It
reconciles persistent tensor shapes, strides, scale bytes and distinct backing
allocations. A/R must match both requested bytes and the complete allocation
size multiset. Those quantities do not include allocator fragmentation or
equalize whole-process reserved memory. The reader changes no score or gate.

A quality-reference pass alone does not qualify production support, memory
capacity under a serving allocator or useful serving throughput. Those need
actual-worker tests and a separate frozen quality/serving protocol.

`serving.py` provides connection tracing for that conditional next stage.
It reuses a persistent HTTP session, records every offered request and has
no retries, semaphore or connector concurrency cap. Its optional
`AcceptedRequests` ASGI middleware records the application acceptance
boundary without rebuilding the streaming request or response. Set
`KNLP_DECISION_HTTP_EVENTS` to a dedicated capture path before installing
that middleware. ASGI acceptance alone does not establish worker execution.

The client requires `aiohttp` and `psutil`. A separately frozen JSON trace
names `mode="open"`, `model`, `seed`, `arrival_seconds`, `drain_seconds`,
`request_timeout_seconds` and `requests`. Each request specifies a unique
`id`, integer `offset_ns`, `prompt_token_ids` and `max_tokens`. Run it with:

```sh
python -m tools.kv.decision.serving --trace trace.json --output capture
```

`goodput.evaluate` applies externally frozen `ttft_ms`, `tpot_ms` and
`eos_ms` limits to the resulting capture. Failed, invalid, wrong-length,
late and timed-out requests contribute zero to useful completions divided
by the common arrival window. It reports completion and SLO fractions of
all offered requests, plus conditional latency distributions. Coalesced
token timestamps remain at SSE-event resolution. `compare_blocks` requires
all ten independent paired blocks, Holm-adjusts sign tests across the full
control family and reports conservative simultaneous Bonferroni median
lower bounds. A practical claim also requires a 5% median relative gain.
These tools do not select SLOs or establish a completed serving result.

`paged_memory.py` reconciles live paged K/V, inline-scale and scalar-scale tensor storages
with Torch's CUDA allocator snapshot. It counts each shared backing once,
includes active-block rounding, and keeps inactive segment space separate.
The worker inventory rejects oversized backing storage and enforces the
declared persistent-cache budget, including scales. Staging and whole-process
allocated/reserved memory are reported separately. Unknown/custom allocations
fail closed. CPU tests cover aliasing, rounding and inconsistent snapshots;
actual kernel read/write behavior still requires worker qualification.

`inline_scales.py` independently separates E4M3 codes from inline FP32
token/head scales in an NHD cache and checks the backend scale views' exact
aliases. The views share their original backing. The caller must still
audit that backing's complete ownership and allocator bytes, including any
larger allocation around an aligned subview. CPU tests exercise disjoint
code/scale writes and reject copied, swapped or misaligned scale planes.
This helper does not establish dynamic-worker numerical or quality support.
Its snapshot decoder reconstructs FP32 oracle operands directly from the
gathered code and scale bytes through a specified causal prefix. It excludes
the unused final-page tail and rejects invalid used scales, codes or overflow.
Literal-byte tests check per-token/head scales and compare resulting attention
with an independent head loop. This reconstruction is for excluded numerical
checks; it is not a serving kernel.

`production_dynamic_oracle.py` computes expected FP32 scale bits from
retained native CPU operands and uses the independent E4M3 codebook. Scale
creation clamps the maximum before multiplying by the rounded FP32 reciprocal
of 448, matching the reference's CUDA scalar-division path. Normalization
uses FP32 division. Tests distinguish the one-ULP scale-construction cases
and cover signed zero, subnormals, per-head independence and rounding ties.
Actual GPU writer integration and byte comparisons remain required.

## Conditional actual-worker comparison

The `production_*` tools implement the next experimental protocol; they do
not yet establish a completed GPU qualification or serving result. Use them
only after both full reference panels and their evidence audits finish. The
worker panel selects fixed per-layer/component AF and dynamic per-token/head
SD against native FlashInfer and FA3 controls. The Triton extension adds an
opt-in writer with the reference scale arithmetic; its storage and attention
paths already support per-token/head FP8. The preparer requires a complete
aggregate-math-qualified SD reference. Numerical checks and actual-worker
quality must still qualify both compressed implementations.

The worker pins the HTTP vLLM source
`e5c8d87fd6bbf9cdedf266d160890599d938abb0` and FlashInfer
`800a550f8b736e49f8cfa136b82bfaf6d9d32711`. The installer's `decision-http`
recipe supplies the original base; explicitly select the worker revision
shown below. It uses Torch 2.11.0+cu129 and
Transformers 5.5.3. The separate `current` batch recipe keeps its existing
source pins. Production requires one H100 80GB HBM3, at least twelve usable
CPU cores of affinity and quota, and an otherwise idle device.

Create `official-model-files.json` from the pinned checkpoint's immutable
Git objects for `config.json`, `generation_config.json` and
`model.safetensors.index.json`. The mapping `files` contains each filename's
`bytes`, `digest` and `algorithm="git_blob_sha1"`; the top-level
`model` and `revision` match the original checkpoint. This supplements the
safetensor weight SHA-256 manifest. Preserve the original tokenizer and
prompt token IDs rather than re-encoding displayed text.

```sh
python -m tools.kv.decision.production_prepare \
  --analysis reference-analysis.json --evidence reference-evidence.json \
  --reference-inputs inputs/qwen3 \
  --reference-scales calibration/scales.json \
  --official-model-files official-model-files.json --out production-inputs
python -m tools.kv.native.october_setup install --recipe decision-http \
  --vllm-revision e5c8d87fd6bbf9cdedf266d160890599d938abb0
```

The installer can transfer an unpublished dependency change for qualification.
`--vllm-revision` selects its full commit identity. Add
`--vllm-source-bundle source.bundle --vllm-bundle-sha256 "$BUNDLE_SHA256"`
for a version-2 Git bundle created with
`git bundle create --version=2 source.bundle "$BASE_COMMIT..$FEATURE_BRANCH"`.
It must contain one branch head and one prerequisite: the original
`decision-http` vLLM commit. The installer checks the bundle digest, Git
objects, ancestry and unchanged compiled-wheel sources, and records the
transfer in `setup.json`. Once that exact commit is published, omit the
bundle flags. These options only install source; they do not update or
bypass the worker protocol's separate source and numerical quality checks.

Preparation requires committed, clean source and freezes all runner/input
hashes in `production-inputs/protocol.json`. Preserve that file's SHA-256 as
`PROTOCOL_SHA`. Download the pinned checkpoint into the Hugging Face snapshot
cache used by the installed environment, then keep model access offline.
The installer and full setup/package/source receipts remain part of the
evidence. A helper creates no cloud allocation.

For each of `native_fi`, `AF`, `native_fa3`, `SD`, run one fresh numerical and
quality invocation. Use the installed Python for all worker commands:

```sh
install/env/bin/python -m tools.kv.decision.production_stage \
  --install install --inputs production-inputs --protocol-sha256 "$PROTOCOL_SHA" \
  --arm native_fi --phase numerical_quality --output worker-quality/native_fi
install/env/bin/python -m tools.kv.decision.production_analysis \
  --inputs production-inputs --protocol-sha256 "$PROTOCOL_SHA" \
  --root worker-quality --output worker-quality-analysis.json
```

Q remains FP16 in every arm; fixed K/V scales transfer before warmup and
graph capture. SD uses inline FP32 token/head scales with native queries.
Native controls keep normal optimized dispatch. Numerical
probes check raw page/slot routing, physical codes, independent FP32 causal
attention, fresh/mixed/decode execution, near-capacity 8K batches and repeated
graph/eager replay. They sample selected rows and do not prove every possible
operand. Their bounds and case lists are frozen before worker outcomes.
For SD, probes retain the actually launched writer and 2D/3D attention
kernels, exact scale aliases, and code-plus-scale byte checks. Graph checks
also retain inline scales; segmented-softmax buffers are audited separately.
Numerical instrumentation has an explicit memory exemption; quality and
timing do not. The full quality panel retains 64 documents/context, 200
retrieval cases/context and all 1,319 GSM8K questions with the original parser
and truncation semantics.

The two quality objectives each retain four candidate/control slots at
alpha .025/4. Serving preselects aggregate-math and requires the candidate
to qualify against both native controls. Strict preservation is reported
separately. A declared SD support gap requires retained launch/log/exit
evidence from a failure before any quality item; its unused alpha slots
remain unused. Neither a support failure nor an unestablished certificate
is proof of task-quality degradation.

Run `--phase equal_slots` in another fresh invocation for each completed
worker arm. `production_memory_analysis` joins those captures with the
quality analysis using `--equal-slot-root` and `--quality-analysis`. It
reports 17,000-block equal-slot bytes separately from physical capacity
under the common 40-GiB KV-plus-scale allocation budget. The worker uses
the same payload request across arms after reserving 36 MiB for unsplit
KV allocation tails and 55,296 bytes for scale allocations. This follows
the pinned [Torch 2.11 native allocator's splitting rule](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp#L2894-L2920).
The live allocator inventory, rather than this request bound, determines
whether the physical budget passes. The worker uses
eight server, two client and two control CPUs, a 68-GiB Torch reserved cap,
and a 76-GiB driver envelope sampled every 100 ms. Gaps above 500 ms fail
the resource audit; shorter peaks can be missed. Staging, weights and graph
pools are separate from persistent cache storage.

Memory observation excludes optional utilization and power queries. A bounded
writer queue keeps log I/O off the sampling path; write errors, queue exhaustion
or an unflushed boundary reject the invocation. New samples record entry and
return times around each memory query. The audit brackets the whole request
window with those intervals and uses the largest possible gap between actual
reads, retaining the 500-ms limit. Legacy point-timestamp captures keep their
original audit semantics; they are not upgraded by the new collector.

## Gated serving plans and analysis

`production_serving_plan --phase initial` creates a native-FA3-only soak
and a bounded calibration fallback sequence. Pass the frozen worker inputs,
protocol hash, `--quality-analysis` and `--output`. Run the soak once, then
calibration entries in manifest order until the first stable point. Keep all
failed or unstable observations; later fallback plans remain unexecuted.

For each plan, `production_stage --phase timing` additionally requires
`--plan`, `--plan-sha256` and `--quality-analysis`. Every invocation has its
own output directory and fresh server. `production_timing_analysis` replays
the exact raw client/ASGI/worker, resource and build evidence using those
same inputs plus `--root` and `--output`.

`production_serving_plan --phase loads` requires `--previous-plans`,
`--previous-manifest-sha256` and `--previous-results`. It independently
replays the native evidence and freezes absolute limits at 110% of the
selected pilot's p95. Run its complete nine-load native grid once. The
`confirmation` phase uses the same previous-phase arguments to select the
eligible 70–90% SLO-attainment load closest to 80%, or record that the bounded
grid found no informative primary cell. No candidate timing selects the
limits, load or quality route.

Freeze and retain the complete confirmation manifest and every named plan
before candidate timing. It specifies ten counterbalanced blocks, matching
Poisson arrival traces across all qualified arms, 120-second arrival windows,
120-second drain, 128 forced outputs and a fresh server per arm/block. The
secondary 1K cell uses the same absolute limits; it and other 8K loads are
descriptive. Requests ending after the fixed drain still contribute zero,
even while bounded post-client cleanup waits for an empty scheduler.

`production_serving_analysis` takes `--plans`, `--manifest-sha256`,
`--root`, `--quality-analysis`, the frozen inputs/protocol and `--output`.
It requires all ten primary paired blocks and the complete qualified control
family. Genuine overload preemptions and failed offered requests remain in
the data. Missing/changed captures, invalid resource evidence or client
saturation prevent a serving improvement claim; there are no replacement
blocks. Compiled-module/package identities must match worker quality, and
generated-kernel inventories remain available at each boundary.

## Reference figures

With Matplotlib installed, `python -m tools.kv.decision.figures --analysis
REFERENCE-ANALYSIS.json --evidence REFERENCE-EVIDENCE.json --receipt
REFERENCE-ANALYSIS-RECEIPT.json --out figures` verifies the complete reader
receipt and both models' panel counts before exporting aggregate-only JSON,
SVG and PNG files. It excludes raw question text and operator metadata.
The figures show descriptive paired math intervals and tensor backing bytes;
neither a descriptive interval nor storage alone establishes a certificate
or serving benefit. Certificate labels in the JSON retain the full objective.
The aggregate bundle also includes both retrieval and NLL endpoints and all
20 direct contrasts per model, with paired intervals and the eight-comparison
Holm adjustment for each endpoint. Direct result fields named `native_correct`
or `native_mean` refer to that row's explicitly named compressed baseline.
Significance for one endpoint does not establish overall superiority.

Use `--data figures/reference-aggregates.json --out redrawn` instead to
redraw a published aggregate bundle. This reproduces figures without
independently re-auditing the raw experiment. The figure manifest records
the renderer, plotting version and output hashes. Render checks use a
visible synthetic-data watermark and are not experimental evidence.
