# Family residency: establish the trace evidence first

A family lease would keep an existing vLLM ancestor resident between related
requests. It consumes cache capacity that could serve other requests. A useful
comparison therefore needs observed branch timing, exact token identities,
competing requests and physical cache observations. Text reuse alone does not
establish that holding an ancestor improves latency.

The CPU auditor checks the public
[LMCache agent-trace corpus](https://github.com/LMCache/lmcache-agent-trace)
at revision `780bcc2979715150d8b9fd4737e026e625444cc9`. It authenticates the clean
checkout, hashes each selected tracked JSONL file, reports malformed records, removes
exact duplicate records within each framework, and groups descriptive text
statistics by explicit session identifier. Session identity does not establish
family lineage. Derived `_result` directories are excluded. The output contains
counts and hashes, without copying prompts or answers. The source corpus uses
the Apache-2.0 license; its README labels PyCodeAGI as a synthetic pipeline.

```sh
git clone https://github.com/LMCache/lmcache-agent-trace.git /path/to/traces
git -C /path/to/traces checkout 780bcc2979715150d8b9fd4737e026e625444cc9
make defconfig-kmesh-trace-audit
KMESH_TRACE_ROOT=/path/to/traces make kmesh-trace-plan
KMESH_TRACE_ROOT=/path/to/traces make
```

Use Python 3.10 or later. No extra Python packages, model or GPU are required.
Set `KMESH_TRACE_OUTPUT` to a fresh output directory; existing output is
rejected. Recheck portable contracts with `make kmesh-trace-test`.

The median character-prefix fraction compares consecutive input strings within
one session. It is descriptive text overlap, with no character-to-token
conversion. Numeric epoch units are inferred only for descriptive spacing
between record timestamps. Neither statistic is an observed idle interval,
active inference interval, physical cache hit, or fork gap. Malformed rows stay
visible in the report rather than silently becoming zero reuse or zero churn.

## Capturing invocation boundaries

`KMESH_TRACE_MODE=invocations` accepts JSONL records with schema
`knlp.kmesh.invocation.v1`. Each record must include globally unique
`request_id`, `family_id`, nullable `parent_request_id`, `engine_id`, `clock_id`,
integer `start_ns` and `end_ns`, exact `input_token_ids` and `output_token_ids`,
terminal `status` (`complete`, `cancelled` or `error`), and a
`cache_observations` list. Record actual invocations, including cancelled work;
retain the clock domain so unrelated monotonic clocks cannot be subtracted.
Root requests explicitly use a null parent. Capture identity must also retain
the model, adapter, cache layout and source/runtime pins in its own manifest.

The auditor rejects duplicate request identities, cyclic lineage and conflicting
family identities. It measures signed parent-completion-to-child-start gaps
only for known parents in the same clock domain. Negative gaps identify children
that start before the parent completes. It reports cross-engine edges and
non-exact parent prefixes separately. Family membership does not imply equal KV.

This first audit validates invocation metadata, not the contents of physical
cache observations. `pressure_replay_qualified` remains false. A pressure study
additionally needs actual pool capacity, block identities, reference counts,
evictions, competing-family demand and displaced hits. No fixed-concurrency
extrapolation or predicted latency verdict is emitted. A bounded capture can
qualify the mechanism without establishing a production-workload benefit.

## Bounded references to an existing ancestor

`FamilyLeases` is an experiment controller over the real vLLM block pool. It
adds references to the producing request's existing full, hashed ancestor
blocks. It allocates no new ancestor blocks and copies no cache data. A cap on
unique physical blocks bounds memory cost; overlapping families still own
separate references. A family-count cap bounds metadata. Generation tokens
prevent a late release from freeing a newer family hold. Expiry, cancellation,
completion and close each drop only the controller's references.

The caller must run on the synchronous scheduler thread, acquire after completed
model execution and before the producing request releases its blocks, and drain
holds before resetting cache or changing model identity. This prototype does
not implement asynchronous synchronization, production admission or a useful
family-selection policy. Cache invalidation and error recovery need integration
before it can become an engine feature.

Use vLLM revision `1b76bf9dd9d52efbd75fb89bf2619898d8f01233` and its matching
compiled CUDA wheel, as in the
[prepared serving runtime](../kmesh_restore/README.md). The runtime must import
`vllm` from that clean source directory. Set `PYTHONPATH` to the prepared source
and dependencies when required by the build. The worker checks the loaded
allocator's path and git identity and records Python, torch and vLLM versions.

```sh
make defconfig-kmesh-family-residency-cpu
export KMESH_RESIDENCY_SOURCE=/path/to/vllm
export KMESH_RESIDENCY_OUTPUT=/path/to/fresh-cpu-results
make kmesh-residency-plan
make KMESH_RESIDENCY_PY=/path/to/prepared/python
```

The CPU child hides GPUs and drives the actual allocator metadata. It exercises
overlapping-family accounting, stale and duplicate releases, budget rejection,
expiry, cancellation, close, invalid blocks and the serving observer. As in the
existing allocator benchmark, successful allocation stands in for completed
prefill; there is no model execution. A controlled synthetic pressure case uses
32 usable blocks, eight protected ancestor blocks and two passes over four
unrelated prefixes. It counts the child's saved prefix and competing cache hits
separately. This request order is chosen to expose capacity competition, not
inferred from the public trace corpus.

## Controlled model capture

The serving worker runs the controller through actual synchronous vLLM
requests. The fixed pool has 129 blocks including its reserved null block,
with 16 tokens per block. A 512-token ancestor occupies 32 blocks, one quarter
of the 128 usable blocks. Each arm runs the parent, two passes over four
unrelated 497-token prompts, then one child extending the parent's exact input
and generated output. The four competing requests each need 32 blocks with
their eight generated tokens, so their working set fills the usable stock pool.
Holding the ancestor reduces capacity available to them. This is a deliberate
synthetic stress case, without an external storage connector.

```sh
make defconfig-kmesh-family-residency-cuda
export KMESH_RESIDENCY_SOURCE=/path/to/vllm
export KMESH_RESIDENCY_MODEL=/path/to/model-snapshot
export KMESH_RESIDENCY_OUTPUT=/path/to/fresh-serving-results
make kmesh-residency-plan
make KMESH_RESIDENCY_PY=/path/to/prepared/python

KMESH_TRACE_MODE=invocations KMESH_TRACE_ROOT=/path/to/fresh-serving-results \
  KMESH_TRACE_OUTPUT=/path/to/fresh-capture-audit make kmesh-trace-audit
```

Use the Qwen/Qwen2.5-0.5B-Instruct snapshot at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The model capture requires a CUDA GPU;
plan and portable-contract commands do not. It alternates stock/held order
across three samples per arm, checks exact generated tokens and chosen
log-probabilities, checks physical ancestor block identity, and requires every
non-null reference to drain. It records actual monotonic invocation intervals,
explicit lineage, exact tokens, allocations, evictions, free counts and family
references. Partial captures remain on failure. Observation hooks are active,
so invocation durations are not a latency benchmark. Expiry and cancellation
are allocator-level checks; the model capture exercises normal completion.

The capture and controller are a mechanism prototype. Even a passing controlled
capture does not qualify production traces, DMA-BUF performance, storage-read
savings or cross-engine sharing. The general trace auditor deliberately keeps
`pressure_replay_qualified` false: this small fixed test is not a universal
physical-cache evidence validator.

Verify saved model output and physical observations, then export a compact
result without request identities or token sequences:

```sh
python -m tools.reproduce.kmesh_residency.publish /path/to/serving-results \
  --output public-results.json
```

This CPU-only exporter checks paired continuations, lineage, held-block alias,
reference counts and drained ownership. It preserves the actual capture file
hashes. The exported result remains a controlled-mechanism qualification; it
makes no production-workload or latency claim.
