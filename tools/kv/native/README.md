# Native packed-cache evaluation

For the current Qwen3-8B recipe, start with the [installation guide](../../../docs/asymmetric-install.md). `october_setup --recipe current` selects the measured mixed fresh-prefill route; the default `historical-compact` preserves the earlier stack. The current route accepts an eligible fresh-prefill suffix alongside decode. The historical fallback description below applies only to the preceding compact source.

These tools evaluate a caller-supplied Qwen3-8B-AWQ checkpoint and native
16-bit K/V, 16-bit K with FP8 V, and symmetric FP8 caches. They preserve the
selected serving stack's attention, allocation and dispatch rules. They do
not force FA3 or change kernels. The specialized inventory expects 36 layers,
32 query heads, eight KV heads and head dimension 128. Native K is FP16.

Use the complete matching FlashInfer and vLLM source branches. Supply an
immutable protocol, checkpoint/tokenizer revisions, separate frozen per-layer
K/V scales, and public-runner NLL/retrieval manifests. The reference JSONL files
can be generated with `tools.kv.prequantized`; no private repository is needed.
GSM8K questions come from the pinned public dataset. `gsm8k.build` renders
stock and explicitly instructed chat adaptations with thinking disabled.
The latter may be selected using training examples before test scoring.
Extraction uses the pinned harness regular expressions. Numeric comparison
removes commas and dollar signs and uses finite decimal equality, so this
is a labelled chat/numeric adaptation rather than an unmodified harness score.

The command-line implementation is provided here. CUDA execution must pass
the supplied runtime gates on the selected stack before a result is claimed.
The caller-owned input directory contains:

- `protocol.json`, `qwen.json.gz`, `scales.json`;
- `vllm/` and `flashinfer/`, exact clean source checkouts;
- `reference/{k16_v16,k16_v8,k8_v8}.jsonl`, frozen QDQ reference rows;
- optional `historical-native/<arm>/nll.jsonl`, separately labelled historical native rows;
- a frozen GSM8K task JSON, and the committed `task-freeze.json` plus
  `SCORING-UNLOCKED.json` before pilot or full test scoring.

`protocol.json` records `source_sha`, `backend_sources`, `torch`,
`runner_files_sha256`, `inputs_sha256`, `torch_allocator_reserved_cap_bytes`
`packed_execution`, and `engine`. The packed execution object freezes the
linear method, kernel class, weight bit width and profiler operator marker.
New serving stacks may repack AWQ weights for Machete without replacing them
with persistent full-precision weights. The engine object contains the standard vLLM LLM configuration,
including explicit FlashInfer attention, AWQ, FP16, TP=1, eager execution,
block size 16 and no automatic KV scale calculation. Hash every Python file
in this package. Input hashes bind all numerical references and manifests.
Loaded package locations, exact Git revisions and code hashes are checked
before constructing the engine. Dependency/binary receipts must also verify
that installed extensions match the selected compiled sources.

Example stages, after installing those exact dependencies and source trees:

```sh
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
python -m tools.kv.native.runner --inputs inputs --output results/writer \
    --arm k16_v8 --stage writer
cp results/writer/geometry.json inputs/geometry.json
python -m tools.kv.native.runner --inputs inputs --output results/quality \
    --arm k16_v16 --stage bridge
python -m tools.kv.native.runner --inputs inputs --output results/quality \
    --arm k16_v16 --stage quality
python -m tools.kv.native.runner --inputs inputs --output results/dev \
    --arm k16_v16 --stage gsm8k --task inputs/gsm8k-stock.json \
    --subset development --batch 8
```

Repeat the complete native screen for each cache arm. Exact repeats and
finite scored logits are required. Raw prefill prediction loss is retained,
then excluded from the 128 scored cached-decode targets. The original
numerical bridge tolerances remain 0.01 mean absolute target-loss error,
0.10 maximum error and 99% top-1 agreement. Bridge failures stay separate
from full preservation against the new engine's own baseline.

After development validates the selected prompt, parser and cap, commit the
frozen task and protocol. `task-freeze.json` binds its file hash and
`primary_cases: 1319`; `SCORING-UNLOCKED.json` records that protocol commit.
Run `--stage gsm8k --subset pilot` in every arm, then `--subset full` in the
same output directory to append the remaining questions. Instrumentation
uses a distinct atomically reserved dispatch subdirectory per subset and
invocation, so pilot and interrupted smoke receipts remain intact. The fixed
200-case pilot is part of the full 1,319, not independent confirmation. Errors,
truncations and strict parser failures remain incorrect in the denominator.
Strict and flexible extraction are reported separately. The useful-baseline
floor is 50%; the primary conservative loss-transition margin is two points.
A zero-discordance bootstrap interval does not imply zero population risk.

```sh
python -m tools.kv.native.analyze --quality results/quality \
    --gsm results/gsm --task inputs/gsm8k-frozen.json \
    --protocol inputs/protocol.json --output inputs/qualification.json
python -m tools.kv.native.runner --inputs inputs --output results/timing-0 \
    --arm k16_v8 --stage timing --repeat 0
```

Timing requires complete qualified native and GSM8K matrices. Use fresh
process repetitions in a frozen randomized arm order. `--blocks` fixes slot
capacity; `--cache-bytes` fixes the cache budget. Neither setting alone proves
an equal-total-memory capacity gain. `--single --batch N` measures one 8K
shape; the default timing grid uses 2K/8K and batches 1/8/32. Generation is
synthetic, forced to 128 tokens, with per-shape warmup excluded. Actual
scheduler admissions and full-active decode steps are recorded. At least
32 full-active steps are required to qualify multi-request concurrency.
Request latency uses observed frontend completion minus wall-clock arrival.
Time to first token uses the engine's frontend latency field; core monotonic
first/last token timestamps supply per-request mean decode intervals.
The effective prefill rate includes admissions and host overhead.
Request percentiles are descriptive, and independent process repetitions
remain the uncertainty units for speed. `--stage profile` collects separate
Chrome traces and actual GPU kernel names. Finite-logit checks apply to
quality, not the timed path. Scheduler observation remains enabled.

Cache-read receipts observe the actual FlashInfer wrapper arguments for
all cache modes, including symmetric storage reinterpreted as FP8. Every
layer must appear in both prefill and decode with the expected physical
dtypes and original model storage identities before qualification. No
asymmetric-only helper is used as a proxy for symmetric cache reads.

Capture continuous external GPU memory telemetry. The saved cold-start
Torch peak precedes the warm-peak reset. Allocated bytes, reserved pools,
backing cache storage and driver memory are distinct measurements. Reconcile
all views by their actual storage identity; do not count shared K/V storage
twice. `storage.reconcile` checks disjoint occupied byte intervals and padding.
Use actual resident requests, no preemption, a predeclared latency limit and
a common total-memory envelope for a capacity claim. `retrieval-batch` adds
a separate full-prompt regression with automatic prefix caching disabled at the claimed schedule;
it is not labelled as a deferred-query prefix-reuse measurement.

For a legacy API pilot, set `VLLM_ENABLE_V1_MULTIPROCESSING=1` and
use `--legacy`. This retains eight maximum sequences, the 64K scheduling
budget and 8K/B1/B8 shapes. A separate `legacy-qualification.json` must
record the configuration bridge before timing that mode. Scheduler labels
are published atomically to the engine process; the observer is installed
there through the uniprocess executor callback. Keep these results separate
from controlled in-process active-concurrency measurements.

Set `engine.worker_cls` to
`tools.kv.native.memory_worker.ObservedWorker` to record device initialization
before model load, raw cache allocation and typed-view materialization. It
delegates to the pinned worker and allocator without changing their results.
The per-process Torch allocator ceiling is applied in the actual worker,
including when the engine runs in a separate process. Driver memory and
non-Torch allocations remain separate measurements.

`prefill.py` captures complete single-request paged prefill inputs from selected
model layers. The component path targets one fixed geometry: 8,192 tokens,
16-token pages, 32 query heads, eight K/V heads, and a head dimension of 128.
Its component runner compares explicit attention backends on the
same query, key, and stored value bytes, including causal page tails. Larger
component batches repeat a captured request in distinct page storage. They
measure a kernel workload; replicas are not independent quality observations.
Fixture copies and workspace allocation stay outside the measured operation.
The inclusive duration includes planning, attention, and required output scaling;
CUDA event and host planning times are also retained. Compilation and separate
profiler calls do not enter the unprofiled measurements. Inspect each trace for
device kernels before using it as dispatch evidence. Numerical checks retain
full-tensor hashes and fixed output samples, rather than every output tensor.

The value-expansion diagnostic uses transient FP16 scratch for the pages
read by each layer. It includes the cast and the scale multiply after the native
attention output has rounded to its output dtype. Its compact page layout does
not include gathering from a serving engine's full pool. It cannot establish a
serving speedup or the workspace required by a serving implementation.

`runner.timing_record(..., output_tokens=512)` changes the forced output length
and derives token accounting and mean decode intervals from that length. The
default remains 128. Compare complete-request rates at each length separately.

`prefill_evidence.py` captures all explicit-backend dispatch calls in one
profiler session after timing. Synchronized named regions bind each call to
complete device kernel events. The combined trace is retained with derived
per-cell slices; a missing expected attention family fails the evidence gate.
No profiled duration enters the unprofiled timing verdict.

The expansion diagnostic allocates one flat V scratch buffer per fixed batch,
sized from actual read pages, and reuses its backing storage across layers.
Each case releases its wrappers and plan closures before the next layer.
Pointer/size records, post-case allocator state, and reset per-case peaks make
that lifetime auditable. Component timing still excludes serving-pool gathering
and scratch allocation. It does not qualify a serving implementation.

`compact.py` checks a caller-supplied bounded serving implementation against
stored quantized cache bytes. It places saved inputs in sparse, repeated,
page-zero layouts, checks exact gathering and untouched guards, and compares
complete attention outputs at fixed tolerances. Its model validator checks
every routed layer against the original mixed-cache attention wrapper.
Route observation joins qualification requests to the actual scheduler.
These hooks run outside serving timing. The worker receipt reports allocated
workspace capacity separately from its active page payload and retains
storage pointers so cross-layer reuse can be checked.

The weight observer also accepts an explicit `fp16` weight format. That path
requires unpacked FP16 parameters, 144 unquantized linear modules and two
unquantized embedding modules. It verifies matrix-multiply dispatch separately
from cache precision and rejects packed weight kernels. This loader check does
not by itself qualify a new checkpoint's cache scales or model quality.


The accepted compact-prefill serving configuration uses an SM90 GPU, one
worker and stream, FP16 compute, NHD layout, 16-token pages, 32 query heads,
eight K/V heads and head dimension 128. Persistent K is FP16 and V is E4M3FN.
Freeze cache scales before graph capture. Keep `VLLM_COMPACT_PREFILL=1` an
explicit opt-in; the existing mixed-cache implementation remains the default
and fallback.

Use compilation mode 0 with `FULL_DECODE_ONLY` graphs and capture sizes
`[1, 8, 32, 44, 45, 59, 60, 96]`, at most 96 sequences and 65,536 batched tokens,
with prefix caching and chunked prefill disabled. Eligible calls contain
complete whole prompts of at most 8,192 tokens and at most 4,096 referenced
page occurrences. Mixed decode/prefill, partial context, unsupported geometry
and excess pages take the fallback. This recipe does not qualify BF16,
parallel workers, other graph modes, sliding attention or attention sinks.

The implementation has no general graph-mode rejection guard. An enabled
unsupported model can also allocate its workspace before falling back. Use
only the accepted configuration; metadata eligibility tests alone do not
qualify broader deployment. The workspace retains 296.140625 MiB of unique
constant GPU storage, including its 256 MiB K/V backing, and a separate 8 MiB
pinned CPU buffer. Per-plan metadata and allocator overhead are additional.
Automatic pool sizing accounts for the initial workspace allocation delta;
an explicit cache-byte pool bypasses that sizing path.
