# Native packed-cache evaluation

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
and `engine`. The engine object contains the standard vLLM LLM configuration,
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
same output directory to append the remaining questions. The fixed 200-case
pilot is part of the full 1,319, not independent confirmation. Errors,
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
Request percentiles are descriptive, and independent process repetitions
remain the uncertainty units for speed. `--stage profile` collects separate
Chrome traces and actual GPU kernel names. Finite-logit checks apply to
quality, not the timed path. Scheduler observation remains enabled.

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
