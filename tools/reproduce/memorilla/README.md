# Memorilla checkpoint evaluation and stress replay

This runner uses the released checkpoints. It never trains. The original
control runs the author's unchanged evaluation driver, mmap embedding store,
reader, prompt construction, frozen decoder and scorers. The storage adapter
uses the same driver with document collections supplied by LMCache. It does
not convert Memorilla into a KV prefix or replace its learned reader.

Start from the public knlp branch `2026-10-23-feature-context-artifacts`.
Select one preset:

| Defconfig | Purpose |
| --- | --- |
| `memorilla-author` | Original source and pinned author CUDA dependencies. |
| `memorilla-port-control` | Original mmap collections on a separately qualified engine. Record runtime differences separately. |
| `memorilla-lmcache` | Same engine with the common cartridge catalog and leased LMCache host storage. |
| `memorilla-port-batch-replay`, `memorilla-lmcache-batch-replay` | Complete-batch admission with V1 multiprocessing retained. Requires level-0 scheduling pause and enqueue APIs. |
| `memorilla-port-reference`, `memorilla-lmcache-reference` | Single-request reference with no prefix reuse. A numerical control rather than a throughput preset. |

```sh
make defconfig DEFCONFIG=memorilla-author
make olddefconfig
make menuconfig
# Select a persistent KNLP_MEM_ROOT and a benchmark (pubmedqa or all).
make memorilla-fetch MEMORILLA_PYTHON=/path/to/python
```

Fetching needs `huggingface_hub>=0.36,<1` and Git. It downloads only selected
test splits, their precomputed embeddings, released readers, decoder weights,
and the scoring encoder when required. It saves exact revision and SHA-256
inventories in `inputs.json`. No training data or training job is needed.
The fetch source pins are in `run.py`; changed or mismatched source/assets
fail validation. Keep the work root and its input manifest together.

For the original author runtime, use a separate Python 3.11 environment and
install the fetched project using its `pyproject.toml`. It pins Torch 2.7.0,
vLLM 0.9.1, Transformers 4.53.3 and datasets 4.1.1. Evaluation does not need
FlashAttention. The author's supported recipes use CUDA and an 80 GiB GPU.
An environment with other versions must use the explicitly separate port
control; it cannot pass the original-runtime doctor.

```sh
/path/to/python -m pip install /path/to/work/source/memorilla-author
make memorilla-doctor MEMORILLA_PYTHON=/path/to/python
make memorilla-benchmark MEMORILLA_PYTHON=/path/to/python
```

Plain `make` performs readiness checks only. `memorilla-benchmark` explicitly
runs inference. Each invocation gets a new run directory with resolved
configuration, input/runtime identities, stdout/stderr, predictions, original
metrics and deltas from the paper. Failures retain partial evidence. The
complete 100-row PubMedQA test is the default, not the earlier eight-row
reader-only probe. Set `KNLP_MEM_BENCHMARK="all"` for all eight benchmarks.
Each benchmark starts a separate engine so its released checkpoint is explicit.

## Public adapter dependencies

The port control and LMCache presets fetch these source branches, then detach
at the exact commits below. Install LMCache in a separate environment using its supported instructions
for the selected Torch/GPU build. The catalog consists of independent Python
modules loaded explicitly from the pinned vLLM checkout under their normal
module names. Only the three catalog files are loaded; the engine package
search path stays unchanged. An existing catalog at another location is
rejected. This does not replace or pretend to rebuild an installed vLLM
engine. Build/install the desired engine separately; preserve its exact runtime
identity. Install Memorilla's remaining dependencies without letting its CUDA
vLLM pin replace that explicitly selected runtime.

| Repository | Public branch | Exact commit |
| --- | --- | --- |
| [vLLM](https://github.com/mcgrof/vllm/tree/2026-10-23-feature-context-artifacts) | `2026-10-23-feature-context-artifacts` | `21a71b62a2a72a12dcf23e44c3fac21143d7b9ee` |
| [LMCache](https://github.com/mcgrof/LMCache/tree/2026-10-23-feature-context-artifacts) | `2026-10-23-feature-context-artifacts` | `d8d0fd7fa5869880384eea883c8f7b59a64cf29f` |

The worker checks catalog and LMCache source revisions and tracked changes,
and records the separately imported engine path/version. Full fork-engine
qualification is distinct from testing these catalog modules.
The recorded port uses Torch 2.9.1 with ROCm 6.4, Transformers 4.57.6
and a locally patched V1 engine derived from vLLM 0.18.1rc1. This is a
qualification of that recorded runtime; the pause/enqueue wrapper uses
public APIs, but other engine builds still need matched control runs.

The storage measurements qualify LMCache's real host allocator and
external-artifact component. They do not qualify its complete package graph
against the older author dependencies. Keep the original native control
in its separate pinned environment. A component import or an installation
that omits dependency resolution does not establish full SDK compatibility.
Full StorageManager and token-KV connector integration remain separate gates.
The recorded port qualification exercises PubMedQA's polarity scorer.
Choice-scored tasks additionally require the author encoder API
(`LLM(task="embed")`), which newer engines may remove. The original runtime
preset retains that API; a ported engine needs separate encoder/scorer
qualification before running the complete eight-benchmark suite.
LMCache arena size is `KNLP_MEM_POOL_BYTES`; this bounds artifact allocations,
not total application memory or the Linux page cache. Existing precomputed
collections are converted and verified before generation. Publication and
conversion time are included in whole-worker time, but not replay batch time.
The unchanged author collator expects owned CPU tensors, so this adapter copies
out of a lease before returning a collection. That ownership copy is explicit.
Question embeddings retain the author's mmap path. No automatic KV attachment,
remote StorageManager routing or speed advantage is claimed.

## Reusable stress command

The [system qualification](../../../docs/context-artifacts.html#system-qualification)
reports complete public-wheel dependency checks, concurrent checkpoint
controls, storage pressure, cancellation and recovery. Mmap remains the
performance baseline: a working catalog and adapter do not establish a speed
advantage. The new port's checks are separate from the original paper scores.

The separate CPU bank stressor uses the same native mmap implementation,
catalog and real LMCache host allocator as checkpoint inference:

```sh
make memorilla-storage-stress MEMORILLA_PYTHON=/path/to/python \
  STORAGE_BENCHMARK=narrativeqa STORAGE_CLIENTS=8 \
  STORAGE_REQUESTS=1024 STORAGE_OUTPUT=/path/to/new/results
make memorilla-storage-faults MEMORILLA_PYTHON=/path/to/python \
  STORAGE_OUTPUT=/path/to/new/fault-fixtures
```

Each CPU bank request validates all bytes and geometry against the original
store. Select a new output directory for each run; existing captures are
rejected before loading inputs. Latency includes the full-byte validating consumer, so a lazy mmap
view is not reported as a completed read. Lookup and consumer times are also
separate. Concurrent capacity exhaustion uses bounded, measured backpressure.
Queued-future cancellation drains reads already running. The fault suite
tests active leases, coalesced misses, capacity, corruption, short/missing
files and identities, and process restart after an interrupted allocated read.
These are storage-service checks, separate from decoder request cancellation.

`STORAGE_CACHE=hot` warms payload files; `cold` requests per-file eviction
only for files inside the configured work root. It never drops the global
page cache. Actual page residency is measured with `mincore`, and the report
includes process and cgroup memory, faults and process I/O counters. A cold
initial cache does not make every subsequent read cold. Use a process memory
budget smaller than the working set for sustained pressure; retain its actual
cgroup limits and counters. No kernel modification is required.

### Clean public-wheel port environment

`requirements-port.lock` pins the qualified Linux Python 3.13 port environment;
`requirements-port.in` contains the top-level resolution inputs. Use a fresh
environment with a C++ compiler and Rust build tools. LMCache's
supported `NO_GPU_EXT=1` host build includes native host extensions and full
runtime dependency resolution. This profile does not qualify GPU KV kernels.

```sh
uv venv --python 3.13 /path/to/env
uv pip install --python /path/to/env/bin/python torch==2.13.0 \
  setuptools==80.10.2 setuptools-scm==10.3.4 ninja==1.13.2 wheel==0.48.0
NO_GPU_EXT=1 uv pip sync --python /path/to/env/bin/python \
  --no-build-isolation tools/reproduce/memorilla/requirements-port.lock
uv pip check --python /path/to/env/bin/python
```

To refresh the lock, use `NO_GPU_EXT=1 uv pip compile --no-build-isolation`
with `requirements-port.in`, then qualify the resulting environment again.
Compile under the same build profile as synchronization. No
dependency-conflict suppression is part of qualification. The runner checks
normal installed LMCache wheels against every Python implementation file in
the pinned public checkout, rejects unexpected Python files and records
extension hashes. It also accepts an
unchanged editable checkout. Memorilla model/evaluation source remains pinned
and unmodified; this is explicitly a port rather than its original packaging.

```sh
make memorilla-stress MEMORILLA_PYTHON=/path/to/python
# Replay the same saved requests with another preset or batch size:
make memorilla-stress TRACE=/path/to/control/trace.json \
  MEMORILLA_PYTHON=/path/to/python
make memorilla-compare REFERENCE=/path/to/control/requests.jsonl \
  OBSERVED=/path/to/adapter/requests.jsonl MEMORILLA_PYTHON=/path/to/python
```

The same stress command works with every preset. Configure request count,
seed, question working set, batch size, and sequential/shuffled/random/hot
accesses in menuconfig. A shuffle visits every selected question before
repeating; hot traffic puts about 90% of requests in the first 10% of questions.
Working-set size counts questions, which can share collections; it is not a
byte budget. `KNLP_MEM_BANK_SWAP=y` holds question zero fixed while changing
the document bank. This is a synthetic input-isolation test, not a scored
paper example. It uses the same original collator, reader and decoder. `KNLP_MEM_EVICT_EVERY` explicitly clears adapter residency every N
batches and leaves the Linux page cache unchanged.

`KNLP_MEM_CLIENTS` controls parallel bank fetches before each complete
decoder batch in port stress mode. The original collator consumes results
in request order, including repeated collections. The learned reader and
decoder stay on the caller thread, so storage completion order cannot change
decoder batch composition. This tests storage contention feeding an offline
engine; it does not simulate independent online decoder clients. Original
author mode requires one bank client. Request receipts separate bank
preparation, collation and model-call times; diagnostic hashing adds overhead
to the model-call interval and must be disabled for timing comparisons.

A `TRACE` replay validates its row bounds, collection, checkpoint, data and
prompt identities before running. It does not silently regenerate a new
schedule. Every replay saves its complete request schedule and append-only predictions,
with stable request IDs, row/collection IDs and input digests. Comparisons
reject duplicate IDs, missing requests, changed inputs and changed predictions.
Keep seed, pattern, request count and working set fixed when changing storage
or batching. Repeated-question output changes are reported.
`KNLP_MEM_ENGINE_JSON` accepts only explicit port-runtime options
(`enforce_eager`, `attention_config`, `max_num_seqs`, `enable_prefix_caching`). They are recorded in
provenance; original author mode rejects them. Model, prompts and sampling
cannot be overridden through this setting. Greedy generation
can differ numerically across batch shapes and devices; investigate changes
before treating an exact mismatch as evidence of wrong-bank reuse.

### Reproducible offline scheduling

The port and LMCache presets now default to
`KNLP_MEM_SCHEDULING="inprocess"`. The worker sets
`VLLM_ENABLE_V1_MULTIPROCESSING=0` before importing vLLM. This follows the
[versioned vLLM offline reproducibility guidance](https://docs.vllm.ai/en/v0.18.1/usage/reproducibility/).
The original-author preset keeps `runtime` scheduling and rejects overrides.
Historical resolved captures without a scheduling key also keep runtime
behavior. Use `runtime` explicitly to reproduce the asynchronous admission
control on a ported engine; an exact mismatch remains a failing comparison.

In V1 multiprocessing mode, the engine can execute a request while the caller
is still submitting its neighbors. Those timing-dependent execution batches
can change floating-point results even with identical prompt embeddings and
greedy sampling. The `batch` mode keeps multiprocessing enabled. It pauses
scheduling at level zero, enqueues the complete call, resumes scheduling and
waits for completion. It neither offloads weights nor clears the prefix cache.
It requires an idle engine owned exclusively by this offline caller. An API
failure invalidates the wrapper; discard that engine rather than execute a
partially submitted batch. Older runtimes without these APIs fail explicitly.

Repeatable scheduling does not imply batch-independent model arithmetic.
Changing neighboring questions or document-bank padding can change reader
and decoder numerics. The reference presets require batch size one,
in-process scheduling, one engine sequence and disabled prefix caching.
They remove neighbors and cached prefixes from this correctness control.
LMCache document-bank residency remains enabled; only decoder prefix reuse
is disabled.
Configuration and effective runtime checks reject a relabeled reference
that does not apply those settings. They preserve the model, prompt,
generation budget and scorer; their scores need not equal a batch-32 run.

Each run records the scheduling selection, effective process model,
prefix-cache setting and engine limits. Compare storage paths with the same
scheduling, batch size and engine settings. No mode promises equal results
across hardware, engine versions or arbitrary batch shapes. vLLM's separate
batch-invariance feature requires its own backend qualification.

Reports include synchronous batch p50/p95/p99, completed requests per second,
whole-worker time, driver RSS/page faults and process I/O observations, plus
LMCache logical range loads/hits. Batch latency is not individual latency.
`KNLP_MEM_DIAGNOSTICS=y` additionally hashes collated tensors, per-request
reader outputs and exact decoder prompt embeddings. It synchronizes GPU-to-CPU
copies and changes timing, so use it to locate correctness differences, not
to claim performance. Comparison reports these tensor differences separately;
equal decoder inputs do not override a failed output-string comparison.
Driver usage excludes engine-child totals. Page-cache state is unknown;
logical bytes and process counters are not attributed NVMe commands.

This first stressor covers sustained synchronous batched inference, workload
order, shared collections and adapter eviction. It does not yet implement
paced concurrent clients, asynchronous cancellation, process-kill recovery,
corruption injection or automatic resource sweeps. The separate artifact
integration suite tests corruption, short files, concurrent misses and lease
pressure. These should later share the inference trace, rather than silently
relabeling transport-only checks as end-to-end stress results.

## What the paper result contains

[Paper v1](https://arxiv.org/html/2610.21375v1), Table 11 and Appendix D,
provide these primary targets. Scores below are percentages, except ROUGE-L,
which the code also reports on a 0–100 scale.

| Benchmark | Questions | Primary metric | Memorilla target |
| --- | ---: | --- | ---: |
| PubMedQA | 100 | Accuracy | 59.00 |
| FactKG | 9,041 | Accuracy | 84.33 |
| TriviaQA | 500 | Exact match | 57.00 |
| NarrativeQA | 10,557 | ROUGE-L | 19.31 |
| LaMP-4 | 1,925 | ROUGE-L | 18.15 |
| LaMP-7 | 1,500 | ROUGE-L | 48.37 |
| PersonaMem-v2 | 5,000 | Short-answer accuracy | 44.60 |
| PersonalizationV4 | 4,237 | Short-answer accuracy | 66.65 |

There are 32,860 questions in total. The code additionally writes F1, BLEU,
ROUGE and METEOR where registered, and PV4 hard-subset accuracy. Choice scoring
embeds generated answers and candidate answers; it does not ask an external
judge. Released `test` files include the paper's selected development/test
partitions. Validate row counts rather than changing their splits or sampling.

Six benchmarks use released `stage3`; PMv2 and PV4 use their named released
folders. The paper describes further personalization training from the
multi-task checkpoint, but the released README labels those folders as trained
from Stage 1 and labels stage3 as epoch-00. Therefore these commands reproduce
*released-checkpoint inference*. Exact checkpoint equivalence to every paper
row remains unverified. Deltas are reported honestly; passing execution is not
proof of matching the table. Preserve checkpoint bytes and investigate a gap
before changing prompts, scoring, generation budgets or checkpoints.
Provenance and paper-comparison receipts include the published release label
and a link to the [pinned model card](https://huggingface.co/memorilla/Memorilla-Qwen3-8B/blob/f6b15af56aa687861c8d2d03b1cd8f7979881361/README.md).
Exact paper-checkpoint verification needs the corresponding authors' run
manifest; the release label alone does not establish which weights produced
the paper's scores.

The author's evaluation guide reports about 15 minutes on one 80 GiB GPU after
inputs are cached. Our per-benchmark process isolation adds startup overhead;
that figure is not a prediction for another GPU. The eight-benchmark suite
excludes retraining, ablations that require new weights, other methods' full
baseline sweeps, TextWorld RL evaluation, and the separate accelerator latency curve.
