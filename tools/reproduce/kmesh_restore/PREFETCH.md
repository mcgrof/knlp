# Does early loading help after demand already reads once?

This experiment compares two ways to restore the same shared context. The
demand control starts at request arrival. The early control starts before
arrival. Both retain the external objects and join overlapping fills, so
each cold group must read exactly sixteen 3 MiB chunks, or 48 MiB. A trial
that changes the amount of storage work fails qualification.

The workload uses the pinned Qwen2.5-0.5B-Instruct model from the
[restoration instructions](README.md#runtime-and-sources), BF16 arithmetic,
4,096 shared tokens, distinct 512-token tails and eight greedy output tokens.
The engine and external cache each have a 512 MiB budget. This is a synthetic
exact-prefix experiment without memory pressure or task-quality evaluation.

Use vLLM `1b76bf9dd9d52efbd75fb89bf2619898d8f01233` and LMCache
`fb71fb40abe8ca3fdbe2b0a6b4823332d1a76f37`. The latter adds optional overlap
joining before backend lookup. Each waiting query then acquires its own
references to the retained objects. Disjoint queries can still proceed
concurrently. The qualified CUDA/native build and patched kernel requirements
in the restoration instructions apply. Host staging and GPU DMA-BUF are
separate transfer paths; an unsupported direct path must fail.

After preparing the clean restore runtime described there, advance its
LMCache checkout to this profile. `LMCACHE_SOURCE` names that checkout:

```sh
git -C "$LMCACHE_SOURCE" fetch --depth=1 origin \
  fb71fb40abe8ca3fdbe2b0a6b4823332d1a76f37
git -C "$LMCACHE_SOURCE" diff --exit-code \
  007a3388331209737ee19c095e430907fc467952 \
  fb71fb40abe8ca3fdbe2b0a6b4823332d1a76f37 \
  -- csrc rust setup_extensions setup.py pyproject.toml
git -C "$LMCACHE_SOURCE" checkout --detach \
  fb71fb40abe8ca3fdbe2b0a6b4823332d1a76f37
```

The compared native sources and build inputs are identical at these pins;
the loading coordination changes are Python code. The vLLM pin stays the
same. Preserve the compiled native hashes with each new capture.

Freeze the arrival deadline 250 ms after each trace origin, before starting
preparation. Keep engine calls on one thread and perform HTTP preparation
in a background owner. Check for due children before each engine step. A
synchronous step can still delay submission; the deadline-based endpoints
include that delay. If children have already been submitted when external
preparation finishes, skip only the unsubmitted engine warm request and
drain the accepted storage work. Such a hint is not ready by arrival.

Measure token delivery at the synchronous engine caller. Trial parameters
request cumulative output; offline preparation receives a deep copy because
`LLM.generate()` selects final-only delivery on its argument. Each child
must first expose exactly one token, with an observation strictly before
its eighth token. Final-response-only delivery fails the clock gate.
These host observations do not measure the device's internal token-ready
time or a remote client's network latency. Arena setup precedes timed groups.

The controls are retained demand, early loading at 0/25/100/250 ms, a cold
declaration without warming, external memory already resident, and engine
cache already resident. The host-resident control starts with the context
in host DRAM and performs no trial SSD reads. Its direct-path counterpart
starts in the external GPU arena. Both placement controls charge the earlier
48 MiB fill separately. The demand arm issues no one-token warm request.
All real warm requests, including an unused hint, remain in the ledger.

Qualification checks exact generated tokens and zero chosen-logprob delta,
actual restored model-cache digests, one ready physical engine ancestor,
native payload paths and complete ownership drainage. The pilot uses two
fresh engine blocks per path, three warmups and five measured groups per
control/block. It selects the shortest common lead that is ready by scheduled
arrival in at least nine of ten measured groups in each path. If none
qualifies, it retains 250 ms as a failed readiness limit.

Confirmation compares that frozen lead against demand, with four and eight
children on both paths. Each combination uses ten fresh paired engine
blocks, three warmups and twenty measured groups per arm/block. There are
no exclusions or outcome-dependent extensions. Confidence intervals resample
paired process blocks with 10,000 draws and seed 20261020. They are nominal
per-endpoint intervals; children within a group are not independent repeats.

Run the phases in order in a matching environment. Set `MODEL_PATH`,
`VLLM_SOURCE` and `LMCACHE_SOURCE` to the prepared paths. Put those clean
source checkouts first when importing the native modules:

```sh
export PYTHONPATH="$LMCACHE_SOURCE:$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/prefetch
python tools/reproduce/kmesh_restore/prefetch_run.py \
  --phase qualification --output results/prefetch/qualification \
  --model "$MODEL_PATH" --vllm-source "$VLLM_SOURCE" \
  --lmcache-source "$LMCACHE_SOURCE" --gpu-provider vulkan_rm
python tools/reproduce/kmesh_restore/prefetch_run.py \
  --phase pilot --output results/prefetch/pilot \
  --qualified results/prefetch/qualification \
  --model "$MODEL_PATH" --vllm-source "$VLLM_SOURCE" \
  --lmcache-source "$LMCACHE_SOURCE" --gpu-provider vulkan_rm
python tools/reproduce/kmesh_restore/prefetch_run.py \
  --phase confirmation --output results/prefetch/confirmation \
  --qualified results/prefetch/qualification --pilot results/prefetch/pilot \
  --model "$MODEL_PATH" --vllm-source "$VLLM_SOURCE" \
  --lmcache-source "$LMCACHE_SOURCE" --gpu-provider vulkan_rm
```

Each phase refuses an existing output directory. Later phases recheck the
complete admitting raw capture and require matching source, native binaries,
model, allocation provider and runtime. A failed or incomplete cell rejects
the campaign. `CONFIG.json` freezes the sampling and late-hint rules before
requests; `RESULT.json` is written only after all gates pass. Raw timings,
warmups, failed receipts, native operations and process identities remain
available beside the all-child `RESULTS.csv`.

The defconfig exposes the same phase commands. Select the Python interpreter
from the prepared environment, then supply its source and model paths:

```sh
make defconfig-kmesh-prefetch-cuda
export KMESH_RESTORE_PY="$(command -v python)"
export KMESH_RESTORE_MODEL_DIR="$MODEL_PATH"
export KMESH_RESTORE_VLLM_SOURCE="$VLLM_SOURCE"
export KMESH_RESTORE_LMCACHE_SOURCE="$LMCACHE_SOURCE"
make kmesh-restore-plan
make kmesh-restore
KMESH_RESTORE_PHASE=pilot \
  KMESH_RESTORE_OUTPUT=results/kmesh-prefetch/pilot make kmesh-restore
KMESH_RESTORE_PHASE=confirmation \
  KMESH_RESTORE_OUTPUT=results/kmesh-prefetch/confirmation make kmesh-restore
```

These commands use the default qualification and pilot capture paths above.
For other locations, set `KMESH_RESTORE_QUALIFIED` and
`KMESH_RESTORE_PILOT` to the actual captures. Set a fresh output for each
phase. `make kmesh-restore-plan` is CPU-only and reports missing inputs.
`make kmesh-restore-verify` rechecks the selected output on CPU.

The first endpoint is request A's first observed output token, timed from
scheduled arrival. A is the first child submitted. The second endpoint is
the last child's eighth observed output token, also timed from scheduled
arrival. Separate clocks retain actual-submission waits, every child,
terminal acknowledgements, preparation and later drainage. Engine-reported
`ttft_ms` remains a diagnostic. Eight tokens need not form a useful answer.

Quiet measured groups have no allocation hooks or residency polling. A
separate diagnostic warmup records allocation/free boundaries and sampled
external control stages. Engine occupancy peaks use those boundaries;
external sampled peaks can miss transients. External payload byte-seconds
are lower/upper bounds from known completion and clear boundaries, rather
than an exact allocation-time integral. Configured arenas, backed bytes,
occupied bytes and the retained 48 MiB copy have separate fields.
The `registered_host_arena_bytes` and `external_gpu_arena_bytes` receipt
fields describe the selected 512 MiB arena configuration. They are not
independent measurements of registered length or pinned-allocation peaks.
Observed external backing and the actual engine pool have separate fields.

Recheck a saved phase without importing a device runtime:

```sh
python tools/reproduce/kmesh_restore/prefetch_campaign.py \
  results/prefetch/confirmation --report results/prefetch/rechecked.json \
  --csv results/prefetch/rechecked.csv
```

A useful engineering result requires at least 5% point improvement in one
primary endpoint, a paired 95% interval excluding zero, and an interval for
the other endpoint excluding regression larger than 5%. Report absolute
milliseconds, placement references, warm costs and failed readiness too.
Mixed or wide intervals remain mixed or inconclusive. Backend payload
counts do not establish physical NVMe command counts or general throughput.

After all three phases pass, export aggregates for review:

```sh
python tools/reproduce/kmesh_restore/prefetch_publish.py \
  --capture results/prefetch \
  --harness-commit "$MEASURED_KNLP_COMMIT" --repository . \
  --output public-prefetch-results.json
```

Set `MEASURED_KNLP_COMMIT` to the full commit ID whose harness actually ran.
The exporter authenticates every captured Python-file hash and the model
manifest against that Git tree. It also rechecks the complete raw phases,
source pins, native binaries and runtime. The exported file keeps declared
scientific fields and native library basenames. Raw paths, process and
request identities stay in the original capture. Offline checks establish
capture consistency; they cannot independently authenticate GPU execution.
