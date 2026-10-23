# Exact-prefix reasoning fork accounting

This experiment asks whether a tree-aware memory manager could reduce the
number of KV-cache blocks needed by reasoning branches beyond vLLM's automatic
prefix caching. A child contains its parent's exact token sequence followed by
an independent tail. For example, five children extend a 4,096-token ancestor
with 512 tokens each. These are synthetic causal-prefix forks, rather than
matching text placed after different private histories.

The harness supplies the matching token sequences. vLLM's automatic prefix
caching supplies physical sharing inside one engine: complete token blocks
are indexed by a hash of their tokens, preceding prefix and relevant cache
identity. A matching request acquires references to those existing KV blocks;
its divergent tail receives separate writable blocks. The replay uses ordinary
requests, without a request-fork API or explicit family declaration. Shared
block IDs save cache capacity. They do not by themselves share attention work,
cross engine boundaries or eliminate external storage transfers.

The benchmark drives the real `KVCacheManager` from the pinned vLLM branch.
It allocates metadata and treats admission as completed prefill, following
vLLM's prefix-cache unit-test convention. No model, scheduler, tensor storage,
GPU or LMCache backend executes. This is the allocator experiment following
the [TokenDance-style recovery baseline](../tokendance/README.md); the original
serving replay and later attention/storage experiments remain separate steps.

[Run](#run-the-allocator-matrix) · [Runtime](#prepare-the-pinned-runtime) ·
[Results](#read-the-results) · [Verification](#verify-and-test)

The workload has two explicit software profiles. The `historical` profile
retains the published source and prepared container. The `current` profile
repeats the same matrix against the newer vLLM integration branch with CUDA
libraries, with GPU visibility disabled. It validates allocator behavior only;
it does not exercise LMCache, DMA-BUF transfers or prefill/decode handoff.
The newer TokenDance recovery implementation requires its own model-execution
qualification; this allocator result cannot establish it.

## Run the allocator matrix

From knlp's root, with Python 3.10 or newer and GNU make:

```sh
make defconfig-kmesh-fork-accounting
make kmesh-fork-plan
make kmesh-fork-test

git clone --single-branch --branch 2026-10-16-feature-kmesh-fork-accounting \
  https://github.com/mcgrof/vllm.git ../vllm-kmesh-forks
git -C ../vllm-kmesh-forks checkout --detach 5ccaa1456af6d2c08fbb39e6c7bc4480df445d07

export KMESH_FORK_VLLM_SOURCE="$(cd ../vllm-kmesh-forks && pwd)"
export KMESH_FORK_RUNTIME_VOLUME=prepared-runtime-volume
export KMESH_FORK_DOCKER=docker
export KMESH_FORK_OUTPUT="$PWD/results/forks-run"
make
```

The source must be clean and match both commit and tree in
[profile.json](profile.json). The source branch extends the published TokenDance
vLLM branch without changing the allocator or updating its upstream base.
LMCache stays at the published TokenDance pin; this experiment does not need a
new LMCache branch. The benchmark retains seed `20260805`, which defines the
original input tokens. New commit dates do not change that seed.

Default `make` executes the complete real allocator matrix and independently
verifies its result. It does not silently fall back to formula-only simulation.
`kmesh-fork-plan` shows the inputs without requiring the native environment;
`kmesh-fork-test` checks the portable verifier against the published capture
and deliberately corrupted evidence.

Every execution needs a fresh output directory. The runner refuses existing
directories, preserving successful and failed attempts. Outputs include
`manifest.json`, the exact container command, `native.log`, `results.json` and
`verified.json`. The manifest records source identity, runtime profile and
execution time. A failed command or missing verification report is not a pass.
Absolute source paths in local manifests belong in private experiment records.

## Prepare the pinned runtime

The automated native run uses the same prepared runtime as the TokenDance
baseline. Docker and a populated runtime volume are prerequisites; `make` does
not build or install packages. The volume must contain
`/opt/kmesh-target/bin/python`, Python 3.12.13, PyTorch
`2.11.0+gitd0c8b1f`, and the compiled vLLM wheel based on
`454ea5b52611c933e00581723e2db56f0144cea7`. Its import version is
`0.26.1rc1.dev221+g454ea5b52`; the distribution build has the `.rocm723` suffix.
The immutable image is recorded in [profile.json](profile.json). See the
[baseline's runtime prerequisites](../tokendance/README.md#pin-the-component-checkouts-and-native-environment)
for the complete prepared environment.

The launcher mounts the source and runtime read-only, disables network access,
hides all GPUs, and mounts no GPU devices. It verifies runtime versions and
that `KVCacheManager` actually imports from the pinned source. The two OpenMP
threads are an execution setting, not a performance measurement. No model
weights, GPU reservation, storage offload, DMA-BUF path or hardware-specific
kernel is required. The ROCm runtime is reused for CPU imports only.

A different runtime needs separate qualification. Building or distributing a
fresh native environment is not automated by this profile. If Docker requires
an administrator-approved prefix, set `KMESH_FORK_DOCKER='sudo -n docker'`.

### Repeat on the current allocator

The current source pin in [profile-current.json](profile-current.json) extends
the newer integration branch with the allocator benchmark and fork tests.
Use the dated branch and exact source pin:

```sh
git clone --single-branch --branch 2026-10-16-feature-kmesh-serving-replay \
  https://github.com/mcgrof/vllm.git ../vllm-kmesh-current
git -C ../vllm-kmesh-current checkout --detach 1b76bf9dd9d52efbd75fb89bf2619898d8f01233
```

The branch retains the current TokenDance code and adds the allocator
benchmark without changing the production memory manager.

Use a clean checkout at that exact commit and a prepared Python 3.13.15
environment with PyTorch `2.13.0+cu130` and the compiled vLLM wheel based on
`84bcbc62644356270aaaa5e2d0237d03adc9bb3a`. Its import version must be
`0.30.1rc1.dev624+g84bcbc626`. As for an editable vLLM installation, the
matching native extensions must be available inside nested source packages,
including `vllm/vllm_flash_attn`; the top-level wheel overlay does not extend
those packages' search paths. The baseline CUDA preparation script also prepares this allocator
environment; see the current CUDA instructions in the baseline README.

```sh
make defconfig-kmesh-fork-accounting-current
export KMESH_FORK_VLLM_SOURCE=/absolute/path/to/pinned-vllm
export KMESH_FORK_PYTHON=/absolute/path/to/prepared-venv/bin/python
export KMESH_FORK_WHEEL_PACKAGE=/absolute/path/to/site-packages/vllm
export KMESH_FORK_OUTPUT="$PWD/results/forks-current-run"
make
KMESH_FORK_OUTPUT="$PWD/results/forks-current-contracts" make kmesh-fork-contracts
```

This profile runs locally, hides GPUs and sets the library offline flags. It
does not apply the historical container's read-only mounts or network
namespace isolation. It verifies source commit/tree and runtime versions,
checks the allocator import path and rejects a GPU-visible runtime. The
contracts report requires all collected prefix-cache tests to pass without
skips. Both profiles retain seed `20260805` and two OpenMP threads.

## Read the results

The complete CPU capture was
repeated on 2026-10-16. Its counters match the archived experiment across five
trees, five admission/cache scenarios per tree and six eviction points. Block
size is 16 tokens. The primary tree has a 4,096-token root and five 512-token
tails. The fork oracle counts 256 ancestor blocks plus five sets of 32 tail
blocks: **416**. An uncached root and five independent full child sequences
require 256 + 5 × 288 = **1,696** simulated writes over the tree lifetime.

| Primary tree | Written blocks | Peak pool usage | Failed admissions |
| --- | ---: | ---: | ---: |
| Concurrent siblings, cached | 416 | 417 | 0 |
| Concurrent siblings, uncached | 1,696 | 1,441 | 0 |
| 520-block pool, cached | 416 | 417 | 0 |
| 520-block pool, uncached | 544 | 289 | 4 |

Written blocks accumulate simulated prefill work; peak usage counts allocated
pool metadata at one instant and includes the reserved null block. The parent
is freed before admitting its children. Therefore 1,696 cumulative uncached
writes must not be described as 1,696 simultaneously resident blocks. The
pressure control writes fewer blocks because four children fail admission.

Every cached tree reaches its fork oracle while ancestors remain available,
including depth-two and non-aligned forks. A 4,101-token ancestor with five
509-token tails writes 422 blocks and recomputes five boundary tokens per
child, 25 total. Stock caching recomputes that partial boundary where an
explicit fork could clone it; the tested block count is the same.

After unrelated requests churn a 1,024-block pool, the child hit counts are:

| Churn / pool | Child prefix hits in tokens |
| --- | --- |
| 0, 0.25 or 0.50 | 4,096 for all five children |
| 0.75 | 4,080, then 4,096 for four children |
| 1.00 or 1.50 | 0, then 4,096 for four children |

In these sequential metadata admissions, the first child's simulated prefill
repopulates lost ancestor blocks for later siblings. This does not establish a
universal bound under scheduler concurrency, repeated interleaved eviction or
preemption. The actual serving replay is a separate experiment.

The result narrows the proposed memory-manager change: full-ancestor block
duplication is not a capacity gap for these exact forks with prefix caching.
Boundary recomputation and ancestor eviction remain observable costs. Sharing
storage capacity also does not prove fewer attention reads: each branch can
still read the same shared KV blocks independently. Cascade attention, future
use policies, external caching and DMA-BUF are outside this measurement.

The byte fields are geometry projections only: K and V × 24 layers × 2 KV
heads × 64 features × 2 BF16 bytes = 12,288 bytes per token, or 196,608 per
16-token block. The metadata test allocates no corresponding KV tensors and
establishes no quality, latency, throughput or end-to-end capacity result.

## Verify and test

The current allocator capture
matches every historical scenario summary, hit vector and eviction point.
The current prefix-cache suite passes all 174 collected tests without skips.
Uncached block ID allocation order differs in some rows, while cached ancestor
aliasing and private tails satisfy the same checks. Neither this run nor the
historical CPU run tests the TokenDance model recovery or a storage transfer.

The portable verifier compares the entire saved counter matrix, hit vectors
and eviction sweep. It also checks observed block tables for ancestor aliasing
and sibling-private tails; optimistic summary counts alone cannot pass.

```sh
python3 tools/reproduce/kmesh_forks/verify.py \
  results/forks-run/results.json --output results/forks-run/rechecked.json
```

Add `--profile current` when checking a current-runtime capture. A capture
from one runtime is rejected under the other profile.

Run the real vLLM prefix-cache unit suite in a separate fresh output directory:

```sh
KMESH_FORK_OUTPUT="$PWD/results/forks-contracts" make kmesh-fork-contracts
```

The suite exercises the real allocator, including non-aligned fork boundaries,
shared references surviving a sibling's release, and the negative control of
matching suffix tokens after different causal prefixes. The portable tests
reject missing controls, missing eviction points, optimistic uncached counts,
aliased writable tails, inconsistent counters and changed runtime versions.

## Replay forks through the serving engine

The serving replay checks whether the allocator result survives real model
execution and scheduler admission. A child uses its parent's exact token
sequence plus an independent tail. The model runs locally in the real vLLM
engine; this is not a scheduler simulation or a network load test. LMCache
is not used in these seven cases.

The current CUDA preparation command in the
[baseline README](../tokendance/README.md#current-cuda-profile) prepares the
pinned runtime, model and separate vLLM serving checkout. From that environment:

```sh
. /workspace/kmesh-runtime/run.env
make defconfig-kmesh-serving-cuda
make kmesh-serving-plan
KMESH_SERVING_OUTPUT=/workspace/serving-forks make
make kmesh-fork-test
```

Run the GPU command with the prepared interpreter on an available GPU.
The launcher requires the pinned image declaration and a clean checkout matching
[profile-current.json](profile-current.json). It authenticates the model
snapshot and runs each case in a fresh process. Source, commands, script
hashes, runtime versions and resolved scheduler settings remain in the bundle.
It refuses to overwrite an earlier output directory.

The workload retains seed `20260805`, a 4,096-token ancestor, five 512-token
tails and 16-token blocks. Warm cached and uncached controls generate one
parent token followed by up to 64 tokens per child. Burst and churn cases
generate up to eight child tokens. Churn submits unrelated 2,048-token prompts
through a 1,024-block pool before the children. The decode-fork case generates
up to 128 parent tokens and builds three child prompts from those actual
output tokens plus fresh 512-token tails.

The recorder observes actual allocation and free boundaries, including active
request block tables. An independent verifier checks physical prefix aliases,
separate writable tails, event-derived peaks, released request ownership and
all seven completed cases. The original five-millisecond pool sampler is
retained as a separate observation. These counters describe pool allocation;
they do not measure device memory traffic or serving speed. Recording and
synchronization make the saved timings unsuitable for performance claims.

A CPU recheck needs the original capture directory and the same committed
reproducer inputs:

```sh
KMESH_SERVING_BUNDLE=/workspace/serving-forks \
KMESH_SERVING_OUTPUT=/workspace/serving-rechecked \
make kmesh-serving-verify
```

### H100 serving result

All seven cases and the independent physical-block checks pass on the rebased
source branch. The
sanitized serving record gives
the source tree, script and result digests. Raw operational captures remain
separately retained; knlp distributes the reproducer and bounded summary.

| Case | Child prefix hits, tokens | Peak used pool blocks |
| --- | --- | ---: |
| Warm parent, prefix cache on | 4,096 × 5 | 437 |
| Warm parent, prefix cache off | 0 × 5 | 1,461 |
| Parent and children in one burst | 4,096 × 5 | 423 |
| Cold sibling burst | 0, then 4,096 × 4 | 422 |
| Churn equal to half the pool | 4,096 × 5 | 513 |
| Churn equal to the full pool | 0, then 4,096 × 4 | 897 |
| Fork after 128 generated parent tokens | 4,208; 4,224; 4,224 | 364 |

The warm control demonstrates prefix aliasing in all ten live sibling pairs.
Its 437-block peak includes the null block and 20 decode-tail blocks beyond
the allocator experiment's 417-block peak. The uncached control has disjoint
physical blocks and produces the same generated token IDs for the parent and
all five children in this run. This bounded output check is not a quality
benchmark.

Cold arrival and full-pool churn leave one child to compute the missing
ancestor, after which four siblings reuse it. Decode-created prefixes also
share: the first child fills the last incomplete 16-token parent block and
later children hit the full 4,224-token fork boundary. These are observed
outcomes for this workload, not universal guarantees under interleaved traffic
or preemption.

Model: `Qwen/Qwen2.5-0.5B-Instruct` at the model revision pinned in
`tokendance/profile.json`. Runtime: one H100, Python 3.13.15, PyTorch `2.13.0+cu130`, CUDA 13.0 and vLLM
`0.30.1rc1.dev624+g84bcbc626`, with BF16, eager execution and two OpenMP
threads. Stock defaults resolve to FlashAttention, 16,384 maximum batched
tokens, 1,024 maximum sequences, chunked prefill and asynchronous scheduling.
The replay records these choices rather than assuming that another engine
version has the same defaults.

The engine already shares exact-prefix storage capacity in these cases. This
experiment does not measure shared attention reads, cascade attention,
storage offload or DMA-BUF latency. Those remain separate questions.
