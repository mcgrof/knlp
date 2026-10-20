# MoT author-code baseline

This wrapper prepares and audits the external implementation of
Mixture-of-Translators (MoT). It pins the author repository to commit
`9753626a11b49b86ab940d87d9e5640958eb87d2` and keeps that checkout
unchanged. The author code uses the PolyForm Noncommercial License. It is not
copied into knlp.

The bounded diagnostic uses an isolated modified-MoT chain published in
`https://github.com/mcgrof/MoT.git` on branch
`research/mot-modified-sparse-20261017`, ending at
`9fb5e6f2a63981a53da1769c72e000a840ac976b`. It preserves exact token IDs for
the verified GPT-2 tokenizer pair, applies that identity contract in channel
profiling, and uses a dense selected-key correctness backend for reduced Top-S
replay. It consumes frozen token rows offline and writes step-level training
records. This is a corrected diagnostic, not an unchanged-author or
paper-reproduction result.

Prepare the exact modified source checkout with:

```bash
git clone https://github.com/mcgrof/MoT.git modified-MoT
git -C modified-MoT checkout 9fb5e6f2a63981a53da1769c72e000a840ac976b
```

The first target is a GPT-2 to GPT-2-medium mechanics baseline. It uses the
paper's shared 500-step OpenWebText training recipe, bfloat16 precision, two
translators with Top-1 routing, depth-ratio channels, and the author evaluator.
It is not yet an exact paper-result reproduction. The paper does not fully
specify the Table 8 F1 row, and the repository does not contain a downloadable
trained checkpoint.

The cost comparison must count more than translated KV tensors. The inspected
author path performs one source prefill for KV, a separate source forward for
attention guidance, every translator expert, and a complete receiver-stack
replay. The generated receipt records these as candidate work. A later serving
comparison must measure them against native fresh prefill and an exact retained
receiver cache under the same residency contract.

Load the readiness defconfig and point it at an unchanged author checkout:

```bash
make defconfig-mot-author-readiness
make MOT_SOURCE=/path/to/MoT \
  MOT_AUTHOR_PYTHON=/path/to/python
```

Plain `make` runs the knlp unit tests, the upstream mock-based test suite,
source provenance checks, a router-gradient audit, an expert-execution audit,
and a constructed causal-alignment audit. It writes generated receipts and
configs outside the repository. Mock tests do not establish real decoder or
paper parity.

Before training, run the CPU-only real-model preflight with the exact pinned
snapshots. It confirms that GPT-2 and GPT-2-medium use the same token grid,
that native continuation from KV matches a full forward pass, and that the
author GPT-2 replay helper matches native logits and cache tensors:

```bash
make mot-author-real-preflight \
  MOT_SOURCE=/path/to/MoT \
  MOT_AUTHOR_PYTHON=/path/to/python \
  MOT_OUTPUT=/path/to/results \
  MOT_SOURCE_MODEL=/path/to/pinned-gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned-gpt2-medium
```

This preflight does not train a translator and cannot test the unequal-token-
grid alignment path. Its thresholds are numerical-parity checks, not model-
quality gates.

The unchanged-author mechanics preflight forces the full-key sparse and
injected-KV branches at 20 and 64 tokens. The modified preflight reuses those
frozen controls and tests the 160-token, Top-S=128 dense selected-key backend
against an independent masked reference. It also runs an FP64 operator oracle,
malformed-selection rejection, and a four-pair mixed injected/recomputed
window. Run the two source variants into separate output directories:

```bash
make defconfig-mot-author-mechanics
make mot-author-mechanics-preflight \
  MOT_VARIANT=unchanged-author \
  MOT_SOURCE=/path/to/unchanged-MoT \
  MOT_OUTPUT=/path/to/unchanged-results \
  MOT_SOURCE_MODEL=/path/to/pinned-gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned-gpt2-medium

make mot-author-mechanics-preflight \
  MOT_VARIANT=modified-mot \
  MOT_SOURCE=/path/to/pinned-modified-MoT \
  MOT_OUTPUT=/path/to/modified-results \
  MOT_SOURCE_MODEL=/path/to/pinned-gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned-gpt2-medium
```

Modified MoT preserves the failed gathered backend as a compatibility path.
The admitted quality pilot uses `attention_backend=dense_selected`: it forms
the exact selected causal mask and calls native GPT-2 attention math. This is
a correctness backend with quadratic working storage, not a sparse-compute or
serving-speed result. Unequal-token-grid execution remains unadmitted.

After both mechanics controls pass, freeze the reserved training-only batch
from the pinned OpenWebText revision. This CPU step saves the exact 32 x
128-token inputs, shuffled source-row content hashes, tokenizer hashes, and
dataset recipe in a private manifest:

```bash
make mot-author-freeze-smoke-data \
  MOT_AUTHOR_PYTHON=/path/to/python \
  MOT_TARGET_MODEL=/path/to/pinned/gpt2-medium \
  MOT_DATA_MANIFEST=/path/to/private/TRAINING_SMOKE_MANIFEST.json
```

The separate training-smoke target then runs two BF16 optimizer steps with
microbatch 4 and four-way gradient accumulation. It uses the frozen manifest
offline and a fixed four-channel terminal window. The payload uses the
scheduler-neutral `KNLP_GPU_RUN_ID` and `KNLP_GPU_RUN_DIR` variables. It stores
the manifest, result, telemetry context and checkpoint under that run directory:

```bash
export KNLP_GPU_RUN_ID=mot-modified-smoke-001
export KNLP_GPU_RUN_DIR=/path/to/private/run-directory
export KNLP_GPU_PROVIDER=external
make mot-author-training-smoke \
  MOT_SOURCE_MODEL=/path/to/pinned/gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned/gpt2-medium \
  MOT_DATA_MANIFEST=/path/to/private/TRAINING_SMOKE_MANIFEST.json \
  MOT_OUTPUT="$KNLP_GPU_RUN_DIR"
```

The smoke checks finite component losses and gradients, selected-expert
updates, frozen language-model weights, and save/reload parity on the same
batch. It is not training-quality evidence and does not select channels. Pass
the frozen input through `MOT_DATA_MANIFEST`.

Training is an explicit operation. It requires local model snapshot paths so
an unpinned hub lookup cannot silently change the models:

```bash
make defconfig-mot-author-gpt2-medium
make mot-author-train \
  MOT_SOURCE=/path/to/MoT \
  MOT_AUTHOR_PYTHON=/path/to/python \
  MOT_OUTPUT=/path/to/results \
  MOT_SOURCE_MODEL=/path/to/pinned-gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned-gpt2-medium
```

Evaluation also requires an explicit checkpoint through `MOT_CHECKPOINT`.
The generated `FROZEN_BASELINE.json`, `PROVENANCE.json`,
`CPU_READINESS.json`, and `REPRODUCTION_MATRIX.md` belong with the private
experiment record, not in knlp. The real-model preflight also writes
`REAL_MODEL_PREFLIGHT.json` there.

For the bounded diagnostic, first freeze the private data manifest. The
manifest contains 128 profiling rows, 64 profiling-validation rows, 8,000
training row positions, and five 100-item evaluation cohorts. OpenWebText
partitions are disjoint by source document. MMLU contains 100 items total,
not 100 items per subject.

```bash
make defconfig-mot-modified-diagnostic
make mot-author-freeze-diagnostic-data \
  MOT_SOURCE=/path/to/pinned-modified-MoT \
  MOT_SOURCE_MODEL=/path/to/pinned-gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned-gpt2-medium \
  MOT_SMOKE_MANIFEST=/path/to/private/TRAINING_SMOKE_MANIFEST.json \
  MOT_DATA_MANIFEST=/path/to/private/DIAGNOSTIC_DATA.json

make mot-author-diagnostic-controls \
  MOT_SOURCE=/path/to/pinned-modified-MoT \
  MOT_SOURCE_MODEL=/path/to/pinned-gpt2 \
  MOT_TARGET_MODEL=/path/to/pinned-gpt2-medium \
  MOT_DATA_MANIFEST=/path/to/private/DIAGNOSTIC_DATA.json
```

Set `MOT_FROZEN_DATA_MANIFEST` to that manifest for `mot-author-train`. After
training, `mot-author-diagnostic-eval` runs only the five frozen cohorts and
writes 500 paired JSONL records. `mot-author-replay-diagnostic` rebuilds every
per-task aggregate and paired 95% bootstrap interval without models.

The inference-only attribution diagnostic keeps that checkpoint and cohort
fixed. It evaluates four arms: exact native receiver cache (N), translated-KV
selected replay (T), selected replay with no injection (R), and selected replay
with native receiver K/V injected at the same target layers (O). It computes
the source-derived selected indices once per item and reuses the same tensors
for T, R, and O. N and T must reproduce the retained token-level anchors.

```bash
make defconfig-mot-modified-attribution
make mot-author-attribution-eval \
  MOT_SOURCE=/path/to/pinned-modified-MoT \
  MOT_CHECKPOINT=/path/to/frozen/checkpoint \
  MOT_CHECKPOINT_SHA256=<translator-file-sha256> \
  MOT_DATA_MANIFEST=/path/to/private/DIAGNOSTIC_DATA.json \
  MOT_ANCHOR_RECORDS=/path/to/private/PAIRED_RECORDS.jsonl.zst \
  MOT_OUTPUT=/path/to/results
```

The target writes 500 records and 2,000 scored arm outputs. It reports T-R,
T-O, R-N, and O-R paired differences. SQuAD uncertainty groups by the executed
context-token digest; MMLU groups by subject. This is an attribution diagnostic,
not a speed benchmark or a paper reproduction.
