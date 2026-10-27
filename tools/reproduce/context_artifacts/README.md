# Context artifact file transport

A context artifact is an immutable tensor payload with the identities needed
by its consumer. A document embedding bank and a learned key/value prefix have
different shapes and dependencies. The typed manifest keeps these contracts
separate while giving both the same import, lookup, and byte-restore operations.

This implementation provides standalone CPU file transport and an external
artifact adapter in LMCache. The LMCache path uses the vLLM cartridge catalog
and real LMCache host memory objects. It does not attach KV to a running vLLM
engine, execute a decoder, or qualify GPU-direct storage.
Lookup-or-create encoding is not implemented.
The control presets fail with an explicit blocker until their engine recipes
are qualified. The LMCache artifact presets support import/load/restore. The default acceptance command also fails until inference
is implemented. `BYTE_TRANSPORT_PASS` never means `INTERFACE_PASS`.

`artifacts.py` supports `document_embeddings`, `input_embeddings`, and
unquantized `prefix_kv`. Payloads use safetensors. Document vectors bind their
encoder, preprocessing, and source identities. Input embeddings additionally
bind their reader, question, document bank, prompt placement, and positions.
Key/value prefixes bind model, tokenizer, rotary-position configuration,
layer/head geometry, exact logical positions, and physical tail padding.
Consumers must mask padding when they later attach a prefix to an engine.

The artifact ID hashes payload digest and semantic metadata. Storage location
and working-copy residency do not enter the ID. Ordered attachment keys bind
the consumer and prompt dependencies, but engine cache-key propagation remains
unimplemented. Existing cartridge classes and connector APIs are untouched.

Import copies and validates the complete payload, synchronizes the file and
directories, then publishes its manifest in a FULL-synchronous SQLite
transaction. A failed import is not discoverable. Restart can recover a
published artifact; an interrupted publication can leave an undiscoverable
object for a verified retry. These operations do not establish hardware
power-loss guarantees. Eviction releases only the registry's working bytes.
Caller-owned tensor copies and Linux page cache have separate lifetimes.

Install `requirements-cpu.txt` from this directory into a separate Python 3.12
environment. Select a preset with the normal configuration commands:

```sh
make defconfig DEFCONFIG=context-artifacts-memorilla-files
make olddefconfig
```

Set `KNLP_CA_WORK_ROOT`, `BUILD_ROOT`, `STORE_ROOT`, `RESULTS_ROOT`, and
`SOURCE_LOCK` through `make menuconfig`. These symbols use the `KNLP_CA_`
prefix. Supply local persistent paths; the runner rejects temporary memory
filesystems and unresolved overlay/network backing. The source lock contains
an exact interpreter path and a `sources` mapping. Each source has a checkout
`path` and complete Git `revision`. The runner checks tracked changes, Git
metadata backing, and object alternates before executing storage operations.

Set `KNLP_CA_MANIFEST` and `KNLP_CA_PAYLOAD` to an existing typed JSON manifest
and safetensors file. Set the exact `CONSUMER` model and `CONSUMER_REVISION`.
Use the selected environment explicitly:

```sh
make context-artifacts-doctor CA_PYTHON=/path/to/environment/bin/python
make context-artifacts-import CA_PYTHON=/path/to/environment/bin/python
```

Import prints the immutable ID. Set `KNLP_CA_ARTIFACT_ID` to that value before
load or restore. Loading never encodes documents or trains a missing prefix.

```sh
make context-artifacts-load CA_PYTHON=/path/to/environment/bin/python
make context-artifacts-restore RESTORE_KIND=eviction \
  CA_PYTHON=/path/to/environment/bin/python
make context-artifacts-restore RESTORE_KIND=restart \
  CA_PYTHON=/path/to/environment/bin/python
```

Each configured run writes its resolved configuration and JSON outcome into a
new results directory. Restore reports page-cache state as unknown. It does
not claim physical storage traffic or latency improvement.

`reader_probe.py --help` describes a separate CPU check for Memorilla's released
reader. It uses unchanged author code and precomputed PubMedQA vectors. One
writer process exits before a second process restores document vectors and
compares reader outputs and persisted soft tokens exactly. It renders the
author's prompt to bind real memory positions. It does not execute the decoder
or verify generation quality. Keep the author source/checkpoint fixed and record the exact Torch,
Transformers, and device runtime for each native-versus-storage control.
The CPU paper-stage control and a ROCm port are separate environments.

Run the CPU contract and configuration tests with:

```sh
python -m pytest tools/reproduce/context_artifacts/tests -q
```

## LMCache artifact reads

Select `context-artifacts-memorilla-lmcache` or
`context-artifacts-lpc-lmcache`, then configure the same manifest, payload,
consumer and source-lock fields as the file path. Set `KNLP_CA_POOL_BYTES`
to the host arena size needed by active artifacts. This bounds the arena,
not total process memory or the Linux page cache. The interpreter must import
the matching vLLM and LMCache checkouts recorded in the source lock.

The vLLM `CartridgeRegistry` now inherits an engine-independent artifact
catalog. Legacy cartridge manifests and labels retain their existing API;
typed context records share the database. Compatibility imports preserve
the existing connector module paths. The catalog resolves identity and a
bounded file location; LMCache owns loaded host buffers, reference counts,
idle-object eviction, and coalescing of simultaneous misses for one artifact.
The adapter is an external-artifact store, not the prompt-token KV cache or
an automatic connection to every LMCache storage-manager backend.

Repeated leases admit an immutable read plan and typed tensor layout once.
Both metadata caches default to 1,024 entries; `max_prepared` changes that
budget independently of the payload arena. This is an entry count, not a
total memory limit. Active leases, caller-retained plans, tensor copies and
the page cache require their own memory accounting. A prepared plan belongs
to the store that issued it and does not reserve payload capacity.

Resident hits reuse the admitted metadata and compare the safetensors header
before creating fresh tensor views. Each payload miss still reads the current
catalog record and rejects a change, then hashes every newly loaded byte.
Preparing a plan does not make mutable source files safe. Consumer identity
is checked on every typed lease, including resident hits.

`lease(..., timeout=seconds)` can wait for another lease to release capacity
without repeating admission. The deadline bounds coalescing and capacity
waiting; it does not interrupt file or catalog operations already running.
The default of zero retains immediate capacity failure and ordinary miss
coalescing. An oversized artifact with no releasable capacity fails rather
than waiting indefinitely. Catalog transactions close their SQLite
connections when each operation finishes.

A collection remains the logical read unit. It can occupy its own file or
an exact byte range in a larger file. Each range currently contains a
complete safetensors artifact, including its header. Direct registration of
raw collection rows inside the author's single-tensor shard is not implemented;
the reader probe serializes each selected bank once during publication.
The reader fills an LMCache buffer
through ordinary buffered `readinto`, then checks its digest in memory.
A load does not first scan the whole source file for hashing. Publication
verifies and synchronizes existing source bytes before metadata becomes
visible. Callers must retain those immutable files; relocation in the same
catalog needs an explicit future placement update.

Tensor views borrow the host allocation. Keep the lease until all consumers
and device copies complete; do not modify borrowed tensors or use them after
lease exit. Active leases prevent eviction. Resident hits reuse the allocation
and do not reread the file. No kernel patch is required and no kernel-patch
speedup is claimed.

The released reader probe accepts `--storage-backend lmcache --device cuda`
for a CUDA-compatible PyTorch device, including ROCm, or `--device cpu`.
It checks native versus restored reader outputs, persisted soft tokens,
working-copy eviction and a second process. It also checks two questions
over one shared bank. The probe retains `interface_pass=false` because it
does not execute the decoder. A different Torch/device environment is a
separately labeled reader control, not a reproduction of the complete
paper environment.

With the pinned integration environments installed, run:

```sh
python -m pytest tools/reproduce/context_artifacts/integration_tests -q
python -m tools.reproduce.context_artifacts.reader_probe --help
```

The integration tests use the real cartridge registry and LMCache host
allocator. They cover all three representations, packed ranges, concurrent
misses, consumer rejection, corrupt/short files, and active/idle eviction.

Full checkpoint inference and reusable request replay use the separate
[`memorilla-author`, `memorilla-port-control`, and `memorilla-lmcache`
defconfigs](../memorilla/README.md). They run decoder generation and original
scorers; the artifact import/load commands above retain their transport scope.
