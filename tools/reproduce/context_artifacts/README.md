# Context artifact file transport

A context artifact is an immutable tensor payload with the identities needed
by its consumer. A document embedding bank and a learned key/value prefix have
different shapes and dependencies. The typed manifest keeps these contracts
separate while giving both the same import, lookup, and byte-restore operations.

This implementation provides CPU file transport. It does not attach artifacts
to vLLM, integrate with LMCache, execute a decoder, or qualify device transfers.
Lookup-or-create encoding is not implemented.
The control and LMCache presets reserve separate configurations and fail with
an explicit blocker. The default acceptance command also fails until inference
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

Run the CPU contract and configuration tests with:

```sh
python -m pytest tools/reproduce/context_artifacts/tests -q
```
