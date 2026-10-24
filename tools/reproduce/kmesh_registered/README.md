<!-- SPDX-License-Identifier: MIT -->
# Registered serial model replay

This bounded replay moves the TokenDance-style serial repair baseline from
in-process retrieval to completed native span restores in a separate cache
process. It uses the actual Qwen model, rotary table, engine cache allocations
and live request slots. The leader publishes four partial spans containing 416
tokens. Four siblings restore them at shifted positions and consume them through
the existing serial CacheBlend repair.

LocalCPU lookup and storage remain scheduling controls. This duplicates storage;
elapsed time cannot establish serving latency, capacity savings or a DMA-BUF
benefit. The ordinary multiprocess serving profile remains disabled. This replay
qualifies CPU-L1 model consumption before changing the storage route.

Use the exact source commits in `profile.json` with the existing
[RTX runtime and model pins](../tokendance/profile.json). Install the matching
native extensions and generated protobuf modules as described by the component
projects. Keep source checkouts clean. Generate the frozen trace using the
[TokenDance reproducer](../tokendance/README.md), which also produces the
independently verified serial reference used by the comparison.

Run the lower-level gates first with `defconfig-kmesh-registered-inputs-cpu`
and `defconfig-kmesh-registered-inputs-cuda`. These use the existing
`kmesh-contracts` and `kmesh-native` entry points and their caller-selected source
and external output paths. Both pin the model replay's LMCache source. The
CUDA gate covers separate and fused NHD/HND views, exact partial spans,
synthetic layer input/writeback, physical drain and exporter retirement. Its
synthetic transform does not replace the real model replay below.

```sh
make defconfig-kmesh-registered-model-cuda
make kmesh-registered-plan
```

Set `KMESH_REGISTERED_VLLM_SOURCE`, `KMESH_REGISTERED_LMCACHE_SOURCE`,
`KMESH_REGISTERED_WHEEL_PACKAGE` and `KMESH_REGISTERED_MODEL_DIR` to the pinned
local inputs. Set `KMESH_REGISTERED_TRACE` and `KMESH_REGISTERED_MANIFEST` to the
generated trace files. Set `KMESH_REGISTERED_OUTPUT` to a fresh external capture
directory. First set `KMESH_REGISTERED_ROUTE=reference` and run
`make kmesh-registered` with the matching Python interpreter selected through
`KMESH_REGISTERED_PY`. This captures the in-process reference on the new source.
Verify it against the independently verified historical serial result. Then use
`KMESH_REGISTERED_ROUTE=registered` with another fresh output directory and run
the model again. Both runs use the same source checkout, workload and runtime.

For the first comparison, set `KMESH_REGISTERED_REFERENCE` to the historical
bundle's `arms/01-serial_oracle/artifacts` directory and
`KMESH_REGISTERED_VERIFICATION` to its independently generated `verified.json`.
Run `make kmesh-registered-verify` to compare saved CPU tensors. For the paired
comparison, point `REFERENCE` to the new reference capture and `VERIFICATION`
to that capture's passing `comparison.json`. The pair also requires identical
actual repair selections at every layer. The policy requires
bitwise K/V, identical selected spans and retrieval masks, logits within the
existing serial comparison tolerance, and identical generated tokens. A failed
comparison is retained as evidence; it must not be relabeled a passing run.

All captures remain outside the source trees. The service process is owned by
the replay and exits before engine allocations can be freed. Checked retirement
must cover all imported allocations; an uncertain reply requires actual importer
exit. This runner does not change kernels, drivers or storage media.

## Model failure controls

Select a new `KMESH_REGISTERED_OUTPUT` directory and set
`KMESH_REGISTERED_ORDINARY_RESULT` to the independently verified historical
bundle's `arms/00-ordinary/artifacts/result.json`. Set
`KMESH_REGISTERED_ORDINARY_VERIFICATION` to that bundle's `verified.json`, then
run `make kmesh-registered-controls` with the same pinned source and runtime.
Each control runs in a fresh process and records its actual operations:

- Omit one registered handle while the source remains intact. The model must
  consume the other three spans and recompute all missing and private positions.
- Corrupt an actual CPU-L1 tensor, or delete its actual L1 key before lookup.
  Both must refuse before native scatter and complete through fallback.
- Raise after a genuine native restore enqueue. Successful physical drain must
  permit full-prompt invalidation and recomputation.
- Refuse one service drain, then require an actual later drain before fallback.
- Keep refusing service drain, or refuse worker drain after a real layer gather.
  Neither may mark blocks reusable for fallback or return a completed request.
  Disarming the control must allow real cleanup and checked retirement.

A revoked lease disables this bounded replay backend; later drained requests
recompute. This is an explicit availability limit. Controls compare completed
generated tokens with the ordinary baseline. They do not replace the successful
pair's raw K/V, logit and selection comparison.
