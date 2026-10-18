# Shared-prefix attention on the current vLLM

Automatic prefix caching lets sibling requests occupy the same ancestor
pages, but ordinary attention can still read that ancestor separately for
each request at every decode step. Cascade attention computes attention over
the common prefix as one group, then merges each request's private suffix.
This experiment checks that GPU-memory read opportunity. It does not use
LMCache or measure DMA-BUF, NVMe traffic or storage coalescing.

The original synthetic workload is Llama-3.1-8B-Instruct with an 8,192-token
ancestor, 32 different 512-token tails, seed 20260809 and 128 greedy generated
tokens per child. The source and ten model-artifact hashes are fixed in
`profile.json`. Use the same prepared current CUDA runtime as the
[restore comparison](../kmesh_restore/README.md#runtime-and-sources), with
its clean pinned vLLM checkout. The model snapshot must already be available
under its access terms; this runner does not download it.

Three fresh processes explicitly select the cascade-capable V1 model runner
and FlashAttention 2: cascade disabled, enabled with the normal heuristic,
and enabled with the heuristic's performance decision bypassed. The forced
mode retains the no-ALiBi, no-sliding/local-attention, single decode-context
rank and minimum-prefix support guards. It is a process-local diagnostic,
not a change to the installed vLLM source or a proposed production default.

Each process warms its prefix cache, then records actual calls to the cascade
kernel during a separate untimed generation. The timing repetitions remove
the observation wrappers; the forced policy stays in effect in that arm.
Three generation-wall samples include cached-prefill scheduling and decode.
These samples are exploratory, not independent paired trials establishing a
stable speedup. Every timed request must report at least the complete cached
ancestor. The verifier rejects an unused forced path, and marks a naturally
enabled path with no kernel calls ineligible for a speed comparison.

The runner disables batch-invariant arithmetic because current vLLM disables
cascade under that policy. Greedy token changes are counted against the
disabled control and reported separately; passing engagement and timing
checks is not a numerical-equivalence or task-quality qualification.

BF16 K/V uses 32 layers, eight KV heads and 128 values per head: 128 KiB per
token. A 4 GiB cache fits the shared 8k ancestor plus 32 private tails and
128-token continuations with block slack. Both arms use that capacity,
4,096 maximum batched tokens, 32 maximum sequences, eager execution,
synchronous scheduling and two OpenMP/MKL threads, without a CPU-affinity
cap. Use an available GPU with enough memory for this configuration.

```sh
make defconfig-kmesh-cascade-cuda
make kmesh-cascade-plan

# Set these inputs explicitly inside an owned shell wrapper.
export KMESH_CASCADE_MODEL_DIR=/path/to/authenticated/llama/snapshot
export KMESH_CASCADE_VLLM_SOURCE=/path/to/clean/pinned/vllm
export KMESH_CASCADE_OUTPUT=/path/to/fresh/results
make kmesh-cascade KMESH_CASCADE_PY=/path/to/prepared/python
```

Run that wrapper with `/bin/bash /path/to/wrapper.sh`.
Preserve the runtime's CUDA and Python library paths explicitly. The launcher
retains and drains its child processes.
An unsupported backend or capacity failure is a failed run, not a result.

Recheck completed evidence on CPU with
`python3 tools/reproduce/kmesh_cascade/verify.py /path/to/results` or
`KMESH_CASCADE_OUTPUT=/path/to/results make kmesh-cascade-verify`.
Run the rejection checks with
`python3 -m unittest discover -s tests -p test_kmesh_cascade.py -v`.
Raw provenance contains local paths and belongs in private evidence storage.
