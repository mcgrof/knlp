# Capture branching reasoning demand before choosing a retention policy

Run the official [Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm)
Game of 24 search at revision `8050e67d0e3a0fddc424d7fa5801538722a4c4cc`,
substituting a local vLLM model provider. Its model-generated arithmetic
proposals, evaluations and greedy selections determine the later calls. The
search, prompts, application value cache and dataset stay in the clean upstream
checkout. The upstream project uses the MIT license, copyright Shunyu Yao.

This is a bounded benchmark adaptation. It uses Qwen2.5-7B-Instruct instead of
the paper's model, keeps two states instead of five, evaluates each state once
instead of three times, and caps each output at 192 tokens. It is not a
reproduction of the paper's accuracy or production arrival patterns. Dataset
indices 0, 1 and 2 are chosen in file order before observing answers. All
selected answers are graded with exact rational arithmetic and a restricted
syntax parser; model text is never executed. Report failed tasks too.

The capture distinguishes logical parentage from exact token identity. A
proposal's producer is the parent of the calls that evaluate or extend it.
The official prompts can rewrite the state rather than append the producer's
complete output. Family membership therefore does not establish reusable KV.
Application memoization avoids some model calls; pruned states are not labelled
as cancelled engine requests. This workload makes sequential model calls and
does not issue in-flight cancellation.

## Prepare the inputs

Use the pinned vLLM serving source and compiled runtime described in the
[residency README](README.md). The model is `Qwen/Qwen2.5-7B-Instruct`, snapshot
`a09a35458c702b33eeacc393d103063234e8bc28`; its eleven model/tokenizer files
must match `tot-model-files.json`. Checkpoint hashes are checked before GPU
allocation. Use the matching Python 3.13 environment. Its existing NumPy and
SymPy are recorded at capture time; install the extra task dependencies into
an isolated directory without replacing the serving environment's packages.

```sh
git clone https://github.com/princeton-nlp/tree-of-thought-llm.git /path/to/tot
git -C /path/to/tot checkout 8050e67d0e3a0fddc424d7fa5801538722a4c4cc
curl -fsSL https://bootstrap.pypa.io/pip/pip.pyz -o /path/to/pip.pyz
/path/to/prepared/python /path/to/pip.pyz install --no-deps \
  --target /path/to/tot-deps \
  -r tools/reproduce/kmesh_residency/requirements-tot.txt
```

## Capture the actual search

```sh
make defconfig-kmesh-tot-cuda
export KMESH_RESIDENCY_SOURCE=/path/to/vllm
export KMESH_RESIDENCY_WORKLOAD_SOURCE=/path/to/tot
export KMESH_RESIDENCY_MODEL=/path/to/model-snapshot
export KMESH_RESIDENCY_OUTPUT=/path/to/fresh-search-capture
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# Include the pinned source and its prepared dependencies as required by the build.
export PYTHONPATH=/path/to/tot-deps:/path/to/vllm
make kmesh-residency-plan
make KMESH_RESIDENCY_PY=/path/to/prepared/python
```

The profile uses one synchronous eager engine, batch-invariant arithmetic,
16-token full-attention blocks, a 512 MiB engine cache and a 2,048-token context
limit. The cache budget matches the earlier serving baseline; it is not tuned
to force eviction after inspecting this workload. Calls use temperature 0.7 and
seed `123 + call_index`. The provider admits at most 384 calls and checks a
75-minute workload deadline at call boundaries. This is a safety ceiling,
not an expected duration. Each admitted call has a 120-second execution
limit. The wall-clock allowance leaves room for verbose evaluation replies
while the call, output, search and cache limits stay fixed. No external
storage connector is installed. The captured configuration records both the
request limit and workload deadline.

`workload.json` records source/model/runtime identities, task quality and actual
pool capacity. `invocations.jsonl` contains exact input/output IDs, parent and
family IDs, a shared monotonic clock and physical cache evidence. Scheduler
computed-token evidence must agree with returned token IDs. Records are flushed
and synced after each invocation. `search.json` retains selected and pruned
states; `events.json` retains allocator observations, including cache evictions.
Keep raw text and tokens with private experiment evidence. Allocation logging
records changed block tables rather than repeating them at every decode step.
Observation and capture writes can alter request gaps: this is not a latency
benchmark. Partial files remain on failure and cannot qualify the complete run.
