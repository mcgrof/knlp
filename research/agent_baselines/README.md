# Agent cache baselines

Use mini-SWE-agent as the default live SWE-bench workload. It gives a small,
inspectable agent loop and patches that the official harness can grade.
Keep EfficientAgent as a separate fixed-trace analysis and replay baseline.
Together they measure both useful task completion and the serving cost of a
controlled sequence of calls.

This directory is a standalone research harness; it does not change knlp's
training or serving paths. All commands print a plan unless `--execute` is
supplied. Each run requires a new output directory. Python 3.10+ is required;
run commands from the knlp repository root.

| Track | What it answers | What it cannot establish |
|---|---|---|
| mini-SWE-agent live run + SWE-bench grading | Do agents still solve tasks, and what cache reuse and latency occur? | Reproduction of the paper's OpenHands agent |
| EfficientAgent trace profile | How much exact-prefix reuse exists and what working set is implied? | Actual hardware cache hit rate or speedup |
| EfficientAgent forced-output replay | How do serving/cache policies handle the same recorded calls? | Whether a freely generating agent still succeeds |

The paper uses OpenHands/CodeActAgent; substituting mini-SWE-agent produces an
adaptation. The public EfficientAgent release can build traces from recorded
calls or make synthetic ones. Neither is automatically the original paper
trace. Treat original trace availability, OpenHands environment/configuration,
model weights, tokenizer, and hardware matching as separate reproduction gates.

## Set up isolated baselines

```bash
python -m research.agent_baselines.bootstrap mini --out build/agent-mini --execute
python -m research.agent_baselines.bootstrap efficientagent --out build/agent-ea --execute
python -m research.agent_baselines.bootstrap swebench --out build/agent-grader --execute
```

Omit `--execute` to inspect the clone/install commands. These environments are
separate from knlp's dependencies and from the GPU serving environment. Source
revisions are fixed in the modules; the mini transport also pins LiteLLM and
OpenAI. Remaining Python dependencies are resolved at installation and recorded
in `requirements-resolved.txt`, not presented as a fully locked environment.

| Component | Pinned commit |
|---|---|
| mini-SWE-agent | `04d809ceab9df28f9adaed044884180159172930` |
| EfficientAgent | `b21b12a07a3ccf473c82acf00820b1aa2073044b` |
| SWE-bench grader (v4.1.0) | `726c5461e2ef52d83cf1ea2107870a8bb3328d57` |

## Freeze the workload

Choose an immutable Hugging Face dataset revision, then reuse the same file
for every arm. Without `--ids`, selection uses lexically sorted IDs, not a
random or success-filtered sample. Start with two tasks, then predeclare a
larger subset before comparing outcomes.

```bash
build/agent-mini/venv/bin/python -m research.agent_baselines.live select \
  --dataset princeton-nlp/SWE-bench_Verified \
  --revision "$DATASET_COMMIT" --limit 2 \
  --out build/agent-inputs/instances.json --execute
```

The frozen file retains the original dataset rows for grading. Only each
`problem_statement` reaches the agent. Gold patches and test outcomes are not
added to prompts. Docker instance images follow the pinned mini convention;
record their resolved digests on the test host because image tags can move.

## Run the live comparison

Start a dedicated inference server and copy `server-info.example.json` outside
the source tree. Replace every placeholder with the actual configuration.
Use a model that fits the target GPU with room left for KV state. The server
must support native bash tool calls and the `return_token_ids` extension:
top-level `prompt_token_ids` and output `choices[0].token_ids`, including correct
usage counts. vLLM 0.13 is the reference for this token-ID contract. The recorder
also handles LiteLLM's relocation of output IDs into `provider_specific_fields`.
Missing IDs fail capture; text is never retokenized and labeled exact.

Set `OPENAI_API_KEY` in the process environment if the endpoint needs one; do
not put credentials in URLs or provenance. The endpoint is explicit and uses
the `openai/<served-name>` provider prefix. Docker runs the generated commands
inside SWE-bench containers. The harness does not provision a GPU or launch
a serving process.

```bash
build/agent-mini/venv/bin/python -m research.agent_baselines.live run \
  --checkout build/agent-mini/source \
  --instances build/agent-inputs/instances.json \
  --server-info /path/to/server-info.json \
  --endpoint http://127.0.0.1:8000/v1 --model openai/served-model \
  --metrics-url http://127.0.0.1:8000/metrics \
  --arm stock --workers 1 --steps 30 --task-seconds 1800 \
  --out build/agent-runs/stock --execute
```

Restart the server with the same configuration and repeat with
`--arm stable-prefix --out build/agent-runs/stable-prefix`. That arm moves the
intact shared `<instructions>` block ahead of `<pr_description>` within the
same user message. It preserves both blocks and changes their order. This can
increase cross-task shared prefix length; it can also change model behavior.
Only live grading can assess the latter. Nothing removes, summarizes, or
rewrites task content.

For a useful first matrix, compare stock with prefix caching disabled, stock
with prefix caching enabled, and stable-prefix with prefix caching enabled.
Run each from a fresh server, use the same model, seeds, task set and budgets,
and repeat in reversed arm order. The runner records the operator-declared
cache start state; it does not flush or attest to server state. Do not change
temperature, truncation, tool definitions, or output limits between arms.

The defaults are a bounded smoke workload, not the paper's task budget.
Increase steps/context only after the two-task capture and grading gates pass.
Task wall time is checked at agent steps, so an in-flight request or upstream
retry can exceed it. Request timeout is recorded separately.

Outputs include `manifest.json`, the exact config and selected rows,
`predictions.jsonl`, trajectories, `task_results.json`, and `summary.json`.
`tasks/` and `attempts/` hold EfficientAgent-compatible token/timing records.
The manifest records installed package versions. Raw trajectories and tokens
contain workload content; keep them in the experiment's results storage.

## Grade every live arm

```bash
python -m research.agent_baselines.live evaluate \
  --checkout build/agent-grader/source \
  --python "$PWD/build/agent-grader/venv/bin/python" \
  --run build/agent-runs/stock --out build/agent-grades/stock --execute
```

Repeat for the other arm with a fresh output directory. Grading reads the
run's frozen dataset rows, not the current dataset head. Keep harness reports,
test logs, errors and empty patches. A zero harness process exit code does not
mean all tasks resolved. The live summary deliberately stays `ungraded`;
derive resolved rate from the grader reports using **all selected tasks** as
the denominator and report infrastructure failures separately.

## Profile and replay the captured calls

```bash
build/agent-ea/venv/bin/python -m research.agent_baselines.replay build-trace \
  --checkout build/agent-ea/source --source-run build/agent-runs/stock \
  --data-kind recorded-agent --out build/agent-trace/stock --execute

build/agent-ea/venv/bin/python -m research.agent_baselines.replay profile \
  --checkout build/agent-ea/source --trace build/agent-trace/stock/trace \
  --out build/agent-profile/stock --execute
```

The adapter validates recorded IDs, timing, trial identity and trace hashes.
It preserves origin/provenance beside the upstream trace. Transport failures
are counted as omitted from replay; do not treat their omission as success.
CPU profiles estimate within-task prefix reuse potential; the pinned upstream
profiler does not attribute cross-task reuse. It also does not measure cache
admission or eviction. Cross-task potential and unique shared KV occupancy
need a separate trace analysis before making claims about either.

The strict replay path requires vLLM **0.13.0** and LMCache **0.3.12** in the
interpreter supplied with `--python`. Install that serving stack separately.
Use a local model directory and match the trace's model/tokenizer revisions.
Replay uses `--dtype bfloat16` by default, with `float16` also supported, and
sets KV dtype to the same resolved model dtype. The recorded KV dtype must
match; an unresolved `auto` label or FP8 trace is rejected by this reference
adapter. Model revision labels are operator receipts, not a hash of all
weight files; archive weight and tokenizer manifests on the serving host.
`--kv-bytes-per-token` is the model's KV size **per tensor-parallel rank**;
compute it for the actual architecture and dtype, not from the paper's GPU
count. For ordinary decoder attention without replicated KV heads, this is
`2 * layers * local_kv_heads * head_dim * bytes_per_element`; architectures
with compressed or recurrent state need their own accounting.

```bash
python -m research.agent_baselines.replay grid \
  --checkout build/agent-ea/source --python /path/to/serve/bin/python \
  --trace build/agent-trace/stock/trace --model /path/to/local/model \
  --model-revision "$MODEL_COMMIT" --tokenizer-revision "$TOKENIZER_COMMIT" \
  --max-model-len 16384 --kv-bytes-per-token "$KV_BYTES_PER_RANK" \
  --tp 1 --host-capacities 1 4 --worker-counts 1 2 \
  --out build/agent-grid/stock --execute
```

`grid` writes commands only, even with `--execute`. It includes GPU-only
prefix caching and matched host capacities with unfiltered, fixed-threshold,
and conditioned admission. Inspect `grid.json`, then explicitly execute the
selected command with `--execute`. The replay wrapper writes its own manifest
and log; upstream output is under `run/`. A replay passes only if every
expected task/call completes, with no interruption, forced-output mismatch,
prompt-length mismatch, or request error. Increase the default replay time
budget for full traces; a time-limited partial run is a failure.

A W7900 run with a smaller model or a ROCm-specific stack is a portability
experiment. It is not the paper's 8-H20 result. The full 30B BF16 weights alone
need roughly 60 GB before KV state and runtime allocations, so a single
48-GB W7900 needs a smaller or quantized model, or a different placement.
Quantization changes the baseline and must be reported. The strict replay
version guard is intentional; a newer ROCm port needs separate validation.

## Measurement and validation

Report task resolved rate, submitted/empty patches, call count, total prompt
tokens, observed cached tokens, task wall time and client call latency. The
non-streaming recorder measures complete HTTP-call latency, **not TTFT**.
TTFT, prefill compute time and prefill tokens actually computed require engine
metrics or additional server instrumentation. Do not infer them from total
prompt length alone. Unique KV occupancy and GPU/host memory high-water marks
likewise require server-side evidence.

Local prefix hit ratio is `sum(hit-token deltas) / sum(query-token deltas)`
from matching vLLM model/engine counter series. Missing counters, changed
labels or observed resets produce an unknown value, never an invented zero.
Only use these deltas on a dedicated, uninterrupted server and allow engine
stats to flush. Two samples cannot detect every restart. Global counters do
not distinguish same-agent from cross-agent reuse or local hits from external
host-cache hits. Report those categories separately only when trace analysis
or engine instrumentation supports the attribution.

For reuse analysis, compare token-weighted reuse potential with measured hits,
vary worker count and host capacity independently, and retain the exact task
order/timing. Whole-task uniqueness is not the same as longest-prefix reuse,
and matching text fragments at different positions are not automatically
valid KV reuse.

```bash
python -m unittest discover -s research/agent_baselines/tests -v
PYTHONPATH=build/agent-mini/source/src LITELLM_LOCAL_MODEL_COST_MAP=True \
  build/agent-mini/venv/bin/python -m unittest \
  research.agent_baselines.tests.test_recording_model -v
```

The local HTTP contract fixture exercises the pinned mini model class and
LiteLLM conversion without a GPU. A separate test agent should then perform:
two-task live capture; stock and stable-prefix free-generation grading;
exact trace construction and CPU profiling; a tiny replay with complete
postconditions; and finally the matched worker/capacity matrix. Archive the
code/source SHAs, frozen inputs, config, resolved packages, hardware/driver
details, raw logs and grading reports. No quality or performance improvement
is claimed by these automation scripts.

References: [mini-SWE-agent SWE-bench workflow](https://mini-swe-agent.com/latest/usage/swebench/),
[EfficientAgent release](https://github.com/KunmingSHAO/efficientagent_release),
[paper](https://arxiv.org/html/2609.33762v1),
[official grading guide](https://www.swebench.com/SWE-bench/guides/evaluation/).
