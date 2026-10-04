# Agent cache baselines

Use mini-SWE-agent as the default live SWE-bench workload. It gives a small,
inspectable agent loop and patches that the official harness can grade.
Keep EfficientAgent as a separate fixed-trace analysis and replay baseline.
Together they measure both useful task completion and the serving cost of a
controlled sequence of calls.

This directory is a standalone research harness; it does not change knlp's
training or serving paths. The local bootstrap, selection, live, grading and
replay commands print a plan unless `--execute` is supplied. The documented
`modal run` and `modal deploy` commands execute directly, can allocate cloud
resources, and do not have this dry-run contract. Each local run requires a
new output directory. Python 3.10+ is required; run commands from the knlp
repository root.

PrefixPlan's first backend-neutral contracts are in `workflow.py`: prompt
segments, ordered layout hashes, request dependencies, tool-state effects and
cache-event provenance. They deliberately distinguish offline reuse potential,
client-requested actions and engine-observed physical events. The SGLang
adapter and its exact source boundaries are documented in
[`BASELINE_AUDIT.md`](BASELINE_AUDIT.md).

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
Use a model that fits the target GPU with room left for KV state. Set the
server-info `backend` to `vllm` or `sglang`. A vLLM server must support native
bash tool calls and the `return_token_ids` extension:
top-level `prompt_token_ids` and output `choices[0].token_ids`, including correct
usage counts. vLLM 0.13 is the reference for this token-ID contract. The recorder
also handles LiteLLM's relocation of output IDs into `provider_specific_fields`.
Missing IDs fail capture; text is never retokenized and labeled exact.

The initial SGLang adapter is pinned to
`cbe070b6bfce18936620f406b9b0d71bdc17452b`. Its server-info must also include
that value as `server_revision` and list `openai_chat` and `sglext_token_ids`
in `enabled_capabilities`; add `cache_source_report` when requesting tiered
device/host/storage counts. Exact IDs come from response-level `sglext`, and
missing or conflicting identities abort the call. Capability declarations do
not establish cache hits: only returned engine details count as observations.
SGLang server-info must also set an explicit `cache_scope`. All tasks in the
same authorized experiment share that value so reusable prefixes have one
identity; workflow and request IDs remain task-local receipts. Use a different
scope whenever the tenant, model, adapter, or experiment changes. The adapter
passes this scope as SGLang's `extra_key` and never derives it from a task ID.

`modal_sglang_cache_scope.py` is the bounded live mechanism check for this
contract. This command executes immediately on Modal. It sends the same token
prefix under two task-local request IDs in one shared scope, then under a
different scope. A pass requires a positive live cache hit only for the shared
scope and identical freely generated output IDs.

```bash
modal run research/agent_baselines/modal_sglang_cache_scope.py \
  --run-name prefixplan-YYYYMMDD-sglang-cache-scope \
  --knlp-commit "$(git rev-parse HEAD)"
```

For Modal, `modal_live_server.py` provides the pinned vLLM endpoint used by the
bounded live pilot. Create the named secret with a random `VLLM_API_KEY`, fetch
the exact model revision into `knlp-hf`, and deploy the app. Export the same
value as `OPENAI_API_KEY` only in the client process; do not put it in a URL or
archive it. Stop the deployed app after each arm and redeploy it before the next
arm so every comparison starts with a fresh serving process.

```bash
modal secret create knlp-prefixplan-live-api VLLM_API_KEY="$RANDOM_API_KEY"
modal run research/agent_baselines/modal_live_server.py --fetch
modal deploy research/agent_baselines/modal_live_server.py
```

The deployed URL ends immediately before `/v1`; copy it into the live command's
`--endpoint` with `/v1` appended. The public endpoint still requires the vLLM
API key. Delete the temporary Modal secret after the campaign.

## Qualify SGLang host restoration

`modal_host_restore.py` is a bounded Modal H100 mechanism test for the physical
host-cache gate. It uses an immutable SGLang image, overlays the exact pinned
source revision, downloads an exact model revision, constrains device KV to
8,192 tokens, and retains a 1 GiB HiCache host pool. It inserts four unrelated
3,000-token prompts between two visits to a 3,000-token target, then flushes
the caches and recomputes the target cold.

```bash
modal run research/agent_baselines/modal_host_restore.py \
  --run-name prefixplan-YYYYMMDD-modal-h100-cbe070b \
  --knlp-commit "$(git rev-parse HEAD)"
```

The gate requires positive engine-reported host retrieval, no host retrieval
in the post-flush cold control, and exact equality of the freely generated
greedy output token IDs. Chosen-token logprobs are retained as a stricter
diagnostic with both `1e-5` and `5e-3` thresholds. Three independently
cache-flushed cold recomputations also provide a cold-vs-cold repeatability
baseline for interpreting the restore-vs-cold delta. The harness writes
`result.json`, `server.log`, and `REPORT.md` to the `knlp-runs` Modal volume,
and preserves `failure.json` on errors.

The 2026-10-02 H100 qualification restored 2,994 host tokens and produced the
same 16 output token IDs as cold recomputation. Its maximum chosen-token
logprob delta was 0.00193237, so the `1e-5` diagnostic did not pass. This is a
synthetic forced-eviction check, not evidence of agent quality, throughput
benefit, or conditioned admission under concurrent pressure.

A follow-up run repeated the cache-flushed cold control three times. All three
cold generations had identical output IDs and chosen-token logprobs, for a
maximum cold-to-cold delta of zero. The restored generation again matched the
output IDs but differed from every cold control by `0.0019323714`. The relaxed
`5e-3` threshold therefore describes a narrow continuation check; it is not a
numerical-equivalence tolerance justified by ordinary cold-run variability.

## Qualify conditioned admission under pressure

`modal_conditioned_pressure.py` is the corresponding bounded mechanism test
for EfficientAgent's conditioned-admission path. It pins EfficientAgent,
vLLM, LMCache, the container digest, and the model revision. Four concurrent
three-step synthetic tasks share a 2,048-token prefix while a 0.1 GiB host
tier is driven to capacity. Its threshold is measured in new chunks that a
request would write, not in prior reuses.

```bash
modal run research/agent_baselines/modal_conditioned_pressure.py \
  --run-name prefixplan-YYYYMMDD-conditioned-pressure \
  --knlp-commit "$(git rev-parse HEAD)"
```

The 2026-10-02 Modal H100 run completed all 12 forced-output requests exactly.
At 14/14 host entries with active eviction, the scheduler's first-hit receipt
recorded one pressure decision and skipped the write of 11 new chunks; final
telemetry recorded 28 host evictions in the observation window. This closes
the narrow mechanism gate that conditioned admission encounters real pressure
and takes its skip-write branch. It does not establish a performance or
quality benefit.

The pinned upstream stack has a reporting defect: its shutdown aggregate
counter files retain only connector-initialization counters even though the
same run's first-hit receipts and final telemetry preserve the pressure event,
decision counts, and evictions. The qualification therefore checks those
authoritative receipts and archives `server.log` as well as `run.json`. Fix the
aggregate-counter lifecycle before using it for comparative storage-traffic
claims.

`efficientagent_stats_instances.patch` supplies that experimental reporting
repair for short comparative runs. It gives each connector instance a unique
role/PID/instance file and flushes after every counter update, so SIGTERM at
server shutdown cannot erase the final decision totals. The Modal image tests
the patch against the pinned EfficientAgent checkout before launching a GPU.

## Compare pressure decisions with later restoration

`modal_policy_restore_compare.py` connects admission, pressure, eviction, and
later host restoration in the same pinned EfficientAgent/vLLM/LMCache stack.
Every arm uses the same connector and copy-time deduplication. A 4,096-token
GPU pool and 0.16 GiB host tier receive a sacrificial prefix, the target,
three pressure prefixes, and a target revisit. The arm order is rotated over
three repetitions.

```bash
modal run research/agent_baselines/modal_policy_restore_compare.py \
  --campaign prefixplan-YYYYMMDD-policy-restore \
  --knlp-commit "$(git rev-parse HEAD)" --repetitions 3
```

The unfiltered arm writes every request. The fixed arm rejects requests above
the two-new-chunk threshold without consulting pressure. The conditioned arm
writes above-threshold requests until fresh telemetry reports a full, evicting
host tier, then skips later writes. No post-replay probe is issued, so serving
counters and LMCache transfer logs cover the same request window. The summary
reports computed prefill tokens, host transfer tokens, target-revisit latency,
makespan, and preserved policy counters. This remains a synthetic mechanism
comparison, not evidence of live-agent quality or a general PrefixPlan win.

The 2026-10-02 Modal H100 campaign completed all nine arms and all 54 forced
requests exactly. In each of three counterbalanced repetitions, conditioned
admission stored 6,144 tokens, skipped three later pressure writes, physically
restored all 3,072 target tokens, and computed 15,361 prefill tokens.
Unfiltered admission stored 18,432 tokens, restored none, and computed 18,432;
the fixed threshold stored and restored none and also computed 18,432. Median
makespans were 8.670, 8.739, and 8.639 seconds for conditioned, unfiltered,
and fixed respectively. Those timings are descriptive for this gap-paced
micro-workload; the result establishes the intended traffic/recomputation
mechanism, not a general task-completion-time benefit.

## Compare fixed layouts with bounded joint selection

`prefix_plan.py` enumerates only layouts supplied by the caller. It scores the
prefix tokens each candidate could reuse while accounting for the KV bytes its
reusable prefixes would occupy in the declared host tier. Its receipt contains
every fixed and per-family assignment, the strongest fixed candidate, the
selected joint candidate, and the capacity calculation. These are structural
estimates; they never become physical cache events without engine telemetry.

`modal_joint_layout_compare.py` gives that contract a bounded mechanism test.
One synthetic request family shares repository context while another shares
tool instructions. The two uniform arms place the same segment type first for
both families. The planner arm selects from those same approved layouts per
family. A static per-family control makes the same selection without consulting
the planner; the harness requires its rendered replay inputs to match the
planner arm exactly. All arms use the identical pressure-conditioned connector,
copy-time deduplication, GPU pool, host tier, logical requests, and forced
outputs. Unique pressure requests evict device KV between revisits, so saved
prefill must be backed by observed host restoration rather than device hits.

```bash
modal run research/agent_baselines/modal_joint_layout_compare.py \
  --campaign prefixplan-YYYYMMDD-joint-layout \
  --knlp-commit "$(git rev-parse HEAD)" --repetitions 3
```

The counterbalanced repetitions are a synthetic systems gate. The planner and
static per-family control are expected to have equal replay inputs, restoration,
and computed prefill. Any improvement over the uniform arms therefore supports
the selected static layout, not the compiler that selected it. The constructed
workload establishes neither end-to-end latency benefit nor task quality. The
next separate gate is freely generated live SWE-bench work with official grading.

The 2026-10-02 Modal H100 campaign predated the static per-family control. It
passed all nine arms and all 180 forced
outputs. In every repetition, each fixed layout physically restored 4,096
tokens and computed 57,344 prefill tokens, while the selected per-family
layout restored 8,192 and computed 53,248. The observed 4,096-token reduction
exactly matched the planner's incremental estimate (7.14 percent of the fixed
arm's computed prefill). Median makespans were 28.718, 28.730, and 28.731
seconds for fixed-repository, fixed-tool, and joint respectively, so this run
does not establish a latency benefit or a compiler advantage over a fixed
per-family rule.

A separate two-task live pilot used fresh vLLM 0.13 servers, the exact
`Qwen/Qwen3-Coder-30B-A3B-Instruct` revision, native prefix caching, one
sequence at a time, and official SWE-bench grading. Stock ordering made 56
successful calls and submitted one incorrect patch; stable-prefix ordering
made 60 successful calls and submitted no patches. Both scored 0/2. Stock's
cold-server absolute counters reported 585,360 hits from 605,517 queried
prefix tokens (96.67 percent); stable-prefix's interval counters reported
600,800 from 616,908 (97.39 percent). Mean client latencies were 4.432 and
2.611 seconds, but the call counts, completions, and generated trajectories
differ, so the pilot is neither a matched performance comparison nor evidence
of a quality improvement. Larger predeclared campaigns remain necessary.

Set `OPENAI_API_KEY` in the process environment if the endpoint needs one; do
not put credentials in URLs or provenance. The endpoint is explicit and uses
the `openai/<served-name>` provider prefix. Docker runs the generated commands
inside SWE-bench containers. The live-agent harness does not provision a GPU
or launch a serving process; the bounded Modal qualification above is the
explicit exception.

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

### Quality and layout matrix

PrefixPlan treats official tasks resolved within fixed budgets as the primary
objective. Reuse, prefill, decode, latency and residency are measured costs and
mechanisms; a cache-hit ratio is not a quality result. The live runner exposes
a frozen two-by-two pilot:

| Arm | Instruction profile | Layout |
| --- | --- | --- |
| `stock` | pinned upstream wording | task first |
| `stable-prefix` | pinned upstream wording | instructions first |
| `clear` | `clear-technical-v1` | task first |
| `clear-stable-prefix` | `clear-technical-v1` | instructions first |

`clear-technical-v1` preserves the operational information, technical names,
boundaries and exact submission command while removing repetition and using
shorter, explicit instructions. It is an STE-inspired experimental profile,
not a claim of ASD-STE100 compliance. Each manifest records the wording and
layout independently, hashes the complete template and instruction block, and
records the task step, wall-time, request-time and output-token budgets.

Hold the selected tasks, model, tools, decoding settings and budgets constant.
Use unforced generation and official grading. Start every arm with a fresh
serving process and an isolated cache scope, counterbalance arm order across
repetitions, and count all selected tasks in the denominator. Compare wording
with layout held fixed and layout with wording held fixed. Add rationales,
worked examples or explicit generated state only in later ablations; otherwise
the first matrix cannot separate clearer wording from additional information.

The follow-up completion diagnostic keeps stock wording and task-first layout
fixed. Compare stock at 30 calls, stock at 60 calls, and `finish-protocol` at
30 calls. The 60-call arm measures additional opportunity and is not an
equal-cost treatment. `finish-protocol` appends the exact remaining-call count
to every tool observation, requires a verification decision with five calls
left, and reserves the last three calls for creating, inspecting and submitting
`patch.txt`. It also requires short observed-fact, unresolved-check and
next-action state. Freeze those thresholds before running; do not tune them per
task.

The next diagnostic decomposes that combined treatment at a fixed 30-call
budget. Run `stock`, `evidence-state`, `finish-window`, and `finish-protocol`
as a two-by-two matrix. `evidence-state` adds only the observed-fact,
unresolved-check, next-action, and executable-assertion guidance.
`finish-window` adds only the live remaining-call state, five-call checkpoint,
and final create/inspect/submit window. The combined arm remains byte-identical
to the earlier `finish-protocol-v1` treatment. Freeze task selection and arm
order before grading, repeat with a counterbalanced order, and report protocol
adherence alongside official task success.

Every live task writes `terminal_captures/<instance>.json` before its container
is removed. Capture schema v4 records the terminal commit, the combined tracked
diff from `HEAD`, and separate unstaged and staged diffs. When terminal export
is enabled, it also records the combined tracked diff from the immutable task
base which the runner verified before the agent started. The receipt includes
changed and untracked path lists, exit reason, submission identity, last
command result and the last command selected by an explicit
verification-command heuristic. Untracked contents are not captured. The
capture is never added to model context or the submitted patch. Missing,
nonzero, timeout-shaped or malformed capture-command results retain the partial
receipt, fail the run as infrastructure, and do not replace the original patch
or submission status.

`--terminal-export terminal-export-v1` enables an optional zero-model-call
runner protocol. Before agent execution, the runner resolves the dataset's
40-character base commit and keeps that identity outside agent control. It
preserves a nonempty explicit Git diff that applies to a clean index at that
base. If the agent instead reaches its call or time limit, or submits proven
invalid content, it validates and exports the final tracked diff from the same
base. This includes committed, staged and unstaged tracked changes while it
excludes untracked files.

Validation has three states. `valid` means the complete patch check succeeded.
`invalid` means a completed check rejected the patch content. `unavailable`
means the base, clean index, tool response or validation operation failed. An
unavailable check never permits replacement of an explicit submission. The
validator first checks the frozen base tree for missing objects. It treats only
recognized Git content diagnostics, such as an inapplicable hunk or malformed
patch, as `invalid`; ambiguous nonzero checks remain `unavailable`. The task
receipt exposes that outcome, and the campaign summary counts it as an
infrastructure failure independently of the agent's original terminal reason.
The protocol does not search earlier checkpoints, repair code, retry the agent,
add model calls, or use hidden grades. The runner keeps the original exit
reason and records task and dataset identity, frozen base, terminal commit,
patch hashes, validation states, model-call count and timings in
`terminal_exports/<instance>.json`. This is a separately named reliability
protocol; it does not revise historical explicit-submission scores.

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
