# Reproducing the tool-use capability ladder

This directory reproduces the published results of *AgentFloor: How Far Up the
tool use Ladder Can Small Open-Weight Models Go?*
([arXiv:2605.00334](https://arxiv.org/abs/2605.00334),
[code](https://github.com/rkarmaka/AgentFloor), MIT) from the authors' released
run corpus. It needs no GPU and runs no inference.

It exists because that ladder is the only published measurement of how far
open-weight models in the 0.27B-32B range get on multi-step tool use, and
because anything built on top of it inherits whatever is wrong with it. Scoring
the corpus first is the cheapest way to find out whether the foundation holds.

## What the ladder measures

Thirty deterministic tasks over six tiers, from instruction following through
single tool call, two-tool chaining, branching, multi-source synthesis, and
long-horizon planning. Sixteen open-weight models plus a frontier anchor,
16,542 scored runs at temperature zero.

The result that matters for anyone reasoning about small models as agents is
the ceiling rather than the floor. Two-tool chaining clears an 80% reliability
bar at no size in the corpus, and branching and above never clear it at any
threshold. Long-horizon planning is 0% for every open-weight model tested. So
the band where these models can be measured doing anything reliably is roughly
one to four tool calls, and any evaluation built above that band is measuring
noise.

The spread inside that band is wide and, usefully, is not predicted by parameter
count. `ministral-3_14b` is beaten cell for cell by its own 8B sibling.
`mistral-small3.2_24b` scores 96% and 93% on the first two tiers and then falls
to 16% on two-tool chaining, a tier where a 2B model reaches 80%. Measured
capability carries information that size does not.

## Running it

```sh
./reproduce.sh [checkout-dir]      # defaults to ~/agentfloor
```

It clones the upstream repository, builds a virtualenv, downloads the 25 MB run
corpus and the judge cache from the `v1.0-data` release, applies the patch
described below, and scores the corpus.

## Result

The published table reproduces. Scoring the released corpus loads exactly 16,542
runs, filters to 12,330 in the `paper_baseline` subset, and returns per-tier pass
rates matching the paper for every open-weight model in the corpus. The full
output is in [tcr_matrix_reproduced.txt](tcr_matrix_reproduced.txt).

## Three defects block the released code

Scoring the corpus and running even a three-task sweep each hit a separate
defect. All three are one-liners, all three are carried here as patches, and all
three should go upstream. Taken together they mean the released tree cannot
reproduce the paper and cannot run its own primary sweep, which is worth stating
plainly: the published reproduction path had not been exercised end to end
before release.

### 1. Scoring crashes on undefined diagnostics

`runs/run_metrics.py results/ --subset paper_baseline`, the command the upstream
README gives for reproducing the paper's tables, fails as released:

```
TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'
```

`_diagnostics_from_ingredients` returns `None` when a ratio's denominator is
undefined, and documents that it does so deliberately rather than returning a
zero that would flatter the model. The bootstrap loop in
`add_cis_to_diagnostics` then filters `float("inf")` but not `None`, so a `None`
reaches the sample list and sorting it raises.

Both ends of the ladder reach it. A model good enough never to emit a malformed
call leaves the error-recovery ratio undefined; a model too weak to emit any call
leaves the malformed-call and hallucination ratios undefined.

Fix: [metrics-none-ci.patch](metrics-none-ci.patch).

### 2. The sweep looks for tasks in the wrong directory

`runs/run_sweep.py --eval` resolves the task set as `_HERE / "tasks"`, where
`_HERE` is the `runs/` directory, so it looks for `runs/tasks`, which does not
exist. Every scored file fails with `could not find task YAML`. The file already
defines `_REPO_ROOT = _HERE.parent` on the line after `_HERE` and uses it
elsewhere, so the correct constant was to hand.

Fix: [sweep-taskdir.patch](sweep-taskdir.patch).

### 3. The runner crashes for every provider except Gemini

`harness/runner.py` calls `provider.reset_run_state()` unconditionally at the
start of every run. That method is declared on `Provider`, which is a
`typing.Protocol` — so it supplies no implementation to the provider classes,
none of which inherit from it. Only `GeminiProvider` defines it, to reset a
synthetic tool-call id counter.

So every run through the OpenAI-compatible provider, which is what the ollama
and vLLM backends use, dies with

```
AttributeError: 'OpenAICompatibleProvider' object has no attribute 'reset_run_state'
```

The primary sweep of the paper is `ollama_full.yaml`, which runs through exactly
that provider. The protocol's own docstring says "Default is a no-op; providers
that hold per-run state override this", so the faithful fix is to give the
providers that hold no per-run state the documented no-op.

Fix: [provider-reset-run-state.patch](provider-reset-run-state.patch).

This one implies the released code has diverged from whatever produced the
corpus, since the corpus contains 12,330 runs through this provider that the
released tree cannot generate.

## Running it locally

The upstream sweep ships in two passes and this matters for any comparison.
`ollama_full.yaml` is pass one: a single prompt variant at one run per
combination, 480 runs across the full model set. `ollama_full_pass2.yaml` is
pass two: five variants at five runs, 12,000 runs, and it is pass two that the
paper's numbers come from. A pass-one run therefore yields five observations per
tier against the paper's hundred and twenty five, which is a smoke test and not
a reproduction.

[compare_tcr.py](compare_tcr.py) compares two `run_metrics.py` outputs. It
reports interval overlap first, because that is the defensible check when the
two sides carry different sampling, and the stricter point-in-interval result
alongside it.

A pass-one smoke run of four models spanning the published spread, on one 48 GB
card, gives overall rates of 33%, 37%, 60% and 53% against published 29%, 44%,
55% and 48%, with 20 of 28 tier intervals overlapping. Most of the
non-overlapping cells are degenerate: five of five passes collapses a bootstrap
interval to [100,100], which cannot overlap anything below it. The full output
is in [tcr_matrix_local_1run.txt](tcr_matrix_local_1run.txt).

Throughput is the useful number from that run. Tasks complete in roughly one to
thirty seconds depending on model size, so the 12,000-run pass-two sweep is on
the order of ten hours on a single workstation card rather than the "O(days)"
the upstream README reports for its own host. Renting equivalent hardware is
therefore unnecessary unless the goal is wall-clock parallelism.

## Watching a model actually call a tool

[walkthrough.py](walkthrough.py) reads a run record out loud. It exists because
a pass rate tells you nothing about how tool calling fails, and the run records
hold the whole exchange: the raw argument text the model emitted, whether that
text parsed, whether the tool it named exists, what the environment replied, and
what the model did next.

```sh
./walkthrough.py walk    <a-run>.json      # narrate one run, turn by turn
./walkthrough.py taxonomy <results-dir>    # count how calls go wrong, per model
```

A 0.6B model on a one-step approval task, in
[example_walkthrough.txt](example_walkthrough.txt), is the clearest lesson
available in three turns. It emits `{"status":"ok","submission_id":"REQ-220"}`.
The JSON is valid and the schema passes. The tool answers `action mismatch:
expected 'approve', got 'None'`. The model reads that error and corrects itself,
changing `status` from `ok` to `approve` — and fails identically, because the
field the tool wants is `action`, and `status` was never the problem. It matched
the word in the error message and put it in the wrong field.

Syntax was never the difficulty. Emitting well-formed JSON against a published
schema is the easy half; knowing which field an error is about is the hard half.

### What the failure counts say

From a pass-one run of four models, in
[call_failure_taxonomy.txt](call_failure_taxonomy.txt):

| model | calls | invented a tool | malformed args | runs with no call at all |
| --- | ---: | ---: | ---: | ---: |
| qwen3:0.6b | 32 | 0% | 3% | **43%** |
| qwen3.5:2b | 117 | 3% | **13%** | 20% |
| ministral-3:8b | 85 | 0% | 0% | 17% |
| qwen3:14b | 40 | 0% | 0% | 33% |

Read the first column before the others. The 0.6B model has the lowest malformed
rate of the three small models, and that is not a competence result: it emitted
32 calls where the 2B emitted 117, and in 43% of its runs it never called
anything, answering in prose instead. Its protocol looks clean because it barely
enters the protocol.

Protocol failure peaks in the middle. The 2B model tries hardest, emits the most
calls, and is the only one that invents tools that do not exist and malforms
arguments at a double-digit rate. Above it, protocol errors go to zero and what
remains is failure at the task.

That shape is the project's own confound appearing inside the protocol metrics.
A rate computed against calls emitted rewards a model for not participating, in
exactly the way a safety rate computed against all runs rewards a model for being
unable to act.

## Where the fixes go

The three patches here exist so that `reproduce.sh` works against the released
upstream tree. This directory is not a fork of that project and is not where the
fixes get submitted. They belong upstream, and a branch carrying them as three
separate commits against that project's own history is prepared outside this
repository.

## Provenance

- Upstream commit: `rkarmaka/AgentFloor` default branch, cloned 2026-08-24.
- Corpus: `v1.0-data` release, `agentfloor-runs-v1.tar.gz`,
  sha256 `cb450cab2b69d86581f0fa7afaeb9e7b534eaa539bac4f49bef8cd9d7f70d106`.
- Judge cache: same release, `llm_judge_cache.jsonl`,
  sha256 `b67fc7c62435da30fe6586748117232d6a03c18a8306c0c4556bdfb3ec635166`.
- The upstream README states the release carries SHA-256 checksums; the release
  body does not contain them, so the two hashes above are recorded here instead.
- Scored with Python 3.14.6, PyYAML 6.0.3, jsonschema 4.26.0.
