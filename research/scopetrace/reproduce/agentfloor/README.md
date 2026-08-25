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

## The released metrics code cannot score its own corpus

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

[metrics-none-ci.patch](metrics-none-ci.patch) is a one-line fix that skips
undefined values alongside infinite ones, which is what the surrounding code
already intends. It should go upstream.

That this is reachable on the first documented command, against the authors' own
data, is worth stating plainly: it means the published reproduction path had not
been exercised end to end before release.

## Provenance

- Upstream commit: `rkarmaka/AgentFloor` default branch, cloned 2026-08-24.
- Corpus: `v1.0-data` release, `agentfloor-runs-v1.tar.gz`,
  sha256 `cb450cab2b69d86581f0fa7afaeb9e7b534eaa539bac4f49bef8cd9d7f70d106`.
- Judge cache: same release, `llm_judge_cache.jsonl`,
  sha256 `b67fc7c62435da30fe6586748117232d6a03c18a8306c0c4556bdfb3ec635166`.
- The upstream README states the release carries SHA-256 checksums; the release
  body does not contain them, so the two hashes above are recorded here instead.
- Scored with Python 3.14.6, PyYAML 6.0.3, jsonschema 4.26.0.
