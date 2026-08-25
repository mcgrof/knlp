Three defects stop the released tree from reproducing the paper or running the
sweep the paper is built on. Each is a one-liner and each is a separate commit.

I found these while reproducing the ladder as a foundation for other work, so
the motivation is entirely selfish: I wanted the numbers to come out, and they
now do.

### 1. Scoring crashes on undefined diagnostics

`runs/run_metrics.py results/ --subset paper_baseline`, the command the README
gives for reproducing Tables 1 and 2, fails on the released corpus:

```
TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'
```

`_diagnostics_from_ingredients` returns `None` for ratios with an undefined
denominator and documents that as deliberate. `add_cis_to_diagnostics` filters
`float("inf")` but not `None`, so a `None` reaches the bootstrap sample list and
sorting raises. Both ends of the ladder reach it: a model good enough never to
emit a malformed call leaves `err` undefined, and a model too weak to emit any
call leaves `sdr` and `thi` undefined.

### 2. The sweep resolves its task set from the wrong directory

`run_sweep.py` computes `tasks_dir = _HERE / "tasks"`, where `_HERE` is the
`runs/` directory, so `--eval` looks for `runs/tasks` and every scored file
fails with `could not find task YAML`. The module defines
`_REPO_ROOT = _HERE.parent` on the next line and already uses it elsewhere.

### 3. The runner crashes for every provider except Gemini

`harness/runner.py` calls `provider.reset_run_state()` at the start of every
run. `Provider` declares it, but `Provider` is a `typing.Protocol`, which
supplies no implementation to the adapter classes, none of which inherit from
it. Only `GeminiProvider` defines it. So every run through `OpenAIProvider` or
`AnthropicProvider` raises `AttributeError`, and `OpenAICompatibleProvider`
subclasses `OpenAIProvider` and is what the ollama and vLLM backends use. That
means `ollama_full.yaml` and `ollama_full_pass2.yaml` cannot run as shipped.

The protocol docstring says "Default is a no-op; providers that hold per-run
state override this", so the fix gives that no-op to the two adapters that hold
none.

Worth flagging: the released corpus contains 12,330 runs through a provider the
released code cannot drive, so the published tree looks to have drifted from
whatever generated it. Something else may have drifted with it, and I have only
checked what blocked me.

### Verification

Scoring was checked from a clean checkout at this branch: 16,542 runs load,
12,330 filter into `paper_baseline`, and the per-tier table matches the paper.

The sweep fixes were checked by running the sweep on one 48 GB card through
ollama. The tool-support probe reports native tool calling, tasks execute with
no errors, and pass rates line up with the published per-tier rates. A
four-model pass-one run gives overall rates of 33%, 37%, 60% and 53% against
published 29%, 44%, 55% and 48%; a matched pass-two run is in progress.

One unrelated note: the README says the `v1.0-data` release carries SHA-256
checksums, and the release body does not appear to contain them.

### Disclosure

These fixes were written with AI assistance. I ran and verified every one of
them myself on real hardware, and the analysis above is mine to defend.
