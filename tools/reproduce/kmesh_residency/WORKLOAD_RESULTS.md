# Export the reasoning-demand result

After the complete search capture described in [WORKLOAD.md](WORKLOAD.md),
recompute its aggregate evidence on CPU:

```sh
python3 -m unittest discover -s tests -p 'test_kmesh_workload_publish.py'
python3 -m tools.reproduce.kmesh_residency.workload_publish \
  /path/to/fresh-search-capture --output public-workload-results.json
```

The exporter recalculates the optimistic prefill bound from exact private
tokens, regrades every selected answer with restricted rational arithmetic,
and checks all three task summaries, allocator events and drained references.
It checks pinned workload/model file hashes, the selected puzzles, four-step
greedy search continuity and model-call accounting. It rejects incomplete
captures, changed recorded identities and contradictory evidence. The output
retains task outcomes, whitespace-only selected extensions, output-limit versus
normal-stop counts by call purpose, aggregate cache counts, runtime versions
and capture hashes. Prompts, answers, token IDs,
request/family identities and operator paths stay in the private capture.

These checks establish consistency within the captured artifacts. They do not
independently replay tokenized search or authenticate a device run. Preserve
the measured source and raw evidence separately when qualifying an execution.
The 512 MiB value is the configured cache budget. The bound uses the observed
physical pool capacity after subtracting the reserved null block.

This is a bounded local-model adaptation of the official search. A failed
task remains in the result. The bound has future knowledge and can retain
independent blocks without real prefix or active-request constraints. Zero
headroom closes retention-policy testing for this case and budget; positive
headroom still needs attribution and displaced-work accounting. Neither
outcome establishes production utility, paper accuracy, latency or storage
performance.
