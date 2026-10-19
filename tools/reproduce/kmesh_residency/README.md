# Family residency: establish the trace evidence first

A family lease would keep an existing vLLM ancestor resident between related
requests. It consumes cache capacity that could serve other requests. A useful
comparison therefore needs observed branch timing, exact token identities,
competing requests and physical cache observations. Text reuse alone does not
establish that holding an ancestor improves latency.

The CPU auditor checks the public
[LMCache agent-trace corpus](https://github.com/LMCache/lmcache-agent-trace)
at revision `780bcc2979715150d8b9fd4737e026e625444cc9`. It authenticates the clean
checkout, hashes each selected tracked JSONL file, reports malformed records, removes
exact duplicate records within each framework, and groups descriptive text
statistics by explicit session identifier. Session identity does not establish
family lineage. Derived `_result` directories are excluded. The output contains
counts and hashes, without copying prompts or answers. The source corpus uses
the Apache-2.0 license; its README labels PyCodeAGI as a synthetic pipeline.

```sh
git clone https://github.com/LMCache/lmcache-agent-trace.git /path/to/traces
git -C /path/to/traces checkout 780bcc2979715150d8b9fd4737e026e625444cc9
make defconfig-kmesh-trace-audit
KMESH_TRACE_ROOT=/path/to/traces make kmesh-trace-plan
KMESH_TRACE_ROOT=/path/to/traces make
```

Use Python 3.10 or later. No extra Python packages, model or GPU are required.
Set `KMESH_TRACE_OUTPUT` to a fresh output directory; existing output is
rejected. Recheck portable contracts with `make kmesh-trace-test`.

The median character-prefix fraction compares consecutive input strings within
one session. It is descriptive text overlap, with no character-to-token
conversion. Numeric epoch units are inferred only for descriptive spacing
between record timestamps. Neither statistic is an observed idle interval,
active inference interval, physical cache hit, or fork gap. Malformed rows stay
visible in the report rather than silently becoming zero reuse or zero churn.

## Capturing invocation boundaries

`KMESH_TRACE_MODE=invocations` accepts JSONL records with schema
`knlp.kmesh.invocation.v1`. Each record must include globally unique
`request_id`, `family_id`, nullable `parent_request_id`, `engine_id`, `clock_id`,
integer `start_ns` and `end_ns`, exact `input_token_ids` and `output_token_ids`,
terminal `status` (`complete`, `cancelled` or `error`), and a
`cache_observations` list. Record actual invocations, including cancelled work;
retain the clock domain so unrelated monotonic clocks cannot be subtracted.
Root requests explicitly use a null parent. Capture identity must also retain
the model, adapter, cache layout and source/runtime pins in its own manifest.

The auditor rejects duplicate request identities, cyclic lineage and conflicting
family identities. It measures signed parent-completion-to-child-start gaps
only for known parents in the same clock domain. Negative gaps identify children
that start before the parent completes. It reports cross-engine edges and
non-exact parent prefixes separately. Family membership does not imply equal KV.

This first audit validates invocation metadata, not the contents of physical
cache observations. `pressure_replay_qualified` remains false. A pressure study
additionally needs actual pool capacity, block identities, reference counts,
evictions, competing-family demand and displaced hits. No fixed-concurrency
extrapolation or predicted latency verdict is emitted. A bounded capture can
qualify the mechanism without establishing a production-workload benefit.
