# PrefixPlan baseline source audit

Audit date: 2026-10-02. This records public source availability and adapter
boundaries for the first PrefixPlan implementation. A paper description is
not treated as executable evidence, and a generic SGLang feature is not
relabeled as a paper system.

| Baseline | Public source audited | License | Adapter status |
|---|---|---|---|
| SGLang | [`sgl-project/sglang`](https://github.com/sgl-project/sglang) at `cbe070b6bfce18936620f406b9b0d71bdc17452b` | Apache-2.0 | Initial OpenAI-compatible live adapter implemented |
| KVFlow | [`PanZaifeng/KVFlow`](https://github.com/PanZaifeng/KVFlow) at `7ef897eaedc16626a281ccfe0d453fae80251e8f` | Apache-2.0 | Public SGLang-based server and ScaleSim code exist; workload translation remains to be implemented |
| TOPAS | [arXiv:2608.25523](https://arxiv.org/abs/2608.25523) | No official code repository found | Paper comparison only; not executable in this harness |
| PBKV | [arXiv:2605.06472](https://arxiv.org/abs/2605.06472) | No official code repository found | Paper comparison only; not executable in this harness |

The official KVFlow repository identifies itself as the implementation of the
NeurIPS 2025 system and contains its SGLang-derived server and scheduler. It
still needs a pinned, reviewed adaptation before a matched run can be claimed.
Repository searches for the exact TOPAS and PBKV paper/system names found no
official public implementation on the audit date. If source appears later, pin
and audit it rather than silently changing the meaning of an existing arm.

## SGLang capability boundary

The SGLang revision above has:

- an OpenAI-compatible chat endpoint;
- response-level `sglext.input_ids` and `sglext.output_ids` for exact token
  capture;
- `sglext.cached_tokens_details` with device, host, and optional storage token
  counts;
- caller-supplied request IDs, classification keys, priorities, and an exact
  `input_ids` bypass;
- language IR/tracing, fork/join primitives, and HiCache host/storage modes.

The audit did **not** find a prompt-reordering compiler pass or a public
per-request API that enforces retain, release, prefetch, or cache residency.
SGLang's language IR and tracing support are therefore not called the TOPAS
compiler. Cache actions produced by offline planning remain requested or
analytical events until engine telemetry demonstrates physical enforcement.

`sglang_adapter.py` binds these findings to the audited source revision. Its
capability receipt separates source availability from operator-declared
runtime enablement. Live capture requires the chat and `sglext` token-ID
features, and the recorder aborts on missing, malformed, or conflicting token
identities. The receipt is evidence about adapter compatibility, not proof that
a cache tier was exercised.

## Qualification status

The bounded 2026-10-02 Modal H100 qualification closed the physical
host-restore mechanism gate for the pinned SGLang revision. With an 8,192-token
device pool and 1 GiB host pool, a target revisit reported 2,994 host-restored
tokens after four unrelated 3,000-token fillers. A cache-flushed cold control
reported no host retrieval, and its 16 freely generated greedy output token IDs
exactly matched the restored run. The maximum chosen-token logprob delta was
0.00193237: within 0.005 but not the stricter 1e-5 diagnostic.

Three independently cache-flushed cold repeats then produced identical chosen-
token logprobs, with a maximum cold-to-cold delta of zero. The restored run's
`0.0019323714` delta is therefore larger than the measured repeatability floor
on this pinned stack. Exact greedy-token agreement remains established;
numerical equivalence does not.

The conditioned-admission mechanism gate was then exercised with four active
synthetic tasks on Modal H100, pinned EfficientAgent
`b21b12a07a3ccf473c82acf00820b1aa2073044b`, vLLM 0.13.0, and LMCache 0.3.12.
The 0.1 GiB host tier reached 14/14 registered entries and recorded 28 recent
evictions. A scheduler first-hit receipt recorded one pressure request, one
skipped request, 11 skipped new chunks, an above-capacity estimate, and active
full-tier eviction. `SKIP_WRITE` was observed, and all 12 forced replay outputs
matched exactly.

The aggregate counter files from that pinned stack retained only connector
initialization at shutdown. The pressure qualification consequently uses the
preserved first-hit event receipts and final telemetry and records this
reporting limitation explicitly. Repair that counter lifecycle before relying
on aggregate counts for comparative storage-traffic analysis.

A bounded 2026-10-03 adapter qualification sent one 3,000-token prompt through
three task-local request IDs. The second request shared the first request's
explicit cache scope and reported 2,999 device-cache tokens. The third used a
different scope and reported no cached tokens. All three freely generated the
same eight output token IDs. This establishes live compatibility and isolation
for the adapter's scope mapping; it does not establish host-tier restoration or
agent quality.

The follow-on comparison carries a pinned reporting patch that assigns unique
role/PID/instance files and flushes each counter update. Its image-level tests
and Modal calibration preserved all six scheduler decisions and the associated
saved/skipped chunk totals across shutdown.

The matched follow-on campaign then completed nine H100 arms: three policies
over three counterbalanced repetitions, with identical connector deduplication
and fresh serving processes. All 54 forced requests matched exactly. In every
repetition, conditioned admission observed pressure, skipped later writes,
physically restored the 3,072-token target, and reduced computed prefill from
18,432 to 15,361 tokens. It stored 6,144 tokens versus unfiltered admission's
18,432; unfiltered and fixed-threshold arms restored zero target tokens.

This is a synthetic admission-policy mechanism result. It does not test a
joint layout/planning compiler, live SWE-bench quality, model-scale behavior,
or a general task-completion-time improvement. The strongest next test is the
joint planner against this conditioned baseline, followed by live grading.

The synthetic checks establish host restoration, the conditioned pressure
branch, and their connection in the matched micro-workload. They do not
establish live task quality or correctness across models and workloads. Keep
serving and post-replay probe measurement windows separate: the earlier
qualification's storage-write counter included greedy probes while serving
counters covered only replay; the matched campaign issues no post-replay
probe.
