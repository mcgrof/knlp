# Read the matched HTTP diagnostic profiles

`serving_profile_audit` is a CPU analysis tool for the separately frozen
Qwen3 HTTP profiles. It adds no confirmation blocks and cannot establish
a throughput or latency gain from instrumented timings.

The diagnostic uses one run per arm. Each run retains the full first-block
closed-concurrency-32 prelude, then profiles the first 64 arrivals of the
original near-load trace. The shorter profile window bounds trace volume;
it is not the full primary measurement interval. Model, scales, source,
runtime policy and inputs remain fixed across the three arms.

Retain the frozen profile directory (`PROFILE-PLAN.json`, the three
`plan-ARM.json` files, `run.py`, qualifications and input traces), the
independent completed parent confirmation, and the raw captures. Each
capture contains the server's `trace.json.gz`, scheduler `steps.json`,
operation records, client events, worker receipts and source/input hashes.

```sh
python -m tools.kv.native.serving_profile_audit \
  --root ./http-captures --freeze ./profile-plan \
  --confirmation ./confirmation-audit.json \
  --output ./profile-audit.json
```

The reader verifies the original parent-result and design hashes, each
arm's exact runtime-protocol match to the first primary block, the wrapper
and plan identities, qualification, calibrated inputs and physical storage.
It also checks the installed FP16 weight inventory. A changed parent or
runtime protocol is rejected. Primary failures remain in the parent result;
a successful diagnostic does not erase them.

CUDA launch correlation attributes kernel events to recorded scheduler
steps. Prefill/mixed and decode phases follow the actual scheduled token
counts; prefill can include recomputation after preemption. The separate
`serving_admission_audit` distinguishes newly admitted and resumed requests.
Named kernel counts, duration sums, unions and unmatched events remain
explicit. Gather includes its fused value conversion.

Kernel-duration sums are not causal decompositions of request wall time.
Host `execute_model` ranges can include waiting and asynchronous launches,
so they are not pure CPU compute. CUDA graphs can execute without entering
per-layer Python wrappers. Missing wrapper rows do not prove an absent
attention operation or a graph fallback. Native FA3 has a different wrapper
path from FlashInfer; all of its observed kernels still remain in the data.
No client/server clock subtraction is used for request latency.

## Supplementary driver memory samples

When a complete capture includes `environment/gpu-samples.csv` and its
UTC timezone receipt, the CPU driver reader can report observed memory
maxima across each full arm command, including startup and shutdown:

```sh
python -m tools.kv.native.serving_driver_audit \
  --root ./http-captures --prefix fp16-1k \
  --output ./driver-memory-audit.json
```

Read the immutable capture after recording stops. The reader requires all
33 planned command windows and retains sample gaps, edge coverage, raw
hashes and envelope violations. Its 2.5-second coverage flag describes
sampling completeness; it is not a new performance gate. One-second
samples can miss shorter peaks. Same-host wall times select the command
windows and never replace monotonic request latency. A missing sampler
cannot be reconstructed from the separate boundary snapshots.
