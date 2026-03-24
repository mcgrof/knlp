# Memory-Traffic Saturation in Autoregressive Decode: Lineage and Script Provenance

This document explains why the standalone memory-traffic saturation script and
documentation exist, how they connect back to the original BPA regime review,
and which earlier scripts they were derived from.

The short version is simple. The standalone script exists so that researchers
and developers do not have to scrape random pieces of a research regime that
were accumulated over time for different questions, machines, and experimental
passes. The BPA work produced real results, but it also produced the normal
mess of an active R\&D line: one-off runners, hardware-specific probes,
intermediate comparisons, and paper-facing orchestration. That history matters,
but it should not be the default interface for someone who wants to reproduce
this specific result.

The standalone script therefore serves a narrower purpose. It gives one clean
entrypoint for reproducing the memory-traffic saturation characterization
without asking a new reader to reconstruct the entire BPA workflow first. This
document preserves the lineage so the cleanup does not erase why the result was
measured this way in the first place.

For the main result, use:
- [Memory-Traffic Saturation in Autoregressive Decode](https://github.com/mcgrof/knlp/blob/main/docs/memory_traffic_saturation_in_autoregressive_decode.md)

For BPA background, use:
- [BPA overview](https://github.com/mcgrof/knlp/blob/main/docs/bpa.md)
- [RGSA, BPA, and fused KV quantization](https://github.com/mcgrof/knlp/blob/main/docs/paper/bpa/evolution.md)

## Why split this out at all

The memory-traffic saturation result is now useful outside the narrower BPA
context. It does three jobs at once. It establishes the decode bottleneck, it
provides the empirical basis for later fused quantization work, and it gives a
measurement framework for practical planning questions around bandwidth,
capacity, and tiering. Those are broad enough uses that the result needs a
clean home of its own.

At the same time, the lineage should not disappear. The result did not appear
from nowhere. BPA is what forced the workload to be framed in terms of
measured decode traffic instead of abstract attention structure. The standalone
document keeps the result clean; this lineage document keeps the history and
motivation visible.

## Why the standalone script exists

The standalone script exists to prevent a bad reproduction workflow.

Without it, a new reader or researcher is likely to do one of two bad things.
They either ignore the prior work and reinvent an incomplete benchmark, or they
start scraping older BPA scripts one by one, mixing hardware-specific runners,
private result conventions, and intermediate experiment scaffolding into an ad
hoc reproduction path. Both outcomes are bad. The first loses provenance. The
second creates a brittle reconstruction of the result from pieces that were not
originally designed to be a single public interface.

The standalone script fixes that by giving one stable entrypoint:
- `scripts/memory_traffic_saturation.py`

That script wraps the more detailed dataset framework underneath it and makes
this one result reproducible without pretending the whole BPA regime was always
organized as a single clean package.

## Why keep a separate lineage document

The standalone script is intentionally narrower than the full BPA history. That
is a feature, not a bug. It keeps reproduction simple. But once you do that,
you need somewhere to explain where the result came from, why those underlying
scripts exist, and how the work moved from BPA diagnosis to fused quantization.
That is what this document is for.

Keep the main memory-traffic document focused on the result itself. Keep this
lineage document focused on provenance, motivation, and script history.

## How the work evolved

The older path was not a single script. It was a research regime.

Early BPA work reframed the problem around measured decode traffic. That led to
H100 decode characterization, then to more explicit kernel latency work,
long-context checks, W7900 comparisons, A100/B200 cross-GPU confirmation, and
paper-facing orchestration that made the final empirical dataset easier to
collect consistently. The current standalone runner sits on top of that evolved
stack rather than replacing its historical importance.

That is also why the standalone result and the lineage document should both
exist. One is the clean interface. The other explains why the interface had to
be extracted from a broader, messier, but still valuable R\&D history.

## Current script structure

If you want the clean reproduction path for this specific result, start here:
- `scripts/memory_traffic_saturation.py`

That script wraps the current dataset framework:
- `scripts/paper/bpa_paper/run_dataset.py`
- `scripts/paper/bpa_paper/run_smoke.py`
- `scripts/paper/bpa_paper/run_matrix.py`
- `scripts/paper/bpa_paper/fit_scaling.py`
- `scripts/paper/bpa_paper/package_results.py`

Those scripts define the current clean interface for the memory-traffic
saturation dataset.

## Earlier script provenance

The standalone workflow was derived from a broader script lineage that remains
useful for provenance and for understanding how the result was assembled.

H100 decode characterization and follow-on kernel analysis came through scripts
such as:
- `scripts/v35_h100_bench.py`
- `scripts/bpa_h100_exp3_kernel_latency.py`
- `scripts/bpa_h100_exp2_long_context.py`

The W7900 line evolved through scripts such as:
- `scripts/bpa_v50_w7900.py`
- `scripts/bpa_v51_w7900.py`
- `scripts/bpa_v52_w7900.py`
- `scripts/bpa_v53_w7900.py`

Supporting comparison and summary helpers include:
- `scripts/check_a100_baseline.py`
- `scripts/plot_scaling_laws_b200.py`
- `scripts/generate_unified_comparison.py`
- `scripts/regenerate_unified_summary.py`

These earlier scripts should not be the default entrypoint for new users. They
exist as provenance and as evidence of how the regime evolved, not as the
simplest way to reproduce the final standalone result.

## Relationship to BPA

BPA remains the background that explains why this work happened. It asked the
right systems question: what decode traffic is actually paid, and what changes
that bill in practice? The memory-traffic saturation result is one concrete
answer to that question. Fused KV quantization is the strongest current
intervention that followed from it.

That ordering matters:
1. BPA reframed the problem around measured decode traffic.
2. Memory-traffic saturation characterized the regime empirically.
3. Fused KV quantization attacked the measured bottleneck directly.

Keep that chain visible, but do not force every reader of the main result to
walk through the entire chain before they can run the standalone script.
