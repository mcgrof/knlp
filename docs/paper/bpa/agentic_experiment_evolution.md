# Agentic Experiment Evolution for BPA

BPA did not emerge from a single idea. It emerged from an experimentation
style that evolved in phases, and the operational lineage matters as much
as the technical one.

This is the abstracted version -- it preserves the methodology and lineage
while leaving out private infrastructure details.

## Roots

The deepest roots are systems-engineering roots, not AI-agent roots.

**Strict grammar.** The early discipline came from kernel-style development:
explicit configuration grammars, constrained option surfaces, stable naming,
small changes that can be reasoned about. Experiments are parameterized and
inspectable, not vague prose requests.

**Reproducibility.** An experiment is not real because it ran once. It is real
when the inputs are identifiable, the execution path is repeatable, the
outputs land in a durable result sink, and the result survives later review.

**Hard testing.** Smoke-test before scale. Separate correctness from
performance. Distrust one-off wins. Assume fragile setups fail at the edges
first. This matters especially for decode and quantization work where a result
can look exciting while actually being measurement drift, kernel noise, or
proxy-metric gaming.

## Operational phases

**Phase A: manual but disciplined.** Direct manual experiment steering. Slower,
but it forced the important habits: write down the question, define the metric,
keep the surface area bounded, record what changed.

**Phase B: harness-driven.** Once the questions got larger, harnesses took over
the mechanical work -- matrix generation, config pinning, benchmark replay,
artifact naming, result collation. Automation reduced clerical drift without
removing human judgment.

**Phase C: agent-assisted.** Agentic orchestration layered on top of that
foundation. Agents propose bounded experiment batches, draft manifests, check
durable state, summarize deltas, rank promising branches, and manage
topic-separated experiment lanes. The loop remains constrained. Free agent
autonomy is not automatically good.

## Technical lineage

The BPA line itself is the best example of this method in action:

1. **RGSA** asked whether routing and selective access could avoid paying dense
   attention cost everywhere.
2. **BPA** reframed the problem around the measurable decode bottleneck:
   repeated KV-cache traffic under memory-bandwidth constraints.
3. **Fused KV quantization** turned that diagnosis into a concrete kernel-level
   win by reducing real decode traffic, not just tensor size on paper.

The pattern: architectural instinct, then systems diagnosis, then
implementation that cashes out the diagnosis. Guided phase evolution, not
random search.

## Public story boundaries

Leave out private hostnames, machine inventory, control-channel topology,
internal operational chatter, ephemeral pod identifiers, credentials, and
result claims without durable artifacts.

Keep the kernel-style roots, the reproducibility doctrine, the testing-first
mentality, the manual-to-automated-to-agentic evolution, the
RGSA-to-BPA-to-fused-KV lineage, and the rule that a result does not count
until it lands durably.

## BPA Loop v0

The next step is not an unconstrained autoresearch swarm. It is a bounded
loop:

1. **Propose** a small number of bounded experiments, each tied to a parent
   finding.
2. **Smoke-test** with cheap probes. Reject broken or noisy ideas early.
3. **Gate** on correctness, stability, and budget before promotion.
4. **Land** results in the durable sink. No artifact, no result.
5. **Rank** what improved, what failed, and what deserves the next budget.

Every iteration declares: parent finding, hypothesis, metric, budget cap,
artifact destination, promotion condition, kill condition. This preserves
lineage and makes later review possible.

## H2 2026 review direction

A strong use of BPA Loop v0 is retrospective re-analysis -- not only
inventing new experiments but reviewing prior routing and KV ideas:

- which experiments were underpowered,
- which claims were overfit to a narrow setup,
- which phases failed to connect,
- which newer routing papers (MoBA, cartridge-adjacent directions) suggest a
  better framing.

The core question: what experiment did we miss, mis-specify, or fail to
connect across phases?

## Relation to the Karpathy loop

The Karpathy-style loop is a useful reference, but BPA needs a stricter
version. Multiple metrics matter, experiments are more hardware-sensitive,
false positives are more expensive, and durable provenance matters more than
local optimization wins. BPA Loop v0 keeps the iterative hypothesis-testing
cycle but adds stronger gates for reproducibility, cost, promotion, and
durability.
