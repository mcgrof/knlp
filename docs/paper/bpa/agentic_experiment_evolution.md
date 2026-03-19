# Agentic Experiment Evolution for BPA

Use this document to explain how the BPA line evolved operationally, not only
technically.

This is the abstracted, security-conscious version. It keeps the methodology
and leaves out private infrastructure details, internal chat routing, exact
runtime topology, and anything that would overfit the story to one private
setup.

## Why this matters

BPA did not emerge from a single isolated idea. It emerged from an
experimentation style that evolved in phases:

1. **kernel-style discipline**
2. **reproducible experiment harnesses**
3. **incremental automation**
4. **agent-assisted orchestration**
5. **topic-specific experiment lanes with durable result sinks**

The technical lineage and the operational lineage matter together.

## Methodology roots

The deepest roots are not "AI agent" roots. They are systems-engineering roots.

### 1. Strict grammar

The early discipline came from kernel-style development:

- explicit configuration grammars,
- constrained option surfaces,
- stable naming,
- and small changes that can be reasoned about.

In practice this means experiments should not be vague prose requests. They
should be parameterized and inspectable.

### 2. Reproducibility first

An experiment is not real because it ran once. It is real when:

- the inputs are identifiable,
- the execution path is repeatable,
- the outputs land in a durable result sink,
- and the result can survive later review.

### 3. Hard testing discipline

The methodology also inherits a strong testing instinct:

- smoke-test before scale,
- separate correctness from performance,
- distrust one-off wins,
- and assume fragile setups will fail at the edges first.

This matters especially for decode, KV, routing, and quantization work where a
result can look exciting while actually being measurement drift, kernel noise,
or proxy-metric gaming.

## Operational evolution

That foundation evolved gradually.

### Phase A: manual but disciplined iteration

The earliest phase relied on direct manual experiment steering. This was slower,
but it forced the important habits:

- write down the question,
- define the metric,
- keep the surface area bounded,
- and record what changed.

### Phase B: harness-driven automation

Once the questions became larger, harnesses took over more of the mechanical
work:

- matrix generation,
- config pinning,
- benchmark replay,
- artifact naming,
- and result collation.

The key lesson was that automation should reduce clerical drift, not remove
human judgment.

### Phase C: agent-assisted orchestration

The current phase adds agentic orchestration on top of that foundation.

Agents help with:

- proposing bounded experiment batches,
- drafting manifests,
- checking durable state,
- summarizing deltas,
- ranking promising branches,
- and managing topic-separated experiment lanes.

But the loop remains constrained. The methodology does **not** assume that free
agent autonomy is automatically good.

## Technical phase evolution

The BPA line itself is a good example of this method.

### RGSA -> BPA -> fused KV quantization

The evolution can be summarized as:

1. **RGSA** asked whether routing / selective access could avoid paying dense,
   uniform attention cost everywhere.
2. **BPA** reframed the problem around the measurable decode bottleneck:
   repeated KV-cache traffic under memory-bandwidth constraints.
3. **Fused KV quantization** turned that diagnosis into a concrete kernel-level
   win by reducing real decode traffic instead of only reducing tensor size on
   paper.

This is the important pattern:

- architectural instinct,
- then systems diagnosis,
- then implementation that cashes out the diagnosis.

That is not random search. It is guided phase evolution.

## What to leave out of the public story

For security and long-term maintainability, leave out:

- private hostnames and machine inventory,
- exact control-channel topology,
- internal operational chatter,
- ephemeral pod identifiers,
- credentials or provider-specific internal procedures,
- and any result claims that do not yet have durable artifacts.

The public story should preserve **method**, **lineage**, and **validated
claims**, not private control-plane detail.

## What to keep in the public story

Keep these elements:

- the kernel-style roots,
- the reproducibility doctrine,
- the testing-first mentality,
- the evolution from manual to automated to agent-assisted execution,
- the RGSA -> BPA -> fused-KV lineage,
- and the rule that a result does not count until it lands durably.

These are stable ideas that explain the project without oversharing.

## BPA Loop v0

The next step is not an unconstrained autoresearch swarm. The next step is a
**bounded BPA Loop v0**.

### Goal

Scale useful parts of the existing workflow without losing rigor.

### Loop shape

1. **Hypothesis proposal**
   - propose a small number of bounded next experiments
   - tie each proposal to a parent finding

2. **Smoke / mini-matrix validation**
   - run cheap probes first
   - reject broken or noisy ideas early

3. **Promotion gate**
   - only promote candidates that pass correctness, stability, and budget gates

4. **Durable landing**
   - results only count when artifacts land in the durable result sink

5. **Ranking and next-step proposal**
   - summarize what improved,
   - what failed,
   - and what deserves the next hardware or context budget

### Required per-run metadata

Every loop iteration should declare:

- parent finding,
- hypothesis,
- metric,
- budget cap,
- artifact destination,
- promotion condition,
- and kill condition.

This preserves lineage and makes later review possible.

## H2 2026 review direction

A strong use of BPA Loop v0 is retrospective re-analysis.

The point is not only to invent new experiments. It is also to review prior
routing and KV ideas and ask:

- which experiments were underpowered,
- which claims were overfit to a narrow setup,
- which phases failed to connect to each other,
- and which newer routing papers suggest a better framing.

That review should include:

- prior routing strategies,
- RGSA-era ideas,
- BPA-era bandwidth and protection results,
- fused quantization results,
- and newer routing / selective-attention work such as MoBA- and
  cartridge-adjacent directions.

The core question is simple:

> what experiment did we miss, mis-specify, or fail to connect across phases?

## Relation to the Karpathy loop

The Karpathy-style loop is a useful reference point, but BPA needs a stricter
version.

The BPA loop differs in a few important ways:

- multiple metrics matter,
- experiments are more hardware-sensitive,
- false positives are more expensive,
- and durable provenance matters more than local optimization wins.

So the right adaptation is:

- keep the iterative search,
- keep the hypothesis-testing cycle,
- but add stronger gates for reproducibility, cost, promotion, and durability.

That is the intended direction for BPA Loop v0.
