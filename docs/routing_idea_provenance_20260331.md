# Routing idea provenance — 2026-03-31

This note explains where the current routing-prior-quality work came from.

## Provenance chain
The current routing work did **not** appear out of nowhere as a kernel micro-optimization project.
It grew out of earlier reciprocal-attention / FIM-guided analysis and later reflection on what signals are actually most useful for routing.

### Step 1 — FIM-guided reciprocal-attention and structural approximation work
Earlier work established the general idea that FIM-like signals are useful for **continuous approximation / restructuring** problems.
This shows up repeatedly in prior notes and session logs around:
- reciprocal attention
- pruning
- KVSplice
- the broader "FIM-guided model approximation" framing

Relevant session-log evidence exists in the main-agent session archive, for example:
- `~/.openclaw/agents/main/sessions/16d16ea8-41ca-410b-a2db-822f9fa7d829.jsonl`

That session includes a summarized note stating:
- FIM works well for continuous perturbation problems
- reciprocal attention belongs in that family
- quantization is a different regime

### Step 2 — Attention probability traces became the more compelling routing prior
While analyzing reciprocal attention and thinking about layer/head selection, the user revisited:
- attention probability traces across layers
- eigenvalues/eigenvectors for head structure
- how these might guide selective routing

The key conceptual shift was:
- **FIM traces** are valuable for understanding perturbation sensitivity / parameter-space structure
- **attention probability traces** provide more direct **functional priors** for routing, because they reflect what attention is actually doing

So the newer routing idea became:
- use attention probability traces as a stronger prior for where routing should focus
- use eigenvalues as a notion of how much mass/importance to allocate
- use eigenvectors as a directional / structural signal
- treat FIM as important historical scaffolding, but not necessarily the best direct routing prior

### Step 3 — Kernel work proved routing speed was viable
Once the fused Triton routed-attention kernel landed, the bottleneck stopped being Python overhead and became:
- prior quality
- operating-point quality/latency tradeoff
- serving integration path

That is why the current phase is no longer “can routing be fast?”
It is:
- “can practical priors make routing accurate enough to be worth integrating?”

## Current thesis
The current routing thesis is therefore:
1. FIM-guided approximation work provided the structural starting point
2. attention probability traces likely provide a **better functional prior** for routing than FIM alone
3. eigen-structure may help modulate allocation / importance across heads
4. practical routing value depends on converting those signals into real routing priors that survive downstream evaluation

## Relationship to paper writing
This provenance should be reflected in any future routing report written in the style of `~/devel/cartridges-engineering-guide/`, but placed under the routing-analysis line.

Important associated repo/path:
- routing analysis/report draft area: `~/devel/routing-analysis` (target location for future report work)
- initial draft/prototype repo to build up over time: `~/devel/paper-router`

## Important nuance for the follow-up thread
Do **not** present this as if FIM was "wrong" and attention traces replaced it completely.
The more accurate story is:
- FIM helped motivate structured approximation and selective intervention thinking
- attention probability traces look more promising as **functional routing priors**
- the current routing experiments are where that hypothesis starts getting tested seriously
