# Linear attention and bounded memory: DeltaNet, Gated DeltaNet, Trellis

This page is the standing summary of knlp's work on the linear-attention /
bounded-memory family of sequence mixers. A dense Transformer keeps a key-value
cache that grows with context; the models here replace it with a fixed-size
recurrent state that is constant in sequence length. We reimplemented three of
them from scratch in one matched harness — DeltaNet, Gated DeltaNet, and Trellis
— to ask whether Trellis's *nonlinear* memory writer beats its *linear* cousins
at matched size. The work is preliminary and still in progress, aimed at matching
the paper's original intent, so what follows is where the line stands today rather
than a settled verdict.

The short version, stated up front: at the scale we can afford to train (≈4.7M
non-embedding parameters), our Trellis *currently* loses to both DeltaNet and
Gated DeltaNet. That is a normal *below-crossover* result — worse than its linear
cousins at toy scale, which is where the Trellis paper says it should be, because
the paper only claims a win at 125M parameters and up. We have not trained at that
scale, and we are still working on it. A same-shell causal control we added on
2026-06-24 — the nonlinearity removed so the write reduces to the linear delta
rule — first looked like a *2×* win for linear (127.9 vs 248.4), but a ChatGPT-Pro
review flagged that as an over-read, and a six-step validation loop (2026-06-25)
confirmed it: the **2× was an under-specified-reimplementation artifact**. It is
*not* a meta-gradient bug (the exact backward is bit-exact for both writes), *not*
the stale-chunk gradient approximation (exact gradients don't close it), and *not*
the inner step size γ (sweeping it doesn't either) — it was the **missing paper
shell fidelity** (the final value-readout activation + output block), which once
restored narrows the gap to **~1.3–1.4×** (ln_silu+shell 182.8 vs identity 127.9),
robust across budgets. So the honest statement is: with fidelity restored, the
nonlinear write is **modestly behind** the linear delta rule at 5M, consistent with
below-crossover (the paper's nonlinear win is 125M+). One fidelity gap stays open —
the compression φ is a guess (the paper specifies `f` but not φ) — and the decisive
test is the gap-versus-scale ladder, not more 5M tuning (see
[Next: forward ablation ideas](#next-forward-ablation-ideas-to-evaluate)). The
rest of this doc lays out the architecture, our reconstruction and where it
departs from the paper, the kernel work that made the comparison tractable, the
numbers so far, and the forward ablations that remain to bring the reconstruction
in line with the paper.

The phase-by-phase lab log lives in
[reports/trellis_full_eval.md](../reports/trellis_full_eval.md); its §13–§14
verdict is older and superseded on the points here.

## Contents

- [Why bounded memory](#why-bounded-memory)
- [The lineage](#the-lineage)
- [What each model is](#what-each-model-is)
- [Our implementation, and where it departs from the paper](#our-implementation-and-where-it-departs-from-the-paper)
- [Kernels: why Trellis is the slow one](#kernels-why-trellis-is-the-slow-one)
- [What we trained](#what-we-trained)
- [What the Trellis paper claims](#what-the-trellis-paper-claims)
- [Next: forward ablation ideas to evaluate](#next-forward-ablation-ideas-to-evaluate)
- [Where it stands and what is next](#where-it-stands-and-what-is-next)
- [Reproduce](#reproduce)

## Why bounded memory

Autoregressive decode is memory-traffic bound: every generated token rereads the
entire KV cache, and that cache grows linearly with context. A model whose
inference state is a *fixed-size* recurrent memory breaks that growth — its
per-token decode cost and its memory footprint are constant in sequence length.
That is the structural prize the linear-attention family chases, and it is why
the line connects directly to knlp's decode and KV-compression work: a bounded
state is the most aggressive form of KV compression there is, because there is no
cache to compress.

The catch is quality. A growing cache stores every token exactly; a bounded
state must compress, and compression can lose the token you needed. The research
question for the whole family is whether a learned bounded memory can hold enough
of the right information to match — and at long context, beat — a dense
Transformer that pays the full cache cost.

## The lineage

The family descends from the **delta rule**
([Widrow & Hoff, 1960](https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf)):
an online least-squares update that nudges a weight matrix to reduce a per-step
prediction error. Schlag, Irie & Schmidhuber
([2021](https://arxiv.org/pdf/2102.11174)) recast linear attention as a
*fast-weight* memory written by exactly this kind of update; that delta-rule
model — later parallelized and scaled as
[**DeltaNet**](https://arxiv.org/abs/2406.06484) — is the linear baseline here.
Adding a learned forget gate to the fast-weight state gives
[**Gated DeltaNet**](https://arxiv.org/abs/2412.06464)
([podcast](https://podcast.do-not-panic.com/episodes/gated-delta-networks-for-long-context-retrieval/)).
A separate thread — test-time training, where the "memory" is a small model
optimized on the fly (TTT, Titans, Atlas) — converges on the same idea from the
optimization side.
[**Trellis**](https://podcast.do-not-panic.com/episodes/trellis-and-bounded-memory-transformer-kv-compression/)
(Karami, Behrouz, Kacham, Mirrokni; Google Research;
[arXiv:2512.23852](https://arxiv.org/abs/2512.23852)) sits at that junction: it is
a gated delta-rule fast-weight memory whose inner write is a *nonlinear* gradient
step rather than the linear one DeltaNet uses. (The 1960 delta-rule paper even
opens with a railroad-interlock switching analogy before arriving at the rule.)

The nonlinearity is the entire bet. It is what the paper credits for Trellis's
long-context win — and, as the kernel section shows, it is also what costs Trellis
the cheap exact training kernel that its linear cousins get for free.

## What each model is

All three keep a per-head recurrent state and read it write-before-read (the
state is updated with the current token, then queried).

**DeltaNet.** A fast-weight matrix `S` is written by the delta rule:
`S ← S + β_t (v_t − S k_t) k_tᵀ`, output `o_t = S q_t`. The inner step `v_t −
S k_t` is linear in `S`. Keys are L2-normalized so the implied `(I − β k kᵀ)`
factor stays a contraction.

**Gated DeltaNet.** The same, with a learned scalar decay: `S ← α_t S + β_t (v_t
− α_t S k_t) k_tᵀ`. The gate lets the memory forget, which matters for long
sequences. Still linear in `S`.

**Trellis.** A fixed `m`-slot memory `M` (we use `m = 64` slots of head
dimension 64). Each token performs one online gradient step on the inner
objective `½‖φ(M w_t) − α_t‖²`, whose gradient with respect to `z_t = M w_t` is
the error code `u_t = J_φ(z_t)ᵀ(φ(z_t) − α_t)`, giving the gated update `M ← β_t
M − γ outer(u_t, w_t)`. Here `φ` is a normalized SiLU, `β_t` a per-head forget
gate, `γ` a per-head step size, and `α_t` the learned write target. The model is
two-pass: a key pass (write keys, read queries) produces an intermediate `ŷ`, a
small map `f` turns it into a read vector, and a value pass (write values, read
`f(ŷ)`) produces the output. Because `φ` is nonlinear, `u_t` depends nonlinearly
on `M` — that is the single difference from DeltaNet, and everything downstream
follows from it.

## Our implementation, and where it departs from the paper

The code is in [`trellis_lm/`](https://github.com/mcgrof/knlp/tree/main/trellis_lm):
`trellis_memory.py` (the recurrence and the chunked operator), `trellis_mixer.py`
(the two-pass mixer), `trellis_triton.py` (the fused kernel), `linear_baselines*.py`
(the DeltaNet/Gated-DeltaNet baselines), and `model.py` (the LMs and a matched
dense control).

What is faithful: the bounded `m`-slot memory, the two-pass key-then-value
readout, the gated online-gradient write with a forget gate and a per-head step
size, the nonlinear inner objective and its exact VJP, and write-before-read
ordering. The baselines are the *reference* DeltaNet and Gated DeltaNet from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) —
the layers the fla authors ship, with the short convolution, qk-normalization,
and output gate that the published architectures carry — so the linear cousins
are not strawmen. Trellis uses two separate fast-weight states (a key memory and
a value memory) that share the write target `α`, which is the natural reading of
the paper's two-pass description.

Where it departs, and this is the part the forward work targets: several
paper-faithful choices are implemented as toggles but default *off*, and a few
details the paper leaves unspecified are our guesses.

- The paper applies a final activation to the value-pass readout (`y = φ(Mᵀr)`);
  our default leaves it off.
- The paper's output block is `PostNorm → GeLU gate → output projection`; our
  default is `output projection → ·SiLU(gate)`, a different order. These
  operations do not commute.
- The paper distinguishes `f` (the inter-pass map, defined in the method as an
  L2-normalized SiLU) from `φ` (the inner compression activation, never cleanly
  specified). We use one LayerNorm-SiLU for both. A change to `φ` alters the
  entire inner optimizer, not just one activation.
- The paper's Trellis write target is a plain linear projection `α = Wₐ x`, which
  **is** our default — so this is faithful, not a departure. (A ChatGPT-Pro review
  corrected an earlier note here: the `Softmax(Wₐ x)` target belongs to the paper's
  ABC discussion, not the Trellis definition. We have a softmax option but it is
  off-thesis for Trellis, and pairing it with `φ=ln_silu` explodes at init — a
  simplex-vs-zero-mean type mismatch — so it should be dropped from any Trellis
  conclusion.)
- The inner step size `γ` is fixed at `1e-2` and never swept. The paper states
  no initialization for `γ` or for the forget gate at all. Our objective is
  `½‖·‖²` to the paper's `‖·‖²`, so a paper step `γ` corresponds to roughly
  `2γ` in our convention — harmless if documented and swept, not if fixed once.

None of these is individually likely to explain the full gap to the linear
baselines. Together they are why the result below is honestly labeled "our
implementation," not "Trellis," and why the forward ablations matter.

## Kernels: why Trellis is the slow one

DeltaNet and Gated DeltaNet get an *exact* chunk-parallel training kernel as a
gift of linearity: each step is a fixed affine map of the state, affine maps
compose, so an entire chunk collapses into matmuls and a triangular solve.
Trellis cannot have this. Its nonlinear `φ` makes the within-chunk dependency
non-affine; the obvious causal fixed-point (Jacobi iteration) is nilpotent and
reaches the exact answer in `C − 1` passes for chunk size `C`, but that costs the
same `~T` inner steps as the sequential loop. We did not prove a lower bound — a
cleverer structured solver might exist — but there is no fixed-size, WY-style
associative representation analogous to the linear delta rule, so the cheap exact
kernel is forfeited. A single Newton/Gauss-Newton linearization restores a block
solve only approximately and at higher cost, because LayerNorm couples all slots.
The paper itself trains with a *stale*-gradient chunk approximation (freeze the
state used for the inner gradient at the chunk start), which is what our fast path
also does; the authors most likely ate the same approximation with a more mature
kernel rather than holding a secret exact form.

That said, most of Trellis's apparent slowness was not algorithmic — it was a
Python launch-bound loop. The true-stale state evolution is sequential over
chunks but trivial in FLOPs, and running it as `T/C` separate kernel launches was
≈40 ms of pure overhead at sequence length 2048. A fused Triton kernel
([`trellis_triton.py`](https://github.com/mcgrof/knlp/blob/main/trellis_lm/trellis_triton.py))
collapses the whole chunk loop into one persistent program per `(batch, head)`,
keeping the `[64, 64]` state tile resident, doing the two matmuls with `tl.dot`
and the LayerNorm-SiLU VJP rowwise in-register, with a hand-derived backward
kernel. On an A100 this is 26–44× on the forward and 6.4× on the full mixer
(120.7k → 769.8k tok/s), pulling Trellis from 0.04× to 0.26× of dense throughput
and shrinking the gap to fla Gated DeltaNet from ~25× to ~4×. The same Triton
kernel runs unmodified on the W7900 (ROCm) for an 8–12× win — no separate HIP
port. The closed-form VJP matches autograd to ~1e-7; the full kernel matches the
PyTorch path to the fp32 floor (forward `<1e-4`, backward `<1e-3`).

Two caveats keep this honest. The Trellis state is forced to fp32 (bf16 is a
documented dead end for the recurrence), and the segmented decay product
`cumprod(β)` underflows fp32 beyond chunk size ≈32. And the ~4× residual gap to
the linear baselines is the real algorithmic floor: even fused, Trellis has no
parallel scan. At *decode*, however, the bounded fixed-size state is an advantage,
not a tax — there is no growing cache to reread. The throughput numbers above are
from our own kernel benchmark on the same hardware.

## What we trained

The decisive comparison is a single A100 run done under one clean protocol: C4,
`d256/L4`, sequence length 2048, 20 tokens per parameter (≈352M tokens), fp32
master weights with bf16 autocast,
AdamW, the fla *reference* DeltaNet/Gated-DeltaNet baselines, and the Trellis
fused kernel — all in the same pod.

| model | val PPL | non-embed params | bounded state? |
|---|---:|---:|---|
| Gated DeltaNet (fla reference) | **78.49** | 4.48M | yes (constant in T) |
| DeltaNet (fla reference) | 89.77 | 4.21M | yes |
| Trellis | **152.54** | 4.74M | yes |
| dense Transformer | 224.89 | 4.72M | no (KV grows with T) |

Trellis at 152.54 sits clearly between the dense baseline and the linear family —
a normal *below-crossover* result, not near-dense. But it still loses to plain
DeltaNet and to Gated DeltaNet, at a small *parameter disadvantage to the
baselines* (Trellis carries ~5–12% more non-embedding parameters and still loses,
which makes the loss conservative). In nats at matched tokens, Trellis is +0.53
nat behind DeltaNet and +0.66 behind Gated DeltaNet.

A supporting result. On PG19, a real long-range corpus, scratch Trellis beats a
matched *dense* Transformer by 6–18% with the margin widening as context grows —
the bounded-memory-at-length effect is real — but it still loses to Gated
DeltaNet. (That PG19 family comparison used the under-featured hand-rolled
baselines, so read the Trellis-beats-dense *direction* as solid and the family
*ordering* there as superseded by the fair A100 run above.) Separately, our fla
DeltaNet at the 125M *architecture* trains normally and keeps descending; that
shows the stack is healthy, not that we have reproduced the paper's absolute loss
(perplexity is not comparable across tokenizers, and the paper's is unstated).

## What the Trellis paper claims

Verified against the arXiv HTML (arXiv:2512.23852):

- **790M params / 30B tokens (C4), Table 1.** Trellis **20.28** (best) <
  Gated DeltaNet 21.40 < Transformer++ 25.89 < TTT 27.05 < Mamba2 28.91.
- **125M params (The Pile), Table 2 ablation.** Trellis **10.87** < TTT 11.44 <
  DeltaNet 11.58. There is no Gated DeltaNet row in *this ablation table* — but a
  ChatGPT-Pro review reports Gated DeltaNet **does** appear at 125M elsewhere
  (reportedly Table 4 at 11.31, vs Trellis 10.87 / DeltaNet 11.58), so the earlier
  "Gated DeltaNet only at 790M" reading was wrong and needs reconciling against the
  arXiv HTML. Replacing the
  nonlinear `φ` with the identity (reducing Trellis to the delta rule) gives
  11.65 — so the nonlinear write is worth ~0.78 ppl here. Removing the forget
  gate gives 11.28; non-chunked (`b=1`) gives 10.75, slightly *better* than the
  chunked 10.87.
- **1B params, RULER / S-NIAH recall, Table 3.** Trellis 79.8 > Gated DeltaNet
  75.8 > TTT 66.1 > DeltaNet 57.9 > Mamba2 52.0, with the edge largest at long
  context.

The smallest published configuration is 125M / 2.4B tokens. The paper specifies
no initialization or schedule for the inner step size `γ` or the forget gate, and
leaves the tokenizer, sequence length, optimizer settings, and the exact baseline
specs unstated — the same under-specification that an independent reimplementation
of the group's earlier Titans model ("Titans Revisited", arXiv:2510.09551) blamed
for being unable to match the reported numbers. No official Trellis code exists,
and it is not yet in flash-linear-attention.

## Next: forward ablation ideas to evaluate

The reconstruction is faithful in its bones but not yet in every detail, and the
honest way to close the gap to the paper is to turn its specified choices back on
and sweep the knobs it leaves unstated, before paying for anything at scale. The
ablations to run, roughly in order of expected leverage:

- Turn the paper-faithful defaults on: the final value-readout activation
  `y = φ(Mᵀr)`, the `PostNorm → GeLU → projection` output block, and the softmax
  write target `α = Softmax(Wₐ x)`.
- Resolve the `f` / `φ` distinction. The paper defines `f` as an L2-normalized
  SiLU in the method but calls the baseline `f` "LN-SiLU" in the ablation, and it
  never cleanly specifies `φ`. Screen `φ` ∈ {LN-SiLU, L2-SiLU, SiLU, identity},
  then screen `f` under the winning `φ`.
- Sweep the inner step size `γ` ∈ {3e-4 … 1e-1} (per-head learned, and lightly
  input-conditioned), accounting for the ½-vs-1 objective convention. DeltaNet
  has no such knob, so an unswept `γ` is exactly where a Trellis-vs-DeltaNet
  comparison is most exposed.

The single most informative experiment is the **same-shell `φ = identity`
control**: the same Trellis mixer, projections, two-pass shell, and parameter
budget, with the nonlinearity removed so the model reduces to the delta rule.
That isolates "does the nonlinear write help?" cleanly — external DeltaNet differs
in shell, normalization, gating, and convolution and so is a strong *practical*
baseline but not the clean causal control. The paper runs exactly that ablation
(11.65 vs 10.87).

**We added the identity-`φ` option and ran it (2026-06-24): a gated linear-delta
control strongly outperforms our current LN-SiLU reimplementation.** At matched
LR, matched tokens, identical shell (d256/L4, C4, val PPL), the linear delta rule
reaches **127.9** vs the nonlinear ln_silu write's **248.4** at 120M tokens, and
**150.4 ± 0.8 vs 306.4 ± 3.3** across 3 seeds at 80M — a robust ~2× gap with the
forget gate held constant. Two supporting results: turning the paper-faithful
defaults on cuts ln_silu 260.6 → 182.8 (a real free lever, but still above
linear), and identity only trains at LR 3e-4 (it diverges at 1e-3 / 3e-3 —
`φ=identity` drops the LayerNorm that bounds `u = Mw − α`, so the state blows up).

**But a ChatGPT-Pro review and our own meta-gradient check say this is not yet
"linear beats Trellis" — it is "a gated linear control beats our *current
under-specified* reimplementation."** The exact-inner backward is bit-exact
correct for both φ (no Hessian-dropping bug), but the **stale-chunk gradient
approximation our training used (chunk 16) is ~100% wrong for the nonlinear write
vs ~10% for the linear one** — an asymmetric handicap that alone could inflate the
gap. On top of that the nonlinear arm carries reconstruction gaps that are *no-ops
under φ=identity*: the compression φ is undefined in the paper (we guessed
LN-SiLU for both f and φ), the final value-readout activation `y=φ(Mᵀr)` is off by
default, the output-block order differs, and γ is fixed/un-swept. The honest
read: the *direction* may survive a faithful, separately-tuned rerun (Pro puts
~25–30% on the full 2× surviving), but the magnitude is probably an artifact. The
validation sequence — meta-gradient check (done: no bug), exact/chunk-1 rerun,
one-at-a-time shell ablation, per-φ LR/γ tuning, faithful DeltaNet/GDN with
key-norm, then the gap-versus-scale ladder — is what resolves it. Full writeup,
caveats, and artifacts: `knlp-key-results/trellis-fidelity-20260624/`.

On scale: we are 26.6× below the paper's parameter floor but only 6.8× below its
token floor, and at ≈75 tokens per non-embedding parameter we are if anything
*over*-trained — so "Trellis just needs more tokens" is weak. "Trellis needs more
outer-model capacity before its extra flexibility pays off" remains plausible, but
it is a hypothesis: the paper shows nothing below 125M, so there is no evidence of
a crossover from losing at 5M to winning at 125M. A gap-versus-scale ladder is the
way to find out without committing to the full reproduction up front.

## Where it stands and what is next

The bounded-memory family beats a matched dense Transformer at long context —
that part is solid and is the result worth keeping. Among the family, Gated
DeltaNet is the standout at the scale we can train: simpler, lower-variance,
exact-chunkable, faster, and better quality. Our Trellis is the weakest member,
but a normal below-crossover result, not a broken one.

The gate before any 125M scaling is the fidelity-closure sweep above, run at 5M
parameters, with a pre-registered bar:

1. Turn the paper-faithful defaults on and add the identity-`φ` control.
2. Screen `φ`, then `γ`, then chunk size, sequentially rather than as a full grid.
3. Confirm the best two configurations and the identity-`φ` control over three
   seeds, keeping the fla Gated DeltaNet baseline.

Fund the 125M run only if Trellis beats its own identity-`φ` control by ≥0.02 nat
across seeds, the stale-chunk penalty is ≤0.05 nat at the chosen chunk, and the
gap to Gated DeltaNet shrinks below 0.25 nat. A gap-versus-scale ladder
(5M → 20M → 50M → 125M at ~20 tokens per parameter) decides whether the gap is
closing toward a crossover before paying for the full reproduction. In parallel,
the single highest-leverage action is non-compute: ask the authors for `φ`, `f`,
the `γ` and `β` parameterizations, the chunk size, the one-vs-two-state question,
and the tokenizer — any of which could save more than another kernel.

## Reproduce

```bash
make defconfig-trellis-tiny
make trellis          # train + eval the tiny Trellis vs dense
make trellis-bench    # kernel + memory-footprint benchmark
```

Code: [`trellis_lm/`](https://github.com/mcgrof/knlp/tree/main/trellis_lm).
Phase-by-phase lab log:
[reports/trellis_full_eval.md](../reports/trellis_full_eval.md) (older verdict,
superseded here).
Interactive version: [linear_attention_visualization.html](linear_attention_visualization.html).
