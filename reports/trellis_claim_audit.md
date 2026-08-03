# Trellis public-claim audit

Scope: the published linear-attention / Trellis surfaces — `docs/linear_attention.html`
(canonical; `linear_attention_visualization.html` is a symlink to it),
`docs/linear-attention.md` (long-form), `docs/index.html` (home card). Triggered by a
2026-07-20 external review. Every claim below was checked against the code that produced
the numbers, the archived run artifacts, and the arXiv source (2512.23852). The goal is one
evidence-scoped account: what was proved, implemented, measured, inferred, or merely chosen.

## Summary

The canonical HTML overstated three things that our own artifacts (and, for two of them,
our own `linear-attention.md`) do not support. The top-line verdict — Gated DeltaNet is the
matched-state quality leader and Trellis's nonlinear write does not pay for itself at
reachable scale — is unaffected: it rests on the clean param-matched A100 run
(Trellis 152.54 vs GDN 78.49), not on the disputed claims.

## Ledger

| # | Public claim | Category | Status | Evidence / correction |
|---|---|---|---|---|
| 1 | The 127.9-vs-248.4 (~2×) gap "changes only the write" / isolates the nonlinear write / is "not a reconstruction artifact" | measured empirical + causal inference | **contradicted (confounded)** | Runs came from commit `2c210da0` via `scripts/trellis_phi_sweep.py`; the identity arm passes only `activation="identity"`, and `trellis_mixer.py` ties **both** `self.phi` (write) and `self.f` (inter-pass map) to `cfg.activation` (lines 156-157). φ and f were never separate. So `identity` linearized both — a joint φ/f control, not write-only. Second confound (already noted in our MD): training used stale-chunk-16, ~100% wrong for the nonlinear arm vs ~10% for the linear one. The paper's clean φ-only control (f held at LN-SiLU) is 11.65 vs 10.87 = **0.78 ppl**, not 2×. Keep: the direction (linear ≥ nonlinear at this scale) matches the fair A100 run. Retire: the clean write-only magnitude, pending a rerun holding f=LN-SiLU fixed. |
| 2 | "Trellis and Gated DeltaNet are compared head-to-head only at 790M" and "the 125M ablation has no Gated DeltaNet row" | paper-reported fact | **contradicted** | Paper (pdftotext + arXiv HTML, cross-confirmed): GDN vs Trellis appears at 125M (Table 4, Pile-2k: Trellis 10.87 vs **GDN 11.31**), 790M (Table 1: 20.28 vs 21.40), and 1B (Table 3, RULER). Table 2 (the 125M *component ablation*) genuinely has no GDN row — the page conflated Table 2 with Table 4. Our own MD had already half-retracted this. |
| 3 | "forfeits the exact kernel — a theorem" / "exactly what Trellis cannot have" / "the theorem rules out" / "genuine no-parallel-scan floor" | mathematical theorem (prose around it) | **supported but overbroad** | The theorem *box* is correct and careful: sufficient-condition only, converse explicitly disclaimed, and it already retired the "nonlinear ⇒ no exact chunk" overclaim. The surrounding headline/caption prose reads as a proven impossibility, which the sufficient-only theorem does not license. Correct wording: the standard exact WY/UT transform does not apply; no compact exact transform is known for this nonlinear write; the ~4× gap is measured for these implementations, not a proven lower bound. |
| 4 | "~4× residual is the genuine no-parallel-scan floor" | measured empirical (mislabeled as bound) | **supported but overbroad** | ~4× is a measurement for our implementations (two passes, nonlinear VJP, fp32 recurrent state), not a mathematical floor. |
| 5 | GatedDeltaProduct = "the other nonlinear bet" | implementation fact | **contradicted (mislabel)** | Its multi-Householder transition is more expressive but still state-linear/affine. Call it "a more expressive affine transition." |
| 6 | "We reimplemented the family from scratch — DeltaNet, Gated DeltaNet, GatedDeltaProduct, and Trellis" | implementation fact | **supported but overbroad** | The DeltaNet/GDN/DeltaProduct baselines are the FLA reference layers (page itself says so). We implemented Trellis and its controls in a matched harness and compared against FLA reference layers. |
| 7 | PG19 "Bounded memory beats dense at length" (universal); "solid regardless of the cousin comparison" | measured empirical (over-generalized) | **supported but overbroad** | Evidence is Trellis-vs-dense at 512/1024/2048/4096; deltas are not monotonic (largest at 4096). Scope to "in our tested PG19 configs, Trellis beat matched dense at every tested length," and keep the MD caveat that Trellis still trails GDN. |
| 8 | Decode: bounded state is "an advantage, not a tax" | inference | **supported but overbroad** | O(1) in context length is true, but the state is a dense, hot, read-modify-write object per token. Not automatically smaller/faster than modern GQA KV at every context; crossover depends on state size, precision, batch, hardware. |
| 9 | "an independent fairness audit put the odds it reproduces at about 0.30" | project-planning estimate | **supported but mis-placed** | Fine as a labeled project-planning estimate; label it as such, not as a scientific result. |
| 10 | "Line closed" | project-management decision | **supported (keep, separate from science)** | Valid as a spend decision. Must not read as "Trellis disproved." Already separated in the footer; keep that separation everywhere. |
| 11 | HTML "closed / verdict in / not an artifact" vs MD + index "preliminary / still in progress / the 2× was an artifact" | consistency | **contradicted (cross-file)** | Reconcile to one claim set: line closed as a spend decision, verdict scoped, the ~2× magnitude confounded (attribution open), paper scale claims parked not disproven. |

## The one open experiment (needs GPU, not run here)

A clean write-only ablation: φ = identity with the inter-pass map **f held at LN-SiLU**, all
else fixed, ideally without the stale-chunk-16 asymmetry (chunk=1 exact or a fidelity-matched
chunk). Until then, the write's isolated cost is bounded above by the paper's own 0.78 ppl
(11.65 vs 10.87) and is not the ~2× the tied-φ/f control reported. Requires the config split
(phi_activation / f_activation / value_readout_activation) that does not yet exist.

## What is untouched and solid

The fair param-matched A100 run (Trellis 152.54 vs GDN 78.49 vs DeltaNet 89.77 vs dense
224.89), the structural point that Trellis loses the *standard* WY/UT transform, and the
fused Triton kernel and its throughput numbers all stand as stated.

## Addendum 2026-07-21: GatedDeltaProduct-2 LM comparison scoped

An external review (arXiv:2502.10297, DeltaProduct) prompted a fairness/faithfulness
check of the "GatedDeltaProduct wins the binding grid but loses to GDN on C4 loss" claim
(GDN 215.72 vs GatedDeltaProduct-2 224.39, 3 seeds).

| # | Public claim | Status | Evidence / correction |
|---|---|---|---|
| 12 | "GatedDeltaProduct's extra-overwrite rotations do not carry to LM loss" (architecture-wide) | **supported but overbroad** | Faithfulness confirmed: the arm is the fla reference `GatedDeltaProduct`, nh2, `allow_neg_eigval=True`, `expand_v=1.0` (matches the paper repo), and — verified — `use_forget_gate=True` via the FLA v0.4.2 layer default (we build the layer directly, not the HF-config path that defaults it off), so it is the paper's exact Gated DeltaProduct-2[-1,1]. But our run is a d512/L10, 20M-token pilot (0.43 tokens/param) vs the paper's 340-393M / 35B tokens (~89-103 tokens/param). The comparison is width-matched, **not** parameter- or FLOP-matched: DeltaProduct-2 carries 49.9M params vs GDN 44.7M (+12%) and ~2x the recurrence work (two Householder micro-steps/token), so it got *more* compute and still lost by ~0.039 nats/token (~4%). Scoped to an early-training pilot result; report nats/relative, not raw PPL. |
| 13 | Implicit "the paper's DeltaProduct>GDN LM claim is the strong one" | **corrected** | The paper's headline gated LM pair is **neither parameter- nor FLOP-matched** (393M GatedDeltaProduct-2 vs 340M GDN, same 8 heads, ~2x recurrence FLOPs). Its only parameter-matched control (ungated, DeltaNet-12h 392M vs DeltaProduct-2 392M) shrinks the win to **0.14 ppl**; no iso-FLOP comparison exists. So the paper's LM edge (~0.85 ppl / ~0.033 nats at 340M/35B) is real but thin and compute-favored. Our pilot result neither tests nor refutes it. |

Two of my own review claims were retracted after the external red-team: the `[0,1]` vs
`[-1,1]` eigenvalue-range asymmetry is a confound of **unknown** LM sign (not "generous to
DeltaProduct"), and "scale caused the reversal" is an unverified story, not a finding — the
honest statement is "at this pilot scale GDN wins; the paper's claim is untested." The
binding-grid win is directional evidence of synthetic overwrite behaviour, not a
replication of the paper's permutation-group state-tracking. Open items if the line reopens:
a GDN[-1,1] arm (to match the paper's config) and disclosure of the optimizer-step / warmup
schedule (a 512-step warmup at 20M tokens can dominate).

## Addendum 2026-08-02: gradient-mode defect and the bounded reopening

A code audit found that `TrellisConfig.exact_inner` — the flag selecting the
paper's full bilevel training objective — was read only by the sequential
dispatch branch. Every chunked path (the PyTorch state evolution and the fused
Triton kernel) detached z unconditionally, and `chunk_refine` could repair only
the forward. Consequence: every chunked nonlinear-Trellis run in this program
trained first-order with a chunk-start-stale forward, whatever the config
stored, and the sequential "seq" cells driven by `trellis_firmup.py` were
exact-forward but also first-order (`exact_inner=False`). No production-scale
nonlinear LM run ever trained the bilevel objective.

| # | Public claim | Status | Evidence / correction |
|---|---|---|---|
| 14 | The fair param-matched A100 run is "untouched and solid" as stated | **supported but underspecified** | The numbers stand for what was run, but the Trellis arm trained first-order + chunk-start-stale (the fused kernel is first-order by construction) against fla baselines with exact gradients for their linear recurrences. The doc treats the stale approximation as paper-faithful; the paper's outer-gradient treatment through its chunk approximation is not determinable from its text, and its B=1 sequential ablation is slightly BETTER than its chunked baseline, so the axis was live. Relabel: "GDN is the matched-state quality leader over the FIRST-ORDER Trellis reconstruction"; the bilevel variant is untested at this scale. |
| 15 | Row 1's residual "~1.3-1.4x under exact gradients" (the chunk-1 rerun) | **supported but underpowered and still confounded** | That rerun (5M tokens, seq256) really was bilevel — the only bilevel LM datapoints in the archive besides the toy phases — but phi and f were still tied, so it bounds the joint phi/f effect, not the write-only cost. The clean write-only rerun in row 1's "open experiment" remains unrun for LM. |

The code now names both axes explicitly (`trellis_state_mode`,
`trellis_outer_gradient_mode`), fails loudly on unservable requests, resolves
legacy configs to what actually executed, and carries an oracle-checked
full-bilevel chunk-start reference plus a float64 gradient test matrix
(`trellis_lm/tests/test_trellis_semantics.py`). The litmus test that froze the
closure leaderboard as assertions was removed with the same change: scientific
conclusions are records for reports, not API invariants — especially when the
frozen numbers were first-order-trained. The line is reopened for a bounded
correctness audit: prior negatives remain valid for the implementations
tested, and the toy-scale bilevel evidence on file (the recall-toy pivot,
the chunk-1 rerun, the write-only recall probe) leans against a bilevel
rescue rather than for it. The audit ends with labeled full-bilevel versus
first-order comparisons; it does not presume their outcome.
