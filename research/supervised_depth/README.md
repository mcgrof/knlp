# Supervised MNIST depth study

This entry point compares supervised digit learning across residual,
LayerRoPE, manifold-constrained Hyper-Connections (mHC), and combined
models. It trains ten logits on MNIST images and labels. It has separate
selection and final seeds, validation-only selection, and a fixed endpoint.

The LayerRoPE component generates existing block input LayerNorm gains.
It is a vision-style adaptation of Eq. 1 in arXiv:2610.09179, not the
full input/output RMSNorm Transformer construction. The new coded writes
use the same angular parameters as normalization, with ordered, shuffled,
and constant assignment controls. This routing intervention is a proposed
variant, not part of the published LayerRoPE method.

Use `make -f research/supervised_depth/Makefile check`, then `qualify`,
`probe`, and `plan`. Set PYTHON, ROOT, DATA and CHECKPOINTS explicitly.
Provide `KNLP_SUPERVISED_SOURCE_JSON` with a clean source commit and source
digest. Freeze the plan before selection with `FROZEN.json` containing
its `plan_sha256`. The runner refuses unqualified or over-budget stages.

`run-core` executes the four core arms before `run-channels` executes the
channel controls. `analyze` retains missing and censored runs and uses
paired seed intervals. Fixed accuracy and cross-entropy crossing costs
are distinct from endpoint quality and measured time. The GPU-host budget
ledger charges command wall time conservatively, including setup and
evaluation, under an eight-hour cap. Resume is deliberately unsupported.

Checkpoint payloads stay outside the results directory. Existing run and
checkpoint paths are never overwritten. Reference qualification uses a
converged FP64 matrix projection and its independently derived implicit
vector-Jacobian product, including competing sharp rows. Failed reference
settings do not consume deep training qualification time.

Training FLOPs are analytical estimates: multiply-add counts as two;
backward plus forward uses three times forward arithmetic. Elementwise
and transcendental operations count as one. The estimate omits optimizer,
evaluation, kernel launches, and memory traffic. These conventions must
accompany FLOP comparisons; measured time remains a separate outcome.
