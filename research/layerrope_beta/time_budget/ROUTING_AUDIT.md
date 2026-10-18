# Finite routing normalization

This route tests whether observed hyper-connection row and column errors
come from the finite iteration budget or arithmetic precision. It preserves
the executed 20-repeat operator, softmax, epsilon placement and final column
normalization as the control. A higher-repeat candidate is a changed finite
operator. Passing a stronger engineering bound does not establish an
upstream bug or an exact projection.

The [mHC paper](https://arxiv.org/html/2512.24880v1#S4.SS2) uses 20 iterations
as a practical approximation and explicitly discusses residual deviations.
The vendored [reference operator](../../tinystories_depth/upstream/tile_kernels/tile_kernels/torch/mhc.py)
is pinned separately. Paper-level ideal constraints and a finite numerical
recipe are distinct contracts.

## Reproduce the diagnostics

Use the [reproduction dependencies](../../reproduction/README.md). The CPU
smoke route checks upstream finite-function fidelity, exact capture parity,
fixed sample selection, same-recipe gradients and finite differences. It
also runs a synthetic training check with common-update checkpoint hooks.

```sh
make defconfig-layerrope-mhc-routing-audit-smoke
make CONFIG_RESIDUAL_OUTPUT_DIR=/path/to/audit-smoke \
     CONFIG_RESIDUAL_CPU_PYTHON=/path/to/python
```

The full audit uses an existing GPU allocation and external checkpoint/data
receipts. It never rents compute or downloads private artifacts. Keep the
original executed source and state receipts before diagnosing a saved run.
Generated tensors, JSON and checkpoints belong outside this checkout.

Prepare an external input card containing:

- `source`: the current `time_budget.train.source_identity()` dictionary.
- `executed_forward_source`: relative forward-path filenames and their
  hashes from the original executed source receipt. The current files must
  match those bytes; do not silently substitute newer forward code.
- `validation`: a file `path` and `sha256` receipt for the original 32,769-token
  validation array. The diagnostic shape is 32 windows of 256 tokens.
- `diagnostic_seed`: a fixed seed for cotangents and directions.
- `checkpoints`: six receipts for M and ML at targets 0, 200 and 787 seconds.
  Each names its `path`, file `sha256`, `model_sha256`, `identity_sha256`,
  `row`, `seed`, `step`, actual `training_seconds` and `target_seconds`.
  Identity hashes use `time_budget.state.json_hash()` on the saved identity.
  Include `validation` with the original same-shape `ce` and `window_ce`
  values. Replay requires identical coverage and absolute differences at
  most 1e-6, with TF32 disabled as in the recorded training recipe.

```sh
make defconfig-layerrope-mhc-routing-audit
make CONFIG_RESIDUAL_AUDIT_INPUTS=/path/to/AUDIT-INPUTS.json \
     CONFIG_RESIDUAL_OUTPUT_DIR=/path/to/audit-output \
     CONFIG_RESIDUAL_CPU_PYTHON=/path/to/python
```

Capture observes all 64 routing sites and requires unchanged logits and
parameters. Every solver receives identical captured FP32 preprojection
logits; FP64 casts those logits rather than recomputing the router. Compare
20, 40, 80, 160 and 320 repeats in each precision. Preserve normalization
axes, order and epsilon 1e-6. The original 20-repeat calculation must match
the executed operator exactly.

Each calculation reports all-matrix row and column maxima, percentiles and
fractions above 1e-4, the final single-iteration change, and the maximum
FP32/FP64 output difference at the same repeat count. Sample IDs combine
deterministic evenly spaced inputs
and the original operator's worst inputs at each site. Their window/token
IDs remain fixed across recipes. Raw sampled logits, matrices, gates and
dtypes are retained. Historical maxima alone cannot reconstruct distributions.

Same-recipe FP32/FP64 gradient comparisons require finite gradients and
per-matrix L2 error at most 1e-6 plus 1e-3 times the reference norm. Two
FP64 directional finite differences use steps 1e-4 and 1e-5, with error at
most 1e-8 plus 1e-3 times the derivative magnitude. These comparisons do
not require gradients of different iteration counts to match.

Select the least FP32 repeat count above 20 that passes row/column bounds
and the gradient checks on every captured checkpoint. FP64 remains a
numerical reference. If none passes by 320, stop without adding another
solver. A selected candidate then receives complete validation forwards,
reporting changed logits, CE and routing distributions. It must still pass
the original bounds on those changed forward states. This measures inference
sensitivity, not the effect of training with that operator.

## Conditional training pilot

The pilot requires a completed `AUDIT.json` with `candidate_qualified` status
and a frozen proposal. That proposal contains `source`, a hashed
`audit_inputs` receipt, a `training` directory/manifest receipt, two fresh
`seeds`, hashed `seed_audit` manifests, an `order_seed` and a finite
`max_process_seconds`. Training inputs contain only train and validation
arrays. No final holdout is used for selection or evaluation in this route.

```sh
make defconfig-layerrope-mhc-routing-audit
make CONFIG_RESIDUAL_AUDIT_ACTION=pilot \
     CONFIG_RESIDUAL_AUDIT_RESULT=/path/to/AUDIT.json \
     CONFIG_RESIDUAL_AUDIT_PROPOSAL=/path/to/PILOT-PROPOSAL.json \
     CONFIG_RESIDUAL_OUTPUT_DIR=/path/to/pilot-output \
     CONFIG_RESIDUAL_CPU_PYTHON=/path/to/python
```

The pilot reserves one H100 80GB GPU. It compares residual at LR 0.001,
original/candidate M at 0.003, and original/candidate ML at 0.01, with the
same candidate in both families. Both LayerRoPE slopes start at zero and
remain trainable. Each of ten fresh trajectories targets 787 synchronized
training seconds under the existing complete-update clock and overshoot
rules. The extra solver work and scheduled checks count toward training.
Offline validation and checkpoint costs remain separate.

Validation at common updates 25 and 100 compares learning per token; the
787-second endpoint compares learning per time. Never extend a trajectory
to obtain a common-update point. A clock violation makes the attempted
comparison incomplete and stops execution. Numerical failures stay recorded
without replacement seeds. Per-check matrix distributions and state RMS/extrema
complement finite model, gradient and optimizer checks. Common-update states
also retain gradient norms and full model/optimizer/RNG/order checkpoints.

For at least one family, continuation requires every observed candidate
routing bound to pass and validation CE to improve by at least 0.01 nat
against its matched original in both seed pairs. This is a descriptive
engineering screen. It does not establish significance, an optimal learning
rate, a population-mean effect or permission to launch a confirmation study.
No quantization, depth expansion or additional beta search runs here.
