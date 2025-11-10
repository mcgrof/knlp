"""
agent_task_ra_lens_sinkgd.py
=====================================================================
Task spec for an AI/code agent to extend and evaluate GPT-2 with
Lens-Gated Attention (RA + Discoverability + Route-Gated MLP)
AND introduce a new optimizer ablation: SinkGD (replacing Adam/AdamW).

This file is the "source of truth" for what to do, why, and how to judge success.
It references code you already have and a paper you uploaded.

---------------------------------------------------------------------
CONTEXT (what we’re doing)
---------------------------------------------------------------------
We have:
  - ra_lens_gpt2.py
      * Implements "Lens-Gated Attention": a parameter-efficient recipe
        combining:
          1) reciprocity (S^T lens),
          2) discoverability (column bias via u_h),
          3) per-head softmax gates,
          4) a route-gate to shift compute from attention→MLP across training,
          5) optional low-rank context into the MLP.
      * Includes a patcher for HF GPT-2 blocks.

  - train_ra_mla.py
      * Training driver supporting:
          - Baseline, RA, MLA, Lens-Gated modes (+ many ablations)
          - FlashAttention when available
          - Metrics logging (entropy, reciprocity), pruning hooks, etc.
      * CLI arguments include a lot of ablation switches and ratios.
      * Uses create_optimizer(...) from lib.optimizers (AdamW etc).

  - sinkgd.pdf  (uploaded alongside this spec)
      * A paper/doc describing "SinkGD" (Sinkhorn Gradient Descent).
      * Goal: add a clean SinkGD optimizer to the training stack and run
        ablations comparing SinkGD vs Adam/AdamW under Lens-Gated settings.

NOTE: This spec assumes PyTorch training. We will **add** a new optimizer
class `SinkGD` in `lib/optimizers.py` (or adjacent) and wire it up so
`--optimizer sinkgd` just works in train_ra_mla.py.

---------------------------------------------------------------------
WHY (rationale)
---------------------------------------------------------------------
1) Lens-Gated Attention aims to reduce KV-cache pressure by learning to route
   capacity into the MLP over time. It preserves (or improves) quality at
   similar compute by mixing S, S^T, and a discoverability column with
   per-head softmax gates → stable, small parameter overhead.

2) SinkGD (per the uploaded doc) promises structured, OT/Sinkhorn-inspired
   gradient transport that can improve optimization geometry in settings
   where vanilla AdamW is over-adaptive or noisy. We want to see if:
     - SinkGD reaches equal/better val-loss at similar or fewer steps
     - plays nice with route-gate annealing
     - stabilizes attention entropy or reciprocity metrics
     - changes memory/speed tradeoffs meaningfully

---------------------------------------------------------------------
HIGH-LEVEL TASKS (what to implement)
---------------------------------------------------------------------
T1. Add optimizer: SinkGD
    - Create `SinkGD` as a PyTorch optimizer (e.g., in lib/optimizers.py).
    - API parity with AdamW entry points so the training script can select
      it via `--optimizer sinkgd`.
    - Hyperparams: lr, weight_decay, beta (if needed), tau/epsilon entropic
      regularization (if required by Sinkhorn step), n_iter (inner Sinkhorn
      iters), and any transport balancing knobs defined in sinkgd.pdf.
    - Mixed precision & AMP-compat safe.
    - Proper state dict save/load.

    # Minimal behavioral contract (pseudo):
    #   grad -> precondition via Sinkhorn-like transport / normalization
    #   update: param -= lr * transformed_grad (+ weight decay if decoupled)
    #
    #   class SinkGD(torch.optim.Optimizer):
    #       def __init__(self, params, lr, weight_decay, tau, n_iter, ...):
    #           ...
    #       @torch.no_grad()
    #       def step(self):
    #           for group in self.param_groups:
    #               for p in group['params']:
    #                   if p.grad is None: continue
    #                   g = p.grad
    #                   # (1) optional decoupled weight decay
    #                   if wd := group.get('weight_decay', 0.0):
    #                       p.add_(p, alpha=-wd * group['lr'])
    #                   # (2) Sinkhorn-like transform of g (see sinkgd.pdf)
    #                   #     e.g., normalize rows/cols in a structured view
    #                   #     or apply entropic OT projection with 'n_iter'
    #                   g_tilde = sinkhorn_like_transform(g, tau, n_iter, ...)
    #                   p.add_(g_tilde, alpha=-group['lr'])

T2. Wire `--optimizer sinkgd` into train_ra_mla.py
    - In create_optimizer(...), add a branch for "sinkgd".
    - Respect existing LR schedulers, clipping, AMP.
    - Ensure state dict checkpointing works.

T3. Add a new ablation family using SinkGD
    - Extend `--ra_mla_ablation_step` with `SX` variants:
        * "S0": Baseline GPT-2 + SinkGD (no Lens/RA/MLA)
        * "S1": Lens-Gated (L3 setup: reciprocity+discoverability) + SinkGD
        * "S2": Full Lens (L6: +low-rank ctx, +KV compression) + SinkGD
        * "S3": Full Lens (L7: +conductor) + SinkGD
      Map these to the corresponding lens flags (like L3/L6/L7) but swap
      optimizer to sinkgd regardless of `--optimizer`.
    - Add CLI flags if needed (e.g., --sinkgd-tau, --sinkgd-iters).

T4. Maintain dry-run
    - `--dry-run` should still pass (CPU one-batch FWD/BWD/OPT-step).
    - Ensure no device-assumptions in SinkGD path.

T5. Metrics & logging (unchanged APIs)
    - Keep existing logs:
        * Attention entropy, reciprocity (if available)
        * Route-gate mean/std/min/max
        * Time per iter, VRAM (if CUDA)
        * Train/val loss
    - Optionally log SinkGD-specific stats (e.g., avg Sinkhorn iters taken).

T6. Documentation & discoverability
    - Put a docstring header on the SinkGD class describing the update rule
      and citing `sinkgd.pdf` (no verbatim >25 words).
    - Mention expected hyper-param ranges in comments.

---------------------------------------------------------------------
FILES TO READ / TOUCH
---------------------------------------------------------------------
INPUTS (read):
  - ./ra_lens_gpt2.py
  - ./train_ra_mla.py
  - ./lib/optimizers.py (or wherever create_optimizer lives)
  - ./lib/scaling_curves.py (leave as-is)
  - ./sinkgd.pdf  <-- reference for the algorithm details

OUTPUTS (modify/add):
  - ./lib/optimizers.py        : add SinkGD class + factory branch
  - ./train_ra_mla.py          : wire "--optimizer sinkgd" and SX ablations
  - ./docs/optimizers_sinkgd.md: (NEW) short rationale, hyperparams, examples
  - ./checkpoints_ra_mla/*.pt  : produced by runs
  - ./checkpoints_ra_mla/metrics.json : produced by runs

---------------------------------------------------------------------
NEW CLI (additions)
---------------------------------------------------------------------
# New optimizer choice:
  --optimizer sinkgd

# SinkGD hyper-params (examples; adjust to paper):
  --sinkgd-lr 3e-4
  --sinkgd-weight-decay 0.1
  --sinkgd-tau 0.1
  --sinkgd-iters 10
  --sinkgd-eps 1e-8

# New ablation steps (SX):
  --ra_mla_ablation_step S0|S1|S2|S3
    S0: Baseline + SinkGD
    S1: Lens L3   + SinkGD   (reciprocity + discoverability)
    S2: Lens L6   + SinkGD   (L3 + low-rank ctx + KV compression)
    S3: Lens L7   + SinkGD   (S2 + conductor mode)

---------------------------------------------------------------------
ACCEPTANCE CRITERIA (done when all pass)
---------------------------------------------------------------------
A1) `--dry-run` succeeds on CPU:
    python train_ra_mla.py --dry-run --optimizer sinkgd

A2) Smoke training (GPU or CPU ok) runs ≥200 iters without error:
    python train_ra_mla.py --use-lens --ra_mla_ablation_step S1 \
        --optimizer sinkgd --max-iters 200 --log-interval 10

A3) Checkpoints and metrics saved; route-gate values logged in S1–S3.

A4) Ablation switch correctly reconfigures model:
    - S0: no Lens/RA/MLA toggles active
    - S1: reciprocity+discoverability on, no ctx/kv-compression
    - S2: as S1 + low-rank ctx + KV compression
    - S3: as S2 + conductor mode

A5) Training parity sanity:
    With comparable LR, S1 SinkGD reaches a similar or better val loss
    than AdamW baseline L3 in ~the same wall time on a small run
    (not a strict guarantee—just observe and record).

---------------------------------------------------------------------
SUGGESTED DEFAULTS (tune as needed)
---------------------------------------------------------------------
# For small GPT-2 and 1k context:
  --batch-size 12
  --gradient-accumulation 4
  --block-size 1024
  --dtype bfloat16 (if supported), else float32
  --max-iters 10000  (for real runs)
  --eval-interval 200
  --log-interval 10

# SinkGD starting point (adjust after quick grid):
  --sinkgd-lr 3e-4
  --sinkgd-weight-decay 0.1
  --sinkgd-tau 0.1
  --sinkgd-iters 5
  --sinkgd-eps 1e-8

---------------------------------------------------------------------
EVALUATION MATRIX (compact)
---------------------------------------------------------------------
Baseline (AdamW):
  - L3: Reciprocity+Discoverability, no ctx/kv-comp
  - L6: L3 + low-rank ctx + KV compression
  - L7: L6 + conductor

SinkGD:
  - S0: Baseline (no Lens/RA/MLA)
  - S1: L3 + SinkGD
  - S2: L6 + SinkGD
  - S3: L7 + SinkGD

Track (per run):
  - final/val loss, time/iter, VRAM/iter
  - attention entropy (if enabled), reciprocity score
  - route_gate mean/std/min/max (Lens runs)
  - any SinkGD inner-loop stats (optional)

---------------------------------------------------------------------
CODE CHANGES (concrete to-dos)
---------------------------------------------------------------------
[lib/optimizers.py]
  1) Add class SinkGD(torch.optim.Optimizer) with:
     - state dict (if you keep per-param buffers)
     - decoupled weight decay option
     - `sinkhorn_like_transform(...)` helper implementing the core step
       per sinkgd.pdf (entropic OT / row-col normalization n_iter times).
     - AMP-friendly (no grad graph ops in .step, use @torch.no_grad()).
  2) In create_optimizer(...):
     elif name == "sinkgd":
         opt = SinkGD(model.parameters(),
                      lr=args.learning_rate or args.sinkgd_lr,
                      weight_decay=args.weight_decay or args.sinkgd_weight_decay,
                      tau=args.sinkgd_tau,
                      n_iter=args.sinkgd_iters,
                      eps=args.sinkgd_eps)
         scheduler = existing_scheduler_or_None
         return opt, scheduler, gradient_clip_norm, spam_state, adamprune_state

[train_ra_mla.py]
  3) argparse: add SinkGD flags (--sinkgd-*)
  4) Extend ablation switch:
       if step == "S0":  # Baseline + SinkGD
           force_use_lens = False
           optimizer = "sinkgd"
       elif step == "S1":  # L3 + SinkGD
           set lens: reciprocity=True, discoverability=True
           ctx=False, kv=False, conductor=False
           optimizer="sinkgd"
       elif step == "S2":  # L6 + SinkGD
           as above + ctx=True, kv=True
           optimizer="sinkgd"
       elif step == "S3":  # L7 + SinkGD
           as S2 + conductor=True
           optimizer="sinkgd"

[docs/optimizers_sinkgd.md]  (NEW)
  5) Short note:
     - Problem SinkGD tries to solve.
     - Update rule sketch.
     - Default hyperparams and stability tips.
     - How to run (examples below).

---------------------------------------------------------------------
COMMANDS (quickstart)
---------------------------------------------------------------------
# Dry-run:
python train_ra_mla.py --dry-run --optimizer sinkgd

# Baseline Lens L3 with AdamW (control):
python train_ra_mla.py --use-lens --ra_mla_ablation_step L3 \
  --optimizer adamw --max-iters 200 --log-interval 10

# Lens S1 with SinkGD:
python train_ra_mla.py --use-lens --ra_mla_ablation_step S1 \
  --optimizer sinkgd --sinkgd-lr 3e-4 --sinkgd-tau 0.1 --sinkgd-iters 5 \
  --max-iters 200 --log-interval 10

# Deeper lens (ctx + KV comp) with SinkGD:
python train_ra_mla.py --use-lens --ra_mla_ablation_step S2 \
  --optimizer sinkgd --sinkgd-lr 3e-4 --sinkgd-tau 0.1 --sinkgd-iters 5 \
  --max-iters 200 --log-interval 10

# Full lens + conductor with SinkGD:
python train_ra_mla.py --use-lens --ra_mla_ablation_step S3 \
  --optimizer sinkgd --sinkgd-lr 3e-4 --sinkgd-tau 0.1 --sinkgd-iters 5 \
  --max-iters 200 --log-interval 10

---------------------------------------------------------------------
NOTE ON IMPLEMENTING SinkGD TRANSFORM
---------------------------------------------------------------------
- Read ./sinkgd.pdf and implement the gradient transform accordingly.
- A simple starting point mimicking Sinkhorn normalization:
    * View gradient tensor in a structured way (per-parameter or batched)
    * Iteratively normalize rows/cols with temperature tau for n_iter steps
      (stabilize with eps; detach from autograd in .step()).
    * Return normalized g_tilde with preserved scale (you may re-scale by
      original grad norm or use tau to control sharpness).
- Keep the code numerically stable (avoid log(0), clamp small values).

---------------------------------------------------------------------
RISKS & GUARDRAILS
---------------------------------------------------------------------
- Don’t change public signatures used by train_ra_mla except where noted.
- Keep AdamW/other opts unaffected.
- If SinkGD is too slow, allow --sinkgd-iters=0 to degenerate to a light
  normalization fallback for debugging.
- Respect AMP & bfloat16 (do transforms in float32 if needed, then cast back).

---------------------------------------------------------------------
DELIVERABLES SUMMARY
---------------------------------------------------------------------
- Working `--optimizer sinkgd` + SX ablations
- Clean docs page with hyper-param guidance
- Metrics & checkpoints for S0–S3 and L3/L6/L7 (AdamW) baselines
- Brief comparison table (val loss, time/iter, VRAM, entropy, route_gate mean)
"""

# Optional: include a tiny reference stub the agent can copy into lib/optimizers.py
SINKGD_REFERENCE_STUB = r'''
import torch
from torch.optim.optimizer import Optimizer

class SinkGD(Optimizer):
    """Sinkhorn-like Gradient Descent (SinkGD).
    See sinkgd.pdf for details on the gradient transform and suggested hyperparams.
    """
    def __init__(self, params, lr=3e-4, weight_decay=0.0, tau=0.1, n_iter=5, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, tau=tau, n_iter=n_iter, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr   = group["lr"]
            wd   = group.get("weight_decay", 0.0)
            tau  = group.get("tau", 0.1)
            iters= int(group.get("n_iter", 5))
            eps  = group.get("eps", 1e-8)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # decoupled weight decay
                if wd != 0.0:
                    p.add_(p, alpha=-wd * lr)
                # sinkhorn-like transform of the gradient
                g_tilde = _sinkhorn_like_transform(g, tau=tau, iters=iters, eps=eps)
                p.add_(g_tilde, alpha=-lr)
        return loss

def _sinkhorn_like_transform(g, tau=0.1, iters=5, eps=1e-8):
    # Implementation detail should follow sinkgd.pdf.
    # Safe default: operate in fp32 for stability, return original dtype.
    orig_dtype = g.dtype
    x = g.detach().float()
    # Example placeholder: symmetric row/col normalization on a flattened view.
    # Replace this with the correct transport projection per the paper.
    v = x
    for _ in range(max(0, iters)):
        # row-norm (treat last dim as "cols")
        denom = v.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
        v = v / denom
        # col-norm
        denom = v.abs().sum(dim=-2, keepdim=True).clamp_min(eps) if v.ndim >= 2 else v.abs().sum().clamp_min(eps)
        v = v / denom
        # temperature smoothing
        v = (v / max(eps, tau)).tanh()  # cheap bounded smoothing
    return v.to(orig_dtype)
'''
