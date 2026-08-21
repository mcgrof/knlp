"""State-synthesis pruning signals for the open-weight lane.

Program O scores a finished checkpoint without its training
history. Two of its signal builders live here (the fresh Fisher
and K-FAC factor arms come from kfac_capture via program_o):

Frozen Adam replay. Initialize v = 0, keep the weights frozen (no
optimizer step is ever taken), feed calibration batches, and EMA
the squared full-weight minibatch gradients:

    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

g_t is the gradient of the batch-mean LM loss w.r.t. each target
weight — Adam's own minibatch-mean-square construction. This is
deliberately distinct from the per-token empirical Fisher diagonal
D = E[g_i^2 x_j^2] captured by FactorAccumulator: D squares
per-token gradients before averaging, the replay squares the
already-averaged minibatch gradient. v is snapshotted at the
requested batch counts so score stability versus calibration
length can be measured. This is a synthetic calibration EMA, not
recovered training history. beta1 is accepted for signature parity
with Adam, but the first moment never enters the
|w| * (v + eps)^q score, so only the second moment is accumulated.
The plain (bias-uncorrected) v is returned: at a fixed batch count
the Adam bias correction 1/(1 - beta2^t) is a uniform per-run
scalar and cannot change any ranking or mask.

Checkpoint-trajectory proxy. For two published checkpoints of the
same model at steps t-k and t:

    trajectory_score = |w_t * (w_t - w_{t-k})|

This estimates recent weight movement, not Adam's moments; the
distinction is the point. load_hf_target_weights pulls only the
target linear weights of the earlier revision onto the CPU, so the
proxy never holds two full models on the GPU.

Both signals are wired into program_o.build_scores (the replay_v /
w_prev keyword arguments, same pattern as stale_v) and configured
in cmd_prune_eval via the "replay" and "trajectory" config keys.

Design contract: knlp-key-results/fisher-factored-pruning-20260820/
CLOUD_WAVE_PLAN.md, "O.3 State-synthesis methods".
"""

import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def frozen_adam_replay(
    model: nn.Module,
    targets: Dict[str, nn.Linear],
    batch_iter: Iterable[torch.Tensor],
    betas: Tuple[float, float] = (0.9, 0.999),
    checkpoints: Sequence[int] = (16, 50, 200),
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Synthetic Adam second moment from a frozen-weight replay.

    Iterates [B, T] token batches, backwards the mean LM loss with
    the weights frozen (model.eval, no optimizer step), and EMAs
    each target weight's squared minibatch gradient into v with
    beta2. Gradients are zeroed between batches. Returns
    {n_batches: {target name: v}} with one independent CPU fp32
    snapshot per requested checkpoint; iteration stops after the
    largest checkpoint. Raises RuntimeError if batch_iter is
    exhausted before every checkpoint is reached.
    """
    wanted = sorted({int(c) for c in checkpoints})
    if not wanted or wanted[0] < 1:
        raise ValueError(f"checkpoints must be positive ints, got {checkpoints!r}")
    beta2 = float(betas[1])
    model.eval()
    device = next(model.parameters()).device
    v = {
        n: torch.zeros(m.weight.shape, dtype=torch.float32, device=device)
        for n, m in targets.items()
    }
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    model.zero_grad(set_to_none=True)
    n_seen = 0
    wanted_set = set(wanted)
    for x in batch_iter:
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        with torch.no_grad():
            for n, m in targets.items():
                g = m.weight.grad
                if g is None:
                    raise RuntimeError(f"no gradient for {n}.weight")
                v[n].mul_(beta2).add_((1.0 - beta2) * g.float() ** 2)
        model.zero_grad(set_to_none=True)
        n_seen += 1
        if n_seen in wanted_set:
            snapshots[n_seen] = {
                n: t.detach().to(device="cpu", dtype=torch.float32, copy=True)
                for n, t in v.items()
            }
        if n_seen >= wanted[-1]:
            break
    missing = [c for c in wanted if c not in snapshots]
    if missing:
        raise RuntimeError(
            f"replay batches exhausted after {n_seen}; no snapshot for {missing}"
        )
    return snapshots


def trajectory_scores(
    w_now: Dict[str, torch.Tensor], w_prev: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """|w_now * (w_now - w_prev)| per target, CPU fp32.

    w_prev may be a superset of w_now (e.g. a full state dict);
    missing or shape-mismatched entries raise.
    """
    missing = sorted(set(w_now) - set(w_prev))
    if missing:
        raise KeyError(f"prev weights missing targets: {missing[:3]}")
    out: Dict[str, torch.Tensor] = {}
    for n, w in w_now.items():
        prev = w_prev[n]
        if tuple(prev.shape) != tuple(w.shape):
            raise ValueError(
                f"{n}: prev shape {tuple(prev.shape)} != now {tuple(w.shape)}"
            )
        w32 = w.detach().float().cpu()
        out[n] = (w32 * (w32 - prev.detach().float().cpu())).abs()
    return out


def load_hf_target_weights(
    model_id: str, revision: str, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Target linear weights of one HF checkpoint revision.

    Loads the revision on the requested device (CPU by default —
    never a second full model on the GPU), discovers the same
    target linears program_o prunes, copies their weights out as
    fp32 CPU tensors, and drops the model.
    """
    from transformers import AutoModelForCausalLM

    # Imported lazily: program_o imports this module at top level.
    from fim.fisher_pruning.program_o import discover_target_linears

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    weights = {
        n: m.weight.detach().float().cpu().clone()
        for n, m in discover_target_linears(model).items()
    }
    del model
    return weights
