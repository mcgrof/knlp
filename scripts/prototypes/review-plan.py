
"""
ra_ablation_plan.py
===================

Purpose
-------
This script is a *specification + scaffold* you can hand to another AI/agent/developer
to implement and run the next round of ablations for **Reciprocal Attention (RA)**
and **Discoverability**. It captures:

1) **Geometry-driven metrics** computed on per-head score matrices S = QK^T:
   - Symmetry skew (how near-undirected the head's graph is)
   - Column hubness via Gini (which tokens want to broadcast)
   - Row entropy (how peaky queries are)
   - Column contrast (std of column mass)
   - Spectral gap of the symmetrized score matrix

2) **Policy** (how to decide where RA/Discoverability help):
   - Turn RA on where symmetry is high and spectral structure is clear.
   - Turn Discoverability on where hubness is high and stable.

3) **Ablation grid** with small, surgical knobs:
   - stabilize_mix, gate_temp vs independent gates
   - discoverability {additive, multiplicative}, disc_temp, std-norm
   - schedule for w_rec (anneal-up warmup) and layer placement (top-half only)

4) **Initialization** from yesterday's metrics:
   - Heads showing utility get a small positive bias toward the useful lens.
   - Others stay biased to standard attention.

Plug-in Points
--------------
- This file is *framework-agnostic*. Fill in the TODOs to connect with your model:
  - Where to fetch per-head score matrices S (or S logits before softmax).
  - Where to apply per-head gate inits and schedules.
  - How to run short eval passes to estimate per-head utility.

Outputs
-------
- Functions that compute metrics from S (B,H,T,T).
- A reference ablation grid.
- Gate initialization policy.
- Lightweight schedules for annealing.
- An example "orchestrate_ablation()" sketch showing how to stitch things together.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional, Any
import numpy as np

# -----------------------------------------------------------------------------
# Utils: metrics on attention score matrices
# -----------------------------------------------------------------------------

def _safe_softmax(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    denom = np.sum(ex, axis=axis, keepdims=True) + eps
    return ex / denom

def fro_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)) + 1e-12)

def gini(x: np.ndarray) -> float:
    """Gini coefficient for nonnegative 1D array x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    # Gini = 1 - 2 * sum((n - i + 0.5) * x_i) / (n * sum(x))
    # Using cumulative form for stability
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(max(0.0, min(1.0, g)))

def spectral_gap_sym(S: np.ndarray) -> float:
    """
    Spectral gap of symmetrized matrix S_sym = (S + S^T)/2 on the last two dims.
    Returns (lambda1 - lambda2)/lambda1 in [0,1].
    For large T, consider using a power method (left as TODO).
    """
    T = S.shape[-1]
    S_sym = 0.5 * (S + S.transpose(-1, -2))
    # Small regularization to avoid NaN on degenerate cases
    S_sym = S_sym + 1e-6 * np.eye(T, dtype=S.dtype)
    # Full eigendecomp (fine for small T; replace with power-iteration for large T)
    w = np.linalg.eigvalsh(S_sym)  # ascending
    lam1, lam2 = float(w[-1]), float(w[-2]) if T > 1 else (float(w[-1]), 0.0)
    if lam1 <= 1e-12:
        return 0.0
    return float((lam1 - lam2) / lam1)

@dataclass
class HeadMetrics:
    symmetry: float          # 1 - ||S - S^T||_F / ||S||_F       (right = more symmetric)
    hub_gini: float          # Gini of column mass c = sum_i S_ij (up = more broadcasty)
    row_entropy: float       # mean entropy over query rows of softmax(S)
    col_std: float           # std of column mass
    spectral_gap: float      # (lambda1 - lambda2)/lambda1 of symmetrized S
    # Optional: raw values for debugging
    col_mass_mean: float
    col_mass_std: float
    skew_fro_ratio: float    # ||S - S^T||_F / ||S||_F

def compute_head_metrics(S: np.ndarray) -> HeadMetrics:
    """
    Compute geometry metrics for a single head's score matrix S [T,T].
    S can be the *logits* prior to softmax; that's preferable for geometry.
    """
    assert S.ndim == 2 and S.shape[0] == S.shape[1], "S must be [T,T]"
    T = S.shape[0]

    # Symmetry / skew
    S_t = S.T
    skew_norm = fro_norm(S - S_t)
    base_norm = fro_norm(S)
    skew_ratio = skew_norm / (base_norm + 1e-12)
    symmetry = float(max(0.0, min(1.0, 1.0 - skew_ratio)))

    # Column mass and stats
    col_mass = np.sum(S, axis=0)  # before softmax, serves as "broadcast intention"
    col_mean = float(np.mean(col_mass))
    col_std = float(np.std(col_mass))
    hub = gini(np.maximum(col_mass, 0.0))  # clamp negatives if present

    # Row entropy after softmax over keys
    P = _safe_softmax(S, axis=-1)
    row_ent = float(np.mean(-np.sum(P * np.log(np.maximum(P, 1e-12)), axis=-1)))

    # Spectral gap of symmetrized S
    gap = spectral_gap_sym(S)

    return HeadMetrics(
        symmetry=symmetry,
        hub_gini=hub,
        row_entropy=row_ent,
        col_std=col_std,
        spectral_gap=gap,
        col_mass_mean=col_mean,
        col_mass_std=col_std,
        skew_fro_ratio=skew_ratio,
    )

# -----------------------------------------------------------------------------
# Policy: where RA / Discoverability should help
# -----------------------------------------------------------------------------

@dataclass
class HeadPolicy:
    use_reciprocity: bool
    use_discoverability: bool
    reason: str

def decide_head_policy(m: HeadMetrics,
                       sym_thresh: float = 0.4,
                       gap_thresh: float = 0.15,
                       hub_thresh: float = 0.30) -> HeadPolicy:
    """
    Heuristics from our discussion:
    - RA good when symmetry is high and spectral gap is decent.
    - Discoverability good when hubness (Gini) is high.
    """
    use_ra = (m.symmetry >= sym_thresh) and (m.spectral_gap >= gap_thresh)
    use_disc = (m.hub_gini >= hub_thresh)
    reason = f"sym={m.symmetry:.2f}, gap={m.spectral_gap:.2f}, hub={m.hub_gini:.2f}"
    return HeadPolicy(use_reciprocity=use_ra, use_discoverability=use_disc, reason=reason)

# -----------------------------------------------------------------------------
# Gate initialization & schedules
# -----------------------------------------------------------------------------

@dataclass
class GateInit:
    """Logit-space initialization for [std, rec, disc] per head."""
    std: float = 2.0
    rec: float = -1.0
    disc: float = -1.0

def init_from_policy(policy: HeadPolicy) -> GateInit:
    """
    Bias heads to standard attention, then lift the useful lens slightly.
    This maps to softmax over [std, rec, disc].
    """
    gi = GateInit()
    if policy.use_reciprocity:
        gi.rec = 0.5   # small but nonzero
        gi.std = 1.5
    if policy.use_discoverability:
        gi.disc = 0.3
        gi.std = min(gi.std, 1.5)
    return gi

def anneal_weight(step: int, warmup_steps: int, max_w: float) -> float:
    """
    Cosine anneal up from ~0 to max_w over warmup_steps.
    Use to scale w_rec or w_disc globally during warmup.
    """
    if warmup_steps <= 0:
        return max_w
    x = min(1.0, max(0.0, step / float(warmup_steps)))
    # half-cosine from 0 -> 1
    return float(max_w * 0.5 * (1 - np.cos(np.pi * x)))

# -----------------------------------------------------------------------------
# Ablation grid
# -----------------------------------------------------------------------------

@dataclass
class AblationConfig:
    name: str
    stabilize_mix: bool = True
    gate_mode: str = "softmax"            # "softmax" | "independent"
    gate_temp: float = 1.0                # used if gate_mode == "softmax"
    disc_form: str = "additive"           # "additive" | "multiplicative" | "both"
    disc_temp: float = 2.0
    disc_stdnorm: bool = False
    layer_placement: str = "top_half"     # "all" | "top_half"
    warmup_steps: int = 0                 # anneal w_rec, w_disc
    notes: str = ""

def default_ablation_grid() -> List[AblationConfig]:
    """
    Minimal, cheap ablation set:
      - A: baseline (std only) lives outside this grid
      - B: RA only with stabilize_mix
      - C: RA + multiplicative discoverability
      - D: C + annealed w_rec
      - E: D + per-head init from metrics + top-half placement
    """
    return [
        AblationConfig(
            name="B_RA_only",
            stabilize_mix=True,
            gate_mode="softmax",
            gate_temp=1.0,
            disc_form="additive",
            disc_temp=2.0,
            disc_stdnorm=False,
            layer_placement="top_half",
            warmup_steps=0,
            notes="Reciprocity only; top-half layers; stabilized mix."
        ),
        AblationConfig(
            name="C_RA_plus_DiscMult",
            stabilize_mix=True,
            gate_mode="softmax",
            gate_temp=1.0,
            disc_form="multiplicative",
            disc_temp=2.0,
            disc_stdnorm=False,
            layer_placement="top_half",
            warmup_steps=0,
            notes="RA + multiplicative discoverability; top-half layers."
        ),
        AblationConfig(
            name="D_RA_DiscMult_Anneal",
            stabilize_mix=True,
            gate_mode="softmax",
            gate_temp=1.0,
            disc_form="multiplicative",
            disc_temp=2.0,
            disc_stdnorm=False,
            layer_placement="top_half",
            warmup_steps= int(0.15 * 100000),  # TODO: replace 100k with total steps * 0.15
            notes="As C, but cosine-anneal w_rec over first ~15% of steps."
        ),
        AblationConfig(
            name="E_RA_DiscMult_Anneal_PerHeadInit",
            stabilize_mix=True,
            gate_mode="softmax",
            gate_temp=1.0,
            disc_form="multiplicative",
            disc_temp=2.0,
            disc_stdnorm=False,
            layer_placement="top_half",
            warmup_steps= int(0.20 * 100000),  # TODO: replace w/ total steps * 0.2
            notes="As D, plus per-head gate init from geometry metrics; top-half only."
        ),
        AblationConfig(
            name="F_RA_independent_gates",
            stabilize_mix=True,
            gate_mode="independent",
            disc_form="multiplicative",
            disc_temp=2.0,
            disc_stdnorm=False,
            layer_placement="top_half",
            warmup_steps= int(0.20 * 100000),
            notes="Independent [0,1] gates (no sum-to-1); tends to avoid early collapse."
        ),
    ]

# -----------------------------------------------------------------------------
# Orchestration sketch (fill in the TODOs to integrate with your trainer)
# -----------------------------------------------------------------------------

def collect_metrics_from_batch(S_bhtT: np.ndarray) -> List[List[HeadMetrics]]:
    """
    Given a batch of score matrices S with shape [B, H, T, T],
    compute per-head metrics averaged (or just on the first element) over B.
    """
    assert S_bhtT.ndim == 4, "Expected [B,H,T,T]"
    B, H, T, _ = S_bhtT.shape
    metrics: List[List[HeadMetrics]] = []
    for h in range(H):
        # average S over batch for this head (or sample a subset for speed)
        S_h = np.mean(S_bhtT[:, h, :, :], axis=0)
        metrics.append([compute_head_metrics(S_h)])
    return metrics

def suggest_gate_inits_from_metrics(metrics_per_head: List[List[HeadMetrics]]) -> List[GateInit]:
    """
    For each head, decide policy and produce a GateInit.
    """
    inits: List[GateInit] = []
    for per_head in metrics_per_head:
        # use the last (most recent) metrics for decision
        m = per_head[-1]
        pol = decide_head_policy(m)
        inits.append(init_from_policy(pol))
    return inits

def orchestrate_ablation(*,
                         fetch_S_fn,
                         apply_gate_inits_fn,
                         run_eval_fn,
                         total_steps: int,
                         steps_between_measures: int = 500,
                         grid: Optional[List[AblationConfig]] = None) -> Dict[str, Any]:
    """
    High-level pseudo-pipeline. Pass in project-specific hooks:

      fetch_S_fn(step) -> np.ndarray[B,H,T,T]
          Return a representative batch of score logits S (pre-softmax).

      apply_gate_inits_fn(gate_inits: List[GateInit], layer_placement: str):
          Apply per-head gate inits to the model, optionally restricting to top-half layers.

      run_eval_fn(config: AblationConfig) -> Dict[str,float]
          Run a training/eval segment under the given config and return metrics
          (e.g., {'val_loss': ..., 'latency_ms': ..., 'power_w': ...}).

    Returns a dict mapping config name to the final reported metrics.
    """
    results: Dict[str, Any] = {}
    grid = grid or default_ablation_grid()

    # 1) Early measurement: get geometry to seed per-head inits
    S = fetch_S_fn(step=0)  # [B,H,T,T] logits pre-softmax
    metrics_per_head = collect_metrics_from_batch(S)
    gate_inits = suggest_gate_inits_from_metrics(metrics_per_head)

    for cfg in grid:
        # 2) Apply inits and placement
        apply_gate_inits_fn(gate_inits, layer_placement=cfg.layer_placement)

        # 3) Configure runtime flags in your model/trainer (pseudocode):
        #    - model.stabilize_mix = cfg.stabilize_mix
        #    - model.gate_mode = cfg.gate_mode
        #    - model.gate_temp = cfg.gate_temp
        #    - model.disc_form = cfg.disc_form
        #    - model.disc_temp = cfg.disc_temp
        #    - model.disc_stdnorm = cfg.disc_stdnorm
        #    - trainer.warmup_steps = cfg.warmup_steps
        # TODO: bind these to your actual configuration system.

        # 4) Run eval/training slice and collect metrics
        out = run_eval_fn(cfg)
        results[cfg.name] = {"config": asdict(cfg), "report": out}

    return results

# -----------------------------------------------------------------------------
# Example stubs (to be implemented in your project)
# -----------------------------------------------------------------------------

def example_fetch_S_fn(step: int) -> np.ndarray:
    """
    TODO: Replace with a real hook to your model.
    Should return logits S (pre-softmax) with shape [B,H,T,T].
    This is just a stub returning a small random matrix for illustration.
    """
    B, H, T = 2, 4, 16
    rng = np.random.default_rng(seed=1337 + step)
    S = rng.normal(size=(B, H, T, T)).astype(np.float32)
    # enforce causal mask pattern if you want to mirror training conditions
    return S

def example_apply_gate_inits_fn(gate_inits: List[GateInit], layer_placement: str) -> None:
    """
    TODO: Wire these inits into your model's per-head gate parameters.
    'layer_placement' can be used to select which layers (e.g., top-half) receive nonzero rec/disc inits.
    """
    # Example log printout; replace with actual model param assignment.
    print(f"[INIT] layer_placement={layer_placement}")
    for h, gi in enumerate(gate_inits):
        print(f"  head {h}: init logits [std, rec, disc] = [{gi.std:.2f}, {gi.rec:.2f}, {gi.disc:.2f}]")

def example_run_eval_fn(cfg: AblationConfig) -> Dict[str, float]:
    """
    TODO: Execute a short train/eval slice with the given config enabled on your model.
    Return key scalars to compare across configs.
    """
    # Dummy values; replace with real eval.
    rng = np.random.default_rng(hash(cfg.name) % (2**32))
    return {
        "val_loss": float(3.6 + 0.02 * rng.standard_normal() - (0.03 if 'RA' in cfg.name else 0.0)),
        "latency_ms": float(7.0 + (0.2 if 'Disc' in cfg.name else 0.05) + 0.05 * rng.standard_normal()),
        "power_w": float(205 + 2 * rng.standard_normal()),
    }

if __name__ == "__main__":
    # Quick smoke test of the scaffold with example stubs
    grid = default_ablation_grid()
    results = orchestrate_ablation(
        fetch_S_fn=example_fetch_S_fn,
        apply_gate_inits_fn=example_apply_gate_inits_fn,
        run_eval_fn=example_run_eval_fn,
        total_steps=100_000,
        steps_between_measures=1_000,
        grid=grid,
    )
    for name, payload in results.items():
        print(f"\n=== {name} ===")
        print("config:", payload["config"])
        print("report:", payload["report"])
