"""
bitter10_variants.py

Implements three Bitter10 coherence-style pruning scorers designed to keep memory
overhead negligible while capturing aspects of temporal stability:

1) bitter10z (Zero-New-Memory)
   - Stateless, uses only existing Adam states (m, v) and current weights.
   - Score ~ bitter3 * (|m| / (sqrt(v)+eps))  --> favors strong, low-noise directions.

2) bitter10s (Sketch Tracker)
   - Tiny, fixed-size per-tensor sketch (e.g., k=64 indices) to estimate temporal stats:
     sign stability, directional consistency (via m as Δw proxy), jitter from v.
   - Applies layer-level multipliers to bitter3 base. ~hundreds of bytes per tensor.

3) bitter10c (Checkpoint Δw)
   - Infrequent 1×-weights snapshot (CPU/disk) every N steps.
   - Long-horizon Δw = w_t - w_{t-N}; uses cos(m, Δw) and |Δw| as stability signals
     at prune time. No per-step history.

Integration notes
-----------------
- All scorers return a single flattened tensor of scores (keep-high → prune-low).
- Plug into your global thresholding/masking the same way as other bitter variants.
- Optimizer must be Adam/AdamW so exp_avg/exp_avg_sq exist.

License: Same as your project. Adjust and inline into your repo structure as needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Any, Tuple
import torch
import torch.nn as nn


# =============================================================================
# Utilities
# =============================================================================

def _device_of_params(params: Iterable[nn.Parameter]) -> torch.device:
    for p in params:
        if p is not None:
            return p.device
    return torch.device("cpu")

def _reduce(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    if mode == "median":
        return x.flatten().median()
    return x.flatten().mean()

def _safe_cos_elemwise(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Stable, elementwise surrogate cosine in L1 form (keeps shape)
    num = a * b
    den = (a.abs() + eps) * (b.abs() + eps)
    return (num / den).clamp(-1.0, 1.0)

def _bitter3_base(p: nn.Parameter, m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Base score used in several variants: |w| * sqrt(|m|)
    return p.data.abs() * torch.sqrt(m.abs() + eps)


# =============================================================================
# 1) Bitter10z — ZERO-NEW-MEMORY, Stateless Scorer
# =============================================================================

@dataclass
class Bitter10ZCfg:
    eps_m: float = 1e-12
    include_bias: bool = False
    include_norm: bool = False  # set True if you want to score LayerNorm/BatchNorm weights too

def bitter10z_scores(params: Iterable[nn.Parameter],
                     optimizer: torch.optim.Optimizer,
                     cfg: Bitter10ZCfg = Bitter10ZCfg()) -> torch.Tensor:
    """
    Stateless score:
      S = (|w| * sqrt(|m| + eps)) * (|m| / (sqrt(v) + eps))

    Interprets |m|/sqrt(v) as a stability/noise ratio for Adam signal.
    """
    chunks = []
    for p in params:
        if p.grad is None:
            continue
        is_bias = (p.ndim == 1)
        is_norm = hasattr(p, "is_norm") or ("norm" in (getattr(p, "name", "") or "").lower())
        if (is_bias and not cfg.include_bias) or (is_norm and not cfg.include_norm):
            continue

        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", None)
        v = st.get("exp_avg_sq", None)
        if (m is None) or (v is None):
            continue

        base = _bitter3_base(p, m, cfg.eps_m)
        stability = m.abs() / (v.sqrt() + cfg.eps_m)
        score = base * stability
        chunks.append(score.flatten())

    if not chunks:
        dev = _device_of_params(params)
        return torch.tensor([], device=dev)
    return torch.cat(chunks, dim=0)


# =============================================================================
# 2) Bitter10s — SKETCH TRACKER (few KB per tensor)
# =============================================================================

@dataclass
class Bitter10SCfg:
    k: int = 64               # samples per tensor
    ema: float = 0.9          # EMA for layer-level stats
    eps_m: float = 1e-12
    include_bias: bool = False
    include_norm: bool = False
    reduce: str = "mean"      # "mean" or "median"

class SketchTracker:
    """
    Tiny per-tensor tracker:
      - idx: fixed sampled indices (stable via deterministic seed)
      - prev_m: previous sampled m-values
      - ema_cos: EMA of |cos(m_now, m_prev)| on sample
      - ema_flip: EMA of sign flip rate on sample
      - jitter: EMA of sqrt(v) (dispersion proxy) on sample
    """
    def __init__(self, cfg: Bitter10SCfg = Bitter10SCfg()):
        self.cfg = cfg
        self.state: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        self.state.clear()

    def _init_tensor(self, p: nn.Parameter, m: torch.Tensor):
        numel = p.data.numel()
        device = p.data.device
        if numel <= self.cfg.k:
            sel = torch.arange(numel, device=device)
        else:
            # deterministic sample per tensor size (stable across steps)
            g = torch.Generator(device=device).manual_seed(numel)
            sel = torch.randperm(numel, generator=g, device=device)[: self.cfg.k]
        st = {
            "idx": sel,
            "prev_m": m.flatten()[sel].clone(),
            "ema_cos": torch.tensor(0.0, device=device),
            "ema_flip": torch.tensor(0.0, device=device),
            "jitter": torch.tensor(0.0, device=device),
        }
        self.state[id(p)] = st

    @torch.no_grad()
    def update(self, p: nn.Parameter, m: torch.Tensor, v: torch.Tensor):
        pid = id(p)
        if pid not in self.state:
            self._init_tensor(p, m)

        st = self.state[pid]
        sel = st["idx"]

        m_now = m.flatten()[sel]
        m_prev = st["prev_m"]

        # |cos| on sampled m (proxy for directional consistency of Δw)
        num = (m_now * m_prev).abs()
        den = (m_now.abs() + 1e-12) * (m_prev.abs() + 1e-12)
        cos_elem = (num / den).clamp(0.0, 1.0)
        cos_dir = cos_elem.mean()

        # sign flips
        flips = (torch.sign(m_now) != torch.sign(m_prev)).float().mean()

        # jitter from sqrt(v) on sample
        jitter = v.flatten()[sel].sqrt().mean().clamp_min(0.0)

        # EMAs
        ema = self.cfg.ema
        st["ema_cos"] = st["ema_cos"] * ema + cos_dir * (1 - ema)
        st["ema_flip"] = st["ema_flip"] * ema + flips * (1 - ema)
        st["jitter"] = st["jitter"] * ema + jitter * (1 - ema)

        st["prev_m"].copy_(m_now)

    def layer_modifiers(self, p: nn.Parameter) -> Tuple[float, float, float]:
        st = self.state.get(id(p))
        if not st:
            return (1.0, 1.0, 1.0)
        # Higher is better for cos; lower is better for flips & jitter
        flip_mod = float((1.0 - st["ema_flip"]).clamp(0.0, 1.0))
        cos_mod = float(st["ema_cos"].clamp(0.0, 1.0))
        jit_mod = float(1.0 / (1.0 + st["jitter"]))
        return (flip_mod, cos_mod, jit_mod)

def bitter10s_scores(params: Iterable[nn.Parameter],
                     optimizer: torch.optim.Optimizer,
                     tracker: SketchTracker,
                     cfg: Bitter10SCfg = Bitter10SCfg()) -> torch.Tensor:
    """
    Layer-modulated bitter3:
      score = (|w| * sqrt(|m|+eps)) * flip_mod * cos_mod * jit_mod

    where flip_mod, cos_mod, jit_mod are layer-level scalars from a tiny sketch.
    """
    chunks = []
    for p in params:
        if p.grad is None:
            continue
        is_bias = (p.ndim == 1)
        is_norm = hasattr(p, "is_norm") or ("norm" in (getattr(p, "name", "") or "").lower())
        if (is_bias and not cfg.include_bias) or (is_norm and not cfg.include_norm):
            continue

        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", None)
        v = st.get("exp_avg_sq", None)
        if (m is None) or (v is None):
            continue

        # Update sketch with current states (cheap)
        tracker.update(p, m, v)

        flip_mod, cos_mod, jit_mod = tracker.layer_modifiers(p)
        base = _bitter3_base(p, m, cfg.eps_m)
        score = base * flip_mod * cos_mod * jit_mod
        chunks.append(score.flatten())

    if not chunks:
        dev = _device_of_params(params)
        return torch.tensor([], device=dev)
    return torch.cat(chunks, dim=0)


# =============================================================================
# 3) Bitter10c — CHECKPOINT Δw (infrequent, 1× snapshot)
# =============================================================================

@dataclass
class Bitter10CCfg:
    eps_m: float = 1e-12
    include_bias: bool = False
    include_norm: bool = False
    cos_reduce: str = "mean"  # "mean" or "median"

class DeltaCheckpoint:
    """
    Stores a 1× snapshot of parameters on CPU (or disk via torch.save if you prefer).
    Use: take_snapshot(params); later compute bitter10c_scores(...) at prune time.
    """
    def __init__(self):
        self.snap: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def take_snapshot(self, params: Iterable[nn.Parameter], to_cpu: bool = True):
        self.snap.clear()
        for p in params:
            if p is None:
                continue
            t = p.data.detach().clone()
            if to_cpu:
                t = t.cpu()
            self.snap[id(p)] = t

    def has_snapshot(self) -> bool:
        return len(self.snap) > 0

def bitter10c_scores(params: Iterable[nn.Parameter],
                     optimizer: torch.optim.Optimizer,
                     ckpt: DeltaCheckpoint,
                     cfg: Bitter10CCfg = Bitter10CCfg()) -> torch.Tensor:
    """
    Long-horizon Δw scorer at prune time:
      Δw = w_now - w_snapshot

    score = bitter3 * cos_plus * inv_jitter_plus
    where:
      - cos_plus = clamp( cos(m, Δw), 0, 1 ), reduced to scalar per tensor
      - inv_jitter_plus = 1 / (1 + mean(sqrt(v)))  (tensor-level)

    Intuition:
      - Large |Δw| consistent with current momentum m (positive cosine) → keep.
      - High jitter (sqrt(v)) → downweight.
    """
    if not ckpt.has_snapshot():
        raise RuntimeError("DeltaCheckpoint is empty. Call take_snapshot(...) before bitter10c_scores().")

    chunks = []
    for p in params:
        if p.grad is None:
            continue
        is_bias = (p.ndim == 1)
        is_norm = hasattr(p, "is_norm") or ("norm" in (getattr(p, "name", "") or "").lower())
        if (is_bias and not cfg.include_bias) or (is_norm and not cfg.include_norm):
            continue

        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", None)
        v = st.get("exp_avg_sq", None)
        if (m is None) or (v is None):
            continue

        snap = ckpt.snap.get(id(p))
        if snap is None:
            # Parameter added after snapshot; treat as neutral (base only)
            base = _bitter3_base(p, m, cfg.eps_m)
            chunks.append(base.flatten())
            continue

        # Δw across the long window
        w_now = p.data
        delta = (w_now.cpu() - snap).to(w_now.device)

        # tensor-level cosine(m, Δw) (positive only)
        cos_elem = _safe_cos_elemwise(m, delta, eps=1e-12)
        cos_plus = torch.clamp(_reduce(cos_elem, cfg.cos_reduce), min=0.0, max=1.0)

        # jitter from v
        inv_jitter_plus = 1.0 / (1.0 + v.sqrt().mean().clamp_min(0.0))

        base = _bitter3_base(p, m, cfg.eps_m)
        score = base * cos_plus * inv_jitter_plus
        chunks.append(score.flatten())

    if not chunks:
        dev = _device_of_params(params)
        return torch.tensor([], device=dev)
    return torch.cat(chunks, dim=0)


# =============================================================================
# Example usage sketch (optional)
# =============================================================================

if __name__ == "__main__":
    # Tiny smoke test with a linear layer and synthetic AdamW states
    torch.manual_seed(0)
    lin = nn.Linear(16, 16).to("cpu")
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)

    # Fake a few steps to populate Adam states
    for _ in range(5):
        x = torch.randn(8, 16)
        y = torch.randint(0, 16, (8,))
        opt.zero_grad(set_to_none=True)
        out = lin(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()

    params = list(lin.parameters())

    # --- bitter10z ---
    s_z = bitter10z_scores(params, opt)
    print("[bitter10z] shape:", s_z.shape, "min/max:", float(s_z.min()), float(s_z.max()))

    # --- bitter10s ---
    tracker = SketchTracker(Bitter10SCfg(k=32))
    s_s = bitter10s_scores(params, opt, tracker)
    print("[bitter10s] shape:", s_s.shape, "min/max:", float(s_s.min()), float(s_s.max()))

    # --- bitter10c ---
    ckpt = DeltaCheckpoint()
    ckpt.take_snapshot(params, to_cpu=True)  # take snapshot now
    # do a few more steps to build a delta
    for _ in range(3):
        x = torch.randn(8, 16)
        y = torch.randint(0, 16, (8,))
        opt.zero_grad(set_to_none=True)
        out = lin(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
    s_c = bitter10c_scores(params, opt, ckpt)
    print("[bitter10c] shape:", s_c.shape, "min/max:", float(s_c.min()), float(s_c.max()))
