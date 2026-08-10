"""Signed reciprocal credit spectrum utilities (spectral_delta_ra lane).

Math contract. For one attention head, flatten batch and token
dimensions into N rows of d = d_head features:

    R = Y_rec - Y_std      [N, d]   reciprocal delta
    G = dL / dY_std        [N, d]   output gradient
    M = G^T R / N          [d, d]   signed cross-feature credit
    H = -(M + M^T) / 2              signed reciprocal credit operator

H is real symmetric; eigendecompose H = U diag(lambda) U^T and order
modes by descending |lambda|. For a mode-gated correction
Y = Y_std + R U_r diag(beta) U_r^T the first-order loss change is

    dL ~= -N * sum_i beta_i * lambda_i

so a scalar gate (beta_i all equal) sees only trace(H). The sign
convention makes a positive eigenvalue mean "a positive gate is
locally loss-reducing".

H is NOT a Fisher Information Matrix and must not be described as one.
Diagnostic companions (do not conflate; see the private plan in
knlp-key-results/ra-spectral-modes-20260810/):

    C_z = E[Z^T Z], Z = G*R   elementwise-credit second moment (PSD)
    C_r = E[R^T R]            reciprocal activation covariance (PSD)
    C_g = E[G^T G]            gradient covariance (PSD)
    A   = skew(Q^T K / N)     forward-only query/key asymmetry (skew)

Numerics: accumulation is FP64 on CPU regardless of input dtype;
matrices are symmetrized before eigh; eigenvector signs are arbitrary
so every consumer must be sign-invariant; near-degenerate eigenvalues
are compared as subspaces via ||U^T V||_F^2 / r.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

TOPK_FRACTIONS = (1, 2, 4, 8)


def _to_cpu64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64)


@dataclass
class HeadCreditAccumulator:
    """Streaming accumulator for one head's credit matrices.

    update() takes G and R chunks of shape [n, d]; totals are kept in
    FP64 on CPU. When retain_raw is set the raw rows are also kept
    (FP32 CPU) so permutation nulls and split-half stability can be
    computed after the fact; size that against calibration length.
    """

    d: int
    retain_raw: bool = False
    n: int = 0
    m_sum: torch.Tensor = field(init=False)
    cz_sum: torch.Tensor = field(init=False)
    cr_sum: torch.Tensor = field(init=False)
    cg_sum: torch.Tensor = field(init=False)
    gr_norm_sum: float = 0.0
    raw_g: List[torch.Tensor] = field(default_factory=list)
    raw_r: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.m_sum = torch.zeros(self.d, self.d, dtype=torch.float64)
        self.cz_sum = torch.zeros(self.d, self.d, dtype=torch.float64)
        self.cr_sum = torch.zeros(self.d, self.d, dtype=torch.float64)
        self.cg_sum = torch.zeros(self.d, self.d, dtype=torch.float64)

    def update(self, g: torch.Tensor, r: torch.Tensor) -> None:
        if g.shape != r.shape or g.ndim != 2 or g.shape[1] != self.d:
            raise ValueError(f"bad shapes g={tuple(g.shape)} r={tuple(r.shape)}")
        g64 = _to_cpu64(g)
        r64 = _to_cpu64(r)
        z64 = g64 * r64
        self.m_sum += g64.T @ r64
        self.cz_sum += z64.T @ z64
        self.cr_sum += r64.T @ r64
        self.cg_sum += g64.T @ g64
        self.gr_norm_sum += float((g64.norm(dim=1) * r64.norm(dim=1)).sum().item())
        self.n += g.shape[0]
        if self.retain_raw:
            self.raw_g.append(g.detach().to("cpu", torch.float32))
            self.raw_r.append(r.detach().to("cpu", torch.float32))

    def raw(self) -> Dict[str, torch.Tensor]:
        if not self.retain_raw:
            raise RuntimeError("retain_raw was not enabled")
        return {
            "G": torch.cat(self.raw_g, dim=0),
            "R": torch.cat(self.raw_r, dim=0),
        }

    def finalize(self) -> Dict[str, torch.Tensor]:
        if self.n == 0:
            raise RuntimeError("no rows accumulated")
        m = self.m_sum / self.n
        return {
            "M": m,
            "H": signed_credit_from_m(m),
            "C_z": 0.5 * (self.cz_sum + self.cz_sum.T) / self.n,
            "C_r": 0.5 * (self.cr_sum + self.cr_sum.T) / self.n,
            "C_g": 0.5 * (self.cg_sum + self.cg_sum.T) / self.n,
            "N": torch.tensor(self.n),
            "gr_norm_mean": torch.tensor(self.gr_norm_sum / self.n),
        }


def signed_credit_from_m(m: torch.Tensor) -> torch.Tensor:
    """H = -sym(M). Positive eigenvalue == positive gate reduces loss."""
    return -0.5 * (m + m.T)


def sym_eig_by_abs(h: torch.Tensor):
    """Eigendecompose a symmetric matrix; order by descending |lambda|.

    Returns (lam, U) with lam[i] the eigenvalue of column U[:, i].
    """
    h64 = 0.5 * (h + h.T).to(torch.float64)
    lam, u = torch.linalg.eigh(h64)
    order = torch.argsort(lam.abs(), descending=True)
    return lam[order], u[:, order]


def signed_spectrum_stats(
    lam: torch.Tensor, gr_norm_mean: Optional[float] = None, eps: float = 1e-12
) -> Dict[str, float]:
    """Diagnostics for the signed credit spectrum (see module docstring)."""
    a = lam.abs()
    mass = float(a.sum())
    trace = float(lam.sum())
    stats = {
        "scalar_signal": abs(trace),
        "trace": trace,
        "spectral_mass": mass,
        "positive_mass": float(lam.clamp(min=0).sum()),
        "negative_mass": float((-lam).clamp(min=0).sum()),
        "cancellation_ratio": 1.0 - abs(trace) / (mass + eps),
        "max_abs_eigenvalue": float(a.max()) if lam.numel() else 0.0,
        "n_pos_modes": int((lam > 0).sum()),
        "n_neg_modes": int((lam < 0).sum()),
    }
    for k in TOPK_FRACTIONS:
        stats[f"top{k}_mass_fraction"] = float(a[: min(k, a.numel())].sum()) / (
            mass + eps
        )
    p = a / (mass + eps)
    entropy = float(-(p * (p + eps).log()).sum())
    stats["spectral_entropy"] = entropy
    stats["effective_rank"] = float(math.exp(entropy))
    if gr_norm_mean is not None:
        stats["normalized_spectral_mass"] = mass / (gr_norm_mean + eps)
    return stats


def psd_spectrum_stats(
    c: torch.Tensor, tau: Optional[float] = None, eps: float = 1e-12
) -> Dict[str, float]:
    """Diagnostics for PSD companions (C_z, C_r, C_g, C_asym).

    logdet uses logdet(I + C/tau); raw determinants are deliberately
    not reported (they are an elaborate underflow benchmark).
    """
    lam, _ = sym_eig_by_abs(c)
    lam = lam.clamp(min=0.0)
    total = float(lam.sum())
    d = lam.numel()
    if tau is None:
        tau = total / d if total > 0 else 1.0
    stats = {
        "trace": total,
        "tau": tau,
        "logdet_I_plus_C_over_tau": float((1.0 + lam / tau).log().sum()),
        "stable_rank": total / (float(lam.max()) + eps) if d else 0.0,
    }
    for k in TOPK_FRACTIONS:
        stats[f"top{k}_explained_variance"] = float(lam[: min(k, d)].sum()) / (
            total + eps
        )
    p = lam / (total + eps)
    entropy = float(-(p * (p + eps).log()).sum())
    stats["spectral_entropy"] = entropy
    stats["effective_rank"] = float(math.exp(entropy))
    return stats


def diag_ridge(c: torch.Tensor, eps: float = 1e-8):
    """Ridge for numerical diagnostics only; the value is recorded and
    must never be used to manufacture rank."""
    d = c.shape[0]
    ridge = eps * float(c.abs().trace()) / d
    return c + ridge * torch.eye(d, dtype=c.dtype), ridge


def subspace_overlap(u: torch.Tensor, v: torch.Tensor) -> float:
    """||U^T V||_F^2 / r for two [d, r] orthonormal bases; in [0, 1]."""
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch {tuple(u.shape)} vs {tuple(v.shape)}")
    r = u.shape[1]
    return float((u.T.to(torch.float64) @ v.to(torch.float64)).norm() ** 2) / r


def split_half_overlap(
    g: torch.Tensor,
    r: torch.Tensor,
    ranks: Sequence[int] = (1, 2, 4, 8),
    seed: int = 0,
    rows_per_block: int = 1,
) -> Dict[str, float]:
    """Stability of the top-r credit subspace across two disjoint halves.

    With rows_per_block > 1 the split happens at sequence granularity
    so rows of one sequence never land in both halves — a row-level
    split would let within-sequence autocorrelation inflate the
    apparent stability.
    """
    n = g.shape[0]
    gen = torch.Generator().manual_seed(seed)
    if rows_per_block > 1:
        n = (n // rows_per_block) * rows_per_block
        g, r = g[:n], r[:n]
        n_blocks = n // rows_per_block
        block_perm = torch.randperm(n_blocks, generator=gen)
        perm = (
            block_perm[:, None] * rows_per_block + torch.arange(rows_per_block)[None, :]
        ).reshape(-1)
    else:
        perm = torch.randperm(n, generator=gen)
    half = n // 2
    ia, ib = perm[:half], perm[half : 2 * half]
    out: Dict[str, float] = {}
    lam_a, u_a = sym_eig_by_abs(
        signed_credit_from_m(_to_cpu64(g[ia]).T @ _to_cpu64(r[ia]) / half)
    )
    lam_b, u_b = sym_eig_by_abs(
        signed_credit_from_m(_to_cpu64(g[ib]).T @ _to_cpu64(r[ib]) / half)
    )
    for rank in ranks:
        rank = min(rank, u_a.shape[1])
        out[f"split_half_overlap_r{rank}"] = subspace_overlap(
            u_a[:, :rank], u_b[:, :rank]
        )
    return out


def permutation_null(
    g: torch.Tensor,
    r: torch.Tensor,
    n_perm: int = 100,
    seed: int = 0,
    rows_per_block: int = 1,
) -> Dict[str, object]:
    """Null distribution of normalized spectral mass under shuffling.

    rows_per_block=1 permutes individual rows of R against G: it
    preserves both marginal distributions while destroying the G-R
    row pairing. Rows from contiguous token positions are strongly
    autocorrelated, so the row-level null destroys within-sequence
    structure too and is EASY to beat — empirically, control heads
    beat it almost as often as trusted heads. With rows_per_block =
    tokens-per-sequence, whole sequences of R are permuted against G
    instead: within-sequence autocorrelation survives in both
    tensors and only the cross-tensor pairing breaks. Audits report
    both; the block null is the conservative one.
    """
    n = g.shape[0]
    if rows_per_block > 1:
        n = (n // rows_per_block) * rows_per_block
        g, r = g[:n], r[:n]
    g64 = _to_cpu64(g)
    r64 = _to_cpu64(r)
    gr_norm_mean = float((g64.norm(dim=1) * r64.norm(dim=1)).mean())
    actual_lam, _ = sym_eig_by_abs(signed_credit_from_m(g64.T @ r64 / n))
    actual = signed_spectrum_stats(actual_lam, gr_norm_mean)
    gen = torch.Generator().manual_seed(seed)
    n_blocks = n // rows_per_block
    null_mass = []
    for _ in range(n_perm):
        if rows_per_block == 1:
            perm = torch.randperm(n, generator=gen)
        else:
            block_perm = torch.randperm(n_blocks, generator=gen)
            perm = (
                block_perm[:, None] * rows_per_block
                + torch.arange(rows_per_block)[None, :]
            ).reshape(-1)
        lam_p, _ = sym_eig_by_abs(signed_credit_from_m(g64.T @ r64[perm] / n))
        null_mass.append(
            signed_spectrum_stats(lam_p, gr_norm_mean)["normalized_spectral_mass"]
        )
    null = torch.tensor(null_mass, dtype=torch.float64)
    actual_mass = actual["normalized_spectral_mass"]
    return {
        "actual_normalized_spectral_mass": actual_mass,
        "rows_per_block": rows_per_block,
        "null_mean": float(null.mean()),
        "null_p95": float(null.quantile(0.95)),
        "null_p99": float(null.quantile(0.99)),
        "exceeds_p95": bool(actual_mass > float(null.quantile(0.95))),
        "percentile_of_actual": float((null < actual_mass).float().mean()),
        "n_perm": n_perm,
        "null_samples": [float(x) for x in null_mass],
    }


class QKAsymmetryAccumulator:
    """Forward-only query/key feature-space asymmetry (no gradients).

    Accumulates B = Q^T K / N over calibration rows, then
    A = skew(B). A is real skew-symmetric, so its modes pair up; the
    basis comes from the SVD of A (equivalently eigh of A^T A) — a
    symmetric eigendecomposition of A itself would be wrong. Use even
    ranks with this basis.
    """

    def __init__(self, d: int):
        self.d = d
        self.n = 0
        self.b_sum = torch.zeros(d, d, dtype=torch.float64)

    def update(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if q.shape != k.shape or q.ndim != 2 or q.shape[1] != self.d:
            raise ValueError(f"bad shapes q={tuple(q.shape)} k={tuple(k.shape)}")
        self.b_sum += _to_cpu64(q).T @ _to_cpu64(k)
        self.n += q.shape[0]

    def finalize(self, eps: float = 1e-12) -> Dict[str, object]:
        if self.n == 0:
            raise RuntimeError("no rows accumulated")
        b = self.b_sum / self.n
        a = 0.5 * (b - b.T)
        s = 0.5 * (b + b.T)
        w, sv, _ = torch.linalg.svd(a)
        return {
            "B": b,
            "A": a,
            "rho_asym": float(a.norm()) / (float(s.norm()) + eps),
            "singular_values": sv,
            "U": w,  # left singular vectors, ordered by descending sv
            "C_asym": a.T @ a,
        }


def s_pm_ratio(s: torch.Tensor, eps: float = 1e-12) -> float:
    """||S_minus||_F / ||S_plus||_F for pre-mask scores [..., T, T].

    Mechanistic descriptor of logit asymmetry only — not a utility
    signal. Compute on raw scores BEFORE the causal mask for the exact
    algebraic decomposition.
    """
    st = s.transpose(-2, -1)
    s_plus = 0.5 * (s + st)
    s_minus = 0.5 * (s - st)
    return float(s_minus.norm()) / (float(s_plus.norm()) + eps)


def haar_random_basis(d: int, r: int, seed: int = 0) -> torch.Tensor:
    """Haar-distributed [d, r] orthonormal basis (control arm O10)."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, rr = torch.linalg.qr(a)
    # Fix the QR sign ambiguity so the distribution is Haar.
    q = q * torch.sign(torch.diagonal(rr)).unsqueeze(0)
    return q[:, :r]


def _sha256_of_file(path: Path, max_bytes: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(max_bytes))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def save_basis(
    out_dir: Path,
    u_by_head: Dict[str, torch.Tensor],
    lam_by_head: Dict[str, torch.Tensor],
    meta: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Serialize a basis bundle as basis.pt + basis.json.

    Head keys are "L{layer}H{head}". meta must identify basis_source,
    model commit/checkpoint hash, config/dataset hashes, calibration
    seed and token count, selection, d_head, rank, and the creation
    command; the caller owns that contract (validated by manifest
    checks, not here).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "basis.pt"
    json_path = out_dir / "basis.json"
    payload = {
        "U_by_layer_head": {k: v.clone() for k, v in u_by_head.items()},
        "eigenvalues_by_layer_head": {k: v.clone() for k, v in lam_by_head.items()},
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, pt_path)
    doc = dict(meta)
    doc.setdefault("schema_version", 1)
    doc["heads"] = sorted(u_by_head.keys())
    doc["basis_pt_sha256"] = _sha256_of_file(pt_path)
    doc["eigenspectrum"] = {k: [float(x) for x in v] for k, v in lam_by_head.items()}
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return {"basis_pt": str(pt_path), "basis_json": str(json_path)}


def load_basis(out_dir: Path) -> Dict[str, object]:
    out_dir = Path(out_dir)
    payload = torch.load(out_dir / "basis.pt", weights_only=True)
    meta = json.loads((out_dir / "basis.json").read_text())
    return {
        "U_by_layer_head": payload["U_by_layer_head"],
        "eigenvalues_by_layer_head": payload["eigenvalues_by_layer_head"],
        "extra": payload.get("extra"),
        "meta": meta,
    }
