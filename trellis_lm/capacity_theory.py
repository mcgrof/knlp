"""Associative-memory capacity theory for the Trellis memory update.

Exact reference algorithms and metrics for the operator-level capacity
study: Hebbian, delta/LMS (one-pass, K-cyclic-pass, converged), the exact
closed-form unrolling of the delta recurrence, least-squares and
recursive-least-squares oracles, and the nonlinear Trellis update
u = J_phi(z)^T (phi(z) - a). Pure functions over torch tensors,
float64-friendly. No argparse, no I/O.

Notation (fixed throughout; one letter never names two things):

    P   number of stored associations
    d   write/key dimension
    m   target/code dimension (Trellis slots)
    W   = [w_1, ..., w_P]  in R^{d x P}   (keys as columns)
    A   = [a_1, ..., a_P]  in R^{m x P}   (target codes as columns)
    G_W = W^T W,  G_A = A^T A             (P x P Gram matrices)
    M   in R^{m x d}                      (memory state)

Nonlinearities phi act over the code (m) dimension, matching the Trellis
slot axis. Closed-form VJPs for identity/silu/ln_silu are reused from
trellis_lm.activations via trellis_lm.trellis_memory; any other callable
falls back to the generic autograd VJP. Pass phi by name ("identity",
"silu", "ln_silu", ...) or as one of the trellis_lm.activations callables
so the closed-form dispatch fires.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import torch

from .activations import get_activation
from .trellis_memory import _trellis_vjp, _trellis_vjp_bilevel

_EPS = 1e-12

PhiLike = Union[str, Callable[[torch.Tensor], torch.Tensor]]


def _phi_fn(phi: PhiLike):
    return get_activation(phi) if isinstance(phi, str) else phi


def _as_schedule(x, P: int, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or length-P sequence to a per-column schedule."""
    t = torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    if t.dim() == 0:
        t = t.expand(P)
    if t.shape != (P,):
        raise ValueError(f"schedule shape {tuple(t.shape)} != ({P},)")
    return t


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------


def hebbian_memory(W: torch.Tensor, A: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """Heteroassociative Hebbian memory M_H = eta * A @ W^T  in R^{m x d}.

    Exact readout identity: M_H W = eta A G_W, i.e. per item
    z_j = eta a_j + eta sum_{i != j} a_i <w_i, w_j>  (signal + crosstalk).
    """
    return eta * A @ W.T


def delta_memory(
    W: torch.Tensor,
    A: torch.Tensor,
    gamma,
    beta,
    passes: Optional[int] = 1,
    M0: Optional[torch.Tensor] = None,
    tol: float = 1e-12,
    max_passes: int = 100_000,
) -> torch.Tensor:
    """Online LMS / identity-delta memory, sequential over columns.

    Per association t (in column order, cyclically over passes):
        M <- beta_t * M - gamma_t * (M w_t - a_t) w_t^T

    This is the phi = identity Trellis update (u = z - a). gamma/beta may
    be scalars or length-P per-column schedules, reused on every pass.
    passes: 1 = one-pass, K = K cyclic passes, None = iterate cyclic
    passes until the per-pass max-abs state change drops below tol
    (raises RuntimeError past max_passes — e.g. an inconsistent P > d
    system with constant gamma limit-cycles instead of converging).
    """
    d, P = W.shape
    m = A.shape[0]
    gammas = _as_schedule(gamma, P, W)
    betas = _as_schedule(beta, P, W)
    M = torch.zeros(m, d, dtype=W.dtype, device=W.device) if M0 is None else M0.clone()

    def one_pass(M: torch.Tensor) -> torch.Tensor:
        for t in range(P):
            w = W[:, t]
            u = M @ w - A[:, t]
            M = betas[t] * M - gammas[t] * torch.outer(u, w)
        return M

    if passes is not None:
        for _ in range(passes):
            M = one_pass(M)
        return M
    for _ in range(max_passes):
        M_next = one_pass(M)
        delta = (M_next - M).abs().max()
        M = M_next
        if delta < tol:
            return M
    raise RuntimeError(
        f"delta_memory did not converge in {max_passes} passes "
        f"(last per-pass change {delta:.3e}; inconsistent P > d systems "
        "with constant gamma do not converge)"
    )


def delta_unrolled_closed_form(
    W: torch.Tensor,
    A: torch.Tensor,
    gammas,
    betas,
    M0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact closed-form unrolling of the one-pass delta recurrence.

    With R_t = beta_t I - gamma_t w_t w_t^T (d x d, acting on M from the
    right), one pass of delta_memory equals

        M_P = M_0 R_1 R_2 ... R_P
              + sum_{i=1}^{P} gamma_i a_i w_i^T R_{i+1} ... R_P.

    Sign conventions match delta_memory exactly:
    M_t = M_{t-1} R_t + gamma_t a_t w_t^T. Computed via explicit suffix
    products (a genuinely different evaluation path from the recurrence,
    so the two implementations cross-check each other).
    """
    d, P = W.shape
    m = A.shape[0]
    gs = _as_schedule(gammas, P, W)
    bs = _as_schedule(betas, P, W)
    eye = torch.eye(d, dtype=W.dtype, device=W.device)
    R = [bs[t] * eye - gs[t] * torch.outer(W[:, t], W[:, t]) for t in range(P)]
    # suffix[i] = R_{i+1} ... R_P in 1-indexed math = R[i] @ ... @ R[P-1]
    suffix = [eye] * (P + 1)
    for i in range(P - 1, -1, -1):
        suffix[i] = R[i] @ suffix[i + 1]
    M = torch.zeros(m, d, dtype=W.dtype, device=W.device)
    if M0 is not None:
        M = M0 @ suffix[0]
    for t in range(P):
        M = M + gs[t] * torch.outer(A[:, t], W[:, t]) @ suffix[t + 1]
    return M


def least_squares_memory(W: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Offline least-squares memory M_LS = A W^+ (pseudoinverse oracle).

    For P <= d with full column rank: M_LS W = A exactly (interpolation).
    For P > d: M_LS W = A (W^+ W) — A projected onto the row space of W —
    and the residual satisfies the normal equations (M_LS W - A) W^T = 0.
    """
    return A @ torch.linalg.pinv(W)


def rls_memory(
    W: torch.Tensor,
    A: torch.Tensor,
    lam: float = 1e-8,
    return_state: bool = False,
):
    """Online recursive least squares (RLS) oracle.

    One presentation per column, exact ridge solution at every step;
    after all P columns M equals A W^T (W W^T + lam I)^{-1}. NOTE: RLS
    carries a d x d inverse-covariance state Pcov on top of the m x d
    memory — an extra O(d^2) state the Trellis update does not have. It
    is an online oracle, not a matched-state baseline. Pass
    return_state=True to get (M, Pcov) and inspect that state.
    """
    d, P = W.shape
    m = A.shape[0]
    Pcov = torch.eye(d, dtype=W.dtype, device=W.device) / lam
    M = torch.zeros(m, d, dtype=W.dtype, device=W.device)
    for t in range(P):
        w = W[:, t]
        Pw = Pcov @ w
        k = Pw / (1.0 + w @ Pw)
        err = A[:, t] - M @ w
        M = M + torch.outer(err, k)
        Pcov = Pcov - torch.outer(k, Pw)
    if return_state:
        return M, Pcov
    return M


def trellis_memory(
    W: torch.Tensor,
    A: torch.Tensor,
    phi: PhiLike,
    gamma,
    beta,
    passes: int = 1,
    M0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Nonlinear Trellis memory update, sequential over columns.

    Per association t:
        z_t = M w_t
        u_t = J_phi(z_t)^T (phi(z_t) - a_t)
        M  <- beta_t * M - gamma_t * u_t w_t^T

    phi is a name or callable over the last (m) dimension. The VJP reuses
    trellis_lm.trellis_memory._trellis_vjp: closed forms for
    identity/silu/ln_silu, generic autograd for any other phi. With
    phi = identity this reproduces delta_memory exactly.
    """
    phi = _phi_fn(phi)
    d, P = W.shape
    m = A.shape[0]
    gammas = _as_schedule(gamma, P, W)
    betas = _as_schedule(beta, P, W)
    M = torch.zeros(m, d, dtype=W.dtype, device=W.device) if M0 is None else M0.clone()
    for _ in range(passes):
        for t in range(P):
            w = W[:, t]
            z = M @ w
            u = _trellis_vjp(phi, z, A[:, t])
            M = betas[t] * M - gammas[t] * torch.outer(u, w)
    return M


# ---------------------------------------------------------------------------
# recall metrics
# ---------------------------------------------------------------------------


def score_matrix(M: torch.Tensor, W: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Matched-code score matrix S = A^T (M W), P x P; S[i, j] = <a_i, z_j>.

    For the Hebbian memory this is exactly S = eta G_A G_W.
    """
    return A.T @ (M @ W)


def readout_metrics(Z: torch.Tensor, A: torch.Tensor) -> dict:
    """Recall metrics for a readout Z = [z_1..z_P] against the code book A.

    normalized_mse   mean_j ||z_j - a_j||^2 / ||a_j||^2
    mean_cosine      mean_j cos(z_j, a_j)
    top1             fraction of j with argmax_i <a_i, z_j> == j
    mean/min_margin  S[j,j] - max_{i != j} S[i,j] over the score matrix
                     S = A^T Z (nan margins when P == 1: no impostors)
    """
    m, P = A.shape
    diff_sq = ((Z - A) ** 2).sum(dim=0)
    a_sq = (A * A).sum(dim=0).clamp_min(_EPS)
    nmse = (diff_sq / a_sq).mean()
    denom = (Z.norm(dim=0) * A.norm(dim=0)).clamp_min(_EPS)
    cosine = ((Z * A).sum(dim=0) / denom).mean()
    S = A.T @ Z
    idx = torch.arange(P, device=A.device)
    top1 = (S.argmax(dim=0) == idx).to(A.dtype).mean()
    if P > 1:
        diag = S.diagonal()
        off = S.masked_fill(torch.eye(P, dtype=torch.bool, device=A.device), -math.inf)
        margin = diag - off.max(dim=0).values
        mean_margin = margin.mean()
        min_margin = margin.min()
    else:
        mean_margin = torch.tensor(math.nan, dtype=A.dtype)
        min_margin = torch.tensor(math.nan, dtype=A.dtype)
    return {
        "normalized_mse": nmse,
        "mean_cosine": cosine,
        "top1": top1,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
    }


def recall_metrics(M: torch.Tensor, W: torch.Tensor, A: torch.Tensor) -> dict:
    """readout_metrics for the linear readout Z = M W."""
    return readout_metrics(M @ W, A)


# ---------------------------------------------------------------------------
# frame metrics
# ---------------------------------------------------------------------------


def welch_bound(dim: int, P: int) -> float:
    """Welch lower bound on max coherence of P unit vectors in R^dim.

    sqrt((P - dim) / (dim (P - 1))) for P > dim; 0 otherwise (orthonormal
    sets exist). The regular simplex frame (P = dim + 1) attains it.
    """
    if P <= dim or P < 2:
        return 0.0
    return math.sqrt((P - dim) / (dim * (P - 1)))


def frame_metrics(X: torch.Tensor, normalize: bool = True) -> dict:
    """Frame-geometry metrics for a code matrix X in R^{dim x P}.

    Columns are normalized first unless normalize=False. Reported:
    max_coherence, mean_sq_off_coherence (mean of G_ij^2 over i != j),
    frame_potential (||X^T X||_F^2, diagonal included, so orthonormal
    columns give exactly P), effective_rank (Roy-Vetterli exponential of
    singular-value entropy), tight_frame_distance
    (||(dim/P) X X^T - I||_F, zero iff a unit-norm tight frame), and
    welch_bound. frame_potential_bound is the matching lower bound
    max(P, P^2/dim) for unit-norm columns.
    """
    dim, P = X.shape
    Xn = X / X.norm(dim=0, keepdim=True).clamp_min(_EPS) if normalize else X
    G = Xn.T @ Xn
    eye = torch.eye(P, dtype=X.dtype, device=X.device)
    off = G - G * eye
    if P > 1:
        max_coh = off.abs().max()
        mean_sq_off = (off**2).sum() / (P * (P - 1))
    else:
        max_coh = torch.zeros((), dtype=X.dtype)
        mean_sq_off = torch.zeros((), dtype=X.dtype)
    frame_potential = (G**2).sum()
    s = torch.linalg.svdvals(Xn)
    p = s / s.sum().clamp_min(_EPS)
    p = p[p > 0]
    erank = torch.exp(-(p * p.log()).sum())
    eye_d = torch.eye(dim, dtype=X.dtype, device=X.device)
    tf_dist = ((dim / P) * (Xn @ Xn.T) - eye_d).norm()
    return {
        "max_coherence": max_coh,
        "mean_sq_off_coherence": mean_sq_off,
        "frame_potential": frame_potential,
        "frame_potential_bound": float(max(P, P**2 / dim)),
        "effective_rank": erank,
        "tight_frame_distance": tf_dist,
        "welch_bound": welch_bound(dim, P),
    }


# ---------------------------------------------------------------------------
# update-Jacobian helpers
# ---------------------------------------------------------------------------


def update_vector(z: torch.Tensor, a: torch.Tensor, phi: PhiLike) -> torch.Tensor:
    """u = J_phi(z)^T (phi(z) - a), differentiable in z (live z).

    The gradient of the inner loss l(z, a) = 1/2 ||phi(z) - a||^2. Reuses
    the bilevel (live-z) VJP dispatch from trellis_lm.trellis_memory.
    """
    return _trellis_vjp_bilevel(_phi_fn(phi), z, a)


def loss_update_jacobian(
    z: torch.Tensor, a: torch.Tensor, phi: PhiLike
) -> torch.Tensor:
    """H_ell(z, a) = du/dz in R^{m x m} via autograd.

    H_ell is the Hessian of the inner loss, hence symmetric:
    H_ell = J_phi^T J_phi + sum_k (phi_k(z) - a_k) Hess(phi_k). Identity
    phi gives H_ell = I; silu gives a diagonal; ln_silu couples all m
    coordinates.
    """
    phi_fn = _phi_fn(phi)
    return torch.autograd.functional.jacobian(
        lambda zz: _trellis_vjp_bilevel(phi_fn, zz, a), z.detach()
    )


def update_jacobian_kron(
    w: torch.Tensor, H: torch.Tensor, gamma: float, beta: float
) -> torch.Tensor:
    """One-step update Jacobian of F(M) = beta M - gamma u(M w, a) w^T.

    Local perturbation rule: dF = beta dM - gamma H_ell (dM w) w^T. In the
    COLUMN-MAJOR vec convention (vec stacks columns of M):

        J = beta I_{m d} - gamma (w w^T (x) H_ell)

    where (x) is the Kronecker product. CONVENTION TRAP: torch's reshape
    is row-major; flattening the [m,d,m,d] autograd Jacobian directly
    yields the transposed order H_ell (x) w w^T. update_jacobian_autograd
    permutes into column-major so the two agree; do not trust the
    ordering without that float64 cross-check.
    """
    d = w.shape[0]
    m = H.shape[0]
    eye = torch.eye(m * d, dtype=w.dtype, device=w.device)
    return beta * eye - gamma * torch.kron(torch.outer(w, w), H)


def update_jacobian_autograd(
    M: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    phi: PhiLike,
    gamma: float,
    beta: float,
) -> torch.Tensor:
    """Full autograd Jacobian of F(M) = beta M - gamma u(M w, a) w^T.

    Returned in the same column-major vec convention as
    update_jacobian_kron (index p = j*m + i for entry M[i, j]).
    """
    phi_fn = _phi_fn(phi)

    def F(Mm: torch.Tensor) -> torch.Tensor:
        u = _trellis_vjp_bilevel(phi_fn, Mm @ w, a)
        return beta * Mm - gamma * torch.outer(u, w)

    m, d = M.shape
    J = torch.autograd.functional.jacobian(F, M.detach())  # [m, d, m, d]
    return J.permute(1, 0, 3, 2).reshape(m * d, m * d)


# ---------------------------------------------------------------------------
# ensembles
# ---------------------------------------------------------------------------


def random_unit_columns(
    dim: int,
    P: int,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """P iid Gaussian columns in R^dim, normalized to unit length."""
    X = torch.randn(dim, P, generator=generator, dtype=dtype, device=device)
    return X / X.norm(dim=0, keepdim=True).clamp_min(_EPS)


def orthonormal_columns(
    dim: int,
    P: int,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """P random orthonormal columns in R^dim (requires P <= dim)."""
    if P > dim:
        raise ValueError(f"orthonormal ensemble needs P <= dim, got P={P} dim={dim}")
    X = torch.randn(dim, P, generator=generator, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(X)
    sign = torch.where(torch.diagonal(R) >= 0, 1.0, -1.0).to(dtype)
    return Q * sign.unsqueeze(0)


def low_coherence_frame(
    dim: int,
    P: int,
    steps: int = 2000,
    lr: float = 0.05,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Optimized low-coherence unit-norm frame (P > dim allowed).

    Simple projected gradient descent on the off-diagonal frame potential
    sum_{i != j} <x_i, x_j>^2 with column renormalization after each
    step. Frame-potential minimization drives the frame to a unit-norm
    TIGHT frame, so the MEAN-SQUARE off-diagonal coherence attains its
    Welch level (welch_bound^2); max coherence is not directly targeted
    and can stay well above the Welch bound (equiangularity is a stronger
    property than tightness). Large lr oscillates — keep lr modest.
    """
    X = random_unit_columns(dim, P, generator=generator, dtype=dtype, device=device)
    X = X.clone().requires_grad_(True)
    eye = torch.eye(P, dtype=dtype, device=device)
    for _ in range(steps):
        Xn = X / X.norm(dim=0, keepdim=True).clamp_min(_EPS)
        G = Xn.T @ Xn
        loss = ((G - eye) ** 2).sum()
        (grad,) = torch.autograd.grad(loss, X)
        with torch.no_grad():
            X -= lr * grad
            X /= X.norm(dim=0, keepdim=True).clamp_min(_EPS)
    return X.detach()


def clustered_columns(
    dim: int,
    P: int,
    n_clusters: int = 4,
    spread: float = 0.1,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Unit columns clustered around n_clusters random unit centers.

    Column t belongs to cluster t mod n_clusters and is the center plus
    isotropic Gaussian noise of scale `spread`, renormalized.
    """
    centers = random_unit_columns(
        dim, n_clusters, generator=generator, dtype=dtype, device=device
    )
    noise = torch.randn(dim, P, generator=generator, dtype=dtype, device=device)
    idx = torch.arange(P) % n_clusters
    X = centers[:, idx] + spread * noise
    return X / X.norm(dim=0, keepdim=True).clamp_min(_EPS)


def near_duplicate_columns(
    dim: int,
    P: int,
    eps: float = 1e-2,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Adversarial near-duplicate pairs: column 2k+1 is column 2k plus
    eps-scale noise, renormalized. Odd P leaves the last column unpaired."""
    n_base = (P + 1) // 2
    base = random_unit_columns(
        dim, n_base, generator=generator, dtype=dtype, device=device
    )
    noise = torch.randn(dim, n_base, generator=generator, dtype=dtype, device=device)
    dup = base + eps * noise
    dup = dup / dup.norm(dim=0, keepdim=True).clamp_min(_EPS)
    X = torch.empty(dim, 2 * n_base, dtype=dtype, device=device)
    X[:, 0::2] = base
    X[:, 1::2] = dup
    return X[:, :P]


def simplex_frame(
    dim: int, dtype: torch.dtype = torch.float64, device=None
) -> torch.Tensor:
    """Regular simplex frame: dim + 1 unit vectors in R^dim with pairwise
    inner products exactly -1/dim. Attains the Welch bound and is a
    unit-norm tight frame — the standard known construction for frame
    metric tests."""
    P = dim + 1
    E = torch.eye(P, dtype=dtype, device=device) - 1.0 / P
    U, S, _ = torch.linalg.svd(E)
    X = U[:, :dim].T @ E  # isometric drop into R^dim
    return X / X.norm(dim=0, keepdim=True).clamp_min(_EPS)


def range_matched_targets(
    phi: PhiLike,
    m: int,
    P: int,
    generator: Optional[torch.Generator] = None,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Range-matched target codes a_i = phi(r_i), r_i ~ scale * N(0, I_m).

    phi is applied per column over the m dimension (the code axis), so
    coordinate-coupling phis like ln_silu normalize over m, not over P.
    Avoids asking an activation to reconstruct targets outside its
    natural output range (Protocol A of the operator study).
    """
    phi_fn = _phi_fn(phi)
    R = scale * torch.randn(m, P, generator=generator, dtype=dtype, device=device)
    return phi_fn(R.T).T
