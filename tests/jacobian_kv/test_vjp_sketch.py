# SPDX-License-Identifier: GPL-2.0
"""The probe identity the whole screen rests on: ``E[u u'] = J' F J``.

If this does not hold, every Jacobian arm is measuring something other than
what it claims, and the failure is silent -- the tensors stay finite and the
numbers stay plausible.  So it is pinned here on maps whose Jacobian is known
in closed form, where the claim can be checked rather than assumed.

Three things are established:

  * the probe sampler's covariance is *exactly* the categorical Fisher
    ``diag(p) - p p'``, which is what makes ``onehot(y) - p`` a legitimate
    substitute for a matrix square root;
  * the second moment of the resulting VJPs converges to ``J' F J``, and the
    *mean* of those same VJPs converges to zero -- which is precisely why the
    signed mean-Jacobian control is a different object and not a cheaper
    estimator of the same one;
  * ``torch.func.vjp`` and ``torch.autograd.grad`` agree, so the portable route
    is a choice rather than a constraint.

CPU-only, seeded, no network, seconds to run.
"""

import math

import pytest
import torch

from research.jacobian_kv.sensitivity import fisher_probe, gaussian_probe

torch.manual_seed(0)


def _fisher(p: torch.Tensor) -> torch.Tensor:
    return torch.diag(p) - torch.outer(p, p)


def _many_probes(logits_row: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """``n`` independent probes at one distribution, drawn in a single call.

    ``fisher_probe`` samples independently per position, so broadcasting one
    row to ``n`` positions yields ``n`` iid draws without an ``n``-long Python
    loop.
    """
    logits = logits_row.reshape(1, 1, -1).expand(1, n, -1).contiguous()
    g = torch.Generator().manual_seed(seed)
    return fisher_probe(logits, g)[0]  # [n, V]


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------


def test_probe_covariance_is_exactly_the_categorical_fisher():
    logits = torch.tensor([2.0, 0.5, -1.0, 0.25, -0.75], dtype=torch.float32)
    p = torch.softmax(logits, dim=-1)
    z = _many_probes(logits, 400_000, seed=1).double()

    assert z.mean(dim=0).abs().max() < 0.01, "probe must be zero mean"
    emp = (z.T @ z) / z.shape[0]
    exact = _fisher(p.double())
    assert (emp - exact).abs().max() < 0.01


def test_probe_is_supported_on_the_fisher_range():
    """The all-ones direction is annihilated, exactly, for every draw.

    Not a statistical claim: ``onehot(y) - p`` sums to zero by construction, so
    the probe never excites the Fisher's null space.  A sampler that failed
    this would inject energy the receiver metric cannot see.
    """
    logits = torch.tensor([1.0, -2.0, 0.3, 0.9], dtype=torch.float32)
    z = _many_probes(logits, 2000, seed=2).double()
    # exact up to the float32 the probe is built in, not up to float64
    assert z.sum(dim=-1).abs().max() < 1e-6


def test_gaussian_probe_is_a_different_metric():
    """The isotropic control really is isotropic, so it is a fair contrast."""
    g = torch.Generator().manual_seed(3)
    z = gaussian_probe(torch.zeros(1, 200_000, 4), g)[0].double()
    emp = (z.T @ z) / z.shape[0]
    assert (emp - torch.eye(4, dtype=torch.float64)).abs().max() < 0.02


# ---------------------------------------------------------------------------
# the identity
# ---------------------------------------------------------------------------


def test_second_moment_of_vjps_converges_to_jt_f_j():
    """``E[u u'] -> J' F J`` on a linear map whose Jacobian is known exactly."""
    torch.manual_seed(4)
    D, V = 6, 5
    W = torch.randn(D, V, dtype=torch.float64)  # logits = x @ W, so J = W'
    logits_row = torch.tensor([1.5, -0.5, 0.2, 0.9, -1.2])
    p = torch.softmax(logits_row, dim=-1).double()
    F = _fisher(p)

    z = _many_probes(logits_row, 300_000, seed=5).double()  # [n, V]
    u = z @ W.T  # u_n = J' z_n = W z_n
    emp = (u.T @ u) / u.shape[0]
    exact = W @ F @ W.T

    rel = (emp - exact).abs().max() / exact.abs().max()
    assert rel < 0.02, f"relative error {rel:.4f}"


def test_the_mean_vjp_is_zero_so_the_control_is_a_different_object():
    """``E[J' z] = 0`` while ``E[u u'] != 0``.

    This is the whole reason a signed mean Jacobian cannot stand in for the
    second moment: averaging the VJPs destroys exactly the information the
    second moment keeps.  Any implementation where the two arms agree is
    measuring one of them wrongly.
    """
    torch.manual_seed(6)
    D, V = 5, 4
    W = torch.randn(D, V, dtype=torch.float64)
    logits_row = torch.tensor([0.7, -0.4, 1.1, -0.9])
    z = _many_probes(logits_row, 300_000, seed=7).double()
    u = z @ W.T

    mean_norm = u.mean(dim=0).norm()
    second_norm = ((u.T @ u) / u.shape[0]).norm()
    assert mean_norm < 0.02
    assert second_norm > 0.1
    assert mean_norm < 0.05 * second_norm


def test_directional_score_converges_at_the_monte_carlo_rate():
    """``mean_p (u_p . e)^2 -> e' G e``, with error shrinking like 1/sqrt(P)."""
    torch.manual_seed(8)
    D, V = 5, 4
    W = torch.randn(D, V, dtype=torch.float64)
    logits_row = torch.tensor([0.4, 0.9, -1.3, 0.2])
    p = torch.softmax(logits_row, dim=-1).double()
    G = W @ _fisher(p) @ W.T
    e = torch.randn(D, dtype=torch.float64)
    exact = float(e @ G @ e)

    errs = []
    for n in (64, 1024, 16384, 262144):
        u = _many_probes(logits_row, n, seed=9).double() @ W.T
        est = float((u @ e).pow(2).mean())
        errs.append(abs(est - exact) / abs(exact))
    assert errs[-1] < 0.05
    assert errs[-1] < errs[0], f"no convergence: {errs}"


# ---------------------------------------------------------------------------
# the two autograd routes
# ---------------------------------------------------------------------------


def test_func_vjp_and_autograd_grad_agree():
    """Both routes give the same ``J' z`` through a nonlinear path."""
    torch.manual_seed(10)
    D, H, V = 6, 8, 5
    w1 = torch.randn(D, H, dtype=torch.float64)
    w2 = torch.randn(H, V, dtype=torch.float64)

    def f(x):
        return torch.tanh(x @ w1) @ w2

    x = torch.randn(3, D, dtype=torch.float64)
    z = torch.randn(3, V, dtype=torch.float64)

    xg = x.clone().requires_grad_(True)
    (u_auto,) = torch.autograd.grad(f(xg), [xg], grad_outputs=z)

    from torch.func import vjp as func_vjp

    _, pullback = func_vjp(f, x)
    (u_func,) = pullback(z)

    assert torch.allclose(u_auto, u_func, atol=1e-10, rtol=0)


def test_vjp_matches_a_finite_difference_in_float64():
    """The linearisation check, at a precision where it is actually valid.

    The same check fails in bfloat16 on real models -- measured on the W7900,
    the bf16 VJP agrees with the fp32 VJP to cosine 0.999 while the bf16 finite
    difference is wrong by 56 to 109 percent and flips sign as the step
    shrinks.  That is the difference of two large logit sums falling below the
    format's resolution, not a bad gradient, which is why the portability probe
    refuses to certify outside float32.
    """
    torch.manual_seed(11)
    D, V = 5, 4
    w = torch.randn(D, V, dtype=torch.float64)

    def f(x):
        return torch.tanh(x @ w)

    x = torch.randn(1, D, dtype=torch.float64)
    z = torch.randn(1, V, dtype=torch.float64)
    d = torch.randn(1, D, dtype=torch.float64) * 1e-4

    xg = x.clone().requires_grad_(True)
    (u,) = torch.autograd.grad(f(xg), [xg], grad_outputs=z)
    predicted = float((u * d).sum())
    fd = float(((f(x + d) * z).sum() - ((f(x - d) * z).sum())) / 2.0)
    assert abs(fd - predicted) / abs(fd) < 1e-6


def test_probe_count_stability_is_measurable():
    """4-vs-8-probe ordering agreement, the shape of the Stage 0 stability rule."""
    from research.jacobian_kv.evaluate import kendall_distance

    torch.manual_seed(12)
    D, V = 6, 5
    W = torch.randn(D, V, dtype=torch.float64)
    logits_row = torch.tensor([0.3, -0.8, 1.2, 0.1, -0.2])
    errors = [torch.randn(D, dtype=torch.float64) * (1.0 + 0.4 * i) for i in range(20)]

    def scores(n_probes, seed):
        u = _many_probes(logits_row, n_probes, seed=seed).double() @ W.T
        return [float((u @ e).pow(2).mean()) for e in errors]

    s_small = scores(4, 13)
    s_large = scores(4096, 13)
    assert not math.isnan(kendall_distance(s_small, s_large))
    # many probes must order the same errors more stably than four do
    a = kendall_distance(scores(4096, 14), s_large)
    b = kendall_distance(scores(4, 15), s_large)
    assert a < b
