# SPDX-License-Identifier: MIT
"""Independent converged FP64 Sinkhorn reference and VJP qualification."""

import copy

import torch

from research.subliminal_learning.connections import sinkhorn_knopp_log


class ConvergedSinkhorn(torch.autograd.Function):
    """Implicit VJP at the converged projection, independent of loop length.

    The row/column multipliers solve the normalization constraints. A
    Fix the final column multiplier to zero to remove the additive gauge.
    """

    @staticmethod
    def forward(ctx, logits):
        la = logits.clone()
        for i in range(50000):
            la -= torch.logsumexp(la, -2, keepdim=True)
            la -= torch.logsumexp(la, -1, keepdim=True)
            if i % 20 == 19:
                p = la.exp()
                if (
                    max((p.sum(-1) - 1).abs().max(), (p.sum(-2) - 1).abs().max())
                    <= 1e-12
                ):
                    ctx.save_for_backward(p)
                    return p
        raise RuntimeError("FP64 reference failed to converge")

    @staticmethod
    def backward(ctx, cotangent):
        (p,) = ctx.saved_tensors
        n = p.shape[-1]
        eye = torch.eye(n, dtype=p.dtype, device=p.device).expand(p.shape)
        system = torch.cat(
            (torch.cat((eye, p), -1), torch.cat((p.transpose(-1, -2), eye), -1)), -2
        )
        weighted = p * cotangent
        rhs = torch.cat((weighted.sum(-1), weighted.sum(-2)), -1)
        reduced = torch.linalg.solve(
            system[..., :-1, :-1], rhs[..., :-1, None]
        ).squeeze(-1)
        solution = torch.cat((reduced, torch.zeros_like(reduced[..., :1])), -1)
        return weighted - p * (solution[..., :n, None] + solution[..., None, n:])


def relative_error(ours, reference):
    delta = (ours.double() - reference).norm().item()
    norm = reference.norm().item()
    return {
        "reference_norm": norm,
        "absolute_error": delta,
        "relative_error": delta / norm if norm >= 1e-8 else None,
        "pass": delta / norm <= 1e-3 if norm >= 1e-8 else delta <= 1e-6,
    }


def map_reference(conn, h):
    m, b, n, d = h.shape
    x = h.reshape(m, b, n * d)
    x = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + conn.eps)
        * conn.rms_weight[:, None]
    )
    z = x @ conn.phi
    pre = (conn.alpha_pre[:, None] * z[..., :n] + conn.b_pre[:, None]).sigmoid()
    post = (
        2
        * (
            conn.alpha_post[:, None] * z[..., n : 2 * n] + conn.b_post[:, None]
        ).sigmoid()
    )
    logits = (
        conn.alpha_res.view(m, 1, 1, 1) * z[..., 2 * n :].view(m, b, n, n)
        + conn.b_res[:, None]
    )
    return pre, post, ConvergedSinkhorn.apply(logits)


def qualify_reference(iters, device):
    from research.subliminal_learning.connections import ManifoldHyperConnection

    g = torch.Generator().manual_seed(75091)
    cases, inputs = [], {}
    # Include a genuine sharp, competing-row matrix, not only near-identity
    # or symmetric matrices on which finite Sinkhorn loops look exact.
    sharp = torch.tensor(
        [
            [6.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ],
        dtype=torch.double,
    )
    for name, logits in (
        ("moderate", torch.randn(3, 4, 4, generator=g, dtype=torch.double)),
        ("sharp_random", torch.randn(3, 4, 4, generator=g, dtype=torch.double) * 4),
        ("sharp_competing_rows", sharp[None]),
    ):
        cot = torch.randn(logits.shape, generator=g, dtype=torch.double)
        ref_x = logits.clone().requires_grad_()
        ref = ConvergedSinkhorn.apply(ref_x)
        ref_vjp = torch.autograd.grad((ref * cot).sum(), ref_x)[0]
        x = logits.float().to(device).requires_grad_()
        p = sinkhorn_knopp_log(x, iters)
        vjp = torch.autograd.grad((p * cot.to(x)).sum(), x)[0]
        residual = max(
            (p.sum(-1) - 1).abs().max().item(), (p.sum(-2) - 1).abs().max().item()
        )
        ref_residual = max(
            (ref.sum(-1) - 1).abs().max().item(), (ref.sum(-2) - 1).abs().max().item()
        )
        error = relative_error(vjp.cpu(), ref_vjp)
        cases.append(
            {
                "case": name,
                "row_column_residual": residual,
                "reference_residual": ref_residual,
                "map_max_error": (p.cpu().double() - ref.detach()).abs().max().item(),
                "vjp": error,
                "pass": residual <= 1e-4 and ref_residual <= 1e-10 and error["pass"],
            }
        )
        inputs[name] = {
            "logits": logits,
            "cotangent": cot,
            "reference": ref.detach(),
            "reference_vjp": ref_vjp,
            "actual": p.detach().cpu(),
            "actual_vjp": vjp.cpu(),
        }
    for name in ("equal_streams", "unequal_streams", "sharp_maps"):
        conn = ManifoldHyperConnection(
            1, 4, 8, 1, iters, 1.0, 1e-5, log_domain=True
        ).double()
        with torch.no_grad():
            conn.phi.copy_(
                torch.randn(conn.phi.shape, generator=g, dtype=torch.double) * 0.3
            )
            if name == "sharp_maps":
                conn.b_res.copy_(sharp[None])
                conn.alpha_res.fill_(0.1)
        h = torch.randn(1, 3, 4, 8, generator=g, dtype=torch.double)
        if name == "equal_streams":
            h = h[:, :, :1].expand_as(h).clone()
        ref_h = h.clone().requires_grad_()
        ref = map_reference(conn, ref_h)
        cot = [torch.randn(y.shape, generator=g, dtype=torch.double) for y in ref]
        ref_grad = torch.autograd.grad(
            sum((a * b).sum() for a, b in zip(ref, cot)),
            [ref_h] + list(conn.parameters()),
        )
        ours = copy.deepcopy(conn).float().to(device)
        ours_h = h.float().to(device).requires_grad_()
        actual = ours.maps(ours_h)
        grad = torch.autograd.grad(
            sum((a * b.to(a)).sum() for a, b in zip(actual, cot)),
            [ours_h] + list(ours.parameters()),
        )
        errors = {
            key: relative_error(a.cpu(), b)
            for key, a, b in zip(
                ["h"] + [n for n, _ in conn.named_parameters()], grad, ref_grad
            )
        }
        p, rp = actual[2], ref[2]
        residual = max(
            (p.sum(-1) - 1).abs().max().item(), (p.sum(-2) - 1).abs().max().item()
        )
        rr = max(
            (rp.sum(-1) - 1).abs().max().item(), (rp.sum(-2) - 1).abs().max().item()
        )
        cases.append(
            {
                "case": name,
                "row_column_residual": residual,
                "reference_residual": rr,
                "vjp": errors,
                "pass": residual <= 1e-4
                and rr <= 1e-10
                and all(v["pass"] for v in errors.values()),
            }
        )
        inputs[name] = {
            "h": h,
            "state": conn.state_dict(),
            "cotangent": cot,
            "reference": [v.detach() for v in ref],
            "reference_vjp": ref_grad,
            "actual": [v.detach().cpu() for v in actual],
            "actual_vjp": [v.cpu() for v in grad],
        }
    return {
        "iters": iters,
        "cases": cases,
        "pass": all(c["pass"] for c in cases),
    }, inputs
