"""CPU finite and converged diagnostics; no uniform training claims."""

import os
import argparse
import json
from pathlib import Path

import torch

from .data import sha
from .model import mhc
from .qualify import reference_sink


class DoublyStochastic(torch.autograd.Function):
    """Validated log-space/implicit oracle, reused from the prior audit."""

    count = 0
    method = "log Sinkhorn"

    @staticmethod
    def forward(ctx, logits):
        x = logits.clone()
        for i in range(400):
            x -= torch.logsumexp(x, -2, keepdim=True)
            x -= torch.logsumexp(x, -1, keepdim=True)
            if i % 20 == 19:
                p = x.exp()
                if (
                    max((p.sum(-1) - 1).abs().max(), (p.sum(-2) - 1).abs().max())
                    <= 1e-12
                ):
                    ctx.save_for_backward(p)
                    DoublyStochastic.count = i + 1
                    DoublyStochastic.method = "log Sinkhorn"
                    return p
        # Newton solves the same diagonal-scaling equations in log space.
        # Fix the last column potential to remove the additive gauge.
        n = x.shape[-1]
        for iteration in range(200):
            p = x.exp()
            rows, cols = p.sum(-1), p.sum(-2)
            residual = torch.cat((rows - 1, cols - 1), -1)
            if residual.abs().max() <= 1e-12:
                ctx.save_for_backward(p)
                DoublyStochastic.count = 400 + iteration
                DoublyStochastic.method = "400 log sweeps + damped Newton"
                return p
            system = torch.cat(
                (
                    torch.cat((torch.diag_embed(rows), p), -1),
                    torch.cat((p.transpose(-1, -2), torch.diag_embed(cols)), -1),
                ),
                -2,
            )
            direction = torch.linalg.solve(
                system[..., :-1, :-1], -residual[..., :-1, None]
            ).squeeze(-1)
            direction = torch.cat((direction, torch.zeros_like(direction[..., :1])), -1)
            delta = direction[..., :n, None] + direction[..., None, n:]
            alpha = torch.ones_like(x[..., :1, :1])
            merit = residual.square().sum(-1)
            accepted = merit < 1e-24
            for _ in range(60):
                candidate = x + alpha * delta
                value = candidate.exp()
                rr = torch.cat((value.sum(-1) - 1, value.sum(-2) - 1), -1)
                good = torch.isfinite(rr).all(-1) & (
                    (rr.square().sum(-1) <= merit * (1 - 1e-4 * alpha[..., 0, 0]))
                    | (rr.abs().amax(-1) <= 1e-12)
                )
                accepted = accepted | good
                if accepted.all():
                    break
                alpha = torch.where(accepted[..., None, None], alpha, alpha / 2)
            assert accepted.all(), "Newton reference line search failed"
            x = x + alpha * delta
        raise RuntimeError("CPU reference did not converge after damped Newton")

    @staticmethod
    def backward(ctx, cot):
        (p,) = ctx.saved_tensors
        n = p.shape[-1]
        eye = torch.eye(n, dtype=p.dtype).expand_as(p)
        system = torch.cat(
            (torch.cat((eye, p), -1), torch.cat((p.transpose(-1, -2), eye), -1)), -2
        )
        weighted = p * cot
        rhs = torch.cat((weighted.sum(-1), weighted.sum(-2)), -1)
        v = torch.linalg.solve(system[..., :-1, :-1], rhs[..., :-1, None]).squeeze(-1)
        v = torch.cat((v, torch.zeros_like(v[..., :1])), -1)
        return weighted - p * (v[..., :n, None] + v[..., None, n:])


def limit(logits):
    # The eps-denominator stationary row/column sum is 1-eps, not one.
    # All strictly positive initial entries are diagonal-scaled by the loop.
    initial = logits.softmax(-1) + 1e-6
    return (1 - 1e-6) * DoublyStochastic.apply(initial.log())


def compare(a, b):
    norm = b.double().norm().item()
    delta = (a.double() - b.double()).norm().item()
    return {
        "ref_norm": norm,
        "absolute_l2": delta,
        "relative_l2": delta / norm if norm >= 1e-8 else None,
    }


def diagnose(logits, cot):
    x = logits.detach().float().requires_grad_()
    y = mhc.sinkhorn_normalize_ref(x, repeat=20, eps=1e-6)
    grad = torch.autograd.grad((y * cot.float()).sum(), x)[0]
    z = x.detach().double().requires_grad_()
    same = reference_sink(z)
    sg = torch.autograd.grad((same * cot.double()).sum(), z)[0]
    z = x.detach().double().requires_grad_()
    converged = limit(z)
    cg = torch.autograd.grad((converged * cot.double()).sum(), z)[0]
    n = x.numel() // 16
    return {
        "matrices": n,
        "reference_iters": DoublyStochastic.count,
        "reference_solver": DoublyStochastic.method,
        "row_residual": (y.sum(-1) - 1).abs().max().item(),
        "column_residual": (y.sum(-2) - 1).abs().max().item(),
        "min_entry": y.min().item(),
        "batch_same_count_output": compare(y, same),
        "batch_same_count_vjp": compare(grad, sg),
        "batch_converged_output": compare(y, converged),
        "batch_converged_vjp": compare(grad, cg),
        "per_matrix": [
            {
                "index": i,
                "row_residual": (y.reshape(n, 4, 4)[i].sum(-1) - 1).abs().max().item(),
                "column_residual": (y.reshape(n, 4, 4)[i].sum(-2) - 1)
                .abs()
                .max()
                .item(),
                "same_count_vjp": compare(
                    grad.reshape(n, 4, 4)[i], sg.reshape(n, 4, 4)[i]
                ),
                "same_count_output": compare(
                    y.reshape(n, 4, 4)[i], same.reshape(n, 4, 4)[i]
                ),
                "converged_output": compare(
                    y.reshape(n, 4, 4)[i], converged.reshape(n, 4, 4)[i]
                ),
                "converged_vjp": compare(
                    grad.reshape(n, 4, 4)[i], cg.reshape(n, 4, 4)[i]
                ),
            }
            for i in range(n)
        ],
    }


def stress(root, payload):
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    inputs = torch.load(payload, map_location="cpu", weights_only=True)
    result = {
        "payload": str(payload),
        "sha256": sha(payload),
        "cases": {},
        "policy": "20 upstream repeats; infinite error is diagnostic only",
        "converged_reference": "(1-eps) DS(softmax(logits)+eps); row/column limit=1-eps; implicit VJP includes softmax and epsilon floor",
    }
    for name, case in inputs.items():
        if "logits" in case:
            z, cot = case["logits"].float(), case["cotangent"].float()
        else:
            h = case["h"][0].float().unsqueeze(0)
            s = {k: v.float() for k, v in case["state"].items()}
            mixes = mhc.mhc_pre_norm_fn_ref(h, s["phi"][0].T, s["rms_weight"][0], 1e-5)
            scale = torch.cat(
                (
                    s["alpha_pre"].flatten(),
                    s["alpha_post"].flatten(),
                    s["alpha_res"].flatten(),
                )
            )
            base = torch.cat(
                (s["b_pre"].flatten(), s["b_post"].flatten(), s["b_res"].flatten())
            )
            _, _, z = mhc.mhc_pre_split_mixes_ref(mixes, scale, base, 4, 2.0, 0.0)
            cot = case["cotangent"][2][0].float().unsqueeze(0)
        result["cases"][name] = diagnose(z, cot)
    # Check the stationary-limit oracle against a long finite loop on a
    # moderate input and its implicit derivative via finite differences.
    z = torch.randn(3, 4, 4, dtype=torch.double).requires_grad_()
    a = reference_sink(z, 2000)
    b = limit(z)
    assert (a - b).abs().max() < 1e-11
    cot = torch.randn_like(z)
    g = torch.autograd.grad((b * cot).sum(), z)[0]
    d = torch.randn_like(z)
    d /= d.norm()
    estimates = []
    exact = (g * d).sum().item()
    for eps in (1e-4, 1e-5, 1e-6):
        fd = (
            (limit(z.detach() + eps * d) - limit(z.detach() - eps * d)) * cot
        ).sum().item() / (2 * eps)
        assert abs(fd - exact) < 1e-8 + 1e-5 * max(abs(fd), abs(exact))
        estimates.append({"eps": eps, "autograd": exact, "fd": fd})
    result["limit_oracle_check"] = {
        "long_loop_max_delta": (a - b).abs().max().item(),
        "finite_differences": estimates,
    }
    Path(root, "STRESS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: v for k, v in result["cases"][name].items() if k != "per_matrix"
                }
                for name in result["cases"]
            }
        ),
        flush=True,
    )


def actual(root, paths):
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.manual_seed(75290)
    result = []
    directory = Path(root, "numerics")
    directory.mkdir(exist_ok=True)
    for path in paths:
        p = torch.load(path, map_location="cpu", weights_only=False)
        samples = p["samples"]
        products = torch.eye(4, dtype=torch.double).expand(16, 4, 4).clone()
        records = []
        for s in samples:
            logits = s["logits"]
            cot = torch.randn_like(logits)
            records.append(
                {
                    "sublayer": s["sublayer"],
                    "indices": s["indices"].tolist(),
                    "numerics": diagnose(logits, cot),
                    "mean_energy": s["mean_energy"],
                    "difference_energy": s["difference_energy"],
                    "write_rms": s["write_rms"],
                    "write_l2_change": s["write_l2_change"],
                }
            )
            products = s["carry"][:16].double().transpose(-1, -2) @ products
            records[-1]["carry_singular_values"] = torch.linalg.svdvals(
                s["carry"].double()
            ).tolist()
            records[-1]["product_singular_values_fixed16"] = torch.linalg.svdvals(
                products
            ).tolist()
        record = {
            "path": str(path),
            "sha256": sha(path),
            "run_id": p["run_id"],
            "step": p["step"],
            "records": records,
            "schedule": p["schedule"],
        }
        destination = directory / (path.stem + ".json")
        destination.write_text(json.dumps(record, indent=2) + "\n")
        result.append(
            {
                "path": str(destination),
                "sha256": sha(destination),
                "run_id": p["run_id"],
                "step": p["step"],
                "sublayers": len(records),
            }
        )
    Path(root, "ACTUAL_NUMERICS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"diagnostic_payloads": len(result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stress", type=Path)
    parser.add_argument("--actual", nargs="*", type=Path)
    args = parser.parse_args()
    if args.stress:
        stress(args.root, args.stress)
    if args.actual is not None:
        actual(args.root, args.actual)
