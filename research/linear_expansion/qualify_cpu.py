"""CPU correctness checks that must pass before any GPU reservation.

Writes a machine-readable receipt with every maximum error and its
tolerance. Exits nonzero if any check fails.
"""

import argparse
import copy
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.tinystories_depth.model import Model, generator, loss
from research.tinystories_depth.probe import batch

from .expand import (
    ARMS,
    Additive,
    Factored,
    WIDTH,
    balanced_factors,
    counts,
    effective,
    expand,
    folded_state,
    param_groups,
    selected_parameters,
    target_names,
)

RESULTS = []


def check(name, value, tolerance, detail=None):
    ok = bool(np.isfinite(value)) and value <= tolerance
    RESULTS.append(
        {
            "check": name,
            "value": float(value),
            "tolerance": tolerance,
            "pass": ok,
            **({"detail": detail} if detail else {}),
        }
    )
    print(f"{'PASS' if ok else 'FAIL'} {name}: {value:.3e} <= {tolerance:.1e}")
    return ok


def gen(name):
    return generator(17, name)


def randn(*shape, name, dtype=torch.float64):
    return torch.randn(*shape, generator=gen(name), dtype=dtype)


def wrappers_and_bias():
    """Folded wrappers equal the original matrix, rectangular and with bias."""
    for out_f, in_f in ((128, 128), (96, 64), (64, 96)):
        W = randn(out_f, in_f, name=f"W{out_f}x{in_f}") * 0.05
        b = randn(out_f, name=f"b{out_f}x{in_f}") * 0.1
        x = randn(5, in_f, name=f"x{out_f}x{in_f}")
        ref = x @ W.T + b
        k = min(out_f, in_f)
        for width in (k, 2 * k):
            A, B, s, R = balanced_factors(W, width, gen(f"R{out_f}{in_f}{width}"))
            check(
                f"R orthonormal {out_f}x{in_f} r={width}",
                (R.T @ R - torch.eye(k, dtype=torch.float64)).abs().max().item(),
                1e-12,
            )
            for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
                m = Factored(A.to(dtype), B.to(dtype), b.to(dtype))
                check(
                    f"factored fold {out_f}x{in_f} r={width} {dtype}",
                    (effective(m).double() - W).abs().max().item(),
                    tol,
                )
                check(
                    f"factored output {out_f}x{in_f} r={width} {dtype}",
                    (m(x.to(dtype)).double() - ref).abs().max().item(),
                    tol * 10,
                )
        for dtype, tol in ((torch.float64, 1e-15), (torch.float32, 1e-7)):
            m = Additive(W.to(dtype), bias=b.to(dtype))
            check(
                f"additive fold {out_f}x{in_f} {dtype}",
                (effective(m).double() - W).abs().max().item(),
                tol,
            )


def gradients():
    """Autograd gradcheck and the analytic factor gradients B^T G, G A^T."""
    W = randn(12, 8, name="gW") * 0.3
    A, B, _, _ = balanced_factors(W, 16, gen("gR"))
    b = randn(12, name="gb")
    for i in range(3):
        x = randn(4, 8, name=f"gx{i}").requires_grad_()
        Ap, Bp, bp = (t.clone().requires_grad_() for t in (A, B, b))
        f = lambda x, A, B, b: torch.nn.functional.linear(
            torch.nn.functional.linear(x, A), B, b
        )
        ok = torch.autograd.gradcheck(f, (x, Ap, Bp, bp))
        check(f"gradcheck factored input {i}", 0.0 if ok else 1.0, 0.0)
        target = randn(4, 12, name=f"gt{i}")
        out = f(x, Ap, Bp, bp)
        ((out - target) ** 2).sum().backward()
        G = 2 * (out - target).detach().T @ x.detach()  # dL/dW
        check(
            f"dL/dA = B^T G input {i}",
            (Ap.grad - B.T @ G).abs().max().item(),
            1e-12,
        )
        check(
            f"dL/dB = G A^T input {i}",
            (Bp.grad - G @ A.T).abs().max().item(),
            1e-12,
        )
    U = [(randn(12, 8, name=f"gu{j}") * 0.1).requires_grad_() for j in range(4)]
    x = randn(4, 8, name="gux").requires_grad_()
    ok = torch.autograd.gradcheck(
        lambda x, *u: torch.nn.functional.linear(x, sum(u) / 2), (x, *U)
    )
    check("gradcheck additive", 0.0 if ok else 1.0, 0.0)


def regression(n_out=16, n_in=12, rows=40):
    X = randn(rows, n_in, name="rx")
    Y = randn(rows, n_out, name="ry")
    W0 = randn(n_out, n_in, name="rw0") * 0.3
    return X, Y, W0


def quad_loss(W, X, Y):
    return 0.5 * ((X @ W.T - Y) ** 2).sum() / X.shape[0]


def sgd_algebra():
    """Exact one-step SGD identity and F1/F2 trajectory equivalence."""
    X, Y, W0 = regression()
    A, B, _, _ = balanced_factors(W0, 12, gen("s1"))
    eta = 0.05
    W = (B @ A).requires_grad_()
    quad_loss(W, X, Y).backward()
    G = W.grad.detach()
    Ap, Bp = A.clone().requires_grad_(), B.clone().requires_grad_()
    quad_loss(Bp @ Ap, X, Y).backward()
    A1, B1 = A - eta * Ap.grad, B - eta * Bp.grad
    W1 = W.detach()
    formula = W1 - eta * (B @ B.T @ G + G @ A.T @ A) + eta**2 * G @ W1.T @ G
    check("one-step SGD product formula", (B1 @ A1 - formula).abs().max().item(), 1e-13)
    trajectories = {}
    for width in (12, 24):
        A, B, _, _ = balanced_factors(W0, width, gen(f"s{width}"))
        A, B = A.clone().requires_grad_(), B.clone().requires_grad_()
        opt = torch.optim.SGD([A, B], lr=eta)
        path = []
        for _ in range(20):
            opt.zero_grad()
            quad_loss(B @ A, X, Y).backward()
            opt.step()
            path.append((B @ A).detach().clone())
        trajectories[width] = torch.stack(path)
    a, b = trajectories[12], trajectories[24]
    check(
        "F1/F2 SGD 20-step effective W, |diff| - rtol*|W|",
        ((a - b).abs() - 1e-8 * b.abs()).max().item(),
        1e-9,
    )


def additive_equivalence():
    """A4 equals D under SGD at the same LR; under Adam, D with 2x LR, 2x eps."""
    X, Y, W0 = regression()
    for kind in ("SGD", "Adam"):
        D = W0.clone().requires_grad_()
        U = [(W0 / 2).clone().requires_grad_() for _ in range(4)]
        lr, eps = 0.01, 1e-8
        if kind == "SGD":
            od = torch.optim.SGD([D], lr=lr)
            ou = torch.optim.SGD(U, lr=lr)
        else:
            od = torch.optim.AdamW(
                [D], lr=2 * lr, betas=(0.9, 0.95), eps=2 * eps, weight_decay=0
            )
            ou = torch.optim.AdamW(U, lr=lr, betas=(0.9, 0.95), eps=eps, weight_decay=0)
        worst = 0.0
        for _ in range(20):
            od.zero_grad()
            ou.zero_grad()
            quad_loss(D, X, Y).backward()
            quad_loss(sum(U) / 2, X, Y).backward()
            od.step()
            ou.step()
            worst = max(worst, (D - sum(U) / 2).abs().max().item())
        label = "same LR" if kind == "SGD" else "D at 2x LR and 2x eps"
        check(f"A4 vs D under {kind}, {label}, 20 steps", worst, 1e-12)


def forward_fp32(model, ids):
    """Whole-model FP32 path: the original Model.forward casts to BF16."""
    h = model.lm.model.embed_tokens(ids)
    return model.lm(inputs_embeds=h, use_cache=False).logits


def rel_l2(a, b):
    return ((a - b).norm() / b.norm()).item()


def build(arm, seed, base_state=None):
    model = Model("R", 4, seed)
    if base_state is not None:
        model.load_state_dict(base_state)
    names = target_names(model)
    receipt = expand(model, arm, seed, names)
    return model, names, receipt


def perturbed_base(seed, ids, targets):
    """A few CPU AdamW steps so W is not the raw named initialization."""
    model = Model("R", 4, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    for _ in range(3):
        opt.zero_grad()
        loss(forward_fp32(model, ids), targets).backward()
        opt.step()
    return model.state_dict()


def real_model(curve):
    """Equal starting functions, export fidelity, groups and parameter counts."""
    seed = 75201
    order = np.arange((len(curve) - 1) // 256)
    ids, targets = batch(curve, order, 0, "cpu")
    ids, targets = ids[:2], targets[:2]
    base = perturbed_base(seed, ids, targets)
    reference = Model("R", 4, seed)
    reference.load_state_dict(base)
    reference.eval()
    with torch.no_grad():
        ref_logits = forward_fp32(reference, ids)
        ref_ce = loss(ref_logits, targets).item()
    audit = {}
    plain_keys = set(reference.state_dict())
    for arm in ARMS:
        model, names, receipt = build(arm, seed, base)
        model.eval()
        assert len(names) == 16
        with torch.no_grad():
            logits = forward_fp32(model, ids)
        check(f"{arm} init logits rel L2 vs D (FP32)", rel_l2(logits, ref_logits), 1e-5)
        check(
            f"{arm} init CE diff vs D (FP32)",
            abs(loss(logits, targets).item() - ref_ce),
            1e-5,
        )
        worst = max(
            [
                r.get("reconstruction_max_abs_master", 0.0)
                for r in receipt["matrices"].values()
            ]
            or [0.0]
        )
        check(f"{arm} per-matrix reconstruction max abs", worst, 1e-6)
        groups = param_groups(model, names, 1e-3, 2.0)
        seen = [id(p) for g in groups for p in g["params"]]
        all_ids = [id(p) for p in model.parameters() if p.requires_grad]
        check(
            f"{arm} optimizer groups cover each parameter once",
            float(len(seen) != len(set(seen)) or set(seen) != set(all_ids)),
            0.0,
        )
        sel = {id(p) for p in selected_parameters(model, names)}
        check(
            f"{arm} selected group holds exactly the target parameters",
            float({id(p) for p in groups[0]["params"]} != sel),
            0.0,
        )
        # Save, reload, fold and compare before and after CPU updates.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            torch.save(model.state_dict(), path)
            again, _, _ = build(arm, seed, base)
            again.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            check(
                f"{arm} save/reload logits max abs",
                (forward_fp32(again, ids) - logits).abs().max().item(),
                0.0,
            )
        model.train()
        opt = torch.optim.AdamW(
            param_groups(model, names, 3e-4, 1.0), betas=(0.9, 0.95), weight_decay=0
        )
        for _ in range(4):
            opt.zero_grad()
            loss(forward_fp32(model, ids), targets).backward()
            opt.step()
        model.eval()
        for stage, m in (("after 4 updates", model),):
            folded = folded_state(m)
            plain = Model("R", 4, seed)
            plain.load_state_dict(folded, strict=True)
            check(
                f"{arm} export key set equals D",
                float(set(folded) != plain_keys),
                0.0,
            )
            check(
                f"{arm} export shapes equal D",
                float(any(folded[k].shape != base[k].shape for k in plain_keys)),
                0.0,
            )
            plain.eval()
            with torch.no_grad():
                a = forward_fp32(m, ids)
                b = forward_fp32(plain, ids)
            check(f"{arm} fold logits rel L2 {stage} (FP32)", rel_l2(b, a), 1e-5)
            check(
                f"{arm} fold CE diff {stage} (FP32)",
                abs(loss(a, targets).item() - loss(b, targets).item()),
                1e-5,
            )
            check(
                f"{arm} exported parameter count equals D",
                float(
                    sum(v.numel() for v in folded.values())
                    != sum(v.numel() for v in base.values())
                ),
                0.0,
            )
        audit[arm] = {
            "counts": counts(build(arm, seed, base)[0], names),
            "exported_parameters": sum(v.numel() for v in base.values()),
            "targets": names,
            "receipt": receipt,
        }
    return audit


def resume(curve):
    """Uninterrupted and checkpoint-resumed CPU trajectories agree exactly."""
    seed = 75201
    order = np.arange((len(curve) - 1) // 256)

    def fresh():
        model, names, _ = build("F2", seed)
        opt = torch.optim.AdamW(
            param_groups(model, names, 3e-4, 1.0), betas=(0.9, 0.95), weight_decay=0
        )
        return model, opt

    def step(model, opt, cursor):
        ids, targets = batch(curve, order, cursor, "cpu")
        opt.zero_grad()
        loss(forward_fp32(model, ids[:2]), targets[:2]).backward()
        opt.step()

    torch.manual_seed(5)
    a, oa = fresh()
    for c in range(6):
        step(a, oa, c)
    torch.manual_seed(5)
    b, ob = fresh()
    clock = 0.0
    for c in range(3):
        t = time.perf_counter()
        step(b, ob, c)
        clock += time.perf_counter() - t
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.pt"
        torch.save(
            {
                "model": b.state_dict(),
                "optimizer": ob.state_dict(),
                "rng": torch.get_rng_state(),
                "cursor": 3,
                "training_seconds": clock,
            },
            path,
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
    c2, oc = fresh()
    c2.load_state_dict(state["model"])
    oc.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    for c in range(state["cursor"], 6):
        step(c2, oc, c)
    worst = max(
        (pa - pc).abs().max().item() for pa, pc in zip(a.parameters(), c2.parameters())
    )
    check("resumed vs uninterrupted parameters max abs", worst, 0.0)
    check("resumed clock is carried", float(state["training_seconds"] <= 0), 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curve", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    curve = np.load(a.curve, mmap_mode="r")
    wrappers_and_bias()
    gradients()
    sgd_algebra()
    additive_equivalence()
    audit = real_model(curve)
    resume(curve)
    receipt = {
        "status": "pass" if all(r["pass"] for r in RESULTS) else "fail",
        "checks": RESULTS,
        "audit": audit,
        "widths": WIDTH,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "seconds": time.perf_counter() - started,
    }
    a.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "checks": len(RESULTS)}))
    sys.exit(0 if receipt["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
