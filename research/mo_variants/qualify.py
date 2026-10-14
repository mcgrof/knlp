"""Independent references for the finite depth-code matrix, without training."""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch

from research.tinystories_depth.model import Model as Baseline
from research.tinystories_depth.model import generator, mhc
from research.tinystories_depth.qualify import error, reference_sink
from .preflight import clients, write
from .model import Model


def code_parameters(model):
    return {
        name: p
        for name, p in model.named_parameters()
        if name.startswith("angular.")
        or name
        in (
            "strength_logits",
            "normalized_frequency_logits",
            "normalized_offsets",
            "free_angles",
        )
        or name.startswith("lm.model.layerrope.out_")
        and name.endswith("_rot")
    }


def reference_code(model, layer, site, theta, dtype, device):
    """Separate equations; never call the production phase/code/strength methods."""
    v = model.variant
    index = model.code_map[site][layer]
    u = index / (model.depth - 1)
    number = lambda x: torch.tensor(x, device=device, dtype=dtype)
    if v == "V05":
        angle = model.fixed_angles[site, index].to(device=device, dtype=dtype)
    elif v in ("V17", "V18", "V19"):
        angle = number(2 * math.pi * {"V17": 0.5, "V18": 1.0, "V19": 2.0}[v] * u)
    elif v == "V20":
        angle = (
            4 * math.pi * theta["normalized_frequency_logits"][site].sigmoid() * u
            + theta["normalized_offsets"][site]
        )
    elif v == "V21":
        angle = number(2 * math.pi * u * u)
    elif v == "V24":
        angle = theta["free_angles"][site, index]
    elif v in ("V22", "V23"):
        angle = None
    else:
        selected = "out_attn" if v in ("V15", "V16") or site == 0 else "out_mlp"
        prefix = "lm.model.layerrope." if v == "V31" else "angular."
        angle = torch.exp(
            theta[prefix + selected + "_alpha_rot"]
            + theta[prefix + selected + "_beta_rot"] * math.log(index + 1)
        )
        if v == "V16" and site == 1:
            angle = angle + math.pi / 2
    if v == "V22":
        p = number(2 * math.pi * u)
        cosine, sine = (p.cos() + (2 * p).cos()) / 2, (p.sin() + (2 * p).sin()) / 2
    elif v == "V23":
        return number(
            ((1, 0, -1, 0), (0, 1, 0, -1), (-1, 0, 1, 0), (0, -1, 0, 1))[index % 4]
        )
    else:
        cosine, sine = angle.cos(), angle.sin()
    return torch.stack((cosine, sine, -cosine, -sine))


def reference_strength(model, layer, site, theta):
    if model.strength_override is not None:
        return model.strength_override
    v = model.variant
    if v == "V11" and site == 1 or v == "V12" and site == 0:
        return 0.0
    if (
        v == "V13"
        and layer >= model.depth // 2
        or v == "V14"
        and layer < model.depth // 2
    ):
        return 0.0
    if v == "V09":
        return 1 / (1 + (-theta["strength_logits"][site]).exp())
    if v == "V10":
        return 0.5 * min(model.step / 500, 1)
    return {"V07": 0.25, "V08": 0.75}.get(v, 0.5)


def reference_route(model, state, args, theta, layer, site, cast=False):
    with torch.autocast(device_type=state.device.type, enabled=False):
        return _reference_route(model, state, args, theta, layer, site, cast)


def _reference_route(model, state, args, theta, layer, site, cast=False):
    phi, weight, scale, base = args
    x = state.flatten(-2).float() if cast else state.flatten(-2)
    mixes = (x @ (phi * weight).T) * (x.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    z = (
        mixes * torch.cat((scale[0].repeat(4), scale[1].repeat(4), scale[2].repeat(16)))
        + base
    )
    read, write_logits = z[..., :4], z[..., 4:8]
    carry_logits = z[..., 8:].reshape(*state.shape[:2], 4, 4)
    v = model.variant
    if v != "V00":
        c = reference_code(model, layer, site, theta, phi.dtype, phi.device)
        rho = reference_strength(model, layer, site, theta)
        if v == "V27":
            write_logits = write_logits + rho * c
        if v in ("V28", "V29"):
            read = read + rho * c
        if v == "V30":
            carry_logits = carry_logits + rho * (c[:, None] * c[None, :]) * 0.5
    pre = read.sigmoid().unsqueeze(-1)
    original = write_logits.sigmoid() * 2
    post = original
    if v not in ("V00", "V27", "V28", "V30"):
        if v == "V25":
            a = torch.minimum(
                torch.minimum(original[..., 0], 2 - original[..., 0]),
                torch.minimum(original[..., 2], 2 - original[..., 2]),
            )
            b = torch.minimum(
                torch.minimum(original[..., 1], 2 - original[..., 1]),
                torch.minimum(original[..., 3], 2 - original[..., 3]),
            )
            perturb = torch.stack((a * c[0], b * c[1], a * c[2], b * c[3]), -1)
        elif v == "V26":
            margins = torch.cat((original, 2 - original), -1)
            good = (margins > 0).all(-1, keepdim=True)
            safe = torch.where(margins > 0, margins, torch.ones_like(margins))
            slack = torch.where(
                good,
                1 / (1 / safe).sum(-1, keepdim=True),
                torch.zeros_like(good, dtype=phi.dtype),
            )
            perturb = slack * c
        else:
            slack = torch.minimum(
                original.min(-1, keepdim=True).values,
                (2 - original).min(-1, keepdim=True).values,
            )
            perturb = slack * c
        post = original + rho * perturb
    if cast:
        # Match the prescribed FP32 primitive at the cast boundary. The
        # continuous FP64 oracle uses independent exp/max/divide above.
        carry = carry_logits.softmax(-1) + 1e-6
        carry = carry / (carry.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            carry = carry / (carry.sum(-1, keepdim=True) + 1e-6)
            carry = carry / (carry.sum(-2, keepdim=True) + 1e-6)
    else:
        carry = reference_sink(carry_logits)
    return pre, post.unsqueeze(-1), carry


def reference_update(state, pre, post, carry, cast=False):
    branch = (state * pre).sum(-2)
    if cast:
        branch = branch.bfloat16()
    branch = branch.square()
    value = branch.float().unsqueeze(-2) * post if cast else branch.unsqueeze(-2) * post
    original = state.float() if cast else state
    value = value + (carry.transpose(-2, -1) @ original)
    return value.bfloat16() if cast else value


def reference_network(model, ids):
    h = model.lm.model.embed_tokens(ids).bfloat16().unsqueeze(-2).repeat(1, 1, 4, 1)
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    theta = code_parameters(model)
    for layer, block in enumerate(model.lm.model.layers):
        for site, name in enumerate(("attn", "mlp")):
            r = model.routes[2 * layer + site]
            pre, post, carry = reference_route(
                model,
                h,
                (r.phi, r.norm_weight, r.scale, r.base),
                theta,
                layer,
                site,
                True,
            )
            inp = (h * pre).sum(-2).bfloat16()
            if site == 0:
                inp = block.input_layernorm(inp, block._gain("in_attn"))
                branch = block.self_attn(inp, position_ids=positions, use_cache=False)[
                    0
                ]
            else:
                inp = block.post_attention_layernorm(inp, block._gain("in_mlp"))
                branch = block.mlp(inp)
            branch = block._block_output(branch, name)
            with torch.autocast(device_type=ids.device.type, enabled=False):
                h = (
                    branch.float().unsqueeze(-2) * post
                    + (carry.transpose(-2, -1) @ h.float())
                ).bfloat16()
    return model.lm.lm_head(model.lm.model.norm(h.float().mean(-2).bfloat16()))


def check_vjps(out, refs, actual_vars, reference_vars):
    cots = [torch.randn_like(o) for o in out]
    grads = torch.autograd.grad(
        sum((a * c).sum() for a, c in zip(out, cots)), actual_vars, allow_unused=True
    )
    refgrads = torch.autograd.grad(
        sum((b * c.to(b)).sum() for b, c in zip(refs, cots)),
        reference_vars,
        allow_unused=True,
    )
    tests = []
    for index, (a, b, var, ref) in enumerate(
        zip(grads, refgrads, actual_vars, reference_vars)
    ):
        a = torch.zeros_like(var) if a is None else a
        b = torch.zeros_like(ref) if b is None else b
        try:
            tests.append(error(a.cpu(), b.cpu()))
        except AssertionError as exc:
            raise AssertionError(
                {"variable_index": index, "shape": list(var.shape), "error": exc.args}
            ) from exc
    return tests


def run(root, device):
    if device == "cuda":
        clients()
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.manual_seed(76390)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"VARIANT-FIDELITY-{device}.json"
    record = {
        "seed": 76390,
        "device": device,
        "status": "running",
        "cases": {},
        "definitions": {},
        "finite_differences": {},
        "baseline_parity": {},
        "zero_strength_parity": {},
        "full_network": {},
    }
    started = time.perf_counter()
    write(path, record)
    try:
        common = None
        for number in range(32):
            v = f"V{number:02}"
            model = Model(v, 4, 76390, vocab=128).to(device)
            model.step = 250
            if common is None:
                common = model.common_hashes
            assert model.common_hashes == common
            record["definitions"][v] = model.identities()
            if v == "V24":
                inherited = Baseline("M-O", 4, 76390, vocab=128).to(device)
                for l in range(4):
                    expected = (
                        inherited.angular.out_attn_alpha_rot
                        + inherited.angular.out_attn_beta_rot * math.log(l + 1)
                    ).exp()
                    assert torch.equal(model.free_angles[0, l], expected)
            if v in ("V02", "V03", "V04"):
                for mapping in model.code_map:
                    if v == "V02":
                        assert sorted(mapping) == list(range(4)) and mapping != list(
                            range(4)
                        )
                    elif v == "V03":
                        assert mapping == [1] * 4
                    else:
                        assert mapping == [3, 2, 1, 0]
            if v == "V05":
                assert not code_parameters(model)
            if v in ("V15", "V16"):
                assert sum(p.numel() for p in code_parameters(model).values()) == 2
            for site in (0, 1):
                r = model.routes[5]
                with torch.no_grad():
                    r.phi.normal_(std=0.002)
                    r.base[:4].copy_(
                        torch.tensor([-0.4, 0.6, -0.7, 0.9], device=device)
                    )
                    r.base[4:8].copy_(
                        torch.logit(
                            torch.tensor([0.33, 0.74, 1.14, 1.61], device=device) / 2
                        )
                    )
                    r.base[8:].normal_(std=0.5)
                state = torch.randn(2, 3, 4, 128, device=device, requires_grad=True)
                theta = code_parameters(model)
                args = (r.phi, r.norm_weight, r.scale, r.base)
                refargs = tuple(
                    p.detach().cpu().double().requires_grad_() for p in args
                )
                refstate = state.detach().cpu().double().requires_grad_()
                reftheta = {
                    n: p.detach().cpu().double().requires_grad_(p.requires_grad)
                    for n, p in theta.items()
                }
                result = model.routing(r, state, 2, site)
                out = (result[0], result[2], result[3])
                refs = reference_route(model, refstate, refargs, reftheta, 2, site)
                actual_vars = (
                    state,
                    *args,
                    *(p for p in theta.values() if p.requires_grad),
                )
                refvars = (
                    refstate,
                    *refargs,
                    *(p for p in reftheta.values() if p.requires_grad),
                )
                key = f"{v}-site{site}"
                record["cases"][key] = {
                    "forward": [error(a.cpu(), b) for a, b in zip(out, refs)],
                    "vjps": check_vjps(out, refs, actual_vars, refvars),
                }
                assert out[1].min() >= 0 and out[1].max() <= 2
                if v not in ("V27",):
                    assert torch.allclose(
                        out[1].sum(-2), result[1].sum(-2), atol=1e-6, rtol=0
                    )
                # Matched BF16 cast function; it is not the continuous FP64 map.
                state = state.detach().bfloat16().requires_grad_()
                a = model.routing(r, state, 2, site)
                out = (
                    a[0],
                    a[2],
                    a[3],
                    mhc.mhc_post_ref(
                        mhc.mhc_pre_apply_mix_ref(state, a[0]).square(),
                        state,
                        a[2],
                        a[3],
                    ),
                )
                matched_args = tuple(p.detach().clone().requires_grad_() for p in args)
                matched_state = state.detach().clone().requires_grad_()
                matched_theta = {
                    n: p.detach().clone().requires_grad_(p.requires_grad)
                    for n, p in theta.items()
                }
                refs = reference_route(
                    model, matched_state, matched_args, matched_theta, 2, site, True
                )
                refs = (*refs, reference_update(matched_state, *refs, True))
                actual_vars = (
                    state,
                    *args,
                    *(p for p in theta.values() if p.requires_grad),
                )
                refvars = (
                    matched_state,
                    *matched_args,
                    *(p for p in matched_theta.values() if p.requires_grad),
                )
                record["cases"][key]["bf16_forward"] = [
                    error(a.cpu(), b.cpu()) for a, b in zip(out, refs)
                ]
                record["cases"][key]["bf16_vjps"] = check_vjps(
                    out, refs, actual_vars, refvars
                )
                if v in ("V09", "V20", "V24", "V25", "V26", "V27", "V28", "V29", "V30"):
                    # FP64 finite differences of the independent, smooth oracle.
                    cots = [
                        torch.randn_like(a)
                        for a in reference_route(
                            model, refstate, refargs, reftheta, 2, site
                        )
                    ]
                    objective = lambda: sum(
                        (a * c).sum()
                        for a, c in zip(
                            reference_route(
                                model, refstate, refargs, reftheta, 2, site
                            ),
                            cots,
                        )
                    )
                    grads = torch.autograd.grad(objective(), refvars, allow_unused=True)
                    tests = []
                    for variable, gradient in zip(refvars, grads):
                        if gradient is None:
                            continue
                        original = variable.detach().clone()
                        direction = torch.ones_like(variable) / math.sqrt(
                            variable.numel()
                        )
                        exact = (gradient * direction).sum().item()
                        estimates = []
                        for eps in (1e-4, 1e-5, 1e-6):
                            with torch.no_grad():
                                variable.copy_(original + eps * direction)
                            plus = objective().item()
                            with torch.no_grad():
                                variable.copy_(original - eps * direction)
                            minus = objective().item()
                            with torch.no_grad():
                                variable.copy_(original)
                            numeric = (plus - minus) / (2 * eps)
                            estimates.append(
                                {
                                    "eps": eps,
                                    "numeric": numeric,
                                    "absolute_error": abs(numeric - exact),
                                }
                            )
                        assert any(
                            abs(t["numeric"] - exact)
                            <= (abs(exact) * 1e-3 if abs(exact) >= 1e-8 else 1e-6)
                            for t in estimates
                        ), (key, exact, estimates)
                        tests.append(
                            {
                                "exact_directional_derivative": exact,
                                "estimates": estimates,
                            }
                        )
                    record["finite_differences"][key] = tests
                write(path, record)
            # Fresh common weights for the exact, executed-baseline comparisons.
            model = Model(v, 4, 76390, vocab=128).to(device)
            model.step = 250
            ids = torch.randint(0, 128, (1, 8), device=device)
            if v in ("V00", "V01", "V31"):
                baseline = Baseline(
                    {"V00": "M", "V01": "M-O", "V31": "ML-O"}[v], 4, 76390, vocab=128
                ).to(device)
                model.load_state_dict(baseline.state_dict(), strict=True)
                with torch.no_grad(), torch.autocast(
                    device_type=device, dtype=torch.bfloat16
                ):
                    actual, expected = model(ids), baseline(ids)
                assert torch.equal(actual, expected)
                record["baseline_parity"][v] = "bit-identical executed forward"
            baseline = Baseline("ML" if v == "V31" else "M", 4, 76390, vocab=128).to(
                device
            )
            model.strength_override = 0.0
            with torch.no_grad(), torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ):
                actual, expected = model(ids), baseline(ids)
            assert torch.equal(actual, expected), (v, "zero strength")
            record["zero_strength_parity"][v] = "bit-identical to M (ML for V31 gains)"
            model.strength_override = None
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                actual = model(ids)
                expected = reference_network(model, ids)
            # Common forward and all trainable parameter VJPs in the full wrapper.
            variables = tuple(p for p in model.parameters() if p.requires_grad)
            record["full_network"][v] = {
                "forward": error(actual.cpu(), expected.cpu()),
                "parameter_vjps": check_vjps(
                    (actual,), (expected,), variables, variables
                ),
            }
            write(path, record)
            print(v, "passed", flush=True)
        # Explicit boundary behavior for the reciprocal and pair slack families.
        from .model import bounded_write

        beta = torch.tensor([0.0, 0.6, 1.2, 2.0], device=device).reshape(1, 1, 4, 1)
        code = torch.tensor([0.6, 0.8, -0.6, -0.8], device=device)
        assert torch.equal(bounded_write(beta, code, 0.5, "reciprocal"), beta)
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write(path, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = p.parse_args()
    run(args.root, args.device)
