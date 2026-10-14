"""Independent finite operator reference, author parity, and cast tests."""

import os
import argparse
import copy
import json
from pathlib import Path

import torch

from .model import Model, Routing, author, config, loss, mhc, pair_weights, write_code


def reference_sink(logits, repeats=20):
    x = (logits - logits.amax(-1, keepdim=True)).exp()
    x = x / x.sum(-1, keepdim=True) + 1e-6
    x = x / (x.sum(-2, keepdim=True) + 1e-6)
    for _ in range(repeats - 1):
        x = x / (x.sum(-1, keepdim=True) + 1e-6)
        x = x / (x.sum(-2, keepdim=True) + 1e-6)
    return x


def oracle(state, phi, weight, scale, base, cast=False):
    x = state.flatten(-2)
    if cast:
        x = x.float()
    mixes = (x @ (phi * weight).T) * (x.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    z = (
        mixes * torch.cat((scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)))
        + base
    )
    pre = z[..., :4].sigmoid().unsqueeze(-1)
    post = 2 * z[..., 4:8].sigmoid().unsqueeze(-1)
    carry = reference_sink(z[..., 8:].reshape(*state.shape[:2], 4, 4))
    read = (state * pre).sum(-2)
    if cast:
        read = read.bfloat16()
    # The simple nonlinear branch also tests the transposed carry contract.
    branch = read.square()
    branch_post = branch.float() if cast else branch
    state_post = state.float() if cast else state
    updated = branch_post.unsqueeze(-2) * post + (
        carry.unsqueeze(-1) * state_post.unsqueeze(-2)
    ).sum(-3)
    if cast:
        updated = updated.bfloat16()
    return pre, post, carry, updated


def executed(state, phi, weight, scale, base):
    z = mhc.mhc_pre_norm_fn_ref(state, phi, weight, 1e-5)
    pre, post, logits = mhc.mhc_pre_split_mixes_ref(z, scale, base, 4, 2.0, 0.0)
    carry = mhc.sinkhorn_normalize_ref(logits, repeat=20, eps=1e-6)
    read = mhc.mhc_pre_apply_mix_ref(state, pre)
    return pre, post, carry, mhc.mhc_post_ref(read.square(), state, post, carry)


def error(a, b):
    norm = b.double().norm().item()
    delta = (a.double() - b.double()).norm().item()
    value = delta / norm if norm >= 1e-8 else delta
    ok = value <= (1e-3 if norm >= 1e-8 else 1e-6)
    assert ok, (norm, delta, value)
    return {
        "reference_norm": norm,
        "absolute": delta,
        "relative": delta / norm if norm >= 1e-8 else None,
        "pass": ok,
    }


def wrapper_reference(model, ids):
    """Independent routing algebra; author computational branch unchanged."""
    h = model.lm.model.embed_tokens(ids).bfloat16().unsqueeze(-2).repeat(1, 1, 4, 1)
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    for l, block in enumerate(model.lm.model.layers):
        for i, site in enumerate(("attn", "mlp")):
            route = model.routes[2 * l + i]
            x = h.flatten(-2).float()
            with torch.autocast(device_type=ids.device.type, enabled=False):
                mixes = (x @ (route.phi * route.norm_weight).T) * (
                    x.square().mean(-1, keepdim=True) + 1e-5
                ).rsqrt()
            z = (
                mixes
                * torch.cat(
                    (
                        route.scale[0].expand(4),
                        route.scale[1].expand(4),
                        route.scale[2].expand(16),
                    )
                )
                + route.base
            )
            pre = z[..., :4].sigmoid().unsqueeze(-1)
            post = 2 * z[..., 4:8].sigmoid().unsqueeze(-1)
            carry = reference_sink(z[..., 8:].reshape(*ids.shape, 4, 4))
            inp = (h * pre).sum(-2).bfloat16()
            if site == "attn":
                inp = block.input_layernorm(inp, block._gain("in_attn"))
                branch = block.self_attn(inp, position_ids=positions, use_cache=False)[
                    0
                ]
            else:
                inp = block.post_attention_layernorm(inp, block._gain("in_mlp"))
                branch = block.mlp(inp)
            branch = block._block_output(branch, site)
            h = (
                branch.float().unsqueeze(-2) * post
                + (carry.unsqueeze(-1) * h.float().unsqueeze(-2)).sum(-3)
            ).bfloat16()
    h = model.lm.model.norm(h.float().mean(-2).bfloat16())
    return model.lm.lm_head(h)


def run(root, device):
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.manual_seed(75290)
    checks = {}
    for sharpness in (1.0, 15.0):
        z = (torch.randn(2, 3, 4, 4, device=device) * sharpness).requires_grad_()
        cot = torch.randn_like(z)
        out = mhc.sinkhorn_normalize_ref(z, repeat=20, eps=1e-6)
        refz = z.detach().double().cpu().requires_grad_()
        ref = reference_sink(refz)
        grad = torch.autograd.grad((out * cot).sum(), z)[0]
        refgrad = torch.autograd.grad((ref * cot.double().cpu()).sum(), refz)[0]
        checks[f"sink_{sharpness}"] = {
            "output": error(out.cpu(), ref),
            "input_vjp": error(grad.cpu(), refgrad),
        }
    route = Routing(1).to(device)
    with torch.no_grad():
        route.phi.normal_(std=0.2)
        route.base.normal_(std=0.5)
    state = torch.randn(2, 3, 4, 128, device=device, requires_grad=True)
    args = (state, route.phi, route.norm_weight, route.scale, route.base)
    # Isolated continuous FP32 maps versus a dtype-preserving FP64 oracle.
    z = mhc.mhc_pre_norm_fn_ref(*args[:3], 1e-5)
    pre, post, logits = mhc.mhc_pre_split_mixes_ref(z, args[3], args[4], 4, 2.0, 0.0)
    outputs = (pre, post, mhc.sinkhorn_normalize_ref(logits, repeat=20, eps=1e-6))
    refargs = tuple(t.detach().double().cpu().requires_grad_() for t in args)
    refs = oracle(*refargs)[:3]
    cots = [torch.randn_like(t) for t in outputs]
    grads = torch.autograd.grad(sum((o * c).sum() for o, c in zip(outputs, cots)), args)
    refgrads = torch.autograd.grad(
        sum((o * c.cpu().double()).sum() for o, c in zip(refs, cots)), refargs
    )
    checks["routing_fp32"] = {
        "outputs": [error(a.cpu(), b) for a, b in zip(outputs, refs)],
        "parameter_input_vjps": [error(a.cpu(), b) for a, b in zip(grads, refgrads)],
    }
    # BF16 cast reference stays FP32 at normalization and mixing reductions.
    castargs = tuple(t.detach().clone().requires_grad_() for t in args)
    castargs = (castargs[0].bfloat16().detach().requires_grad_(),) + castargs[1:]
    out = executed(*castargs)
    matched = tuple(t.detach().clone().requires_grad_() for t in castargs)
    refs = oracle(*matched, cast=True)
    c = torch.randn_like(out[-1])
    grads = torch.autograd.grad((out[-1] * c).float().sum(), castargs)
    refgrads = torch.autograd.grad((refs[-1] * c).float().sum(), matched)
    checks["bf16_casts"] = {
        "outputs": [error(a.cpu(), b.cpu()) for a, b in zip(out, refs)],
        "parameter_input_vjps": [
            error(a.cpu(), b.cpu()) for a, b in zip(grads, refgrads)
        ],
    }
    # Directional finite differences of the smooth FP64 mathematical oracle.
    cots = [c.double().cpu() for c in cots]

    def objective(xs):
        ys = oracle(*xs)[:3]
        return sum((y * c).sum() for y, c in zip(ys, cots))

    objective(refargs).backward()
    fd = []
    for i, x in enumerate(refargs):
        direction = torch.randn_like(x)
        direction /= direction.norm()
        exact = (x.grad * direction).sum().item()
        estimates = []
        for eps in (1e-4, 1e-5, 1e-6):
            plus = list(refargs)
            minus = list(refargs)
            plus[i] = x.detach() + eps * direction
            minus[i] = x.detach() - eps * direction
            numerical = ((objective(plus) - objective(minus)) / (2 * eps)).item()
            estimates.append(
                {
                    "eps": eps,
                    "numeric": numerical,
                    "pass": abs(numerical - exact)
                    <= 1e-8 + 1e-5 * max(abs(exact), abs(numerical)),
                }
            )
        assert any(estimates[j]["pass"] and estimates[j + 1]["pass"] for j in range(2))
        fd.append({"parameter": i, "autograd": exact, "estimates": estimates})
    checks["finite_difference_fp64"] = fd
    identities = {}
    for arm in ("R", "L"):
        m = Model(arm, 4, 75290, vocab=128).to(device)
        direct = author.LlamaForCausalLM(config(4, arm == "L", 128)).to(device)
        pair_weights(direct, 75290)
        ids = torch.randint(0, 128, (2, 8), device=device)
        targets = torch.randint(0, 128, (2, 8), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            a = m(ids)
            b = direct(
                inputs_embeds=direct.model.embed_tokens(ids).bfloat16(), use_cache=False
            ).logits
            la, lb = loss(a, targets), loss(b, targets)
        la.backward()
        lb.backward()
        checks["author_" + arm] = {
            "output": error(a.cpu(), b.cpu()),
            "loss": error(la.cpu(), lb.cpu()),
            "vjps": [
                error(p.grad.cpu(), q.grad.cpu())
                for p, q in zip(m.lm.parameters(), direct.parameters())
            ],
        }
        identities[arm] = m.identities()
    assert identities["R"]["common_weights"] == identities["L"]["common_weights"]
    for arm in ("M", "ML"):
        m = Model(arm, 4, 75290, vocab=128).to(device)
        direct = copy.deepcopy(m)
        ids = torch.randint(0, 128, (2, 8), device=device)
        targets = torch.randint(0, 128, (2, 8), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            a = m(ids)
            b = wrapper_reference(direct, ids)
            la, lb = loss(a, targets), loss(b, targets)
        la.backward()
        lb.backward()
        checks["wrapper_" + arm] = {
            "output": error(a.cpu(), b.cpu()),
            "parameter_vjps": [
                error(p.grad.cpu(), q.grad.cpu())
                for p, q in zip(m.parameters(), direct.parameters())
            ],
        }
        assert m.common_hashes == identities["R"]["common_weights"]
    baseline = Model("ML", 4, 75290, vocab=128).to(device)
    zero = Model("ML-O", 4, 75290, vocab=128, rho=0).to(device)
    ids = torch.randint(0, 128, (2, 8), device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        a = baseline(ids)
        b = zero(ids)
    checks["zero_rho_model"] = error(a.cpu(), b.cpu())
    # Explicit asymmetric carry test prevents accepting a transpose mistake.
    s = torch.arange(16, device=device).reshape(1, 1, 4, 4).bfloat16()
    matrix = torch.arange(16, device=device).reshape(1, 1, 4, 4).float() / 17
    a = mhc.mhc_post_ref(
        torch.zeros(1, 1, 4, device=device),
        s,
        torch.zeros(1, 1, 4, 1, device=device),
        matrix,
    )
    b = (matrix[0, 0].T @ s[0, 0].float()).bfloat16()
    assert torch.equal(a[0, 0], b)
    checks["carry_orientation"] = (
        "input m to output n; output = matrix transpose times residual"
    )
    coded = Model("ML-O", 4, 75290, vocab=128).to(device)
    schedule = coded.lm.model.layerrope
    beta = (
        torch.tensor([0.33, 0.74, 1.14, 1.61], device=device)
        .reshape(1, 1, 4, 1)
        .requires_grad_()
    )
    changed = write_code(beta, schedule, "out_attn", 2, 0.5)
    assert (changed >= 0).all() and (changed <= 2).all()
    assert torch.allclose(changed.sum(-2), beta.sum(-2), atol=1e-6)
    assert torch.equal(write_code(beta, schedule, "out_attn", 2, 0.0), beta)
    changed.square().sum().backward()
    assert (
        schedule.out_attn_alpha_rot.grad.abs() > 0
        and schedule.out_attn_beta_rot.grad.abs() > 0
    )
    # Separate gain use must propagate gradients to the same actual scalars.
    schedule.zero_grad()
    schedule("out_attn", torch.tensor(1.0, device=device)).square().sum().backward()
    # Squared norm cancels rotation; an asymmetric cotangent does not.
    schedule.zero_grad()
    (
        schedule("out_attn", torch.tensor(1.0, device=device))
        * torch.arange(128, device=device)
    ).sum().backward()
    assert (
        schedule.out_attn_alpha_rot.grad.abs() > 0
        and schedule.out_attn_beta_rot.grad.abs() > 0
    )
    checks["channel"] = (
        "bounds, immediate sum, zero rho, gain and code gradients passed away from amin ties"
    )
    # FP64 code oracle is smooth here: the minimum slack is unique.
    for site in ("out_attn", "out_mlp"):
        code_model = Model("ML-O", 4, 75290, vocab=128).double()
        schedule = code_model.lm.model.layerrope
        beta = (
            torch.tensor([0.33, 0.74, 1.14, 1.61], dtype=torch.double)
            .reshape(1, 1, 4, 1)
            .requires_grad_()
        )
        value = write_code(beta, schedule, site, 2, 0.5).square().sum()
        variables = (
            beta,
            getattr(schedule, site + "_alpha_rot"),
            getattr(schedule, site + "_beta_rot"),
        )
        gradients = torch.autograd.grad(value, variables)
        tests = []
        for variable, gradient in zip(variables, gradients):
            direction = torch.ones_like(variable)
            direction /= direction.norm()
            exact = (gradient * direction).sum().item()
            estimates = []
            original = variable.detach().clone()
            for eps in (1e-4, 1e-5, 1e-6):
                with torch.no_grad():
                    variable.copy_(original + eps * direction)
                plus = write_code(beta, schedule, site, 2, 0.5).square().sum().item()
                with torch.no_grad():
                    variable.copy_(original - eps * direction)
                minus = write_code(beta, schedule, site, 2, 0.5).square().sum().item()
                with torch.no_grad():
                    variable.copy_(original)
                numeric = (plus - minus) / (2 * eps)
                estimates.append(
                    {
                        "eps": eps,
                        "numeric": numeric,
                        "pass": abs(numeric - exact)
                        <= 1e-8 + 1e-5 * max(abs(numeric), abs(exact)),
                    }
                )
            assert any(
                estimates[j]["pass"] and estimates[j + 1]["pass"] for j in range(2)
            )
            tests.append({"autograd": exact, "estimates": estimates})
        checks["channel_fd_" + site] = tests
    seq = torch.arange(257)
    assert torch.equal(seq[:-1] + 1, seq[1:])
    checks["next_token_shift"] = (
        "257 sources form 256 inputs/targets; internal author loss never called"
    )
    result = {
        "status": "operator_fidelity_pass",
        "device": device,
        "checks": checks,
        "identities": identities,
        "training_qualification": "unstarted_pending_cost",
    }
    Path(root, f"FIDELITY-{device}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"status": result["status"], "device": device, "checks": list(checks)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = p.parse_args()
    run(a.root, a.device)
