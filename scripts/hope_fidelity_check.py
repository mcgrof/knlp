#!/usr/bin/env python3
"""Check the Hope candidate against explicit double-precision references.

The candidate is `kmccleary3301/nested_learning` (pinned by commit in the
evaluation's source manifest), a community reconstruction of the Hope
architecture from the Nested Learning paper (arXiv 2512.24695). Its
paper-defined block is self-modifying Titans followed by a continuum
memory system (CMS), and the paper-to-code map left four things for a
numerical pass to establish. This battery is that pass, on the CPU in
float64, where speed cannot hide semantic errors.

What is referenced, and how:

    self-modifying core   an explicit re-implementation of the chunked
                          update: keys, values, learning rate and decay
                          produced by the fast memories themselves, the
                          inner gradient of ||M(k) - sg(M(v))||^2 derived
                          BY HAND (no autograd anywhere in the chain),
                          per-token gradients at the chunk-boundary
                          weights, and the sequential application with
                          the rank-1 retention on the input-side matrix.
    CMS write             the gradient-shaping identity: the code trains
                          each level against a detached target built so
                          the gradient equals the teach-signal direction;
                          the reference differentiates a *linearized*
                          loss with that same claimed gradient, then
                          applies a hand-rolled replica of the inner
                          optimizer (EMA momentum with Adam-style
                          preconditioning) and the level clock.
    model level           no reference; the library against itself:
                          streaming versus monolithic logits with
                          teach-driven updates, and a full fast-state
                          suspend/resume through serialized bytes.

Beyond equivalence, the battery settles the map's open questions: which
target the inner loss uses (M(v), not v), whether the four meta
initializations the read path never touches are inert to autograd while
the true function is sensitive to them, and how the per-call partial
flush of the self-modifying core breaks streaming equivalence at
misaligned call boundaries.

    PYTHONPATH=<nested_learning>/src python3 scripts/hope_fidelity_check.py
"""

import argparse
import io
import json
import math
import platform
import subprocess
import sys

import torch
import torch.nn.functional as F

DTYPE = torch.float64

MEM_NAMES = ("k", "v", "q", "eta", "alpha", "memory")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def compare(a, b):
    a = a.detach()
    b = b.detach()
    if a.numel() == 0 and b.numel() == 0:
        return dict(max_abs=0.0, rel_l2=0.0, finite=True)
    denom = b.norm().item()
    return dict(
        max_abs=float((a - b).abs().max()),
        rel_l2=float((a - b).norm() / denom) if denom > 0 else float("nan"),
        finite=bool(torch.isfinite(a).all()),
    )


def compare_dicts(da, db):
    worst = dict(max_abs=0.0, rel_l2=0.0, finite=True)
    assert set(da) == set(db), f"key mismatch: {set(da) ^ set(db)}"
    for name in db:
        r = compare(da[name], db[name])
        worst["max_abs"] = max(worst["max_abs"], r["max_abs"])
        worst["rel_l2"] = max(worst["rel_l2"], r["rel_l2"])
        worst["finite"] = worst["finite"] and r["finite"]
    return worst


def gelu_prime(x):
    # d/dx of the exact (erf) gelu: Phi(x) + x * phi(x)
    phi = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    big_phi = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return big_phi + x * phi


def normalize_ref(x, eps):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def causal_conv_ref(x, conv):
    # depthwise causal conv, left-padded, no bias
    if conv is None:
        return x
    w = conv.weight  # (dim, 1, K)
    kernel = w.shape[-1]
    xp = F.pad(x.transpose(1, 2), (kernel - 1, 0))  # (B, D, T+K-1)
    b_sz, d, tk = xp.shape
    t = tk - (kernel - 1)
    out = torch.zeros(b_sz, d, t, dtype=x.dtype)
    for j in range(kernel):
        out += w[:, 0, j].view(1, -1, 1) * xp[:, :, j : j + t]
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# explicit self-modifying-core reference
# ---------------------------------------------------------------------------
# A memory state is a dict with keys w1 (B, out, hidden), w2 (B, hidden, in),
# optionally w_skip (B, out, in), and momentum buffers m_w1/m_w2/m_w_skip.


def mem_state_from_module(mem_module, batch):
    def expand(w):
        return w.weight.detach().clone().unsqueeze(0).repeat(batch, 1, 1)

    st = {"w1": expand(mem_module.w1), "w2": expand(mem_module.w2)}
    if mem_module.w_skip is not None:
        st["w_skip"] = expand(mem_module.w_skip)
    return st


def mem_forward_ref(st, x):
    # x: (B, T, in) or (B, in); batched weights
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(1)
    hidden = F.gelu(torch.matmul(x, st["w2"].transpose(-1, -2)))
    out = torch.matmul(hidden, st["w1"].transpose(-1, -2))
    if "w_skip" in st:
        out = out + torch.matmul(x, st["w_skip"].transpose(-1, -2))
    elif out.shape[-1] == x.shape[-1]:
        out = out + x
    return out.squeeze(1) if squeeze else out


def mem_weight_vjp(st, x, cot):
    """Hand-derived d<cot, f(x)>/dweights for one token batch.

    x: (B, in), cot: (B, out).  The residual paths (identity or w_skip)
    contribute to the input gradient only, except w_skip's own grad.
    """
    pre = torch.einsum("bd,bhd->bh", x, st["w2"])  # (B, hidden)
    hidden = F.gelu(pre)
    g1 = torch.einsum("bo,bh->boh", cot, hidden)
    dh = torch.einsum("bo,boh->bh", cot, st["w1"])
    dpre = dh * gelu_prime(pre)
    g2 = torch.einsum("bh,bd->bhd", dpre, x)
    gskip = torch.einsum("bo,bd->bod", cot, x) if "w_skip" in st else None
    return g1, g2, gskip


def inner_token_grads(st, k_t, v_t, objective, stopgrad_vhat, target="mv"):
    """Hand-derived gradient of the inner objective for one token.

    l2:  L = ||f(k) - vhat||^2 summed over the output dim,
    dot: L = -<f(k), vhat>, with vhat = f(v), detached when stopgrad.
    target="v" uses the raw value v as the target instead of f(v); that
    variant only typechecks when the memory's output dim matches v.
    """
    if target == "v":
        pred = mem_forward_ref(st, k_t)
        return mem_weight_vjp(st, k_t, 2.0 * (pred - v_t))
    pred = mem_forward_ref(st, k_t)
    vhat = mem_forward_ref(st, v_t)
    if objective == "dot":
        cot_k = -vhat
        cot_v = -pred
    else:
        r = pred - vhat
        cot_k = 2.0 * r
        cot_v = -2.0 * r
    g1, g2, gskip = mem_weight_vjp(st, k_t, cot_k)
    if not stopgrad_vhat:
        h1, h2, hskip = mem_weight_vjp(st, v_t, cot_v)
        g1 = g1 + h1
        g2 = g2 + h2
        if gskip is not None:
            gskip = gskip + hskip
    return g1, g2, gskip


def ref_apply_chunk_update(
    state, cfg, k_seq, v_seq, eta_seq, alpha_seq, memories, target="mv"
):
    steps = k_seq.shape[1]
    boundary = {
        name: {kk: vv.clone() for kk, vv in state[name].items() if kk.startswith("w")}
        for name in memories
    }
    grads = {}
    for name in memories:
        per_tok = [
            inner_token_grads(
                boundary[name],
                k_seq[:, t],
                v_seq[:, t],
                cfg.objective,
                cfg.stopgrad_vhat,
                target=target,
            )
            for t in range(steps)
        ]
        grads[name] = per_tok

    beta = float(cfg.momentum)

    def with_momentum(st, key, g):
        if beta <= 0.0:
            return g
        buf = st.get(key)
        buf = g if buf is None else beta * buf + g
        st[key] = buf
        return buf

    for t in range(steps):
        k_t = k_seq[:, t]  # (B, D)
        eta_t = eta_seq[:, t].view(-1, 1, 1)
        alpha_t = alpha_seq[:, t].view(-1, 1, 1)
        for name in memories:
            st = state[name]
            g1, g2, gskip = grads[name][t]
            g1 = with_momentum(st, "m_w1", g1)
            g2 = with_momentum(st, "m_w2", g2)
            if cfg.use_rank1_precond:
                # w2 (aI - eta k k^T) = a w2 - eta (w2 k) k^T, expanded so the
                # reference never materializes the preconditioner matrix
                w2k = torch.einsum("bhd,bd->bh", st["w2"], k_t)
                st["w2"] = (
                    alpha_t * st["w2"]
                    - eta_t * torch.einsum("bh,bd->bhd", w2k, k_t)
                    - eta_t * g2
                )
            else:
                st["w2"] = alpha_t * st["w2"] - eta_t * g2
            st["w1"] = alpha_t * st["w1"] - eta_t * g1
            if "w_skip" in st:
                gskip = with_momentum(st, "m_w_skip", gskip)
                if cfg.use_rank1_precond:
                    wsk = torch.einsum("bod,bd->bo", st["w_skip"], k_t)
                    st["w_skip"] = (
                        alpha_t * st["w_skip"]
                        - eta_t * torch.einsum("bo,bd->bod", wsk, k_t)
                        - eta_t * gskip
                    )
                else:
                    st["w_skip"] = alpha_t * st["w_skip"] - eta_t * gskip


def ref_selfmod_state(module, batch):
    return {
        name: mem_state_from_module(getattr(module, f"m_{name}"), batch)
        for name in MEM_NAMES
    }


def ref_selfmod_forward_with_updates(module, x, state, memory_target="mv"):
    """Explicit replica of SelfModifyingTitans.forward_with_updates.

    memory_target="v" swaps M_memory's inner target from M(v) to the raw
    value v, to demonstrate numerically which one the code trains toward.
    (Only M_memory admits the swap; the eta/alpha memories emit scalars,
    so a raw-v target does not even typecheck for them.)
    """
    cfg = module.config
    b_sz, seq_len, _ = x.shape
    xc = causal_conv_ref(x, module.local_conv)

    other_chunk = int(cfg.chunk_size_other)
    memory_chunk = int(cfg.chunk_size_memory)
    other_names = (
        ("k", "v", "eta")
        + (("q",) if cfg.adaptive_q else ())
        + (("alpha",) if cfg.use_alpha else ())
    )

    outputs = []
    buf_other = {"k": [], "v": [], "eta": [], "alpha": []}
    buf_memory = {"k": [], "v": [], "eta": [], "alpha": []}

    def flush(buf, names, target="mv"):
        if not buf["k"]:
            return
        ref_apply_chunk_update(
            state,
            cfg,
            torch.cat(buf["k"], dim=1),
            torch.cat(buf["v"], dim=1),
            torch.cat(buf["eta"], dim=1),
            torch.cat(buf["alpha"], dim=1),
            names,
            target=target,
        )
        for key in buf:
            buf[key].clear()

    idx = 0
    while idx < seq_len:
        next_other = min((idx // other_chunk + 1) * other_chunk, seq_len)
        next_memory = min((idx // memory_chunk + 1) * memory_chunk, seq_len)
        end = min(next_other, next_memory)
        xs = xc[:, idx:end]

        k_chunk = mem_forward_ref(state["k"], xs)
        v_chunk = mem_forward_ref(state["v"], xs)
        q_chunk = (
            mem_forward_ref(state["q"], xs)
            if cfg.adaptive_q
            else xs @ module.w_q.weight.t()
        )
        if cfg.qk_l2_norm:
            k_chunk = normalize_ref(k_chunk, cfg.eps)
            q_chunk = normalize_ref(q_chunk, cfg.eps)
        eta_chunk = (
            F.softplus(mem_forward_ref(state["eta"], xs).squeeze(-1)) * cfg.eta_scale
        )
        if cfg.use_alpha:
            alpha_chunk = torch.sigmoid(mem_forward_ref(state["alpha"], xs).squeeze(-1))
        else:
            alpha_chunk = torch.ones_like(eta_chunk)
        outputs.append(mem_forward_ref(state["memory"], q_chunk))

        for buf in (buf_other, buf_memory):
            buf["k"].append(k_chunk)
            buf["v"].append(v_chunk)
            buf["eta"].append(eta_chunk)
            buf["alpha"].append(alpha_chunk)

        idx = end
        if idx == next_other:
            flush(buf_other, other_names)
        if idx == next_memory:
            flush(buf_memory, ("memory",), target=memory_target)

    flush(buf_other, other_names)
    flush(buf_memory, ("memory",), target=memory_target)
    return torch.cat(outputs, dim=1), state


def ref_selfmod_forward_with_state(module, x, state):
    cfg = module.config
    xc = causal_conv_ref(x, module.local_conv)
    q = (
        mem_forward_ref(state["q"], xc)
        if cfg.adaptive_q
        else xc @ module.w_q.weight.t()
    )
    if cfg.qk_l2_norm:
        q = normalize_ref(q, cfg.eps)
    return mem_forward_ref(state["memory"], q)


def lib_state_to_dict(state):
    out = {}
    for name in MEM_NAMES:
        mem = getattr(state, name)
        for key in ("w1", "w2", "w_skip", "m_w1", "m_w2", "m_w_skip"):
            t = getattr(mem, key)
            if t is not None:
                out[f"{name}.{key}"] = t
    return out


def ref_state_to_dict(state):
    out = {}
    for name, st in state.items():
        for key, t in st.items():
            out[f"{name}.{key}"] = t
    return out


# ---------------------------------------------------------------------------
# explicit CMS reference (level forward, gradient shaping, inner optimizer)
# ---------------------------------------------------------------------------


def cms_theta(block, level_name, deltas):
    module = block.cms.blocks[level_name]
    return {
        name: (param + deltas[name]).detach()
        for name, param in module.named_parameters()
    }


def cms_level_forward_ref(theta, x):
    # CMSBlock in eval mode: x + Linear(gelu(Linear(LayerNorm(x))))
    ln_w, ln_b = theta["net.0.weight"], theta["net.0.bias"]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    h = (x - mu) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    h = F.gelu(h @ theta["net.1.weight"].t() + theta["net.1.bias"])
    delta = h @ theta["net.3.weight"].t() + theta["net.3.bias"]
    return x + delta


def cms_shaped_grads_ref(block, level_name, deltas, x, teach, active):
    """Gradient of the level's shaped loss, taken via a linearized loss.

    The library builds target = sg(pred - teach) and differentiates
    ||pred - target||^2 * mask summed; the claimed gradient equals that
    of the linear functional 2 <teach * mask, pred>.  Differentiating the
    linear form instead makes the equality a real identity check.
    """
    module = block.cms.blocks[level_name]
    theta = {
        name: (param.detach() + deltas[name].detach()).requires_grad_(True)
        for name, param in module.named_parameters()
    }
    pred = cms_level_forward_ref(theta, x)
    mask = active.unsqueeze(-1).to(pred.dtype)
    linear_loss = (2.0 * teach * mask * pred).sum()
    names = list(theta)
    grads = torch.autograd.grad(linear_loss, [theta[n] for n in names])
    return dict(zip(names, grads))


class RefDeepMomentum:
    """Hand-rolled replica of the default inner optimizer."""

    def __init__(self, beta=0.9, beta2=0.999, eps=1e-8):
        self.beta, self.beta2, self.eps = beta, beta2, eps
        self.grad_avg = {}
        self.sq_avg = {}

    def step(self, key, grad):
        sq = self.sq_avg.get(key)
        sq = torch.zeros_like(grad) if sq is None else sq
        sq = self.beta2 * sq + (1.0 - self.beta2) * grad * grad
        self.sq_avg[key] = sq
        update = grad / (sq.sqrt() + self.eps)
        ga = self.grad_avg.get(key)
        ga = torch.zeros_like(grad) if ga is None else ga
        ga = self.beta * ga + (1.0 - self.beta) * update
        self.grad_avg[key] = ga
        return ga


class RefCmsState:
    def __init__(self, block):
        self.levels = [spec.name for spec in block.config.cms_levels]
        self.periods = {
            spec.name: int(spec.update_period) for spec in block.config.cms_levels
        }
        self.deltas = {
            name: {
                pname: torch.zeros_like(p)
                for pname, p in block.cms.blocks[name].named_parameters()
            }
            for name in self.levels
        }
        self.buffers = {
            name: {"inputs": [], "teach": [], "active": [], "count": 0}
            for name in self.levels
        }
        self.optim = {name: RefDeepMomentum() for name in self.levels}
        self.updates = {name: 0 for name in self.levels}
        self.lr = block.config.self_mod_lr


def _ref_pop(buffer, count):
    inputs, teach, active = [], [], []
    remaining = count
    while remaining > 0:
        take = min(remaining, buffer["inputs"][0].shape[1])
        inputs.append(buffer["inputs"][0][:, :take])
        teach.append(buffer["teach"][0][:, :take])
        active.append(buffer["active"][0][:, :take])
        if take == buffer["inputs"][0].shape[1]:
            for key in ("inputs", "teach", "active"):
                buffer[key].pop(0)
        else:
            for key in ("inputs", "teach", "active"):
                buffer[key][0] = buffer[key][0][:, take:]
        remaining -= take
    return (
        torch.cat(inputs, dim=1),
        torch.cat(teach, dim=1),
        torch.cat(active, dim=1),
    )


def ref_cms_forward_online(block, ref, x, teach, finalize):
    cfg = block.config
    base_chunk = min(ref.periods.values())
    active_mask = teach.abs().sum(dim=-1) > 0
    outputs = []
    for start in range(0, x.shape[1], base_chunk):
        end = min(start + base_chunk, x.shape[1])
        current = x[:, start:end]
        level_inputs = {}
        for name in ref.levels:
            level_inputs[name] = current
            current = cms_level_forward_ref(
                cms_theta(block, name, ref.deltas[name]), current
            )
        outputs.append(current)
        for name in ref.levels:
            buf = ref.buffers[name]
            buf["inputs"].append(level_inputs[name])
            buf["teach"].append(teach[:, start:end])
            buf["active"].append(active_mask[:, start:end])
            buf["count"] += end - start
            period = ref.periods[name]
            while buf["count"] >= period:
                ci, ct, ca = _ref_pop(buf, period)
                buf["count"] -= period
                ref_apply_cms_update(block, ref, name, ci, ct, ca)
    if finalize:
        if cfg.cms_flush_partial_at_end:
            for name in ref.levels:
                buf = ref.buffers[name]
                remaining = buf["count"]
                if remaining <= 0:
                    continue
                ci, ct, ca = _ref_pop(buf, remaining)
                buf["count"] -= remaining
                if bool(ca.any()):
                    ref_apply_cms_update(block, ref, name, ci, ct, ca)
        for name in ref.levels:
            buf = ref.buffers[name]
            for key in ("inputs", "teach", "active"):
                buf[key].clear()
            buf["count"] = 0
    return torch.cat(outputs, dim=1)


def ref_apply_cms_update(block, ref, name, inputs, teach, active):
    grads = cms_shaped_grads_ref(block, name, ref.deltas[name], inputs, teach, active)
    for pname, grad in grads.items():
        step = ref.optim[name].step(pname, grad)
        ref.deltas[name][pname] = ref.deltas[name][pname] - ref.lr * step
    ref.updates[name] += 1


def ref_selfmod_block_forward(block, x, teach, sm_state, cms_ref, finalize):
    o, sm_state = ref_selfmod_forward_with_updates(block.selfmod, x, sm_state)
    out = ref_cms_forward_online(block, cms_ref, o, teach, finalize)
    return out, sm_state


# ---------------------------------------------------------------------------
# library plumbing
# ---------------------------------------------------------------------------


def make_selfmod(seed, **overrides):
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )

    cfg = dict(
        dim=8,
        eta_scale=1e-3,
        chunk_size_other=4,
        chunk_size_memory=8,
        objective="l2",
        stopgrad_vhat=True,
        use_rank1_precond=True,
        use_alpha=True,
        momentum=0.0,
        qk_l2_norm=True,
        adaptive_q=False,
        use_skip=True,
        local_conv_window=4,
    )
    cfg.update(overrides)
    torch.manual_seed(seed)
    module = SelfModifyingTitans(SelfModifyingTitansConfig(**cfg)).double()
    module.eval()
    return module


def batched_lib_state(module, batch):
    return module._ensure_batched_state(module.init_fast_state(), batch)


SELFMOD_CONFIGS = {
    "s1_paperish": dict(),
    "s2_adaptive_q": dict(adaptive_q=True),
    "s3_momentum": dict(momentum=0.9),
    "s4_dot_nostopgrad": dict(objective="dot", stopgrad_vhat=False),
    "s5_bare": dict(
        local_conv_window=None,
        use_rank1_precond=False,
        use_skip=False,
        qk_l2_norm=False,
        use_alpha=False,
    ),
    "s6_interleave": dict(chunk_size_other=3, chunk_size_memory=5, stopgrad_vhat=False),
}


def run_selfmod_equivalence(report, tol):
    section = {}
    ok = True
    for name, overrides in SELFMOD_CONFIGS.items():
        module = make_selfmod(300, **overrides)
        torch.manual_seed(31)
        seq_len = 13 if name == "s6_interleave" else 16
        x = torch.randn(2, seq_len, module.config.dim, dtype=DTYPE)
        lib_state = batched_lib_state(module, 2)
        with torch.no_grad():
            lib_out, lib_state = module.forward_with_updates(x, lib_state)
        ref_state = ref_selfmod_state(module, 2)
        ref_out, ref_state = ref_selfmod_forward_with_updates(module, x, ref_state)
        r = compare(lib_out, ref_out)
        r["state"] = compare_dicts(
            lib_state_to_dict(lib_state), ref_state_to_dict(ref_state)
        )

        torch.manual_seed(32)
        x_read = torch.randn(2, 6, module.config.dim, dtype=DTYPE)
        with torch.no_grad():
            lib_read = module.forward_with_state(x_read, lib_state)
        ref_read = ref_selfmod_forward_with_state(module, x_read, ref_state)
        r["read"] = compare(lib_read, ref_read)
        r["pass"] = (
            r["rel_l2"] < tol
            and r["state"]["rel_l2"] < tol
            and r["read"]["rel_l2"] < tol
        )
        ok = ok and r["pass"]
        section[name] = r
        print(
            f"  {name}: out rel={r['rel_l2']:.3e} state rel={r['state']['rel_l2']:.3e} "
            f"read rel={r['read']['rel_l2']:.3e} {'PASS' if r['pass'] else 'FAIL'}",
            flush=True,
        )
    report["selfmod_equivalence"] = section
    return ok


def run_target_discrimination(report):
    module = make_selfmod(300)
    torch.manual_seed(33)
    x = torch.randn(1, 16, module.config.dim, dtype=DTYPE)
    lib_state = batched_lib_state(module, 1)
    with torch.no_grad():
        _, lib_state = module.forward_with_updates(x, lib_state)

    diffs = {}
    for target in ("mv", "v"):
        ref_state = ref_selfmod_state(module, 1)
        _, ref_state = ref_selfmod_forward_with_updates(
            module, x, ref_state, memory_target=target
        )
        diffs[target] = compare_dicts(
            lib_state_to_dict(lib_state), ref_state_to_dict(ref_state)
        )
    report["target_discrimination"] = {
        "memory_target_Mv_vs_library": diffs["mv"],
        "memory_target_v_vs_library": diffs["v"],
        "note": "raw-v targets only typecheck for M_memory; the scalar "
        "eta/alpha memories force the M(v) form",
    }
    print(
        f"  M_memory target M(v): rel={diffs['mv']['rel_l2']:.3e}   "
        f"target v: rel={diffs['v']['rel_l2']:.3e}",
        flush=True,
    )
    return diffs["mv"]["rel_l2"] < 1e-12 and diffs["v"]["rel_l2"] > 1e-3


def run_selfmod_causality_isolation(report, tol):
    module = make_selfmod(300)
    torch.manual_seed(34)
    x = torch.randn(2, 16, module.config.dim, dtype=DTYPE)
    with torch.no_grad():
        base, _ = module.forward_with_updates(x, batched_lib_state(module, 2))
    ok = True
    section = {}
    for s in (5, 9, 15):
        pert = x.clone()
        pert[:, s] += 10.0
        with torch.no_grad():
            out, _ = module.forward_with_updates(pert, batched_lib_state(module, 2))
        before = float((out[:, :s] - base[:, :s]).abs().max())
        passed = before == 0.0
        ok = ok and passed
        section[f"perturb_t{s}"] = dict(max_abs_before=before, passed=passed)
        print(
            f"  perturb t={s}: before={before:.1e} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    for row in range(2):
        with torch.no_grad():
            alone, st_alone = module.forward_with_updates(
                x[row : row + 1], batched_lib_state(module, 1)
            )
            both, st_both = module.forward_with_updates(x, batched_lib_state(module, 2))
        r = compare(both[row : row + 1], alone)
        row_state = {k: v[row : row + 1] for k, v in lib_state_to_dict(st_both).items()}
        r["state"] = compare_dicts(row_state, lib_state_to_dict(st_alone))
        r["pass"] = r["rel_l2"] < tol and r["state"]["rel_l2"] < tol
        ok = ok and r["pass"]
        section[f"isolation_row{row}"] = r
        print(
            f"  row {row} batched vs alone: out rel={r['rel_l2']:.3e} "
            f"state rel={r['state']['rel_l2']:.3e} {'PASS' if r['pass'] else 'FAIL'}",
            flush=True,
        )
    report["selfmod_causality_isolation"] = section
    return ok


def run_meta_gradient(report):
    """The read path touches only w_q, the conv, and M_memory; the other
    meta initializations get no autograd gradient while the true function
    is sensitive to them (they seed the no-grad state trajectory)."""
    module = make_selfmod(301, eta_scale=0.1)
    torch.manual_seed(35)
    x_prefix = torch.randn(1, 8, module.config.dim, dtype=DTYPE)
    x_read = torch.randn(1, 4, module.config.dim, dtype=DTYPE)

    def pipeline_loss():
        state = module._ensure_batched_state(module.init_fast_state(), 1)
        with torch.no_grad():
            _, state = module.forward_with_updates(x_prefix, state)
        return module.forward_with_state(x_read, state).sum()

    module.zero_grad(set_to_none=True)
    pipeline_loss().backward()
    grad_norms = {
        name: (0.0 if p.grad is None else float(p.grad.norm()))
        for name, p in module.named_parameters()
    }
    reached = {n for n, g in grad_norms.items() if g > 0}
    unreached_mems = ("m_k", "m_v", "m_eta", "m_alpha", "m_q")
    autograd_ok = (
        all(not any(n.startswith(mem + ".") for n in reached) for mem in unreached_mems)
        and any(n.startswith("m_memory.") for n in reached)
        and "w_q.weight" in reached
    )

    # central finite difference along a random direction in m_k.w1
    fd = {}
    for mem in ("m_k", "m_v", "m_eta", "m_alpha", "m_memory"):
        param = dict(module.named_parameters())[f"{mem}.w1.weight"]
        torch.manual_seed(36)
        direction = torch.randn_like(param)
        direction /= direction.norm()
        eps = 1e-5
        with torch.no_grad():
            param.add_(direction, alpha=eps)
            up = float(pipeline_loss())
            param.add_(direction, alpha=-2 * eps)
            down = float(pipeline_loss())
            param.add_(direction, alpha=eps)
        fd_dir = (up - down) / (2 * eps)
        auto_dir = 0.0
        grad = param.grad
        if grad is not None:
            auto_dir = float((grad * direction).sum())
        fd[mem] = dict(finite_difference=fd_dir, autograd=auto_dir)

    inert_to_autograd_sensitive_in_truth = all(
        fd[m]["autograd"] == 0.0 and abs(fd[m]["finite_difference"]) > 1e-8
        for m in ("m_k", "m_v", "m_eta", "m_alpha")
    )
    report["meta_gradient"] = dict(
        autograd_reached=sorted(reached),
        finite_difference=fd,
        autograd_partition_ok=autograd_ok,
        unreached_metas_sensitive=inert_to_autograd_sensitive_in_truth,
    )
    for m, entry in fd.items():
        print(
            f"  {m}.w1: autograd={entry['autograd']:+.3e} "
            f"finite-diff={entry['finite_difference']:+.3e}",
            flush=True,
        )
    print(
        f"  autograd partition {'PASS' if autograd_ok else 'FAIL'}; "
        f"unreached metas sensitive in truth "
        f"{'PASS' if inert_to_autograd_sensitive_in_truth else 'FAIL'}",
        flush=True,
    )
    return autograd_ok and inert_to_autograd_sensitive_in_truth


def run_selfmod_roundtrip(report, tol):
    # the causal depthwise conv keeps no cross-call state (each call is
    # left-padded with zeros), so the gates below run without it; its
    # boundary effect is characterized separately at the end
    module = make_selfmod(300, local_conv_window=None)
    torch.manual_seed(37)
    x = torch.randn(1, 16, module.config.dim, dtype=DTYPE)
    with torch.no_grad():
        full_out, full_state = module.forward_with_updates(
            x, batched_lib_state(module, 1)
        )
    section = {}
    ok = True
    for split in (8, 4, 6):  # both-aligned, memory-partial, both-partial
        with torch.no_grad():
            head, state = module.forward_with_updates(
                x[:, :split], batched_lib_state(module, 1)
            )
        buf = io.BytesIO()
        torch.save(state, buf)
        del state
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        with torch.no_grad():
            tail, state2 = module.forward_with_updates(x[:, split:], restored)
        r = compare(torch.cat([head, tail], dim=1), full_out)
        r["state"] = compare_dicts(
            lib_state_to_dict(state2), lib_state_to_dict(full_state)
        )
        aligned = split % module.config.chunk_size_other == 0 and (
            split % module.config.chunk_size_memory == 0
        )
        r["aligned"] = aligned
        if aligned:
            r["pass"] = r["rel_l2"] < tol and r["state"]["rel_l2"] < tol
            ok = ok and r["pass"]
            verdict = "PASS" if r["pass"] else "FAIL"
        else:
            r["pass"] = None
            verdict = "measured (early flush expected)"
        section[f"split_{split}"] = r
        print(
            f"  split at {split}: out rel={r['rel_l2']:.3e} "
            f"state rel={r['state']['rel_l2']:.3e}  {verdict}",
            flush=True,
        )

    # with the conv enabled, even a fully aligned split diverges, because
    # the conv re-pads with zeros at every call boundary
    module_c = make_selfmod(300)
    torch.manual_seed(37)
    xc = torch.randn(1, 16, module_c.config.dim, dtype=DTYPE)
    with torch.no_grad():
        full_c, full_state_c = module_c.forward_with_updates(
            xc, batched_lib_state(module_c, 1)
        )
        head, st = module_c.forward_with_updates(
            xc[:, :8], batched_lib_state(module_c, 1)
        )
        tail, st = module_c.forward_with_updates(xc[:, 8:], st)
    r = compare(torch.cat([head, tail], dim=1), full_c)
    r["state"] = compare_dicts(lib_state_to_dict(st), lib_state_to_dict(full_state_c))
    section["conv_boundary_aligned_split_8"] = r
    print(
        f"  conv on, aligned split at 8: out rel={r['rel_l2']:.3e} "
        f"state rel={r['state']['rel_l2']:.3e}  measured (conv keeps no cross-call state)",
        flush=True,
    )
    report["selfmod_roundtrip"] = section
    return ok


# ---------------------------------------------------------------------------
# block level: CMS write reference + cadence
# ---------------------------------------------------------------------------


def make_selfmod_block(seed, periods=(1, 3), flush=False, chunk=4, chunk_mem=8):
    from nested_learning.hope.block import HOPESelfModBlock, HOPESelfModBlockConfig
    from nested_learning.levels import LevelSpec

    torch.manual_seed(seed)
    cfg = HOPESelfModBlockConfig(
        dim=8,
        cms_levels=tuple(LevelSpec(name=f"cms_p{p}", update_period=p) for p in periods),
        cms_hidden_multiplier=2,
        cms_flush_partial_at_end=flush,
        selfmod_chunk_size=chunk,
        selfmod_chunk_size_memory=chunk_mem,
        eta_scale=1e-3,
        self_mod_lr=1e-3,
    )
    block = HOPESelfModBlock(cfg).double()
    block.eval()
    return block


def block_fast_state(block):
    from nested_learning.fast_state import build_block_fast_state

    return build_block_fast_state(
        titan_module=None,
        cms_blocks=dict(block.cms.blocks.items()),
        selfmod_module=block.selfmod,
        specs=list(block.config.cms_levels),
        optimizer_configs=block.config.optimizer_configs,
        default_lr=block.config.self_mod_lr,
    )


def cms_internals_dict(block, fast_state):
    out = {}
    for spec in block.config.cms_levels:
        name = spec.name
        for pname, delta in fast_state.cms_params[name].items():
            out[f"{name}.delta.{pname}"] = delta
        optim = fast_state.level_manager.optimizers[name]
        for key, st in optim.state.items():
            if st.grad_avg is not None:
                out[f"{name}.grad_avg.{key}"] = st.grad_avg
            if st.sq_avg is not None:
                out[f"{name}.sq_avg.{key}"] = st.sq_avg
    return out


def ref_cms_internals_dict(ref):
    out = {}
    for name in ref.levels:
        for pname, delta in ref.deltas[name].items():
            out[f"{name}.delta.{pname}"] = delta
        for key, t in ref.optim[name].grad_avg.items():
            out[f"{name}.grad_avg.{key}"] = t
        for key, t in ref.optim[name].sq_avg.items():
            out[f"{name}.sq_avg.{key}"] = t
    return out


def run_cms_reference(report, tol):
    block = make_selfmod_block(400)
    fs = block_fast_state(block)
    sm_ref = ref_selfmod_state(block.selfmod, 1)
    cms_ref = RefCmsState(block)
    torch.manual_seed(41)
    ok = True
    section = {}
    for call, (seq_len, finalize) in enumerate(((7, False), (9, True))):
        x = torch.randn(1, seq_len, 8, dtype=DTYPE)
        teach = 0.1 * torch.randn(1, seq_len, 8, dtype=DTYPE)
        with torch.no_grad():
            lib_out = block(
                x, teach_signal=teach, fast_state=fs, finalize_updates=finalize
            )
        ref_out, sm_ref = ref_selfmod_block_forward(
            block, x, teach, sm_ref, cms_ref, finalize
        )
        r = compare(lib_out, ref_out)
        r["cms_internals"] = compare_dicts(
            cms_internals_dict(block, fs), ref_cms_internals_dict(cms_ref)
        )
        r["selfmod_state"] = compare_dicts(
            lib_state_to_dict(fs.selfmod_state), ref_state_to_dict(sm_ref)
        )
        clock_stats = fs.level_manager.clock.stats()
        r["update_counts"] = {
            name: dict(lib=clock_stats[name].updates, ref=cms_ref.updates[name])
            for name in cms_ref.levels
        }
        counts_ok = all(v["lib"] == v["ref"] for v in r["update_counts"].values())
        r["pass"] = (
            r["rel_l2"] < tol
            and r["cms_internals"]["rel_l2"] < tol
            and r["selfmod_state"]["rel_l2"] < tol
            and counts_ok
        )
        ok = ok and r["pass"]
        section[f"call{call}_len{seq_len}_final{finalize}"] = r
        print(
            f"  call {call} (len {seq_len}, finalize {finalize}): "
            f"out rel={r['rel_l2']:.3e} cms rel={r['cms_internals']['rel_l2']:.3e} "
            f"selfmod rel={r['selfmod_state']['rel_l2']:.3e} "
            f"counts {[(k, v['lib']) for k, v in r['update_counts'].items()]} "
            f"{'PASS' if r['pass'] else 'FAIL'}",
            flush=True,
        )
    report["cms_reference"] = section
    return ok


def run_cms_cadence(report):
    section = {}
    ok = True
    for flush in (False, True):
        for splits in ((14,), (5, 9)):
            block = make_selfmod_block(400, periods=(1, 3, 8), flush=flush)
            fs = block_fast_state(block)
            torch.manual_seed(42)
            x = torch.randn(1, 14, 8, dtype=DTYPE)
            teach = 0.1 * torch.randn(1, 14, 8, dtype=DTYPE)
            pos = 0
            for i, size in enumerate(splits):
                finalize = i == len(splits) - 1
                with torch.no_grad():
                    block(
                        x[:, pos : pos + size],
                        teach_signal=teach[:, pos : pos + size],
                        fast_state=fs,
                        finalize_updates=finalize,
                    )
                pos += size
            stats = fs.level_manager.clock.stats()
            got = {
                spec.name: stats[spec.name].updates for spec in block.config.cms_levels
            }
            expected = {}
            for spec in block.config.cms_levels:
                p = int(spec.update_period)
                n = 14 // p
                if flush and 14 % p:
                    n += 1
                expected[spec.name] = n
            passed = got == expected
            ok = ok and passed
            key = f"flush_{flush}_splits_{'x'.join(map(str, splits))}"
            section[key] = dict(got=got, expected=expected, passed=passed)
            print(
                f"  {key}: got {got} expected {expected} "
                f"{'PASS' if passed else 'FAIL'}",
                flush=True,
            )
    report["cms_cadence"] = section
    return ok


# ---------------------------------------------------------------------------
# model level: streaming, suspend/resume
# ---------------------------------------------------------------------------


def make_model(seed, chunk=4, chunk_mem=8, periods=(1, 4), flush=False, conv=None):
    from nested_learning.levels import LevelSpec
    from nested_learning.model import HOPEModel, ModelConfig

    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=50,
        dim=16,
        num_layers=2,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=tuple(LevelSpec(name=f"cms_p{p}", update_period=p) for p in periods),
        cms_flush_partial_at_end=flush,
        self_mod_chunk_size=chunk,
        self_mod_chunk_size_memory=chunk_mem,
        self_mod_local_conv_window=conv,
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg).double()
    model.eval()
    return model, cfg


def model_state_dicts(fast_state):
    out = {}
    for i, blk in enumerate(fast_state.blocks):
        if blk.selfmod_state is not None:
            for k, v in lib_state_to_dict(blk.selfmod_state).items():
                out[f"block{i}.selfmod.{k}"] = v
        for level, params in blk.cms_params.items():
            for pname, delta in params.items():
                out[f"block{i}.cms.{level}.{pname}"] = delta
        for level, optim in blk.level_manager.optimizers.items():
            for key, st in optim.state.items():
                if st.grad_avg is not None:
                    out[f"block{i}.opt.{level}.ga.{key}"] = st.grad_avg
                if st.sq_avg is not None:
                    out[f"block{i}.opt.{level}.sq.{key}"] = st.sq_avg
    return out


def run_model_streaming(report, tol):
    torch.manual_seed(51)
    tokens = torch.randint(0, 50, (1, 16))
    teach = None
    section = {}
    ok = True

    def fresh():
        model, _ = make_model(500)
        return model

    model = fresh()
    torch.manual_seed(52)
    teach = 0.1 * torch.randn(1, 16, 16, dtype=DTYPE)

    def run_calls(model, splits, with_teach):
        fs = model.init_fast_state()
        outs = []
        pos = 0
        for i, size in enumerate(splits):
            finalize = i == len(splits) - 1
            kwargs = dict(fast_state=fs, finalize_updates=finalize)
            if with_teach:
                kwargs["teach_signal"] = teach[:, pos : pos + size]
            with torch.no_grad():
                outs.append(model(tokens[:, pos : pos + size], **kwargs))
            pos += size
        return torch.cat(outs, dim=1), fs

    mono, mono_fs = run_calls(model, (16,), True)

    for label, splits in (("aligned_8x8", (8, 8)), ("misaligned_6x10", (6, 10))):
        out, fs = run_calls(fresh(), splits, True)
        r = compare(out, mono)
        r["state"] = compare_dicts(model_state_dicts(fs), model_state_dicts(mono_fs))
        aligned = label.startswith("aligned")
        if aligned:
            r["pass"] = r["rel_l2"] < tol and r["state"]["rel_l2"] < tol
            ok = ok and r["pass"]
            verdict = "PASS" if r["pass"] else "FAIL"
        else:
            r["pass"] = None
            verdict = "measured (early self-mod flush expected)"
        section[f"teach_{label}"] = r
        print(
            f"  with updates, {label}: logits rel={r['rel_l2']:.3e} "
            f"state rel={r['state']['rel_l2']:.3e}  {verdict}",
            flush=True,
        )

    # read-only streaming (no teach signal): state never changes
    mono_ro, _ = run_calls(fresh(), (16,), False)
    out_ro, _ = run_calls(fresh(), (6, 10), False)
    r = compare(out_ro, mono_ro)
    r["pass"] = r["rel_l2"] < tol
    ok = ok and r["pass"]
    section["read_only_misaligned"] = r
    print(
        f"  read-only, misaligned 6x10: logits rel={r['rel_l2']:.3e} "
        f"{'PASS' if r['pass'] else 'FAIL'}",
        flush=True,
    )

    # the shipped default keeps the conv (window 4); with it on, even
    # aligned streaming diverges, because the conv re-pads every call
    def fresh_conv():
        model, _ = make_model(500, conv=4)
        return model

    mono_c, mono_c_fs = run_calls(fresh_conv(), (16,), True)
    out_c, fs_c = run_calls(fresh_conv(), (8, 8), True)
    r = compare(out_c, mono_c)
    r["state"] = compare_dicts(model_state_dicts(fs_c), model_state_dicts(mono_c_fs))
    r["pass"] = None
    section["conv4_teach_aligned_8x8"] = r
    print(
        f"  conv on, with updates, aligned 8x8: logits rel={r['rel_l2']:.3e} "
        f"state rel={r['state']['rel_l2']:.3e}  measured (conv keeps no cross-call state)",
        flush=True,
    )
    report["model_streaming"] = section
    return ok


def run_model_suspend_resume(report, tol):
    torch.manual_seed(51)
    tokens = torch.randint(0, 50, (1, 16))
    model, cfg = make_model(500)
    sd = model.state_dict()
    torch.manual_seed(52)
    teach = 0.1 * torch.randn(1, 16, 16, dtype=DTYPE)

    # uninterrupted two-call stream is the baseline
    fs = model.init_fast_state()
    with torch.no_grad():
        model(
            tokens[:, :8],
            teach_signal=teach[:, :8],
            fast_state=fs,
            finalize_updates=False,
        )
        base_out = model(
            tokens[:, 8:],
            teach_signal=teach[:, 8:],
            fast_state=fs,
            finalize_updates=True,
        )
    base_state = model_state_dicts(fs)

    # suspend after call one, serialize, rebuild the model, restore, resume
    model2, _ = make_model(999)  # different init on purpose
    model2.load_state_dict(sd)
    fs2 = model2.init_fast_state()
    with torch.no_grad():
        model2(
            tokens[:, :8],
            teach_signal=teach[:, :8],
            fast_state=fs2,
            finalize_updates=False,
        )
    buf = io.BytesIO()
    torch.save(fs2, buf)
    del fs2, model2

    model3, _ = make_model(998)
    model3.load_state_dict(sd)
    buf.seek(0)
    fs3 = torch.load(buf, weights_only=False)
    with torch.no_grad():
        out3 = model3(
            tokens[:, 8:],
            teach_signal=teach[:, 8:],
            fast_state=fs3,
            finalize_updates=True,
        )
    r = compare(out3, base_out)
    r["state"] = compare_dicts(model_state_dicts(fs3), base_state)
    clock_ok = all(
        fs3.blocks[i].level_manager.clock.stats()[spec.name].updates
        == fs.blocks[i].level_manager.clock.stats()[spec.name].updates
        for i in range(len(fs.blocks))
        for spec in model.blocks[i].config.cms_levels
    )
    r["clock_counts_match"] = clock_ok
    r["pass"] = r["rel_l2"] < tol and r["state"]["rel_l2"] < tol and clock_ok
    report["model_suspend_resume"] = r
    print(
        f"  serialized suspend/resume: logits rel={r['rel_l2']:.3e} "
        f"state rel={r['state']['rel_l2']:.3e} clocks "
        f"{'match' if clock_ok else 'DIFFER'} {'PASS' if r['pass'] else 'FAIL'}",
        flush=True,
    )
    return r["pass"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", default="hope_fidelity.json")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    import nested_learning

    pkg_dir = nested_learning.__file__.rsplit("/", 3)[0]
    try:
        commit = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    report = {
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nested_learning_commit": commit,
            "device": "cpu",
            "dtype": "float64",
            "tolerance": args.tol,
        }
    }

    gates = {}
    print("self-modifying core vs hand-derived reference:", flush=True)
    gates["selfmod"] = run_selfmod_equivalence(report, args.tol)
    print("inner-target discrimination:", flush=True)
    gates["target"] = run_target_discrimination(report)
    print("causality and batch isolation:", flush=True)
    gates["causality_isolation"] = run_selfmod_causality_isolation(report, args.tol)
    print("meta-gradient reachability:", flush=True)
    gates["meta_gradient"] = run_meta_gradient(report)
    print("self-mod state round trip:", flush=True)
    gates["selfmod_roundtrip"] = run_selfmod_roundtrip(report, args.tol)
    print("CMS write vs linearized-loss reference:", flush=True)
    gates["cms_reference"] = run_cms_reference(report, args.tol)
    print("CMS cadence vs closed form:", flush=True)
    gates["cms_cadence"] = run_cms_cadence(report)
    print("model-level streaming (paper block variant):", flush=True)
    gates["model_streaming"] = run_model_streaming(report, args.tol)
    print("model-level suspend/resume through bytes:", flush=True)
    gates["model_suspend_resume"] = run_model_suspend_resume(report, args.tol)

    report["gates"] = gates
    ok = all(gates.values())
    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=1)
    print(("PASS" if ok else "FAIL") + f"  ({args.json_out})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
