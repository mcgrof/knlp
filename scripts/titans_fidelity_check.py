#!/usr/bin/env python3
"""Check the Titans neural-memory candidate against explicit references.

The candidate is `lucidrains/titans-pytorch` (pinned by commit in the
evaluation's source manifest). There is no author-released implementation
of the Titans paper (arXiv 2501.00663), so before any training result can
mean anything, the candidate's update rule has to be pinned down against
slow, explicit reference code in double precision on the CPU, where speed
cannot hide semantic errors.

Two references are implemented here:

    code mode    an explicit re-implementation of what the library does:
                 per-chunk shared hyperparameters, every surprise gradient
                 taken at a fixed anchor weight, momentum and decay applied
                 as scans across chunks, retrieval reading the weights
                 written through the previous chunk.  Agreement validates
                 the reading of the code, not the paper.
    paper mode   the per-token recurrence as the paper defines it: the
                 surprise gradient of token t is taken at the memory the
                 previous token left behind, then momentum and forgetting.
                 The library is expected to match this only with
                 chunk_size=1 and ttt_batch_size=1.

The inner gradient of the code-mode reference is built with plain
`torch.autograd.grad` on an explicitly assembled loss, independent of the
library's `vmap(grad(functional_call))` machinery, and is itself anchored
by a closed-form hand-derived gradient on a linear memory model.

Beyond equivalence, the battery measures the approximations the code
makes (gradient anchoring, within-chunk sharing), and asserts causality,
batch isolation, token-by-token streaming equivalence, and a replay-free
state round trip through serialized bytes.

    PYTHONPATH=<deps>:<titans-clone> python3 scripts/titans_fidelity_check.py
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
    for name in db:
        r = compare(da[name], db[name])
        worst["max_abs"] = max(worst["max_abs"], r["max_abs"])
        worst["rel_l2"] = max(worst["rel_l2"], r["rel_l2"])
        worst["finite"] = worst["finite"] and r["finite"]
    return worst


def rmsnorm(x, weight):
    # nn.RMSNorm with default eps, which torch resolves to the dtype epsilon
    eps = torch.finfo(x.dtype).eps
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y * weight


def layernorm_no_affine(x, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def linear(x, mod):
    return x @ mod.weight.t() + (mod.bias if mod.bias is not None else 0.0)


def first_linear(seq_or_linear):
    # the library's Sequential helper collapses to the bare module when it
    # wraps only one; index into it only when it is an actual Sequential
    if isinstance(seq_or_linear, torch.nn.Sequential):
        return seq_or_linear[0]
    return seq_or_linear


# ---------------------------------------------------------------------------
# explicit memory-model forward (MemoryMLP, optionally ResidualNorm-wrapped)
# ---------------------------------------------------------------------------


def mem_forward(params, x):
    """Run the memory model explicitly from a flat name->tensor dict."""
    wrapped = "norm.gamma" in params
    prefix = "model.weights." if wrapped else "weights."
    ws = []
    i = 0
    while prefix + str(i) in params:
        ws.append(params[prefix + str(i)])
        i += 1
    assert ws, f"unrecognized memory model parameters: {list(params)}"
    h = x
    for i, w in enumerate(ws):
        if i > 0:
            h = F.gelu(h)
        h = h @ w
    if wrapped:
        h = layernorm_no_affine(h) * (params["norm.gamma"] + 1.0) + x
    return h


def chunk_grad_autograd(params, ks, vs, lrs):
    """d/dW of sum_i lr_i * mean((M(k_i) - v_i)^2), via plain autograd."""
    live = {name: p.detach().clone().requires_grad_(True) for name, p in params.items()}
    pred = mem_forward(live, ks)
    loss = ((pred - vs).pow(2).mean(dim=-1) * lrs).sum()
    names = list(live)
    grads = torch.autograd.grad(loss, [live[n] for n in names])
    return dict(zip(names, grads))


def chunk_grad_analytic(params, ks, vs, lrs):
    """Closed form for a single linear weight, no norm wrapper.

    loss_i = lr_i * mean((k_i W - v_i)^2)  =>
    dL/dW = sum_i lr_i * (2/d) k_i^T (k_i W - v_i)
    """
    assert list(params) == ["weights.0"]
    w = params["weights.0"]
    d = w.shape[-1]
    resid = ks @ w - vs
    grad = 2.0 / d * einsum_t(ks, resid * lrs.unsqueeze(-1))
    return {"weights.0": grad}


def einsum_t(a, b):
    return a.transpose(-1, -2) @ b


# ---------------------------------------------------------------------------
# code-mode reference: what the library computes, made explicit
# ---------------------------------------------------------------------------


def ref_hparams(mod, sn, b, heads):
    """Adaptive lr per token, decay and momentum per chunk, from the
    normalized store sequence.  Returns tensors indexed [b, h, ...]."""
    c = mod.store_chunk_size
    u = mod.num_kv_per_token
    n = sn.shape[1]
    num_chunks = n // c

    la = first_linear(mod.to_adaptive_step)
    raw = linear(sn, la)  # b n (h u)
    raw = raw.view(b, n, heads, u)
    lr = mod.adaptive_step_transform(raw)  # sigmoid * max_lr by default

    chunked = sn.view(b, num_chunks, c, -1).mean(dim=2) if num_chunks else None
    decay = eta = None
    if num_chunks:
        ld = first_linear(mod.to_decay_factor)
        decay = torch.sigmoid(linear(chunked, ld))  # b n h
        if mod.to_momentum is not None:
            lm = first_linear(mod.to_momentum)
            o = mod.momentum_order
            eta = torch.sigmoid(linear(chunked, lm)).view(b, num_chunks, heads, o)
    return lr, decay, eta, chunked


def ref_keys_values(mod, sn, values_seq, b, heads):
    c = mod.store_chunk_size
    u = mod.num_kv_per_token
    dh = linear(sn, first_linear(mod.to_keys)).shape[-1] // (heads * u)
    keys = linear(sn, first_linear(mod.to_keys)).view(b, -1, heads, u, dh)
    vals = linear(values_seq, first_linear(mod.to_values)).view(b, -1, heads, u, dh)
    # b n h u d -> b h (n u) d
    keys = keys.permute(0, 2, 1, 3, 4).reshape(b, heads, -1, dh)
    vals = vals.permute(0, 2, 1, 3, 4).reshape(b, heads, -1, dh)
    return keys, vals


def ref_store(mod, seq, weights, past_state, seq_index, grad_fn=chunk_grad_autograd):
    """Explicit replica of NeuralMemory.store_memories.

    weights: anchor dict name -> (b*h, *w); the gradient of every chunk in
    this call is taken at these weights.  past_state: (last_update,
    last_momentum) dicts or None.  Returns (updates, next_state) where
    updates entries run [W_prev, W_1, ..., W_n] as the library's
    remove_prev=False decay scan does.
    """
    b, n = seq.shape[:2]
    heads = mod.heads
    c = mod.store_chunk_size
    u = mod.num_kv_per_token
    n_rd = n // c * c
    num_chunks = n_rd // c
    seq, remainder = seq[:, :n_rd], seq[:, n_rd:]
    next_seq_index = seq_index + n_rd

    sn = rmsnorm(seq, mod.store_norm.weight) if n_rd else seq
    lr, decay, eta, _ = ref_hparams(mod, sn, b, heads) if n_rd else (None,) * 4
    if n_rd:
        keys, vals = ref_keys_values(mod, sn, sn, b, heads)

    if past_state is None:
        last_update = {k: v.clone() for k, v in weights.items()}
        last_momentum = {
            k: torch.zeros(mod.momentum_order, *v.shape, dtype=v.dtype)
            for k, v in weights.items()
        }
        past_state = (last_update, last_momentum)
    last_update, last_momentum = past_state

    if num_chunks == 0:
        updates = {k: v.unsqueeze(1) for k, v in weights.items()}
        state = (
            seq_index if False else next_seq_index,
            weights,
            remainder,
            past_state,
            updates,
        )
        return updates, state

    names = list(weights)
    # per-chunk surprises at the anchor weights
    surprises = {name: [] for name in names}
    for bi in range(b):
        for h in range(heads):
            bh = bi * heads + h
            params = {name: weights[name][bh] for name in names}
            for j in range(num_chunks):
                ks = keys[bi, h, j * c * u : (j + 1) * c * u]
                vs = vals[bi, h, j * c * u : (j + 1) * c * u]
                # token-major (n u) flattening within the chunk
                lrs = lr[bi, j * c : (j + 1) * c, h, :].reshape(-1)
                g = grad_fn(params, ks, vs, lrs)
                for name in names:
                    surprises[name].append((bh, j, -g[name]))

    updates = {}
    next_last_update = {}
    next_last_momentum = {}
    for name in names:
        w_shape = weights[name].shape[1:]
        sur = torch.zeros(b * heads, num_chunks, *w_shape, dtype=DTYPE)
        for bh, j, g in surprises[name]:
            sur[bh, j] = g

        upd_in = sur
        if eta is not None:
            mom_orders = []
            mom = sur
            for o in range(mod.momentum_order):
                prev = last_momentum[name][o]
                outs = torch.zeros_like(mom)
                state_t = prev
                for j in range(num_chunks):
                    gate = eta[:, j, :, o].reshape(b * heads, *([1] * len(w_shape)))
                    state_t = gate * state_t + mom[:, j]
                    outs[:, j] = state_t
                mom_orders.append(outs)
                mom = outs
            next_last_momentum[name] = torch.stack([m[:, -1] for m in mom_orders])
            upd_in = mom_orders[-1]

        outs = torch.zeros(b * heads, num_chunks + 1, *w_shape, dtype=DTYPE)
        state_t = last_update[name]
        outs[:, 0] = state_t
        for j in range(num_chunks):
            gate = (1.0 - decay[:, j, :]).reshape(b * heads, *([1] * len(w_shape)))
            state_t = gate * state_t + upd_in[:, j]
            outs[:, j + 1] = state_t
        updates[name] = outs
        next_last_update[name] = outs[:, -1]

    if eta is None:
        next_last_momentum = last_momentum
    next_state = (
        next_seq_index,
        weights,
        remainder,
        (next_last_update, next_last_momentum),
        updates,
    )
    return updates, next_state


def ref_retrieve(mod, seq, updates):
    """Explicit replica of NeuralMemory.retrieve_memories."""
    b, n = seq.shape[:2]
    heads = mod.heads
    c = mod.retrieve_chunk_size
    m = next(iter(updates.values())).shape[1]

    is_single = n == 1 and m == 1
    if is_single:
        c = 1
    need_pad = c > 1 or m > 1

    x = seq
    if need_pad:
        x = F.pad(x, (0, 0, 1, 0))
    n_pad = x.shape[1]
    n_up = math.ceil(n_pad / c) * c
    x = F.pad(x, (0, 0, 0, n_up - n_pad))
    num_q_chunks = n_up // c
    assert num_q_chunks == m, f"query chunks {num_q_chunks} != weight entries {m}"

    rn = rmsnorm(x, mod.retrieve_norm.weight)
    queries = linear(rn, first_linear(mod.to_queries))
    dh = queries.shape[-1] // heads
    queries = queries.view(b, n_up, heads, dh).permute(0, 2, 1, 3)  # b h n d

    values = torch.zeros(b, heads, n_up, dh, dtype=DTYPE)
    for bi in range(b):
        for h in range(heads):
            bh = bi * heads + h
            for j in range(m):
                params = {name: updates[name][bh, j] for name in updates}
                q = queries[bi, h, j * c : (j + 1) * c]
                values[bi, h, j * c : (j + 1) * c] = mem_forward(params, q)

    if mod.retrieve_gate is not None:
        gate = torch.sigmoid(linear(rn, first_linear(mod.retrieve_gate)))  # b n h
        values = values * gate.permute(0, 2, 1).unsqueeze(-1)

    out = values.permute(0, 2, 1, 3).reshape(b, n_up, heads * dh)
    if not isinstance(mod.combine_heads, torch.nn.Identity):
        out = linear(out, mod.combine_heads)
    if need_pad:
        out = out[:, 1:]
    return out[:, :n]


def ref_forward(mod, seq, state=None, ttt_batch_size=None, grad_fn=chunk_grad_autograd):
    """Explicit replica of NeuralMemory.forward, without the optional paths
    the canonical configuration does not use (multi-input views, gated
    transition, store masks, lookahead values)."""
    b = seq.shape[0]
    is_single_token = seq.shape[1] == 1

    if state is None:
        state = (0, None, None, None, None)
    seq_index, weights, cache, past_state, _ = state

    if weights is None:
        weights = {k: v.detach().clone() for k, v in dict(mod.init_weights(b)).items()}

    store_seq = (
        seq if cache is None or cache.numel() == 0 else torch.cat([cache, seq], dim=1)
    )
    L = store_seq.shape[1]
    batch_size = ttt_batch_size if ttt_batch_size is not None else mod.batch_size

    if batch_size is not None:
        update_after_final = (seq_index + L) % batch_size == 0
        seq_range = torch.arange(L) + seq_index + 1
        idx = seq_range[(seq_range % batch_size) == 0] - seq_index
        idx = F.pad(idx, (1, 0), value=0)
        if idx[-1] != L:
            idx = F.pad(idx, (0, 1), value=L)
        split_sizes = (idx[1:] - idx[:-1]).tolist()
    else:
        split_sizes = [L]
        update_after_final = False

    updates = None
    pos = 0
    for ind, size in enumerate(split_sizes):
        chunk_seq = store_seq[:, pos : pos + size]
        pos += size
        next_updates, next_state = ref_store(
            mod, chunk_seq, weights, past_state, seq_index, grad_fn=grad_fn
        )
        seq_index = next_state[0]
        weights = next_state[1]
        remainder = next_state[2]
        past_state = next_state[3]

        if updates is None:
            updates = next_updates
        else:
            updates = {
                k: torch.cat([updates[k][:, :-1], next_updates[k]], dim=1)
                for k in updates
            }

        is_last = ind == len(split_sizes) - 1
        if is_last and not update_after_final:
            continue
        weights = {k: v.clone() for k, v in past_state[0].items()}

    final_state = (seq_index, weights, remainder, past_state, updates)

    retrieve_updates = updates
    if is_single_token:
        retrieve_updates = {k: v.unsqueeze(1) for k, v in past_state[0].items()}

    retrieved = ref_retrieve(mod, seq, retrieve_updates)
    return retrieved, final_state


# ---------------------------------------------------------------------------
# paper-mode reference: the per-token recurrence of the paper
# ---------------------------------------------------------------------------


def paper_forward(mod, seq):
    """Eq (8)-(13) of the paper, token by token, sharing the module's
    projections: the surprise gradient of token t is taken at M_{t-1},
    S_t = eta_t S_{t-1} - theta_t grad, M_t = (1 - alpha_t) M_{t-1} + S_t,
    and token t retrieves from M_t.  This is what the library computes
    only when chunk_size=1 and ttt_batch_size=1."""
    b, n = seq.shape[:2]
    heads = mod.heads
    assert mod.store_chunk_size == 1 and mod.retrieve_chunk_size == 1
    weights = {k: v.detach().clone() for k, v in dict(mod.init_weights(b)).items()}
    momentum = {
        k: torch.zeros(mod.momentum_order, *v.shape, dtype=DTYPE)
        for k, v in weights.items()
    }
    outs = []
    for t in range(n):
        tok = seq[:, t : t + 1]
        sn = rmsnorm(tok, mod.store_norm.weight)
        lr, decay, eta, _ = ref_hparams(mod, sn, b, heads)
        keys, vals = ref_keys_values(mod, sn, sn, b, heads)
        new_w = {}
        for name, w in weights.items():
            grads = []
            for bi in range(b):
                for h in range(heads):
                    bh = bi * heads + h
                    params = {nm: weights[nm][bh] for nm in weights}
                    g = chunk_grad_autograd(
                        params, keys[bi, h], vals[bi, h], lr[bi, 0, h, :].reshape(-1)
                    )
                    grads.append(g[name])
            sur = -torch.stack(grads)
            mom = sur
            for o in range(mod.momentum_order):
                gate = eta[:, 0, :, o].reshape(b * heads, *([1] * (w.ndim - 1)))
                momentum[name][o] = gate * momentum[name][o] + mom
                mom = momentum[name][o]
            gate = (1.0 - decay[:, 0, :]).reshape(b * heads, *([1] * (w.ndim - 1)))
            new_w[name] = gate * w + mom
        weights = new_w
        outs.append(
            ref_retrieve(mod, tok, {k: v.unsqueeze(1) for k, v in weights.items()})
        )
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# state utilities
# ---------------------------------------------------------------------------


def state_tensors(state, prefix=""):
    """Flatten a NeuralMemState into name -> tensor."""
    out = {}
    seq_index, weights, cache, states, updates = state
    if weights is not None:
        for k, v in dict(weights).items():
            out[f"weights.{k}"] = v
    if cache is not None:
        out["cache_store_segment"] = cache
    if states is not None:
        last_update, last_momentum = states
        for k, v in dict(last_update).items():
            out[f"last_update.{k}"] = v
        for k, v in dict(last_momentum).items():
            out[f"last_momentum.{k}"] = v
    if updates is not None:
        for k, v in dict(updates).items():
            out[f"updates.{k}"] = v
    return out


def fresh_module(cfg, seed):
    from titans_pytorch import NeuralMemory

    torch.manual_seed(seed)
    mod = NeuralMemory(**cfg).double()
    mod.eval()
    return mod


def lib_forward(mod, seq, state=None, ttt=None):
    with torch.no_grad():
        return mod(seq, state=state, ttt_batch_size=ttt)


# ---------------------------------------------------------------------------
# the battery
# ---------------------------------------------------------------------------

CONFIGS = {
    "c1_default": dict(dim=16, heads=1, chunk_size=4),
    "c2_two_heads": dict(dim=16, heads=2, dim_head=8, chunk_size=4),
    "c3_multi_kv": dict(dim=16, heads=2, dim_head=8, chunk_size=4, num_kv_per_token=2),
    "c4_momentum2": dict(dim=16, heads=1, chunk_size=2, momentum_order=2),
}


def run_equivalence(report, tol):
    section = {}
    for name, cfg in CONFIGS.items():
        mod = fresh_module(cfg, seed=100)
        torch.manual_seed(7)
        for label, T in (("aligned", 24), ("with_remainder", 26)):
            seq = torch.randn(2, T, cfg["dim"], dtype=DTYPE)
            lib_out, lib_state = lib_forward(mod, seq)
            ref_out, ref_state = ref_forward(mod, seq)
            r = compare(lib_out, ref_out)
            r["state"] = compare_dicts(
                state_tensors(tuple(lib_state)), state_tensors(ref_state)
            )
            r["pass"] = r["rel_l2"] < tol and r["state"]["rel_l2"] < tol
            section[f"{name}.{label}"] = r
            print(
                f"  {name}.{label}: out rel={r['rel_l2']:.3e} "
                f"state rel={r['state']['rel_l2']:.3e} "
                f"{'PASS' if r['pass'] else 'FAIL'}",
                flush=True,
            )
    # ttt batching: the anchor advances at declared boundaries
    cfg = CONFIGS["c1_default"]
    mod = fresh_module(cfg, seed=100)
    torch.manual_seed(7)
    seq = torch.randn(2, 24, cfg["dim"], dtype=DTYPE)
    lib_out, lib_state = lib_forward(mod, seq, ttt=8)
    ref_out, ref_state = ref_forward(mod, seq, ttt_batch_size=8)
    r = compare(lib_out, ref_out)
    r["state"] = compare_dicts(
        state_tensors(tuple(lib_state)), state_tensors(ref_state)
    )
    r["pass"] = r["rel_l2"] < tol and r["state"]["rel_l2"] < tol
    section["c1_default.ttt_batch_8"] = r
    print(
        f"  c1_default.ttt_batch_8: out rel={r['rel_l2']:.3e} "
        f"state rel={r['state']['rel_l2']:.3e} {'PASS' if r['pass'] else 'FAIL'}",
        flush=True,
    )
    report["equivalence"] = section
    return all(r["pass"] for r in section.values())


def run_analytic_gradient(report, tol):
    from titans_pytorch.memory_models import MemoryMLP

    from titans_pytorch import NeuralMemory

    torch.manual_seed(11)
    mod = NeuralMemory(
        dim=8,
        heads=1,
        chunk_size=2,
        model=MemoryMLP(8, depth=1),
        mem_model_norm_add_residual=False,
        momentum=False,
    ).double()
    mod.eval()
    torch.manual_seed(12)
    seq = torch.randn(1, 8, 8, dtype=DTYPE)
    lib_out, _ = lib_forward(mod, seq)
    ref_out, _ = ref_forward(mod, seq, grad_fn=chunk_grad_analytic)
    r = compare(lib_out, ref_out)
    r["pass"] = r["rel_l2"] < tol
    report["analytic_linear_gradient"] = r
    print(
        f"  closed-form linear gradient: rel={r['rel_l2']:.3e} "
        f"{'PASS' if r['pass'] else 'FAIL'}",
        flush=True,
    )
    return r["pass"]


def run_paper_mode(report, tol):
    cfg = dict(dim=16, heads=1, chunk_size=1)
    mod = fresh_module(cfg, seed=200)
    torch.manual_seed(13)
    seq = torch.randn(1, 12, 16, dtype=DTYPE)

    paper = paper_forward(mod, seq)
    lib_out, _ = lib_forward(mod, seq, ttt=1)
    r_exact = compare(lib_out, paper)
    r_exact["pass"] = r_exact["rel_l2"] < tol

    # token-by-token with state carry must equal the one-call run
    state, outs = None, []
    for t in range(seq.shape[1]):
        o, state = lib_forward(mod, seq[:, t : t + 1], state=state, ttt=1)
        outs.append(o)
    r_stream = compare(torch.cat(outs, dim=1), lib_out)
    r_stream["pass"] = r_stream["rel_l2"] < tol

    # without ttt batching the anchor never advances: measured, not gated
    lib_anchor, _ = lib_forward(mod, seq)
    r_anchor = compare(lib_anchor, paper)

    report["paper_mode"] = {
        "chunk1_ttt1_vs_paper": r_exact,
        "stream_vs_one_call": r_stream,
        "chunk1_no_ttt_vs_paper_measured": r_anchor,
    }
    print(
        f"  chunk1+ttt1 vs paper recurrence: rel={r_exact['rel_l2']:.3e} "
        f"{'PASS' if r_exact['pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  token-by-token vs one call:      rel={r_stream['rel_l2']:.3e} "
        f"{'PASS' if r_stream['pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  fixed-anchor (no ttt) vs paper:  rel={r_anchor['rel_l2']:.3e}  (measured)",
        flush=True,
    )
    return r_exact["pass"] and r_stream["pass"]


def run_approximation_measures(report):
    """Size of the two named approximations, against the paper recurrence."""
    section = {}
    torch.manual_seed(14)
    seq64 = torch.randn(1, 64, 16, dtype=DTYPE)

    cfg1 = dict(dim=16, heads=1, chunk_size=1)
    mod1 = fresh_module(cfg1, seed=200)
    paper = paper_forward(mod1, seq64)

    for T in (8, 16, 32, 64):
        lib_out, _ = lib_forward(mod1, seq64[:, :T])
        section[f"anchoring_T{T}"] = compare(lib_out, paper[:, :T])

    for c in (2, 4, 8):
        mod_c = fresh_module(dict(dim=16, heads=1, chunk_size=c), seed=200)
        lib_out, _ = lib_forward(mod_c, seq64, ttt=c)
        section[f"chunk{c}_ttt{c}_T64"] = compare(lib_out, paper)
        lib_out_na, _ = lib_forward(mod_c, seq64)
        section[f"chunk{c}_no_ttt_T64"] = compare(lib_out_na, paper)

    report["approximation_size"] = section
    for k, v in section.items():
        print(f"  {k}: rel={v['rel_l2']:.4f}", flush=True)
    return True


def run_causality(report):
    cfg = CONFIGS["c2_two_heads"]
    mod = fresh_module(cfg, seed=100)
    torch.manual_seed(15)
    seq = torch.randn(1, 24, cfg["dim"], dtype=DTYPE)
    base, _ = lib_forward(mod, seq)
    section = {}
    ok = True
    for s in (5, 12, 23):
        pert = seq.clone()
        pert[:, s] += 10.0
        out, _ = lib_forward(mod, pert)
        before = float((out[:, :s] - base[:, :s]).abs().max())
        at_or_after = float((out[:, s:] - base[:, s:]).abs().max())
        passed = before == 0.0
        ok = ok and passed
        section[f"perturb_t{s}"] = dict(
            max_abs_before=before, max_abs_at_or_after=at_or_after, passed=passed
        )
        print(
            f"  perturb t={s}: before={before:.1e} after={at_or_after:.1e} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    report["causality"] = section
    return ok


def run_batch_isolation(report, tol):
    cfg = CONFIGS["c2_two_heads"]
    mod = fresh_module(cfg, seed=100)
    torch.manual_seed(16)
    seq = torch.randn(2, 24, cfg["dim"], dtype=DTYPE)
    both, _ = lib_forward(mod, seq)
    ok = True
    section = {}
    for row in range(2):
        alone, _ = lib_forward(mod, seq[row : row + 1])
        r = compare(both[row : row + 1], alone)
        r["pass"] = r["rel_l2"] < tol
        ok = ok and r["pass"]
        section[f"row{row}"] = r
        print(
            f"  row {row} batched vs alone: rel={r['rel_l2']:.3e} "
            f"{'PASS' if r['pass'] else 'FAIL'}",
            flush=True,
        )
    report["batch_isolation"] = section
    return ok


def run_streaming(report, tol):
    """Token-by-token decoding with chunk_size > 1 must match the one-call
    run; partial multi-token continuations are characterized as found."""
    cfg = CONFIGS["c1_default"]
    mod = fresh_module(cfg, seed=100)
    torch.manual_seed(17)
    seq = torch.randn(1, 16, cfg["dim"], dtype=DTYPE)
    full, _ = lib_forward(mod, seq)

    state, outs = None, []
    for t in range(16):
        o, state = lib_forward(mod, seq[:, t : t + 1], state=state)
        outs.append(o)
    r = compare(torch.cat(outs, dim=1), full)
    r["pass"] = r["rel_l2"] < tol
    print(
        f"  token-by-token (chunk 4) vs one call: rel={r['rel_l2']:.3e} "
        f"{'PASS' if r['pass'] else 'FAIL'}",
        flush=True,
    )

    # feeding a partial chunk mid-stream: characterize, do not gate
    partial = dict(behavior=None)
    try:
        _, st = lib_forward(mod, seq[:, :4])
        o2, st = lib_forward(mod, seq[:, 4:6], state=st)
        r2 = compare(o2, full[:, 4:6])
        partial["behavior"] = "runs"
        partial["vs_one_call"] = r2
        try:
            o3, st = lib_forward(mod, seq[:, 6:8], state=st)
            r3 = compare(o3, full[:, 6:8])
            partial["continuation"] = "runs"
            partial["continuation_vs_one_call"] = r3
        except Exception as e:  # noqa: BLE001
            partial["continuation"] = f"raises: {type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        partial["behavior"] = f"raises: {type(e).__name__}: {e}"

    # the same partial-chunk path with batch*heads > 1, where the silent
    # batch broadcast in retrieval cannot line up
    cfg2 = CONFIGS["c2_two_heads"]
    mod2 = fresh_module(cfg2, seed=100)
    torch.manual_seed(21)
    seq2 = torch.randn(2, 16, cfg2["dim"], dtype=DTYPE)
    try:
        _, st = lib_forward(mod2, seq2[:, :4])
        lib_forward(mod2, seq2[:, 4:6], state=st)
        partial["multi_bh"] = "runs"
    except Exception as e:  # noqa: BLE001
        partial["multi_bh"] = f"raises: {type(e).__name__}: {e}"

    print(f"  partial-chunk continuation: {partial['behavior']}", flush=True)
    if "vs_one_call" in partial:
        print(
            f"    tokens 5-6 vs one call: rel={partial['vs_one_call']['rel_l2']:.3e}",
            flush=True,
        )
    if "continuation" in partial:
        print(f"    next partial call: {partial['continuation']}", flush=True)
    print(f"    with batch*heads > 1: {partial['multi_bh']}", flush=True)

    report["streaming"] = {"token_by_token": r, "partial_chunk": partial}
    return r["pass"]


def run_roundtrip(report, tol):
    """Save the state at a split, serialize it, rebuild the module, restore,
    continue, and require agreement with the uninterrupted run."""
    cfg = CONFIGS["c2_two_heads"]
    mod = fresh_module(cfg, seed=100)
    torch.manual_seed(18)
    seq = torch.randn(1, 24, cfg["dim"], dtype=DTYPE)
    full, full_state = lib_forward(mod, seq)
    sd = mod.state_dict()

    from titans_pytorch.neural_memory import NeuralMemState

    def rebuild_and_restore(buf):
        mod2 = fresh_module(cfg, seed=999)  # different init on purpose
        try:
            mod2.load_state_dict(sd)
            section.setdefault("state_dict_reload", "plain load works")
        except RuntimeError as e:
            # the per-head learned parameters are stride-0 expanded views;
            # copying into them fails, so a fresh instance cannot be
            # rehydrated the ordinary way
            section.setdefault(
                "state_dict_reload",
                f"plain load fails ({type(e).__name__}); assign=True used",
            )
            mod2.load_state_dict(sd, assign=True)
        buf.seek(0)
        return mod2, NeuralMemState(*torch.load(buf, weights_only=False))

    def state_rel(state2):
        # the updates field accumulates only the current call's entries and
        # is rebuilt every forward; compare the load-bearing fields instead
        load_bearing = {
            k: v
            for k, v in state_tensors(tuple(state2)).items()
            if not k.startswith("updates.")
        }
        full_lb = {
            k: v
            for k, v in state_tensors(tuple(full_state)).items()
            if not k.startswith("updates.")
        }
        return compare_dicts(load_bearing, full_lb)

    section = {}
    ok = True
    for split in (6, 8, 12):  # mid-chunk, boundary, boundary
        mid_chunk = split % cfg["chunk_size"] != 0
        _, state = lib_forward(mod, seq[:, :split])
        buf = io.BytesIO()
        torch.save(tuple(state), buf)
        del state

        # whole-suffix continuation in one call
        mod2, restored = rebuild_and_restore(buf)
        r = {}
        try:
            out2, state2 = lib_forward(mod2, seq[:, split:], state=restored)
            r = compare(out2, full[:, split:])
            r["final_state_load_bearing"] = state_rel(state2)
            r["pass"] = (
                r["rel_l2"] < tol and r["final_state_load_bearing"]["rel_l2"] < tol
            )
            msg = (
                f"out rel={r['rel_l2']:.3e} "
                f"state rel={r['final_state_load_bearing']['rel_l2']:.3e} "
                f"{'PASS' if r['pass'] else 'FAIL'}"
            )
        except Exception as e:  # noqa: BLE001
            # a cached remainder that crosses a chunk boundary in the
            # continuation breaks retrieval alignment upstream; only a
            # boundary restore is required to work in one call
            r["behavior"] = f"raises: {type(e).__name__}: {e}"
            r["pass"] = None if mid_chunk else False
            msg = f"raises {type(e).__name__} ({'characterized' if mid_chunk else 'FAIL'})"
        if r["pass"] is False:
            ok = False
        section[f"split_{split}_one_call"] = r
        print(f"  split at {split}, one-call suffix: {msg}", flush=True)

        # token-by-token continuation, the decode pattern; must always work
        mod2, restored = rebuild_and_restore(buf)
        state2, outs = restored, []
        try:
            for t in range(split, seq.shape[1]):
                o, state2 = lib_forward(mod2, seq[:, t : t + 1], state=state2)
                outs.append(o)
            r = compare(torch.cat(outs, dim=1), full[:, split:])
            r["final_state_load_bearing"] = state_rel(state2)
            r["pass"] = (
                r["rel_l2"] < tol and r["final_state_load_bearing"]["rel_l2"] < tol
            )
            msg = (
                f"out rel={r['rel_l2']:.3e} "
                f"state rel={r['final_state_load_bearing']['rel_l2']:.3e} "
                f"{'PASS' if r['pass'] else 'FAIL'}"
            )
        except Exception as e:  # noqa: BLE001
            r = {"behavior": f"raises: {type(e).__name__}: {e}", "pass": False}
            msg = f"raises {type(e).__name__} FAIL"
        ok = ok and bool(r["pass"])
        section[f"split_{split}_stepwise"] = r
        print(f"  split at {split}, stepwise suffix: {msg}", flush=True)

    # the updates field must not be load-bearing for continuation
    _, state = lib_forward(mod, seq[:, :8])
    stripped = state._replace(updates=None)
    out_stripped, _ = lib_forward(mod, seq[:, 8:], state=stripped)
    out_kept, _ = lib_forward(mod, seq[:, 8:], state=state)
    r = compare(out_stripped, out_kept)
    r["pass"] = r["max_abs"] == 0.0
    ok = ok and r["pass"]
    section["updates_field_not_required"] = r
    print(
        f"  updates field stripped: max_abs={r['max_abs']:.1e} "
        f"{'PASS' if r['pass'] else 'FAIL'}",
        flush=True,
    )

    # the anchor weights are load-bearing: clobbering them must change output
    clobbered = state._replace(
        weights=state.weights.apply(lambda t: torch.randn_like(t))
    )
    out_clobbered, _ = lib_forward(mod, seq[:, 8:], state=clobbered)
    r = compare(out_clobbered, out_kept)
    r["pass"] = r["max_abs"] > 1e-6
    ok = ok and r["pass"]
    section["anchor_weights_are_live_state"] = r
    print(
        f"  anchor weights clobbered: max_abs={r['max_abs']:.1e} "
        f"{'PASS (live state)' if r['pass'] else 'FAIL (inert?)'}",
        flush=True,
    )

    # branching: one prefix, two suffixes, each equal to its own straight run
    torch.manual_seed(19)
    suf_a = torch.randn(1, 8, cfg["dim"], dtype=DTYPE)
    suf_b = torch.randn(1, 8, cfg["dim"], dtype=DTYPE)
    _, state = lib_forward(mod, seq[:, :8])
    branch_ok = True
    for label, suf in (("a", suf_a), ("b", suf_b)):
        out_branch, _ = lib_forward(mod, suf, state=state)
        straight, _ = lib_forward(mod, torch.cat([seq[:, :8], suf], dim=1))
        r = compare(out_branch, straight[:, 8:])
        r["pass"] = r["rel_l2"] < tol
        branch_ok = branch_ok and r["pass"]
        section[f"branch_{label}"] = r
        print(
            f"  branch {label}: rel={r['rel_l2']:.3e} "
            f"{'PASS' if r['pass'] else 'FAIL'}",
            flush=True,
        )
    ok = ok and branch_ok

    report["roundtrip"] = section
    return ok


def run_state_inventory(report):
    cfg = dict(dim=64, heads=4, dim_head=16, chunk_size=16)
    mod = fresh_module(cfg, seed=100)
    torch.manual_seed(20)
    seq = torch.randn(1, 50, cfg["dim"], dtype=DTYPE)  # remainder of 2
    _, state = lib_forward(mod, seq)
    inv = []
    for name, t in state_tensors(tuple(state)).items():
        restore_required = not name.startswith("updates.")
        inv.append(
            dict(
                name=name,
                shape=list(t.shape),
                dtype=str(t.dtype).replace("torch.", ""),
                logical_bytes=t.numel() * t.element_size(),
                restore_required=restore_required,
            )
        )
    fast = sum(e["logical_bytes"] for e in inv if e["name"].startswith("last_update."))
    total = sum(e["logical_bytes"] for e in inv if e["restore_required"])
    report["state_inventory"] = dict(
        config=cfg,
        seq_len=int(seq.shape[1]),
        seq_index=int(state.seq_index),
        tensors=inv,
        fast_weight_bytes=fast,
        restore_required_bytes=total,
        ratio_total_over_fast=total / fast,
    )
    print(
        f"  fast weights {fast}B, full restore set {total}B "
        f"({total / fast:.2f}x the fast weights)",
        flush=True,
    )
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", default="titans_fidelity.json")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    import titans_pytorch

    titans_dir = titans_pytorch.__file__.rsplit("/", 2)[0]
    try:
        commit = subprocess.run(
            ["git", "-C", titans_dir, "rev-parse", "HEAD"],
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
            "titans_pytorch_commit": commit,
            "device": "cpu",
            "dtype": "float64",
            "tolerance": args.tol,
        }
    }

    gates = {}
    print("closed-form gradient anchor:", flush=True)
    gates["analytic"] = run_analytic_gradient(report, args.tol)
    print("code-mode equivalence:", flush=True)
    gates["equivalence"] = run_equivalence(report, args.tol)
    print("paper-mode:", flush=True)
    gates["paper"] = run_paper_mode(report, args.tol)
    print("approximation size (vs paper recurrence, measured):", flush=True)
    run_approximation_measures(report)
    print("causality:", flush=True)
    gates["causality"] = run_causality(report)
    print("batch isolation:", flush=True)
    gates["isolation"] = run_batch_isolation(report, args.tol)
    print("streaming:", flush=True)
    gates["streaming"] = run_streaming(report, args.tol)
    print("state round trip:", flush=True)
    gates["roundtrip"] = run_roundtrip(report, args.tol)
    print("state inventory:", flush=True)
    run_state_inventory(report)

    report["gates"] = gates
    ok = all(gates.values())
    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=1)
    print(("PASS" if ok else "FAIL") + f"  ({args.json_out})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
