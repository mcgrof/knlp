#!/usr/bin/env python3
"""Select source layers, fit both mappers from captured traces, and apply them.

Everything here reads the files the capture stage wrote, so no model is loaded
while fitting. The two arms share the traces, the selected layers, the
regulariser and the centring; they differ only in which source heads feed a
target head and in whether rows are weighted.

The mapper on disk is float32, which is what the published artifact sizes
imply. The solves run in float64.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import attn_repair
from .methods import CACHE_BRIDGE, FULL_HEAD
from .support import identity_head_assignment
from .torch_ops import Accumulator, apply_rope, r_squared, strip_rope

COMPONENTS = ("k", "v")


def load_meta(trace_dir):
    with open(os.path.join(trace_dir, "CAPTURE.json")) as f:
        return json.load(f)


def open_layer(trace_dir, layer, comp):
    return np.load(os.path.join(trace_dir, f"L{layer:03d}.{comp}.npy"), mmap_mode="r")


def _rows(arr, lo, hi, device):
    return torch.from_numpy(np.array(arr[lo:hi])).to(device=device, dtype=torch.float64)


def check_pair(src_meta, tgt_meta):
    """The precondition both papers require, checked rather than assumed."""
    s, t = src_meta["geometry"], tgt_meta["geometry"]
    if s["head_dim"] != t["head_dim"]:
        raise ValueError(f"head widths differ: {s['head_dim']} vs {t['head_dim']}")
    identity_head_assignment(s["kv_heads"], t["kv_heads"])
    for key in ("rows", "sequences", "sequence_length", "stride"):
        if src_meta[key] != tgt_meta[key]:
            raise ValueError(
                f"the two traces disagree on {key}: {src_meta[key]} vs "
                f"{tgt_meta[key]}; they were not captured from the same rows"
            )


def select_layers(src_dir, tgt_dir, k, *, lam=0.01, device="cpu", chunk=2048):
    """Per target layer, the k source layers whose single-layer probes fit best.

    The probe is a centred fit from one source head to the target head of the
    same index, scored by held-in R-squared, averaged over heads and over keys
    and values. One set serves every head of the target layer and both
    components.

    A probe's Gram depends only on the source layer and head, so it is formed
    once and shared by every target layer; only the cross terms are per target.
    That makes the whole selection a single pass over the traces.
    """
    sm, tm = load_meta(src_dir), load_meta(tgt_dir)
    check_pair(sm, tm)
    Ls, Lt = sm["geometry"]["layers"], tm["geometry"]["layers"]
    H, d, rows = tm["geometry"]["kv_heads"], tm["geometry"]["head_dim"], tm["rows"]
    f64 = torch.float64
    scores = np.zeros((Lt, Ls), dtype=np.float64)
    for comp in COMPONENTS:
        src = [open_layer(src_dir, j, comp) for j in range(Ls)]
        tgt = [open_layer(tgt_dir, layer, comp) for layer in range(Lt)]
        sxx = torch.zeros(Ls, H, d, d, device=device, dtype=f64)
        sx = torch.zeros(Ls, H, d, device=device, dtype=f64)
        sxy = torch.zeros(Lt, Ls, H, d, d, device=device, dtype=f64)
        sy = torch.zeros(Lt, H, d, device=device, dtype=f64)
        syy = torch.zeros(Lt, H, device=device, dtype=f64)
        for lo in range(0, rows, chunk):
            hi = min(rows, lo + chunk)
            X = torch.stack([_rows(s_, lo, hi, device) for s_ in src], 1)  # [n,Ls,H,d]
            sxx += torch.einsum("njhd,njhe->jhde", X, X)
            sx += X.sum(0)
            for layer in range(Lt):
                Y = _rows(tgt[layer], lo, hi, device)  # [n,H,d]
                sxy[layer] += torch.einsum("njhd,nhe->jhde", X, Y)
                sy[layer] += Y.sum(0)
                syy[layer] += (Y * Y).sum((0, 2))
        n = float(rows)
        xbar = sx / n  # [Ls,H,d]
        A = sxx - n * torch.einsum("jhd,jhe->jhde", xbar, xbar)
        A_reg = A + lam * torch.eye(d, device=device, dtype=f64)
        chol = torch.linalg.cholesky(A_reg)  # [Ls,H,d,d]
        for layer in range(Lt):
            ybar = sy[layer] / n  # [H,d]
            B = sxy[layer] - n * torch.einsum("jhd,he->jhde", xbar, ybar)
            Wp = torch.cholesky_solve(B, chol)  # [Ls,H,d,d]
            ss_tot = syy[layer] - n * (ybar * ybar).sum(-1)  # [H]
            ss_res = (
                ss_tot[None]
                - 2.0 * (Wp * B).sum((-1, -2))
                + (Wp * torch.einsum("jhde,jhef->jhdf", A, Wp)).sum((-1, -2))
            )  # [Ls,H]
            r2 = 1.0 - ss_res / ss_tot[None].clamp_min(1e-300)
            scores[layer] += r2.mean(1).cpu().numpy() / len(COMPONENTS)
    selected = []
    for layer in range(Lt):
        order = sorted(range(Ls), key=lambda j: (-scores[layer, j], j))
        selected.append(sorted(order[:k]))
    return selected, scores


def repair_weights(tgt_dir, feature_width):
    """Calibration weights per (layer, component, head), with their receipts."""
    sens = np.load(os.path.join(tgt_dir, "sensitivities.npy"), mmap_mode="r")
    L, _, H, rows = sens.shape
    w = np.ones((L, 2, H, rows), dtype=np.float32)
    info = {}
    for layer in range(L):
        for c in range(2):
            for h in range(H):
                wi, inf = attn_repair.weights(
                    np.asarray(sens[layer, c, h], dtype=np.float64), feature_width
                )
                w[layer, c, h] = wi
                info[f"L{layer}.{COMPONENTS[c]}.h{h}"] = {
                    "alpha": inf["alpha"],
                    "ess_after": inf["ess_after"],
                    "cv_squared": inf["cv_squared"],
                    "degenerated_to_unweighted": inf["degenerated_to_unweighted"],
                }
    return w, info


def fit(
    method_id,
    src_dir,
    tgt_dir,
    selected,
    *,
    lam=0.01,
    device="cpu",
    chunk=8192,
    weights=None,
    progress=None,
):
    """Fit one arm. Returns coefficients and per-map held-in R-squared.

    W is [target_layers, 2, heads, feature_width, head_dim] and b is
    [target_layers, 2, heads, head_dim], float32, component 0 the keys.
    """
    sm, tm = load_meta(src_dir), load_meta(tgt_dir)
    check_pair(sm, tm)
    Lt = tm["geometry"]["layers"]
    H, d, rows = tm["geometry"]["kv_heads"], tm["geometry"]["head_dim"], tm["rows"]
    Hs = sm["geometry"]["kv_heads"]
    k = len(selected[0])
    full = method_id == FULL_HEAD
    if not full and method_id != CACHE_BRIDGE:
        raise ValueError(f"unknown method {method_id!r}")
    p = k * Hs * d if full else k * d
    if full and weights is not None:
        raise ValueError("the baseline is unweighted by definition")
    W = torch.zeros(Lt, 2, H, p, d, dtype=torch.float32)
    b = torch.zeros(Lt, 2, H, d, dtype=torch.float32)
    r2 = np.zeros((Lt, 2, H), dtype=np.float64)
    assign = identity_head_assignment(Hs, H)
    for layer in range(Lt):
        for ci, comp in enumerate(COMPONENTS):
            src = [open_layer(src_dir, j, comp) for j in selected[layer]]
            tgt = open_layer(tgt_dir, layer, comp)
            if full:
                # every head reads the same features: one Gram, heads as columns
                acc = [Accumulator(p, H * d, device)]
            else:
                acc = [Accumulator(p, d, device) for _ in range(H)]
            for lo in range(0, rows, chunk):
                hi = min(rows, lo + chunk)
                Y = _rows(tgt, lo, hi, device)
                Xs = [_rows(s_, lo, hi, device) for s_ in src]  # each [n, Hs, d]
                if full:
                    X = torch.cat([x.reshape(x.shape[0], -1) for x in Xs], dim=1)
                    acc[0].add(X, Y.reshape(Y.shape[0], H * d))
                    continue
                for h in range(H):
                    X = torch.cat([x[:, assign[h]] for x in Xs], dim=1)
                    wr = None
                    if weights is not None:
                        wr = torch.from_numpy(
                            np.array(weights[layer, ci, h, lo:hi])
                        ).to(device=device, dtype=torch.float64)
                    acc[h].add(X, Y[:, h], wr)
            if full:
                Wa, ba, st = acc[0].solve(lam)  # [p, H*d], [H*d]
                W[layer, ci] = (
                    Wa.reshape(p, H, d).permute(1, 0, 2).to(torch.float32).cpu()
                )
                b[layer, ci] = ba.reshape(H, d).to(torch.float32).cpu()
                for h in range(H):
                    r2[layer, ci, h] = r_squared(st, slice(h * d, (h + 1) * d))
            else:
                for h in range(H):
                    Wh, bh, st = acc[h].solve(lam)
                    W[layer, ci, h] = Wh.to(torch.float32).cpu()
                    b[layer, ci, h] = bh.to(torch.float32).cpu()
                    r2[layer, ci, h] = r_squared(st)
        if progress:
            progress(layer + 1, Lt)
    return {
        "W": W,
        "b": b,
        "r2": r2,
        "feature_width": p,
        "method": method_id,
        "selected": selected,
        "lambda": lam,
    }


def save_mapper(path, fitted, extra=None):
    from safetensors.torch import save_file

    meta = {
        "method": fitted["method"],
        "feature_width": str(fitted["feature_width"]),
        "lambda": str(fitted["lambda"]),
        "selected": json.dumps(fitted["selected"]),
        "coefficient_dtype": "float32",
    }
    meta.update({k: str(v) for k, v in (extra or {}).items()})
    save_file(
        {"W": fitted["W"].contiguous(), "b": fitted["b"].contiguous()},
        path,
        metadata=meta,
    )
    return os.path.getsize(path)


def load_mapper(path):
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt") as f:
        md = f.metadata()
        out["W"], out["b"] = f.get_tensor("W"), f.get_tensor("b")
    out["method"] = md["method"]
    out["selected"] = json.loads(md["selected"])
    out["feature_width"] = int(md["feature_width"])
    out["metadata"] = md
    return out


@torch.no_grad()
def transfer(mapper, src_pairs, src_theta, tgt_theta, *, dtype=torch.bfloat16):
    """Map a source cache to a target cache.

    src_pairs is the source's per-layer (keys, values), each
    [1, kv_heads, T, d] with keys rotated as cached. Positions are 0..T-1 on
    both sides: the prefix is installed where it was computed.
    """
    W, b = mapper["W"], mapper["b"]
    dev = src_pairs[0][0].device
    T = src_pairs[0][0].shape[2]
    pos = torch.arange(T, device=dev)
    Lt, _, H, p, d = W.shape
    full = mapper["method"] == FULL_HEAD
    content = {}
    out = []
    for layer in range(Lt):
        pair = []
        for ci in range(2):
            blocks = []
            for j in mapper["selected"][layer]:
                if (j, ci) not in content:
                    x = src_pairs[j][ci][0].to(torch.float32)  # [Hs, T, d]
                    if ci == 0:
                        x = strip_rope(x, pos, src_theta)
                    content[(j, ci)] = x
                blocks.append(content[(j, ci)])
            Wl = W[layer, ci].to(dev)
            bl = b[layer, ci].to(dev)
            if full:
                # [T, k*Hs*d], shared by every target head
                X = torch.cat([x.permute(1, 0, 2).reshape(T, -1) for x in blocks], 1)
                y = torch.einsum("tp,hpd->htd", X, Wl) + bl[:, None, :]
            else:
                X = torch.cat(blocks, dim=-1)  # [H, T, k*d]
                y = torch.einsum("htp,hpd->htd", X, Wl) + bl[:, None, :]
            if ci == 0:
                y = apply_rope(y, pos, tgt_theta)
            pair.append(y.unsqueeze(0).to(dtype))
        out.append((pair[0], pair[1]))
    return out
