"""TrellisFastMemory — exact sequential bounded-memory recurrence.

One pass of the Trellis memory. The state M_state in R^{B,H,M,D} is a fixed-size
compressed memory (M slots, D per head). For each token the memory is updated
by an online-gradient-descent step on the inner objective
    L_t(M) = 1/2 || phi(M @ write_t) - alpha_t ||^2
whose gradient w.r.t. z_t = M @ write_t is
    u_t = J_phi(z_t)^T (phi(z_t) - alpha_t)          (the Trellis "error" code)
giving the gated update (Trellis Eq. with forget gate beta and lr gamma)
    M_state <- beta_t * M_state - gamma * outer(u_t, write_t).
u_t is computed by an exact autograd VJP (correct for any phi); a closed-form /
chunked version is a later optimization.

Readout is write-before-read (from the UPDATED state):
    read_mode "M_q"   : y_t = M_state @ read_t          -> [B,H,M]   (key pass)
    read_mode "M_T_r" : y_t = M_state^T @ read_t         -> [B,H,D]   (value pass)

Memory is O(B*H*M*D) — independent of sequence length T. State is created fresh
per call (per independent sequence); a streaming generate() carries it forward.
"""

from __future__ import annotations

from typing import Optional

import torch

from .activations import (
    ln_silu,
    ln_silu_vjp,
    ln_silu_vjp_from_alpha,
    ln_silu_alpha_adjoint,
)

# --- Phase-1 iso-wall-clock refactor (Codex collab): split the true-stale
# chunked operator into a sequential state-evolution (returns per-chunk
# start-states + u) and a batched-over-chunks readout, so the readout einsums
# leave the Python loop and the two passes' states evolve stacked on a 2x batch
# axis. Math is identical to run_trellis_memory_chunked(refine_passes=0); proven
# by the fp64 A/B test in tests/test_trellis_phase1.py.


def _pad_time(x: torch.Tensor, pad: int, value: float = 0.0) -> torch.Tensor:
    if pad == 0:
        return x
    shape = list(x.shape)
    shape[2] = pad
    return torch.cat([x, x.new_full(shape, value)], dim=2)


def _trellis_vjp(phi, zin: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    z = zin.detach()
    if phi is ln_silu:
        return ln_silu_vjp_from_alpha(z, alpha)
    zr = z.requires_grad_(True)
    cg = alpha.requires_grad
    with torch.enable_grad():
        pred = phi(zr)
        err = pred - alpha
        (u,) = torch.autograd.grad(
            pred, zr, grad_outputs=err, create_graph=cg, retain_graph=True
        )
    return u if cg else u.detach()


def _trellis_residual(phi, zin: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    z = zin.detach()
    pred = phi(z)
    return pred - alpha


def _innovation_cap_enabled(stabilizer: str, cap: float) -> bool:
    return (
        stabilizer
        in (
            "innovation_rms_cap",
            "innovation_rms_cap_plus_layer0_gamma",
            "innovation_rms_cap_plus_layerwise_gamma",
        )
        and cap > 0.0
    )


def _delta_ratio_cap_enabled(stabilizer: str, cap: float) -> bool:
    return stabilizer == "delta_ratio_cap" and cap > 0.0


def _apply_innovation_rms_cap(
    err: torch.Tensor,
    cap: float,
    detach_scale: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided RMS cap over the memory/write feature dimension.

    This is update clipping, not unit normalization: rows already below the cap
    are unchanged. The default detached scale keeps the cap out of the learned
    architecture and matches the GPT-Pro recommendation.
    """
    rms = err.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    scale = (cap / (rms + eps)).clamp(max=1.0).to(err.dtype)
    if detach_scale:
        scale = scale.detach()
    return err * scale, scale


def _trellis_vjp_stabilized(
    phi,
    zin: torch.Tensor,
    alpha: torch.Tensor,
    stabilizer: str = "none",
    innovation_rms_cap: float = 0.0,
    detach_scale: bool = True,
) -> torch.Tensor:
    z = zin.detach()
    if not _innovation_cap_enabled(stabilizer, innovation_rms_cap):
        return _trellis_vjp(phi, z, alpha)
    zr = z.requires_grad_(True)
    cg = alpha.requires_grad
    with torch.enable_grad():
        pred = phi(zr)
        err = pred - alpha
        err, _ = _apply_innovation_rms_cap(
            err,
            innovation_rms_cap,
            detach_scale=detach_scale,
        )
        (u,) = torch.autograd.grad(
            pred, zr, grad_outputs=err, create_graph=cg, retain_graph=True
        )
    return u if cg else u.detach()


def _apply_delta_ratio_cap_to_u(
    u: torch.Tensor,
    write: torch.Tensor,
    gamma: torch.Tensor,
    state: torch.Tensor,
    cap: float,
    state_floor: float,
    detach_scale: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale a token/chunk update code so gamma*outer/update is bounded.

    `u` may be [B,H,M] for the sequential path or [B,H,C,M] for chunk state
    evolution. The chunk form uses the aggregate chunk update, then applies one
    scale per [B,H] to all codes in that chunk so batched readout can reuse the
    same stored `u`.
    """
    if u.dim() == 3:
        raw = gamma * torch.einsum("bhm,bhd->bhmd", u, write)
        scale_for_u = lambda scale: scale.squeeze(-1)
    elif u.dim() == 4:
        raw = gamma * torch.einsum("bhcm,bhcd->bhmd", u, write)
        scale_for_u = lambda scale: scale.squeeze(-1).unsqueeze(2)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported u shape {tuple(u.shape)}")
    state_rms = state.float().pow(2).mean(dim=(-1, -2), keepdim=True).sqrt()
    delta_rms = raw.float().pow(2).mean(dim=(-1, -2), keepdim=True).sqrt()
    ref = torch.maximum(state_rms, state.new_tensor(float(state_floor)).float())
    scale = (float(cap) * ref / (delta_rms + eps)).clamp(max=1.0).to(u.dtype)
    if detach_scale:
        scale = scale.detach()
    return u * scale_for_u(scale), scale


def trellis_chunk_decay(beta: torch.Tensor, chunk_size: int):
    """beta [B,H,T,1] -> P [B,H,nC,C,1], rmat [B,H,nC,C,C], pad."""
    assert beta.shape[-1] == 1, "chunked path supports per-head beta only"
    B, H, T, _ = beta.shape
    C = chunk_size
    pad = (C - T % C) % C
    bp = _pad_time(beta, pad, value=1.0)
    nC = bp.shape[2] // C
    b = bp.view(B, H, nC, C, 1)
    logP = torch.cumsum(torch.log(b.clamp_min(1e-9)), dim=3)
    P = torch.exp(logP)
    lp = logP.squeeze(-1)
    tri = torch.tril(torch.ones(C, C, device=beta.device, dtype=beta.dtype))
    rmat = torch.exp((lp[..., :, None] - lp[..., None, :]).clamp_max(0.0))
    rmat = rmat * tri.view(1, 1, 1, C, C)
    return P, rmat, pad


def run_trellis_memory_chunked_state_evolution(
    write,
    alpha,
    beta,
    gamma,
    phi,
    chunk_size,
    P=None,
    rmat=None,
    update_gate=None,
    residual_update_mix: float = 0.0,
    trellis_update_stabilizer: str = "none",
    trellis_innovation_rms_cap: float = 0.0,
    trellis_delta_ratio_cap: float = 0.0,
    trellis_state_rms_floor: float = 1e-3,
    trellis_stabilizer_detach_scale: bool = True,
):
    """True-stale chunk state evolution. No readout. Returns per-chunk
    start-states M0s [B,H,nC,M,D], codes us [B,H,nC,C,M], and P/rmat/pad."""
    B, H, T, D = write.shape
    M = alpha.shape[-1]
    C = chunk_size
    if P is None or rmat is None:
        if beta is None:
            raise ValueError("beta required when P/rmat are not supplied")
        P, rmat, pad = trellis_chunk_decay(beta, C)
    else:
        pad = P.shape[2] * C - T
    nC = P.shape[2]
    Wp = _pad_time(write, pad, 0.0).contiguous().view(B, H, nC, C, D)
    Ap = _pad_time(alpha, pad, 0.0).contiguous().view(B, H, nC, C, M)
    Gp = None
    if update_gate is not None:
        Gp = _pad_time(update_gate, pad, 1.0).contiguous()
        Gp = Gp.view(B, H, nC, C, update_gate.shape[-1])
    Mstate = torch.zeros(B, H, M, D, device=write.device, dtype=write.dtype)
    g = gamma.view(1, H, 1, 1)
    N = B * H
    M0s, us = [], []
    for c in range(nC):
        M0 = Mstate
        W = Wp[:, :, c]
        A = Ap[:, :, c]
        Wf = W.reshape(N, C, D)
        M0f = M0.reshape(N, M, D).to(Wf.dtype)
        M0W = torch.bmm(Wf, M0f.transpose(1, 2)).view(B, H, C, M)
        u = _trellis_vjp_stabilized(
            phi,
            M0W,
            A,
            stabilizer=trellis_update_stabilizer,
            innovation_rms_cap=trellis_innovation_rms_cap,
            detach_scale=trellis_stabilizer_detach_scale,
        )
        if residual_update_mix:
            u = u + residual_update_mix * _trellis_residual(phi, M0W, A)
        if Gp is not None:
            u = u * Gp[:, :, c]
        if _delta_ratio_cap_enabled(trellis_update_stabilizer, trellis_delta_ratio_cap):
            u, _ = _apply_delta_ratio_cap_to_u(
                u,
                W,
                g,
                Mstate,
                trellis_delta_ratio_cap,
                trellis_state_rms_floor,
                detach_scale=trellis_stabilizer_detach_scale,
            )
        rC = rmat[:, :, c, -1, :].unsqueeze(-1)
        upd = torch.bmm((u * rC).reshape(N, C, M).transpose(1, 2), Wf).view(B, H, M, D)
        Plast = P[:, :, c, -1:, :]
        Mstate = Plast * Mstate - g * upd
        M0s.append(M0)
        us.append(u)
    return torch.stack(M0s, dim=2), torch.stack(us, dim=2), P, rmat, pad


def run_trellis_memory_chunked_batched_readout(
    write, read, M0s, us, P, rmat, gamma, read_mode, chunk_size, T_out=None
):
    B, H, T, D = write.shape
    C = chunk_size
    nC = M0s.shape[2]
    M = M0s.shape[-2]
    T_out = T if T_out is None else T_out
    pad = nC * C - T
    Wp = _pad_time(write, pad, 0.0).contiguous().view(B, H, nC, C, D)
    Rp = _pad_time(read, pad, 0.0).contiguous()
    N = B * H * nC
    Wf = Wp.reshape(N, C, D)
    M0f = M0s.reshape(N, M, D)
    uf = us.reshape(N, C, M)
    Pf = P.reshape(N, C, 1)
    rf = rmat.reshape(N, C, C)
    gf = gamma.view(1, H, 1, 1, 1).expand(B, H, nC, 1, 1).reshape(N, 1, 1)
    if read_mode == "M_q":
        Rf = Rp.view(B, H, nC, C, D).reshape(N, C, D)
        M0q = torch.bmm(Rf, M0f.transpose(1, 2).to(Rf.dtype))
        S = torch.bmm(Rf, Wf.transpose(1, 2)) * rf
        y = Pf * M0q - gf * torch.bmm(S, uf)
        return y.view(B, H, nC, C, M).reshape(B, H, nC * C, M)[:, :, :T_out]
    if read_mode == "M_T_r":
        Rf = Rp.view(B, H, nC, C, M).reshape(N, C, M)
        first = Pf * torch.bmm(Rf, M0f.to(Rf.dtype))
        G = torch.bmm(Rf, uf.transpose(1, 2)) * rf
        y = first - gf * torch.bmm(G, Wf)
        return y.view(B, H, nC, C, D).reshape(B, H, nC * C, D)[:, :, :T_out]
    raise ValueError(read_mode)


class TrellisStateEvolutionFn(torch.autograd.Function):
    """Phase-2 layer 1: the true-stale state-evolution with an explicit
    hand-derived backward (no autograd through the chunk loop). Same forward as
    run_trellis_memory_chunked_state_evolution; the explicit backward is the
    foundation the Triton/HIP kernels will implement. ln_silu / chunk_size=16
    only (the fused-kernel regime). z is detached, so u's only gradient is to
    alpha (ln_silu_alpha_adjoint)."""

    @staticmethod
    def forward(ctx, write, alpha, P, rmat, gamma):
        B, H, T, D = write.shape
        M = alpha.shape[-1]
        nC, C = P.shape[2], P.shape[3]
        pad = nC * C - T
        Wp = _pad_time(write, pad, 0.0).contiguous().view(B, H, nC, C, D)
        Ap = _pad_time(alpha, pad, 0.0).contiguous().view(B, H, nC, C, M)
        Mstate = torch.zeros(B, H, M, D, device=write.device, dtype=write.dtype)
        g = gamma.view(1, H, 1, 1)
        N = B * H
        M0s, us, zs = [], [], []
        for c in range(nC):
            M0 = Mstate
            W = Wp[:, :, c]
            Wf = W.reshape(N, C, D)
            z = torch.bmm(Wf, M0.reshape(N, M, D).transpose(1, 2)).view(B, H, C, M)
            u = ln_silu_vjp_from_alpha(z, Ap[:, :, c])
            rC = rmat[:, :, c, -1, :].unsqueeze(-1)
            upd = torch.bmm((u * rC).reshape(N, C, M).transpose(1, 2), Wf).view(
                B, H, M, D
            )
            Mstate = P[:, :, c, -1:, :] * Mstate - g * upd
            M0s.append(M0)
            us.append(u)
            zs.append(z)
        M0s = torch.stack(M0s, dim=2)
        us = torch.stack(us, dim=2)
        zs = torch.stack(zs, dim=2)
        ctx.save_for_backward(Wp, P, rmat, gamma, M0s, us, zs)
        ctx.dims = (B, H, T, D, M, nC, C, pad)
        return M0s, us

    @staticmethod
    def backward(ctx, grad_M0s, grad_us):
        Wp, P, rmat, gamma, M0s, us, zs = ctx.saved_tensors
        B, H, T, D, M, nC, C, pad = ctx.dims
        g4 = gamma.view(1, H, 1, 1)
        g3 = gamma.view(1, H, 1)
        grad_W = torch.zeros_like(Wp)
        grad_A = torch.zeros(B, H, nC, C, M, device=Wp.device, dtype=Wp.dtype)
        grad_P = torch.zeros_like(P)
        grad_rmat = torch.zeros_like(rmat)
        grad_gamma = torch.zeros(H, device=gamma.device, dtype=gamma.dtype)
        bar_M = torch.zeros(B, H, M, D, device=Wp.device, dtype=Wp.dtype)
        for c in reversed(range(nC)):
            bar_M1 = bar_M
            W = Wp[:, :, c]  # [B,H,C,D]
            u = us[:, :, c]  # [B,H,C,M]
            M0 = M0s[:, :, c]  # [B,H,M,D]
            rC = rmat[:, :, c, -1, :]  # [B,H,C]
            Plast = P[:, :, c, -1:, :]  # [B,H,1,1]
            BW = torch.einsum("bhmd,bhcd->bhcm", bar_M1, W)  # bar_M1 @ W_s
            UB = torch.einsum("bhcm,bhmd->bhcd", u, bar_M1)  # u_s @ bar_M1
            grad_W[:, :, c] = -g4 * rC.unsqueeze(-1) * UB
            bar_u_state = -g4 * rC.unsqueeze(-1) * BW
            ub = (u * BW).sum(-1)  # <bar_M1, outer(u_s,W_s)> per s -> [B,H,C]
            grad_rmat[:, :, c, -1, :] = -g3 * ub
            grad_gamma += (-(rC * ub).sum(-1)).sum(0)
            grad_P[:, :, c, -1, 0] = (bar_M1 * M0).sum((-1, -2))
            bar_u_total = grad_us[:, :, c] + bar_u_state
            grad_A[:, :, c] = ln_silu_alpha_adjoint(zs[:, :, c], bar_u_total)
            bar_M = grad_M0s[:, :, c] + Plast * bar_M1
        gw = grad_W.reshape(B, H, nC * C, D)[:, :, :T]
        ga = grad_A.reshape(B, H, nC * C, M)[:, :, :T]
        return gw, ga, grad_P, grad_rmat, grad_gamma


def run_trellis_memory_chunked_phase1(
    write,
    read,
    alpha,
    beta,
    gamma,
    phi,
    read_mode,
    chunk_size,
    update_gate=None,
    residual_update_mix: float = 0.0,
):
    """True-stale equivalent of run_trellis_memory_chunked(refine_passes=0)."""
    P, rmat, _ = trellis_chunk_decay(beta, chunk_size)
    M0s, us, P, rmat, _ = run_trellis_memory_chunked_state_evolution(
        write,
        alpha,
        beta,
        gamma,
        phi,
        chunk_size,
        P=P,
        rmat=rmat,
        update_gate=update_gate,
        residual_update_mix=residual_update_mix,
    )
    return run_trellis_memory_chunked_batched_readout(
        write,
        read,
        M0s,
        us,
        P,
        rmat,
        gamma,
        read_mode,
        chunk_size,
        T_out=write.shape[2],
    )


def run_trellis_memory(
    write: torch.Tensor,  # [B,H,T,D]
    read: torch.Tensor,  # [B,H,T,D] (key pass) or [B,H,T,M] (value pass)
    alpha: torch.Tensor,  # [B,H,T,M]
    beta: torch.Tensor,  # [B,H,T,1] or [B,H,T,M]
    gamma: torch.Tensor,  # [H] positive
    phi,  # callable over last dim (M)
    read_mode: str,  # "M_q" | "M_T_r"
    training: bool,
    exact_inner: bool = True,  # True: backprop through the inner VJP (2nd order,
    # slow/exact). False: stale-gradient (detach u from
    # the param graph) — the sanctioned fast mode.
    M_init: Optional[torch.Tensor] = None,  # [B,H,M,D] carried state (generation)
    return_state: bool = False,
    update_gate: Optional[torch.Tensor] = None,  # [B,H,T,1] or [B,H,T,M]
    residual_update_mix: float = 0.0,
    trellis_update_stabilizer: str = "none",
    trellis_innovation_rms_cap: float = 0.0,
    trellis_delta_ratio_cap: float = 0.0,
    trellis_state_rms_floor: float = 1e-3,
    trellis_stabilizer_detach_scale: bool = True,
    input_gate: Optional[torch.Tensor] = None,  # [B,H,T,M] per-slot gate a(x_t)
):
    B, H, T, D = write.shape
    M = alpha.shape[-1]
    dev, dt = write.device, write.dtype
    if M_init is None:
        Mstate = torch.zeros(B, H, M, D, device=dev, dtype=dt)
    else:
        Mstate = M_init
    g = gamma.view(1, H, 1, 1)
    per_slot = beta.shape[-1] == M
    outs = []
    create_graph = training and torch.is_grad_enabled()

    for t in range(T):
        w = write[:, :, t, :]  # [B,H,D]
        a = alpha[:, :, t, :]  # [B,H,M]
        b = beta[:, :, t, :]  # [B,H,1] or [B,H,M]
        # z_t = M_state @ write_t
        z = torch.einsum("bhmd,bhd->bhm", Mstate, w)  # [B,H,M]
        if input_gate is not None:
            # input-conditioned write: u_t = a(x_t) ⊙ z_t − alpha_t. a is a
            # per-slot gate from the token INPUT (not the state code z), so the
            # update stays AFFINE in M -> exact-chunkable, and a≡1 recovers the
            # (gated) delta rule. z keeps its M dependency so the outer backward
            # flows through the recurrence; no inner VJP graph is needed.
            a_gate = input_gate[:, :, t, :]  # [B,H,M]
            u = a_gate * z - a
            if update_gate is not None:
                u = u * update_gate[:, :, t, :]
            outer = torch.einsum("bhm,bhd->bhmd", u, w)  # [B,H,M,D]
            b_e = b.unsqueeze(-1)  # [B,H,M,1] or [B,H,1,1]
            Mstate = b_e * Mstate - g * outer
            r = read[:, :, t, :]
            if read_mode == "M_q":
                y = torch.einsum("bhmd,bhd->bhm", Mstate, r)
            elif read_mode == "M_T_r":
                y = torch.einsum("bhmd,bhm->bhd", Mstate, r)
            else:
                raise ValueError(read_mode)
            outs.append(y)
            continue
        # u_t = J_phi(z)^T (phi(z) - alpha) via VJP.
        # exact_inner: keep z in the graph -> u carries both the M (2nd-order)
        # and alpha gradients. stale: detach z (cut the expensive M 2nd-order
        # through the recurrence) but KEEP alpha in `err` so alpha_proj still
        # trains (first-order). create_graph only while training.
        exact = exact_inner and z.requires_grad
        z_req = z if exact else z.detach().requires_grad_(True)
        cg = z.requires_grad
        with torch.enable_grad():
            pred = phi(z_req)  # [B,H,M]
            err = pred - a  # alpha stays in graph
            if _innovation_cap_enabled(
                trellis_update_stabilizer, trellis_innovation_rms_cap
            ):
                err, _ = _apply_innovation_rms_cap(
                    err,
                    trellis_innovation_rms_cap,
                    detach_scale=trellis_stabilizer_detach_scale,
                )
            (u,) = torch.autograd.grad(
                pred,
                z_req,
                grad_outputs=err,
                create_graph=cg,
                retain_graph=True,
            )
        if not cg:
            u = u.detach()
        if residual_update_mix:
            if exact:
                resid = err
            else:
                resid = _trellis_residual(phi, z, a)
            u = u + residual_update_mix * resid
        if update_gate is not None:
            u = u * update_gate[:, :, t, :]
        if _delta_ratio_cap_enabled(trellis_update_stabilizer, trellis_delta_ratio_cap):
            u, _ = _apply_delta_ratio_cap_to_u(
                u,
                w,
                g,
                Mstate,
                trellis_delta_ratio_cap,
                trellis_state_rms_floor,
                detach_scale=trellis_stabilizer_detach_scale,
            )
        # gated OGD update: M <- beta*M - gamma * outer(u, write)
        outer = torch.einsum("bhm,bhd->bhmd", u, w)  # [B,H,M,D]
        b_e = b.unsqueeze(-1) if per_slot else b.unsqueeze(-1)  # [B,H,M,1] or [B,H,1,1]
        Mstate = b_e * Mstate - g * outer
        # readout from UPDATED state (write-before-read)
        r = read[:, :, t, :]
        if read_mode == "M_q":
            y = torch.einsum("bhmd,bhd->bhm", Mstate, r)  # [B,H,M]
        elif read_mode == "M_T_r":
            y = torch.einsum("bhmd,bhm->bhd", Mstate, r)  # [B,H,D]
        else:
            raise ValueError(read_mode)
        outs.append(y)

    out = torch.stack(outs, dim=2)  # [B,H,T,*]
    if return_state:
        return out, Mstate
    return out


def run_trellis_memory_chunked(
    write: torch.Tensor,  # [B,H,T,D]
    read: torch.Tensor,  # [B,H,T,D] (key) or [B,H,T,M] (value)
    alpha: torch.Tensor,  # [B,H,T,M]
    beta: torch.Tensor,  # [B,H,T,1]  (per-head only)
    gamma: torch.Tensor,  # [H]
    phi,
    read_mode: str,  # "M_q" | "M_T_r"
    chunk_size: int,
    refine_passes: int = 0,
    update_gate: Optional[torch.Tensor] = None,  # [B,H,T,1] or [B,H,T,M]
    residual_update_mix: float = 0.0,
    trellis_update_stabilizer: str = "none",
    trellis_innovation_rms_cap: float = 0.0,
    trellis_delta_ratio_cap: float = 0.0,
    trellis_state_rms_floor: float = 1e-3,
    trellis_stabilizer_detach_scale: bool = True,
    input_gate: Optional[torch.Tensor] = None,  # [B,H,T,M] per-slot gate a(x_t)
):
    """Faithful chunkwise form of run_trellis_memory (per-head beta only).

    The state recurrence within a chunk is handled exactly via segmented decay
    products (Mstate is reconstructed as M_t = P_t*M0 - gamma*sum_{s<=t}
    (P_t/P_s) u_s w_s), which turns readout + update into matmuls. The only
    approximation is the inner code z_t = M_{t-1} @ w_t.

    Two regimes, and only two are sane (see reports/trellis_full_eval.md sec 11):

      refine_passes == 0   "true-stale": z_t uses the chunk-start state M0 for
                           all t. Cheapest (T/C inner steps -> ~15x speedup) but
                           lossy (+35-46% PPL): it discards within-chunk state
                           evolution.

      refine_passes <  0   "exact": auto-sets passes = chunk_size-1. The inner
                           dependency z_t<-u_{s<t} is strictly lower-triangular,
                           so Jacobi iteration is NILPOTENT: after k passes the
                           first k tokens are bit-exact, and C-1 passes reproduce
                           the sequential forward exactly (verified < 1e-3). This
                           is a correctness ORACLE, not a speed win -- it costs
                           (T/C)*(C-1) ~= T inner steps, same as the sequential
                           loop. A single-pass closed-form solve would need the
                           inner step to be linear in z; Trellis's nonlinear phi
                           forfeits that delta-rule free lunch, so there is no
                           cheap-AND-exact chunkwise form for this architecture.

    Any fixed 0 < refine_passes < C-1 is a TRAP: Jacobi's converged prefix is
    only `refine_passes` tokens long and the unconverged tail is erratic (the
    non-monotone transient), which is why intermediate values trained worse than
    true-stale. Use 0 (fast/lossy) or <0 (exact/slow); nothing between.

    Exact (== sequential stale) at chunk_size=1 for any refine_passes (the
    in-chunk correction is empty). u is detached from M (no 2nd-order) but keeps
    the alpha gradient; outer backprop flows through the chunkwise matmuls.
    """
    if refine_passes < 0:
        refine_passes = chunk_size - 1
    assert beta.shape[-1] == 1, "chunked path supports per-head beta only"
    B, H, T, D = write.shape
    M = alpha.shape[-1]
    dev, dt = write.device, write.dtype
    # state in the input dtype; TrellisMixer forces fp32 inputs (bf16 is a dead
    # end here), so this is fp32 in practice and fp64 A/B tests still work.
    Mstate = torch.zeros(B, H, M, D, device=dev, dtype=dt)
    g = gamma.view(1, H, 1, 1)
    outs = []

    def _vjp(zin, A):
        z = zin.detach()  # cut M 2nd-order
        if phi is ln_silu and not _innovation_cap_enabled(
            trellis_update_stabilizer, trellis_innovation_rms_cap
        ):
            # fused closed-form phi(z) + J_phi^T(phi(z)-A): exact, no per-chunk
            # autograd graph (the kernel's dominant overhead) and no double
            # LN/SiLU. Linear in A, so the outer backward to alpha_proj flows.
            return ln_silu_vjp_from_alpha(z, A)
        zr = z.requires_grad_(True)
        cg = A.requires_grad
        with torch.enable_grad():
            pred = phi(zr)
            err = pred - A  # keep alpha -> alpha_proj trains
            if _innovation_cap_enabled(
                trellis_update_stabilizer, trellis_innovation_rms_cap
            ):
                err, _ = _apply_innovation_rms_cap(
                    err,
                    trellis_innovation_rms_cap,
                    detach_scale=trellis_stabilizer_detach_scale,
                )
            (uu,) = torch.autograd.grad(
                pred, zr, grad_outputs=err, create_graph=cg, retain_graph=True
            )
        return uu if cg else uu.detach()

    # masks are shape-fixed per call (chunk_size, dev, dt) -> allocate once
    tri_full = torch.tril(torch.ones(chunk_size, chunk_size, device=dev, dtype=dt))
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        C = c1 - c0
        W = write[:, :, c0:c1, :]  # [B,H,C,D]
        R = read[:, :, c0:c1, :]  # [B,H,C,*]
        A = alpha[:, :, c0:c1, :]  # [B,H,C,M]
        b = beta[:, :, c0:c1, :]  # [B,H,C,1]
        G = update_gate[:, :, c0:c1, :] if update_gate is not None else None
        # cumulative inclusive decay P_t = prod_{i<=t} b_i, in log space so the
        # bounded ratio rmat[t,s] = P_t/P_s (<=1 for s<=t) replaces every u/P and
        # P/P division -- overflow-safe and bf16-friendly (Codex review).
        logP = torch.cumsum(torch.log(b.clamp_min(1e-9)), dim=2)  # [B,H,C,1]
        P = torch.exp(logP)
        tri = tri_full[:C, :C]  # s<=t (cached)
        lp = logP.squeeze(-1)  # [B,H,C]
        rmat = torch.exp((lp[..., :, None] - lp[..., None, :]).clamp_max(0.0))
        rmat = rmat * tri.view(1, 1, C, C)  # [B,H,C,C] = P_t/P_s for s<=t
        # inner code: stale from M0, then refine to z_t = M_{t-1} @ w_t using
        # the in-chunk segmented-product reconstruction of M_{t-1}.
        M0W = torch.einsum("bhmd,bhcd->bhcm", Mstate, W)  # M0 @ w_t  (= stale z)
        if input_gate is not None:
            # Input-conditioned affine write: u_t = a(x_t) ⊙ z_t − alpha_t, with
            # z_t = M_{t-1} @ w_t reconstructed EXACTLY inside the chunk. Because
            # u_t is affine in z_t (a from the token input, not the state code),
            # one FORWARD-substitution pass over the chunk is exact -- no Jacobi
            # iteration and no VJP graph. z_t uses the same segmented-product
            # reconstruction as the refine path: z_t = P_{t-1}(M0@w_t − gamma
            # sum_{s<t} (w_t·w_s)(u_s/P_s)). This reproduces the sequential
            # recurrence; the readout/advance below are already exact given u.
            Pprev = P / (b + 1e-12)  # P_{t-1}  [B,H,C,1]
            WW = torch.einsum("bhtd,bhsd->bhts", W, W)  # w_t . w_s  [B,H,C,C]
            strict = torch.tril(torch.ones(C, C, device=dev, dtype=dt), diagonal=-1)
            a_g = input_gate[:, :, c0:c1, :]  # [B,H,C,M]
            gval = g.reshape(1, H, 1, 1, 1)  # per-head gamma
            # The chunk innovation solves (I + Λ_m) u_m = rhs_m per slot m, a
            # strictly-lower-triangular (nilpotent) system. Λ_m and rhs_m differ
            # per slot only through the per-slot gate a_{t,m}; the coupling
            # coef[t,s] = (P_{t-1}/P_s)(w_t·w_s) is shared. Solve all slots at
            # once with a batched unit-lower-triangular solve -- this replaces the
            # length-C token loop with one matmul-class kernel, leaving only the
            # T/C chunk loop sequential (the DeltaNet chunk structure).
            lp = logP.squeeze(-1)  # log P_t  [B,H,C]
            lpprev = lp - torch.log(b.clamp_min(1e-9)).squeeze(-1)  # log P_{t-1}
            ratio = torch.exp((lpprev[..., :, None] - lp[..., None, :]).clamp_max(0.0))
            coef = ratio * WW * strict.view(1, 1, C, C)  # (P_{t-1}/P_s) w_t·w_s
            # Λ[b,h,m,t,s] = γ a_{t,m} coef[t,s]      (strict lower, per slot)
            Lam = gval * a_g.permute(0, 1, 3, 2).unsqueeze(-1) * coef.unsqueeze(2)
            eye = torch.eye(C, device=dev, dtype=dt).view(1, 1, 1, C, C)
            Lmat = eye + Lam  # [B,H,M,C,C] unit-lower-triangular
            rhs = a_g.permute(0, 1, 3, 2) * Pprev.squeeze(-1).unsqueeze(
                2
            ) * M0W.permute(0, 1, 3, 2) - A.permute(
                0, 1, 3, 2
            )  # [B,H,M,C]
            u = torch.linalg.solve_triangular(
                Lmat, rhs.unsqueeze(-1), upper=False, unitriangular=True
            ).squeeze(
                -1
            )  # [B,H,M,C]
            u = u.permute(0, 1, 3, 2)  # [B,H,C,M]
            if G is not None:
                u = u * G
            if read_mode == "M_q":
                M0q = torch.einsum("bhmd,bhcd->bhcm", Mstate, R)
                S = torch.einsum("bhtd,bhsd->bhts", R, W) * rmat
                term2 = torch.einsum("bhts,bhsm->bhtm", S, u)
                y = P * M0q - g * term2
            elif read_mode == "M_T_r":
                first = P * torch.einsum("bhcm,bhmd->bhcd", R, Mstate)
                Gr = torch.einsum("bhtm,bhsm->bhts", R, u) * rmat
                term2 = torch.einsum("bhts,bhsd->bhtd", Gr, W)
                y = first - g * term2
            else:
                raise ValueError(read_mode)
            outs.append(y)
            Plast = P[:, :, -1:, :]
            rC = rmat[:, :, -1, :].unsqueeze(-1)
            upd = torch.einsum("bhcm,bhcd->bhmd", u * rC, W)
            Mstate = Plast * Mstate - g * upd
            continue
        u = _vjp(M0W, A)
        if residual_update_mix:
            u = u + residual_update_mix * _trellis_residual(phi, M0W, A)
        if G is not None:
            u = u * G
        if _delta_ratio_cap_enabled(trellis_update_stabilizer, trellis_delta_ratio_cap):
            u, _ = _apply_delta_ratio_cap_to_u(
                u,
                W,
                g,
                Mstate,
                trellis_delta_ratio_cap,
                trellis_state_rms_floor,
                detach_scale=trellis_stabilizer_detach_scale,
            )
        if refine_passes > 0 and C > 1:
            # exact-mode only (not the ladder path); kept in the original u/P
            # form, which is verified-exact. The bounded rewrite above covers
            # the readouts + state advance that the true-stale ladder uses.
            Pprev = P / (b + 1e-12)  # P_{t-1}
            WW = torch.einsum("bhtd,bhsd->bhts", W, W)  # w_t . w_s
            strict = torch.tril(torch.ones(C, C, device=dev, dtype=dt), diagonal=-1)
            WWs = WW * strict.view(1, 1, C, C)  # s < t
            for _ in range(refine_passes):
                Util = u / (P + 1e-12)
                corr = torch.einsum("bhts,bhsm->bhtm", WWs, Util)  # sum_{s<t}
                z = Pprev * (M0W - g * corr)  # z_t = M_{t-1} @ w_t
                u = _vjp(z, A)
                if residual_update_mix:
                    u = u + residual_update_mix * _trellis_residual(phi, z, A)
                if G is not None:
                    u = u * G
                if _delta_ratio_cap_enabled(
                    trellis_update_stabilizer, trellis_delta_ratio_cap
                ):
                    u, _ = _apply_delta_ratio_cap_to_u(
                        u,
                        W,
                        g,
                        Mstate,
                        trellis_delta_ratio_cap,
                        trellis_state_rms_floor,
                        detach_scale=trellis_stabilizer_detach_scale,
                    )
        if read_mode == "M_q":
            M0q = torch.einsum("bhmd,bhcd->bhcm", Mstate, R)  # [B,H,C,M]
            S = torch.einsum("bhtd,bhsd->bhts", R, W) * rmat  # (q_t.w_s)(P_t/P_s)
            term2 = torch.einsum("bhts,bhsm->bhtm", S, u)  # sum_{s<=t}
            y = P * M0q - g * term2  # [B,H,C,M]
        elif read_mode == "M_T_r":
            first = P * torch.einsum("bhcm,bhmd->bhcd", R, Mstate)  # P_t (r_t@M0)
            G = torch.einsum("bhtm,bhsm->bhts", R, u) * rmat  # (r_t.u_s)(P_t/P_s)
            term2 = torch.einsum("bhts,bhsd->bhtd", G, W)
            y = first - g * term2  # [B,H,C,D]
        else:
            raise ValueError(read_mode)
        outs.append(y)
        # advance state: M_end = P_C M0 - gamma sum_s (P_C/P_s) u_s w_s
        Plast = P[:, :, -1:, :]  # [B,H,1,1]
        rC = rmat[:, :, -1, :].unsqueeze(-1)  # [B,H,C,1] = P_C/P_s
        upd = torch.einsum("bhcm,bhcd->bhmd", u * rC, W)
        Mstate = Plast * Mstate - g * upd
    return torch.cat(outs, dim=2)  # [B,H,T,*]
