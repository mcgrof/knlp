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

from .activations import ln_silu, ln_silu_vjp, ln_silu_vjp_from_alpha


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
            (u,) = torch.autograd.grad(
                pred,
                z_req,
                grad_outputs=err,
                create_graph=cg,
                retain_graph=True,
            )
        if not cg:
            u = u.detach()
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
    # state accumulates over chunks -> keep it fp32 even under bf16 autocast
    # (Codex review); the matmuls still run bf16 via autocast for the speed.
    Mstate = torch.zeros(B, H, M, D, device=dev, dtype=torch.float32)
    g = gamma.view(1, H, 1, 1)
    outs = []

    def _vjp(zin, A):
        z = zin.detach()  # cut M 2nd-order
        if phi is ln_silu:
            # fused closed-form phi(z) + J_phi^T(phi(z)-A): exact, no per-chunk
            # autograd graph (the kernel's dominant overhead) and no double
            # LN/SiLU. Linear in A, so the outer backward to alpha_proj flows.
            return ln_silu_vjp_from_alpha(z, A)
        zr = z.requires_grad_(True)
        cg = A.requires_grad
        with torch.enable_grad():
            pred = phi(zr)
            err = pred - A  # keep alpha -> alpha_proj trains
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
        u = _vjp(M0W, A)
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
