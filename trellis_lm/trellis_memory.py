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


def run_trellis_memory(
    write: torch.Tensor,        # [B,H,T,D]
    read: torch.Tensor,         # [B,H,T,D] (key pass) or [B,H,T,M] (value pass)
    alpha: torch.Tensor,        # [B,H,T,M]
    beta: torch.Tensor,         # [B,H,T,1] or [B,H,T,M]
    gamma: torch.Tensor,        # [H] positive
    phi,                        # callable over last dim (M)
    read_mode: str,             # "M_q" | "M_T_r"
    training: bool,
    exact_inner: bool = True,   # True: backprop through the inner VJP (2nd order,
                                # slow/exact). False: stale-gradient (detach u from
                                # the param graph) — the sanctioned fast mode.
    M_init: Optional[torch.Tensor] = None,   # [B,H,M,D] carried state (generation)
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
        w = write[:, :, t, :]                       # [B,H,D]
        a = alpha[:, :, t, :]                        # [B,H,M]
        b = beta[:, :, t, :]                         # [B,H,1] or [B,H,M]
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
            pred = phi(z_req)                        # [B,H,M]
            err = pred - a                           # alpha stays in graph
            (u,) = torch.autograd.grad(
                pred, z_req, grad_outputs=err,
                create_graph=cg, retain_graph=True,
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
            y = torch.einsum("bhmd,bhd->bhm", Mstate, r)   # [B,H,M]
        elif read_mode == "M_T_r":
            y = torch.einsum("bhmd,bhm->bhd", Mstate, r)   # [B,H,D]
        else:
            raise ValueError(read_mode)
        outs.append(y)

    out = torch.stack(outs, dim=2)                   # [B,H,T,*]
    if return_state:
        return out, Mstate
    return out


def run_trellis_memory_chunked(
    write: torch.Tensor,        # [B,H,T,D]
    read: torch.Tensor,         # [B,H,T,D] (key) or [B,H,T,M] (value)
    alpha: torch.Tensor,        # [B,H,T,M]
    beta: torch.Tensor,         # [B,H,T,1]  (per-head only)
    gamma: torch.Tensor,        # [H]
    phi,
    read_mode: str,             # "M_q" | "M_T_r"
    chunk_size: int,
    refine_passes: int = 2,
):
    """Faithful chunkwise form of run_trellis_memory (per-head beta only).

    The state recurrence within a chunk is handled exactly via segmented decay
    products (Mstate is reconstructed as M_t = P_t*M0 - gamma*sum_{s<=t}
    (P_t/P_s) u_s w_s), which turns readout + update into matmuls. The only
    approximation is the inner code z_t = M_{t-1} @ w_t: a "true-stale" version
    uses M0 for all t (cheap but lossy). Here we REFINE it -- reconstruct
    M_{t-1} from the in-chunk updates and recompute z (and u) `refine_passes`
    times -- which recovers most of the sequential quality. Exact (==sequential
    stale) at chunk_size=1 for any refine_passes (the in-chunk correction is
    empty). u is detached from M (no 2nd-order) but keeps the alpha gradient;
    outer backprop flows through the chunkwise matmuls. T/C sequential steps.
    """
    assert beta.shape[-1] == 1, "chunked path supports per-head beta only"
    B, H, T, D = write.shape
    M = alpha.shape[-1]
    dev, dt = write.device, write.dtype
    Mstate = torch.zeros(B, H, M, D, device=dev, dtype=dt)
    g = gamma.view(1, H, 1, 1)
    outs = []

    def _vjp(zin, A):
        zr = zin.detach().requires_grad_(True)    # cut M 2nd-order
        cg = A.requires_grad
        with torch.enable_grad():
            pred = phi(zr)
            err = pred - A                        # keep alpha -> alpha_proj trains
            (uu,) = torch.autograd.grad(pred, zr, grad_outputs=err,
                                        create_graph=cg, retain_graph=True)
        return uu if cg else uu.detach()

    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        C = c1 - c0
        W = write[:, :, c0:c1, :]                 # [B,H,C,D]
        R = read[:, :, c0:c1, :]                  # [B,H,C,*]
        A = alpha[:, :, c0:c1, :]                 # [B,H,C,M]
        b = beta[:, :, c0:c1, :]                  # [B,H,C,1]
        # cumulative inclusive decay P_t = prod_{i<=t} b_i  (per head)
        P = torch.cumprod(b, dim=2)                           # [B,H,C,1]
        tri = torch.tril(torch.ones(C, C, device=dev, dtype=dt))  # s<=t
        # inner code: stale from M0, then refine to z_t = M_{t-1} @ w_t using
        # the in-chunk segmented-product reconstruction of M_{t-1}.
        M0W = torch.einsum("bhmd,bhcd->bhcm", Mstate, W)      # M0 @ w_t  (= stale z)
        u = _vjp(M0W, A)
        if refine_passes > 0 and C > 1:
            Pprev = P / (b + 1e-12)                           # P_{t-1}
            WW = torch.einsum("bhtd,bhsd->bhts", W, W)        # w_t . w_s
            strict = torch.tril(torch.ones(C, C, device=dev, dtype=dt), diagonal=-1)
            WWs = WW * strict.view(1, 1, C, C)                # s < t
            for _ in range(refine_passes):
                Util = u / (P + 1e-12)
                corr = torch.einsum("bhts,bhsm->bhtm", WWs, Util)  # sum_{s<t}
                z = Pprev * (M0W - g * corr)                  # z_t = M_{t-1} @ w_t
                u = _vjp(z, A)
        if read_mode == "M_q":
            M0q = torch.einsum("bhmd,bhcd->bhcm", Mstate, R)  # [B,H,C,M]
            S = torch.einsum("bhtd,bhsd->bhts", R, W)         # S[t,s]=q_t.w_s
            S = S * tri.view(1, 1, C, C)
            Util = u / (P + 1e-12)                            # u_s / P_s
            term2 = torch.einsum("bhts,bhsm->bhtm", S, Util)  # sum_{s<=t}
            y = P * (M0q - g * term2)                         # [B,H,C,M]
        elif read_mode == "M_T_r":
            first = P * torch.einsum("bhcm,bhmd->bhcd", R, Mstate)   # P_t (r_t@M0)
            G = torch.einsum("bhtm,bhsm->bhts", R, u)         # G[t,s]=r_t.u_s
            Pmat = P.squeeze(-1).unsqueeze(-1) / (P.squeeze(-1).unsqueeze(-2) + 1e-12)  # [B,H,C(t),C(s)] = P_t/P_s
            G = G * Pmat * tri.view(1, 1, C, C)
            term2 = torch.einsum("bhts,bhsd->bhtd", G, W)
            y = first - g * term2                             # [B,H,C,D]
        else:
            raise ValueError(read_mode)
        outs.append(y)
        # advance state to chunk end: M_end = P_last M0 - gamma sum (P_last/P_s) u_s w_s
        Plast = P[:, :, -1:, :]                               # [B,H,1,1]
        coef = (Plast / (P + 1e-12))                          # [B,H,C,1]
        upd = torch.einsum("bhcm,bhcd->bhmd", u * coef, W)    # weighted outer sum
        Mstate = Plast * Mstate - g * upd
    return torch.cat(outs, dim=2)                             # [B,H,T,*]
