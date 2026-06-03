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
        # u_t = J_phi(z)^T (phi(z) - alpha)  via exact VJP
        # Build the double-graph only when z is actually in a grad graph (real
        # training); otherwise compute the VJP on a detached, grad-enabled copy
        # so it still works for plain-tensor / eval calls.
        make_graph = create_graph and z.requires_grad
        z_req = z if z.requires_grad else z.detach().requires_grad_(True)
        with torch.enable_grad():
            pred = phi(z_req)                        # [B,H,M]
            err = pred - a
            (u,) = torch.autograd.grad(
                pred, z_req, grad_outputs=err,
                create_graph=make_graph, retain_graph=True,
            )
        if not make_graph:
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
