"""fla-backed fast path for the smd_identity arm.

smd_identity is a gated delta rule (the write gate G=I is a no-op), so its two
memory passes map onto flash-linear-attention's gated-delta operator under the
transform in PHASE1_FLA_SPIKE_FINDINGS.md: state S=M^T, so the fla key-axis is
our head_dim D and the fla value-axis is our slot count M, with
g_fla=log(beta), beta_fla=gamma/beta, v_fla=beta*alpha, scale=1.

The key pass reads M_q (contract D) which is fla's native per-token output. The
value pass reads M_T_r (contract the slot axis), which fla cannot emit
directly; it is recovered by a delta self-probe (query = the write key) plus a
gated-linear-attention read (PHASE2_FLA_FULL_LAYER_DESIGN.md):

    p_t = M_t v_t                                   # fla delta, query = write key
    w_t = gamma (beta*value_alpha - p_t) / (beta - gamma*||v||^2)   # WY innovation
    y_t = sum_{s<=t} (P_t/P_s)(w_s . r_t) v_s       # GLA: Q=r, K=w, V=v, g=log beta

Two backends: "fla" uses the Triton ops (CUDA only); "ref" reproduces the exact
decomposition with our own chunk kernel and a PyTorch GLA (device-agnostic, for
W7900 validation). Both return the same tensors as run_trellis_memory_chunked.
"""

from __future__ import annotations

import torch

from .activations import ln_silu
from .trellis_memory import run_trellis_memory_chunked


def _to_bt(x: torch.Tensor) -> torch.Tensor:
    # [B,H,T,*] -> [B,T,H,*] contiguous for fla
    return x.transpose(1, 2).contiguous()


def _from_bt(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _delta_fla(read, write, v_fla, g, beta_fla):
    """fla gated-delta native output o_t = read_t^T S_t (contract D)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    o = chunk_gated_delta_rule(
        _to_bt(read),
        _to_bt(write),
        _to_bt(v_fla),
        _to_bt(g).float(),  # arg order (q,k,v,g,beta) -- g and beta SWAPPED vs naive
        _to_bt(beta_fla).float(),
        scale=1.0,
    )[0]
    return _from_bt(o)


def _gla_fla(q, k, v, g):
    """fla scalar-decay gated linear attention, causal/inclusive."""
    from fla.ops.simple_gla import chunk_simple_gla

    o = chunk_simple_gla(
        _to_bt(q),
        _to_bt(k),
        _to_bt(v),
        _to_bt(g).float(),
        scale=1.0,
    )[0]
    return _from_bt(o)


def _gla_ref(q, k, v, g):
    """PyTorch scalar-decay GLA reference. g is per-head log-decay [B,H,T].

    y_t = sum_{s<=t} exp(cumsum_g_t - cumsum_g_s) (q_t . k_s) v_s  (s=t term undecayed).
    """
    T = g.shape[-1]
    logP = torch.cumsum(g, dim=-1)  # [B,H,T]
    diff = logP[:, :, :, None] - logP[:, :, None, :]  # log(P_t/P_s)
    # mask the exponent BEFORE exp so the acausal upper triangle is exactly 0
    # (exp(-inf)) with clean gradients -- masking after exp would leave inf*0=nan.
    causal = torch.ones(T, T, dtype=torch.bool, device=g.device).tril()
    diff = diff.masked_fill(~causal, float("-inf"))
    A = torch.exp(diff)  # P_t/P_s, causal inclusive
    sim = torch.einsum("bhtm,bhsm->bhts", q, k)
    return torch.einsum("bhts,bhsd->bhtd", A * sim, v)


def smd_identity_pass(
    write, read, alpha, beta, gamma, read_mode, chunk_size, backend="fla"
):
    """Compute one smd_identity memory pass, matching run_trellis_memory_chunked
    with an all-ones input_gate. read_mode "M_q" (key pass -> [B,H,T,M]) or
    "M_T_r" (value pass -> [B,H,T,D]). beta is [B,H,T,1], gamma is [H]."""
    B, Hh, T, D = write.shape
    M = alpha.shape[-1]
    g = torch.log(beta.clamp_min(1e-9)).squeeze(-1)  # [B,H,T] log-decay
    beta_fla = gamma.view(1, Hh, 1) / beta.squeeze(-1)  # [B,H,T]
    v_fla = beta * alpha  # [B,H,T,M]

    if read_mode == "M_q":
        # key pass: native fla output = M_t @ read
        if backend == "fla":
            return _delta_fla(read, write, v_fla, g, beta_fla)
        # ref: our exact kernel
        return run_trellis_memory_chunked(
            write,
            read,
            alpha,
            beta,
            gamma,
            ln_silu,
            "M_q",
            chunk_size,
            input_gate=write.new_ones(B, Hh, T, M),
        )

    if read_mode == "M_T_r":
        # value pass: delta self-probe (query = write key) -> p = M_t @ write
        if backend == "fla":
            p = _delta_fla(write, write, v_fla, g, beta_fla)
        else:
            p = run_trellis_memory_chunked(
                write,
                write,
                alpha,
                beta,
                gamma,
                ln_silu,
                "M_q",
                chunk_size,
                input_gate=write.new_ones(B, Hh, T, M),
            )
        n = (write * write).sum(dim=-1, keepdim=True)  # ||v||^2 [B,H,T,1]
        denom = beta - gamma.view(1, Hh, 1, 1) * n
        w = gamma.view(1, Hh, 1, 1) * (beta * alpha - p) / denom  # [B,H,T,M]
        # read along V: GLA Q=read(slots), K=w(slots), V=write(D), g=log beta
        gla = _gla_fla if backend == "fla" else _gla_ref
        return gla(read, w, write, g)

    raise ValueError(read_mode)
