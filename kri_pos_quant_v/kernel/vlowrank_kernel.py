#!/usr/bin/env python3
"""Triton low-rank-V decode attention (the V6 read-path accumulation trick).

Standard decode attention reads every past V_i [head_dim] to form
o = sum_i softmax(q.k_i) V_i. The V6 result stores V at rank r (r=32 << head_dim
128) as coefficients proj_i = B^T v_i with a per-head basis B; the value
contribution is linear in B, so the weighted sum accumulates in r-dim and lifts
once:  o = B ( sum_i alpha_i proj_i ). The K side (scores) is unchanged.

This kernel fuses the score + online-softmax + r-dim accumulation in one pass over
the context, reading K [T,dh] (full, for scores) and proj [T,r] (NOT V [T,dh]) for
the value aggregation -- head_dim/r = 4x less V read + a 4x smaller accumulator.
It outputs za [nq, r] per head; the [dh,r] lift (za @ B^T) is a cheap per-head
GEMV done outside. The win is the V-read/compute reduction on the ~99% of tokens
stored low-rank.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lowrank_decode(
    Q_ptr,  # bf16 [Hkv, nq, dh]
    K_ptr,  # bf16 [Hkv, T, dh]
    P_ptr,  # fp32 [Hkv, T, r]
    O_ptr,  # fp32 [Hkv, nq, r]  (za, pre-lift)
    scale,
    nq,
    T,
    dh,
    r,
    BLOCK_T: tl.constexpr,
    DH: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid // nq
    qi = pid % nq
    dvec = tl.arange(0, DH)
    rvec = tl.arange(0, R)
    q = tl.load(Q_ptr + head * nq * dh + qi * dh + dvec, mask=dvec < dh, other=0.0)
    q = q.to(tl.float32)
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([R], dtype=tl.float32)
    for t0 in range(0, T, BLOCK_T):
        tt = t0 + tl.arange(0, BLOCK_T)
        mask_t = tt < T
        # K block [BLOCK_T, DH]
        k_ptrs = K_ptr + head * T * dh + tt[:, None] * dh + dvec[None, :]
        k = tl.load(
            k_ptrs, mask=mask_t[:, None] & (dvec[None, :] < dh), other=0.0
        ).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_T]
        s = tl.where(mask_t, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        p = tl.exp(s - m_new)  # [BLOCK_T]
        corr = tl.exp(m - m_new)
        l = l * corr + tl.sum(p, axis=0)
        # proj block [BLOCK_T, R]
        p_ptrs = P_ptr + head * T * r + tt[:, None] * r + rvec[None, :]
        pj = tl.load(
            p_ptrs, mask=mask_t[:, None] & (rvec[None, :] < r), other=0.0
        )
        acc = acc * corr + tl.sum(p[:, None] * pj, axis=0)  # [R]
        m = m_new
    za = acc / l
    o_ptrs = O_ptr + head * nq * r + qi * r + rvec
    tl.store(o_ptrs, za, mask=rvec < r)


@triton.jit
def _dense_decode(
    Q_ptr,  # bf16 [Hkv, nq, dh]
    K_ptr,  # bf16 [Hkv, T, dh]
    V_ptr,  # bf16 [Hkv, T, dh]
    O_ptr,  # fp32 [Hkv, nq, dh]
    scale,
    nq,
    T,
    dh,
    BLOCK_T: tl.constexpr,
    DH: tl.constexpr,
):
    """Fusion-matched dense flash-decode: identical structure to _lowrank_decode
    but accumulates o[DH] reading V[T,dh] -- so the latency delta vs the low-rank
    kernel isolates the V-read/accumulator reduction (dh vs r), not fusion."""
    pid = tl.program_id(0)
    head = pid // nq
    qi = pid % nq
    dvec = tl.arange(0, DH)
    q = tl.load(Q_ptr + head * nq * dh + qi * dh + dvec, mask=dvec < dh, other=0.0)
    q = q.to(tl.float32)
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([DH], dtype=tl.float32)
    for t0 in range(0, T, BLOCK_T):
        tt = t0 + tl.arange(0, BLOCK_T)
        mask_t = tt < T
        k_ptrs = K_ptr + head * T * dh + tt[:, None] * dh + dvec[None, :]
        k = tl.load(
            k_ptrs, mask=mask_t[:, None] & (dvec[None, :] < dh), other=0.0
        ).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(mask_t, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        corr = tl.exp(m - m_new)
        l = l * corr + tl.sum(p, axis=0)
        v_ptrs = V_ptr + head * T * dh + tt[:, None] * dh + dvec[None, :]
        v = tl.load(
            v_ptrs, mask=mask_t[:, None] & (dvec[None, :] < dh), other=0.0
        ).to(tl.float32)
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    o_ptrs = O_ptr + head * nq * dh + qi * dh + dvec
    tl.store(o_ptrs, acc / l, mask=dvec < dh)


def dense_decode_triton(Q, K, V, BLOCK_T=128):
    """Fusion-matched dense flash-decode baseline -> o [Hkv,nq,dh] fp32."""
    Hkv, nq, dh = Q.shape
    T = K.shape[1]
    scale = 1.0 / (dh ** 0.5)
    DH = triton.next_power_of_2(dh)
    o = torch.empty(Hkv, nq, dh, device=Q.device, dtype=torch.float32)
    grid = (Hkv * nq,)
    _dense_decode[grid](Q, K, V, o, scale, nq, T, dh, BLOCK_T=BLOCK_T, DH=DH)
    return o


def lowrank_decode(Q, K, proj, Bbasis, BLOCK_T=128):
    """Q [Hkv,nq,dh] bf16, K [Hkv,T,dh] bf16, proj [Hkv,T,r] fp32,
    Bbasis [Hkv,dh,r] fp32 -> o [Hkv,nq,dh] fp32."""
    Hkv, nq, dh = Q.shape
    T = K.shape[1]
    r = proj.shape[2]
    scale = 1.0 / (dh ** 0.5)
    DH = triton.next_power_of_2(dh)
    R = triton.next_power_of_2(r)
    za = torch.empty(Hkv, nq, r, device=Q.device, dtype=torch.float32)
    grid = (Hkv * nq,)
    _lowrank_decode[grid](
        Q, K, proj, za, scale, nq, T, dh, r,
        BLOCK_T=BLOCK_T, DH=DH, R=R,
    )
    # per-head lift: o[h] = za[h] @ Bbasis[h]^T  -> [nq, dh]
    o = torch.bmm(za, Bbasis.transpose(1, 2))  # [Hkv, nq, dh]
    return o


@torch.no_grad()
def dense_decode(Q, K, V):
    """Reference/baseline dense decode attention -> o [Hkv,nq,dh] fp32."""
    dh = Q.shape[-1]
    scale = 1.0 / (dh ** 0.5)
    s = torch.einsum("hqd,htd->hqt", Q.float(), K.float()) * scale
    a = torch.softmax(s, dim=-1)
    return torch.einsum("hqt,htd->hqd", a, V.float())


@torch.no_grad()
def lowrank_decode_ref(Q, K, proj, Bbasis):
    """Torch reference for the low-rank path (validation oracle)."""
    dh = Q.shape[-1]
    scale = 1.0 / (dh ** 0.5)
    s = torch.einsum("hqd,htd->hqt", Q.float(), K.float()) * scale
    a = torch.softmax(s, dim=-1)
    za = torch.einsum("hqt,htr->hqr", a, proj.float())
    return torch.bmm(za, Bbasis.transpose(1, 2))
