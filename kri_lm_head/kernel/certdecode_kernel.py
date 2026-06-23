#!/usr/bin/env python3
"""Triton certified-decode kernel for the shadow-bound LM head.

Replaces the dense LM-head GEMV (H @ W_U^T over a 150k vocab) with a two-stage
certified decode that reads ~25% of the bytes losslessly:

  Stage 1 (this kernel): the SHADOW bound. For a decode hidden state h, with the
  offline shadow head aq[V,r] (int8, per-column scale) and delta[V], compute per
  token the valid upper bound  U_v = (aq_v . (scale*q)) + rho*delta_v + cst,
  where q = B^T h, rho = ||h - B q||, cst = aq_err_norm*||q||. Reduce U to idblock
  slab maxima U_b. This is an [V,r] int8 GEMV -- r<<d, int8<bf16 -- so it streams
  ~18% of the dense head's bytes.

  Stage 2 (host loop + exact GEMV): open idblock slabs in descending U_b, fetch
  the TRUE logits for each opened slab from W_U, track ell* = max fetched logit,
  stop (strict) when ell* > the largest U_b among unopened slabs. The dense argmax
  is then certified having touched only sink+opened rows (~7% mean on qwen-7b).

The bound is the same one validated losslessly in shadow_bound_replay.py
(argmax_match=1.0 at r=1280/int8); this module realizes its DEPLOYMENT cost (no
dense logits) and times it against the dense GEMV on the actual GPU.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _shadow_gemv(
    aq_ptr,  # int8 [V, r]
    qs_ptr,  # fp32 [r]   (= scale * q)
    delta_ptr,  # fp32 [V]
    u_ptr,  # fp32 [V]   output upper bound
    rho,  # fp32 scalar
    cst,  # fp32 scalar  (aq_err_norm * ||q||)
    V,
    r,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = row < V
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for r0 in range(0, r, BLOCK_R):
        rj = r0 + tl.arange(0, BLOCK_R)
        mask_r = rj < r
        a_ptrs = aq_ptr + row[:, None] * r + rj[None, :]
        a = tl.load(
            a_ptrs, mask=mask_v[:, None] & mask_r[None, :], other=0
        ).to(tl.float32)
        qv = tl.load(qs_ptr + rj, mask=mask_r, other=0.0)
        acc += tl.sum(a * qv[None, :], axis=1)
    d = tl.load(delta_ptr + row, mask=mask_v, other=0.0)
    tl.store(u_ptr + row, acc + rho * d + cst, mask=mask_v)


def shadow_upper_bound(aq, scale, delta, q, rho, aq_err_norm, BLOCK_V=64, BLOCK_R=256):
    """Per-token upper bound U[V] via the Triton shadow GEMV. q is [r] fp32."""
    V, r = aq.shape
    qs = (scale * q).contiguous()
    cst = float(aq_err_norm) * float(q.norm())
    U = torch.empty(V, device=aq.device, dtype=torch.float32)
    grid = (triton.cdiv(V, BLOCK_V),)
    _shadow_gemv[grid](
        aq, qs, delta, U, float(rho), cst, V, r,
        BLOCK_V=BLOCK_V, BLOCK_R=BLOCK_R,
    )
    return U


@triton.jit
def _shadow_slabmax_b1(
    aq_ptr,  # int8 [V, r]
    qs_ptr,  # fp32 [r]   (= scale * q)
    delta_ptr,  # fp32 [V]
    rho_ptr,  # fp32 [1]   DEVICE scalar (graph-safe, no .item())
    cst_ptr,  # fp32 [1]   DEVICE scalar (= aq_err_norm * ||q||)
    ub_ptr,  # fp32 [C]   output: per-slab upper-bound maxima (pre-filled -inf)
    V,
    r,
    S,
    C,
    BLOCK_R: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Graph-safe per-token (Bt=1) stage-1: writes the per-slab maxima Ub[C]
    DIRECTLY (no V-wide U intermediate), with rho/cst read from DEVICE scalars so
    the whole thing is CUDA-graph capturable (no host sync). One program per
    (slab, row-subblock); atomic-max into ub[slab]. fp32 throughout (the
    certificate path), so no bf16-rounding margin is needed."""
    slab = tl.program_id(0)
    sub = tl.program_id(1)
    rho = tl.load(rho_ptr)
    cst = tl.load(cst_ptr)
    vloc = sub * BLOCK_V + tl.arange(0, BLOCK_V)
    row = slab * S + vloc
    mask_v = (vloc < S) & (row < V)
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for r0 in range(0, r, BLOCK_R):
        rj = r0 + tl.arange(0, BLOCK_R)
        mask_r = rj < r
        a = tl.load(aq_ptr + row[:, None] * r + rj[None, :],
                    mask=mask_v[:, None] & mask_r[None, :], other=0).to(tl.float32)
        qv = tl.load(qs_ptr + rj, mask=mask_r, other=0.0)
        acc += tl.sum(a * qv[None, :], axis=1)
    d = tl.load(delta_ptr + row, mask=mask_v, other=0.0)
    U = acc + rho * d + cst
    U = tl.where(mask_v, U, -float("inf"))
    local = tl.max(U, axis=0)
    tl.atomic_max(ub_ptr + slab, local)


def shadow_slabmax_b1(aq, qs, delta, rho_t, cst_t, S, ub_out, BLOCK_R=128, BLOCK_V=64):
    """Graph-safe Ub[C] for one token. qs=scale*q [r], rho_t/cst_t are [1] device
    scalars, ub_out [C] is pre-filled -inf by the caller (so the fill is captured
    too). Returns ub_out."""
    V, r = aq.shape
    C = V // S
    grid = (C, triton.cdiv(S, BLOCK_V))
    _shadow_slabmax_b1[grid](aq, qs, delta, rho_t, cst_t, ub_out, V, r, S, C,
                             BLOCK_R=BLOCK_R, BLOCK_V=BLOCK_V)
    return ub_out


@triton.jit
def _shadow_slabmax_batched(
    aq_ptr,  # int8 [V, r]  (exact in bf16)
    qsbf_ptr,  # bf16 [Bt, r]  (= bf16(scale * Q), the WMMA operand)
    delta_ptr,  # fp32 [V]
    rho_ptr,  # fp32 [Bt]
    cst_ptr,  # fp32 [Bt]
    aql2_ptr,  # fp32 [V]   per-row ||aq_v||_2
    epsx_ptr,  # fp32 [Bt]  ||scale*Q_b - bf16(scale*Q_b)||_2
    ub_ptr,  # fp32 [Bt, C]  output: per-slab upper-bound maxima
    V,
    r,
    Bt,
    S,
    C,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BT: tl.constexpr,
):
    """One program per (slab, row-subblock). The score dot runs in BF16 WMMA --
    RDNA3 has matrix units for f16/bf16/int8 but NOT fp32, so the old ieee-fp32
    tl.dot fell back to the vector ALUs (~32x the bandwidth floor). aq is exact in
    bf16 (int8 in [-127,127]); only the online operand scale*Q is bf16-rounded, so
    a certified margin ||aq_v||*||x-bf16(x)|| = aql2[v]*epsx[b] is added back to
    keep U_v a valid upper bound. Output is the tiny [Bt,C] via an atomic slab-max
    (ub pre-filled -inf)."""
    slab = tl.program_id(0)
    sub = tl.program_id(1)
    b = tl.arange(0, BT)
    mask_b = b < Bt
    rho = tl.load(rho_ptr + b, mask=mask_b, other=0.0)
    cst = tl.load(cst_ptr + b, mask=mask_b, other=0.0)
    epsx = tl.load(epsx_ptr + b, mask=mask_b, other=0.0)
    vloc = sub * BLOCK_V + tl.arange(0, BLOCK_V)
    row = slab * S + vloc
    mask_v = (vloc < S) & (row < V)
    acc = tl.zeros([BT, BLOCK_V], dtype=tl.float32)
    for r0 in range(0, r, BLOCK_R):
        rj = r0 + tl.arange(0, BLOCK_R)
        mask_r = rj < r
        a = tl.load(
            aq_ptr + row[:, None] * r + rj[None, :],
            mask=mask_v[:, None] & mask_r[None, :], other=0,
        ).to(tl.bfloat16)  # [BLOCK_V, BLOCK_R] exact
        x = tl.load(
            qsbf_ptr + b[:, None] * r + rj[None, :],
            mask=mask_b[:, None] & mask_r[None, :], other=0.0,
        )  # [BT, BLOCK_R] bf16
        acc += tl.dot(x, tl.trans(a), out_dtype=tl.float32)  # BF16 WMMA -> fp32
    d = tl.load(delta_ptr + row, mask=mask_v, other=0.0)
    aql2 = tl.load(aql2_ptr + row, mask=mask_v, other=0.0)
    # certified upper bound incl. the bf16 q-rounding margin aql2[v]*epsx[b]
    U = acc + rho[:, None] * d[None, :] + cst[:, None] + epsx[:, None] * aql2[None, :]
    U = tl.where(mask_v[None, :], U, -float("inf"))
    local = tl.max(U, axis=1)  # [BT]
    tl.atomic_max(ub_ptr + b * C + slab, local, mask=mask_b)


def shadow_slab_bounds(aq, scale, delta, Q, rho, aq_err_norm, S, aq_l2=None,
                       BLOCK_V=64, BLOCK_R=128):
    """Batched per-slab shadow upper bounds U_b [Bt, C] directly (slab-max fused).
    Dispatches Bt==1 to the simple memory-bound per-token kernel. The batched dot
    runs in BF16 WMMA (RDNA3 has no fp32 matrix unit) with a certified margin for
    the bf16-rounded online operand. aq_l2 (per-row ||aq||_2) is a property of the
    offline shadow head -- pass it in to avoid recomputing per call."""
    V, r = aq.shape
    Bt = Q.shape[0]
    C = V // S
    if Bt == 1:
        U = shadow_upper_bound(aq, scale, delta, Q[0], float(rho[0]), aq_err_norm)
        return U.view(C, S).amax(1).unsqueeze(0)  # [1, C]
    qs = scale.unsqueeze(0) * Q  # [Bt, r] fp32
    qsbf = qs.to(torch.bfloat16).contiguous()
    epsx = (qs - qsbf.float()).norm(dim=1).contiguous()  # [Bt] bf16-round residual
    cst = (aq_err_norm * Q.norm(dim=1)).contiguous()  # [Bt]
    if aq_l2 is None:
        aq_l2 = aq.float().pow(2).sum(1).sqrt().contiguous()  # [V]
    BT = max(16, triton.next_power_of_2(Bt))
    ub = torch.full((Bt, C), float("-inf"), device=aq.device, dtype=torch.float32)
    n_sub = triton.cdiv(S, BLOCK_V)
    _shadow_slabmax_batched[(C, n_sub)](
        aq, qsbf, delta, rho.contiguous(), cst, aq_l2, epsx, ub, V, r, Bt, S, C,
        BLOCK_V=BLOCK_V, BLOCK_R=BLOCK_R, BT=BT,
    )
    return ub


@triton.jit
def _shadow_slabmax_perrow(
    aq_ptr, qsbf_ptr, delta_ptr, rho_ptr, qn_ptr, aqerr_row_ptr, aql2_ptr,
    epsx_ptr, ub_ptr, V, r, Bt, S, C,
    BLOCK_V: tl.constexpr, BLOCK_R: tl.constexpr, BT: tl.constexpr,
):
    """Per-ROW-error batched shadow slab-max: like _shadow_slabmax_batched but the
    quantization slack is qn[b]*aq_err_up[v] (per row) instead of a scalar cst --
    the scalar is dominated by one bad row and taxes every token; per-row tightens
    the bound. int8 aq, bf16 WMMA dot + bf16-rounding margin, atomic slab-max."""
    slab = tl.program_id(0)
    sub = tl.program_id(1)
    b = tl.arange(0, BT)
    mask_b = b < Bt
    rho = tl.load(rho_ptr + b, mask=mask_b, other=0.0)
    qn = tl.load(qn_ptr + b, mask=mask_b, other=0.0)
    epsx = tl.load(epsx_ptr + b, mask=mask_b, other=0.0)
    vloc = sub * BLOCK_V + tl.arange(0, BLOCK_V)
    row = slab * S + vloc
    mask_v = (vloc < S) & (row < V)
    acc = tl.zeros([BT, BLOCK_V], dtype=tl.float32)
    for r0 in range(0, r, BLOCK_R):
        rj = r0 + tl.arange(0, BLOCK_R)
        mask_r = rj < r
        a = tl.load(aq_ptr + row[:, None] * r + rj[None, :],
                    mask=mask_v[:, None] & mask_r[None, :], other=0).to(tl.bfloat16)
        x = tl.load(qsbf_ptr + b[:, None] * r + rj[None, :],
                    mask=mask_b[:, None] & mask_r[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(a), out_dtype=tl.float32)
    d = tl.load(delta_ptr + row, mask=mask_v, other=0.0)
    er = tl.load(aqerr_row_ptr + row, mask=mask_v, other=0.0)
    aql2 = tl.load(aql2_ptr + row, mask=mask_v, other=0.0)
    U = (acc + rho[:, None] * d[None, :] + qn[:, None] * er[None, :]
         + epsx[:, None] * aql2[None, :])
    U = tl.where(mask_v[None, :], U, -float("inf"))
    local = tl.max(U, axis=1)
    tl.atomic_max(ub_ptr + b * C + slab, local, mask=mask_b)


def shadow_slab_bounds_perrow(aq, scale, delta, Q, rho, aq_err_row, S, aq_l2=None,
                              BLOCK_V=64, BLOCK_R=128):
    """Batched Ub[Bt,C] from the int8 shadow with PER-ROW quant error aq_err_row[V]
    (= ||W_U[v]@B - dequant(aq[v])||). Reads int8 aq (not fp32 a_q), slab-max fused,
    no [Bt,V] materialization. Returns ub[Bt,C]."""
    V, r = aq.shape
    Bt = Q.shape[0]
    C = V // S
    qs = scale.unsqueeze(0) * Q
    qsbf = qs.to(torch.bfloat16).contiguous()
    epsx = (qs - qsbf.float()).norm(dim=1).contiguous()
    qn = Q.norm(dim=1).contiguous()
    if aq_l2 is None:
        aq_l2 = aq.float().pow(2).sum(1).sqrt().contiguous()
    BT = max(16, triton.next_power_of_2(Bt))
    ub = torch.full((Bt, C), float("-inf"), device=aq.device, dtype=torch.float32)
    n_sub = triton.cdiv(S, BLOCK_V)
    _shadow_slabmax_perrow[(C, n_sub)](
        aq, qsbf, delta, rho.contiguous(), qn, aq_err_row.contiguous(), aq_l2, epsx,
        ub, V, r, Bt, S, C, BLOCK_V=BLOCK_V, BLOCK_R=BLOCK_R, BT=BT,
    )
    return ub


def shadow_upper_bound_ref(aq, scale, delta, q, rho, aq_err_norm):
    """Torch reference for the bound (validation oracle)."""
    qs = scale * q
    a_deq = aq.float() * qs.unsqueeze(0)  # [V,r] dequant*q
    U = a_deq.sum(1) + rho * delta + aq_err_norm * float(q.norm())
    return U


@torch.no_grad()
def certified_decode(h, B, aq, scale, delta, W_U, S, aq_err_norm, use_kernel=True):
    """One token: returns (argmax_id, fetched_rows, U_b). Stage-1 shadow bound via
    the Triton kernel (or torch ref), Stage-2 greedy slab fetch with TRUE W_U for
    a lossless certificate. S = rows per (contiguous idblock) slab; C = V/S."""
    device = h.device
    V, d = W_U.shape
    C = V // S
    q = (h.float() @ B)  # [r]
    rho = (h.float() - q @ B.t()).norm()
    if use_kernel:
        U = shadow_upper_bound(aq, scale, delta, q, rho, aq_err_norm)
    else:
        U = shadow_upper_bound_ref(aq, scale, delta, q, rho, aq_err_norm)
    U_b = U.view(C, S).amax(1)  # slab upper bounds [C]
    order = U_b.argsort(descending=True)
    ell = torch.tensor(float("-inf"), device=device)
    best_id = -1
    fetched = 0
    hf = h.float()
    for i in range(C):
        b = int(order[i])
        # certify BEFORE opening: if current incumbent already beats this slab's
        # upper bound, every remaining (lower) slab too -> stop, argmax certified.
        if ell > U_b[b]:
            break
        lo = b * S
        rows = W_U[lo : lo + S].float()  # TRUE weights -> exact logits
        logits = rows @ hf  # [S]
        m, j = logits.max(0)
        if m > ell:
            ell = m
            best_id = lo + int(j)
        fetched += S
    return best_id, fetched, U_b


@torch.no_grad()
def certified_decode_waves(h, B, aq, scale, delta, W_U, S, aq_err_norm,
                           wave=8, fallback_frac=0.30, WUt=None, use_kernel=True):
    """Certified lossless decode with a FUSED, BOUNDED stage-2: open slabs in WAVES
    of `wave` (one gathered GEMV per wave, not a per-slab loop), and if a token has
    not certified after fetching `fallback_frac` of the vocab, bail to the exact
    dense GEMV (still lossless -- dense argmax is exact). The waves cut the launch
    count; the fallback caps the tail at ~dense cost. Returns (argmax, fetched, n)."""
    device = h.device
    V, d = W_U.shape
    C = V // S
    cap_rows = int(fallback_frac * V)
    hf = h.float()
    q = hf @ B
    # exact residual; the sqrt(||h||^2-||q||^2) shortcut catastrophically cancels
    # when h is largely in span(B) (in-sample basis -> tiny residual), spuriously
    # inflating rho and loosening the bound.
    rho = (hf - q @ B.t()).norm()
    if use_kernel:
        U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err_norm)
    else:
        U = shadow_upper_bound_ref(aq, scale, delta, q, float(rho), aq_err_norm)
    U_b = U.view(C, S).amax(1)
    order = U_b.argsort(descending=True)
    Ub_sorted = U_b[order]
    arangeS = torch.arange(S, device=device)
    ell = torch.tensor(float("-inf"), device=device)
    best_id, fetched, nwave = -1, 0, 0
    i = 0
    while i < C:
        if bool(ell > Ub_sorted[i]):
            break
        if fetched >= cap_rows:                  # tail -> exact dense GEMV, lossless
            if WUt is not None:                   # bf16 head (deployed precision)
                logits = h.to(WUt.dtype) @ WUt
            else:
                logits = W_U.float() @ hf
            return int(logits.argmax()), V, nwave
        j2 = min(i + wave, C)
        slabs = order[i:j2]
        idx = (slabs.unsqueeze(1) * S + arangeS.unsqueeze(0)).reshape(-1)
        m, jj = (W_U[idx].float() @ hf).max(0)
        if bool(m > ell):
            ell = m
            best_id = int(idx[int(jj)])
        fetched += int(slabs.numel()) * S
        nwave += 1
        i = j2
    return best_id, fetched, nwave


@triton.jit
def _fused_gather_gemv(
    wu_ptr,  # [V, d]  the dense head (any float dtype)
    idx_ptr,  # int64 [N]  the gathered row ids (union of opened slabs)
    h_ptr,  # fp32 [d]
    out_ptr,  # fp32 [N]  output logits
    N,
    d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather-GEMV: out[i] = W_U[idx[i], :] . h, reading each W_U row DIRECTLY
    through a gathered pointer -- no W_U[idx] materialization, no bf16->fp32 copy.
    One program per BLOCK_M output rows; reduction over d. This is the fused
    stage-2 fetch that replaces the per-wave host loop (device sync + row-copy +
    cast each wave)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < N
    ri = tl.load(idx_ptr + rows, mask=mask_m, other=0)  # row ids
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for d0 in range(0, d, BLOCK_D):
        dj = d0 + tl.arange(0, BLOCK_D)
        mask_d = dj < d
        w = tl.load(
            wu_ptr + ri[:, None] * d + dj[None, :],
            mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        ).to(tl.float32)
        hh = tl.load(h_ptr + dj, mask=mask_d, other=0.0)
        acc += tl.sum(w * hh[None, :], axis=1)
    tl.store(out_ptr + rows, acc, mask=mask_m)


@triton.jit
def _fused_gather_gemm(
    wu_ptr,  # [V, d]  dense head (bf16/fp16)
    idx_ptr,  # int64 [N]  gathered row ids (union of opened slabs)
    h_ptr,  # [B, d]  hidden states (same dtype as wu for the tensor-core dot)
    out_ptr,  # fp32 [B, N]  output logits
    B,
    N,
    d,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Batched gather-GEMM: out[b,n] = H[b,:] . W_U[idx[n],:], reading the gathered
    W_U rows DIRECTLY through pointers (no W_U[idx] materialization/copy). The dot
    runs on tensor cores (bf16 in, fp32 acc). This is the batched stage-2 fetch
    that replaces the union materialization + torch GEMM."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_n = rn < N
    mask_b = rb < B
    ri = tl.load(idx_ptr + rn, mask=mask_n, other=0)  # [BLOCK_N] row ids
    acc = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.float32)
    for d0 in range(0, d, BLOCK_D):
        dj = d0 + tl.arange(0, BLOCK_D)
        mask_d = dj < d
        h = tl.load(h_ptr + rb[:, None] * d + dj[None, :],
                    mask=mask_b[:, None] & mask_d[None, :], other=0.0)  # [BB,BD]
        w = tl.load(wu_ptr + ri[:, None] * d + dj[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)  # [BN,BD]
        acc += tl.dot(h, tl.trans(w), out_dtype=tl.float32)            # [BB,BN]
    tl.store(out_ptr + rb[:, None] * N + rn[None, :], acc,
             mask=mask_b[:, None] & mask_n[None, :])


@triton.jit
def _slab_maxarg(
    wu_ptr,    # [V,d] dense head (bf16/fp16)
    slabs_ptr,  # int32 [L] opened slab ids
    h_ptr,     # [B,d] hidden states (W_U dtype)
    maxv_ptr,  # fp32 [L,B] out: per (slab,token) max logit
    argr_ptr,  # int32 [L,B] out: row-within-slab achieving it
    L, S, d, Bn, C,
    BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Shared-open exact stage: one program per (opened slab, token-block) reads
    the slab's S CONTIGUOUS rows ONCE (direct slab-address arithmetic, no row
    gather / no union materialization), dots with all B tokens on tensor cores,
    and tracks each token's max logit + row index within the slab. A SENTINEL
    slab id >= C is a no-op (writes -inf) so a fixed-L round can be padded and
    stay CUDA-graph-capturable."""
    li = tl.program_id(0)
    bt = tl.program_id(1)
    slab = tl.load(slabs_ptr + li)
    valid = slab < C
    b = bt * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = b < Bn
    best = tl.full([BLOCK_B], -float("inf"), tl.float32)
    barg = tl.zeros([BLOCK_B], tl.int32)
    for r0 in range(0, S, BLOCK_R):
        rloc = r0 + tl.arange(0, BLOCK_R)
        row = slab.to(tl.int64) * S + rloc
        mask_r = (rloc < S) & valid
        acc = tl.zeros([BLOCK_R, BLOCK_B], tl.float32)
        for d0 in range(0, d, BLOCK_D):
            dj = d0 + tl.arange(0, BLOCK_D)
            md = dj < d
            w = tl.load(wu_ptr + row[:, None] * d + dj[None, :],
                        mask=mask_r[:, None] & md[None, :], other=0.0)
            hh = tl.load(h_ptr + b[:, None] * d + dj[None, :],
                         mask=mask_b[:, None] & md[None, :], other=0.0)
            acc += tl.dot(w, tl.trans(hh), out_dtype=tl.float32)   # [BR,BB]
        acc = tl.where(mask_r[:, None], acc, -float("inf"))
        cmax = tl.max(acc, axis=0)                                 # [BB]
        carg = (tl.argmax(acc, axis=0) + r0).to(tl.int32)
        upd = cmax > best
        best = tl.where(upd, cmax, best)
        barg = tl.where(upd, carg, barg)
    tl.store(maxv_ptr + li * Bn + b, best, mask=mask_b)
    tl.store(argr_ptr + li * Bn + b, barg, mask=mask_b)


@triton.jit
def _slab_maxarg_rc(
    wu_ptr, slabs_ptr, h_ptr, maxv_ptr, argr_ptr, L, S, d, Bn, C, NRC,
    BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Row-chunked slab-major max+argmax: grid (L*NRC, B-blocks) so each (slab,
    row-chunk) is its own program -> L*NRC programs (vs L), saturating the GPU.
    Writes a PARTIAL (max,argrow) per (slab,row-chunk); the caller reduces over
    the NRC chunks. argrow is the GLOBAL row-in-slab. Sentinel slab>=C no-ops."""
    pi = tl.program_id(0)
    bt = tl.program_id(1)
    li = pi // NRC
    rc = pi % NRC
    slab = tl.load(slabs_ptr + li)
    valid = slab < C
    b = bt * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = b < Bn
    best = tl.full([BLOCK_B], -float("inf"), tl.float32)
    barg = tl.zeros([BLOCK_B], tl.int32)
    r0 = rc * BLOCK_R
    rloc = r0 + tl.arange(0, BLOCK_R)
    row = slab.to(tl.int64) * S + rloc
    mask_r = (rloc < S) & valid
    acc = tl.zeros([BLOCK_R, BLOCK_B], tl.float32)
    for d0 in range(0, d, BLOCK_D):
        dj = d0 + tl.arange(0, BLOCK_D)
        md = dj < d
        w = tl.load(wu_ptr + row[:, None] * d + dj[None, :],
                    mask=mask_r[:, None] & md[None, :], other=0.0)
        hh = tl.load(h_ptr + b[:, None] * d + dj[None, :],
                     mask=mask_b[:, None] & md[None, :], other=0.0)
        acc += tl.dot(w, tl.trans(hh), out_dtype=tl.float32)
    acc = tl.where(mask_r[:, None], acc, -float("inf"))
    best = tl.max(acc, axis=0)
    barg = (tl.argmax(acc, axis=0) + r0).to(tl.int32)
    tl.store(maxv_ptr + pi * Bn + b, best, mask=mask_b)
    tl.store(argr_ptr + pi * Bn + b, barg, mask=mask_b)


def slab_maxarg_rc(W_U, slabs, H, S, BLOCK_R=128, BLOCK_D=128, BLOCK_B=32):
    """Parallel slab-major max+argmax over L slabs for all B tokens. Returns
    (best_logit[B], best_slab[B], best_row[B]) -- the global incumbent over the
    opened slabs, with the cross-(slab,row-chunk) reduction done here in torch
    (cheap: [L*NRC, B] -> [B]). NRC = ceil(S/BLOCK_R)."""
    L = slabs.numel()
    V, d = W_U.shape
    C = V // S
    Bn = H.shape[0]
    NRC = triton.cdiv(S, BLOCK_R)
    P = L * NRC
    maxv = torch.empty(P, Bn, device=W_U.device, dtype=torch.float32)
    argr = torch.empty(P, Bn, device=W_U.device, dtype=torch.int32)
    grid = (P, triton.cdiv(Bn, BLOCK_B))
    _slab_maxarg_rc[grid](W_U, slabs.to(torch.int32), H.to(W_U.dtype).contiguous(),
                          maxv, argr, L, S, d, Bn, C, NRC,
                          BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D, BLOCK_B=max(16, BLOCK_B))
    m, flat = maxv.max(0)                      # [B] over all (slab,row-chunk)
    li = (flat // NRC)
    best_slab = slabs.to(torch.long)[li]
    best_row = argr.gather(0, flat[None, :]).squeeze(0).long()
    return m, best_slab, best_row


def slab_maxarg(W_U, slabs, H, S, BLOCK_R=64, BLOCK_D=64, BLOCK_B=64):
    """Per (opened slab, token) max logit + row, via the slab-major kernel. Slab
    ids >= C are sentinels (no-op -> -inf), so a fixed-L round can be padded.
    Returns maxv[L,B] (fp32), argr[L,B] (int32). H cast to W_U dtype."""
    L = slabs.numel()
    V, d = W_U.shape
    C = V // S
    Bn = H.shape[0]
    maxv = torch.empty(L, Bn, device=W_U.device, dtype=torch.float32)
    argr = torch.empty(L, Bn, device=W_U.device, dtype=torch.int32)
    grid = (L, triton.cdiv(Bn, BLOCK_B))
    _slab_maxarg[grid](W_U, slabs.to(torch.int32), H.to(W_U.dtype).contiguous(),
                       maxv, argr, L, S, d, Bn, C,
                       BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D, BLOCK_B=max(16, BLOCK_B))
    return maxv, argr


def fused_gather_gemm(W_U, idx, H, BLOCK_B=16, BLOCK_N=64, BLOCK_D=64):
    """logits[B,N] = H @ W_U[idx].T via the fused Triton gather-GEMM (no row-copy).
    H is cast to W_U's dtype for the tensor-core dot; logits are bf16-precise =
    the DEPLOYED head's truth (match_deployed is exact; match_fp32 may differ on
    near-ties, which is the deployed reality)."""
    assert W_U.is_contiguous() and idx.dtype == torch.int64 and W_U.is_cuda
    Bn = H.shape[0]
    N = idx.numel()
    V, d = W_U.shape
    out = torch.empty(Bn, N, device=W_U.device, dtype=torch.float32)
    Hc = H.to(W_U.dtype).contiguous()
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(Bn, BLOCK_B))
    _fused_gather_gemm[grid](
        W_U, idx, Hc, out, Bn, N, d,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out


def fused_gather_gemv(W_U, idx, h, BLOCK_M=64, BLOCK_D=256):
    """logits[N] = W_U[idx] . h via the fused Triton gather-GEMV (no row-copy)."""
    assert W_U.is_contiguous(), "W_U must be row-major contiguous for gathered ptrs"
    assert idx.dtype == torch.int64, "idx must be int64 for pointer arithmetic"
    assert h.is_cuda and W_U.is_cuda, "fused gather-GEMV is GPU-only"
    N = idx.numel()
    V, d = W_U.shape
    out = torch.empty(N, device=W_U.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M),)
    _fused_gather_gemv[grid](
        W_U, idx, h.float().contiguous(), out, N, d,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )
    return out


@torch.no_grad()
def certified_decode_oneshot(h, B, aq, scale, delta, W_U, S, aq_err_norm, WUt,
                             k_frac=0.125, use_kernel=True, fused=True):
    """Sync-free stage-2: open a FIXED top-K-by-bound budget in ONE fused
    gather-GEMV, then ONE certification check. No Python while-loop, so no
    per-wave device sync and no per-wave row-copy. If the incumbent over the
    opened budget already beats the largest UNopened slab bound, the dense argmax
    is certified losslessly; otherwise fall back to the exact dense GEMV (also
    lossless). k_frac is the fraction of SLABS opened up front (calibrate so the
    fallback rate is small). Returns (argmax_id, fetched_rows, certified_bool)."""
    device = h.device
    V, d = W_U.shape
    C = V // S
    hf = h.float()
    q = hf @ B
    rho = (hf - q @ B.t()).norm()  # exact; sqrt-shortcut cancels (see _waves)
    if use_kernel:
        U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err_norm)
    else:
        U = shadow_upper_bound_ref(aq, scale, delta, q, float(rho), aq_err_norm)
    U_b = U.view(C, S).amax(1)
    Ub_sorted, order = U_b.sort(descending=True)
    K = max(1, min(C, int(math.ceil(k_frac * C))))
    top = order[:K]
    idx = (top.unsqueeze(1) * S + torch.arange(S, device=device).unsqueeze(0)).reshape(-1)
    if fused:
        logits = fused_gather_gemv(W_U, idx, hf)
    else:
        logits = W_U[idx].float() @ hf
    m, jj = logits.max(0)
    # certify: incumbent must beat the largest unopened bound (Ub_sorted[K])
    bound = Ub_sorted[K] if K < C else torch.tensor(float("-inf"), device=device)
    if bool(m > bound):
        return int(idx[int(jj)]), K * S, True
    return int((h.to(WUt.dtype) @ WUt).argmax()), V, False


@torch.no_grad()
def certified_decode_routed(h, B, aq, scale, delta, W_U, S, aq_err_norm, WUt,
                            tau_gap=None, tau_rho=None, K=52, wave=4,
                            fallback_frac=0.20):
    """Predictive-routing certified decode (Codex thread 019eb249). Avoids the
    wasted partial fetch on hard tokens by routing them to the dense GEMV up front.
    Returns (argmax_id, route) where route in {'rho_veto','gap_route','certified'}.

    rho_veto (pre-stage-1): if rho/||h|| >= tau_rho the bound is too loose -> dense
    WITHOUT paying stage-1 (the only path to ~dense p99). gap_route (post-stage-1):
    if the U_b spread over the top-K slabs gK = Ub[0]-Ub[K-1] <= tau_gap, the bound
    can't separate the winner from ~K slabs -> dense. Else cheap certified decode
    with the dense fallback kept only as a miss-safety net. All paths lossless."""
    device = h.device
    V, d = W_U.shape
    C = V // S
    hf = h.float()
    q = hf @ B
    hn = torch.sqrt((hf * hf).sum()).clamp_min(1e-9)
    rho = (hf - q @ B.t()).norm()  # exact; sqrt-shortcut cancels (see _waves)
    if tau_rho is not None and float(rho / hn) >= tau_rho:
        return int((h.to(WUt.dtype) @ WUt).argmax()), "rho_veto"
    U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err_norm)
    U_b = U.view(C, S).amax(1)
    Ub_sorted, order = U_b.sort(descending=True)
    if tau_gap is not None:
        gK = float(Ub_sorted[0] - Ub_sorted[min(K, C) - 1])
        if gK <= tau_gap:
            return int((h.to(WUt.dtype) @ WUt).argmax()), "gap_route"
    # cheap certified path (reuse the wave fetch over the precomputed order)
    arangeS = torch.arange(S, device=device)
    cap_rows = int(fallback_frac * V)
    ell = torch.tensor(float("-inf"), device=device)
    best_id, fetched, i = -1, 0, 0
    while i < C:
        if bool(ell > Ub_sorted[i]):
            break
        if fetched >= cap_rows:
            return int((h.to(WUt.dtype) @ WUt).argmax()), "certified"
        j2 = min(i + wave, C)
        slabs = order[i:j2]
        idx = (slabs.unsqueeze(1) * S + arangeS.unsqueeze(0)).reshape(-1)
        m, jj = (W_U[idx].float() @ hf).max(0)
        if bool(m > ell):
            ell = m
            best_id = int(idx[int(jj)])
        fetched += int(slabs.numel()) * S
        i = j2
    return best_id, "certified"
