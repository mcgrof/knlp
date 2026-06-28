"""Fused Triton forward for the Trellis true-stale state-evolution.

The state-evolution is sequential over chunks (Mstate carries) with parallel
bmm's inside each chunk; the PyTorch path pays a Python launch per chunk
(nC = T/chunk_size launches, each several bmm's). This collapses the whole
chunk loop into one persistent kernel: grid = (B*H,), each program owns one
(b,h) state Mstate[M,D] resident in registers and loops the chunks internally,
doing the two matmuls with tl.dot and the LayerNorm-SiLU VJP rowwise in-kernel.

Crucially D here is the *head* dim (d_head=64), not the model dim, so Mstate is
a single [64,64] tile (~16KB fp32) — no D-tiling needed. Occupancy is one CTA
per (b,h); the win materialises when B*H is large enough to fill the SMs (the
big ladder rungs: B*H = 128, 192).

Output matches run_trellis_memory_chunked_state_evolution: per-chunk start
states M0s [B,H,nC,M,D] and codes us [B,H,nC,C,M], which then feed the existing
batched readout. fp32 only (the mixer forces fp32; the recurrence is sensitive).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _trellis_evo_fwd_kernel(
        W_ptr,
        A_ptr,
        rC_ptr,
        P_ptr,
        g_ptr,
        M0_ptr,
        U_ptr,
        nC,
        M: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        ACT: tl.constexpr,
        EPS: tl.constexpr,
    ):
        n = tl.program_id(0)
        rm = tl.arange(0, M)
        rd = tl.arange(0, D)
        rc = tl.arange(0, C)
        # per-(b,h) base offsets into the [N, nC, ...] tensors
        gamma = tl.load(g_ptr + n)
        state = tl.zeros((M, D), dtype=tl.float32)  # Mstate, resident
        for c in range(nC):
            wbase = (n * nC + c) * C * D
            W = tl.load(W_ptr + wbase + rc[:, None] * D + rd[None, :])  # [C,D]
            # z = W @ state^T  -> [C, M]
            z = tl.dot(W, tl.trans(state), allow_tf32=False)  # [C,M]
            abase = (n * nC + c) * C * M
            alpha = tl.load(A_ptr + abase + rc[:, None] * M + rm[None, :])  # [C,M]
            if ACT == 0:
                # LayerNorm(SiLU(z)) VJP w.r.t. z, carrying alpha.
                sig = tl.sigmoid(z)
                s = z * sig
                silu_grad = sig + s * (1.0 - sig)
                mean = tl.sum(s, axis=1) / M  # [C]
                sc = s - mean[:, None]
                var = tl.sum(sc * sc, axis=1) / M  # [C]
                std = tl.sqrt(var + EPS)
                y = sc / std[:, None]
                err = y - alpha
                emean = tl.sum(err, axis=1) / M
                eymean = tl.sum(err * y, axis=1) / M
                ds = (err - emean[:, None] - y * eymean[:, None]) / std[:, None]
                u = silu_grad * ds  # [C,M]
            elif ACT == 1:
                # SiLU VJP: u = silu'(z) * (silu(z) - alpha).
                sig = tl.sigmoid(z)
                s = z * sig
                silu_grad = sig + s * (1.0 - sig)
                u = silu_grad * (s - alpha)
            else:
                # Identity VJP: u = z - alpha.
                u = z - alpha
            # store start-state M0 (= current state) and code u
            m0base = (n * nC + c) * M * D
            tl.store(M0_ptr + m0base + rm[:, None] * D + rd[None, :], state)
            tl.store(U_ptr + abase + rc[:, None] * M + rm[None, :], u)
            # upd = (u * rC)^T @ W  -> [M, D]
            rC = tl.load(rC_ptr + (n * nC + c) * C + rc)  # [C]
            ur = u * rC[:, None]  # [C,M]
            upd = tl.dot(tl.trans(ur), W, allow_tf32=False)  # [M,D]
            Plast = tl.load(P_ptr + n * nC + c)
            state = Plast * state - gamma * upd

    @triton.jit
    def _trellis_evo_bwd_kernel(
        W_ptr,
        rC_ptr,
        P_ptr,
        g_ptr,
        M0_ptr,
        U_ptr,
        gM0_ptr,
        gU_ptr,
        gW_ptr,
        gA_ptr,
        gP_ptr,
        gRC_ptr,
        gG_ptr,
        nC,
        M: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        ACT: tl.constexpr,
        EPS: tl.constexpr,
    ):
        n = tl.program_id(0)
        rm = tl.arange(0, M)
        rd = tl.arange(0, D)
        rc = tl.arange(0, C)
        gamma = tl.load(g_ptr + n)
        bar_M = tl.zeros((M, D), dtype=tl.float32)  # carried cotangent of state
        gG = tl.zeros((1,), dtype=tl.float32)
        for ci in range(nC):
            c = nC - 1 - ci  # reverse chunk order
            wbase = (n * nC + c) * C * D
            ubase = (n * nC + c) * C * M
            m0base = (n * nC + c) * M * D
            W = tl.load(W_ptr + wbase + rc[:, None] * D + rd[None, :])  # [C,D]
            u = tl.load(U_ptr + ubase + rc[:, None] * M + rm[None, :])  # [C,M]
            M0 = tl.load(M0_ptr + m0base + rm[:, None] * D + rd[None, :])  # [M,D]
            rC = tl.load(rC_ptr + (n * nC + c) * C + rc)  # [C]
            Plast = tl.load(P_ptr + n * nC + c)
            bar_M1 = bar_M
            # BW[c,m] = sum_d bar_M1[m,d] W[c,d]  = W @ bar_M1^T -> [C,M]
            BW = tl.dot(W, tl.trans(bar_M1), allow_tf32=False)  # [C,M]
            # UB[c,d] = sum_m u[c,m] bar_M1[m,d] = u @ bar_M1 -> [C,D]
            UB = tl.dot(u, bar_M1, allow_tf32=False)  # [C,D]
            grad_W = -gamma * rC[:, None] * UB  # [C,D]
            tl.store(gW_ptr + wbase + rc[:, None] * D + rd[None, :], grad_W)
            bar_u_state = -gamma * rC[:, None] * BW  # [C,M]
            ub = tl.sum(u * BW, axis=1)  # [C]
            grad_rC = -gamma * ub  # [C]
            tl.store(gRC_ptr + (n * nC + c) * C + rc, grad_rC)
            gG += -tl.sum(rC * ub, axis=0)
            # grad_P at last step = <bar_M1, M0>
            gP = tl.sum(bar_M1 * M0)
            tl.store(gP_ptr + n * nC + c, gP)
            # bar_u_total = grad_us + bar_u_state ; grad_A = alpha adjoint
            gU = tl.load(gU_ptr + ubase + rc[:, None] * M + rm[None, :])  # [C,M]
            bar_u = gU + bar_u_state  # [C,M]
            if ACT == 0:
                # recompute z = W @ M0^T -> [C,M] for the LN-SiLU adjoint
                z = tl.dot(W, tl.trans(M0), allow_tf32=False)  # [C,M]
                sig = tl.sigmoid(z)
                s = z * sig
                silu_grad = sig + s * (1.0 - sig)
                mean = tl.sum(s, axis=1) / M
                sc = s - mean[:, None]
                var = tl.sum(sc * sc, axis=1) / M
                std = tl.sqrt(var + EPS)
                y = sc / std[:, None]
                h = silu_grad * bar_u
                hmean = tl.sum(h, axis=1) / M
                hymean = tl.sum(h * y, axis=1) / M
                lnop = (h - hmean[:, None] - y * hymean[:, None]) / std[:, None]
                grad_A = -lnop  # [C,M]
            elif ACT == 1:
                z = tl.dot(W, tl.trans(M0), allow_tf32=False)  # [C,M]
                sig = tl.sigmoid(z)
                s = z * sig
                silu_grad = sig + s * (1.0 - sig)
                grad_A = -(silu_grad * bar_u)
            else:
                grad_A = -bar_u
            tl.store(gA_ptr + ubase + rc[:, None] * M + rm[None, :], grad_A)
            # carry: bar_M = grad_M0s + Plast * bar_M1
            gM0 = tl.load(gM0_ptr + m0base + rm[:, None] * D + rd[None, :])
            bar_M = gM0 + Plast * bar_M1
        tl.store(gG_ptr + n, tl.sum(gG))

    def _activation_code(activation):
        if activation == "ln_silu":
            return 0
        if activation == "silu":
            return 1
        if activation in ("identity", "linear"):
            return 2
        raise ValueError(f"unsupported Triton Trellis activation: {activation}")

    def _slot_kernel_dim(actual_m: int, activation_code: int) -> int:
        if activation_code == 0:
            return actual_m
        return max(16, 1 << (actual_m - 1).bit_length())

    def _pad_alpha_slots(alpha, m_kernel: int):
        pad = m_kernel - alpha.shape[-1]
        if pad <= 0:
            return alpha
        return F.pad(alpha, (0, pad))

    def trellis_state_evolution_triton(
        write, alpha, P, rmat, gamma, chunk_size, activation="ln_silu"
    ):
        """Triton fused forward. Returns M0s [B,H,nC,M,D], us [B,H,nC,C,M]."""
        M0s, us = _evo_fwd(write, alpha, P, rmat, gamma, chunk_size, activation)
        return M0s, us

    def _gamma_to_N(gamma, B, H):
        # N = B*H laid out as index b*H+h, so gamma[h] tiles per batch
        return gamma.view(1, H).expand(B, H).reshape(B * H).contiguous().float()

    def _evo_fwd(write, alpha, P, rmat, gamma, C, activation):
        B, H, T, D = write.shape
        M_actual = alpha.shape[-1]
        act = _activation_code(activation)
        M = _slot_kernel_dim(M_actual, act)
        nC = P.shape[2]
        N = B * H
        dev = write.device
        pad = nC * C - T
        if pad > 0:
            write = F.pad(write, (0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, pad))
        alpha = _pad_alpha_slots(alpha, M)
        Wp = write.reshape(N, nC, C, D).contiguous().float()
        Ap = alpha.reshape(N, nC, C, M).contiguous().float()
        rC = rmat[:, :, :, -1, :].reshape(N, nC, C).contiguous().float()
        Pl = P[:, :, :, -1, 0].reshape(N, nC).contiguous().float()
        g = _gamma_to_N(gamma, B, H)
        M0s = torch.empty(N, nC, M, D, device=dev, dtype=torch.float32)
        us = torch.empty(N, nC, C, M, device=dev, dtype=torch.float32)
        _trellis_evo_fwd_kernel[(N,)](
            Wp,
            Ap,
            rC,
            Pl,
            g,
            M0s,
            us,
            nC,
            M=M,
            D=D,
            C=C,
            ACT=act,
            EPS=1e-6,
        )
        M0s = M0s.view(B, H, nC, M, D)[:, :, :, :M_actual, :]
        us = us.view(B, H, nC, C, M)[:, :, :, :, :M_actual]
        return M0s.contiguous(), us.contiguous()

    class TrellisStateEvolutionTriton(torch.autograd.Function):
        """fwd+bwd Triton fused state-evolution. Same math as the validated
        TrellisStateEvolutionFn (z detached for u -> true-stale gradient)."""

        @staticmethod
        def forward(ctx, write, alpha, P, rmat, gamma, chunk_size, activation):
            B, H, T, D = write.shape
            M_actual = alpha.shape[-1]
            act = _activation_code(activation)
            M = _slot_kernel_dim(M_actual, act)
            C = chunk_size
            nC = P.shape[2]
            N = B * H
            dev = write.device
            # pad time to nC*C with zeros (matches the PyTorch state-evolution;
            # the decay P/rmat already carry the padded chunk via beta-pad=1).
            pad = nC * C - T
            if pad > 0:
                write = F.pad(write, (0, 0, 0, pad))
                alpha = F.pad(alpha, (0, 0, 0, pad))
            alpha = _pad_alpha_slots(alpha, M)
            Wp = write.reshape(N, nC, C, D).contiguous().float()
            Ap = alpha.reshape(N, nC, C, M).contiguous().float()
            rCl = rmat[:, :, :, -1, :].reshape(N, nC, C).contiguous().float()
            Pl = P[:, :, :, -1, 0].reshape(N, nC).contiguous().float()
            g = _gamma_to_N(gamma, B, H)
            M0s = torch.empty(N, nC, M, D, device=dev, dtype=torch.float32)
            us = torch.empty(N, nC, C, M, device=dev, dtype=torch.float32)
            _trellis_evo_fwd_kernel[(N,)](
                Wp,
                Ap,
                rCl,
                Pl,
                g,
                M0s,
                us,
                nC,
                M=M,
                D=D,
                C=C,
                ACT=act,
                EPS=1e-6,
            )
            ctx.save_for_backward(Wp, rCl, Pl, g, M0s, us)
            ctx.dims = (B, H, T, D, M_actual, M, nC, C, act)
            M0v = M0s.view(B, H, nC, M, D)[:, :, :, :M_actual, :]
            usv = us.view(B, H, nC, C, M)[:, :, :, :, :M_actual]
            return M0v.contiguous(), usv.contiguous()

        @staticmethod
        def backward(ctx, grad_M0s, grad_us):
            Wp, rCl, Pl, g, M0s, us = ctx.saved_tensors
            B, H, T, D, M_actual, M, nC, C, act = ctx.dims
            N = B * H
            dev = Wp.device
            gW = torch.empty(N, nC, C, D, device=dev, dtype=torch.float32)
            gA = torch.empty(N, nC, C, M, device=dev, dtype=torch.float32)
            gP_last = torch.empty(N, nC, device=dev, dtype=torch.float32)
            gRC = torch.empty(N, nC, C, device=dev, dtype=torch.float32)
            gG_n = torch.empty(N, device=dev, dtype=torch.float32)
            if M != M_actual:
                grad_M0s = F.pad(grad_M0s, (0, 0, 0, M - M_actual))
                grad_us = F.pad(grad_us, (0, M - M_actual))
            gM0s = grad_M0s.reshape(N, nC, M, D).contiguous().float()
            gUs = grad_us.reshape(N, nC, C, M).contiguous().float()
            _trellis_evo_bwd_kernel[(N,)](
                Wp,
                rCl,
                Pl,
                g,
                M0s,
                us,
                gM0s,
                gUs,
                gW,
                gA,
                gP_last,
                gRC,
                gG_n,
                nC,
                M=M,
                D=D,
                C=C,
                ACT=act,
                EPS=1e-6,
            )
            grad_write = gW.view(B, H, nC * C, D)[:, :, :T].contiguous()
            grad_alpha = gA.view(B, H, nC * C, M)[:, :, :T, :M_actual].contiguous()
            grad_P = torch.zeros(B, H, nC, C, 1, device=dev, dtype=torch.float32)
            grad_P[:, :, :, -1, 0] = gP_last.view(B, H, nC)
            grad_rmat = torch.zeros(B, H, nC, C, C, device=dev, dtype=torch.float32)
            grad_rmat[:, :, :, -1, :] = gRC.view(B, H, nC, C)
            grad_gamma = gG_n.view(B, H).sum(0)
            return grad_write, grad_alpha, grad_P, grad_rmat, grad_gamma, None, None
