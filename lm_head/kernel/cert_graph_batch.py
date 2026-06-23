#!/usr/bin/env python3
"""Batched shared-shortlist certified decode, CUDA-graph-safe (v2 -> graph).

The multi-round adaptive shared-open (shared_open_decode.py) is lossless and keeps
the union small (~11% at B=32) but its data-dependent rounds are not graph-safe. This
is the graph-safe single-round version: open a FIXED top-L shared shortlist by
batch-urgency G[s]=max_b Ub[b,s], compute it for all B tokens with the slab-major
max+argmax kernel, certify per token (m_b > max non-shortlist Ub[b,*]); the rare
uncertified token falls back to dense OUTSIDE the graph. Static shapes (Ub GEMM,
topk(L), slab_maxarg over L slabs) -> one CUDA-graph replay, no host sync. Per-row
aq_err, r configurable. Trades the multi-round 11% union for a fixed ~L/C union, but
stays graph-capturable and the Pareto still wins at ~20%.
"""

from __future__ import annotations

import torch

from certdecode_kernel import slab_maxarg, shadow_slab_bounds_perrow

# static round ladder (NEW slabs opened per round); unrolled in the graph, rounds
# no-op on device once all tokens certify (ChatGPT-Pro v3 design)
LADDER = [4, 4, 8, 8, 16, 16, 32, 32, 64, 72]


class CertDecodeGraphBatch:
    def __init__(self, Bb, aq, scale, delta, aq_err_row, aq_l2, W_U, S, L, Bn, device):
        self.Bb = Bb
        self.aq = aq            # int8 [V,r] codes
        self.scale = scale      # fp32 [r]
        self.delta = delta
        self.aq_err_row = aq_err_row
        self.aq_l2 = aq_l2
        self.W_U = W_U.contiguous()
        self.WUt = self.W_U.t().contiguous()
        self.S = S
        self.C = W_U.shape[0] // S
        self.L = L
        self.Bn = Bn
        self.d = Bb.shape[0]
        self.dev = device
        self.H_buf = torch.zeros(Bn, self.d, device=device, dtype=torch.float32)
        self.best_id = torch.zeros(Bn, dtype=torch.long, device=device)
        self.cert = torch.zeros(Bn, dtype=torch.bool, device=device)
        self.graph = None

    def _step(self):
        Hf = self.H_buf
        q = Hf @ self.Bb
        rho = (Hf - q @ self.Bb.t()).norm(dim=1)
        # int8 shadow (reads 97MB int8 not 389MB fp32), per-row err, direct slab-max
        Ub = shadow_slab_bounds_perrow(self.aq, self.scale, self.delta, q, rho,
                                       self.aq_err_row, self.S, self.aq_l2)  # [B,C]
        G = Ub.max(0).values                                    # [C] batch urgency
        short = G.topk(self.L).indices.to(torch.int32)          # [L]
        maxv, argr = slab_maxarg(self.W_U, short, self.H_buf, self.S)  # [L,B]
        rbest, rslab = maxv.max(0)                              # [B]
        rrow = argr.gather(0, rslab[None, :]).squeeze(0).long()
        self.best_id.copy_(short[rslab].long() * self.S + rrow)
        Ub_masked = Ub.scatter(1, short.long().unsqueeze(0).expand(self.Bn, self.L).contiguous(),
                               -1e30)
        rem = Ub_masked.max(1).values                          # [B]
        self.cert.copy_(rbest > rem)

    def capture(self, warmup=5):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        return self

    @torch.no_grad()
    def run_eager(self, H):
        """Eager correctness path: returns (best_id[B], cert[B])."""
        self.H_buf.copy_(H.float())
        self._step()
        ids = self.best_id.clone()
        cert = self.cert.clone()
        if (~cert).any():
            ids[~cert] = (H[~cert].to(self.WUt.dtype) @ self.WUt).argmax(1)
        return ids, cert

    @torch.no_grad()
    def __call__(self, H):
        self.H_buf.copy_(H.float())
        self.graph.replay()
        ids = self.best_id.clone()
        cert = self.cert.clone()
        if (~cert).any():
            ids[~cert] = (H[~cert].to(self.WUt.dtype) @ self.WUt).argmax(1)
        return ids, cert


class CertDecodeGraphLadder:
    """Adaptive all-certify via a STATIC round ladder (ChatGPT-Pro v3 design).
    Unrolled fixed rounds (graph-capturable); each round opens the top-L_round
    UNOPENED slabs by urgency over UNCERTIFIED tokens, computes them for all B
    (slab-major, sentinel-padded no-op), updates incumbents, re-certifies. No
    host sync inside the graph. Dense fallback only for tokens still uncertified
    after the ladder (rare; the last round + full open == exact)."""

    def __init__(self, Bb, aq, scale, delta, aq_err_row, aq_l2, W_U, S, Bn, device,
                 ladder=LADDER):
        self.Bb = Bb
        self.aq, self.scale, self.delta = aq, scale, delta
        self.aq_err_row, self.aq_l2 = aq_err_row, aq_l2
        self.W_U = W_U.contiguous()
        self.WUt = self.W_U.t().contiguous()
        self.S = S
        self.C = W_U.shape[0] // S
        self.Bn = Bn
        self.d = Bb.shape[0]
        self.dev = device
        self.ladder = ladder
        self.H_buf = torch.zeros(Bn, self.d, device=device, dtype=torch.float32)
        self.best_id = torch.zeros(Bn, dtype=torch.long, device=device)
        self.cert = torch.zeros(Bn, dtype=torch.bool, device=device)
        self.graph = None

    def _step(self):
        dev = self.dev
        C, Bn, S = self.C, self.Bn, self.S
        NEG = -1e30
        Hf = self.H_buf
        q = Hf @ self.Bb
        rho = (Hf - q @ self.Bb.t()).norm(dim=1)
        Ub = shadow_slab_bounds_perrow(self.aq, self.scale, self.delta, q, rho,
                                       self.aq_err_row, S, self.aq_l2)   # [B,C]
        opened = torch.zeros(C, device=dev, dtype=torch.float32)         # 1.0 = opened
        m_b = torch.full((Bn,), NEG, device=dev)
        best_id = torch.zeros(Bn, dtype=torch.long, device=dev)
        cert = torch.zeros(Bn, dtype=torch.bool, device=dev)
        for Lr in self.ladder:
            active = ~cert
            cand = torch.where((Ub >= m_b[:, None]) & active[:, None], Ub,
                               torch.full_like(Ub, NEG))
            G = cand.max(0).values                                       # [C]
            G = torch.where(opened > 0.5, torch.full_like(G, NEG), G)
            vals, idx = G.topk(Lr)                                       # [Lr]
            real = (vals > -1e29)                                        # real vs sentinel
            sel = torch.where(real, idx.to(torch.int32),
                              torch.full_like(idx, C, dtype=torch.int32))  # sentinel=C
            # mark real selected slabs opened (graph-safe scatter_reduce)
            opened = opened.scatter_reduce(
                0, idx.long(), real.to(torch.float32), reduce="amax", include_self=True)
            maxv, argr = slab_maxarg(self.W_U, sel, self.H_buf, S)       # [Lr,B]
            rbest, rslab = maxv.max(0)                                   # [B]
            rrow = argr.gather(0, rslab[None, :]).squeeze(0).long()
            sel_slab = sel[rslab].long()                                 # may be C (sentinel)
            rid = sel_slab.clamp(max=C - 1) * S + rrow
            upd = rbest > m_b
            m_b = torch.where(upd, rbest, m_b)
            best_id = torch.where(upd, rid, best_id)
            rem = torch.where(opened[None, :] > 0.5, torch.full_like(Ub, NEG),
                              Ub).max(1).values
            cert = m_b > rem
        self.best_id.copy_(best_id)
        self.cert.copy_(cert)

    def capture(self, warmup=4):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        return self

    @torch.no_grad()
    def run_eager(self, H):
        self.H_buf.copy_(H.float()); self._step()
        ids = self.best_id.clone(); cert = self.cert.clone()
        if (~cert).any():
            ids[~cert] = (H[~cert].to(self.WUt.dtype) @ self.WUt).argmax(1)
        return ids, cert

    @torch.no_grad()
    def __call__(self, H):
        self.H_buf.copy_(H.float()); self.graph.replay()
        ids = self.best_id.clone(); cert = self.cert.clone()
        if (~cert).any():
            ids[~cert] = (H[~cert].to(self.WUt.dtype) @ self.WUt).argmax(1)
        return ids, cert
