#!/usr/bin/env python3
"""CUDA-graph-captured batch-1 certified decode (Codex design, thread 019eb34a).

The eager one-shot is ~10 small launches glued by Python (q=h@B, rho, stage1,
topk, idx, gather-GEMV, max, cert). On the H100 the dense head is only ~0.38 ms,
so that ~0.3 ms orchestration tax is what keeps batch-1 from winning. This
captures the whole certified path (fixed K*S=K*594 gather shape) into ONE CUDA
graph and replays it as a single launch. fp32 gather throughout (the certificate
must be fp32-dense-lossless, not bf16-TC). The cert bool + argmax are read after
replay; the ~7% uncertified tokens fall back to the dense head OUTSIDE the graph.

Graph-safe stage-1 (`shadow_slabmax_b1`) takes rho/cst as DEVICE scalars so there
is no host sync inside the capture.
"""

from __future__ import annotations

import torch

from certdecode_kernel import shadow_slabmax_b1, fused_gather_gemv


class CertDecodeGraphB1:
    def __init__(self, B, aq, scale, delta, W_U, S, aq_err, K, device):
        self.B = B
        self.Bt = B.t().contiguous()
        self.aq = aq
        self.scale = scale
        self.delta = delta
        self.W_U = W_U.contiguous()
        self.WUt = self.W_U.t().contiguous()
        self.S = S
        self.C = W_U.shape[0] // S
        self.aq_err = float(aq_err)
        self.K = K
        self.d = B.shape[0]
        self.dev = device
        self.arangeS = torch.arange(S, device=device)
        # static buffers
        self.h_buf = torch.zeros(self.d, device=device, dtype=torch.float32)
        self.ub_buf = torch.full((self.C,), -float("inf"), device=device)
        self.best_id_buf = torch.zeros(1, device=device, dtype=torch.long)
        self.cert_buf = torch.zeros(1, device=device, dtype=torch.bool)
        self.graph = None

    def _step(self):
        hf = self.h_buf
        q = hf @ self.B                       # [r]
        bq = q @ self.Bt                      # [d]
        rho = (hf - bq).norm().reshape(1)     # [1] device
        cst = (self.aq_err * q.norm()).reshape(1)  # [1] device
        qs = (self.scale * q).contiguous()    # [r]
        self.ub_buf.fill_(-float("inf"))
        shadow_slabmax_b1(self.aq, qs, self.delta, rho, cst, self.S, self.ub_buf)
        top_vals, top_slabs = torch.topk(self.ub_buf, self.K + 1, sorted=True)
        idx = (top_slabs[: self.K].unsqueeze(1) * self.S
               + self.arangeS.unsqueeze(0)).reshape(-1).contiguous()
        logits = fused_gather_gemv(self.W_U, idx, hf)   # [K*S] fp32
        m, jj = logits.max(0)
        # index_select (explicit-tensor gather) is graph-capturable; idx[jj]
        # advanced-indexing is NOT (it errors mid-capture).
        self.best_id_buf.copy_(idx.index_select(0, jj.view(1)))
        self.cert_buf.copy_((m > top_vals[self.K].view(1)))

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
    def __call__(self, h):
        """Return (argmax_id, certified_bool). h is [d] (any dtype)."""
        self.h_buf.copy_(h.float())
        self.graph.replay()
        if not bool(self.cert_buf):           # one sync; fallback outside graph
            return int((h.to(self.WUt.dtype) @ self.WUt).argmax()), False
        return int(self.best_id_buf), True
