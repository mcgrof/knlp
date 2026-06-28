"""TrellisMixer — two-pass bounded-memory sequence mixing sublayer.

Replaces self-attention. Produces its output directly from the compressed
memory state (no [B,H,T,T] mask). Returns the sublayer delta; the block adds
the residual.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .activations import get_activation
from .trellis_memory import (
    run_trellis_memory,
    run_trellis_memory_chunked,
    trellis_chunk_decay,
    run_trellis_memory_chunked_state_evolution,
    run_trellis_memory_chunked_batched_readout,
)

try:
    from .trellis_triton import TrellisStateEvolutionTriton, HAS_TRITON
except Exception:  # pragma: no cover
    HAS_TRITON = False

# Head-on bf16 test (Codex-advised recipe): round the write/read/alpha inputs to
# bf16 while the decay (beta->P/rmat), gamma, the resident Mstate and all the
# LN-SiLU reductions stay fp32 (the kernel accumulates in fp32 internally). This
# is the autocast-bf16 regime for Trellis without a fully-bf16 operator. Off by
# default; the ladder/test flips it to compare PPL vs fp32.
BF16_INPUTS = False


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.w


class CausalDWConv1d(nn.Module):
    """Depthwise causal conv over time on a [B,H,T,D] tensor (per-channel)."""

    def __init__(self, channels, kernel):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=True)

    def forward(self, x):  # x: [B,H,T,D]
        B, H, T, D = x.shape
        xt = x.permute(0, 1, 3, 2).reshape(B, H * D, T)  # [B, C=H*D, T]
        xt = F.pad(xt, (self.kernel - 1, 0))  # left pad = causal
        out = self.conv(xt)  # [B, C, T]
        out = out.reshape(B, H, D, T).permute(0, 1, 3, 2)  # [B,H,T,D]
        return out


class TrellisMixer(nn.Module):
    def __init__(self, cfg: TrellisConfig):
        super().__init__()
        self.cfg = cfg
        H, D, M, d = cfg.n_heads, cfg.d_head, cfg.n_slots, cfg.d_model
        self.H, self.D, self.M = H, D, M
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, H * D, bias=False)
        self.v_proj = nn.Linear(d, H * D, bias=False)
        self.alpha_proj = nn.Linear(d, H * M, bias=False)
        beta_out = H if cfg.beta_mode == "scalar_per_head" else H * M
        self.beta_proj = nn.Linear(d, beta_out, bias=True)
        self.reset_beta_bias()
        # gamma positive per head via softplus(raw); init so softplus(raw)=gamma_init
        raw0 = math.log(math.expm1(cfg.gamma_init))
        self.gamma_raw = nn.Parameter(torch.full((H,), raw0))
        self.out_proj = nn.Linear(H * D, d, bias=False)
        self.output_path = cfg.output_path
        self.post_gate = cfg.post_gate
        if cfg.output_path == "paper":
            # Fig 1 order: Trellis -> PostNorm -> GeLU gate -> out_proj. The gate
            # lives in the inner_dim (H*D) space, applied BEFORE out_proj.
            self.post_norm = RMSNorm(H * D)
            self.gate_in = nn.Linear(d, H * D, bias=False)
        elif cfg.post_gate:
            self.gate_proj = nn.Linear(d, d, bias=False)
        # final phi on the value-pass readout y = phi(M^T r) (paper); None=legacy
        self.value_readout_act = (
            get_activation(cfg.value_readout_act)
            if cfg.value_readout_act != "none"
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.use_conv = cfg.use_short_conv_qk
        if self.use_conv:
            self.q_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
            self.k_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
        self.phi = get_activation(cfg.activation)
        self.f = get_activation(cfg.activation)
        self.alpha_act = get_activation(cfg.alpha_mode)

    def reset_beta_bias(self):
        """Set beta_proj bias = logit(beta_init) so the forget gate STARTS near
        beta_init, not 0.5. beta~0.5 (zero bias) is ~1-token memory half-life;
        the paper wants beta near 1 (ChatGPT-Pro suspect #2). Must be called
        AFTER the model-wide _init_weights (which zeros all Linear biases),
        else it is silently clobbered."""
        with torch.no_grad():
            b0 = math.log(self.cfg.beta_init / (1.0 - self.cfg.beta_init))
            self.beta_proj.bias.fill_(b0)

    def _heads(self, x, last):  # [B,T,H*last] -> [B,H,T,last]
        B, T, _ = x.shape
        return x.view(B, T, self.H, last).permute(0, 2, 1, 3)

    def forward(self, x, training: bool = True):
        cfg = self.cfg
        B, T, d = x.shape
        h = self.norm(x)
        q = self._heads(self.q_proj(h), self.D)  # [B,H,T,D]
        k = self._heads(self.k_proj(h), self.D)
        v = self._heads(self.v_proj(h), self.D)
        if self.use_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
        if cfg.write_l2norm:
            # DeltaNet-style L2 normalization of the write vectors (keys) and the
            # key-pass query, over head_dim, to bound gamma*||w||^2.
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            v = F.normalize(v, dim=-1)
        alpha = self._heads(self.alpha_proj(h), self.M)  # [B,H,T,M]
        alpha = self.alpha_act(alpha)
        # Decay (beta) and inner-LR (gamma) stay fp32 regardless of any outer
        # autocast: bf16 decay precision is poor (Codex) and the head-on bf16
        # test kept these fp32 (q/k/v/alpha may be bf16, which it validated).
        with torch.autocast(device_type=x.device.type, enabled=False):
            hf = h.float()
            if cfg.beta_mode == "scalar_per_head":
                beta = (
                    torch.sigmoid(self.beta_proj(hf))
                    .view(B, T, self.H, 1)
                    .permute(0, 2, 1, 3)
                )
            else:
                beta = torch.sigmoid(self._heads(self.beta_proj(hf), self.M))
            if not cfg.forget_gate:
                beta = torch.ones_like(beta)
            gamma = F.softplus(self.gamma_raw.float())  # [H], positive

        # key pass -> yhat; value pass -> y. The Trellis memory runs in fp32 (the
        # chunk recurrence state, LN-SiLU reductions and decay are fp32; bf16
        # inputs are fine -- the head-on test showed bf16 PPL ~ fp32). Output
        # rejoins any outer autocast at out_proj. Chunked stale path when
        # chunk_size>1 and beta is per-head.
        use_chunk = cfg.chunk_size > 1 and cfg.beta_mode == "scalar_per_head"
        qf, kf, vf = q.float(), k.float(), v.float()
        af, bf, gf = alpha.float(), beta.float(), gamma.float()
        if BF16_INPUTS:
            # round inputs to bf16 precision (decay bf/gf stay fp32)
            qf, kf, vf, af = (t.bfloat16().float() for t in (qf, kf, vf, af))
        with torch.autocast(device_type=x.device.type, enabled=False):
            if use_chunk and cfg.chunk_refine == 0:
                # Phase-1 fast path: evolve key+value states stacked on a 2x
                # batch axis (neither depends on r), then batched readouts.
                cs = cfg.chunk_size
                P, rmat, _ = trellis_chunk_decay(bf, cs)
                nC = P.shape[2]
                write2 = (
                    torch.stack((kf, vf), dim=0)
                    .reshape(2 * B, self.H, T, self.D)
                    .contiguous()
                )
                alpha2 = (
                    af.unsqueeze(0)
                    .expand(2, B, self.H, T, self.M)
                    .reshape(2 * B, self.H, T, self.M)
                    .contiguous()
                )
                P2 = (
                    P.unsqueeze(0)
                    .expand(2, B, self.H, nC, cs, 1)
                    .reshape(2 * B, self.H, nC, cs, 1)
                    .contiguous()
                )
                rmat2 = (
                    rmat.unsqueeze(0)
                    .expand(2, B, self.H, nC, cs, cs)
                    .reshape(2 * B, self.H, nC, cs, cs)
                    .contiguous()
                )
                # The fused Triton kernel handles LN-SiLU only in the square
                # memory case because LN reductions are over the true slot count.
                # Pointwise SiLU/identity can pad slots internally, so the
                # Stage-2 winner/control at n_slots=48,d_head=64 can stay fused.
                triton_pointwise = cfg.activation in ("silu", "identity")
                triton_ln_silu = (
                    cfg.activation == "ln_silu" and cfg.n_slots == cfg.d_head
                )
                nvidia_cuda = write2.is_cuda and torch.version.hip is None
                if HAS_TRITON and nvidia_cuda and (triton_pointwise or triton_ln_silu):
                    # Fused Triton state-evolution: collapses the nC-chunk Python
                    # loop into one kernel (26-44x over the bmm loop), gradient-
                    # equivalent to the PyTorch path (z detached -> true-stale).
                    M0_2, u_2 = TrellisStateEvolutionTriton.apply(
                        write2, alpha2, P2, rmat2, gf, cs, cfg.activation
                    )
                else:
                    M0_2, u_2, _, _, _ = run_trellis_memory_chunked_state_evolution(
                        write2, alpha2, None, gf, self.phi, cs, P=P2, rmat=rmat2
                    )
                M0_2 = M0_2.view(2, B, self.H, nC, self.M, self.D)
                u_2 = u_2.view(2, B, self.H, nC, cs, self.M)
                yhat = run_trellis_memory_chunked_batched_readout(
                    kf, qf, M0_2[0], u_2[0], P, rmat, gf, "M_q", cs, T_out=T
                )
                r = self.f(yhat)
                y = run_trellis_memory_chunked_batched_readout(
                    vf, r, M0_2[1], u_2[1], P, rmat, gf, "M_T_r", cs, T_out=T
                )
            elif use_chunk:
                cs = cfg.chunk_size
                yhat = run_trellis_memory_chunked(
                    kf, qf, af, bf, gf, self.phi, "M_q", cs, cfg.chunk_refine
                )
                r = self.f(yhat)
                y = run_trellis_memory_chunked(
                    vf, r, af, bf, gf, self.phi, "M_T_r", cs, cfg.chunk_refine
                )
            else:
                ex = cfg.exact_inner
                yhat = run_trellis_memory(
                    kf, qf, af, bf, gf, self.phi, "M_q", training, exact_inner=ex
                )
                r = self.f(yhat)  # [B,H,T,M]
                y = run_trellis_memory(
                    vf, r, af, bf, gf, self.phi, "M_T_r", training, exact_inner=ex
                )

        # final phi on the value-pass readout (paper: y = phi(M^T r)). Applied
        # over the head dim D, on the fp32 memory output before head-merge.
        if self.value_readout_act is not None:
            y = self.value_readout_act(y)

        y = y.permute(0, 2, 1, 3).reshape(B, T, self.H * self.D)  # merge heads
        if self.output_path == "paper":
            yn = self.post_norm(y)
            g = F.gelu(self.gate_in(h))
            out = self.out_proj(yn * g)
        else:
            out = self.out_proj(y)
            if self.post_gate:
                out = out * F.silu(self.gate_proj(h))
        return self.drop(out)
