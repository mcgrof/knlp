# SPDX-License-Identifier: MIT
"""Hyper-Connections and Manifold-Constrained Hyper-Connections.

Both operate on a hyper hidden state H of shape [M, B, n, d]: M packed
independent models, B images, n residual streams of width d. Every
parameter carries a leading model axis so packed models never share
weights.

Hyper-Connections (HC) follow Zhu et al., ICLR 2025
(https://arxiv.org/pdf/2409.19606v3), Eqs. 1-14 and Algorithm 2. The
width connection mixes the streams with alpha^T H into one layer input
and n carried streams; the depth connection adds beta_j times the layer
output to carried stream j. Dynamic HC adds, per stream, a LayerNorm,
a zero-initialized linear map, tanh and a learnable scale initialized to
0.01 on top of the static matrices. Static alpha is initialized to
(e_{k mod n}, I) and static beta to ones, which makes the initial
network equivalent to a pre-norm residual network (Eq. 14).

Manifold-Constrained HC (mHC) follow Xie et al.
(https://arxiv.org/pdf/2512.24880v2), Eqs. 3, 7-9: the n streams are
flattened to one n*d vector, RMS-normalized, and linearly projected to
the pre (1 x n), post (1 x n) and residual (n x n) maps, each scaled by
its own gating factor (initialized to 0.01, the paper's Table 5) plus a
static bias. H_pre = sigmoid, H_post = 2 * sigmoid, and H_res is the
Sinkhorn-Knopp projection of exp(logits) with alternating column then
row normalization (Eq. 9, t_max iterations).

The mHC paper does not state its bias initialization. We encode the HC
initialization through those maps with a magnitude c: pre logits +c on
stream k mod n and -c elsewhere, residual logits +c on the diagonal and
-c elsewhere, post logits 0 (so H_post = 1 exactly). Reading a single
stream per block matters: with uniform reads and writes every stream
receives identical gradients forever and n streams collapse into one.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _bcast(t, like, lead=1):
    """View a [M, ...] tensor so it broadcasts against `like` [M, *, ...]."""
    extra = like.dim() - t.dim()
    return t.view(t.shape[:lead] + (1,) * extra + t.shape[lead:])


class MLayerNorm(nn.Module):
    """LayerNorm over the last dim with a separate affine per model."""

    def __init__(self, n_models, dim, eps, affine_gain=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_models, dim)) if affine_gain else None
        self.bias = nn.Parameter(torch.zeros(n_models, dim))

    def forward(self, x, gain=None):
        y = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
        gain = self.weight if gain is None else gain
        if gain is None:
            raise ValueError("LayerNorm without an affine gain needs an external gain")
        return y * _bcast(gain, y) + _bcast(self.bias, y)


def sinkhorn_knopp(logits, iters):
    """Project exp(logits) [..., n, n] towards the doubly stochastic set.

    Each iteration normalizes columns then rows (M <- T_r(T_c(M))), so
    rows sum to one exactly and columns approximately after finite
    iterations. The max subtraction only improves the range of exp and
    cancels in the normalization.
    """
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True).detach())
    for _ in range(iters):
        m = m / m.sum(dim=-2, keepdim=True)
        m = m / m.sum(dim=-1, keepdim=True)
    return m


def sinkhorn_knopp_log(logits, iters):
    """The same column-then-row iteration carried out in the log domain.

    Mathematically identical to sinkhorn_knopp; it avoids underflow of
    exp for very negative logits but does not speed up convergence for
    sharp matrices.
    """
    la = logits
    for _ in range(iters):
        la = la - torch.logsumexp(la, dim=-2, keepdim=True)
        la = la - torch.logsumexp(la, dim=-1, keepdim=True)
    return la.exp()


def sinkhorn_errors(m):
    """Largest |row sum - 1| and |column sum - 1| of [..., n, n]."""
    return (m.sum(-1) - 1).abs().max().item(), (m.sum(-2) - 1).abs().max().item()


class HyperConnection(nn.Module):
    def __init__(self, n_models, streams, dim, layer_id, dynamic, eps, post_scale=1.0):
        super().__init__()
        n = streams
        self.streams = n
        self.dynamic = dynamic
        # A fixed factor on the whole (static plus dynamic) write beta.
        self.post_scale = post_scale
        alpha0 = torch.zeros(n, 1)
        alpha0[layer_id % n, 0] = 1.0
        static_alpha = torch.cat([alpha0, torch.eye(n)], dim=1)
        self.static_alpha = nn.Parameter(static_alpha.repeat(n_models, 1, 1))
        self.static_beta = nn.Parameter(torch.ones(n_models, n))
        if dynamic:
            self.norm = MLayerNorm(n_models, dim, eps)
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(n_models, dim, n + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.full((n_models, 1), 0.01))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(n_models, dim))
            self.dynamic_beta_scale = nn.Parameter(torch.full((n_models, 1), 0.01))
        self.record = False
        self.last_diag = None
        # Rich HC mechanism probes are opt-in and detached.  They must never
        # participate in the training graph or alter the returned update.
        self.capture_terms = False
        self.last_terms = None

    def forward(self, h, layer):
        m, b, n, d = h.shape
        if self.dynamic:
            norm_h = self.norm(h)
            wc = torch.tanh(
                torch.einsum("mbnd,mde->mbne", norm_h, self.dynamic_alpha_fn)
            )
            alpha = wc * self.dynamic_alpha_scale.view(m, 1, 1, 1)
            alpha = alpha + self.static_alpha[:, None]
            dc = torch.tanh(torch.einsum("mbnd,md->mbn", norm_h, self.dynamic_beta_fn))
            beta = dc * self.dynamic_beta_scale.view(m, 1, 1)
            beta = beta + self.static_beta[:, None]
            mix = torch.einsum("mbne,mbnd->mbed", alpha, h)
        else:
            alpha = self.static_alpha
            beta = self.static_beta[:, None].expand(m, b, n)
            mix = torch.einsum("mne,mbnd->mbed", alpha, h)
        block_input = mix[:, :, 0]
        carried = mix[:, :, 1:]
        out = layer(block_input)
        if self.record:
            self.last_diag = {
                "beta_mean": beta.detach().mean().item(),
                "alpha_read_mean": alpha.detach()[..., 0].mean().item(),
            }
        scaled_beta = beta * self.post_scale
        write = scaled_beta.unsqueeze(-1) * out.unsqueeze(2)
        updated = carried + write
        if self.capture_terms:
            alpha_batch = (
                alpha if alpha.dim() == 4 else alpha[:, None].expand(m, b, n, n + 1)
            )
            carry = carried - h
            self.last_terms = {
                "h": h.detach(),
                "carry": carry.detach(),
                "write": write.detach(),
                "combined": (carry + write).detach(),
                "updated": updated.detach(),
                "block_input": block_input.detach(),
                "block_output": out.detach(),
                "beta_unscaled": beta.detach(),
                "beta_scaled": scaled_beta.detach(),
                "read": alpha_batch[..., 0].detach(),
                "carry_matrix": alpha_batch[..., 1:].detach(),
            }
        return updated


class ManifoldHyperConnection(nn.Module):
    def __init__(
        self,
        n_models,
        streams,
        dim,
        layer_id,
        iters,
        init_logit,
        eps,
        write_mode="dense",
        write_scale=1.0,
        post_scale=1.0,
        log_domain=False,
    ):
        super().__init__()
        n = streams
        self.streams = n
        self.target = layer_id % n
        self.write_mode = write_mode
        self.write_scale = write_scale
        # A fixed factor on the whole post (write) map; the doubly
        # stochastic mixer and the read are untouched.
        self.post_scale = post_scale
        self.log_domain = log_domain
        # Largest row/column error of H_res seen in any forward (training
        # included) since the last reset_tracking().
        self.track_row = 0.0
        self.track_col = 0.0
        self.iters = iters
        self.eps = eps
        c = float(init_logit)
        self.rms_weight = nn.Parameter(torch.ones(n_models, n * dim))
        self.phi = nn.Parameter(torch.zeros(n_models, n * dim, n * n + 2 * n))
        self.alpha_pre = nn.Parameter(torch.full((n_models, 1), 0.01))
        self.alpha_post = nn.Parameter(torch.full((n_models, 1), 0.01))
        self.alpha_res = nn.Parameter(torch.full((n_models, 1), 0.01))
        b_pre = torch.full((n,), -c)
        b_pre[layer_id % n] = c
        b_res = torch.full((n, n), -c)
        b_res.fill_diagonal_(c)
        self.b_pre = nn.Parameter(b_pre.repeat(n_models, 1))
        self.b_post = nn.Parameter(torch.zeros(n_models, n))
        self.b_res = nn.Parameter(b_res.repeat(n_models, 1, 1))
        self.record = False
        self.last_diag = None

    def maps(self, h):
        m, b, n, d = h.shape
        x = h.reshape(m, b, n * d)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.rms_weight[:, None]
        proj = torch.bmm(x, self.phi)
        pre = self.alpha_pre[:, None] * proj[..., :n] + self.b_pre[:, None]
        post = self.alpha_post[:, None] * proj[..., n : 2 * n] + self.b_post[:, None]
        res = self.alpha_res.view(m, 1, 1, 1) * proj[..., 2 * n :].reshape(m, b, n, n)
        res = res + self.b_res[:, None]
        return (
            torch.sigmoid(pre),
            2 * torch.sigmoid(post),
            (sinkhorn_knopp_log if self.log_domain else sinkhorn_knopp)(
                res, self.iters
            ),
        )

    def reset_tracking(self):
        self.track_row = 0.0
        self.track_col = 0.0

    def forward(self, h, layer):
        h_pre, h_post, h_res = self.maps(h)
        with torch.no_grad():
            row, col = sinkhorn_errors(h_res.detach())
            self.track_row = max(self.track_row, row)
            self.track_col = max(self.track_col, col)
        out = layer(torch.einsum("mbn,mbnd->mbd", h_pre, h))
        if self.record:
            r = h_res.detach()
            self.last_diag = {
                "row_residual_max": (r.sum(-1) - 1).abs().max().item(),
                "col_residual_max": (r.sum(-2) - 1).abs().max().item(),
                "min_entry": r.min().item(),
                "diag_mean": torch.diagonal(r, dim1=-2, dim2=-1).mean().item(),
                "pre_read_sum_mean": h_pre.detach().sum(-1).mean().item(),
                "post_mean": h_post.detach().mean().item(),
                **write_ratios(h_post.detach(), self.target),
            }
        # Only the post (write) map is routed; the doubly stochastic mixer
        # is left intact, since masking it would break its row/column sums.
        h_post = route_write(h_post, self.write_mode, self.target, self.write_scale)
        h_post = h_post * self.post_scale
        mixed = torch.einsum("mbij,mbjd->mbid", h_res, h)
        return mixed + h_post.unsqueeze(-1) * out.unsqueeze(2)


WRITE_MODES = ("dense", "cyclic", "cyclic_l2", "cyclic_sum")


def route_write(coeff, mode, target, scale=1.0):
    """Route dense per-branch write weights coeff [M, B, n].

    dense keeps them. The cyclic modes put the whole write on branch
    `target` (block index mod n): cyclic uses scale * coeff[target], a
    fixed amplitude; cyclic_l2 and cyclic_sum replace it per sample by the
    L2 norm or the sum of the dense weights, matching the dense write's
    norm or its effect on the branch mean. The per-sample forms change the
    gate's derivatives and are diagnostics, not candidate designs.
    """
    if mode == "dense":
        return coeff
    if mode == "cyclic":
        val = scale * coeff[..., target]
    elif mode == "cyclic_l2":
        val = coeff.norm(dim=-1)
    elif mode == "cyclic_sum":
        val = coeff.sum(dim=-1)
    else:
        raise ValueError(f"unknown write mode {mode!r}")
    onehot = F.one_hot(torch.tensor(target), coeff.shape[-1]).to(coeff)
    return onehot * val.unsqueeze(-1)


def write_ratios(coeff, target):
    """Initial-state calibration: ||s||_2 / s_t and sum(s) / s_t."""
    sel = coeff[..., target]
    return {
        "l2_over_selected": (coeff.norm(dim=-1) / sel).mean().item(),
        "sum_over_selected": (coeff.sum(dim=-1) / sel).mean().item(),
    }


class MRMSNorm(nn.Module):
    """RMSNorm over the last dim with a separate gain per model (and group)."""

    def __init__(self, gain_shape, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(gain_shape))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * _bcast(self.weight, x)


class GatedResidual(nn.Module):
    """Gated Residual (GR) from the Qwen3.8-Next architecture report.

    Qwen Team, "On the Design of Qwen3.8-Next Architecture" (arXiv
    2608.30320v1), Section 2.2, Eqs. 30-34. Each branch is RMS-normalized
    with its own gain; an elementwise sigmoid gate per branch and channel
    is predicted from all normalized branches through a rank-r bottleneck
    with SiLU; the block input is the branch average of the gated branches;
    the block output is written to branch i with weight 2 * sigmoid of a
    linear map of all branches. There is no branch-mixing matrix, and the
    read replaces the block's own pre-normalization. The projections have
    no static term and start from random weights (random_linears), drawn
    from their own generator stream so the common weights stay shared.
    """

    def __init__(
        self,
        n_models,
        streams,
        dim,
        eps,
        rank=None,
        layer_id=0,
        write_mode="dense",
        write_scale=1.0,
        post_scale=1.0,
    ):
        super().__init__()
        n = streams
        self.streams = n
        self.target = layer_id % n
        self.write_mode = write_mode
        self.write_scale = write_scale
        self.post_scale = post_scale
        r = rank or max(1, dim // 8)
        self.norm = MRMSNorm((n_models, n, dim), eps)
        self.w_down = nn.Parameter(torch.empty(n_models, n * dim, r))
        self.w_up = nn.Parameter(torch.empty(n_models, r, n * dim))
        self.w_write = nn.Parameter(torch.empty(n_models, n * dim, n))
        self.record = False
        self.last_diag = None

    def random_linears(self):
        """(tensor, fan_in) pairs initialized N(0, 1/fan_in) per model."""
        return [
            (self.w_down, self.w_down.shape[1]),
            (self.w_up, self.w_up.shape[1]),
            (self.w_write, self.w_write.shape[1]),
        ]

    def forward(self, h, layer):
        m, b, n, d = h.shape
        hn = self.norm(h)
        flat = hn.reshape(m, b, n * d)
        gate = torch.sigmoid(
            torch.bmm(F.silu(torch.bmm(flat, self.w_down) / n), self.w_up)
        ).view(m, b, n, d)
        x = (gate * hn).mean(dim=2)
        dense = 2 * torch.sigmoid(torch.bmm(flat, self.w_write) / n)
        write = route_write(dense, self.write_mode, self.target, self.write_scale)
        write = write * self.post_scale
        if self.record:
            self.last_diag = {
                "gate_mean": gate.detach().mean().item(),
                "write_mean": dense.detach().mean().item(),
                "write_std": dense.detach().std().item(),
                **write_ratios(dense.detach(), self.target),
            }
        return h + write.unsqueeze(-1) * layer(x).unsqueeze(2)


class SimpleAltUp(nn.Module):
    """Width-only baseline: the simplified AltUp of the same report (Eqs. 21-22).

    Block l reads a weighted sum of the branches and adds its output to
    branch l mod n only. The read weights start at 1/n (the report does
    not give an initialization), so the first read is the branch average.
    The block keeps its own pre-normalization.

    Controls: read="uniform" freezes the read at 1/n (then, with
    duplicated input streams, the branch mean follows exactly
    mean' = mean + scale * F(mean) / n, a residual network with its update
    scaled by scale/n); write_mode="dense" adds the output to every branch
    with weight one; cyclic_l2 / cyclic_sum and write_scale match the
    single write's amplitude to the dense write's L2 norm or sum.
    """

    def __init__(
        self,
        n_models,
        streams,
        layer_id,
        write_mode="cyclic",
        write_scale=1.0,
        read="learned",
    ):
        super().__init__()
        self.streams = streams
        self.target = layer_id % streams
        self.write_mode = write_mode
        self.write_scale = write_scale
        init = torch.full((n_models, streams), 1.0 / streams)
        if read == "learned":
            self.read = nn.Parameter(init)
        elif read == "uniform":
            self.register_buffer("read", init)
        else:
            raise ValueError(f"unknown read mode {read!r}")
        self.record = False
        self.last_diag = None

    def forward(self, h, layer):
        m, b, n, d = h.shape
        y = layer(torch.einsum("mn,mbnd->mbd", self.read, h))
        if self.record:
            self.last_diag = {"read": self.read.detach().mean(0).tolist()}
        coeff = torch.ones(m, b, n, dtype=h.dtype, device=h.device)
        write = route_write(coeff, self.write_mode, self.target, self.write_scale)
        return h + write.unsqueeze(-1) * y.unsqueeze(2)
