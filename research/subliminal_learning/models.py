# SPDX-License-Identifier: MIT
"""Packed MNIST models with a common 10 digit + k auxiliary output head.

Every module carries a leading model axis M, as in the reference script's
MultiLinear, so M independent models train in one tensor without sharing
any parameter. An input is either [M, B, 784] (each model sees its own
images) or [B, 784] (all models see the same images, e.g. MNIST test).

paper_mlp is the reference network: 784 -> 256 -> 256 -> 10 + k with
ReLU, 270,093 parameters for k = 3. The residual family is a stem
784 -> d, blocks F(h) = Linear(ReLU(Linear(LayerNorm(h)))), one final
LayerNorm and a d -> 10 + k head; residual_mlp adds F(h) to one stream,
noskip_mlp replaces h by F(h) (the same network without the skip
connection),
hc / hc_static / mhc route it through n streams (see connections.py)
that are initialized as copies of the stem output and averaged before
the final LayerNorm.

Linear weights are drawn N(0, 1/fan_in) with zero biases, in a fixed
order, from the generator of each model's key (init_common), so
architectures that share a shape share the exact weights for a seed and
routing parameters never consume that generator.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .connections import (
    GatedResidual,
    HyperConnection,
    MLayerNorm,
    ManifoldHyperConnection,
    SimpleAltUp,
)
from .data import PIXELS, generator
from .layerrope import LayerRoPE

MULTI_STREAM = ("hc", "hc_static", "mhc", "gr", "altup")

# Read/write/boundary controls; the defaults reproduce each design as
# first implemented (see recipe.py for their meaning).
ROUTING_DEFAULTS = {
    "residual_scale": 1.0,
    "ms_write": "cyclic",
    "ms_write_scale": 1.0,
    "ms_read": "learned",
    "ms_stem": "dup",
    "gr_write": "dense",
    "gr_write_scale": 1.0,
    "mhc_write": "dense",
    "mhc_write_scale": 1.0,
    "post_scale": 1.0,
    "sinkhorn_log": False,
}


class MLinear(nn.Module):
    def __init__(self, n_models, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(n_models, d_out))

    def forward(self, x):
        if x.dim() == 2:
            return torch.matmul(x, self.weight.transpose(1, 2)) + self.bias[:, None]
        return torch.baddbmm(self.bias[:, None], x, self.weight.transpose(1, 2))


class PaperMLP(nn.Module):
    def __init__(self, n_models, total_out, width=256):
        super().__init__()
        self.l1 = MLinear(n_models, PIXELS, width)
        self.l2 = MLinear(n_models, width, width)
        self.head = MLinear(n_models, width, total_out)

    def common_linears(self):
        return [self.l1, self.l2, self.head]

    def forward(self, x):
        return self.head(F.relu(self.l2(F.relu(self.l1(x)))))


class Block(nn.Module):
    def __init__(self, n_models, width, eps, norm=True, layerrope=False):
        super().__init__()
        self.norm = (
            MLayerNorm(n_models, width, eps, affine_gain=not layerrope)
            if norm
            else nn.Identity()
        )
        self.fc1 = MLinear(n_models, width, width)
        self.fc2 = MLinear(n_models, width, width)

    def forward(self, h, gain=None, output_gain=None):
        h = self.norm(h) if gain is None else self.norm(h, gain=gain)
        out = self.fc2(F.relu(self.fc1(h)))
        return out if output_gain is None else out * output_gain[:, None, :]


class ResidualFamily(nn.Module):
    def __init__(
        self,
        n_models,
        arch,
        total_out,
        width,
        blocks,
        streams,
        eps,
        sinkhorn_iters=20,
        mhc_init_logit=4.0,
        norm=True,
        routing=None,
        layerrope=None,
        layerrope_output_gain=False,
    ):
        super().__init__()
        r = {**ROUTING_DEFAULTS, **(routing or {})}
        self.arch = arch
        self.residual_scale = r["residual_scale"]
        self.streams = streams if arch in MULTI_STREAM else 1
        self.stem = MLinear(n_models, PIXELS, width)
        # GR's read normalizes and gates, so it replaces the block pre-norm.
        block_norm = norm and arch != "gr"
        self.layerrope = (
            LayerRoPE(n_models, width, blocks, **layerrope) if layerrope else None
        )
        self.output_layerrope = (
            LayerRoPE(n_models, width, blocks, **layerrope)
            if layerrope and layerrope_output_gain
            else None
        )
        self.blocks = nn.ModuleList(
            Block(n_models, width, eps, block_norm, self.layerrope is not None)
            for _ in range(blocks)
        )
        if arch in ("hc", "hc_static"):
            self.conns = nn.ModuleList(
                HyperConnection(
                    n_models, streams, width, k, arch == "hc", eps, r["post_scale"]
                )
                for k in range(blocks)
            )
        elif arch == "mhc":
            self.conns = nn.ModuleList(
                ManifoldHyperConnection(
                    n_models,
                    streams,
                    width,
                    k,
                    sinkhorn_iters,
                    mhc_init_logit,
                    eps,
                    r["mhc_write"],
                    r["mhc_write_scale"],
                    r["post_scale"],
                    r["sinkhorn_log"],
                )
                for k in range(blocks)
            )
        elif arch == "gr":
            self.conns = nn.ModuleList(
                GatedResidual(
                    n_models,
                    streams,
                    width,
                    eps,
                    layer_id=k,
                    write_mode=r["gr_write"],
                    write_scale=r["gr_write_scale"],
                    post_scale=r["post_scale"],
                )
                for k in range(blocks)
            )
        elif arch == "altup":
            self.conns = nn.ModuleList(
                SimpleAltUp(
                    n_models,
                    streams,
                    k,
                    r["ms_write"],
                    r["ms_write_scale"],
                    r["ms_read"],
                )
                for k in range(blocks)
            )
        else:
            self.conns = None
        # Diversified input streams: fixed (non-trained) random projections
        # of the pixels, centred across streams so the stream mean is still
        # exactly the stem output. Drawn by init_model like routing weights.
        self.stem_mode = r["ms_stem"]
        if self.stem_mode == "diverse":
            assert self.streams > 1, "diverse stem needs several streams"
            self.register_buffer(
                "stem_diverse", torch.zeros(n_models, self.streams, width, PIXELS)
            )
        self.final_norm = MLayerNorm(n_models, width, eps) if norm else nn.Identity()
        self.head = MLinear(n_models, width, total_out)
        self.record = False
        self.last_diag = None

    def random_linears(self):
        if self.stem_mode != "diverse":
            return []
        return [(self.stem_diverse, PIXELS)]

    def common_linears(self):
        out = [self.stem]
        for blk in self.blocks:
            out += [blk.fc1, blk.fc2]
        return out + [self.head]

    def sinkhorn_tracking(self, reset=True):
        """Largest (row, column) H_res error over all forwards since reset."""
        conns = [c for c in (self.conns or []) if hasattr(c, "reset_tracking")]
        if not conns:
            return None
        out = [max(c.track_row for c in conns), max(c.track_col for c in conns)]
        if reset:
            for c in conns:
                c.reset_tracking()
        return out

    def set_record(self, on):
        self.record = on
        for c in self.conns or []:
            c.record = on

    def set_hc_capture(self, on):
        """Capture detached HC carry/write terms without changing its update."""
        if self.arch not in ("hc", "hc_static"):
            raise ValueError("HC term capture requires an HC architecture")
        for conn in self.conns:
            conn.capture_terms = on
            if not on:
                conn.last_terms = None

    def forward(self, x):
        h = self.stem(x)
        gains = self.layerrope() if self.layerrope is not None else None
        output_gains = (
            self.output_layerrope() if self.output_layerrope is not None else None
        )
        block_fns = (
            [
                partial(
                    blk,
                    gain=gains[:, k],
                    output_gain=(
                        output_gains[:, k] if output_gains is not None else None
                    ),
                )
                for k, blk in enumerate(self.blocks)
            ]
            if gains is not None
            else self.blocks
        )
        if self.arch == "noskip_mlp":
            for blk in block_fns:
                h = blk(h)
        elif self.conns is None:
            for blk in block_fns:
                h = h + self.residual_scale * blk(h)
        else:
            hs = h.unsqueeze(2).expand(*h.shape[:2], self.streams, h.shape[-1])
            if self.stem_mode == "diverse":
                if x.dim() == 2:
                    dev = torch.einsum("bi,mndi->mbnd", x, self.stem_diverse)
                else:
                    dev = torch.einsum("mbi,mndi->mbnd", x, self.stem_diverse)
                hs = hs + dev - dev.mean(dim=2, keepdim=True)
            for conn, blk in zip(self.conns, block_fns):
                hs = conn(hs, blk)
            if self.record:
                centre = hs.mean(2, keepdim=True)
                spread = (hs - centre).norm(dim=-1) / centre.norm(dim=-1).clamp_min(
                    1e-12
                )
                self.last_diag = {
                    "stream_spread_max": spread.max().item(),
                    "stream_spread_mean": spread.mean().item(),
                    "blocks": [c.last_diag for c in self.conns],
                }
            h = hs.mean(2)
        return self.head(self.final_norm(h))


def build(recipe, n_models):
    if recipe.arch == "paper_mlp":
        return PaperMLP(n_models, recipe.total_outputs)
    return ResidualFamily(
        n_models,
        recipe.arch,
        recipe.total_outputs,
        recipe.width,
        recipe.blocks,
        recipe.streams,
        recipe.ln_eps,
        recipe.sinkhorn_iters,
        recipe.mhc_init_logit,
        recipe.block_norm,
        {k: getattr(recipe, k) for k in ROUTING_DEFAULTS},
        (
            {
                "base": recipe.layerrope_base,
                "mag_slope": recipe.layerrope_mag_slope,
                "rot_slope": recipe.layerrope_rot_slope,
                "mag_intercept": recipe.layerrope_mag_intercept,
                "rot_intercept": recipe.layerrope_rot_intercept,
            }
            if recipe.layerrope
            else None
        ),
        recipe.layerrope_output_gain,
    )


@torch.no_grad()
def init_common(model, model_keys, stream):
    """Draw every common Linear from each model key's `stream` generator.

    Row m of every weight comes from generator(model_keys[m], stream),
    consumed in common_linears() order; biases are zero. Everything else
    keeps its deterministic construction-time initialization.
    """
    linears = model.common_linears()
    for m, key in enumerate(model_keys):
        g = generator(key, stream)
        for lin in linears:
            d_out, d_in = lin.weight.shape[1:]
            w = torch.randn(d_out, d_in, generator=g) / math.sqrt(d_in)
            lin.weight[m].copy_(w.to(lin.weight.device))
            lin.bias[m].zero_()


@torch.no_grad()
def init_model(model, model_keys, stream):
    """init_common, then any randomly initialized routing weights.

    Routing weights (GR's projections) come from `<stream>/routing`, a
    separate generator, so they never shift the common weights.
    """
    init_common(model, model_keys, stream)
    tensors = []
    for mod in model.modules():
        if hasattr(mod, "random_linears"):
            tensors += mod.random_linears()
    for m, key in enumerate(model_keys):
        g = generator(key, f"{stream}/routing")
        for t, fan_in in tensors:
            w = torch.randn(t.shape[1:], generator=g) / math.sqrt(fan_in)
            t[m].copy_(w.to(t.device))


def count_params_per_model(model, n_models):
    return sum(p.numel() for p in model.parameters()) // n_models
