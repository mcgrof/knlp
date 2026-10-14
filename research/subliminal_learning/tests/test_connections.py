# SPDX-License-Identifier: MIT
"""HC / mHC routing: reference equivalence, reductions, gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.connections import HyperConnection, sinkhorn_knopp
from research.subliminal_learning.models import build, init_common
from research.subliminal_learning.run import student_loss


class PaperHyperConnection(nn.Module):
    """Algorithm 2 of the Hyper-Connections paper, as printed (h: BxLxNxD)."""

    def __init__(self, dim, rate, layer_id, dynamic):
        super().__init__()
        self.rate, self.layer_id, self.dynamic = rate, layer_id, dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,)))
        init_alpha0 = torch.zeros((rate, 1))
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye(rate)], dim=1)
        )
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate + 1)))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim,)))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(dim)

    def width_connection(self, h):
        if self.dynamic:
            norm_h = self.layer_norm(h)
            wc_weight = F.tanh(norm_h @ self.dynamic_alpha_fn)
            alpha = (
                wc_weight * self.dynamic_alpha_scale
                + self.static_alpha[None, None, ...]
            )
            dc_weight = F.tanh(norm_h @ self.dynamic_beta_fn)
            beta = (
                dc_weight * self.dynamic_beta_scale + self.static_beta[None, None, ...]
            )
        else:
            alpha = self.static_alpha[None, None, ...]
            beta = self.static_beta[None, None, ...]
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        return torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]


@pytest.mark.parametrize("dynamic", [True, False])
def test_hc_matches_paper_algorithm(dynamic):
    torch.manual_seed(0)
    n, d, B, layer_id = 4, 8, 6, 5
    paper = PaperHyperConnection(d, n, layer_id, dynamic).double()
    with torch.no_grad():  # move away from the (degenerate) initialization
        for p in paper.parameters():
            p.add_(0.3 * torch.randn_like(p))
    ours = HyperConnection(
        1, n, d, layer_id, dynamic, eps=paper.layer_norm.eps if dynamic else 1e-5
    ).double()
    with torch.no_grad():
        ours.static_alpha[0].copy_(paper.static_alpha)
        ours.static_beta[0].copy_(paper.static_beta)
        if dynamic:
            ours.dynamic_alpha_fn[0].copy_(paper.dynamic_alpha_fn)
            ours.dynamic_alpha_scale[0].copy_(paper.dynamic_alpha_scale)
            ours.dynamic_beta_fn[0].copy_(paper.dynamic_beta_fn)
            ours.dynamic_beta_scale[0].copy_(paper.dynamic_beta_scale)
            ours.norm.weight[0].copy_(paper.layer_norm.weight)
            ours.norm.bias[0].copy_(paper.layer_norm.bias)
    layer = nn.Sequential(nn.Linear(d, d), nn.Tanh()).double()
    h = torch.randn(B, 1, n, d, dtype=torch.double, requires_grad=True)
    mix, beta = paper.width_connection(h)
    ref = paper.depth_connection(mix, layer(mix[..., 0, :]), beta)
    h2 = h.detach().clone().requires_grad_(True)
    out = ours(h2.transpose(0, 1), layer).transpose(0, 1)  # [1,B,n,d] model axis
    torch.testing.assert_close(out, ref)
    g = torch.randn_like(ref)
    (ref * g).sum().backward()
    (out * g).sum().backward()
    torch.testing.assert_close(h2.grad, h.grad)


def residual_pair(arch, **kw):
    r = Recipe(arch=arch, width=16, blocks=3, streams=4, **kw)
    base = build(Recipe(arch="residual_mlp", width=16, blocks=3), 2).double()
    model = build(r, 2).double()
    for m in (base, model):
        init_common(m, ["seed-1", "seed-2"], "init")
    return base, model


@pytest.mark.parametrize("arch", ["hc", "hc_static"])
def test_hc_initialization_is_the_residual_network(arch):
    base, model = residual_pair(arch)
    x = torch.randn(10, 784, dtype=torch.double)
    torch.testing.assert_close(model(x), base(x))


def test_mhc_unit_routing_is_the_residual_network():
    # With the gates at zero and |logit| = 60 the maps are exact in double:
    # H_pre = e_{k mod n}, H_post = 1, H_res = I.
    base, model = residual_pair("mhc", mhc_init_logit=60.0)
    with torch.no_grad():
        for c in model.conns:
            c.alpha_pre.zero_()
            c.alpha_post.zero_()
            c.alpha_res.zero_()
    x = torch.randn(10, 784, dtype=torch.double)
    torch.testing.assert_close(model(x), base(x))


def test_mhc_default_init_is_close_to_residual():
    base, model = residual_pair("mhc")
    x = torch.randn(64, 784, dtype=torch.double)
    rel = (model(x) - base(x)).norm() / base(x).norm()
    assert rel < 0.1, rel


def test_sinkhorn_is_doubly_stochastic_and_differentiable():
    torch.manual_seed(0)
    # Moderate logits, as at and near the mHC initialization. Widely spread
    # logits need more than 20 iterations for 1e-4 column sums, which is
    # why the runner records the residuals instead of assuming them.
    logits = torch.randn(5, 4, 4, dtype=torch.double)
    m = sinkhorn_knopp(logits, 20)
    assert (m.sum(-1) - 1).abs().max() < 1e-12
    assert (m.sum(-2) - 1).abs().max() < 1e-4
    assert m.min() > 0
    small = torch.randn(2, 3, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: sinkhorn_knopp(t, 20), (small,))


@pytest.mark.parametrize("arch", ["hc", "mhc"])
def test_streams_can_diverge_and_gradients_are_finite(arch):
    torch.manual_seed(0)
    r = Recipe(arch=arch, width=16, blocks=4, streams=4)
    model = build(r, 1)
    init_common(model, ["seed-1"], "init")
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.2 * torch.randn_like(p))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.rand(1, 128, 784) * 2 - 1
    model.set_record(True)
    model(x)
    before = model.last_diag["stream_spread_max"]
    for _ in range(30):
        with torch.no_grad():
            tgt = teacher(x)
        loss, _, _ = student_loss(model(x), tgt, "per_example_v1", 1)
        opt.zero_grad()
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        opt.step()
    model(x)
    assert before < 1e-6
    assert model.last_diag["stream_spread_max"] > 1e-3
    if arch == "mhc":
        for blk in model.last_diag["blocks"]:
            assert blk["row_residual_max"] < 1e-4 and blk["col_residual_max"] < 1e-4


def test_parameter_counts():
    r = Recipe(arch="residual_mlp")
    assert sum(p.numel() for p in build(r, 1).parameters()) == 733197


def test_noskip_is_the_residual_network_without_the_addition():
    r = build(Recipe(arch="residual_mlp", width=16, blocks=3), 1).double()
    n = build(Recipe(arch="noskip_mlp", width=16, blocks=3), 1).double()
    for m in (r, n):
        init_common(m, ["seed-1"], "init")
    assert sum(p.numel() for p in r.parameters()) == sum(
        p.numel() for p in n.parameters()
    )
    x = torch.randn(8, 784, dtype=torch.double)
    h = n.stem(x)
    for blk in n.blocks:
        h = blk(h)
    torch.testing.assert_close(n(x), n.head(n.final_norm(h)))
    assert not torch.allclose(n(x), r(x))


def test_block_norm_off_removes_every_layernorm():
    on = build(Recipe(arch="noskip_mlp"), 1)
    off = build(Recipe(arch="noskip_mlp", block_norm=False), 1)
    n_on = sum(p.numel() for p in on.parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_on - n_off == 2 * 256 * (4 + 1)
    with pytest.raises(ValueError):
        from research.subliminal_learning.recipe import validate

        validate(Recipe(arch="hc", block_norm=False))


def test_gated_residual_matches_the_report_equations():
    from research.subliminal_learning.connections import GatedResidual
    from research.subliminal_learning.models import init_model

    torch.manual_seed(0)
    m = build(Recipe(arch="gr", width=16, blocks=2, streams=4), 1).double()
    init_model(m, ["seed-1"], "init")
    gr = m.conns[1]
    assert isinstance(gr, GatedResidual)
    with torch.no_grad():
        gr.norm.weight.add_(0.1 * torch.randn_like(gr.norm.weight))
    h = torch.randn(1, 5, 4, 16, dtype=torch.double)
    layer = m.blocks[1]
    out = gr(h, layer)
    # Eqs. 30-34 written out for the single model, one image at a time.
    for b in range(5):
        R = h[0, b]
        Rh = torch.stack(
            [
                R[i]
                * torch.rsqrt(R[i].pow(2).mean() + gr.norm.eps)
                * gr.norm.weight[0, i]
                for i in range(4)
            ]
        )
        v = Rh.reshape(-1)
        G = torch.sigmoid((F.silu((v @ gr.w_down[0]) / 4) @ gr.w_up[0]).reshape(4, 16))
        x = (G * Rh).mean(0)
        s = 2 * torch.sigmoid((v @ gr.w_write[0]) / 4)
        y = layer(x[None, None])[0, 0]
        ref = R + s[:, None] * y[None]
        torch.testing.assert_close(out[0, b], ref)
    assert isinstance(layer.norm, torch.nn.Identity)  # GR replaces the pre-norm


def test_altup_writes_one_branch_round_robin():
    m = build(Recipe(arch="altup", width=16, blocks=3, streams=4), 1)
    init_common(m, ["seed-1"], "init")
    h = torch.randn(1, 3, 4, 16)
    for k, (c, blk) in enumerate(zip(m.conns, m.blocks)):
        out = c(h, blk)
        changed = [(out[:, :, i] != h[:, :, i]).any().item() for i in range(4)]
        assert changed == [i == k % 4 for i in range(4)]


def test_routing_weights_do_not_shift_common_weights():
    from research.subliminal_learning.models import init_model

    r = build(Recipe(arch="residual_mlp", width=16, blocks=2), 1)
    g = build(Recipe(arch="gr", width=16, blocks=2, streams=4), 1)
    init_model(r, ["seed-3"], "init")
    init_model(g, ["seed-3"], "init")
    for a, b in zip(r.common_linears(), g.common_linears()):
        assert torch.equal(a.weight, b.weight)
    g2 = build(Recipe(arch="gr", width=16, blocks=2, streams=4), 1)
    init_model(g2, ["seed-3"], "init_other")
    assert not torch.equal(g.conns[0].w_down, g2.conns[0].w_down)


@pytest.mark.parametrize("arch", ["gr", "altup"])
def test_new_branches_diverge_and_train(arch):
    torch.manual_seed(0)
    from research.subliminal_learning.models import init_model

    model = build(Recipe(arch=arch, width=16, blocks=4, streams=4), 1)
    init_model(model, ["seed-1"], "init")
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.2 * torch.randn_like(p))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.rand(1, 128, 784) * 2 - 1
    for _ in range(30):
        with torch.no_grad():
            tgt = teacher(x)
        loss, _, _ = student_loss(model(x), tgt, "per_example_v1", 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.set_record(True)
    model(x)
    assert model.last_diag["stream_spread_max"] > 1e-3


def _pair(arch_a, kw_a, arch_b, kw_b, blocks=3, width=16):
    from research.subliminal_learning.models import init_model

    a = build(Recipe(arch=arch_a, width=width, blocks=blocks, **kw_a), 1).double()
    b = build(Recipe(arch=arch_b, width=width, blocks=blocks, **kw_b), 1).double()
    for m in (a, b):
        init_model(m, ["seed-7"], "init")
    return a, b


@pytest.mark.parametrize("n", [2, 4, 8])
def test_uniform_read_cyclic_altup_is_a_scaled_residual_network(n):
    """mean(R') = mean(R) + F(mean(R))/n when reads are frozen at 1/n."""
    alt, res = _pair(
        "altup",
        {"streams": n, "ms_read": "uniform"},
        "residual_mlp",
        {"residual_scale": 1.0 / n},
    )
    x = torch.randn(6, 784, dtype=torch.double)
    torch.testing.assert_close(alt(x), res(x))
    alt(x).square().sum().backward()
    res(x).square().sum().backward()
    for la, lr in zip(alt.common_linears(), res.common_linears()):
        torch.testing.assert_close(la.weight.grad, lr.weight.grad)


def test_route_write_modes():
    from research.subliminal_learning.connections import route_write

    s = torch.tensor([[[1.0, 2.0, 2.0, 4.0]]])
    torch.testing.assert_close(route_write(s, "dense", 1), s)
    torch.testing.assert_close(
        route_write(s, "cyclic", 1, 3.0), torch.tensor([[[0.0, 6.0, 0.0, 0.0]]])
    )
    torch.testing.assert_close(
        route_write(s, "cyclic_l2", 2), torch.tensor([[[0.0, 0.0, 5.0, 0.0]]])
    )
    torch.testing.assert_close(
        route_write(s, "cyclic_sum", 3), torch.tensor([[[0.0, 0.0, 0.0, 9.0]]])
    )


def test_dense_altup_with_copied_streams_is_one_stream():
    alt, res = _pair(
        "altup", {"ms_write": "dense", "ms_read": "uniform"}, "residual_mlp", {}
    )
    x = torch.randn(6, 784, dtype=torch.double)
    torch.testing.assert_close(alt(x), res(x))


def test_diverse_stem_keeps_the_stream_mean():
    from research.subliminal_learning.models import init_model

    m = build(Recipe(arch="altup", width=16, blocks=0, ms_stem="diverse"), 1).double()
    init_model(m, ["seed-1"], "init")
    x = torch.randn(5, 784, dtype=torch.double)
    h = m.stem(x)
    hs = h.unsqueeze(2).expand(1, 5, 4, 16)
    dev = torch.einsum("bi,mndi->mbnd", x, m.stem_diverse)
    streams = hs + dev - dev.mean(2, keepdim=True)
    torch.testing.assert_close(streams.mean(2), h)
    assert (streams - h.unsqueeze(2)).abs().max() > 0.1
    m.set_record(True)
    m(x)


@pytest.mark.parametrize("arch", ["gr", "mhc"])
def test_cyclic_write_touches_one_branch(arch):
    from research.subliminal_learning.models import init_model

    m = build(Recipe(arch=arch, width=16, blocks=2, **{f"{arch}_write": "cyclic"}), 1)
    init_model(m, ["seed-1"], "init")
    h = torch.randn(1, 3, 4, 16)
    for k, (c, blk) in enumerate(zip(m.conns, m.blocks)):
        if arch == "mhc":
            _, _, h_res = c.maps(h)
            base = torch.einsum("mbij,mbjd->mbid", h_res, h)
            assert (h_res.sum(-1) - 1).abs().max() < 1e-5
        else:
            base = h
        delta = c(h, blk) - base
        touched = [delta[:, :, i].abs().max().item() > 0 for i in range(4)]
        assert touched == [i == k % 4 for i in range(4)]


def test_common_weights_shared_across_routing_controls():
    from research.subliminal_learning.models import init_model

    ms = [
        build(Recipe(arch=a, width=16, blocks=2, **kw), 1)
        for a, kw in (
            ("residual_mlp", {}),
            ("altup", {"ms_stem": "diverse"}),
            ("gr", {"gr_write": "cyclic"}),
            ("mhc", {"mhc_write": "cyclic"}),
        )
    ]
    for m in ms:
        init_model(m, ["seed-4"], "init")
    for other in ms[1:]:
        for a, b in zip(ms[0].common_linears(), other.common_linears()):
            assert torch.equal(a.weight, b.weight)


@pytest.mark.parametrize("arch", ["hc", "hc_static"])
def test_hc_post_scale_initially_equals_scaled_residual(arch):
    base, model = _pair(
        "residual_mlp", {"residual_scale": 0.25}, arch, {"post_scale": 0.25}
    )
    x = torch.randn(6, 784, dtype=torch.double)
    torch.testing.assert_close(model(x), base(x))


def test_gr_and_mhc_post_scale_scales_the_write_only():
    from research.subliminal_learning.models import init_model

    for arch in ("gr", "mhc"):
        a = build(Recipe(arch=arch, width=16, blocks=1), 1).double()
        b = build(Recipe(arch=arch, width=16, blocks=1, post_scale=0.25), 1).double()
        for m in (a, b):
            init_model(m, ["seed-2"], "init")
        h = torch.randn(1, 3, 4, 16, dtype=torch.double)
        blk = a.blocks[0]
        base = (
            h
            if arch == "gr"
            else torch.einsum("mbij,mbjd->mbid", a.conns[0].maps(h)[2], h)
        )
        da = a.conns[0](h, blk) - base
        db = b.conns[0](h, blk) - base
        torch.testing.assert_close(db, 0.25 * da)


def test_sinkhorn_log_domain_matches_and_tracking_records():
    from research.subliminal_learning.connections import sinkhorn_knopp_log

    logits = torch.randn(3, 4, 4, dtype=torch.double)
    torch.testing.assert_close(
        sinkhorn_knopp_log(logits, 30), sinkhorn_knopp(logits, 30)
    )
    m = build(Recipe(arch="mhc", width=16, blocks=2, sinkhorn_iters=1), 1)
    with torch.no_grad():  # the initial H_res is symmetric; break that
        for c in m.conns:
            c.b_res.add_(torch.randn_like(c.b_res))
    m(torch.randn(4, 784))
    row, col = m.sinkhorn_tracking()
    assert col > 0 and m.sinkhorn_tracking(reset=False) == [0.0, 0.0]
