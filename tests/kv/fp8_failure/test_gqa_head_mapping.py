"""GQA broadcast correctness for the gauge. Each KV head is shared by g query heads; the inverse
gauge must reach exactly those query heads (query head h <- kv head h // g). A wrong mapping still
'looks' plausible but breaks score invariance only for the misrouted heads -- so we test the index
math directly AND end-to-end with a per-kv-head-distinct gauge (which only stays invariant if the
routing is right)."""

import torch

import _fixtures as fx

from tools.kv.fp8_failure import architecture_discovery as AD
from tools.kv.fp8_failure import qk_gauge_equalization as G


def test_repeat_interleave_routes_query_head_to_kv_head():
    n_kv, hd, g = 2, 8, 3  # n_q = 6
    D = torch.arange(n_kv * hd, dtype=torch.float32).reshape(n_kv, hd) + 1.0
    qD = D.repeat_interleave(g, dim=0)  # the probe's broadcast
    assert tuple(qD.shape) == (n_kv * g, hd)
    for h in range(n_kv * g):
        assert torch.equal(qD[h], D[h // g])  # query head h uses kv head h//g


def test_fold_qscale_matches_inverse_kv_gauge():
    n_kv, hd, g = 2, 8, 2  # n_q = 4
    Ds = torch.rand(n_kv, hd) + 0.5
    q_scale = (1.0 / Ds).repeat_interleave(g, dim=0).reshape(-1)
    for h in range(n_kv * g):
        assert torch.allclose(q_scale[h * hd:(h + 1) * hd], 1.0 / Ds[h // g])


def test_end_to_end_distinct_per_kv_gauge_stays_invariant():
    # If GQA routing were wrong, a per-kv-head-DISTINCT gauge would corrupt scores. Identical
    # invariance under such a gauge is the strong evidence the mapping is correct.
    m = fx.tiny_qwen2(n_q=6, n_kv=2, head_dim=8)
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        base = m(ids).logits
    for info in infos:
        # deliberately different scale per kv head (col-broadcast distinct rows)
        D = torch.stack([torch.full((8,), 0.5), torch.full((8,), 2.0)])
        G.fold_qk_gauge_weights(info, D, rotary_dim=info["rotary_dim"])
    with torch.no_grad():
        got = m(ids).logits
    assert torch.allclose(base, got, atol=1e-4, rtol=1e-4)
