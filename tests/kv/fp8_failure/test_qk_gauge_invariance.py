"""The load-bearing test for Phase 6. The QK diagonal gauge (K_i <- D_i K_i, Q_i <- Q_i / D_i)
must leave attention scores invariant -- otherwise it is not a gauge, it is a model edit. We check:
(1) the raw score identity for any per-channel D, including GQA expansion; (2) the activation probe
leaves a real model's logits unchanged when no quant is applied; (3) the weight-fold leaves logits
unchanged for full-RoPE models (the symmetrize-then-commute claim); (4) D = identity is an exact
no-op; (5) restore() exactly undoes the fold."""

import torch

import _fixtures as fx

from tools.kv.fp8_failure import architecture_discovery as AD
from tools.kv.fp8_failure import qk_gauge_equalization as G


def test_raw_score_identity_any_D():
    torch.manual_seed(0)
    B, Hq, Hkv, T, D, g = 1, 4, 2, 6, 8, 2
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    Dvec = torch.rand(Hkv, D) * 2 + 0.25  # positive per-(kv head, channel) gauge
    base = G.scores_from_qk(q, k, scaling=1.0, num_kv_groups=g)
    qD = Dvec.repeat_interleave(g, dim=0)  # broadcast kv-head gauge to its query heads
    q2 = q / qD[None, :, None, :]
    k2 = k * Dvec[None, :, None, :]
    gauged = G.scores_from_qk(q2, k2, scaling=1.0, num_kv_groups=g)
    assert torch.allclose(base, gauged, atol=1e-5, rtol=1e-5)


def test_symmetrize_shares_within_rotary_pair():
    torch.manual_seed(1)
    D = torch.rand(2, 8) * 2 + 0.5
    Ds = G.symmetrize_rotary_D(D, rotary_dim=4)
    half = 2
    # within the rotary block, a channel and its partner share the gauge; tail is untouched
    assert torch.allclose(Ds[..., :half], Ds[..., half:2 * half])
    assert torch.equal(Ds[..., 4:], D[..., 4:])
    # geometric mean preserves the product of each pair
    assert torch.allclose(Ds[..., :half] * Ds[..., half:2 * half], D[..., :half] * D[..., half:2 * half])


def test_probe_no_quant_preserves_logits():
    m = fx.tiny_llama(n_q=4, n_kv=2, head_dim=8)
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        base = m(ids).logits
    D_by_layer = {i["layer_idx"]: torch.rand(2, 8) * 2 + 0.3 for i in infos}
    with G.QKGaugeProbe(m, infos, D_by_layer=D_by_layer, quant_k=None):
        with torch.no_grad():
            got = m(ids).logits
    assert torch.allclose(base, got, atol=1e-4, rtol=1e-4)


def test_probe_identity_D_is_noop():
    m = fx.tiny_llama()
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        base = m(ids).logits
    D_by_layer = {i["layer_idx"]: torch.ones(2, 8) for i in infos}
    with G.QKGaugeProbe(m, infos, D_by_layer=D_by_layer, quant_k="fp8:per_tensor"):
        with torch.no_grad():
            got = m(ids).logits
    # D=I makes the gauge a no-op; only the (identity-scaled) FP8 of K differs -> still close,
    # and equal to plain FP8-K with no gauge. Compare against gauge-free FP8-K.
    import k_bias_common as kbc

    with kbc.FlexKVHarness(m, infos, kbc.parse_spec("fp8:per_tensor"), kbc.parse_spec("bf16")):
        with torch.no_grad():
            plain_fp8 = m(ids).logits
    assert torch.allclose(got, plain_fp8, atol=1e-4, rtol=1e-4)


def test_weight_fold_fullrope_preserves_logits_and_restores():
    m = fx.tiny_qwen2(n_q=4, n_kv=2, head_dim=8)  # biased -> also tests bias transform
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        base = m(ids).logits
    restores = []
    for info in infos:
        D = torch.rand(info["n_kv_heads"], info["head_dim"]) * 1.5 + 0.4
        restores.append(G.fold_qk_gauge_weights(info, D, rotary_dim=info["rotary_dim"]))
    with torch.no_grad():
        folded = m(ids).logits
    assert torch.allclose(base, folded, atol=1e-4, rtol=1e-4)
    for r in restores:
        r()
    with torch.no_grad():
        restored = m(ids).logits
    assert torch.equal(base, restored)


def test_weight_fold_identity_D_unchanged():
    m = fx.tiny_llama()
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        base = m(ids).logits
    for info in infos:
        D = torch.ones(info["n_kv_heads"], info["head_dim"])
        G.fold_qk_gauge_weights(info, D, rotary_dim=info["rotary_dim"])
    with torch.no_grad():
        got = m(ids).logits
    assert torch.allclose(base, got, atol=1e-5, rtol=1e-5)
