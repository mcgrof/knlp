# SPDX-License-Identifier: GPL-2.0
"""The accelerated path, held to the reference and to a planted case.

The decisive test here transfers a model's cache to itself. Nothing about the
method is being evaluated in that case: a correct pipeline must pick every
layer for itself, fit a map that is the identity, and reproduce the native
logits. Anything else is a defect in positions, rotation, head layout or cache
installation, and it is far cheaper to find on a four-layer random model than
on a rented card.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from research.kv_translate.published import CACHE_BRIDGE, FULL_HEAD  # noqa: E402
from research.kv_translate.published import attn_repair  # noqa: E402
from research.kv_translate.published import capture as cap  # noqa: E402
from research.kv_translate.published import fit_pair, ridge  # noqa: E402
from research.kv_translate.published.torch_ops import (  # noqa: E402
    Accumulator,
    apply_rope,
    head_dim_of,
    r_squared,
    strip_rope,
)

THETA = 1_000_000.0


def tiny(layers, seed, heads=4, kv_heads=2, head_dim=16, hidden=48):
    """A random model with non-square attention, like the real target."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=257,
        hidden_size=hidden,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=512,
        rope_theta=THETA,
    )
    return Qwen3ForCausalLM(cfg).eval()


def tokens(n, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 257, (n, t), generator=g)


# ---- rotation -----------------------------------------------------------


def test_rotation_round_trips():
    x = torch.randn(3, 40, 16, dtype=torch.float64)
    pos = torch.arange(40)
    assert torch.allclose(
        strip_rope(apply_rope(x, pos, THETA), pos, THETA), x, atol=1e-12
    )


def test_stripping_the_cache_recovers_the_keys_the_model_computed():
    """The convention is checked against the model, not against itself.

    Keys are tapped before rotation and compared with cached keys after the
    rotation is removed. A shifted position must break the match, or the check
    could not detect the error it exists for.
    """
    m = tiny(3, 1)
    geom = cap.geometry(m)
    ids = tokens(1, 24)
    pre = {}
    hooks = []
    for i, attn in enumerate(cap.attention_modules(m)):
        mod = getattr(attn, "k_norm", None) or attn.k_proj
        hooks.append(
            mod.register_forward_hook(
                lambda _m, _i, out, i=i: pre.__setitem__(
                    i, out.reshape(1, 24, geom["kv_heads"], geom["head_dim"]).detach()
                )
            )
        )
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    for h in hooks:
        h.remove()
    pos = torch.arange(24)
    for layer, (k, _v) in enumerate(cap.cache_layers(out.past_key_values)):
        content = strip_rope(k[0].to(torch.float64), pos, geom["rope_theta"])
        want = pre[layer][0].permute(1, 0, 2).to(torch.float64)
        assert torch.allclose(content, want, atol=1e-5), layer
        shifted = strip_rope(k[0].to(torch.float64), pos + 1, geom["rope_theta"])
        assert not torch.allclose(shifted, want, atol=1e-3), layer


def test_head_width_is_read_not_derived():
    m = tiny(2, 0)
    assert m.config.hidden_size // m.config.num_attention_heads == 12
    assert head_dim_of(m.config) == 16


# ---- the solve ----------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
def test_the_batched_solve_matches_the_reference(weighted):
    rng = np.random.default_rng(3)
    n, p, d = 500, 12, 5
    X = rng.normal(size=(n, p)) + 2.0
    Y = X @ rng.normal(size=(p, d)) + 0.1 * rng.normal(size=(n, d)) - 3.0
    w = rng.uniform(0.1, 4.0, size=n) if weighted else None
    Wr, br, _ = ridge.solve(X, Y, w=w, lam=0.01)
    acc = Accumulator(p, d, "cpu")
    for lo in range(0, n, 128):  # arriving in batches must not change the answer
        acc.add(
            torch.from_numpy(X[lo : lo + 128]),
            torch.from_numpy(Y[lo : lo + 128]),
            None if w is None else torch.from_numpy(w[lo : lo + 128]),
        )
    W, b, st = acc.solve(0.01)
    assert np.allclose(W.numpy(), Wr, atol=1e-9)
    assert np.allclose(b.numpy(), br, atol=1e-9)
    assert r_squared(st) == pytest.approx(ridge.held_in_r2(X, Y, Wr, br, w=w), abs=1e-9)


def test_stacking_heads_as_columns_equals_solving_each_head():
    """The shared Gram is a speed change and must not be a numerical one."""
    rng = np.random.default_rng(4)
    n, p, H, d = 300, 10, 3, 4
    X = torch.from_numpy(rng.normal(size=(n, p)))
    Y = torch.from_numpy(rng.normal(size=(n, H, d)))
    together = Accumulator(p, H * d, "cpu")
    together.add(X, Y.reshape(n, H * d))
    Wa, ba, st = together.solve(0.01)
    for h in range(H):
        one = Accumulator(p, d, "cpu")
        one.add(X, Y[:, h])
        Wh, bh, sh = one.solve(0.01)
        assert torch.allclose(Wa.reshape(p, H, d)[:, h], Wh, atol=1e-10)
        assert torch.allclose(ba.reshape(H, d)[h], bh, atol=1e-10)
        assert r_squared(st, slice(h * d, (h + 1) * d)) == pytest.approx(
            r_squared(sh), abs=1e-10
        )


# ---- capture ------------------------------------------------------------


def test_capture_writes_content_keys_in_the_declared_layout(tmp_path):
    m = tiny(3, 2)
    meta = cap.capture(m, tokens(4, 32), str(tmp_path), stride=4, batch_size=2)
    assert meta["rows"] == 4 * 8
    k0 = np.load(tmp_path / "L000.k.npy")
    assert k0.shape == (32, 2, 16) and k0.dtype == np.float16
    assert np.isfinite(k0).all()
    assert not (tmp_path / "sensitivities.npy").exists()


def test_batching_does_not_change_what_is_captured(tmp_path):
    m = tiny(2, 5)
    ids = tokens(4, 16)
    cap.capture(m, ids, str(tmp_path / "a"), stride=4, batch_size=1)
    cap.capture(m, ids, str(tmp_path / "b"), stride=4, batch_size=4)
    for f in ("L001.k.npy", "L001.v.npy"):
        a, b = np.load(tmp_path / "a" / f), np.load(tmp_path / "b" / f)
        assert np.allclose(a.astype(np.float32), b.astype(np.float32), atol=2e-3)


def test_sensitivities_are_nonnegative_and_carry_one_unit_per_boundary(tmp_path):
    m = tiny(2, 7)
    bounds = [6, 12, 20, 31]
    cap.capture(
        m, tokens(2, 32), str(tmp_path), stride=4, batch_size=2, boundaries=bounds
    )
    s = np.load(tmp_path / "sensitivities.npy")
    assert s.shape == (2, 2, 2, 16)
    assert (s >= 0).all() and np.isfinite(s).all()
    per_seq = s.reshape(2, 2, 2, 2, 8).sum(-1)  # [L, comp, head, seq]
    assert np.allclose(per_seq, len(bounds), atol=1e-4)


def test_positions_at_or_after_a_boundary_receive_nothing_from_it(tmp_path):
    m = tiny(2, 8)
    cap.capture(
        m, tokens(1, 32), str(tmp_path), stride=4, batch_size=1, boundaries=[10]
    )
    s = np.load(tmp_path / "sensitivities.npy")  # sample positions 0,4,...,28
    assert (s[..., 3:] == 0).all()  # positions 12 and later are not before 10
    assert (s[..., :3].sum(-1) > 0).all()


# ---- the planted case ---------------------------------------------------


@pytest.fixture(scope="module")
def self_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("selfpair")
    m = tiny(4, 11)
    ids = tokens(24, 48, seed=1)
    bounds, _ = attn_repair.log_spaced_boundaries(6, 47, 8)
    cap.capture(m, ids, str(d / "src"), stride=2, batch_size=8)
    cap.capture(m, ids, str(d / "tgt"), stride=2, batch_size=8, boundaries=bounds)
    return m, d


def test_a_model_selects_each_of_its_own_layers_for_itself(self_pair):
    _m, d = self_pair
    sel, scores = fit_pair.select_layers(str(d / "src"), str(d / "tgt"), k=1)
    assert sel == [[0], [1], [2], [3]]
    assert all(scores[layer, layer] > 0.999 for layer in range(4))


@pytest.mark.parametrize("method_id", [FULL_HEAD, CACHE_BRIDGE])
def test_transferring_a_cache_to_itself_reproduces_native_logits(self_pair, method_id):
    m, d = self_pair
    sel, _ = fit_pair.select_layers(str(d / "src"), str(d / "tgt"), k=1)
    weights = None
    if method_id == CACHE_BRIDGE:
        weights, _info = fit_pair.repair_weights(str(d / "tgt"), feature_width=16)
    fitted = fit_pair.fit(
        method_id, str(d / "src"), str(d / "tgt"), sel, lam=1e-6, weights=weights
    )
    assert fitted["r2"].min() > 0.999
    geom = cap.geometry(m)
    ids = tokens(1, 40, seed=99)
    with torch.no_grad():
        prefix = m(input_ids=ids[:, :-8], use_cache=True).past_key_values
        native = m(input_ids=ids[:, -8:], past_key_values=prefix).logits
        src = cap.cache_layers(m(input_ids=ids[:, :-8], use_cache=True).past_key_values)
        moved = fit_pair.transfer(
            fitted, src, geom["rope_theta"], geom["rope_theta"], dtype=torch.float32
        )
        got = m(input_ids=ids[:, -8:], past_key_values=cap.make_cache(moved)).logits
    assert torch.allclose(got, native, atol=5e-2)
    assert (got.argmax(-1) == native.argmax(-1)).all()


def test_the_two_arms_have_the_published_shape_relationship(self_pair):
    _m, d = self_pair
    sel, _ = fit_pair.select_layers(str(d / "src"), str(d / "tgt"), k=2)
    fh = fit_pair.fit(FULL_HEAD, str(d / "src"), str(d / "tgt"), sel)
    cb = fit_pair.fit(CACHE_BRIDGE, str(d / "src"), str(d / "tgt"), sel)
    assert fh["W"].shape == (4, 2, 2, 2 * 2 * 16, 16)
    assert cb["W"].shape == (4, 2, 2, 2 * 16, 16)
    assert fh["W"].numel() == 2 * cb["W"].numel()  # the ratio is the source head count


def test_the_baseline_refuses_weights(self_pair):
    _m, d = self_pair
    sel, _ = fit_pair.select_layers(str(d / "src"), str(d / "tgt"), k=1)
    w, _ = fit_pair.repair_weights(str(d / "tgt"), feature_width=16)
    with pytest.raises(ValueError, match="unweighted by definition"):
        fit_pair.fit(FULL_HEAD, str(d / "src"), str(d / "tgt"), sel, weights=w)


def test_a_mapper_survives_serialization_exactly(self_pair, tmp_path):
    _m, d = self_pair
    sel, _ = fit_pair.select_layers(str(d / "src"), str(d / "tgt"), k=2)
    fitted = fit_pair.fit(CACHE_BRIDGE, str(d / "src"), str(d / "tgt"), sel)
    path = str(tmp_path / "mapper.safetensors")
    size = fit_pair.save_mapper(path, fitted)
    back = fit_pair.load_mapper(path)
    assert torch.equal(back["W"], fitted["W"]) and torch.equal(back["b"], fitted["b"])
    assert back["selected"] == sel and back["method"] == CACHE_BRIDGE
    payload = (fitted["W"].numel() + fitted["b"].numel()) * 4
    assert payload <= size < payload + 4096  # float32 coefficients plus a header


# ---- refusing the wrong inputs ------------------------------------------


def test_traces_from_different_rows_are_refused(tmp_path):
    m = tiny(2, 12)
    cap.capture(m, tokens(4, 16), str(tmp_path / "a"), stride=4, batch_size=4)
    cap.capture(m, tokens(6, 16), str(tmp_path / "b"), stride=4, batch_size=6)
    with pytest.raises(ValueError, match="not captured from the same rows"):
        fit_pair.select_layers(str(tmp_path / "a"), str(tmp_path / "b"), k=1)


def test_a_pair_with_different_head_counts_is_refused(tmp_path):
    a, b = tiny(2, 13, kv_heads=2), tiny(2, 14, kv_heads=4)
    ids = tokens(4, 16)
    cap.capture(a, ids, str(tmp_path / "a"), stride=4, batch_size=4)
    cap.capture(b, ids, str(tmp_path / "b"), stride=4, batch_size=4)
    with pytest.raises(ValueError, match="will not guess"):
        fit_pair.select_layers(str(tmp_path / "a"), str(tmp_path / "b"), k=1)


def test_an_interrupted_capture_is_not_mistaken_for_a_finished_one(tmp_path):
    """The receipt is written last, so its absence marks an incomplete trace."""
    m = tiny(2, 15)
    cap.capture(m, tokens(4, 16), str(tmp_path / "a"), stride=4, batch_size=4)
    os.remove(tmp_path / "a" / "CAPTURE.json")
    with pytest.raises(FileNotFoundError):
        fit_pair.load_meta(str(tmp_path / "a"))
