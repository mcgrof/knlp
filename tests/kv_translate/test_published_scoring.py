# SPDX-License-Identifier: GPL-2.0
"""The scorer, held to cases with known answers.

Scoring from a prefix cache is an optimisation of scoring in one pass, so the
two must agree, and a model's cache transferred to itself must score as its
own cache does.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from research.kv_translate.published import CACHE_BRIDGE, FULL_HEAD  # noqa: E402
from research.kv_translate.published import attn_repair  # noqa: E402
from research.kv_translate.published import capture as cap  # noqa: E402
from research.kv_translate.published import fit_pair  # noqa: E402
from research.kv_translate.published.scoring import PrefixScorer  # noqa: E402

THETA = 1_000_000.0


def tiny(layers, seed, heads=4, kv_heads=2, head_dim=16, hidden=48):
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


def case(seed, ctx_len=30, lengths=(1, 3, 7, 4)):
    ctx = tokens(1, ctx_len, seed=seed)[0].tolist()
    conts = [tokens(1, n, seed=seed + 1 + i)[0].tolist() for i, n in enumerate(lengths)]
    return ctx, conts


def reference(model, ctx, cont):
    """One continuation, one pass, no batching and no cache."""
    ids = torch.tensor([ctx + cont[:-1]])
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids).logits.float(), -1)[0]
    rows = logp[len(ctx) - 1 : len(ctx) - 1 + len(cont)]
    tgt = torch.tensor(cont)
    return (
        rows.gather(1, tgt[:, None]).sum().item(),
        bool((rows.argmax(-1) == tgt).all()),
    )


# ---- the scorer ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["direct", "native"])
def test_batched_scores_match_one_pass_per_continuation(mode):
    m = tiny(3, 5)
    ctx, conts = case(40)
    got = PrefixScorer(m, mode).score(ctx, conts)
    for (ll, greedy), c in zip(got, conts):
        want_ll, want_greedy = reference(m, ctx, c)
        assert ll == pytest.approx(want_ll, abs=1e-4)
        assert greedy == want_greedy


def test_a_longer_neighbour_does_not_change_a_score():
    m = tiny(3, 5)
    ctx, conts = case(41, lengths=(2, 9))
    alone = PrefixScorer(m, "native").score(ctx, conts[:1])[0][0]
    beside = PrefixScorer(m, "native").score(ctx, conts)[0][0]
    assert alone == pytest.approx(beside, abs=1e-5)


def test_the_greedy_flag_is_true_only_for_the_models_own_choice():
    m = tiny(3, 5)
    ctx, _ = case(42)
    own = []
    with torch.no_grad():
        ids = list(ctx)
        for _ in range(4):
            nxt = int(m(input_ids=torch.tensor([ids])).logits[0, -1].argmax())
            own.append(nxt)
            ids.append(nxt)
    other = list(own)
    other[2] = (other[2] + 1) % 257
    got = PrefixScorer(m, "native").score(ctx, [own, other])
    assert got[0][1] is True
    assert got[1][1] is False
    assert got[0][0] > got[1][0]


def test_a_single_token_context_is_scored_without_a_prefix():
    m = tiny(3, 5)
    ctx, conts = case(43, ctx_len=1)
    s = PrefixScorer(m, "native")
    got = s.score(ctx, conts)
    assert s.prefixes == 0
    for (ll, _g), c in zip(got, conts):
        assert ll == pytest.approx(reference(m, ctx, c)[0], abs=1e-4)


def test_the_prefix_work_is_counted():
    m = tiny(3, 5)
    s = PrefixScorer(m, "native")
    for seed, n in ((50, 20), (60, 35)):
        ctx, conts = case(seed, ctx_len=n)
        s.score(ctx, conts)
    assert s.prefixes == 2
    assert s.prefix_tokens == 19 + 34
    d = PrefixScorer(m, "direct")
    d.score(*case(50))
    assert d.prefixes == 0


@pytest.fixture(scope="module")
def fitted_self(tmp_path_factory):
    d = tmp_path_factory.mktemp("scoreself")
    m = tiny(4, 11)
    ids = tokens(24, 48, seed=1)
    bounds, _ = attn_repair.log_spaced_boundaries(6, 47, 8)
    cap.capture(m, ids, str(d / "src"), stride=2, batch_size=8)
    cap.capture(m, ids, str(d / "tgt"), stride=2, batch_size=8, boundaries=bounds)
    sel, _ = fit_pair.select_layers(str(d / "src"), str(d / "tgt"), k=1)
    out = {}
    for mid in (FULL_HEAD, CACHE_BRIDGE):
        w = None
        if mid == CACHE_BRIDGE:
            w, _ = fit_pair.repair_weights(str(d / "tgt"), feature_width=16)
        f = fit_pair.fit(mid, str(d / "src"), str(d / "tgt"), sel, lam=1e-6, weights=w)
        path = str(d / f"{mid}.safetensors")
        fit_pair.save_mapper(path, f)
        out[mid] = fit_pair.load_mapper(path)
    return m, out


@pytest.mark.parametrize(
    "mode,mid", [("full_head", FULL_HEAD), ("cache_bridge", CACHE_BRIDGE)]
)
def test_a_cache_transferred_to_itself_scores_as_the_native_cache(
    fitted_self, mode, mid
):
    m, mappers = fitted_self
    ctx, conts = case(70, ctx_len=40, lengths=(3, 5, 2, 6))
    native = PrefixScorer(m, "native").score(ctx, conts)
    moved = PrefixScorer(m, mode, source=m, mapper=mappers[mid]).score(ctx, conts)
    for (a, _), (b, _) in zip(native, moved):
        assert b == pytest.approx(a, abs=5e-2)
    assert int(np.argmax([x[0] for x in moved])) == int(
        np.argmax([x[0] for x in native])
    )


def test_a_mapper_fitted_for_the_other_arm_is_refused(fitted_self):
    m, mappers = fitted_self
    with pytest.raises(ValueError, match="fitted as"):
        PrefixScorer(m, "full_head", source=m, mapper=mappers[CACHE_BRIDGE])
    with pytest.raises(ValueError, match="fitted as"):
        PrefixScorer(m, "cache_bridge", source=m, mapper=mappers[FULL_HEAD])


def test_a_transfer_mode_without_its_inputs_is_refused(fitted_self):
    m, mappers = fitted_self
    with pytest.raises(ValueError, match="needs a source"):
        PrefixScorer(m, "full_head", mapper=mappers[FULL_HEAD])
    with pytest.raises(ValueError, match="needs a source"):
        PrefixScorer(m, "cache_bridge", source=m)
    with pytest.raises(ValueError, match="unknown mode"):
        PrefixScorer(m, "translated")


def test_an_empty_context_is_refused():
    with pytest.raises(ValueError, match="empty context"):
        PrefixScorer(tiny(2, 3), "native").score([], [[5, 6]])


def test_nonfinite_logits_stop_the_run_rather_than_score_low():
    m = tiny(2, 3)
    with torch.no_grad():
        m.lm_head.weight[7, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        PrefixScorer(m, "native").score(*case(80))


# ---- the harness adapter ------------------------------------------------


class FakeTok:
    eos_token_id = 0

    def encode(self, s, add_special_tokens=False):
        return [1 + (ord(c) % 250) for c in s]


def test_requests_come_back_in_the_order_they_arrived_and_prefixes_are_shared():
    pytest.importorskip("lm_eval")
    from research.kv_translate.published.scoring import make_lm

    m = tiny(3, 5)
    s = PrefixScorer(m, "native")
    lm = make_lm(s, FakeTok())
    a, ac = case(90, ctx_len=20, lengths=(2, 3))
    b, bc = case(95, ctx_len=25, lengths=(4, 1))
    # interleaved, so grouping has to restore the order
    reqs = [
        (("a", "0"), a, ac[0]),
        (("b", "0"), b, bc[0]),
        (("a", "1"), a, ac[1]),
        (("b", "1"), b, bc[1]),
    ]
    got = lm._loglikelihood_tokens(reqs)
    want = [
        reference(m, c, k) for c, k in ((a, ac[0]), (b, bc[0]), (a, ac[1]), (b, bc[1]))
    ]
    for (ll, g), (wl, wg) in zip(got, want):
        assert ll == pytest.approx(wl, abs=1e-4)
        assert g == wg
    assert s.prefixes == 2
