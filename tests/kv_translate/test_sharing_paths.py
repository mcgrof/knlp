# SPDX-License-Identifier: GPL-2.0
"""The three request paths and the cache file, on a model small enough to check.

In float32 a receiver that restores its own cache must reach the token it
reaches by recomputing, and a cache translated from a model to itself must
reach it too. Anything else is a defect in positions, in the boundary between
prefix and suffix, or in how a cache is stored and installed.
"""

from __future__ import annotations

import errno
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from research.kv_translate.published import CACHE_BRIDGE, FULL_HEAD  # noqa: E402
from research.kv_translate.published import attn_repair  # noqa: E402
from research.kv_translate.published import capture as cap  # noqa: E402
from research.kv_translate.published import fit_pair  # noqa: E402
from research.kv_translate.sharing import cache_store, contexts  # noqa: E402
from research.kv_translate.sharing.request_path import Receiver  # noqa: E402

THETA = 1_000_000.0


def tiny(layers, seed):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=257,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=THETA,
    )
    return Qwen3ForCausalLM(cfg).eval()


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 257, (n,), generator=g).tolist()


def meta(prefix, name="tiny", rev="r"):
    return {
        "model": name,
        "revision": rev,
        "prefix_tokens_sha256": contexts.digest(prefix),
    }


def want(prefix, name="tiny", rev="r"):
    return {**meta(prefix, name, rev), "tokens": len(prefix)}


@pytest.fixture(scope="module")
def model():
    return tiny(4, 11)


@pytest.fixture(scope="module")
def mappers(model, tmp_path_factory):
    d = tmp_path_factory.mktemp("self")
    ids = torch.stack([torch.tensor(tokens(48, 100 + i)) for i in range(24)])
    bounds, _ = attn_repair.log_spaced_boundaries(6, 47, 8)
    cap.capture(model, ids, str(d / "src"), stride=2, batch_size=8)
    cap.capture(model, ids, str(d / "tgt"), stride=2, batch_size=8, boundaries=bounds)
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
    return out


def receiver(model, tokens_in_prefix, mapper=None):
    size = cache_store.payload_bytes(4, 2, tokens_in_prefix, 16, itemsize=4)
    return Receiver(model, largest_payload=size, mapper=mapper, source_theta=THETA)


def stored(rx, prefix, path):
    return cache_store.write(str(path), rx.prefix_cache(prefix), meta(prefix))


def test_a_cache_file_holds_exactly_the_cache(model, tmp_path):
    prefix = tokens(40, 1)
    rx = receiver(model, 40)
    pairs = rx.prefix_cache(prefix)
    rec = cache_store.write(str(tmp_path / "a.kv"), pairs, meta(prefix))
    assert rec["payload_bytes"] == 4 * 2 * 2 * 40 * 16 * 4
    assert rec["logical_bytes"] == cache_store.HEADER + rec["payload_bytes"]
    head, io = rx.reader.read(str(tmp_path / "a.kv"), direct=False, expect=want(prefix))
    assert rx.reader.verify(head)
    assert io["bytes"] == rec["payload_bytes"]
    back = cache_store.as_pairs(rx.reader.tensor(head))
    for (k, v), (k2, v2) in zip(pairs, back):
        assert torch.equal(k, k2) and torch.equal(v, v2)
    assert head["tokens"] == 40 and head["dtype"] == "float32"


def test_a_cache_for_another_prefix_or_model_is_refused(model, tmp_path):
    prefix, other = tokens(40, 1), tokens(40, 2)
    rx = receiver(model, 40)
    stored(rx, prefix, tmp_path / "a.kv")
    with pytest.raises(cache_store.WrongCache, match="prefix_tokens_sha256"):
        rx.reader.read(str(tmp_path / "a.kv"), direct=False, expect=want(other))
    with pytest.raises(cache_store.WrongCache, match="model"):
        rx.reader.read(
            str(tmp_path / "a.kv"), direct=False, expect=want(prefix, "bigger")
        )
    (tmp_path / "junk.kv").write_bytes(b"not a cache" * 500)
    with pytest.raises(cache_store.WrongCache, match="not a prefix cache"):
        cache_store.header(str(tmp_path / "junk.kv"))


def test_reading_around_the_page_cache_gives_the_same_bytes(model):
    # direct reads need a real file system; a memory-backed one refuses them
    base = os.environ.get("KNLP_DIRECT_IO_DIR", os.path.expanduser("~"))
    path = os.path.join(base, f".kv-direct-test-{os.getpid()}.kv")
    prefix = tokens(64, 3)
    rx = receiver(model, 64)
    try:
        try:
            stored(rx, prefix, path)
            head, io = rx.reader.read(path, direct=True, expect=want(prefix))
        except OSError as e:
            if e.errno in (errno.EINVAL, errno.EACCES, errno.EROFS, errno.EPERM):
                pytest.skip("this file system does not take direct reads")
            raise
        assert rx.reader.verify(head) and io["direct"] is True
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_restoring_reaches_the_token_recomputing_reaches(model, tmp_path):
    rx = receiver(model, 60)
    for seed in range(5):
        prefix, suffix = tokens(60, 10 + seed), tokens(7, 50 + seed)
        stored(rx, prefix, tmp_path / f"{seed}.kv")
        a = rx.recompute(prefix, suffix)
        b = rx.restore(
            str(tmp_path / f"{seed}.kv"), suffix, direct=False, expect=want(prefix)
        )
        assert a["token"] == b["token"]
        assert torch.allclose(a["logits"], b["logits"], atol=1e-4)
        assert b["io"]["bytes"] > 0
        for stage in ("read", "to_card", "install", "suffix", "first_token", "total"):
            assert f"seconds_{stage}" in b
        assert "seconds_prefill" in a and "seconds_read" not in a


def test_the_first_token_matches_one_pass_over_the_whole_prompt(model):
    rx = receiver(model, 60)
    prefix, suffix = tokens(60, 21), tokens(9, 22)
    with torch.no_grad():
        full = model(input_ids=torch.tensor([prefix + suffix])).logits[0, -1]
    a = rx.recompute(prefix, suffix)
    assert a["token"] == int(full.argmax())
    assert torch.allclose(a["logits"], full.float(), atol=1e-4)


def test_a_boundary_off_by_one_is_caught(model, tmp_path):
    """The check has to be able to fail: a cache one token short must differ."""
    rx = receiver(model, 60)
    prefix, suffix = tokens(60, 31), tokens(7, 32)
    a = rx.recompute(prefix, suffix)
    short = rx.recompute(prefix[:-1], suffix)
    assert not torch.allclose(a["logits"], short["logits"], atol=1e-4)


@pytest.mark.parametrize("mid", [FULL_HEAD, CACHE_BRIDGE])
def test_a_cache_translated_to_itself_reaches_the_same_token(
    model, mappers, mid, tmp_path
):
    rx = receiver(model, 60, mapper=mappers[mid])
    agree = 0
    for seed in range(5):
        prefix, suffix = tokens(60, 40 + seed), tokens(7, 60 + seed)
        stored(rx, prefix, tmp_path / f"{seed}.kv")
        a = rx.recompute(prefix, suffix)
        c = rx.translate(
            str(tmp_path / f"{seed}.kv"), suffix, direct=False, expect=want(prefix)
        )
        assert torch.allclose(a["logits"], c["logits"], atol=5e-2)
        agree += a["token"] == c["token"]
        assert "seconds_map" in c
    assert agree == 5


def test_translation_without_a_mapper_is_refused(model, tmp_path):
    rx = receiver(model, 60)
    prefix = tokens(60, 5)
    stored(rx, prefix, tmp_path / "a.kv")
    with pytest.raises(ValueError, match="needs a fitted mapper"):
        rx.translate(
            str(tmp_path / "a.kv"), tokens(4, 6), direct=False, expect=want(prefix)
        )


def test_generation_continues_greedily_and_stops_where_told(model):
    rx = receiver(model, 60)
    prefix, suffix = tokens(60, 71), tokens(5, 72)
    ids = prefix + suffix
    ref = []
    with torch.no_grad():
        for _ in range(12):
            nxt = int(model(input_ids=torch.tensor([ids + ref])).logits[0, -1].argmax())
            ref.append(nxt)
    a = rx.recompute(prefix, suffix)
    g = rx.generate(a["token"], a["cache"], max_new_tokens=12, stop_ids=set())
    assert g["tokens"] == ref and g["stopped"] is False and g["new_tokens"] == 12
    a = rx.recompute(prefix, suffix)
    h = rx.generate(a["token"], a["cache"], max_new_tokens=12, stop_ids={ref[4]})
    cut = ref.index(ref[4])
    assert h["tokens"] == ref[:cut] and h["stopped"] is True


def test_the_closed_book_route_sees_no_prefix(model):
    rx = receiver(model, 60)
    head, suffix = tokens(6, 81), tokens(5, 82)
    with torch.no_grad():
        full = model(input_ids=torch.tensor([head + suffix])).logits[0, -1]
    got = rx.closed_book(head + suffix)
    assert got["token"] == int(full.argmax())
