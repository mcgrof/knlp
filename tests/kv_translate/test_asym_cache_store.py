# SPDX-License-Identifier: GPL-2.0
"""Contracts for the shape-aware K16/V8 cache store."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from research.kv_translate.sharing import asym_cache_store  # noqa: E402
from research.kv_translate.sharing.run_native_codec_screen import (  # noqa: E402
    choose_qualification_contexts,
    choose_screen_contexts,
)


def test_alignment_rounds_only_up():
    assert asym_cache_store._align(1) == 4096
    assert asym_cache_store._align(4096) == 4096
    assert asym_cache_store._align(4097) == 8192


def test_qualification_uses_one_group_per_domain_and_length():
    rows = []
    for length in (4096, 16384, 32768):
        for domain in ("document", "code"):
            for group in ("g2", "g1"):
                rows.append(
                    {
                        "domain": domain,
                        "length_bucket": length,
                        "source_group_id": f"{domain}-{length}-{group}",
                        "context_id": f"{domain}-{length}-{group}",
                    }
                )
    selected = choose_qualification_contexts(rows, [4096, 16384, 32768])
    assert [(r["length_bucket"], r["domain"]) for r in selected] == [
        (4096, "document"),
        (4096, "code"),
        (16384, "document"),
        (16384, "code"),
        (32768, "document"),
        (32768, "code"),
    ]


def test_development_selection_uses_frozen_count_at_each_length():
    rows = []
    for length, count in ((4096, 2), (16384, 3), (32768, 2)):
        for domain in ("document", "code"):
            for group in range(count + 1):
                rows.append(
                    {
                        "domain": domain,
                        "length_bucket": length,
                        "source_group_id": f"{domain}-{length}-{group}",
                        "context_id": f"{domain}-{length}-{group}",
                    }
                )
    selected = choose_screen_contexts(
        rows, [4096, 16384, 32768], {"4096": 2, "16384": 3, "32768": 2}
    )
    assert len(selected) == 14
    assert [(row["length_bucket"], row["domain"]) for row in selected] == [
        (4096, "document"),
        (4096, "document"),
        (4096, "code"),
        (4096, "code"),
        (16384, "document"),
        (16384, "document"),
        (16384, "document"),
        (16384, "code"),
        (16384, "code"),
        (16384, "code"),
        (32768, "document"),
        (32768, "document"),
        (32768, "code"),
        (32768, "code"),
    ]


def test_pinned_codec_round_trip_preserves_key_bits_and_reference_values(tmp_path):
    checkout = os.environ.get("LMCACHE_CODEC_CHECKOUT")
    if not checkout:
        pytest.skip("set LMCACHE_CODEC_CHECKOUT for pinned codec qualification")
    api = asym_cache_store._codec_api(checkout)
    AsymK16V8Codec, _, ScaleScope = api
    codec = AsymK16V8Codec(scale_scope=ScaleScope.PER_TENSOR)
    generator = torch.Generator().manual_seed(7)
    shape = (1, 2, 4099, 8)
    pairs = []
    for layer in range(2):
        key = torch.randn(shape, generator=generator).to(torch.bfloat16)
        value = torch.randn(shape, generator=generator).to(torch.bfloat16)
        if layer == 0:
            value.zero_()
        pairs.append((key, value))
    path = tmp_path / "native.kv"
    meta = {
        "model": "unit/model",
        "revision": "frozen-revision",
        "prefix_tokens_sha256": "ab" * 32,
        "role": "native",
        "attention_backend": "sdpa",
    }
    receipt = asym_cache_store.write(
        path,
        pairs,
        meta,
        codec_checkout=checkout,
        codec_revision="b1339f7606f7241402312e988c3e730d8f66a52b",
    )
    head = asym_cache_store.header(path)
    assert head["outer_header_bytes"] == 8 * 4096
    largest = max(segment["padded_length"] for segment in head["segments"])
    reader = asym_cache_store.Reader(
        largest,
        device="cpu",
        codec_checkout=checkout,
        staging_buffers=1,
    )
    _, restored, io = reader.read(path, direct=False, expect=meta)
    # This deliberately partial, small tensor pays proportionally large
    # block-padding overhead.  The codec payload itself retains the expected
    # quarter reduction; the full-size screen enforces 24% on actual files.
    assert 1 - head["asym_payload_bytes"] / head["bf16_payload_bytes"] >= 0.249
    assert receipt["serialized_reduction"] > 0.15
    assert io["h2d_bytes"] == head["asym_payload_bytes"]
    for (key, value), (got_key, got_value) in zip(pairs, restored):
        assert torch.equal(got_key.view(torch.int16), key.view(torch.int16))
        encoded = codec.encode(key, value)
        _, reference, _ = codec.decode(encoded, out_v_dtype=torch.bfloat16)
        assert torch.equal(got_value, reference.reshape(value.shape))
    assert torch.count_nonzero(restored[0][1]) == 0


def test_prevalidated_policy_attests_immutable_file(tmp_path):
    checkout = os.environ.get("LMCACHE_CODEC_CHECKOUT")
    if not checkout:
        pytest.skip("set LMCACHE_CODEC_CHECKOUT for pinned codec qualification")
    key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).to(torch.bfloat16)
    value = (key.float() / 7).to(torch.bfloat16)
    path = tmp_path / "attested.kv"
    meta = {
        "model": "unit/model",
        "revision": "frozen-revision",
        "prefix_tokens_sha256": "ef" * 32,
        "role": "native",
        "attention_backend": "sdpa",
    }
    asym_cache_store.write(
        path,
        [(key, value)],
        meta,
        codec_checkout=checkout,
        codec_revision="b1339f7606f7241402312e988c3e730d8f66a52b",
    )
    head = asym_cache_store.header(path)
    reader = asym_cache_store.Reader(
        head["segments"][0]["padded_length"],
        device="cpu",
        codec_checkout=checkout,
        integrity_policy="prevalidated",
    )
    with pytest.raises(asym_cache_store.WrongCache, match="not been"):
        reader.read(path, direct=False, expect=meta)
    reader.validate(path, expect=meta)
    _, restored, io = reader.read(path, direct=False, expect=meta)
    assert torch.equal(restored[0][0].view(torch.int16), key.view(torch.int16))
    assert io["payload_crc"] == "prevalidated_immutable"

    segment = head["segments"][0]
    with open(path, "r+b") as output:
        output.seek(segment["offset"] + segment["length"] - 1)
        byte = output.read(1)
        output.seek(-1, os.SEEK_CUR)
        output.write(bytes([byte[0] ^ 1]))
        output.flush()
        os.fsync(output.fileno())
    with pytest.raises(asym_cache_store.WrongCache, match="changed"):
        reader.read(path, direct=False, expect=meta)


def test_validation_rejects_path_replacement_race(tmp_path):
    checkout = os.environ.get("LMCACHE_CODEC_CHECKOUT")
    if not checkout:
        pytest.skip("set LMCACHE_CODEC_CHECKOUT for pinned codec qualification")
    key = torch.zeros((1, 1, 8, 4), dtype=torch.bfloat16)
    meta = {
        "model": "unit/model",
        "revision": "frozen-revision",
        "prefix_tokens_sha256": "34" * 32,
    }
    path = tmp_path / "attested.kv"
    replacement = tmp_path / "replacement.kv"
    for target, value in ((path, torch.ones_like(key)), (replacement, key.clone())):
        asym_cache_store.write(
            target,
            [(key, value)],
            meta,
            codec_checkout=checkout,
            codec_revision="b1339f7606f7241402312e988c3e730d8f66a52b",
        )
    head = asym_cache_store.header(path)
    reader = asym_cache_store.Reader(
        head["segments"][0]["padded_length"],
        device="cpu",
        codec_checkout=checkout,
        integrity_policy="prevalidated",
    )
    original_check = reader._check_encoded
    replaced = False

    def replace_after_check(encoded, parsed_head, layer):
        nonlocal replaced
        original_check(encoded, parsed_head, layer)
        if not replaced:
            os.replace(replacement, path)
            replaced = True

    reader._check_encoded = replace_after_check
    with pytest.raises(asym_cache_store.WrongCache, match="changed during"):
        reader.validate(path, expect=meta)


def test_restore_rejects_path_replacement_race(tmp_path):
    checkout = os.environ.get("LMCACHE_CODEC_CHECKOUT")
    if not checkout:
        pytest.skip("set LMCACHE_CODEC_CHECKOUT for pinned codec qualification")
    key = torch.zeros((1, 1, 8, 4), dtype=torch.bfloat16)
    meta = {
        "model": "unit/model",
        "revision": "frozen-revision",
        "prefix_tokens_sha256": "56" * 32,
    }
    path = tmp_path / "attested.kv"
    replacement = tmp_path / "replacement.kv"
    for target, value in ((path, torch.ones_like(key)), (replacement, key.clone())):
        asym_cache_store.write(
            target,
            [(key, value)],
            meta,
            codec_checkout=checkout,
            codec_revision="b1339f7606f7241402312e988c3e730d8f66a52b",
        )
    head = asym_cache_store.header(path)
    reader = asym_cache_store.Reader(
        head["segments"][0]["padded_length"],
        device="cpu",
        codec_checkout=checkout,
        integrity_policy="prevalidated",
    )
    reader.validate(path, expect=meta)
    original_read = reader._read_segment
    replaced = False

    def replace_after_read(fd, segment, index):
        nonlocal replaced
        source = original_read(fd, segment, index)
        if not replaced:
            os.replace(replacement, path)
            replaced = True
        return source

    reader._read_segment = replace_after_read
    with pytest.raises(asym_cache_store.WrongCache, match="changed during"):
        reader.read(path, direct=False, expect=meta)


def test_per_read_crc_rejects_payload_corruption(tmp_path):
    checkout = os.environ.get("LMCACHE_CODEC_CHECKOUT")
    if not checkout:
        pytest.skip("set LMCACHE_CODEC_CHECKOUT for pinned codec qualification")
    key = torch.zeros((1, 1, 8, 4), dtype=torch.bfloat16)
    value = torch.ones_like(key)
    path = tmp_path / "corrupt.kv"
    meta = {
        "model": "unit/model",
        "revision": "frozen-revision",
        "prefix_tokens_sha256": "12" * 32,
    }
    asym_cache_store.write(
        path,
        [(key, value)],
        meta,
        codec_checkout=checkout,
        codec_revision="b1339f7606f7241402312e988c3e730d8f66a52b",
    )
    head = asym_cache_store.header(path)
    segment = head["segments"][0]
    with open(path, "r+b") as output:
        output.seek(segment["offset"] + segment["length"] - 1)
        byte = output.read(1)
        output.seek(-1, os.SEEK_CUR)
        output.write(bytes([byte[0] ^ 1]))
        output.flush()
        os.fsync(output.fileno())
    reader = asym_cache_store.Reader(
        segment["padded_length"],
        device="cpu",
        codec_checkout=checkout,
    )
    with pytest.raises(asym_cache_store.WrongCache, match="CRC mismatch"):
        reader.read(path, direct=False, expect=meta)


def test_nonfinite_values_are_rejected(tmp_path):
    checkout = os.environ.get("LMCACHE_CODEC_CHECKOUT")
    if not checkout:
        pytest.skip("set LMCACHE_CODEC_CHECKOUT for pinned codec qualification")
    key = torch.zeros((1, 1, 3, 4), dtype=torch.bfloat16)
    value = key.clone()
    value[..., 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        asym_cache_store.write(
            tmp_path / "bad.kv",
            [(key, value)],
            {
                "model": "unit/model",
                "revision": "frozen",
                "prefix_tokens_sha256": "cd" * 32,
            },
            codec_checkout=checkout,
            codec_revision="b1339f7606f7241402312e988c3e730d8f66a52b",
        )
