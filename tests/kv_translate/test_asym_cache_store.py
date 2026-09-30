# SPDX-License-Identifier: GPL-2.0
"""Contracts for the shape-aware K16/V8 cache store."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from research.kv_translate.sharing import asym_cache_store  # noqa: E402
from research.kv_translate.sharing.run_native_codec_screen import (  # noqa: E402
    choose_qualification_contexts,
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
        largest, device="cpu", codec_checkout=checkout
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
