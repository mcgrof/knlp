# SPDX-License-Identifier: GPL-2.0
"""The raw-BF16 streaming control is bit exact and owns returned storage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from research.kv_translate.sharing import cache_store  # noqa: E402


def test_streaming_bf16_reader_is_bit_exact_and_bounded(tmp_path):
    generator = torch.Generator().manual_seed(19)
    shape = (1, 2, 256, 4)
    pairs = []
    for _ in range(4):
        pairs.append(
            (
                torch.randn(shape, generator=generator).to(torch.bfloat16),
                torch.randn(shape, generator=generator).to(torch.bfloat16),
            )
        )
    path = tmp_path / "bf16.kv"
    meta = {
        "model": "unit/model",
        "revision": "frozen-revision",
        "prefix_tokens_sha256": "78" * 32,
    }
    receipt = cache_store.write(str(path), pairs, meta)
    layer_bytes = receipt["payload_bytes"] // len(pairs)
    reader = cache_store.StreamingReader(
        layer_bytes,
        device="cpu",
        staging_buffers=1,
    )
    head, restored, io = reader.read(path, direct=False, expect=meta)

    assert io["integrity_policy"] == "per_read_sha256"
    assert io["staging_bytes"] == layer_bytes
    assert io["staging_bytes"] < head["payload_bytes"]
    for (key, value), (got_key, got_value) in zip(pairs, restored):
        assert torch.equal(got_key.view(torch.int16), key.view(torch.int16))
        assert torch.equal(got_value.view(torch.int16), value.view(torch.int16))
