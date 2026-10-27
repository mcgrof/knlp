# SPDX-License-Identifier: MIT
"""CPU gates for the native-cartridge bridge on the tiny model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.context_cart import artifact as A
from research.context_cart import cas_bridge as B
from research.context_cart import model as M

from tests.context_cart.test_model import (
    FAKE_CHECKPOINT,
    FAKE_SOURCE_REVISION,
    _identity,
    _random_prefix,
)


def _forward_keys(model, ids: list[int], positions: list[int]):
    input_ids = torch.tensor([ids])
    pos = torch.tensor([positions])
    out = model.model(
        input_ids=input_ids,
        position_ids=pos,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
    )
    layers = M.Geometry.from_config(model.config).layers
    return [
        tuple(t.clone() for t in M.cache_layer(out.past_key_values, layer))
        for layer in range(layers)
    ]


def test_native_file_roundtrip_and_export(tmp_path: Path):
    model = M.tiny_base(seed=11)
    identity = _identity(model)
    prefix = [(k.detach(), v.detach()) for k, v in _random_prefix(model, 5)]
    path = tmp_path / "cart.pt"
    B.save_native_cartridge(path, prefix, frozen_rows=1)
    cart = B.load_native_cartridge(path, dtype=torch.float32)
    assert (cart.layers, cart.kv_heads, cart.head_dim) == (2, 2, 8)
    assert cart.frozen_rows == 1 and cart.trainable_rows == 4 and cart.rows == 5
    assert list(cart.positions) == [0, 1, 2, 3, 4] and cart.reader_start == 5
    assert B.prefix_equal(cart.prefix("cpu"), prefix)
    result = A.export_prefix(
        tmp_path / "art",
        cart.prefix("cpu"),
        identity,
        positions=cart.positions,
        reader_start=cart.reader_start,
        source_revision=FAKE_SOURCE_REVISION,
        checkpoint_digest=FAKE_CHECKPOINT,
        namespace="native",
        extra={"native_digest": B.digest_native(cart)},
    )
    loaded = A.load_prefix(tmp_path / "art", identity, "cpu")
    assert loaded.first_position == 5 and loaded.positions == (0, 1, 2, 3, 4)
    assert loaded.tensor_digest == M.prefix_digest(prefix)
    assert result.tensor_bytes == 2 * 2 * 5 * 2 * 8 * 4


def test_sink_rows_are_the_first_position_of_any_sequence():
    model = M.tiny_base(seed=12)
    sink = B.sink_rows(model, token_id=3)
    assert sink[0][0].shape == (1, 2, 1, 8)
    for ids in ([3, 40, 41, 42], [3, 100, 7], [3]):
        rows = _forward_keys(model, ids, list(range(len(ids))))
        for (sk, sv), (k, v) in zip(sink, rows):
            assert torch.allclose(sk, k[:, :, :1], atol=1e-6)
            assert torch.allclose(sv, v[:, :, :1], atol=1e-6)
    # a different first token is a different sink
    other = B.sink_rows(model, token_id=4)
    assert not B.prefix_equal(sink, other)


def test_rerotation_round_trip_and_position_shift_equivalence():
    model = M.tiny_base(seed=13)
    ids = [5, 9, 13, 17, 21, 25, 29, 33]
    base = list(range(7, 15))
    shifted = [p + 30 for p in base]
    keys_base = _forward_keys(model, ids, base)
    keys_shift = _forward_keys(model, ids, shifted)
    for (k0, v0), (k1, v1) in zip(keys_base, keys_shift):
        # values carry no position
        assert torch.allclose(v0, v1, atol=1e-5)
        moved = B.rerotate_keys(model, k0, base, shifted)
        assert torch.allclose(moved, k1, atol=1e-4), (moved - k1).abs().max()
        back = B.rerotate_keys(model, moved, shifted, base)
        assert torch.allclose(back, k0, atol=1e-5)
        # the identity frame is a no-op
        assert torch.allclose(B.rerotate_keys(model, k0, base, base), k0, atol=1e-6)
    with pytest.raises(ValueError):
        B.rerotate_keys(model, keys_base[0][0], base, base[:-1])


def test_generated_rows_move_into_the_native_frame():
    pytest.importorskip("peft")
    torch.manual_seed(0)
    reader_model = M.tiny_base(seed=14)
    writer_base = M.tiny_base(seed=14)
    spec = M.ModelSpec(
        name="tiny", revision="a" * 40, dtype="float32", memory_slots=3, lora_rank=2
    )
    writer = M.Writer(writer_base, spec, init_rows=[5, 6, 7])
    reader = M.Reader(reader_model, eos_ids=(1,), pad_id=0)
    source = torch.tensor([[11, 12, 13, 14, 15, 16]])
    with torch.no_grad():
        generated, length = writer.encode(source)
    generated = [(k.detach(), v.detach()) for k, v in generated]
    assert length == 6 and generated[0][0].shape[2] == 3
    sink = B.sink_rows(reader_model, token_id=3)
    framed = B.generated_to_native_frame(
        reader_model, generated, length, sink, torch.float32
    )
    assert framed[0][0].shape == (1, 2, 4, 8)
    for (fk, fv), (gk, gv), (sk, sv) in zip(framed, generated, sink):
        assert torch.equal(fk[:, :, :1], sk) and torch.equal(fv[:, :, :1], sv)
        assert torch.equal(fv[:, :, 1:], gv)
        expected = B.rerotate_keys(reader_model, gk, [6, 7, 8], [1, 2, 3])
        assert torch.allclose(fk[:, :, 1:], expected, atol=1e-6)
    # the moved rows behave like the originals: relative distances to the
    # reader tokens are unchanged, so the reader's logits agree
    prompt = torch.tensor([[40, 41, 42]])
    mask = torch.ones_like(prompt)
    with torch.no_grad():
        live = reader.greedy(generated, prompt, mask, 9, 3, keep_logits=True)
        content_only = [(k[:, :, 1:], v[:, :, 1:]) for k, v in framed]
        moved = reader.greedy(content_only, prompt, mask, 4, 3, keep_logits=True)
    assert torch.allclose(live["step_logits"], moved["step_logits"], atol=1e-4)
    assert live["ids"].tolist() == moved["ids"].tolist()
    # and the full native-frame prefix installs at reader start rows
    positions, start = B.native_frame_positions(4)
    assert list(positions) == [0, 1, 2, 3] and start == 4
    with torch.no_grad():
        out = reader.greedy(framed, prompt, mask, start, 2)
    assert out["cache_rows"] == 4 + 3 + 1
