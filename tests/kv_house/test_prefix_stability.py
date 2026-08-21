# SPDX-License-Identifier: GPL-2.0
"""Gate-0 sealed-block prefix-cache contract. CPU, no GPU, no model.

These are the properties Prefix Integrity Analysis polices end to
end; here they are enforced at the codec level with synthetic data:
determinism, suffix independence, raw partial tails, and codec
configuration living in the cache key.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from kv_house.prefix_codec import CodecConfig, SealedBlockCodec

TRANSFORMS_UNDER_TEST = (
    "identity",
    "anchor_delta",
    "dc_householder",
    "dct2",
    "haar",
    "pca_oracle",
    "poweriter_householder",
)


def _codec(transform, b=16, bits=8):
    cfg = CodecConfig(block_size=b, transform=transform, quant_alloc=(bits,) * b)
    return SealedBlockCodec(cfg)


def _ctx(t, d=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(t, d, generator=gen)


def test_encode_is_deterministic():
    for name in TRANSFORMS_UNDER_TEST:
        codec = _codec(name)
        block = _ctx(16)
        a1 = codec.encode_block(block, 0)
        a2 = codec.encode_block(block.clone(), 0)
        assert a1.digest == a2.digest, name


def test_sealed_blocks_are_suffix_independent():
    """The same prefix with two different suffixes must produce
    byte-identical artifacts for every shared full block."""
    for name in TRANSFORMS_UNDER_TEST:
        codec = _codec(name)
        prefix = _ctx(4 * 16, seed=1)
        for seed in (2, 3):
            suffix = _ctx(24, seed=seed)
            arts, _ = codec.seal_context(torch.cat([prefix, suffix]))
            base, _ = codec.seal_context(prefix)
            for a, b in zip(base, arts[: len(base)]):
                assert a.digest == b.digest, name


def test_partial_tail_stays_raw():
    codec = _codec("dc_householder")
    ctx = _ctx(16 * 3 + 5)
    arts, tail = codec.seal_context(ctx)
    assert len(arts) == 3
    assert tail.shape[0] == 5
    assert torch.equal(tail, ctx[48:])


def test_config_changes_cache_key():
    keys = set()
    for name in TRANSFORMS_UNDER_TEST:
        for bits in (8, 4):
            cfg = CodecConfig(
                block_size=16, transform=name, quant_alloc=(bits,) * 16
            )
            keys.add(cfg.cache_key())
    assert len(keys) == len(TRANSFORMS_UNDER_TEST) * 2


def test_decode_requires_matching_config():
    a = _codec("dc_householder").encode_block(_ctx(16), 0)
    other = _codec("dct2")
    try:
        other.decode_block(a)
        raise RuntimeError("config mismatch not caught")
    except AssertionError:
        pass


def test_roundtrip_error_is_quantization_only():
    """At 16-bit allocation the sealed round trip must be fp16-close
    for every transform (the transform itself is lossless)."""
    for name in TRANSFORMS_UNDER_TEST:
        codec = _codec(name, bits=16)
        block = _ctx(16, seed=9)
        out = codec.decode_block(codec.encode_block(block, 0))
        rel = (out - block).norm() / block.norm()
        assert rel < 5e-3, (name, float(rel))
