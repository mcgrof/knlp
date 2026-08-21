# SPDX-License-Identifier: GPL-2.0
"""Gate-0 block random access. CPU, no GPU, no model.

Decoding block j must need nothing but block j's artifact: no delta
chain, no neighbor blocks, no earlier context.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from kv_house.prefix_codec import CodecConfig, SealedBlockCodec


def test_single_block_decode_matches_full_decode():
    cfg = CodecConfig(
        block_size=16, transform="poweriter_householder", quant_alloc=(8,) * 16
    )
    codec = SealedBlockCodec(cfg)
    gen = torch.Generator().manual_seed(5)
    ctx = torch.randn(16 * 6, 64, generator=gen)
    arts, _ = codec.seal_context(ctx)

    full = [codec.decode_block(a) for a in arts]
    for j in (0, 3, 5):
        fresh = SealedBlockCodec(cfg)
        alone = fresh.decode_block(arts[j])
        assert torch.equal(alone, full[j])


def test_subset_decode_order_independent():
    cfg = CodecConfig(block_size=8, transform="dct2", quant_alloc=(8,) * 8)
    codec = SealedBlockCodec(cfg)
    gen = torch.Generator().manual_seed(6)
    ctx = torch.randn(8 * 5, 32, generator=gen)
    arts, _ = codec.seal_context(ctx)
    forward_order = {j: codec.decode_block(arts[j]) for j in range(5)}
    for j in reversed(range(5)):
        assert torch.equal(codec.decode_block(arts[j]), forward_order[j])
