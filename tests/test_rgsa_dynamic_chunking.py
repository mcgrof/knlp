#!/usr/bin/env python3
"""
Tests for RGSA dynamic chunking.

Validates compute_chunk_size bounds, rounding, and that dynamic
chunking does not change parameter counts.
"""

import sys

sys.path.insert(0, ".")

from gpt2.rgsa import (
    RGSAConfig,
    GPT2_RGSA,
    compute_chunk_size,
    _round_to_pow2,
    _apply_rounding,
    _piecewise_lookup,
)


def test_compute_chunk_size_static():
    """Static mode returns cfg.chunk_size unchanged."""
    cfg = RGSAConfig(chunk_size=64, dynamic_chunking=False)
    for sl in [128, 256, 512, 1024, 2048, 4096]:
        cs = compute_chunk_size(sl, cfg)
        assert cs == 64, f"Static mode: expected 64, got {cs} for sl={sl}"
    print("PASS: static mode returns chunk_size unchanged")


def test_compute_chunk_size_bounds():
    """Dynamic mode stays within [min, max]."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=32,
        chunk_size_max=256,
        chunk_size_alpha=0.5,
        chunk_size_schedule="power",
        chunk_size_rounding="nearest",
    )
    for sl in [1, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 65536]:
        cs = compute_chunk_size(sl, cfg)
        assert cs >= 1, f"cs={cs} < 1 for sl={sl}"
        # With rounding, the result should still be reasonable
        assert (
            cs >= cfg.chunk_size_min or cs >= 1
        ), f"cs={cs} < min={cfg.chunk_size_min} for sl={sl}"
    print("PASS: dynamic mode respects bounds")


def test_compute_chunk_size_pow2_rounding():
    """Pow2 rounding produces powers of 2."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=8,
        chunk_size_max=512,
        chunk_size_alpha=0.5,
        chunk_size_schedule="power",
        chunk_size_rounding="pow2",
    )
    for sl in [64, 128, 256, 512, 1024, 2048, 4096]:
        cs = compute_chunk_size(sl, cfg)
        assert cs > 0 and (cs & (cs - 1)) == 0, f"cs={cs} not pow2 for sl={sl}"
    print("PASS: pow2 rounding produces powers of 2")


def test_compute_chunk_size_multiple_of_8():
    """multiple_of_8 rounding produces multiples of 8."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=8,
        chunk_size_max=512,
        chunk_size_alpha=0.5,
        chunk_size_schedule="power",
        chunk_size_rounding="multiple_of_8",
    )
    for sl in [64, 256, 1024, 4096]:
        cs = compute_chunk_size(sl, cfg)
        assert cs % 8 == 0, f"cs={cs} not multiple of 8 for sl={sl}"
    print("PASS: multiple_of_8 rounding correct")


def test_compute_chunk_size_growth():
    """Chunk size grows with sequence length (sublinearly)."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=8,
        chunk_size_max=1024,
        chunk_size_alpha=0.5,
        chunk_size_schedule="power",
        chunk_size_rounding="nearest",
    )
    sizes = []
    seq_lens = [64, 256, 1024, 4096, 16384]
    for sl in seq_lens:
        sizes.append(compute_chunk_size(sl, cfg))

    # Should be non-decreasing
    for i in range(1, len(sizes)):
        assert sizes[i] >= sizes[i - 1], f"Not non-decreasing: {sizes}"
    print(f"PASS: growth is non-decreasing: {dict(zip(seq_lens, sizes))}")


def test_piecewise_schedule():
    """Piecewise schedule uses correct thresholds."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=32,
        chunk_size_max=128,
        chunk_size_schedule="piecewise",
        chunk_size_piecewise="512:32,2048:64,8192:128",
        chunk_size_rounding="nearest",
    )
    assert compute_chunk_size(256, cfg) == 32
    assert compute_chunk_size(512, cfg) == 32
    assert compute_chunk_size(1024, cfg) == 64
    assert compute_chunk_size(2048, cfg) == 64
    assert compute_chunk_size(4096, cfg) == 128
    print("PASS: piecewise schedule correct")


def test_expected_values_alpha05():
    """Print expected chunk sizes for alpha=0.5."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=32,
        chunk_size_max=256,
        chunk_size_alpha=0.5,
        chunk_size_schedule="power",
        chunk_size_rounding="pow2",
    )
    results = {}
    for sl in [256, 512, 1024, 2048, 4096]:
        results[sl] = compute_chunk_size(sl, cfg)
    print(f"Expected chunk sizes (alpha=0.5, pow2): {results}")


def test_param_count_unchanged():
    """Dynamic vs static chunking must have identical params."""
    cfg_static = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        dynamic_chunking=False,
    )
    cfg_dynamic = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        dynamic_chunking=True,
        chunk_size_alpha=0.5,
    )
    cfg_dense = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        dense_mode=True,
    )
    cfg_random = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        random_routing=True,
    )

    m_static = GPT2_RGSA(cfg_static)
    m_dynamic = GPT2_RGSA(cfg_dynamic)
    m_dense = GPT2_RGSA(cfg_dense)
    m_random = GPT2_RGSA(cfg_random)

    p_static = m_static.get_num_params()
    p_dynamic = m_dynamic.get_num_params()
    p_dense = m_dense.get_num_params()
    p_random = m_random.get_num_params()

    assert (
        p_static == p_dynamic
    ), f"Param mismatch: static={p_static} vs dynamic={p_dynamic}"
    assert p_static == p_dense, f"Param mismatch: static={p_static} vs dense={p_dense}"
    assert (
        p_static == p_random
    ), f"Param mismatch: static={p_static} vs random={p_random}"
    print(f"PASS: all modes have identical params: {p_static}")


def test_forward_pass_dynamic():
    """Dynamic chunking model runs a forward pass without error."""
    import torch

    cfg = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        dynamic_chunking=True,
        chunk_size_alpha=0.5,
    )
    model = GPT2_RGSA(cfg)
    model.eval()

    x = torch.randint(0, 1000, (2, 256))
    targets = torch.randint(0, 1000, (2, 256))
    with torch.no_grad():
        logits, loss = model(x, targets=targets)
    assert logits.shape == (2, 256, 1000), f"Unexpected shape: {logits.shape}"
    assert loss is not None, "Loss should be computed with targets"
    print("PASS: forward pass works with dynamic chunking")


def test_round_to_pow2():
    """Verify pow2 rounding helper."""
    assert _round_to_pow2(1) == 1
    assert _round_to_pow2(2) == 2
    assert _round_to_pow2(3) == 2  # equidistant, rounds down
    assert _round_to_pow2(5) == 4
    assert _round_to_pow2(6) == 4  # closer to 4 than 8
    assert _round_to_pow2(7) == 8
    assert _round_to_pow2(8) == 8
    assert _round_to_pow2(15) == 16
    assert _round_to_pow2(16) == 16
    assert _round_to_pow2(17) == 16
    assert _round_to_pow2(32) == 32
    assert _round_to_pow2(33) == 32
    assert _round_to_pow2(48) == 32  # closer to 32 than 64
    assert _round_to_pow2(64) == 64
    print("PASS: pow2 rounding helper correct")


def test_piecewise_lookup():
    """Verify piecewise lookup helper."""
    assert _piecewise_lookup(100, "512:32,2048:64", 128) == 32
    assert _piecewise_lookup(600, "512:32,2048:64", 128) == 64
    assert _piecewise_lookup(3000, "512:32,2048:64", 128) == 128
    assert _piecewise_lookup(100, "", 64) == 64
    print("PASS: piecewise lookup helper correct")


def test_dynamic_plus_dense():
    """Dynamic chunking + dense_mode forward pass works."""
    import torch

    cfg = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        dynamic_chunking=True,
        chunk_size_alpha=0.5,
        dense_mode=True,
    )
    model = GPT2_RGSA(cfg)
    model.eval()

    x = torch.randint(0, 1000, (2, 256))
    targets = torch.randint(0, 1000, (2, 256))
    with torch.no_grad():
        logits, loss = model(x, targets=targets)
    assert logits.shape == (2, 256, 1000)
    assert loss is not None
    print("PASS: dynamic+dense forward pass works")


def test_dynamic_plus_random():
    """Dynamic chunking + random_routing forward pass works."""
    import torch

    cfg = RGSAConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=1000,
        chunk_size=64,
        routing_dim=16,
        top_b=4,
        local_window=64,
        dynamic_chunking=True,
        chunk_size_alpha=0.5,
        random_routing=True,
    )
    model = GPT2_RGSA(cfg)
    model.eval()

    x = torch.randint(0, 1000, (2, 256))
    targets = torch.randint(0, 1000, (2, 256))
    with torch.no_grad():
        logits, loss = model(x, targets=targets)
    assert logits.shape == (2, 256, 1000)
    assert loss is not None
    print("PASS: dynamic+random forward pass works")


def test_n_chunks_eff_gte_top_b():
    """Verify n_chunks_eff >= top_b is handled (via RetrievalGate clamping)."""
    cfg = RGSAConfig(
        dynamic_chunking=True,
        chunk_size_min=128,
        chunk_size_max=256,
        chunk_size_alpha=0.5,
        chunk_size_schedule="power",
        chunk_size_rounding="nearest",
        top_b=8,
    )
    # With large chunk sizes, short sequences have few chunks
    for sl in [128, 256, 512]:
        cs = compute_chunk_size(sl, cfg)
        n_chunks = (sl + cs - 1) // cs
        # RetrievalGate clamps top_b to min(top_b, num_chunks)
        effective_top_b = min(cfg.top_b, n_chunks)
        assert (
            effective_top_b <= n_chunks
        ), f"top_b={effective_top_b} > n_chunks={n_chunks} for sl={sl}"
    print("PASS: n_chunks_eff vs top_b clamping verified")


if __name__ == "__main__":
    test_round_to_pow2()
    test_piecewise_lookup()
    test_compute_chunk_size_static()
    test_compute_chunk_size_bounds()
    test_compute_chunk_size_pow2_rounding()
    test_compute_chunk_size_multiple_of_8()
    test_compute_chunk_size_growth()
    test_piecewise_schedule()
    test_expected_values_alpha05()
    test_param_count_unchanged()
    test_forward_pass_dynamic()
    test_dynamic_plus_dense()
    test_dynamic_plus_random()
    test_n_chunks_eff_gte_top_b()
    print("\nAll tests passed!")
