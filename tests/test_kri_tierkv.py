# SPDX-License-Identifier: GPL-2.0
"""KRI-TierKV emulation-core tests (CPU).

Pin the block bookkeeping, the eviction/retrieval policies, KRI-D-sum scoring,
and the fake-quant round-trip. The load-bearing check: on a cache whose
attention mass is concentrated in a few old blocks, KRI-D-sum retrieval captures
more of that mass than recent-only, which is the whole reason to build it.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kri_tierkv.config import (
    TierKVConfig,
    EvictionPolicy,
    RetrievalPolicy,
    QuantPolicy,
)
from kri_tierkv.block_index import BlockIndex, FAST, SLOW
from kri_tierkv import scoring, eviction, fake_quant, trace


def _cfg(**kw):
    c = TierKVConfig(
        block_size=4, fast_window_tokens=8, protect_prefix_tokens=4, slow_topk_blocks=2
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_block_index_tiers_and_protection():
    cfg = _cfg()  # block_size 4, fast window 8 tokens = 2 fast blocks
    idx = BlockIndex(seq_len=40, cfg=cfg)  # 10 blocks
    assert idx.num_blocks == 10
    assert idx.fast_ids() == [8, 9]  # last 2 blocks
    assert set(idx.slow_ids()) == set(range(8))
    prot = idx.protected_ids()
    assert 0 in prot  # prefix block protected (protect_prefix_tokens=4 -> 1 block)
    assert 8 in prot and 9 in prot  # fast window protected


def test_quant_policy_bits():
    assert TierKVConfig(quant_policy=QuantPolicy.NONE).resolved_bits() == (16, 16)
    assert TierKVConfig(quant_policy=QuantPolicy.V8_ONLY).resolved_bits() == (16, 8)
    assert TierKVConfig(quant_policy=QuantPolicy.V4_ONLY).resolved_bits() == (16, 4)
    assert TierKVConfig(quant_policy=QuantPolicy.K8V8).resolved_bits() == (8, 8)
    assert TierKVConfig(quant_policy=QuantPolicy.K8V4).resolved_bits() == (8, 4)
    assert TierKVConfig(quant_policy=QuantPolicy.K8V4).is_unsafe_quant()
    assert not TierKVConfig(quant_policy=QuantPolicy.V8_ONLY).is_unsafe_quant()


def test_kri_d_sum_scores_shape_and_ranking():
    # one layer, 1 head, 16 tokens, dim 8; make block 2 large in norm
    k = torch.randn(1, 1, 16, 8) * 0.1
    v = torch.randn(1, 1, 16, 8) * 0.1
    k[:, :, 8:12, :] += 3.0  # block 2 (tokens 8..11) has big keys
    scores = scoring.kri_d_sum_scores([(k, v)], block_size=4)
    assert len(scores) == 4
    assert scores[2] == max(scores)  # the loud block scores highest


def test_kri_eviction_keeps_high_score():
    cfg = _cfg(eviction_policy=EvictionPolicy.KRI_D_SUM, slow_topk_blocks=2)
    idx = BlockIndex(seq_len=40, cfg=cfg)  # 10 blocks, slow 0..7, evictable 1..7
    scores = [0.0] * 10
    scores[5] = 10.0
    scores[6] = 9.0  # blocks 5,6 valuable
    ev = eviction.evict(idx, cfg, scores)
    assert 5 not in ev and 6 not in ev  # kept
    assert len(idx.evictable_ids()) - len(ev) == 2  # budget honored


def test_retrieval_recent_only_vs_kri_topk():
    cfg = _cfg(slow_topk_blocks=2)
    idx = BlockIndex(seq_len=40, cfg=cfg)
    scores = [float(b) for b in range(10)]
    scores[3] = 100.0  # an old block with high KRI score
    cfg.retrieval_policy = RetrievalPolicy.RECENT_ONLY
    rec = eviction.retrieve(idx, cfg, scores)
    assert 3 not in rec  # recent-only never fetches the old high-value block
    cfg.retrieval_policy = RetrievalPolicy.KRI_TOPK
    kri = eviction.retrieve(idx, cfg, scores)
    assert 3 in kri  # kri_topk fetches it


def test_kri_topk_beats_recent_only_on_mass_recall():
    # attention mass concentrated in an OLD block (block 2); recent-only misses it
    cfg = _cfg(slow_topk_blocks=2)
    idx = BlockIndex(seq_len=40, cfg=cfg)
    mass = [0.0] * 10
    mass[2] = 0.6  # old block holds most mass
    mass[9] = 0.4  # recent block
    kri_scores = list(mass)  # KRI-D-sum tracks mass here
    cfg.retrieval_policy = RetrievalPolicy.RECENT_ONLY
    r_recent = eviction.retrieve(idx, cfg, kri_scores)
    cfg.retrieval_policy = RetrievalPolicy.KRI_TOPK
    r_kri = eviction.retrieve(idx, cfg, kri_scores)
    recall_recent = trace.attention_mass_recall(r_recent, mass)
    recall_kri = trace.attention_mass_recall(r_kri, mass)
    assert recall_kri > recall_recent
    assert recall_kri >= 0.99  # captures the old high-mass block


def test_fake_quant_roundtrip_error_grows_as_bits_drop():
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)
    _, v8 = fake_quant.fake_quant_kv(k, v, 16, 8)
    _, v4 = fake_quant.fake_quant_kv(k, v, 16, 4)
    e8 = fake_quant.quant_error(v, v8)
    e4 = fake_quant.quant_error(v, v4)
    assert e8 < e4  # fewer bits -> more error
    # K16 pass-through is exact
    k16, _ = fake_quant.fake_quant_kv(k, v, 16, 8)
    assert fake_quant.quant_error(k, k16) == 0.0


def test_high_mass_false_negative_rate():
    mass = [0.0] * 10
    mass[2] = 0.5
    mass[7] = 0.5
    # selection that keeps the top block but drops the other high-mass block
    fnr = trace.high_mass_false_negative_rate([2], mass, quantile=0.7)
    assert 0.0 < fnr <= 1.0
    # keeping both high-mass blocks -> no false negatives
    assert trace.high_mass_false_negative_rate([2, 7], mass, quantile=0.7) == 0.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
