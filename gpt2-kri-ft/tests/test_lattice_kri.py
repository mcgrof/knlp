"""Unit tests for the Lattice-KRI router (training-free, mask-level).

These run on CPU with random K/V/Q tensors — no model checkpoint needed.
They cover the acceptance criteria that do not require a trained model:
strict causality of the produced mask for every variant and summary mode,
selection eligibility / no-future-leak, the budget bound, and the
rel_only == cosine-only-KRI-Q equivalence. Model-level no-future-leak is
covered by the shared build_kri_mask machinery (the final causal AND) and
the existing tests/test_kri_mask_causal.py.
"""

import pytest
import torch

from src.kri_mask import KRIConfig, build_kri_mask, select_kri_blocks, num_blocks
from src.lattice_kri import (
    LatticeConfig,
    select_lattice_blocks,
    LATTICE_ROUTERS,
    _VARIANTS,
)

BS, LW, TOPK, BLOCK = 16, 32, 3, 16


def _kv(B=2, H=3, T=128, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, H, T, D, generator=g),
        torch.randn(B, H, T, D, generator=g),
        torch.randn(B, H, T, D, generator=g),
    )


def _cfg(**kw):
    base = dict(
        block_size=BLOCK,
        local_window_tokens=LW,
        global_topk_blocks=TOPK,
        protected_blocks=(0,),
    )
    base.update(kw)
    return LatticeConfig(**base)


def test_config_is_a_kriconfig():
    assert isinstance(_cfg(), KRIConfig)


def test_routers_registered():
    assert "lattice_kri" in LATTICE_ROUTERS  # canonical alias
    for v in _VARIANTS:
        assert f"lattice_kri_{v}" in LATTICE_ROUTERS


@pytest.mark.parametrize("variant", _VARIANTS)
def test_selection_eligible_and_no_future(variant):
    k, v, q = _kv()
    cfg = _cfg(variant=variant)
    t = 100
    sel = select_lattice_blocks(q[:, :, t, :], k, v, cfg, t_query=t)
    local_first_block = max(0, t - cfg.local_window_tokens) // cfg.block_size
    assert sel.shape[-1] <= cfg.global_topk_blocks
    if sel.numel():
        # every selected block is strictly before the local window (no future,
        # no overlap with recent) and is not a protected/sink block
        assert int(sel.max()) < local_first_block
        for p in cfg.protected_blocks:
            assert int((sel == p).sum()) == 0
        # no duplicate block within a (batch, head)
        for b in range(sel.shape[0]):
            for h in range(sel.shape[1]):
                row = sel[b, h].tolist()
                assert len(set(row)) == len(row)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("summary", ["k", "v", "kv_cat", "kv_sum"])
def test_mask_strictly_causal(variant, summary):
    T = 96
    k, v, q = _kv(T=T)
    cfg = _cfg(variant=variant, summary_mode=summary, prefill_split=24)
    mask = build_kri_mask(
        cfg,
        T,
        2,
        3,
        k_per_layer=[k],
        v_per_layer=[v],
        q_per_layer=[q],
        select_fn=select_lattice_blocks,
    )
    assert mask.dtype == torch.bool and mask.shape == (2, 3, T, T)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert not (mask & upper.view(1, 1, T, T)).any(), "future token leaked"


def test_rel_only_equals_cosine_only_kri_q():
    """rel_only must collapse onto pure cosine-only KRI-Q (same block set)."""
    k, v, q = _kv()
    t = 100
    sel_lat = select_lattice_blocks(q[:, :, t, :], k, v, _cfg(variant="rel_only"), t)
    kri = KRIConfig(
        block_size=BLOCK,
        local_window_tokens=LW,
        global_topk_blocks=TOPK,
        protected_blocks=(0,),
        w_cos=1.0,
        w_value_energy=0.0,
        w_recency=0.0,
        use_novelty=False,
    )
    sel_kri = select_kri_blocks(q[:, :, t, :], k, v, kri, t_query=t)
    for b in range(sel_lat.shape[0]):
        for h in range(sel_lat.shape[1]):
            assert set(sel_lat[b, h].tolist()) == set(sel_kri[b, h].tolist())


def test_rel_orth_differs_from_rel_only():
    """Orthogonal novelty must actually change the selection (else it is a
    no-op). With distinct content it should diverge for at least some head."""
    k, v, q = _kv(T=256, seed=1)
    t = 220
    sel_rel = select_lattice_blocks(q[:, :, t, :], k, v, _cfg(variant="rel_only"), t)
    sel_orth = select_lattice_blocks(
        q[:, :, t, :], k, v, _cfg(variant="rel_orth", lambda_orth=1.0), t
    )
    diverged = any(
        set(sel_rel[b, h].tolist()) != set(sel_orth[b, h].tolist())
        for b in range(sel_rel.shape[0])
        for h in range(sel_rel.shape[1])
    )
    assert diverged


def test_canonical_adapter_runs_and_is_causal():
    T = 128
    k, v, q = _kv(T=T)
    mask = LATTICE_ROUTERS["lattice_kri"](
        [k],
        [v],
        [q],
        seq_len=T,
        block_size=BLOCK,
        local_window_tokens=LW,
        sink_blocks=1,
        topk_blocks=TOPK,
    )
    assert mask.shape == (2, 3, T, T) and mask.dtype == torch.bool
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert not (mask & upper.view(1, 1, T, T)).any()
