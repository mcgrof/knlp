"""Trellis update-stabilizer unit tests."""

import torch

from trellis_lm.activations import silu
from trellis_lm.config import TrellisConfig
from trellis_lm.model import TrellisLM
from trellis_lm.trellis_memory import (
    _apply_delta_ratio_cap_to_u,
    _apply_innovation_rms_cap,
    run_trellis_memory_chunked_state_evolution as evo,
    trellis_chunk_decay,
)


def test_innovation_rms_cap_leaves_small_values_unchanged():
    err = torch.full((2, 3, 4), 0.25)
    capped, scale = _apply_innovation_rms_cap(err, cap=8.0)
    assert torch.equal(capped, err)
    assert torch.equal(scale, torch.ones_like(scale))


def test_innovation_rms_cap_limits_large_values():
    err = torch.full((2, 3, 4), 32.0)
    capped, scale = _apply_innovation_rms_cap(err, cap=8.0)
    rms = capped.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 8.0), atol=1e-5)
    assert torch.all(scale < 1.0)


def test_delta_ratio_cap_leaves_small_update_unchanged():
    u = torch.full((1, 2, 4), 0.01)
    write = torch.full((1, 2, 3), 0.01)
    gamma = torch.full((1, 2, 1, 1), 0.005)
    state = torch.ones(1, 2, 4, 3)
    capped, scale = _apply_delta_ratio_cap_to_u(
        u, write, gamma, state, cap=8.0, state_floor=1e-3
    )
    assert torch.allclose(capped, u)
    assert torch.equal(scale, torch.ones_like(scale))


def test_delta_ratio_cap_limits_pathological_update():
    u = torch.full((1, 2, 4), 100.0)
    write = torch.full((1, 2, 3), 10.0)
    gamma = torch.full((1, 2, 1, 1), 0.005)
    state = torch.full((1, 2, 4, 3), 0.01)
    capped, scale = _apply_delta_ratio_cap_to_u(
        u, write, gamma, state, cap=4.0, state_floor=1e-3
    )
    raw = gamma * torch.einsum("bhm,bhd->bhmd", capped, write)
    state_rms = state.float().pow(2).mean(dim=(-1, -2)).sqrt()
    delta_rms = raw.float().pow(2).mean(dim=(-1, -2)).sqrt()
    assert torch.all(delta_rms <= 4.0 * state_rms + 1e-5)
    assert torch.all(scale < 1.0)


def test_none_stabilizer_matches_default_state_evolution():
    torch.manual_seed(0)
    B, H, T, D, M, C = 2, 3, 17, 8, 8, 8
    write = torch.randn(B, H, T, D)
    alpha = torch.randn(B, H, T, M)
    beta = torch.sigmoid(torch.randn(B, H, T, 1))
    gamma = torch.rand(H) + 0.1
    P, rmat, _ = trellis_chunk_decay(beta, C)
    ref = evo(write, alpha, beta, gamma, silu, C, P=P, rmat=rmat)
    cur = evo(
        write,
        alpha,
        beta,
        gamma,
        silu,
        C,
        P=P,
        rmat=rmat,
        trellis_update_stabilizer="none",
    )
    for a, b in zip(ref[:2], cur[:2]):
        assert torch.allclose(a, b)


def test_layer0_gamma_multiplier_only_changes_layer0_effective_gamma():
    cfg = TrellisConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_head=8,
        n_slots=8,
        trellis_update_stabilizer="layerwise_gamma",
        trellis_layer0_gamma_mult=0.25,
    )
    model = TrellisLM(cfg)
    gamma0 = torch.ones(cfg.n_heads)
    layer0 = model.blocks[0].mixer.effective_gamma(gamma0)
    layer1 = model.blocks[1].mixer.effective_gamma(gamma0)
    assert torch.allclose(layer0, torch.full_like(gamma0, 0.25))
    assert torch.allclose(layer1, gamma0)


def test_layer0_gamma_alias_matches_requested_cap_combo_name():
    cfg = TrellisConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_head=8,
        n_slots=8,
        trellis_update_stabilizer="innovation_rms_cap_plus_layer0_gamma",
        trellis_innovation_rms_cap=24.0,
        trellis_layer0_gamma_mult=0.75,
    )
    model = TrellisLM(cfg)
    gamma0 = torch.ones(cfg.n_heads)
    layer0 = model.blocks[0].mixer.effective_gamma(gamma0)
    layer1 = model.blocks[1].mixer.effective_gamma(gamma0)
    assert torch.allclose(layer0, torch.full_like(gamma0, 0.75))
    assert torch.allclose(layer1, gamma0)


def test_synthetic_large_innovation_has_no_nan_reference_path():
    torch.manual_seed(0)
    B, H, T, D, M, C = 1, 2, 16, 8, 8, 8
    write = torch.randn(B, H, T, D)
    alpha = torch.full((B, H, T, M), 1000.0)
    beta = torch.full((B, H, T, 1), 0.99)
    gamma = torch.full((H,), 0.005)
    M0s, us, _, _, _ = evo(
        write,
        alpha,
        beta,
        gamma,
        silu,
        C,
        trellis_update_stabilizer="innovation_rms_cap",
        trellis_innovation_rms_cap=8.0,
    )
    assert torch.isfinite(M0s).all()
    assert torch.isfinite(us).all()


def test_triton_innovation_cap_matches_reference_if_available():
    try:
        import triton  # noqa: F401

        from trellis_lm.trellis_triton import (
            HAS_TRITON,
            TrellisStateEvolutionTriton,
        )
    except Exception:
        return
    if not (HAS_TRITON and torch.cuda.is_available() and torch.version.hip is None):
        return

    torch.manual_seed(0)
    B, H, T, D, M, C = 2, 4, 33, 64, 48, 16
    dev = "cuda"
    write = torch.randn(B, H, T, D, device=dev)
    alpha = torch.randn(B, H, T, M, device=dev) * 32.0
    beta = torch.sigmoid(torch.randn(B, H, T, 1, device=dev))
    gamma = torch.rand(H, device=dev) + 0.1
    P, rmat, _ = trellis_chunk_decay(beta, C)
    ref_m0, ref_u, _, _, _ = evo(
        write,
        alpha,
        beta,
        gamma,
        silu,
        C,
        P=P,
        rmat=rmat,
        trellis_update_stabilizer="innovation_rms_cap",
        trellis_innovation_rms_cap=8.0,
    )
    tri_m0, tri_u = TrellisStateEvolutionTriton.apply(
        write,
        alpha,
        P,
        rmat,
        gamma,
        C,
        "silu",
        "innovation_rms_cap",
        8.0,
    )
    assert torch.allclose(ref_m0, tri_m0, atol=1e-4, rtol=1e-4)
    assert torch.allclose(ref_u, tri_u, atol=1e-4, rtol=1e-4)
