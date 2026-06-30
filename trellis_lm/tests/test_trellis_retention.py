"""Trellis learned/fixed retention mode tests."""

import math

import torch

from trellis_lm.model import TrellisLM
from trellis_lm.tests._helpers import tiny_cfg


def test_fixed_beta_broadcasts_constant_per_head():
    cfg = tiny_cfg(
        trellis_retention_mode="fixed_beta",
        trellis_beta_init=0.99,
    )
    mixer = TrellisLM(cfg).blocks[0].mixer
    h = torch.randn(2, 7, cfg.d_model)
    beta = mixer.compute_beta(h.float())
    assert mixer.beta_proj is None
    assert mixer.retention_theta is None
    assert beta.shape == (2, cfg.n_heads, 7, 1)
    assert torch.allclose(beta, torch.full_like(beta, 0.99), atol=1e-6)


def test_fixed_beta_matches_constant_token_projection():
    common = dict(
        activation="silu",
        alpha_mode="linear",
        value_readout_act="none",
        use_short_conv_qk=False,
        chunk_size=1,
        beta_init=0.99,
        trellis_beta_init=0.99,
    )
    token_cfg = tiny_cfg(trellis_retention_mode="token_proj", **common)
    fixed_cfg = tiny_cfg(trellis_retention_mode="fixed_beta", **common)
    torch.manual_seed(0)
    token_model = TrellisLM(token_cfg).eval()
    torch.manual_seed(1)
    fixed_model = TrellisLM(fixed_cfg).eval()

    fixed_state = fixed_model.state_dict()
    token_state = token_model.state_dict()
    for key in list(fixed_state):
        if key in token_state and fixed_state[key].shape == token_state[key].shape:
            fixed_state[key] = token_state[key]
    fixed_model.load_state_dict(fixed_state)

    logit = math.log(0.99 / 0.01)
    for block in token_model.blocks:
        beta_proj = block.mixer.beta_proj
        beta_proj.weight.data.zero_()
        beta_proj.bias.data.fill_(logit)

    idx = torch.randint(0, token_cfg.vocab_size, (2, 9))
    with torch.no_grad():
        token_logits, _ = token_model(idx, training=False)
        fixed_logits, _ = fixed_model(idx, training=False)
    assert torch.allclose(token_logits, fixed_logits, atol=1e-5, rtol=1e-5)


def test_learned_per_head_initializes_and_stays_in_bounds():
    cfg = tiny_cfg(
        trellis_retention_mode="learned_per_head",
        trellis_beta_init=0.99,
        trellis_beta_min=0.90,
        trellis_beta_max=0.9995,
        trellis_beta_init_schedule="flat_099",
    )
    mixer = TrellisLM(cfg).blocks[0].mixer
    beta = mixer.retention_beta_values()
    assert beta.shape == (cfg.n_heads,)
    assert torch.allclose(beta, torch.full_like(beta, 0.99), atol=1e-6)
    assert torch.all(beta >= cfg.trellis_beta_min)
    assert torch.all(beta <= cfg.trellis_beta_max)


def test_learned_per_head_broadcasts_to_fused_shape():
    cfg = tiny_cfg(
        trellis_retention_mode="learned_per_head",
        trellis_beta_init_schedule="head_logspace",
    )
    mixer = TrellisLM(cfg).blocks[0].mixer
    h = torch.randn(3, 11, cfg.d_model)
    beta = mixer.compute_beta(h.float())
    assert beta.shape == (3, cfg.n_heads, 11, 1)
    assert beta[..., 0].amin() >= cfg.trellis_beta_min
    assert beta[..., 0].amax() <= cfg.trellis_beta_max


def test_learned_per_channel_broadcasts_over_memory_slots():
    cfg = tiny_cfg(
        trellis_retention_mode="learned_per_channel",
        trellis_beta_init_schedule="head_logspace",
    )
    mixer = TrellisLM(cfg).blocks[0].mixer
    h = torch.randn(2, 5, cfg.d_model)
    beta = mixer.compute_beta(h.float())
    assert beta.shape == (2, cfg.n_heads, 5, cfg.n_slots)
    assert beta.amin() >= cfg.trellis_beta_min
    assert beta.amax() <= cfg.trellis_beta_max


def test_learned_per_head_channel_broadcasts_over_heads_and_slots():
    cfg = tiny_cfg(
        trellis_retention_mode="learned_per_head_channel",
        trellis_beta_init_schedule="layer_head_logspace",
    )
    mixer = TrellisLM(cfg).blocks[0].mixer
    h = torch.randn(2, 5, cfg.d_model)
    beta = mixer.compute_beta(h.float())
    assert mixer.retention_beta_values().shape == (cfg.n_heads, cfg.n_slots)
    assert beta.shape == (2, cfg.n_heads, 5, cfg.n_slots)
    assert beta.amin() >= cfg.trellis_beta_min
    assert beta.amax() <= cfg.trellis_beta_max


def test_beta_gradients_flow_for_learned_per_head():
    cfg = tiny_cfg(
        activation="silu",
        alpha_mode="linear",
        value_readout_act="none",
        use_short_conv_qk=False,
        chunk_size=4,
        trellis_retention_mode="learned_per_head",
        trellis_beta_init_schedule="flat_099",
    )
    model = TrellisLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(idx, labels=idx)
    loss.backward()
    grads = [block.mixer.retention_theta.grad for block in model.blocks]
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum().item() for grad in grads) > 0.0
