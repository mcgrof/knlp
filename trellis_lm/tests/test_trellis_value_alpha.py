"""Trellis value-alpha/keyed-binding branch tests."""

import torch

from trellis_lm.model import TrellisLM
from trellis_lm.tests._helpers import tiny_cfg


def _tiny_model(mode: str = "shared", correction_init: float = 1e-3):
    cfg = tiny_cfg(
        activation="silu",
        alpha_mode="linear",
        beta_init=0.99,
        gamma_init=0.005,
        chunk_size=4,
        output_path="paper",
        use_short_conv_v=True,
        trellis_update_stabilizer="layerwise_gamma",
        trellis_layer0_gamma_mult=0.5,
        residual_update_mix=0.10,
        trellis_value_alpha_mode=mode,
        trellis_value_alpha_correction_init=correction_init,
        trellis_value_alpha_correction_max=0.25,
    )
    return TrellisLM(cfg)


def test_default_value_alpha_mode_is_shared():
    assert tiny_cfg().trellis_value_alpha_mode == "shared"


def test_key_readout_value_alpha_forward_backward_is_finite():
    torch.manual_seed(0)
    model = _tiny_model("key_readout")
    idx = torch.randint(0, model.cfg.vocab_size, (2, 17))
    labels = idx.clone()
    logits, loss = model(idx, labels=labels, training=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_key_readout_changes_outputs_with_same_weights():
    torch.manual_seed(1)
    shared = _tiny_model("shared").eval()
    keyed = _tiny_model("key_readout").eval()
    keyed.load_state_dict(shared.state_dict())
    idx = torch.randint(0, shared.cfg.vocab_size, (2, 19))
    with torch.no_grad():
        shared_logits, _ = shared(idx, training=False)
        keyed_logits, _ = keyed(idx, training=False)
    assert not torch.allclose(shared_logits, keyed_logits)


def test_shared_plus_key_correction_starts_near_configured_scale():
    model = _tiny_model("shared_plus_key_correction", correction_init=1e-3)
    scales = []
    for block in model.blocks:
        raw = block.mixer.value_alpha_correction_raw
        assert raw is not None
        scales.append(
            block.mixer.cfg.trellis_value_alpha_correction_max
            * torch.sigmoid(raw.detach().float())
        )
    flat = torch.cat(scales)
    assert torch.allclose(flat, torch.full_like(flat, 1e-3), atol=1e-6)


def test_shared_plus_key_correction_forward_backward_is_finite():
    torch.manual_seed(2)
    model = _tiny_model("shared_plus_key_correction")
    idx = torch.randint(0, model.cfg.vocab_size, (2, 17))
    labels = idx.clone()
    logits, loss = model(idx, labels=labels, training=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        block.mixer.value_alpha_correction_raw.grad
        for block in model.blocks
        if block.mixer.value_alpha_correction_raw is not None
    ]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_prev_alpha_correction_changes_outputs_with_same_weights():
    torch.manual_seed(3)
    shared = _tiny_model("shared").eval()
    prev = _tiny_model("shared_plus_prev_alpha_correction", correction_init=0.1).eval()
    prev.load_state_dict(shared.state_dict(), strict=False)
    idx = torch.randint(0, shared.cfg.vocab_size, (2, 19))
    with torch.no_grad():
        shared_logits, _ = shared(idx, training=False)
        prev_logits, _ = prev(idx, training=False)
    assert not torch.allclose(shared_logits, prev_logits)


def test_prev_key_correction_forward_backward_is_finite():
    torch.manual_seed(4)
    model = _tiny_model("shared_plus_prev_key_correction")
    idx = torch.randint(0, model.cfg.vocab_size, (2, 17))
    labels = idx.clone()
    logits, loss = model(idx, labels=labels, training=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        block.mixer.value_alpha_correction_raw.grad
        for block in model.blocks
        if block.mixer.value_alpha_correction_raw is not None
    ]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
