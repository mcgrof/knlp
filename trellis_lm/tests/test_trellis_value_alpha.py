"""Trellis value-alpha/keyed-binding branch tests."""

import torch

from trellis_lm.model import TrellisLM
from trellis_lm.tests._helpers import tiny_cfg


def _tiny_model(mode: str = "shared"):
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
