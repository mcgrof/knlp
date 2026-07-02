import torch
from trellis_lm.tests._helpers import tiny_cfg
from trellis_lm.model import TrellisLM, DenseTransformerTiny


def test_trellis_forward_shapes():
    cfg = tiny_cfg()
    m = TrellisLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(idx, labels=idx, training=False)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert m.get_num_params() > 0


def test_dense_baseline_shapes():
    cfg = tiny_cfg()
    m = DenseTransformerTiny(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(idx, labels=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_beta_per_slot_and_no_conv_shapes():
    cfg = tiny_cfg(beta_mode="per_slot", use_short_conv_qk=False, activation="softmax")
    m = TrellisLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (3, 20))
    logits, _ = m(idx, training=False)
    assert logits.shape == (3, 20, cfg.vocab_size)


def test_update_gate_and_repair_knobs_shapes():
    for mode in ("scalar", "channel"):
        cfg = tiny_cfg(
            activation="silu",
            chunk_size=4,
            beta_mode="scalar_per_head",
            update_gate_mode=mode,
            use_short_conv_v=True,
            residual_update_mix=0.25,
        )
        m = TrellisLM(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _ = m(idx, training=False)
        assert logits.shape == (2, 16, cfg.vocab_size)


def test_update_gate_floor_initializes_effective_gate():
    cfg = tiny_cfg(
        update_gate_mode="scalar",
        update_gate_init=0.95,
        trellis_update_gate_floor=0.5,
    )
    m = TrellisLM(cfg).eval()
    mixer = m.blocks[0].mixer
    gate = mixer._update_gate_from_logits(mixer.update_gate_proj.bias.detach())
    assert torch.allclose(gate, torch.full_like(gate, 0.95), atol=1e-6)


def test_value_only_update_gate_forward_backward_is_finite():
    cfg = tiny_cfg(
        activation="silu",
        chunk_size=4,
        beta_mode="scalar_per_head",
        update_gate_mode="scalar",
        update_gate_init=0.95,
        trellis_update_gate_target="value",
        trellis_update_gate_floor=0.5,
        use_short_conv_v=True,
        residual_update_mix=0.10,
    )
    m = TrellisLM(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(idx, labels=idx, training=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    gate_grads = [
        block.mixer.update_gate_proj.bias.grad
        for block in m.blocks
        if block.mixer.update_gate_proj is not None
    ]
    assert gate_grads
    assert all(g is not None and torch.isfinite(g).all() for g in gate_grads)


def test_value_only_update_gate_changes_outputs_vs_both_target():
    torch.manual_seed(0)
    both_cfg = tiny_cfg(
        activation="silu",
        chunk_size=4,
        beta_mode="scalar_per_head",
        update_gate_mode="scalar",
        update_gate_init=0.5,
        trellis_update_gate_target="both",
        use_short_conv_v=True,
    )
    value_cfg = tiny_cfg(
        activation="silu",
        chunk_size=4,
        beta_mode="scalar_per_head",
        update_gate_mode="scalar",
        update_gate_init=0.5,
        trellis_update_gate_target="value",
        use_short_conv_v=True,
    )
    both = TrellisLM(both_cfg).eval()
    value = TrellisLM(value_cfg).eval()
    value.load_state_dict(both.state_dict())
    idx = torch.randint(0, both_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        both_logits, _ = both(idx, training=False)
        value_logits, _ = value(idx, training=False)
    assert not torch.allclose(both_logits, value_logits)


def test_update_gate_layer_mode_and_diagnostics():
    cfg = tiny_cfg(
        activation="silu",
        chunk_size=4,
        beta_mode="scalar_per_head",
        update_gate_mode="scalar",
        update_gate_init=0.8,
        trellis_update_gate_target="value",
        trellis_update_gate_layer_mode="layer0",
        use_short_conv_v=True,
        residual_update_mix=0.10,
    )
    m = TrellisLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits, _ = m(idx, training=False)
    assert torch.isfinite(logits).all()

    layer0 = m.blocks[0].mixer.last_trellis_diag
    layer1 = m.blocks[1].mixer.last_trellis_diag
    assert layer0["value_update_gate"] is not None
    assert layer0["key_update_gate"] is None
    assert layer1["value_update_gate"] is None
    assert layer0["value_state"]["rms"] >= 0.0
    assert layer0["value_update"]["rms"] >= 0.0
    assert layer0["backend"] in (
        "pytorch_state_evolution",
        "triton_state_evolution",
    )
