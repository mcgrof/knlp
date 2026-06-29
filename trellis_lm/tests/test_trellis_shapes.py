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
