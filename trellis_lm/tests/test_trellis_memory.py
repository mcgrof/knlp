import torch
from trellis_lm.tests._helpers import tiny_cfg
from trellis_lm.model import TrellisLM
from trellis_lm.trellis_memory import run_trellis_memory
from trellis_lm.activations import get_activation


def test_run_shapes_both_modes():
    B, H, T, D, M = 2, 2, 8, 16, 4
    phi = get_activation("ln_silu")
    write = torch.randn(B, H, T, D)
    q = torch.randn(B, H, T, D)
    alpha = torch.randn(B, H, T, M)
    beta = torch.sigmoid(torch.randn(B, H, T, 1))
    gamma = torch.full((H,), 0.01)
    yhat = run_trellis_memory(write, q, alpha, beta, gamma, phi, "M_q", training=True)
    assert yhat.shape == (B, H, T, M)
    r = torch.randn(B, H, T, M)
    y = run_trellis_memory(write, r, alpha, beta, gamma, phi, "M_T_r", training=True)
    assert y.shape == (B, H, T, D)


def test_grad_flow_through_all_projections():
    cfg = tiny_cfg()
    m = TrellisLM(cfg)
    m.train()
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    _, loss = m(idx, labels=idx, training=True)
    loss.backward()
    got = [n for n, p in m.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0]
    for needle in ("alpha_proj", "beta_proj", "gamma_raw", "q_proj", "k_proj", "v_proj"):
        assert any(needle in n for n in got), f"no gradient reached {needle}"


def test_deterministic_eval():
    cfg = tiny_cfg()
    m = TrellisLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        a, _ = m(idx, training=False)
        b, _ = m(idx, training=False)
    assert torch.allclose(a, b)


def test_bounded_memory_no_full_kv():
    """Trellis state must be O(layers*heads*slots*d_head), independent of T, and
    far smaller than a full KV cache at long context."""
    cfg = tiny_cfg(max_seq_len=4096)
    m = TrellisLM(cfg)
    B, T, elem = 1, 2048, 4  # fp32
    full_kv = 2 * cfg.n_layers * B * cfg.n_heads * T * cfg.d_head * elem
    state = m.memory_state_bytes(B)
    assert state > 0
    assert state < full_kv / 4, f"state {state} not << full-KV {full_kv}"


def test_forget_gate_off_changes_output():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m_on = TrellisLM(cfg).eval()
    cfg2 = tiny_cfg(forget_gate=False)
    m_off = TrellisLM(cfg2)
    m_off.load_state_dict(m_on.state_dict())
    m_off.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        a, _ = m_on(idx, training=False)
        b, _ = m_off(idx, training=False)
    assert not torch.allclose(a, b), "forget gate had no effect"
