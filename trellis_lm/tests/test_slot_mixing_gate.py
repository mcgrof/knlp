"""Slot-Mixing Delta rank-r gate: fp64 A/B correctness.

The rank-r low-rank gain G(x) = diag(a(x)) + U(x) V(x)^T must reduce EXACTLY to
the diagonal gate when the low-rank factor vanishes, and must be finite/stable
for r=1,2. r=0 builds no projection at all (bit-identical diagonal), so these
tests exercise the additive low-rank TERM in run_trellis_memory directly, and a
model-level smoke for the mixer wiring + reset init.
"""

import torch

from trellis_lm.config import TrellisConfig
from trellis_lm.model import TrellisLM
from trellis_lm.trellis_memory import run_trellis_memory


def _rand_inputs(B, H, T, D, M, r, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    write = rnd(B, H, T, D)
    read = rnd(B, H, T, D)  # M_q pass reads with D-dim query
    alpha = rnd(B, H, T, M) * 0.3
    beta = torch.sigmoid(rnd(B, H, T, 1)) * 0.1 + 0.89  # ~0.89..0.99
    gamma = torch.full((H,), 0.05, dtype=dtype)
    a_gate = 1.0 + 0.2 * rnd(B, H, T, M)  # diagonal gain ~1
    U = rnd(B, H, T, M, r)
    V = rnd(B, H, T, M, r)
    return write, read, alpha, beta, gamma, a_gate, U, V


def _run(write, read, alpha, beta, gamma, a_gate, lowrank):
    phi = lambda x: x  # unused in the input_gate branch
    return run_trellis_memory(
        write,
        read,
        alpha,
        beta,
        gamma,
        phi,
        "M_q",
        training=False,
        input_gate=a_gate,
        input_gate_lowrank=lowrank,
    )


def test_zero_lowrank_matches_diagonal():
    """U V^T = 0 (U zeroed) must reproduce the diagonal path to fp64 machine
    precision -- the low-rank term is a clean additive extension."""
    B, H, T, D, M, r = 2, 3, 12, 8, 6, 2
    w, rd, al, be, ga, ag, U, V = _rand_inputs(B, H, T, D, M, r, torch.float64)
    diag = _run(w, rd, al, be, ga, ag, None)
    U0 = torch.zeros_like(U)
    lr = _run(w, rd, al, be, ga, ag, (U0, V))
    err = (diag - lr).abs().max().item()
    assert err < 1e-12, f"zero-U low-rank != diagonal: max err {err:.3e}"


def test_random_lowrank_finite_and_differs():
    B, H, T, D, M, r = 2, 3, 12, 8, 6, 2
    w, rd, al, be, ga, ag, U, V = _rand_inputs(B, H, T, D, M, r, torch.float64)
    diag = _run(w, rd, al, be, ga, ag, None)
    for rr in (1, 2):
        Ur, Vr = U[..., :rr], V[..., :rr]
        out = _run(w, rd, al, be, ga, ag, (Ur, Vr))
        assert torch.isfinite(out).all(), f"rank{rr} produced non-finite output"
        d = (diag - out).abs().max().item()
        assert d > 1e-6, f"rank{rr} did not change the output (max diff {d:.3e})"


def _cfg(rank, scope="per_slot"):
    return TrellisConfig(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_head=32,
        n_slots=16,
        max_seq_len=64,
        chunk_size=1,
        write_l2norm=True,
        gamma_init=0.05,
        trellis_write_mode="input_conditioned",
        trellis_input_gate_act="sigmoid",
        trellis_input_gate_scope=scope,
        trellis_input_gate_rank=rank,
        dtype="fp32",
    )


def test_model_smoke_rank_ladder():
    """scalar / diagonal / rank1 / rank2 all build, forward, and backprop
    with finite grads; rank>0 starts as a pure diagonal (U V^T = 0 at init)."""
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (2, 24))
    labels = idx.clone()
    for scope, rank in [
        ("scalar", 0),
        ("per_slot", 0),
        ("per_slot", 1),
        ("per_slot", 2),
    ]:
        model = TrellisLM(_cfg(rank, scope))
        if rank > 0:
            # U V^T must be 0 at init: reset zeros U, small-inits V.
            m = model.blocks[0].mixer
            H, M, rr = m.H, m.M, m.write_gate_rank
            w = m.write_lowrank_proj.weight
            assert w[: H * M * rr].abs().max().item() == 0.0, "U not zeroed at init"
        out = model(idx, labels=labels)
        loss = out[1] if isinstance(out, tuple) else out.loss
        assert torch.isfinite(loss).all()
        loss.backward()
        gmax = max(
            p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
        )
        assert gmax < 1e4, f"scope={scope} rank={rank}: grad blew up ({gmax:.3e})"


def test_rank0_no_lowrank_param_and_deterministic():
    """rank=0 must add NO low-rank parameter (so it draws the same RNG stream
    as the pre-feature diagonal path) and must build bit-identically under a
    fixed seed. Guards against the low-rank machinery silently perturbing the
    diagonal baseline it is meant to reduce to."""
    torch.manual_seed(1234)
    m0 = TrellisLM(_cfg(0))
    names = dict(m0.named_parameters())
    assert not any(
        "write_lowrank" in n for n in names
    ), "rank=0 built a write_lowrank parameter"
    assert m0.blocks[0].mixer.write_lowrank_proj is None
    # same seed -> bit-identical state dict (reset ordering is deterministic)
    torch.manual_seed(1234)
    m0b = TrellisLM(_cfg(0))
    for k, v in m0.state_dict().items():
        assert torch.equal(v, m0b.state_dict()[k]), f"non-deterministic build: {k}"


def test_write_gate_bias_survives_init():
    """The input-conditioned write gate's a≡1 bias init must survive the
    model-wide _init_weights (which zeros all Linear biases). Regression for
    the reset loop omitting reset_write_gate_bias -> the diagonal gate started
    at the wrong operating point. sigmoid scope: reset value is bias=0."""
    torch.manual_seed(0)
    for scope, rank in [("scalar", 0), ("per_slot", 0), ("per_slot", 2)]:
        model = TrellisLM(_cfg(rank, scope))
        b = model.blocks[0].mixer.write_gate_proj.bias
        # sigmoid gate resets bias to 0.0 (2*sigmoid(0)=1 == a≡1)
        assert torch.equal(b, torch.zeros_like(b)), (
            f"scope={scope} rank={rank}: write-gate bias not at reset value "
            f"(max |b|={b.abs().max().item():.3e}) -- _init_weights clobber"
        )


if __name__ == "__main__":
    test_zero_lowrank_matches_diagonal()
    test_random_lowrank_finite_and_differs()
    test_model_smoke_rank_ladder()
    test_rank0_no_lowrank_param_and_deterministic()
    test_write_gate_bias_survives_init()
    print("ALL SLOT-MIXING GATE TESTS PASSED")
