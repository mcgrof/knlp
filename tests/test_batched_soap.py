"""BatchedSOAP must be the reference SOAP with batched kernels.

The load-bearing test is equivalence: two identical tiny GPT-2s fed an
identical fixed sequence of batches, one trained by the vendored
reference SOAP and one by BatchedSOAP, must follow the same trajectory.

Everything runs in float64 -- `torch.set_default_dtype(torch.float64)`
plus `model.double()` -- so the comparison measures algebra rather than
float32 rounding. The default dtype matters as much as the parameter
dtype: the reference allocates its `GG` accumulators with a bare
`torch.zeros(sh, sh, device=...)`, so under the float32 default they
would be float32 and the float64 `lerp_` into them would raise.

Read `test_matches_reference_soap_without_refresh` and
`test_matches_reference_soap` together. The first pins the algebra
exactly; the second measures what the eigenbasis refresh does to it,
against a control that shows the same thing happening to the reference
compared with itself.

These run on CPU deliberately. The same comparison on a GPU proves
nothing: measured on an AMD W7900 (ROCm 6.4, torch 2.9.1), a GPT-2
124M backward pass is nondeterministic at ~1e-7 per gradient, and 12
steps of the reference SOAP amplify that to 0.25 in the weights --
against ITSELF, same seed, same batches. BatchedSOAP sits inside that
band (0.11 vs the reference's own 0.10 run-to-run spread with the
refresh disabled), which is all a GPU run can say.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.batched_soap import BatchedSOAP  # noqa: E402
from fim.fisher_pruning.soap import SOAP  # noqa: E402
from fim.reciprocal_attention.gpt2_spectral_ra import build_model  # noqa: E402

MODEL_CFG = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 32,
    "block_size": 8,
    "vocab_size": 64,
    "dropout": 0.0,
    "bias": True,
}

STEPS = 25
PRECOND_FREQ = 5
NEVER_REFRESH = 10**6
LR = 3e-3

# 64 tokens per batch (8 x block_size 8), not 16. Each gradient of a
# [4*n_embd, n_embd] weight has rank at most min(n_embd, tokens), so a
# 16-token batch cannot fill a 128x128 accumulator no matter how many
# steps run, and the refresh would then be picking an eigenbasis for a
# permanently rank-96 matrix. That degeneracy is a property of the
# setup, not of the optimizer, and it swamps the measurement.
BATCH = 8

# Tolerances. See the two equivalence tests for the reasoning; the
# short version is that everything except the eigenbasis refresh
# reproduces to float64 rounding (measured 1.4e-12 over 25 steps), and
# the refresh is chaotic in the reference itself, so it is measured
# against the reference's own sensitivity rather than a fixed number.
NO_REFRESH_ATOL = 1e-11
REFRESH_ABS_CAP = 1e-2
REFRESH_CONTROL_FACTOR = 2.0

# The smallest float32 multiplier greater than one: 1 + 2**-23.
ONE_ULP = float(
    torch.nextafter(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(2.0, dtype=torch.float32),
    )
)


@pytest.fixture(autouse=True)
def _float64_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


class UlpPerturbedSOAP(SOAP):
    """A control, not an implementation.

    The reference, unmodified, except that `GG` is multiplied by
    1 + 2**-23 -- one float32 ulp -- for the duration of the eigenbasis
    refresh. A positive scalar multiple leaves the eigenvectors of `GG`
    unchanged, leaves the descending order of `diag(o.T @ GG @ o)`
    unchanged, and leaves the `Q` of `qr(GG @ o)` unchanged, so the
    refresh is mathematically unaffected. Only float32 rounding moves,
    by the smallest amount float32 can express -- no more than batching
    a gemm moves it.

    Comparing the reference against this measures how much the
    reference's own refresh amplifies a one-ulp difference, which is
    the yardstick `test_matches_reference_soap` uses.
    """

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        original = state["GG"]
        state["GG"] = [(m * ONE_ULP) if len(m) else m for m in original]
        try:
            return super().get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
        finally:
            state["GG"] = original


def _make_model(seed=1234):
    torch.manual_seed(seed)
    model = build_model(dict(MODEL_CFG), selection={})
    model.double()
    model.train()
    return model


def _make_batches(n, seed=7):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(
            0, MODEL_CFG["vocab_size"], (BATCH, MODEL_CFG["block_size"]), generator=g
        )
        y = torch.randint(
            0, MODEL_CFG["vocab_size"], (BATCH, MODEL_CFG["block_size"]), generator=g
        )
        out.append((x, y))
    return out


def _param_groups(model):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build(cls, model, freq=PRECOND_FREQ):
    return cls(
        _param_groups(model),
        lr=LR,
        betas=(0.95, 0.95),
        precondition_frequency=freq,
    )


def _train(model, opt, batches):
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()


def _max_param_diff(a, b):
    worst, where = 0.0, ""
    da = dict(a.named_parameters())
    for name, pb in b.named_parameters():
        d = (da[name].detach() - pb.detach()).abs().max().item()
        if d > worst:
            worst, where = d, name
    return worst, where


def _race(cls_a, cls_b, freq, steps=STEPS):
    """Train two models on the same batches, one step at a time.

    Returns the per-step history of the largest parameter difference,
    so a divergence can be attributed to the step it started at.
    """
    model_a, model_b = _make_model(), _make_model()
    assert _max_param_diff(model_a, model_b)[0] == 0.0, "seeds gave different inits"
    opt_a, opt_b = _build(cls_a, model_a, freq), _build(cls_b, model_b, freq)
    history = []
    for x, y in _make_batches(steps):
        for model, opt in ((model_a, opt_a), (model_b, opt_b)):
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            loss.backward()
            opt.step()
        history.append(_max_param_diff(model_a, model_b))
    return model_a, opt_a, model_b, opt_b, history


def test_matches_reference_soap_without_refresh():
    """The exact test: everything but the eigenbasis refresh.

    With `precondition_frequency` past the horizon the QR refresh never
    fires, so this exercises the skipped first step, the projection and
    project-back, both moments and their bias correction, the per-step
    `exp_avg` re-projection, the `GG` accumulation, the initial `eigh`,
    the weight update and the decoupled weight decay -- batched against
    unbatched, on identical inputs.

    Measured worst case over 25 steps: 1.4e-12 on weights of scale
    ~2e-2, i.e. 1e-10 relative, which is the float64 accumulation of
    the ~1e-15 relative difference between `bmm(G, G^T)` and
    `tensordot(G, G, ...)` -- one gemm-blocking rounding. `Q` comes out
    bitwise identical because batched and unbatched `linalg.eigh` agree
    bitwise on this machine.
    """
    model_a, opt_a, model_b, opt_b, history = _race(SOAP, BatchedSOAP, NEVER_REFRESH)
    worst, where = max(history, key=lambda t: t[0])
    assert worst < NO_REFRESH_ATOL, f"max param divergence {worst:.3e} at {where}"

    # Optimizer state, not just weights.
    named_b = dict(model_b.named_parameters())
    checked = 0
    for name, pa in model_a.named_parameters():
        sa, sb = opt_a.state[pa], opt_b.state[named_b[name]]
        assert sa["step"] == sb["step"], name
        for key in ("exp_avg", "exp_avg_sq"):
            scale = max(1e-30, sa[key].abs().max().item())
            rel = (sa[key] - sb[key]).abs().max().item() / scale
            assert rel < 1e-11, f"{name}.{key} relative divergence {rel:.3e}"
        for i, (ga, gb) in enumerate(zip(sa["GG"], sb["GG"])):
            if len(ga) == 0:
                assert len(gb) == 0
                continue
            scale = max(1e-30, ga.abs().max().item())
            rel = (ga - gb).abs().max().item() / scale
            assert rel < 1e-11, f"{name}.GG[{i}] relative divergence {rel:.3e}"
        for i, (qa, qb) in enumerate(zip(sa["Q"], sb["Q"])):
            if len(qa) == 0:
                assert len(qb) == 0
                continue
            # Orthonormal, so absolute is already scale-free.
            assert (qa - qb).abs().max().item() < 1e-12, f"{name}.Q[{i}]"
        checked += 1
    assert checked == len(list(model_a.parameters()))


def test_matches_reference_soap():
    """25 steps at precondition_frequency 5, against a control.

    Up to the first refresh the two trajectories agree to float64
    rounding. From the first refresh they separate, reaching ~4e-4 by
    step 25, and the tolerance is NOT loosened to cover that -- the
    divergence is attributed instead.

    Where it comes from: `get_orthogonal_matrix_QR` casts `GG` and `Q`
    down to float32 unconditionally (`m.data.float()` in both
    branches), then sorts `diag(o.T @ m @ o)` and QR-factorizes
    `m @ o`. Batched and unbatched float32 gemms differ by about one
    ulp -- verified directly in
    `test_refresh_matches_reference_on_identical_state` -- and `GG`
    always carries an exactly singular direction, because every one of
    these weights feeds a LayerNorm whose backward removes the mean, so
    one direction of the output space receives zero gradient forever.
    A rank-deficient, float32 power iteration does not determine its
    eigenbasis to better than the perturbation, so a one-ulp difference
    reorders and rotates columns, and the trajectories part.

    That is a property of the reference's refresh, not of batching, so
    the assertion is relative to the reference's own sensitivity.
    `UlpPerturbedSOAP` is the reference with `GG` scaled by one float32
    ulp inside the refresh -- mathematically the identity -- and the
    reference diverges from THAT by 9.0e-4, more than twice its 4.0e-4
    difference from BatchedSOAP. (A third control, re-associating the
    refresh's gemms, gives 5.1e-4; batching is the smallest of the
    three.) The bound self-tightens: if the refresh ever stops being
    chaotic, the control shrinks and so must the batched difference.
    """
    _, _, _, _, batched_hist = _race(SOAP, BatchedSOAP, PRECOND_FREQ)
    _, _, _, _, control_hist = _race(SOAP, UlpPerturbedSOAP, PRECOND_FREQ)

    # Before the first refresh (it fires inside step PRECOND_FREQ, so
    # step PRECOND_FREQ + 1 is the first one that can see it) the two
    # implementations must agree to float64 rounding.
    pre = max(d for d, _ in batched_hist[:PRECOND_FREQ])
    assert pre < NO_REFRESH_ATOL, f"diverged before any refresh: {pre:.3e}"

    batched, where = batched_hist[-1]
    control = control_hist[-1][0]
    assert batched < REFRESH_ABS_CAP, f"{batched:.3e} at {where}"
    assert batched <= REFRESH_CONTROL_FACTOR * control, (
        f"batched vs reference {batched:.3e} exceeds "
        f"{REFRESH_CONTROL_FACTOR}x the reference's own one-ulp "
        f"sensitivity {control:.3e}"
    )


def test_refresh_matches_reference_on_identical_state():
    """The batched refresh, on bitwise-identical well-conditioned input.

    Isolated from the trajectory: same `GG`, same `Q`, same
    `exp_avg_sq`. The eigenvalue sort must land on the same
    permutation, `exp_avg_sq` must be re-indexed identically, and the
    new bases must agree to float32 rounding -- the refresh is computed
    in float32 no matter what dtype the state is.
    """
    torch.manual_seed(0)
    K, m, n = 4, 12, 5
    params = [torch.nn.Parameter(torch.randn(m, n)) for _ in range(K)]
    ref = SOAP(params, precondition_frequency=5)
    bat = BatchedSOAP(params, precondition_frequency=5)

    # A well-conditioned accumulator: this test is about kernel
    # agreement, not about degeneracy.
    A = torch.randn(K, m, m)
    GG0 = torch.bmm(A, A.transpose(1, 2)) + m * torch.eye(m)
    B = torch.randn(K, n, n)
    GG1 = torch.bmm(B, B.transpose(1, 2)) + n * torch.eye(n)
    GG0 = 0.5 * (GG0 + GG0.transpose(1, 2))
    GG1 = 0.5 * (GG1 + GG1.transpose(1, 2))
    Q0 = torch.linalg.qr(torch.randn(K, m, m))[0]
    Q1 = torch.linalg.qr(torch.randn(K, n, n))[0]
    eas = torch.rand(K, m, n)

    from fim.fisher_pruning.batched_soap import _Bucket

    bucket = _Bucket(params)
    bucket.state.update(
        step=5,
        precondition_frequency=5,
        shampoo_beta=0.95,
        GG=[GG0.clone(), GG1.clone()],
        Q=[Q0.clone(), Q1.clone()],
        exp_avg=torch.zeros(K, m, n),
        exp_avg_sq=eas.clone(),
    )
    b_final = bat._b_get_orthogonal_matrix_QR(bucket, 10000, False)
    b_eas = bucket.state["exp_avg_sq"]

    for k in range(K):
        state = {
            "step": 5,
            "precondition_frequency": 5,
            "shampoo_beta": 0.95,
            "GG": [GG0[k].clone(), GG1[k].clone()],
            "Q": [Q0[k].clone(), Q1[k].clone()],
            "exp_avg": torch.zeros(m, n),
            "exp_avg_sq": eas[k].clone(),
        }
        r_final = ref.get_orthogonal_matrix_QR(state, 10000, False)
        assert torch.equal(
            state["exp_avg_sq"], b_eas[k]
        ), f"exp_avg_sq re-indexed differently for member {k}"
        for i, (rq, bq) in enumerate(zip(r_final, b_final)):
            d = (rq - bq[k]).abs().max().item()
            # float32 eps is 1.2e-7; the refresh runs in float32.
            assert d < 1e-6, f"member {k} Q[{i}] differs by {d:.3e}"


def test_initial_eigenbasis_matches_reference():
    """`get_orthogonal_matrix` (the eigh path), batched vs unbatched."""
    torch.manual_seed(3)
    K, n = 5, 9
    A = torch.randn(K, n, n)
    GG = torch.bmm(A, A.transpose(1, 2)) + n * torch.eye(n)
    GG = 0.5 * (GG + GG.transpose(1, 2))
    params = [torch.nn.Parameter(torch.randn(n, n)) for _ in range(K)]
    ref = SOAP(params)
    bat = BatchedSOAP(params)

    b_final = bat._b_get_orthogonal_matrix([GG.clone(), None])
    assert b_final[1] is None
    for k in range(K):
        r_final = ref.get_orthogonal_matrix([GG[k].clone(), []])
        assert len(r_final[1]) == 0
        d = (r_final[0] - b_final[0][k]).abs().max().item()
        assert d < 1e-12, f"member {k} eigenbasis differs by {d:.3e}"


def test_state_dict_round_trip():
    """Save, rebuild, load, continue -- same as never stopping."""
    batches = _make_batches(STEPS)
    half = STEPS // 2

    plain_model = _make_model()
    plain_opt = _build(BatchedSOAP, plain_model)
    _train(plain_model, plain_opt, batches)

    rt_model = _make_model()
    rt_opt = _build(BatchedSOAP, rt_model)
    _train(rt_model, rt_opt, batches[:half])
    sd = copy.deepcopy(rt_opt.state_dict())

    fresh_opt = _build(BatchedSOAP, rt_model)
    fresh_opt.load_state_dict(sd)
    assert fresh_opt.bucket_report() == rt_opt.bucket_report()
    _train(rt_model, fresh_opt, batches[half:])

    worst, where = _max_param_diff(plain_model, rt_model)
    assert worst == 0.0, f"round trip changed the trajectory: {worst:.3e} at {where}"


def test_state_dict_is_unstacked_per_parameter():
    """Every state tensor carries its parameter's shape, not the slab's."""
    model = _make_model()
    opt = _build(BatchedSOAP, model)
    _train(model, opt, _make_batches(6))
    assert opt.bucket_report(), "nothing was batched, the test proves nothing"

    params = [p for g in opt.param_groups for p in g["params"]]
    sd = opt.state_dict()
    assert sd["state"], "empty state_dict"
    for i, entry in sd["state"].items():
        p = params[i]
        assert entry["exp_avg"].shape == p.shape
        assert entry["exp_avg_sq"].shape == p.shape
        # Cloned, not a view onto the slab.
        assert entry["exp_avg"].untyped_storage().nbytes() == (
            p.numel() * p.element_size()
        )
        assert isinstance(entry["step"], int)
        for gg in entry["GG"]:
            if len(gg) == 0:
                continue
            assert gg.dim() == 2 and gg.shape[0] == gg.shape[1]
        for q in entry["Q"]:
            if len(q) == 0:
                continue
            assert q.dim() == 2


def test_reference_can_load_a_batched_checkpoint():
    """The state_dict is interchangeable with the reference's."""
    bat_model = _make_model()
    bat_opt = _build(BatchedSOAP, bat_model)
    _train(bat_model, bat_opt, _make_batches(6))

    ref_model = _make_model()
    ref_opt = _build(SOAP, ref_model)
    ref_opt.load_state_dict(copy.deepcopy(bat_opt.state_dict()))

    named = dict(bat_model.named_parameters())
    for name, p in ref_model.named_parameters():
        rs, bs = ref_opt.state[p], bat_opt.state[named[name]]
        assert rs["step"] == bs["step"]
        assert torch.equal(rs["exp_avg"], bs["exp_avg"])
        assert torch.equal(rs["exp_avg_sq"], bs["exp_avg_sq"])


def test_bucketing_groups_identical_shapes():
    model = _make_model()
    opt = _build(BatchedSOAP, model)
    _train(model, opt, _make_batches(2))

    counts = {shape: k for shape, _dt, _dev, k in opt.bucket_report()}
    assert counts, "nothing was bucketed"

    n_embd = MODEL_CFG["n_embd"]
    n_layer = MODEL_CFG["n_layer"]
    # nn.Linear stores [out_features, in_features].
    for shape in (
        (3 * n_embd, n_embd),
        (n_embd, n_embd),
        (4 * n_embd, n_embd),
        (n_embd, 4 * n_embd),
    ):
        assert counts.get(shape) == n_layer, f"{shape} -> {counts}"

    assert all(k > 1 for k in counts.values()), "a bucket has a single member"

    # No parameter is in two buckets, and the ids actually cover them.
    seen = set()
    for buckets in opt._buckets.values():
        for b in buckets:
            for p in b.params:
                assert id(p) not in seen
                seen.add(id(p))
                assert id(p) in opt._batched_ids
    assert len(seen) == sum(counts.values())


def test_unique_shape_parameter_still_trains():
    """A shape that occurs once takes the reference path and moves."""
    model = _make_model()
    opt = _build(BatchedSOAP, model)

    # Position embeddings are the only [block_size, n_embd] tensor.
    wpe = model.transformer.wpe.weight
    assert wpe.shape == (MODEL_CFG["block_size"], MODEL_CFG["n_embd"])
    before = wpe.detach().clone()

    _train(model, opt, _make_batches(6))

    bucketed = {shape for shape, _dt, _dev, _k in opt.bucket_report()}
    assert tuple(wpe.shape) not in bucketed
    assert id(wpe) not in opt._batched_ids
    assert (wpe.detach() - before).abs().max().item() > 0
    assert opt.state[wpe]["step"] > 0
    assert torch.isfinite(wpe).all()


def test_unique_shape_parameter_matches_reference():
    """The unbatched fallback is the reference, exactly."""
    model_a, _, model_b, _, _ = _race(SOAP, BatchedSOAP, NEVER_REFRESH)
    a = model_a.transformer.wpe.weight.detach()
    b = model_b.transformer.wpe.weight.detach()
    d = (a - b).abs().max().item()
    assert d < NO_REFRESH_ATOL, f"unbatched fallback diverged {d:.3e}"


def test_bucket_dissolves_when_a_member_loses_its_gradient():
    """Losing a gradient must not silently desynchronize a bucket."""
    model = _make_model()
    opt = _build(BatchedSOAP, model)
    batches = _make_batches(8)
    _train(model, opt, batches[:4])
    assert opt.bucket_report()

    victim = model.transformer.h[0].attn.c_attn.weight
    steps_before = opt.state[victim]["step"]
    for x, y in batches[4:]:
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        victim.grad = None
        opt.step()

    shapes = {shape for shape, _dt, _dev, _k in opt.bucket_report()}
    assert tuple(victim.shape) not in shapes
    assert opt.state[victim]["step"] == steps_before
    # The rest of the former bucket kept stepping.
    sibling = model.transformer.h[1].attn.c_attn.weight
    assert opt.state[sibling]["step"] > steps_before
    for p in model.parameters():
        if p.requires_grad and p in opt.state:
            assert torch.isfinite(p).all()


def test_dissolution_matches_reference_exactly():
    """When a bucket member loses its gradient the survivors must take
    exactly ONE update on that step, with reference step counters.

    Gradients are injected directly so both optimizers see bitwise
    identical values and the comparison tests the algorithm, not
    autograd nondeterminism.
    """
    import torch

    from fim.fisher_pruning.batched_soap import BatchedSOAP
    from fim.fisher_pruning.soap import SOAP

    torch.manual_seed(0)
    init = [torch.randn(6, 5, dtype=torch.float64) for _ in range(4)]

    def run(cls):
        torch.manual_seed(0)
        ps = [torch.nn.Parameter(t.clone()) for t in init]
        opt = cls(
            ps,
            lr=1e-2,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            precondition_frequency=1000,
        )
        g = torch.Generator().manual_seed(7)
        for step in range(10):
            for i, p in enumerate(ps):
                # member 1 stops producing gradients from step 5
                p.grad = (
                    None
                    if (i == 1 and step >= 5)
                    else torch.randn(6, 5, generator=g, dtype=torch.float64)
                )
            opt.step()
        counters = [int(opt.state[p].get("step", 0)) for p in ps]
        return [p.detach().clone() for p in ps], counters

    ref_p, ref_c = run(SOAP)
    bat_p, bat_c = run(BatchedSOAP)
    assert bat_c == ref_c, f"step counters {bat_c} != reference {ref_c}"
    for i, (a, b) in enumerate(zip(ref_p, bat_p)):
        rel = ((a - b).norm() / a.norm()).item()
        assert rel < 1e-10, f"param {i} diverged by {rel:.3e}"
