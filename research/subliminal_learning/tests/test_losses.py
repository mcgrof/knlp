# SPDX-License-Identifier: MIT
"""Loss, gradient and first-update parity with the reference script."""

import copy
import importlib.machinery
import importlib.util
import os

import pytest
import torch

from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.models import PaperMLP
from research.subliminal_learning.numerical_kl import EXACT_CANCEL_KL, LEGACY_KL
from research.subliminal_learning.run import student_loss, teacher_loss

UPSTREAM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "upstream",
    "run_mnist_experiment.py.orig",
)


@pytest.fixture(scope="module")
def up():
    for mod in ("pandas", "tqdm", "matplotlib", "torchvision"):
        pytest.importorskip(mod)
    loader = importlib.machinery.SourceFileLoader("sl_upstream", UPSTREAM)
    spec = importlib.util.spec_from_loader("sl_upstream", loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # module scope only seeds and defines
    m.DEVICE = "cpu"
    return m


def paired_models(up, n_models, seed=7):
    """Reference and ported models with identical weights, in float64.

    The first Adam step is nearly sign(g) * lr, so an element whose
    gradient is comparable to Adam's epsilon amplifies float32 summation
    order differences between einsum and bmm. Double precision removes
    that noise and leaves any real reduction or update mismatch visible.
    """
    torch.manual_seed(seed)
    ref = up.MultiClassifier(n_models, [784, 256, 256, 13])
    ours = PaperMLP(n_models, 13)
    with torch.no_grad():
        for dst, src in zip(
            ours.common_linears(), [ref.net[0], ref.net[2], ref.net[4]]
        ):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    return ref.double(), ours.double()


def adam(model):
    return torch.optim.Adam(model.parameters(), lr=3e-4)


def ours_params(m):
    return [t for lin in m.common_linears() for t in (lin.weight, lin.bias)]


def up_params(m):
    return [t for i in (0, 2, 4) for t in (m.net[i].weight, m.net[i].bias)]


def assert_close_params(ours, ref):
    for a, b in zip(ours_params(ours), up_params(ref)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


def test_teacher_first_update_matches_upstream(up):
    M, B = 4, 300
    ref, ours = paired_models(up, M)
    x = torch.randn(M, B, 1, 28, 28, dtype=torch.double)
    y = torch.randint(0, 10, (B,))
    up.train(ref, x, y, epochs=1)  # B <= 1024: exactly one Adam step
    # The reference shuffles each model's rows; one full batch makes the
    # mean loss permutation invariant, so the same rows in order suffice.
    opt = adam(ours)
    loss, _ = teacher_loss(ours(x.flatten(2)), y.expand(M, B), "upstream_batched_v1", M)
    ref_loss = up.ce_first10(copy.deepcopy(paired_models(up, M)[0])(x), y.expand(M, B))
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-7)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert_close_params(ours, ref)


@pytest.mark.parametrize("aux", [True, False])
def test_student_first_update_matches_upstream(up, aux):
    M, B = 3, 257
    teacher_ref, teacher = paired_models(up, M)
    torch.manual_seed(11)
    with torch.no_grad():  # a different, "trained" teacher
        for p_ref, p in zip(up_params(teacher_ref), ours_params(teacher)):
            p_ref.add_(0.05 * torch.randn_like(p_ref))
            p.copy_(p_ref)
    student_ref, student = paired_models(up, M, seed=12)
    idx = up.GHOST_IDX if aux else up.ALL_IDX
    src = torch.rand(M, B, 1, 28, 28, dtype=torch.double) * 2 - 1
    up.distill(student_ref, teacher_ref, idx, src, epochs=1)
    opt = adam(student)
    x = src.flatten(2)
    with torch.no_grad():
        tgt = teacher(x)[..., idx]
    loss, _, _ = student_loss(student(x)[..., idx], tgt, "upstream_batched_v1", M)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert_close_params(student, student_ref)


def test_packing_does_not_change_updates():
    """Two packs of 2 with logical cohort 4 == one pack of 4."""
    torch.manual_seed(0)
    for protocol in ("upstream_batched_v1", "per_example_v1"):
        for loss_mode in (LEGACY_KL, EXACT_CANCEL_KL):
            _assert_packing_update_parity(protocol, loss_mode)


def _assert_packing_update_parity(protocol, loss_mode):
    """One four-model step must equal two physical packs of two."""
    full = PaperMLP(4, 13)
    for lin in full.common_linears():
        torch.nn.init.normal_(lin.weight, 0, 0.05)
    x = torch.randn(4, 64, 784)
    teacher = copy.deepcopy(full)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.1 * torch.randn_like(p))
    halves = []
    for rows in (slice(0, 2), slice(2, 4)):
        h = PaperMLP(2, 13)
        t = PaperMLP(2, 13)
        h.load_state_dict({k: v[rows].clone() for k, v in full.state_dict().items()})
        t.load_state_dict({k: v[rows].clone() for k, v in teacher.state_dict().items()})
        halves.append((h, t, x[rows]))

    def step(model, tch, xb):
        opt = adam(model)
        with torch.no_grad():
            tgt = tch(xb)[..., 10:]
        loss, _, _ = student_loss(model(xb)[..., 10:], tgt, protocol, 4, loss_mode)
        opt.zero_grad()
        loss.backward()
        opt.step()

    step(full, teacher, x)
    for h, t, xb in halves:
        step(h, t, xb)
    for k, v in full.state_dict().items():
        joined = torch.cat([halves[0][0].state_dict()[k], halves[1][0].state_dict()[k]])
        torch.testing.assert_close(v, joined, rtol=1e-6, atol=1e-8)


def test_aux_softmax_is_over_aux_logits_only():
    s = torch.randn(2, 5, 13)
    t = torch.randn(2, 5, 13)
    _, kl_sum, kl_mean = student_loss(s[..., 10:], t[..., 10:], "per_example_v1", 2)
    p = torch.softmax(t[..., 10:], -1)
    manual = (p * (p.log() - torch.log_softmax(s[..., 10:], -1))).sum(-1)
    torch.testing.assert_close(kl_sum, manual.sum(1))
    torch.testing.assert_close(kl_mean, manual.mean(1))
    # Slicing a 13-way softmax is a different (wrong) target.
    assert not torch.allclose(p, torch.softmax(t, -1)[..., 10:])


def test_upstream_reduction_is_sum_over_images_divided_by_cohort():
    s, t = torch.randn(3, 7, 3), torch.randn(3, 7, 3)
    loss, kl_sum, _ = student_loss(s, t, "upstream_batched_v1", 3)
    ref = torch.nn.functional.kl_div(
        torch.log_softmax(s, -1), torch.softmax(t, -1), reduction="batchmean"
    )
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(loss, kl_sum.sum() / 3)


def test_recipe_defaults_are_the_reference_recipe():
    r = Recipe()
    assert (r.teacher_epochs, r.student_epochs, r.batch_size) == (5, 5, 1024)
    assert r.lr == 3e-4 and r.adam_betas == (0.9, 0.999) and r.adam_eps == 1e-8
    assert r.weight_decay == 0.0 and r.total_outputs == 13
    assert sum(p.numel() for p in PaperMLP(1, 13).parameters()) == 270093
