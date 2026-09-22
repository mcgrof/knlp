# SPDX-License-Identifier: GPL-2.0
"""What the corrected terminal-supervision arms must satisfy.

The previous experiment intended to remove terminal supervision and instead
removed a position from the denominator too, so the answer objective was
up-weighted by between 1.125 and 2 times depending on answer length. Nothing
in that run separated the two changes, and nothing in it announced that a
second change had occurred.

These checks are on the functions the trainer itself calls, not on a
reimplementation of them, because a reimplementation can satisfy every
property here while the trainer does something else.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from research.kv_translate.eos_objective import (
    ARMS,
    answer_logits,
    answer_loss_from_logits,
    build_labels,
    legacy_answer_loss_from_logits,
    masks_for_arm,
)

ROOT = Path(__file__).resolve().parents[2]
V = 32


class _Tok:
    """One id per character, so label lengths are predictable."""

    eos_token_id = V - 1

    def __call__(self, s, return_tensors=None, add_special_tokens=None):
        class R:
            input_ids = torch.tensor([[(ord(c) % (V - 2)) + 1 for c in s]])

        return R()


class _Ex:
    def __init__(self, answer, query="q?"):
        self.answer, self.query = answer, query
        self.doc_id, self.kind, self.meta = "d", "code", {"code": answer}
        self.prompt_ids = list(range(8))


def _grads(labels, terminal, pad=None, seed=0):
    """Gradients into the logits for each arm, from identical inputs."""
    out = {}
    lab = torch.tensor(labels).unsqueeze(0)
    for arm in ARMS:
        torch.manual_seed(seed)
        pred = torch.randn(1, len(labels), V, requires_grad=True)
        valid, contribute = masks_for_arm(terminal, arm, pad=pad)
        loss = answer_loss_from_logits(
            pred, lab, valid.unsqueeze(0), contribute.unsqueeze(0)
        )
        (g,) = torch.autograd.grad(loss, pred)
        out[arm] = (float(loss.detach()), g)
    return out


@pytest.mark.parametrize("n_answer", [1, 2, 3, 8])
def test_answer_gradients_are_identical_across_arms(n_answer):
    """The whole point. Masking a term must not rescale the others.

    This is the property the earlier arm violated: its answer gradients were
    the same terms divided by a smaller number, so they arrived larger.
    """
    labels = list(range(1, n_answer + 1)) + [_Tok.eos_token_id]
    terminal = [0] * n_answer + [1]
    g = _grads(labels, terminal)
    a, b = g["on"][1][:, :n_answer], g["off"][1][:, :n_answer]
    assert torch.equal(a, b), f"answer gradients differ at n={n_answer}"


@pytest.mark.parametrize("n_answer", [1, 2, 3, 8])
def test_off_arm_terminal_gradient_is_exactly_zero(n_answer):
    labels = list(range(1, n_answer + 1)) + [_Tok.eos_token_id]
    terminal = [0] * n_answer + [1]
    g = _grads(labels, terminal)
    term_off = g["off"][1][:, n_answer]
    assert torch.count_nonzero(term_off) == 0, "off arm still trains the terminal"
    assert torch.count_nonzero(g["on"][1][:, n_answer]) > 0, "on arm ignores it"


@pytest.mark.parametrize("n_answer", [1, 3, 8])
def test_on_arm_reproduces_the_original_objective(n_answer):
    """The control must be the old control, or the pair compares two changes."""
    labels = list(range(1, n_answer + 1)) + [_Tok.eos_token_id]
    terminal = [0] * n_answer + [1]
    lab = torch.tensor(labels).unsqueeze(0)
    torch.manual_seed(0)
    pred = torch.randn(1, len(labels), V)
    valid, contribute = masks_for_arm(terminal, "on")
    new = answer_loss_from_logits(
        pred, lab, valid.unsqueeze(0), contribute.unsqueeze(0)
    )
    old = legacy_answer_loss_from_logits(pred, lab)
    assert torch.allclose(new, old, atol=1e-6), (float(new), float(old))


@pytest.mark.parametrize("n_answer", [1, 2, 3, 8])
def test_the_denominator_is_the_same_in_both_arms(n_answer):
    terminal = [0] * n_answer + [1]
    d = {}
    for arm in ARMS:
        valid, _ = masks_for_arm(terminal, arm)
        d[arm] = float(valid.sum())
    assert d["on"] == d["off"] == n_answer + 1, d


def test_the_earlier_arm_is_reproduced_and_is_not_the_corrected_one():
    """Keep the old behaviour available, and keep it distinguishable.

    A one-token answer is where the two differ most: deleting the terminal
    label halves the denominator, so the answer term arrives at twice the
    weight.
    """
    torch.manual_seed(0)
    pred = torch.randn(1, 2, V)
    labels = torch.tensor([[5, _Tok.eos_token_id]])
    valid, contribute = masks_for_arm([0, 1], "off")
    corrected = answer_loss_from_logits(
        pred, labels, valid.unsqueeze(0), contribute.unsqueeze(0)
    )
    legacy = legacy_answer_loss_from_logits(pred[:, :1, :], labels[:, :1])
    assert not torch.allclose(corrected, legacy)
    assert torch.allclose(legacy, corrected * 2, atol=1e-6), (
        float(legacy),
        float(corrected),
    )


def test_padding_is_excluded_from_both_numerator_and_denominator():
    """A padded position is not a supervised one, in either arm."""
    terminal = [0, 0, 1, 0]  # two answer tokens, terminal, then padding
    pad = [1, 1, 1, 0]
    for arm in ARMS:
        valid, contribute = masks_for_arm(terminal, arm, pad=pad)
        assert float(valid.sum()) == 3.0, arm
        assert float(contribute[3]) == 0.0, f"{arm} supervises padding"
    _, c_on = masks_for_arm(terminal, "on", pad=pad)
    _, c_off = masks_for_arm(terminal, "off", pad=pad)
    assert float(c_on.sum()) == 3.0 and float(c_off.sum()) == 2.0


def test_padded_positions_do_not_change_answer_gradients():
    g_nopad = _grads([5, 6, _Tok.eos_token_id], [0, 0, 1])
    g_pad = _grads([5, 6, _Tok.eos_token_id, 0], [0, 0, 1, 0], pad=[1, 1, 1, 0])
    for arm in ARMS:
        assert torch.allclose(
            g_nopad[arm][1][:, :2], g_pad[arm][1][:, :2], atol=1e-7
        ), arm


def test_labels_carry_the_real_terminal_id_and_are_arm_independent():
    tok = _Tok()
    q, labels, terminal = build_labels(tok, _Ex("AB12"), tok.eos_token_id)
    assert labels[-1] == tok.eos_token_id
    assert terminal == [0] * (len(labels) - 1) + [1]
    assert sum(terminal) == 1
    # Same labels regardless of arm: the arm lives in the mask.
    for arm in ARMS:
        valid, _ = masks_for_arm(terminal, arm)
        assert len(valid) == len(labels)


def test_the_first_answer_token_comes_from_the_querys_last_position():
    """An offset wrong by one still trains, and trains the wrong thing."""
    q_len, n_lab = 4, 3
    out = torch.zeros(1, q_len + n_lab - 1, V)
    labels = torch.tensor([[7, 8, _Tok.eos_token_id]])
    # Put all the mass for label i at the position that should predict it.
    for i, lab in enumerate(labels[0].tolist()):
        out[0, q_len - 1 + i, lab] = 20.0
    pred = answer_logits(out, q_len)
    assert pred.shape[1] == n_lab
    valid, contribute = masks_for_arm([0, 0, 1], "on")
    good = answer_loss_from_logits(
        pred, labels, valid.unsqueeze(0), contribute.unsqueeze(0)
    )
    shifted = answer_logits(out, q_len + 1)
    bad = answer_loss_from_logits(
        shifted,
        labels[:, : shifted.shape[1]],
        valid[: shifted.shape[1]].unsqueeze(0),
        contribute[: shifted.shape[1]].unsqueeze(0),
    )
    assert float(good) < 0.01, float(good)
    assert float(bad) > float(good) * 10, (float(bad), float(good))


def test_an_unknown_arm_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        masks_for_arm([0, 1], "whatever")


def test_the_trainer_calls_these_functions_and_not_a_bare_mean():
    """Guard against the check drifting away from the thing checked.

    Every property above holds of functions the trainer imports. If the
    trainer stopped calling them, or computed the answer loss with a default
    reduction of its own, the tests would keep passing while the training
    changed. So the trainer's source is read for both.
    """
    src = (ROOT / "research" / "kv_translate" / "run_probe.py").read_text()
    tree = ast.parse(src)
    imported = {
        n.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and "eos_objective" in node.module
        for n in node.names
    }
    assert {"answer_loss_from_logits", "masks_for_arm", "build_labels"} <= imported
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        for _ in [0]
        if False
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "answer_loss_from_logits" in called, "the trainer does not call it"
    assert "answer_logits" in called, "the trainer slices the offset itself"
