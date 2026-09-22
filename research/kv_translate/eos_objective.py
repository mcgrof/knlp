# SPDX-License-Identifier: GPL-2.0
"""The answer objective, with the terminal term separable from the denominator.

The previous intervention removed the end-of-sequence label and, by doing so,
also removed a position from the mean it was divided by. The answer terms
themselves never changed, so the off arm trained them at a larger effective
step -- between 1.125 and 2 times, depending on answer length. Whatever that
run showed, it was not an ablation of terminal supervision.

The repair is to stop expressing the arm in the label length. Both arms build
the same labels, feed the same teacher-forced sequence, and divide by the same
count of supervised positions. The arm is a mask over which terms contribute.
Zeroing a term is not the same as deleting the position it occupied, and only
the first leaves everything else where it was.

Two masks are needed, not one. ``valid`` marks real supervised positions and
is identical across arms; it is the denominator. ``contribute`` marks which of
those positions add to the numerator and is where the arms differ. Collapsing
them into one mask reintroduces exactly the bug this file exists to fix,
because the thing that changed last time was the denominator.
"""

from __future__ import annotations

import torch

ARMS = ("on", "off")


def build_labels(tok, example, eos_id):
    """Query ids and the full answer-plus-terminal labels, arm-independent.

    The terminal token is always present. An arm that wants it unsupervised
    masks it; it does not shorten this.
    """
    q = tok(example.query, return_tensors="pt", add_special_tokens=False).input_ids
    a = tok(example.answer, return_tensors="pt", add_special_tokens=False).input_ids
    labels = a[0].tolist() + [eos_id]
    terminal = [0] * (len(labels) - 1) + [1]
    return q, labels, terminal


def masks_for_arm(terminal, arm, pad=None):
    """``(valid, contribute)`` for one arm.

    ``valid`` is every real supervised position, terminal included, and does
    not depend on the arm. ``contribute`` drops the terminal position in the
    off arm and nothing else.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    t = torch.as_tensor(terminal, dtype=torch.float32)
    valid = (
        torch.ones_like(t) if pad is None else torch.as_tensor(pad, dtype=torch.float32)
    )
    contribute = valid.clone()
    if arm == "off":
        contribute = contribute * (1.0 - t)
    return valid, contribute


def answer_logits(out, query_len):
    """The slice of ``out`` whose position i predicts label i.

    The first answer token is predicted from the query's last position, so the
    slice starts one before the labels do. Kept here rather than inline in the
    trainer so the off-by-one can be tested instead of asserted: an offset that
    is wrong by one trains a model to continue an answer it never learns to
    start, and the loss still looks reasonable while it does so.
    """
    return out[:, query_len - 1 :, :]


def answer_loss_from_logits(pred, labels, valid, contribute):
    """Per-position cross entropy, masked in the numerator only.

    ``pred`` holds the logits that predict ``labels``: position i of ``pred``
    predicts ``labels[i]``, so the caller is responsible for the one-step
    offset that makes the first answer token come from the query's last
    position. Getting that wrong trains a model to continue an answer it never
    learns to begin.
    """
    v = pred.shape[-1]
    ce = torch.nn.functional.cross_entropy(
        pred.reshape(-1, v).float(), labels.reshape(-1), reduction="none"
    )
    c = contribute.reshape(-1).to(ce.device, ce.dtype)
    d = valid.reshape(-1).to(ce.device, ce.dtype).sum().clamp(min=1.0)
    return (ce * c).sum() / d


def legacy_answer_loss_from_logits(pred, labels):
    """What the previous off arm actually computed, kept for provenance.

    A plain mean over whatever labels were handed in. With the terminal label
    deleted upstream, the denominator shrank with it. This is not used for new
    work; it exists so the historical arm can be reproduced and compared
    rather than described from memory.
    """
    v = pred.shape[-1]
    return torch.nn.functional.cross_entropy(
        pred.reshape(-1, v).float(), labels.reshape(-1)
    )
