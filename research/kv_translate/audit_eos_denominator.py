# SPDX-License-Identifier: GPL-2.0
"""Check what the termination ablation actually changed in its loss.

The ablation claimed to remove the end-of-sequence label and nothing else. A
label is not the only thing that moves when a label is dropped: the answer
loss is a mean over supervised positions, so removing one position also
shrinks the denominator that the remaining positions are divided by. If that
happened, the off arm trained the answer tokens at a larger effective step
than the on arm and the two runs differ by more than termination.

This is checked on fixed logits rather than by rerunning anything. The
question is arithmetic and does not need a GPU, a model or a trained map.

Run: python research/kv_translate/audit_eos_denominator.py
"""

from __future__ import annotations

import json
import sys

import torch


def answer_loss_as_run(logits, labels):
    """The reduction the trainer actually used: mean over supervised positions."""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1)
    )


def answer_loss_denominator_preserved(logits, labels, eos_id, drop_eos):
    """What an ablation of termination alone would have had to do.

    Keep every position, including the end position, and keep dividing by the
    same count. Zero only the end position's contribution. Then the off arm
    differs from the on arm by that one term and by nothing else.
    """
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
    )
    if drop_eos:
        ce = ce.clone()
        ce[labels.reshape(-1) == eos_id] = 0.0
    return ce.sum() / ce.numel()


def main() -> int:
    torch.manual_seed(0)
    vocab, eos_id = 64, 63
    report = {"cases": []}

    for n_answer in (1, 2, 3, 5, 8):
        # Fixed logits, so every number below is reproducible arithmetic.
        logits_on = torch.randn(1, n_answer + 1, vocab, generator=None)
        labels_on = torch.cat(
            [torch.arange(n_answer).unsqueeze(0), torch.tensor([[eos_id]])], dim=1
        )
        # The off arm as it was actually run: one fewer position, and the mean
        # is taken over that smaller count.
        logits_off = logits_on[:, :n_answer, :]
        labels_off = labels_on[:, :n_answer]

        on = float(answer_loss_as_run(logits_on, labels_on))
        off_as_run = float(answer_loss_as_run(logits_off, labels_off))
        off_preserved = float(
            answer_loss_denominator_preserved(logits_on, labels_on, eos_id, True)
        )

        # The answer terms are identical in both arms. What differs is what
        # they are divided by.
        answer_sum = float(
            torch.nn.functional.cross_entropy(
                logits_off.reshape(-1, vocab).float(),
                labels_off.reshape(-1),
                reduction="sum",
            )
        )
        report["cases"].append(
            {
                "answer_tokens": n_answer,
                "supervised_positions_on": n_answer + 1,
                "supervised_positions_off_as_run": n_answer,
                "denominator_on": n_answer + 1,
                "denominator_off_as_run": n_answer,
                "denominator_off_if_preserved": n_answer + 1,
                "answer_term_sum": answer_sum,
                "loss_on": on,
                "loss_off_as_run": off_as_run,
                "loss_off_denominator_preserved": off_preserved,
                "answer_gradient_scale_off_over_on": (n_answer + 1) / n_answer,
                "as_run_equals_preserved": abs(off_as_run - off_preserved) < 1e-6,
            }
        )

    inflations = [c["answer_gradient_scale_off_over_on"] for c in report["cases"]]
    report["verdict"] = {
        "denominator_preserved": all(
            c["as_run_equals_preserved"] for c in report["cases"]
        ),
        "answer_gradient_inflation_range": [min(inflations), max(inflations)],
        "finding": (
            "The trainer reduces the answer loss with cross_entropy's default "
            "mean, so the denominator is the number of supervised positions. "
            "Dropping the end label removed a position and therefore also "
            "shrank that denominator. The answer terms themselves are "
            "unchanged, so the off arm applied the same answer gradients "
            "divided by a smaller number: for a one-token answer the answer "
            "loss carries twice the weight it carried in the on arm, and for "
            "an eight-token answer 1.125 times. The off arm is therefore not "
            "an ablation of termination alone. It is termination removed and "
            "the answer objective simultaneously up-weighted by a "
            "length-dependent factor."
        ),
        "consequence": (
            "The off arm's divergence cannot be attributed to the absence of "
            "termination supervision. A larger effective step on the answer "
            "objective is an equally consistent explanation and the run does "
            "not separate them. The reported collapse stands as the behaviour "
            "of that recorded recipe; the causal claim about the end token "
            "does not."
        ),
        "what_a_clean_ablation_requires": (
            "Keep every supervised position and the same denominator, and "
            "zero only the end position's contribution, as "
            "answer_loss_denominator_preserved does here."
        ),
    }
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
