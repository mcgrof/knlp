# SPDX-License-Identifier: GPL-2.0

import torch

from research.kv_translate.train_probe_distillation import (
    cyclic_schedule,
    mechanics_decision,
    summarize_task_rows,
    tensor_digest,
)


def test_cyclic_schedule_is_deterministic_and_covers_each_pass():
    first = cyclic_schedule(5, 12, 17)
    second = cyclic_schedule(5, 12, 17)

    assert first == second
    assert sorted(first[:5]) == list(range(5))
    assert sorted(first[5:10]) == list(range(5))


def test_tensor_digest_covers_names_shapes_dtypes_and_values():
    base = {"a": torch.tensor([1.0, 2.0]), "b": torch.zeros(2, 2)}
    reordered = {"b": base["b"].clone(), "a": base["a"].clone()}
    changed = {"a": torch.tensor([1.0, 3.0]), "b": torch.zeros(2, 2)}

    assert tensor_digest(base) == tensor_digest(reordered)
    assert tensor_digest(base) != tensor_digest(changed)


def test_task_summary_keeps_kind_and_condition_separate():
    rows = [
        {"kind": "retrieval", "condition": "native", "primary": 1.0},
        {"kind": "retrieval", "condition": "native", "primary": 0.0},
        {"kind": "cloze", "condition": "native", "primary": 0.5},
    ]

    summary = summarize_task_rows(rows)

    assert summary["retrieval"]["native"] == {"mean": 0.5, "n": 2}
    assert summary["cloze"]["native"] == {"mean": 0.5, "n": 1}


def test_mechanics_gate_requires_change_improvement_and_identity():
    training = {"unrelated": {"parameters_changed": True}}
    diagnostics = {
        "unrelated": {
            "affine": {"mean_kl": 0.5},
            "unrelated": {"mean_kl": 0.4},
        }
    }

    assert mechanics_decision(training, diagnostics, [])["passes"] is True
    assert mechanics_decision(training, diagnostics, [{"item": 1}])["passes"] is False
    training["unrelated"]["parameters_changed"] = False
    assert mechanics_decision(training, diagnostics, [])["passes"] is False
