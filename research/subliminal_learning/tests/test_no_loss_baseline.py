# SPDX-License-Identifier: MIT
import json

import torch

from research.subliminal_learning import no_loss_baseline
from research.subliminal_learning.data import make_noise, tensor_digest
from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.run import accuracy, state_digest


def test_no_loss_baseline_keeps_state_and_digit_scoring_exact(tmp_path, monkeypatch):
    recipe = Recipe(
        arch="residual_mlp",
        protocol="per_example_v1",
        seeds=[91, 92],
        teacher_epochs=1,
        student_epochs=1,
        batch_size=8,
        width=16,
        blocks=2,
        target_scale=0.0,
        noise_samples=16,
        heldout_noise=8,
        device="cpu",
        output_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="null",
    )
    keys = recipe.model_keys
    w0 = build(recipe, len(keys))
    init_model(w0, keys, "init")
    test_x = torch.linspace(-1, 1, 12 * 784).reshape(12, 784)
    test_y = torch.arange(12) % 10
    initial_logits = no_loss_baseline._digit_logits(w0, test_x)
    initial = no_loss_baseline._digit_summary(initial_logits, test_y)
    archived_accuracy = accuracy(w0, test_x, test_y).tolist()
    assert initial["correct_counts"] == [
        round(value * test_y.numel()) for value in archived_accuracy
    ]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "recipe.json").write_text(
        json.dumps({"recipe": recipe.to_json(), "git": {"commit": "test"}})
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "0": {
                    "keys": keys,
                    "results": {
                        "w0_digest": [state_digest(w0, m) for m in range(2)],
                        "noise_digest": [
                            tensor_digest(make_noise(k, 16)) for k in keys
                        ],
                        "acc_init": archived_accuracy,
                    },
                }
            }
        )
    )
    monkeypatch.setattr(
        no_loss_baseline,
        "load_mnist",
        lambda _path: (test_x, test_y, test_x, test_y),
    )
    monkeypatch.setattr(no_loss_baseline, "git_identity", lambda: {"commit": "test"})

    result = no_loss_baseline.run_baseline(str(run_dir), steps=2)
    assert result["schema"] == "subliminal-no-loss-baseline.v1"
    assert result["operations"]["teacher_loaded"] is False
    assert result["operations"]["loss_function_called"] is False
    assert result["operations"]["optimizer_created"] is False
    assert result["checks"]["state_equal_after_each_forward"] == [True, True]
    assert result["checks"]["final_state_equal"]
    assert result["checks"]["digit_logits_bitwise_equal"]
    assert result["checks"]["digit_predictions_equal"]
    assert result["checks"]["digit_accuracy_equal"]
    assert result["initial_digit_scoring"]["accuracy"] == initial["accuracy"]
    assert sum(result["initial_digit_scoring"]["true_digit_counts"]) == 12
