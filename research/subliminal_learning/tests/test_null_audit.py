# SPDX-License-Identifier: MIT
import copy
import json

import torch

from research.subliminal_learning import null_audit
from research.subliminal_learning.data import make_noise, tensor_digest
from research.subliminal_learning.models import build, init_model
from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.run import state_digest


def test_null_audit_records_exact_inputs_and_optimizer_drift(tmp_path, monkeypatch):
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
    teacher = copy.deepcopy(w0)
    with torch.no_grad():
        for prm in teacher.parameters():
            prm.add_(0.01 * torch.randn_like(prm))
    teacher_path = tmp_path / "pack00_teacher.pt"
    torch.save(teacher.state_dict(), teacher_path)
    teacher_sha = null_audit._sha256(teacher_path)
    noise_digest = [tensor_digest(make_noise(k, 16)) for k in keys]

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
                        "teacher_digest": [state_digest(teacher, m) for m in range(2)],
                        "noise_digest": noise_digest,
                    },
                }
            }
        )
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    teacher_path.name: {
                        "sha256": teacher_sha,
                        "bytes": teacher_path.stat().st_size,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(null_audit, "git_identity", lambda: {"commit": "test"})

    result = null_audit.run_audit(str(run_dir), str(teacher_path), steps=2)
    assert result["schema"] == "subliminal-null-audit.v1"
    assert result["forward"]["w0_repeated"]["equal"]
    assert result["forward"]["student_repeated"]["equal"]
    assert result["forward"]["w0_vs_student"]["equal"]
    assert result["forward"]["w0_vs_attenuated_target"]["equal"]
    assert not result["loss_gradient"]["backward_changed_parameters"]
    assert len(result["optimizer_traces"]["adam"]) == 2
    assert len(result["optimizer_traces"]["sgd"]) == 2
    assert result["provenance_checks"]["noise_digest"] == noise_digest
    assert result["source"]["teacher_checkpoint_sha256"] == teacher_sha
