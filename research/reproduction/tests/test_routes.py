"""Check public dispatch, qualification gates and smoke result boundaries."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.reproduction.run import REPO, Runner, configuration, execute, plan

PRESETS = (
    "subliminal-mnist",
    "subliminal-mnist-null-repaired",
    "subliminal-mnist-layerrope-study",
    "subliminal-mnist-smoke",
    "layerrope-mhc-tinystories",
    "layerrope-mhc-tinystories-depth",
    "layerrope-mhc-tinystories-confirmation",
    "layerrope-mhc-tinystories-smoke",
    "layerrope-mhc-tinystories-kv-cache",
    "layerrope-mhc-tinystories-kv-cache-smoke",
    "layerrope-mhc-tinystories-gsm8k",
    "layerrope-mhc-beta",
    "layerrope-mhc-beta-smoke",
)


def workspace(root):
    for path in REPO.iterdir():
        if path.name in (".git", ".config", ".config.old", "config.py"):
            continue
        (root / path.name).symlink_to(path, target_is_directory=path.is_dir())


@pytest.mark.parametrize("preset", PRESETS)
def test_plain_make_selects_the_loaded_study(tmp_path, preset):
    workspace(tmp_path)
    subprocess.run(
        ["make", "defconfig-" + preset, "PYTHON=" + sys.executable],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        ["make", "-n", "-j4", "PYTHON=" + sys.executable],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "research.reproduction.run run" in result.stdout
    assert "lenet5/train" not in result.stdout and "test_matrix.py" not in result.stdout
    config = configuration(tmp_path / ".config")
    assert config["smoke"] == preset.endswith("-smoke")
    if config["study"] in ("tinystories", "confirmation"):
        frozen = plan(config)["tinystories"]["confirmation"]
        assert len(frozen["rows"]) == 7 and len(frozen["seeds"]) == 8
        assert frozen["D_enabled"] is False


def test_conflicting_workload_fails_before_any_runner(tmp_path):
    workspace(tmp_path)
    config = (REPO / "defconfigs/subliminal-mnist-smoke").read_text()
    (tmp_path / ".config").write_text(config + "CONFIG_KNLP_REPRODUCE_DECODE=y\n")
    result = subprocess.run(
        ["make", "PYTHON=" + sys.executable],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Conflicting workloads" in result.stdout


def test_unrelated_default_keeps_its_dispatch(tmp_path):
    workspace(tmp_path)
    (tmp_path / ".config").write_text('CONFIG_MODEL_LENET5=y\nCONFIG_MODEL="lenet5"\n')
    result = subprocess.run(
        ["make", "-n", "PYTHON=" + sys.executable],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "research.reproduction.run run" not in result.stdout


def test_failed_qualification_prevents_mnist_training(tmp_path, monkeypatch):
    config = configuration(REPO / "defconfigs/subliminal-mnist-layerrope-study")
    runner = Runner(config, tmp_path)
    calls = []

    def reject(module, *args, **kwargs):
        calls.append((module, args))
        raise RuntimeError("qualification failed")

    monkeypatch.setattr(runner, "command", reject)
    with pytest.raises(RuntimeError, match="qualification failed"):
        runner.mnist()
    assert len(calls) == 1 and calls[0][1][0] == "qualify"


def test_unavailable_gpu_fails_without_a_cpu_substitute(tmp_path, monkeypatch):
    import torch

    config = configuration(REPO / "defconfigs/subliminal-mnist-layerrope-study")
    config["output_dir"] = str(tmp_path / "output")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        execute(config)
    assert not (tmp_path / "output").exists()


def test_incomplete_smoke_retains_failure_without_confirmation(tmp_path, monkeypatch):
    config = configuration(REPO / "defconfigs/subliminal-mnist-smoke")
    config["output_dir"] = str(tmp_path)

    def fail(_self):
        raise RuntimeError("interrupted smoke")

    monkeypatch.setattr(Runner, "mnist", fail)
    with pytest.raises(RuntimeError, match="interrupted smoke"):
        execute(config)
    report = json.loads(next(tmp_path.glob("*/REPORT.json")).read_text())
    assert report["status"] == "incomplete"
    assert report["scope"] == "smoke"
    assert report["scientific_confirmation_allowed"] is False


def test_make_overrides_are_in_the_effective_plan(tmp_path):
    workspace(tmp_path)
    subprocess.run(
        ["make", "defconfig-subliminal-mnist-smoke", "PYTHON=" + sys.executable],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            "make",
            "residual-plan",
            "PYTHON=" + sys.executable,
            "CONFIG_RESIDUAL_OUTPUT_DIR=" + str(tmp_path / "output"),
            "CONFIG_RESIDUAL_MAX_SECONDS=19",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout[result.stdout.index("{") :])
    assert data["effective_config"]["output_dir"] == str(tmp_path / "output")
    assert data["effective_config"]["max_seconds"] == 19


def test_repaired_null_keeps_the_original_recipe(tmp_path):
    from research.reproduction.mnist import recipes

    config = configuration(REPO / "defconfigs/subliminal-mnist-null-repaired")
    [recipe] = recipes(config, tmp_path)
    assert recipe.seeds == list(range(73001, 73006))
    assert recipe.student_epochs == 80 and recipe.teacher_epochs == 5
    assert recipe.student_loss_mode == "exact_cancel_v1"
    assert recipe.target_scale == 0 and recipe.pack_size == 0
    assert recipe.protocol == "per_example_v1"


def test_confirmation_respects_the_callers_reserved_devices():
    from research.reproduction.run import gpu_slots

    assert gpu_slots({"CUDA_VISIBLE_DEVICES": "2,5"}) == ["2", "5"]
    assert gpu_slots({"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}) == [
        "GPU-first",
        "GPU-second",
    ]
    with pytest.raises(RuntimeError, match="too small"):
        gpu_slots({"CUDA_VISIBLE_DEVICES": ""})


def test_saved_cache_requires_sources_before_device_initialization(tmp_path):
    config = configuration(REPO / "defconfigs/layerrope-mhc-tinystories-kv-cache")
    config["output_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="saved reproduction run"):
        execute(config)


def test_new_outputs_cannot_pollute_the_source_checkout():
    config = configuration(REPO / "defconfigs/layerrope-mhc-tinystories-kv-cache-smoke")
    config["output_dir"] = str(REPO / "results")
    with pytest.raises(ValueError, match="outside"):
        execute(config)


def test_saved_cache_dispatch_does_not_prepare_data_or_train(tmp_path, monkeypatch):
    config = configuration(REPO / "defconfigs/layerrope-mhc-tinystories-kv-cache")
    config.update(
        cache_depth_run=str(tmp_path / "old-depth"),
        cache_confirmation_run=str(tmp_path / "old-confirmation"),
    )
    runner = Runner(config, tmp_path)
    calls = []
    monkeypatch.setattr(
        runner, "command", lambda module, *args, **kwargs: calls.append((module, args))
    )
    runner.saved_cache()
    assert len(calls) == 2
    assert all(module == "research.reproduction.kv_cache" for module, _ in calls)
    assert [args[args.index("--cohort") + 1] for _, args in calls] == [
        "depth",
        "confirmation",
    ]
