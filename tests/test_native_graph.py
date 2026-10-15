# SPDX-License-Identifier: MIT
"""Reject stale identities and late capture-time configuration on CPU."""

import subprocess
import pytest
from tools.kv.native.graph import (
    ExecutionVariant,
    install_capture_hooks,
    resolve_variant,
)


def test_variant_names_do_not_infer_writer_or_scale_flags():
    control = ExecutionVariant({}, False, True)
    candidate = ExecutionVariant({}, True, True)
    definitions = {"control": control, "single": candidate}
    assert resolve_variant("single", definitions) is candidate
    assert control.environment({})["VLLM_ASYM_SINGLE_WRITER"] == "0"
    assert candidate.environment({})["VLLM_ASYM_SINGLE_WRITER"] == "1"
    assert candidate.environment({})["FLASHINFER_FUSE_V_SCALE"] == "1"
    for stale in ["combined-old", "single-guarded", "control-single"]:
        with pytest.raises(ValueError, match="unknown execution variant"):
            resolve_variant(stale, definitions)


def test_source_identity_is_frozen_against_caller_mutation():
    sources = {"backend": "1" * 40}
    variant = ExecutionVariant(sources, True, True)
    sources["backend"] = "2" * 40
    assert variant.sources["backend"] == "1" * 40
    with pytest.raises(TypeError):
        variant.sources["backend"] = "3" * 40


def test_configuration_precedes_warmup_and_capture(monkeypatch):
    events = []
    variant = ExecutionVariant({}, True, True)
    for name, value in variant.environment({}).items():
        monkeypatch.setenv(name, value)

    class Runner:
        scale = None

        def capture_model(self):
            events.append(("capture", self.scale))

    class Worker:
        def compile_or_warm_up_model(self):
            events.append(("warmup", self.model_runner.scale))
            self.model_runner.capture_model()

    runner = Runner()
    worker = Worker()
    worker.model_runner = runner
    original_warmup = Worker.compile_or_warm_up_model
    original_capture = Runner.capture_model

    def configure(value):
        value.model_runner.scale = 0.37
        events.append(("configure", 0.37))

    def check(value):
        assert value.scale == 0.37
        events.append(("check", value.scale))

    restore = install_capture_hooks(Worker, Runner, configure, check, variant)
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            install_capture_hooks(Worker, Runner, configure, check, variant)
        worker.compile_or_warm_up_model()
        assert events == [
            ("configure", 0.37),
            ("warmup", 0.37),
            ("check", 0.37),
            ("capture", 0.37),
        ]
        monkeypatch.setenv("VLLM_ASYM_SINGLE_WRITER", "0")
        with pytest.raises(RuntimeError, match="capture environment differs"):
            runner.capture_model()
        assert len(events) == 4
    finally:
        restore()
    assert Worker.compile_or_warm_up_model is original_warmup
    assert Runner.capture_model is original_capture


def test_source_identity_rejects_wrong_commit_dirty_tree_and_import_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    path = source / "module.py"
    path.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    variant = ExecutionVariant({"backend": sha}, True, True)
    variant.verify_sources({"backend": source}, {"backend": path})
    with pytest.raises(RuntimeError, match="outside source"):
        variant.verify_sources({"backend": source}, {"backend": tmp_path / "other.py"})
    with pytest.raises(RuntimeError, match="checkout differs"):
        ExecutionVariant({"backend": "0" * 40}, True, True).verify_sources(
            {"backend": source}
        )
    path.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="checkout differs"):
        variant.verify_sources({"backend": source})


def test_capture_configuration_refuses_an_unfrozen_experimental_flag():
    variant = ExecutionVariant({}, True, True)
    wrong = variant.environment({})
    wrong["FLASHINFER_ASYM_FP8_PV"] = "1"
    with pytest.raises(RuntimeError, match="capture environment differs"):
        variant.verify_environment(wrong)
