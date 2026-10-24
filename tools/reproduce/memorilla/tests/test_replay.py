# SPDX-License-Identifier: MIT
"""Exercise asynchronous admission, ownership and terminal failure handling."""

from types import SimpleNamespace

import pytest

from tools.reproduce.memorilla.replay import (
    BatchAdmission,
    configure_scheduling,
    runtime_settings,
)
from tools.reproduce.memorilla.run import configuration


class Engine:
    def __init__(self, fail_after=None):
        self.llm_engine = self
        self.paused = False
        self.busy = False
        self.pending = []
        self.admissions = []
        self.results = []
        self.fail_after = fail_after

    def has_unfinished_requests(self):
        return self.busy

    def sleep(self, *, level, mode):
        assert (level, mode) == (0, "keep")
        self.paused = True

    def enqueue(self, prompts, **kwargs):
        self.arguments = kwargs
        for position, prompt in enumerate(prompts):
            if position == self.fail_after:
                raise ValueError("submission failed")
            self.pending.append(prompt)
            if not self.paused:
                self.execute()

    def execute(self):
        self.admissions.append(tuple(self.pending))
        self.results.extend(self.pending)
        self.pending.clear()

    def wake_up(self, *, tags):
        assert tags == ["scheduling"]
        self.paused = False
        self.execute()

    def wait_for_completion(self, *, use_tqdm):
        result, self.results = self.results, []
        return result


def test_barrier_prevents_execution_of_partially_submitted_batches():
    prompts = ["first", "second", "third"]
    default = Engine()
    default.enqueue(prompts)
    assert default.admissions == [("first",), ("second",), ("third",)]
    engine = Engine()
    admission = BatchAdmission(engine)
    sampling = object()
    assert admission.generate(prompts, sampling, use_tqdm=False) == prompts
    assert engine.admissions == [tuple(prompts)]
    assert engine.arguments["sampling_params"] is sampling
    assert not engine.paused
    assert admission.generate(["short-tail"], sampling) == ["short-tail"]
    assert engine.admissions[-1] == ("short-tail",)


def test_failed_submission_never_executes_partial_batch_or_reuses_engine():
    engine = Engine(fail_after=2)
    admission = BatchAdmission(engine)
    with pytest.raises(ValueError, match="submission failed"):
        admission.generate(["first", "second", "third"])
    assert engine.paused and engine.pending == ["first", "second"]
    assert not engine.admissions
    with pytest.raises(RuntimeError, match="discard"):
        admission.generate(["retry"])
    assert not engine.admissions


def test_busy_engine_is_rejected_before_pause_or_submission():
    engine = Engine()
    engine.busy = True
    with pytest.raises(RuntimeError, match="exclusive"):
        BatchAdmission(engine).generate(["ours"])
    assert not engine.paused and not engine.pending and not engine.admissions


def test_old_runtime_is_not_silently_treated_as_batch_capable():
    with pytest.raises(ValueError, match="level-0"):
        BatchAdmission(SimpleNamespace(generate=lambda *args: []))


def test_attached_admission_does_not_keep_decoder_alive_for_scoring():
    import gc
    import weakref

    engine = Engine()
    admission = BatchAdmission(engine)
    engine.generate = admission.generate
    reference = weakref.ref(engine)
    del engine
    gc.collect()
    assert reference() is None


def test_scheduling_selection_is_explicit_and_runtime_preserves_environment(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    configure_scheduling("runtime")
    import os

    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    configure_scheduling("inprocess")
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    configure_scheduling("batch")
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    with pytest.raises(ValueError, match="unknown"):
        configure_scheduling("silent-fallback")


def test_effective_settings_record_unknowns_instead_of_inventing_defaults():
    legacy = SimpleNamespace(llm_engine=SimpleNamespace())
    assert runtime_settings(legacy)["enable_prefix_caching"] is None
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=False),
    )
    modern = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))
    assert runtime_settings(modern)["enable_prefix_caching"] is False
    assert runtime_settings(modern)["max_num_seqs"] == 1


@pytest.mark.parametrize("mode", ["inprocess", "batch", "unsupported"])
def test_original_profile_cannot_silently_change_scheduling(tmp_path, mode):
    config = tmp_path / "config"
    config.write_text(
        "CONFIG_KNLP_REPRODUCE_MEMORILLA=y\nCONFIG_KNLP_MEM_AUTHOR=y\n"
        f'CONFIG_KNLP_MEM_SCHEDULING="{mode}"\n'
    )
    with pytest.raises(ValueError, match="scheduling"):
        configuration(config)


def test_reference_preset_cannot_be_relabeled_after_changing_its_contract(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    original = (root / "defconfigs/memorilla-port-reference").read_text()
    config = tmp_path / "config"
    config.write_text(original)
    assert configuration(config)["batch"] == 1
    for wrong in [
        original.replace("MEM_BATCH=1", "MEM_BATCH=32"),
        original.replace('SCHEDULING="inprocess"', 'SCHEDULING="runtime"'),
        original.replace('caching\\":false', 'caching\\":true'),
    ]:
        config.write_text(wrong)
        with pytest.raises(ValueError, match="singleton reference"):
            configuration(config)
