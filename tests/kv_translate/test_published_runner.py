# SPDX-License-Identifier: GPL-2.0
"""The stage runner, tested for the one property it exists for.

An interrupted run must resume without repeating finished work and without
skipping unfinished work, and a change of configuration must not be mistaken
for a finished stage.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from research.kv_translate.published import run_pair  # noqa: E402

THETA = 1_000_000.0


def tiny(layers, seed, heads=4, kv_heads=2, head_dim=16, hidden=48):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=257,
        hidden_size=hidden,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=512,
        rope_theta=THETA,
    )
    return Qwen3ForCausalLM(cfg).eval()


def tokens(n, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 257, (n, t), generator=g)


def cfg_for_test(**over):
    c = run_pair.default_config()
    c.update(over)
    return c


def test_a_finished_stage_is_not_repeated(tmp_path):
    ran, told = [], []
    st = run_pair.Stages(str(tmp_path), "abc", told.append)
    st.run("one", lambda: ran.append(1) or {"x": 1})
    again = run_pair.Stages(str(tmp_path), "abc", told.append)
    r = again.run("one", lambda: ran.append(2))
    assert ran == [1]
    assert told == ["one"]
    assert r["detail"] == {"x": 1}


def test_a_stage_from_another_configuration_is_repeated(tmp_path):
    ran = []
    run_pair.Stages(str(tmp_path), "abc").run("one", lambda: ran.append(1))
    run_pair.Stages(str(tmp_path), "xyz").run("one", lambda: ran.append(2))
    assert ran == [1, 2]


def test_a_stage_that_raised_left_no_receipt(tmp_path):
    st = run_pair.Stages(str(tmp_path), "abc")

    def boom():
        raise RuntimeError("the instance went away")

    with pytest.raises(RuntimeError):
        st.run("one", boom)
    assert st.done("one") is None
    assert not os.path.exists(st.path("one"))


def test_the_fit_identity_ignores_what_only_the_evaluation_uses():
    a = cfg_for_test()
    b = cfg_for_test(eval_examples=7, tasks=["hellaswag"], capture_batch=2)
    assert run_pair.identity(a) == run_pair.identity(b)
    assert run_pair.eval_identity(a) != run_pair.eval_identity(b)
    for key, val in (("k", 4), ("stride", 2), ("sequences", 10), ("lam", 0.1)):
        assert run_pair.identity(cfg_for_test(**{key: val})) != run_pair.identity(a)
        assert run_pair.eval_identity(cfg_for_test(**{key: val})) != (
            run_pair.eval_identity(a)
        )


def test_systematic_sampling_is_even_and_repeatable():
    assert run_pair.systematic(10, 20) == list(range(10))
    got = run_pair.systematic(1000, 100)
    assert got == run_pair.systematic(1000, 100)
    assert len(got) == 100 and got[0] == 0 and got[-1] == 990
    assert len(set(np.diff(got))) == 1
    assert len(run_pair.systematic(57, 9)) == 9


def test_an_interrupted_run_resumes_where_it_stopped(tmp_path, monkeypatch):
    models = {"src": tiny(3, 21), "tgt": tiny(4, 22)}
    loads, told = [], []

    def fake_load(name, revision, device="cpu"):
        loads.append(name)
        return models[name]

    def fake_tokens(cfg, work):
        arr = tokens(cfg["sequences"], cfg["sequence_length"], seed=3).numpy()
        np.save(os.path.join(work, "calibration_tokens.npy"), arr)
        return {"sequences": int(arr.shape[0])}

    monkeypatch.setattr(run_pair, "load_model", fake_load)
    monkeypatch.setattr(run_pair, "stage_tokens", fake_tokens)
    monkeypatch.setattr(run_pair, "free", lambda *a: None)
    cfg = cfg_for_test(
        source="src",
        target="tgt",
        source_revision="r",
        target_revision="r",
        sequences=16,
        sequence_length=48,
        stride=2,
        k=2,
        boundaries=[6, 47, 8],
        capture_batch=8,
        device="cpu",
    )
    work = str(tmp_path / "w")
    run_pair.run(cfg, work, on_stage_done=told.append, stop_after="capture_target")
    assert told == ["tokens", "capture_source", "capture_target"]
    assert loads == ["src", "tgt"]

    out = run_pair.run(
        cfg, work, on_stage_done=told.append, stop_after="fit_cache_bridge"
    )
    assert told[3:] == ["select", "fit_full_head", "fit_cache_bridge"]
    assert loads == ["src", "tgt"], "a finished capture loaded its model again"
    assert set(out["stage_seconds"]) == set(told)

    with open(os.path.join(work, "stages", "fit_full_head.json")) as f:
        full = json.load(f)["detail"]
    with open(os.path.join(work, "stages", "fit_cache_bridge.json")) as f:
        local = json.load(f)["detail"]
    # 2 selected layers, 2 source heads, width 16: the baseline reads every head
    assert full["feature_width"] == 2 * 2 * 16
    assert local["feature_width"] == 2 * 16
    assert local["repair_maps"] == 4 * 2 * 2
    assert os.path.getsize(
        os.path.join(work, "mapper_full_head_mapping.safetensors")
    ) > os.path.getsize(os.path.join(work, "mapper_cache_bridge.safetensors"))


def test_retention_is_reported_against_both_receiver_paths(tmp_path):
    st = run_pair.Stages(str(tmp_path / "two"), "abc")
    for mode, score in (("direct", 0.5), ("native", 0.4), ("cache_bridge", 0.3)):
        st.run(
            f"eval_t_{mode}",
            lambda mode=mode, score=score: {
                "task": "t",
                "mode": mode,
                "score": score,
                "examples": 10,
            },
        )
    out = run_pair.summarise(str(tmp_path / "two"))
    assert out["retention_percent_of_native"]["t"]["cache_bridge"] == 75.0
    assert out["retention_percent_of_direct"]["t"]["cache_bridge"] == 60.0


def test_retention_is_reported_against_the_receiver_alone(tmp_path):
    st = run_pair.Stages(str(tmp_path), "abc")
    for mode, score in (("direct", 0.5), ("native", 0.5), ("cache_bridge", 0.45)):
        st.run(
            f"eval_t_{mode}",
            lambda mode=mode, score=score: {
                "task": "t",
                "mode": mode,
                "score": score,
                "examples": 10,
            },
        )
    out = run_pair.summarise(str(tmp_path))
    assert out["retention_percent_of_direct"]["t"] == {
        "direct": 100.0,
        "native": 100.0,
        "cache_bridge": 90.0,
    }
    assert out["retention_percent_of_native"]["t"]["cache_bridge"] == 90.0


def test_the_processor_count_is_the_one_this_process_may_use():
    n = run_pair.usable_processors()
    assert 1 <= n <= len(os.sched_getaffinity(0))


def test_per_example_results_are_read_by_subject():
    samples = {
        "a": [{"doc_id": 3, "acc": 1.0, "x": 0}, {"doc_id": 9, "acc": 0.0}],
        "b": [{"doc_id": 3, "acc": 1.0}],
    }
    got = run_pair.extract_outcomes(samples, "acc")
    assert got == {"a": {"3": 1.0, "9": 0.0}, "b": {"3": 1.0}}


def test_a_missing_result_is_not_counted_as_a_wrong_answer():
    with pytest.raises(RuntimeError, match="carries no acc_norm"):
        run_pair.extract_outcomes({"a": [{"doc_id": 1, "acc": 1.0}]}, "acc_norm")
    with pytest.raises(RuntimeError, match="scored twice"):
        run_pair.extract_outcomes(
            {"a": [{"doc_id": 1, "acc": 1.0}, {"doc_id": 1, "acc": 0.0}]}, "acc"
        )
    with pytest.raises(RuntimeError, match="no per-example"):
        run_pair.extract_outcomes({}, "acc")
    with pytest.raises(RuntimeError, match="no per-example"):
        run_pair.extract_outcomes(None, "acc")


def test_the_paired_report_waits_for_per_example_results(tmp_path):
    cfg = cfg_for_test(tasks=["t"], modes=["native", "full_head"])
    assert run_pair.write_paired(cfg, str(tmp_path)) is None
    d = tmp_path / "outcomes"
    d.mkdir()
    (d / "t.native.json").write_text(json.dumps({"t": {"0": 1.0, "1": 1.0}}))
    (d / "t.full_head.json").write_text(json.dumps({"t": {"0": 1.0, "1": 0.0}}))
    out = run_pair.write_paired(cfg, str(tmp_path))
    assert out["arms"]["full_head"]["tasks"]["t"]["only_baseline_right"] == 1
    assert os.path.exists(tmp_path / "PAIRED.json")


def test_the_source_model_is_a_second_baseline_not_a_replacement(tmp_path):
    modes = ["native", "cache_bridge", run_pair.SOURCE_NATIVE]
    cfg = cfg_for_test(tasks=["t"], modes=modes)
    d = tmp_path / "outcomes"
    d.mkdir()
    for mode, marks in (
        ("native", [1.0, 0.0, 0.0, 0.0]),
        ("cache_bridge", [1.0, 0.0, 0.0, 0.0]),
        (run_pair.SOURCE_NATIVE, [1.0, 1.0, 1.0, 0.0]),
    ):
        rows = {str(i): m for i, m in enumerate(marks)}
        (d / f"t.{mode}.json").write_text(json.dumps({"t": rows}))
    out = run_pair.write_paired(cfg, str(tmp_path))
    own = out["arms"]["cache_bridge"]["tasks"]["t"]
    big = out["against_source"]["arms"]["cache_bridge"]["tasks"]["t"]
    assert own["difference"] == 0.0
    assert big["difference"] == pytest.approx(-0.5)
    assert out["against_source"]["baseline"] == run_pair.SOURCE_NATIVE
