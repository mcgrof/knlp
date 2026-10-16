# SPDX-License-Identifier: MIT
"""Check frozen comparison bounds and prefill fixture metadata on CPU."""

import ast
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.kv.native.prefill import compare_close, select_pages, unpack_cache


def test_elementwise_tolerance_and_nonfinite():
    reference = torch.tensor([0.0, 1.0])
    assert compare_close(torch.tensor([0.0009, 1.0019]), reference)["passed"]
    result = compare_close(torch.tensor([0.0011, 1.0]), reference)
    assert not result["passed"] and result["violating_elements"] == 1
    assert not compare_close(torch.tensor([float("nan"), 1.0]), reference)["passed"]
    assert not compare_close(torch.tensor([0.0, float("inf")]), reference)["passed"]


@pytest.mark.parametrize("length", [128, 512])
def test_forced_lengths_account_for_all_outputs(tmp_path, length):
    tree = ast.parse(Path("tools/kv/native/runner.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "timing_record"
    )
    completed = []
    schedule = SimpleNamespace(
        _completed=completed,
        _mode="buffered",
        label=lambda _: None,
        internal_ids=lambda ids: ids,
    )
    memory = dict(cuda_device_used_bytes=1)
    calls = []
    synchronized_after_calls = []

    class Engine:
        llm_engine = SimpleNamespace(step=lambda: [])

        def collective_rpc(self, callback):
            return [] if callback == "completion_events" else [memory]

        def generate(self, prompts, params, use_tqdm):
            calls.append(params.max_tokens)
            completed[:] = [
                dict(
                    batch_id=i,
                    completed_monotonic=i + 1,
                    scheduled_tokens={"r": 1},
                    decode_request_ids=["r"],
                    completed_tokens={"r": 1},
                    preempted=[],
                )
                for i in range(params.max_tokens)
            ]
            output = SimpleNamespace(
                request_id="r",
                finished=True,
                num_cached_tokens=0,
                prompt_token_ids=[1, 2],
                outputs=[SimpleNamespace(token_ids=[3] * params.max_tokens)],
                metrics=SimpleNamespace(
                    arrival_time=time.time() - 1,
                    first_token_ts=1,
                    last_token_ts=params.max_tokens,
                    first_token_latency=0.1,
                ),
            )
            Engine.values = [output]
            self.llm_engine.step()
            return [output]

    engine = Engine()
    engine.llm_engine = SimpleNamespace(step=lambda: Engine.values, log_stats=True)
    namespace = dict(
        __package__="tools.kv.native",
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        schedule=schedule,
        callbacks=SimpleNamespace(
            reset_warm_peak="reset", native_memory_state="memory"
        ),
        OPTIONS=SimpleNamespace(stage="test", arm="test", repeat=0),
        OUT=tmp_path,
        json=json,
        hashlib=hashlib,
        time=time,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: synchronized_after_calls.append(len(calls))
            )
        ),
        save=lambda path, value: path.write_text(json.dumps(value)),
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "timing", "exec"),
        namespace,
    )
    engine.llm_engine.log_stats = False
    with pytest.raises(RuntimeError, match="before warmup"):
        namespace["timing_record"](
            engine,
            [{"prompt_token_ids": [1, 2]}],
            1,
            "test",
            tmp_path / "disabled.json",
            {},
            output_tokens=length,
        )
    assert calls == []
    engine.llm_engine.log_stats = True
    row = namespace["timing_record"](
        engine,
        [{"prompt_token_ids": [1, 2]}],
        1,
        "test",
        tmp_path / "cell.json",
        {},
        output_tokens=length,
    )
    assert calls == [length, length]
    assert synchronized_after_calls == [1, 2]
    assert row["CUDA_synchronized_at_timing_boundaries"]
    assert row["cohort_decode_seconds"] == length - 1
    assert row["output_tokens"] == length and row["output_tokens_per_request"] == length
    assert row["itl_p50_seconds"] == 1
    assert row["completion_linked_decode"]["steps"] == length - 2


def test_numerical_result_reports_full_coverage():
    reference = torch.zeros(3, 8, dtype=torch.float16)
    result = compare_close(reference, reference)
    assert result["elements"] == 24 and result["violating_elements"] == 0
    assert result["max_absolute_error"] == 0


def test_encoded_page_selection_preserves_all_fp8_bytes():
    raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    encoded = raw.view(torch.float8_e4m3fn)
    order = torch.tensor([15, 0, 7, 2], dtype=torch.int64)
    actual = select_pages(encoded, order)
    assert torch.equal(actual.view(torch.uint8), raw.index_select(0, order))


def test_native_cache_uses_second_axis_for_planes():
    key = (torch.arange(7 * 16 * 8 * 128).reshape(7, 16, 8, 128) % 1024).half()
    value = -key - 0.5
    native = torch.stack((key, value), dim=1)
    actual_k, actual_v = unpack_cache(native)
    order = torch.tensor([6, 0, 3], dtype=torch.int64)
    assert torch.equal(select_pages(actual_k, order), key.index_select(0, order))
    assert torch.equal(select_pages(actual_v, order), value.index_select(0, order))
    assert actual_k.stride(0) == actual_v.stride(0) == 2 * 16 * 8 * 128
    assert actual_k.untyped_storage().data_ptr() == native.untyped_storage().data_ptr()


def test_mixed_cache_keeps_original_planes():
    key = torch.zeros(7, 16, 8, 128, dtype=torch.float16)
    value = key.to(torch.float8_e4m3fn)
    actual_k, actual_v = unpack_cache((key, value))
    assert actual_k is key and actual_v is value


def test_wrong_cache_axis_and_bad_page_fail_before_indexing():
    with pytest.raises(ValueError, match="expected a K/V pair"):
        unpack_cache(torch.zeros(2, 7, 16, 8, 128))
    with pytest.raises(ValueError, match="outside the cache page"):
        select_pages(torch.zeros(7, 16, 8, 128), torch.tensor([7]))
    with pytest.raises(ValueError, match="outside the cache page"):
        select_pages(torch.zeros(7, 16, 8, 128), torch.tensor([-1]))
