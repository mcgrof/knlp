# SPDX-License-Identifier: MIT
"""Full numerical gates must retain failures beyond their first chunk."""

import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from tools.kv.native.compact import (
    _observe_route_calls,
    full_comparison,
    tensor_hash,
)


def test_full_comparison_covers_last_chunk_and_all_counters():
    reference = torch.ones(8200, 2, dtype=torch.float16)
    actual = reference.clone()
    actual[4096, 0] = 2
    actual[-1, -1] = 3
    row = full_comparison(actual, reference)
    assert not row["passed"] and row["finite"]
    assert row["elements"] == 16400 and row["violating_elements"] == 2
    assert row["max_absolute_error"] == 2


def test_full_comparison_fixed_tolerance_and_nonfinite_gate():
    reference = torch.ones(2, 2)
    actual = reference + 0.0019
    assert full_comparison(actual, reference)["passed"]
    actual[0, 0] = float("nan")
    row = full_comparison(actual, reference)
    assert not row["passed"] and not row["finite"]


def test_full_comparison_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        full_comparison(torch.zeros(2, 3), torch.zeros(3, 2))


def test_hash_retains_encoded_bytes():
    values = torch.tensor([0.0, -0.0], dtype=torch.float16)
    assert torch.equal(values, values.flip(0))
    assert tensor_hash(values) != tensor_hash(values.flip(0))


@pytest.mark.parametrize("phase,batch_id", [("cohort", 17), (None, None)])
def test_route_observer_releases_model_tensors_without_cyclic_gc(
    tmp_path, phase, batch_id
):
    refs = []

    class Workspace:
        backing = torch.zeros(4)

        def run(self, query, cache, source_indices, page_plan):
            intermediate = query + 1
            refs.append(weakref.ref(intermediate))
            return intermediate

    class Runner:
        workspace = Workspace()

        def prepare_inputs(self, scheduler_output):
            return SimpleNamespace(req_ids=["request-17"])

        def execute_model(self, scheduler_output):
            batch = self.prepare_inputs(scheduler_output)
            assert batch.req_ids == ["request-17"]
            value = torch.zeros(64, 64)
            for _ in range(12):
                value = self.workspace.run(
                    value,
                    (self.workspace.backing,),
                    None,
                    SimpleNamespace(query_lengths=(64,), pages=4),
                )
            return value

    original_execute = Runner.execute_model
    original_prepare = Runner.prepare_inputs
    original_run = Workspace.run
    enabled = gc.isenabled()
    gc.disable()
    try:
        with _observe_route_calls(
            tmp_path / "routes.jsonl", Runner, Workspace, lambda: phase
        ):
            scheduler = (
                SimpleNamespace(_knlp_batch_id=batch_id)
                if batch_id is not None
                else SimpleNamespace()
            )
            result = Runner().execute_model(scheduler)
            assert torch.equal(result, torch.full((64, 64), 12.0))
            del result
        assert all(ref() is None for ref in refs)
        assert Runner.execute_model is original_execute
        assert Runner.prepare_inputs is original_prepare
        assert Workspace.run is original_run
        rows = [
            json.loads(x)
            for x in (tmp_path / "routes.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 12
        assert all(
            row["batch_id"] == batch_id
            and row["label"] == phase
            and row["request_ids"] == ["request-17"]
            for row in rows
        )
    finally:
        if enabled:
            gc.enable()


def test_route_observer_rejects_labeled_batch_without_scheduler_id(tmp_path):
    class Runner:
        def execute_model(self, scheduler_output):
            raise AssertionError("unidentified labeled batch executed")

        def prepare_inputs(self, scheduler_output):
            return SimpleNamespace(req_ids=[])

    class Workspace:
        def run(self, *args, **kwargs):
            raise AssertionError("unidentified labeled attention executed")

    original = Runner.execute_model
    with _observe_route_calls(
        tmp_path / "routes.jsonl", Runner, Workspace, lambda: "cohort"
    ):
        with pytest.raises(RuntimeError, match="missing scheduler identity"):
            Runner().execute_model(SimpleNamespace())
    assert Runner.execute_model is original
