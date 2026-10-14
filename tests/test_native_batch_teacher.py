"""CPU request-map tests using the installed serving-stack API."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
from tools.kv.native_batch_quality import install, stage, finish


class Model:
    def compute_logits(self, hidden):
        return hidden.clone()


class Runner:
    def __init__(self):
        self.model = Model()

    def sample(self, hidden, batch, grammar):
        return self.model.compute_logits(hidden)


def batch(ids, prefill=False):
    return SimpleNamespace(
        req_ids=ids,
        num_draft_tokens=0,
        num_scheduled_tokens=[8 if prefill else 1] * len(ids),
        is_prefilling_np=[prefill] * len(ids),
    )


def test_request_reordering_preserves_raw_scores_and_targets(tmp_path):
    worker = SimpleNamespace(model_runner=Runner())
    original = worker.model_runner.sample
    compute = worker.model_runner.model.compute_logits
    install(worker)
    path = tmp_path / "rows.jsonl"
    requests = {
        "a": {"sample_id": "doc-a", "replica": 0, "teacher": [1, 2]},
        "b": {"sample_id": "doc-b", "replica": 1, "teacher": [3, 0]},
    }
    stage(worker, requests, str(path), "frozen-receipt")
    hidden = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    result = worker.model_runner.sample(hidden, batch(["b", "a"], True), None)
    assert result.argmax(-1).tolist() == [3, 1]
    assert worker.model_runner.model.compute_logits == compute
    worker.model_runner.sample(hidden[:1], batch(["a"]), None)
    worker.model_runner.sample(hidden[:1], batch(["b"]), None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keyed = {(r["request_id"], r["step"]): r for r in rows}
    assert keyed[("b", 0)]["target"] == 3
    assert keyed[("a", 0)]["target"] == 1
    assert keyed[("a", 1)]["target"] == 2
    assert keyed[("b", 1)]["target"] == 0
    assert keyed[("b", 0)]["nll"] == -float(hidden[0].log_softmax(-1)[3])
    assert keyed[("a", 0)]["top1"] == 0
    assert keyed[("b", 0)]["active_decode_rows"] == 0
    assert keyed[("a", 1)]["active_decode_rows"] == 1
    finish(worker)
    assert worker.model_runner.sample == original


def test_unregistered_request_is_rejected(tmp_path):
    worker = SimpleNamespace(model_runner=Runner())
    install(worker)
    stage(worker, {}, str(tmp_path / "rows.jsonl"), "receipt")
    with pytest.raises(RuntimeError, match="unregistered"):
        worker.model_runner.sample(torch.zeros(1, 4), batch(["unknown"]), None)
    finish(worker)


def test_nonfinite_logits_restore_original_method(tmp_path):
    worker = SimpleNamespace(model_runner=Runner())
    compute = worker.model_runner.model.compute_logits
    install(worker)
    stage(
        worker,
        {"a": {"sample_id": "doc", "replica": 0, "teacher": [0]}},
        str(tmp_path / "rows.jsonl"),
        "receipt",
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        worker.model_runner.sample(torch.full((1, 4), float("nan")), batch(["a"]), None)
    assert worker.model_runner.model.compute_logits == compute
    finish(worker)
