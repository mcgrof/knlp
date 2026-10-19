"""HTTP teacher admission must bind frozen prompts to actual worker IDs."""

from types import SimpleNamespace

import pytest

from tools.kv.native.serving_quality import completion_body, register_teacher_requests


def pending():
    return {
        "nll-doc": {
            "sample_id": "doc",
            "prompt_token_ids": [3, 5],
            "teacher": list(range(129)),
        }
    }


def test_teacher_matches_actual_randomized_worker_id_and_prompt():
    requests = pending()
    state = {"requests": {}, "counts": {}}
    seen = set()
    request = SimpleNamespace(req_id="cmpl-nll-doc-0-c0ffee12", prompt_token_ids=[3, 5])
    register_teacher_requests(requests, state, seen, [request])
    assert state["requests"][request.req_id]["teacher"] == list(range(129))
    assert state["counts"] == {request.req_id: 0}
    assert seen == {"doc"}
    with pytest.raises(RuntimeError, match="twice"):
        register_teacher_requests(requests, state, seen, [request])


@pytest.mark.parametrize(
    "rid,tokens", [("unregistered", [3, 5]), ("cmpl-nll-doc-0-abc", [3, 6])]
)
def test_teacher_rejects_unregistered_or_modified_prompts(rid, tokens):
    state = {"requests": {}, "counts": {}}
    with pytest.raises(RuntimeError):
        register_teacher_requests(
            pending(),
            state,
            set(),
            [SimpleNamespace(req_id=rid, prompt_token_ids=tokens)],
        )
    assert state == {"requests": {}, "counts": {}}


def test_natural_quality_does_not_force_length_or_drop_task_stops():
    sample = {"id": "test-0000", "token_ids": [3, 5, 7, 9], "length": 2}
    task = {"max_output_tokens": 2048, "stop": ["Q:"]}
    gsm = completion_body(sample, "gsm8k", task)
    assert gsm["ignore_eos"] is False and gsm["stop"] == task["stop"]
    assert gsm["max_tokens"] == 2048 and gsm["prompt"] == sample["token_ids"]
    retrieval = completion_body(sample, "retrieval")
    assert retrieval["ignore_eos"] is False and "stop" not in retrieval
    nll = completion_body(sample, "nll")
    assert nll["ignore_eos"] is True and nll["max_tokens"] == 129
    assert nll["prompt"] == [3, 5]
