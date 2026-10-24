# SPDX-License-Identifier: MIT
"""Replay schedules must preserve identity across batching and storage paths."""

import pytest

from tools.reproduce.memorilla.run import BENCHMARKS, configuration
from tools.reproduce.memorilla.workloads import compare, fingerprint, schedule


@pytest.mark.parametrize("pattern", ["sequential", "shuffle", "random", "hot"])
def test_schedule_replay_and_bounds(pattern):
    left = schedule(17, 1003, 23, pattern)
    assert left == schedule(17, 1003, 23, pattern)
    assert len(left) == 1003 and min(left) >= 0 and max(left) < 17
    # Changing batch size must not change logical order or input identity.
    for batch in [1, 8, 32]:
        assert [
            row
            for start in range(0, len(left), batch)
            for row in left[start : start + batch]
        ] == left


def test_shuffle_covers_every_row_before_repeat():
    result = schedule(37, 111, 23, "shuffle")
    assert all(set(result[i : i + 37]) == set(range(37)) for i in range(0, 111, 37))


def test_comparison_detects_corruption_missing_and_duplicate():
    reference = [
        {
            "request_id": "q:0",
            "input_digest": fingerprint({"row": 3}),
            "prediction": "yes",
        }
    ]
    assert compare(reference, reference)["exact"]
    assert not compare(reference, [reference[0] | {"input_digest": "wrong-bank"}])[
        "exact"
    ]
    assert not compare(reference, [reference[0] | {"prediction": "no"}])["exact"]
    assert compare(reference, [])["missing"] == ["q:0"]
    with pytest.raises(ValueError, match="duplicate"):
        compare(reference, reference * 2)


def test_compare_cli_preserves_unicode_separators_inside_json_strings(
    tmp_path, monkeypatch, capsys
):
    import json
    import sys

    from tools.reproduce.memorilla.run import main

    row = {
        "request_id": "q:0",
        "input_digest": "same-input",
        "prediction": "first\u2028second\u2029third",
    }
    reference, observed = tmp_path / "reference.jsonl", tmp_path / "observed.jsonl"
    reference.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    observed.write_text(json.dumps(row, ensure_ascii=True) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "memorilla",
            "compare",
            "--reference",
            str(reference),
            "--observed",
            str(observed),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["exact"]


def test_all_paper_rows_and_configuration(tmp_path):
    assert sum(v[0] for v in BENCHMARKS.values()) == 32860
    config = tmp_path / "config"
    config.write_text(
        'CONFIG_KNLP_REPRODUCE_MEMORILLA=y\nCONFIG_KNLP_MEM_AUTHOR=y\nCONFIG_KNLP_MEM_BENCHMARK="all"\n'
    )
    assert len(configuration(config)["benchmarks"]) == 8
    config.write_text(config.read_text() + "CONFIG_KNLP_MEM_LMCACHE=y\n")
    with pytest.raises(ValueError, match="select one"):
        configuration(config)


@pytest.mark.parametrize("rows,requests", [(0, 1), (1, 0), (-1, 5)])
def test_invalid_schedules(rows, requests):
    with pytest.raises(ValueError):
        schedule(rows, requests, 23, "shuffle")


def test_saved_trace_rejects_duplicate_and_out_of_split_rows():
    from tools.reproduce.memorilla.workloads import validate_trace

    request = {"request_id": "q:0", "row": 0}
    validate_trace([request], 1)
    for broken in [
        [request, request],
        [request | {"row": 1}],
        [request | {"row": True}],
        [],
    ]:
        with pytest.raises(ValueError):
            validate_trace(broken, 1)


def test_original_runtime_rejects_port_options(tmp_path):
    config = tmp_path / "config"
    config.write_text(
        'CONFIG_KNLP_REPRODUCE_MEMORILLA=y\nCONFIG_KNLP_MEM_AUTHOR=y\nCONFIG_KNLP_MEM_ENGINE_JSON="{\\"enforce_eager\\":true}"\n'
    )
    with pytest.raises(ValueError, match="rejects engine overrides"):
        configuration(config)


def test_request_identity_uses_collection_without_raw_document_column():
    from tools.reproduce.memorilla.workloads import request_record

    question = {"collection_id": 900, "question": "Does it help?", "answer": "yes"}
    original = request_record(
        question, {"collection_id": 900}, 0, 0, "test:0", {"data": "frozen"}
    )
    swapped = request_record(
        question, {"collection_id": 901}, 0, 1, "test:0", {"data": "frozen"}
    )
    assert original["input_digest"] != swapped["input_digest"]
    assert original["row"] == swapped["row"] == 0
    assert swapped["collection_id"] == 901


def test_bfloat16_diagnostics_bind_values_shape_and_dtype():
    import torch

    from tools.reproduce.memorilla.worker import tensor_digest

    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    assert tensor_digest(x) == tensor_digest(x.clone())
    assert tensor_digest(x) != tensor_digest(x.reshape(2, 1))
    assert tensor_digest(x) != tensor_digest(x.float())
    assert tensor_digest(x) != tensor_digest(x + 1)


def test_tensor_equality_does_not_hide_changed_generated_text():
    ref = {
        "request_id": "q:0",
        "input_digest": "same",
        "prediction": "yes",
        "prompt_sha256": "same",
    }
    result = compare([ref], [ref | {"prediction": "no"}])
    assert result["exact_decoder_inputs"]
    assert not result["exact"]
    assert result["tensor_differences"]["reader_sha256"] is None
