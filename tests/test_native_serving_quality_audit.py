# SPDX-License-Identifier: MIT
"""Check that raw HTTP evidence controls the independently recomputed score."""

import json

import pytest

from tools.kv.native.serving_quality_audit import natural, response_choice


def completion(text="The answer is 42.", finish="stop"):
    choice = {"text": text, "token_ids": [7, 8], "finish_reason": finish}
    return {
        "id": "case",
        "prompt_hash": "frozen-prompt",
        "gold": "42",
        "error": None,
        "http_status": 200,
        "raw_response": json.dumps({"choices": [choice]}),
        "text": text,
        "generated_token_ids": [7, 8],
        "finish_reason": finish,
    }


def write_case(root, row, correct):
    (root / "gsm8k-raw.jsonl").write_text(json.dumps(row) + "\n")
    (root / "gsm8k-scored.json").write_text(
        json.dumps([{"id": "case", "correct": correct}])
    )
    return [{"id": "case", "prompt_hash": "frozen-prompt", "gold": "42"}]


def test_raw_answer_overrules_a_corrupt_saved_score(tmp_path):
    fixture = write_case(tmp_path, completion("The answer is 43."), True)
    with pytest.raises(ValueError):
        natural(tmp_path, "gsm8k", fixture)


def test_truncated_correct_text_does_not_count(tmp_path):
    fixture = write_case(tmp_path, completion(finish="length"), False)
    scores, evidence = natural(tmp_path, "gsm8k", fixture)
    assert scores == {"case": False}
    assert evidence["cases"] == 1 and evidence["correct"] == 0


def test_raw_token_ids_must_match_the_saved_completion():
    row = completion()
    row["generated_token_ids"] = [7, 9]
    with pytest.raises(ValueError, match="raw HTTP response"):
        response_choice(row)


def test_duplicate_rows_cannot_replace_case_coverage(tmp_path):
    row = completion()
    fixture = write_case(tmp_path, row, True)
    (tmp_path / "gsm8k-raw.jsonl").write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError):
        natural(tmp_path, "gsm8k", fixture)
