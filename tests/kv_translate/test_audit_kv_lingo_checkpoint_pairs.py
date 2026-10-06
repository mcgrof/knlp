from research.kv_translate.published.audit_kv_lingo_checkpoint_pairs import (
    Audit,
    ids_sha256,
    reconstruct_history,
    regrade_fields,
    verify_ownership,
)


def test_regrade_fields_recomputes_official_score_and_health():
    row = {
        "translated_answer": "The cat",
        "translated_raw_text": "The cat\n",
        "translated_stop_reason": "newline",
        "translated_token_ids": [1, 2],
    }
    assert regrade_fields(row, "translated", ["cat", "a cat"]) == {
        "f1": 1.0,
        "em": 1.0,
        "health": [],
    }


def test_reconstruct_history_checks_each_incremental_prompt_hash():
    story = {
        "turn_1_token_ids": [10, 11],
        "next_turn_suffix_token_ids": {
            "2": {"after_eos": [12], "after_other": [13]},
        },
    }
    first_prompt = [10, 11]
    second_prompt = first_prompt + [20, 13]
    rows = [
        {
            "turn": 1,
            "prompt_token_ids_sha256": ids_sha256(first_prompt),
            "translated_token_ids": [20],
            "translated_stop_reason": "newline",
        },
        {
            "turn": 2,
            "prompt_token_ids_sha256": ids_sha256(second_prompt),
            "translated_token_ids": [21],
            "translated_stop_reason": "eos",
        },
    ]
    audit = Audit()
    metadata = reconstruct_history(story, rows, "translated", audit, "fixture")
    assert not audit.failures
    assert [item["prompt_tokens"] for item in metadata] == [2, 4]


def test_verify_ownership_reconstructs_both_cache_frontiers():
    record = {
        "models": ["4B", "8B"],
        "total_tokens": 6,
        "line_covered": {"4B": 6, "8B": 4},
        "spans": [
            {"writer": "4B", "start": 0, "end": 2, "serial": 0},
            {"writer": "8B", "start": 2, "end": 4, "serial": 1},
            {"writer": "4B", "start": 4, "end": 6, "serial": 2},
        ],
        "translations": [
            {"span_serial": 0, "receiver": "8B"},
            {"span_serial": 1, "receiver": "4B"},
        ],
        "remaining_auxiliary_spans": [2],
    }
    audit = Audit()
    result = verify_ownership(record, audit, "fixture")
    assert not audit.failures
    assert result["line_covered"] == {"4B": 6, "8B": 4}
    assert result["remaining_auxiliary_spans"] == [2]
