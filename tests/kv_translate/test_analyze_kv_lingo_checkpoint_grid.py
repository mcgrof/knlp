import pytest

from research.kv_translate.published import coqa
from research.kv_translate.published.analyze_kv_lingo_checkpoint_grid import analyze


def cells():
    result = {}
    pairs = (
        "forward_1000_reverse_1000",
        "forward_5000_reverse_5000",
        "forward_5000_reverse_1000",
        "forward_1000_reverse_5000",
    )
    for pair_index, pair in enumerate(pairs):
        rows = {}
        for domain_index, domain in enumerate(coqa.DOMAINS):
            for start in ("4B", "8B"):
                for turn in (1, 2):
                    key = (domain, domain, start, start, turn)
                    change = 0.0 if turn == 1 else pair_index / 100.0
                    rows[key] = {
                        "conversation_id": domain,
                        "domain": domain,
                        "starting_model": start,
                        "receiver": start,
                        "turn": turn,
                        "prompt_token_ids_sha256": f"{pair}-{domain}-{start}-{turn}",
                        "translated_f1": 0.5 + change,
                        "same_transcript_native_f1": 0.6,
                        "native_trajectory_f1": 0.6,
                        "translated_raw_text": "same" if turn == 1 else pair,
                        "translated_token_ids": [1] if turn == 1 else [pair_index],
                        "same_transcript_native_raw_text": "native",
                        "same_transcript_native_token_ids": [2],
                        "native_trajectory_raw_text": "native",
                        "native_trajectory_token_ids": [2],
                    }
                    if turn == 1:
                        rows[key]["prompt_token_ids_sha256"] = f"turn1-{domain}-{start}"
        result[pair] = rows
    return result


def test_grid_pairs_complete_conversations_and_checks_turn1():
    result = analyze(cells(), draws=20)
    comparison = result["paired_comparisons"][
        "forward_1000_reverse_1000_to_forward_5000_reverse_1000"
    ]["conditions"]["4B/same_transcript_native"]
    assert comparison["point"]["treatment_f1_change"] == pytest.approx(0.01)
    assert comparison["point"]["reference_f1_change"] == 0.0
    assert comparison["point"]["deficit_change"] == pytest.approx(-0.01)
    assert result["turn1_control"]["passed"]


def test_turn1_control_detects_a_native_only_difference():
    data = cells()
    pair = "forward_1000_reverse_5000"
    key = next(key for key in data[pair] if key[-1] == 1)
    data[pair][key]["translated_raw_text"] = "changed"
    result = analyze(data, draws=5)
    assert not result["passed_integrity_checks"]
    assert result["turn1_control"]["mismatch_count"] == 1


def test_grid_rejects_incomplete_cell_join():
    data = cells()
    pair = "forward_5000_reverse_1000"
    data[pair].pop(next(iter(data[pair])))
    with pytest.raises(ValueError, match="checkpoint-pair row identities differ"):
        analyze(data, draws=5)
