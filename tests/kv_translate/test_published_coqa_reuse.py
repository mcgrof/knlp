from types import SimpleNamespace

import torch

from research.kv_translate.published import coqa, coqa_reuse
from research.kv_translate.published.coqa_runner import ModelPair


def story(domain, index, passage=None):
    return {
        "source": domain,
        "id": f"{domain}-{index:03d}",
        "story": passage or f"Unique passage {domain} {index}",
        "questions": [
            {"turn_id": turn, "input_text": f"Question {turn}?"}
            for turn in range(1, 11)
        ],
        "answers": [
            {"turn_id": turn, "input_text": f"Answer {turn}"} for turn in range(1, 11)
        ],
        "additional_answers": {},
    }


def test_passage_cut_leaves_a_boundary_crossing_token_in_the_suffix():
    prompt = "Passage text\n\nQuestion: What?\nAnswer:"
    boundary = prompt.index("Question:")
    offsets = [(0, 7), (7, boundary - 1), (boundary - 1, boundary + 2)]
    offsets += [(boundary + 2, len(prompt))]
    result = coqa_reuse.passage_cut(prompt, offsets)
    assert result["early_cut"] == 2
    assert result["token_crossing_boundary"] == 2
    assert result["crossing_token_left_in_suffix"] is True


def test_replication_selection_excludes_prior_and_duplicate_passages():
    rows = [story(domain, index) for domain in coqa.DOMAINS for index in range(30)]
    prior = rows[0]
    duplicate = story("mctest", 999, passage=prior["story"].upper())
    rows.append(duplicate)
    result = coqa_reuse.select_replication_conversations(
        {"data": rows}, {prior["id"]}, per_domain=20
    )
    assert result["sufficient"] is True
    assert all(
        len(result["selected_by_domain"][domain]) == 20 for domain in coqa.DOMAINS
    )
    selected_ids = {row["id"] for row in result["selected"]}
    assert prior["id"] not in selected_ids
    assert duplicate["id"] not in selected_ids
    normalized = [
        coqa_reuse.normalize_passage(row["story"]) for row in result["selected"]
    ]
    assert len(normalized) == len(set(normalized))


class DummyCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class DummyModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, *, input_ids, past_key_values, use_cache):
        assert use_cache is True
        self.inputs.append(input_ids.tolist())
        length = past_key_values.length + input_ids.shape[1]
        return SimpleNamespace(
            logits=torch.zeros((1, input_ids.shape[1], 4)),
            past_key_values=DummyCache(length),
        )


def fake_pair():
    pair = ModelPair.__new__(ModelPair)
    pair.device = torch.device("cpu")
    pair.models = {"32B": DummyModel()}
    pair._decode = lambda _model, _logits, cache, prompt_length: {
        "cache": cache,
        "cache_covered_tokens": prompt_length,
        "generation_seconds": 0.0,
    }
    return pair


def test_generate_split_consumes_every_token_after_an_arbitrary_cut():
    pair = fake_pair()
    result = pair.generate_split(
        "32B", [10, 11, 12, 13, 14], DummyCache(2), cache_cut=2
    )
    assert pair.models["32B"].inputs == [[[12, 13, 14]]]
    assert result["suffix_token_count"] == 3
    assert result["cache_cut"] == 2
    assert result["cache"].length == 5


def test_generate_split_keeps_the_historical_final_token_default():
    pair = fake_pair()
    result = pair.generate_split("32B", [10, 11, 12], DummyCache(2))
    assert pair.models["32B"].inputs == [[[12]]]
    assert result["suffix_token_count"] == 1
    assert result["final_token_seconds"] >= 0.0


def result_rows(turn_one_transfer=0.78):
    rows = []
    for domain in coqa.DOMAINS:
        for conversation in range(2):
            for turn in coqa.Q1_TURNS:
                rows.append(
                    {
                        "row_id": f"{domain}-{conversation}:{turn}",
                        "conversation_id": f"{domain}-{conversation}",
                        "domain": domain,
                        "turn": turn,
                        "native_early_f1": 0.80,
                        "translated_early_f1": (
                            turn_one_transfer if turn == 1 else 0.78
                        ),
                        "native_late_f1": 0.80,
                        "translated_late_f1": 0.79,
                        "native_early_health": [],
                        "translated_early_health": [],
                        "native_late_health": [],
                        "translated_late_health": [],
                    }
                )
    return rows


def test_r1_requires_stable_controls_both_transfers_and_turn_one():
    passed = coqa_reuse.summarize_r1(result_rows(), replicates=50)
    assert passed["next_state"] == "R2"
    assert passed["screens"]["native_cut_stability"]["passed"] is True
    assert passed["screens"]["late_positive_control"]["passed"] is True
    assert passed["screens"]["early_reuse"]["passed"] is True

    failed = coqa_reuse.summarize_r1(result_rows(turn_one_transfer=0.72), replicates=20)
    assert failed["next_state"] == "CLOSE_EARLY_REUSE_PROMOTION"
    assert (
        failed["screens"]["early_reuse"]["requirements"]["turn_1_deficit_at_most_0.03"]
        is False
    )


def test_r2_uses_the_same_early_reuse_screen():
    rows = result_rows()
    for row in rows:
        for field in list(row):
            if field.startswith("native_late") or field.startswith("translated_late"):
                del row[field]
    result = coqa_reuse.summarize_r2(rows, replicates=20)
    assert result["next_state"] == "INDEPENDENT_REPLICATION_POINT_SCREEN_PASS"
