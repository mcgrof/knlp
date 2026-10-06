import hashlib

import pytest

from research.kv_translate.published import coqa


def story(domain, index, turns=10):
    return {
        "source": domain,
        "id": f"{domain}-{index:02d}",
        "story": f"Passage for {domain} {index}.",
        "questions": [
            {"turn_id": turn, "input_text": f"Question {turn}?"}
            for turn in range(1, turns + 1)
        ],
        "answers": [
            {"turn_id": turn, "input_text": f"answer {turn}"}
            for turn in range(1, turns + 1)
        ],
        "additional_answers": {
            "0": [
                {"turn_id": turn, "input_text": f"alternate {turn}"}
                for turn in range(1, turns + 1)
            ]
        },
    }


def test_selection_is_hash_ordered_disjoint_and_model_blind():
    data = {
        "data": [story(domain, index) for domain in coqa.DOMAINS for index in range(25)]
    }
    got = coqa.select_conversations(data)
    assert len(got["q1"]) == 100
    assert len(got["smoke"]) == 5
    assert len(got["q2"]) == 20
    assert not ({row["id"] for row in got["q1"]} & {row["id"] for row in got["smoke"]})
    for domain in coqa.DOMAINS:
        ordered = sorted(
            (row for row in data["data"] if row["source"] == domain),
            key=lambda row: (coqa.selection_digest(row["id"]), row["id"]),
        )
        assert [row["id"] for row in got["q1"] if row["source"] == domain] == [
            row["id"] for row in ordered[:20]
        ]
        assert [row["id"] for row in got["q2"] if row["source"] == domain] == [
            row["id"] for row in ordered[:4]
        ]
        assert (
            next(row for row in got["smoke"] if row["source"] == domain)["id"]
            == ordered[20]["id"]
        )


def test_prompt_is_plain_completion_with_reference_history_and_current_question():
    row = story("cnn", 1, turns=2)
    prompt = coqa.render_prompt(row, 2)
    assert prompt == (
        "Answer each question using the passage. Give only a short answer.\n"
        "If the passage does not contain the answer, say unknown.\n\n"
        "Passage: Passage for cnn 1.\n\n"
        "Question: Question 1?\n"
        "Answer: answer 1\n"
        "Question: Question 2?\n"
        "Answer:"
    )
    assert "answer 2" not in prompt


def test_generated_history_suffix_retains_newline_without_adding_another():
    assert coqa.next_turn_suffix("Next?", "newline") == "Question: Next?\nAnswer:"
    assert coqa.next_turn_suffix("Next?", "eos") == "\nQuestion: Next?\nAnswer:"
    assert coqa.next_turn_suffix("Next?", "cap") == "\nQuestion: Next?\nAnswer:"


def test_official_multiple_reference_rule_is_leave_one_out_not_plain_max():
    score = coqa.turn_score(["red", "blue", "green"], "red")
    assert score == {"em": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3)}
    assert coqa.turn_score(["The red, fox."], "red fox") == {"em": 1.0, "f1": 1.0}
    assert coqa.turn_score([""], "") == {"em": 1.0, "f1": 1.0}


def test_health_separates_bad_generation_from_wrong_but_formed_answer():
    assert (
        coqa.health_events("wrong", "wrong\n", stop_reason="newline", new_tokens=2)
        == []
    )
    assert coqa.health_events("", "\n", stop_reason="newline", new_tokens=1) == [
        "empty"
    ]
    assert "unterminated" in coqa.health_events(
        "still going", "still going", stop_reason="cap", new_tokens=64
    )
    assert "unterminated_thinking" in coqa.health_events(
        "reason", "<think>reason", stop_reason="eos", new_tokens=2
    )


def q1_rows(deficit=0.02, unhealthy=False):
    rows = []
    for domain in coqa.DOMAINS:
        for conversation in range(4):
            for turn in (1, 5, 10):
                rows.append(
                    {
                        "row_id": f"{domain}-{conversation}:{turn}",
                        "conversation_id": f"{domain}-{conversation}",
                        "domain": domain,
                        "turn": turn,
                        "native_f1": 0.8,
                        "transfer_f1": 0.8 - deficit,
                        "native_health": [],
                        "transfer_health": ["empty"] if unhealthy else [],
                        "native_answer": "native",
                        "transfer_answer": "transfer",
                    }
                )
    return rows


def test_q1_screen_uses_equal_domains_and_grouped_conversation_bootstrap():
    result = coqa.summarize_q1(q1_rows(), replicates=100)
    assert result["screen"]["promising"] is True
    assert result["equal_domain"]["native_minus_transfer_f1"] == pytest.approx(0.02)
    assert (
        result["bootstrap"]["unit"] == "complete conversations resampled within domain"
    )
    assert result["turn_counts"] == {1: 20, 5: 20, 10: 20}

    failed = coqa.summarize_q1(q1_rows(deficit=0.06), replicates=50)
    assert failed["screen"]["promising"] is False
    assert failed["screen"]["requirements"]["each_domain_deficit_at_most_0.05"] is False


def test_q2_does_not_average_receiver_or_schedule_failures_away():
    rows = []
    for schedule in ("first_receiver_14B", "first_receiver_32B"):
        for receiver in ("14B", "32B"):
            rows.append(
                {
                    "schedule": schedule,
                    "conversation_id": f"{schedule}-{receiver}",
                    "domain": "cnn",
                    "turn": 1,
                    "receiver": receiver,
                    "transfer_f1": 0.75 if receiver == "14B" else 0.79,
                    "shadow_f1": 0.80,
                    "native_only_f1": 0.80,
                    "transfer_health": [],
                    "shadow_health": [],
                    "native_only_health": [],
                }
            )
    result = coqa.summarize_q2(rows, replicates=10)
    assert result["screen"]["promising"] is False
    assert (
        result["cells"]["first_receiver_14B:14B"]["comparisons"]["native_shadow"][
            "passed"
        ]
        is False
    )
    comparison = result["cells"]["first_receiver_14B:14B"]["comparisons"][
        "native_shadow"
    ]
    assert comparison["reference_unhealthy"] == 0
    assert comparison["transfer_unhealthy"] == 0
    assert comparison["by_turn_descriptive"]["1"]["turn_positions"] == 1


def test_q2_resume_discards_every_part_of_an_incomplete_conversation():
    stories = [story("cnn", 0, turns=2), story("cnn", 1, turns=2)]

    def records(conversation_id, count):
        return [{"conversation_id": conversation_id, "index": i} for i in range(count)]

    answers = records(stories[0]["id"], 12) + records(stories[1]["id"], 12)
    provenance = records(stories[0]["id"], 4) + records(stories[1]["id"], 2)
    analysis = records(stories[0]["id"], 4) + records(stories[1]["id"], 2)

    complete, kept_answers, kept_provenance, kept_analysis = (
        coqa.retain_complete_q2_conversations(
            stories, answers, provenance, analysis, 10
        )
    )
    assert complete == {stories[0]["id"]}
    assert len(kept_answers) == 12
    assert len(kept_provenance) == 4
    assert len(kept_analysis) == 4
    assert {
        row["conversation_id"]
        for rows in (kept_answers, kept_provenance, kept_analysis)
        for row in rows
    } == {stories[0]["id"]}


def test_selection_digest_is_exact_frozen_formula():
    cid = "abc"
    assert (
        coqa.selection_digest(cid) == hashlib.sha256(b"knlp-coqa-q1-v1:abc").hexdigest()
    )
