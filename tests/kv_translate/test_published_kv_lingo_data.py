import json
import os
from pathlib import Path

import pytest

from research.kv_translate.published.kv_lingo_data import (
    Candidate,
    CandidatePool,
    TokenStore,
    TokenStoreWriter,
    assistant_turns,
    build_packed,
    build_packed_v2,
    candidates_from_record,
    candidates_from_record_v2,
    mixture_cell,
    mixture_cell_v2,
    token_sha256,
)
from research.kv_translate.published.freeze_kv_lingo_coqa import (
    frozen_story,
    round_robin_select,
)
from research.kv_translate.published.freeze_kv_lingo_onehop import freeze_rows
from research.kv_translate.published.kv_lingo_onehop_runner import learning_gate
from research.kv_translate.published import coqa
from research.kv_translate.published.kv_lingo_eval import summarize


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize
        rendered = "".join(
            f"<{row['role']}>{row['content']}</{row['role']}>" for row in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return list(rendered.encode())


def candidate(uuid, length, continuation=8, reasoning="off", turn=1):
    return Candidate(
        uuid=uuid,
        reasoning=reasoning,
        assistant_index=turn,
        prefix_ids=tuple(range(length)),
        continuation_ids=tuple(range(continuation)),
    )


def test_the_four_row_cycle_balances_both_axes_and_alternates_reasoning():
    assert [mixture_cell(i) for i in range(4)] == [
        ("natural", "off"),
        ("packed", "on"),
        ("packed", "off"),
        ("natural", "on"),
    ]


def test_a_natural_turn_keeps_the_empty_thinking_marker_in_the_prefix():
    tokenizer = FakeTokenizer()
    record = {
        "uuid": "row-1",
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": "A sufficiently detailed test question for the tokenizer.",
            },
            {"role": "assistant", "content": "answer"},
        ],
    }
    assert assistant_turns(record["messages"]) == [2]
    rows = candidates_from_record(tokenizer, record, "off")
    assert len(rows) == 1
    assert b"<think>\n\n</think>\n\n" in bytes(rows[0].prefix_ids)
    assert bytes(rows[0].continuation_ids).endswith(b"<|im_end|>\n")


def test_reasoning_on_leaves_the_real_trace_in_the_continuation():
    tokenizer = FakeTokenizer()
    record = {
        "uuid": "row-2",
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<think>work</think>answer",
            },
        ],
    }
    row = candidates_from_record(tokenizer, record, "on")[0]
    assert not bytes(row.prefix_ids).endswith(b"</think>\n\n")
    assert bytes(row.continuation_ids).startswith(b"<think>work</think>")


def test_packing_keeps_complete_fillers_before_the_focal_question():
    focal = candidate("focal", 100, reasoning="on")
    fillers = CandidatePool(
        [
            candidate(f"fill-{i}", 1000, continuation=100, reasoning="on")
            for i in range(16)
        ],
        seed=42,
    )
    prefix, continuation, constituents = build_packed(focal, fillers, serial=1)
    assert 12_000 <= len(prefix) <= 16_000
    assert tuple(prefix[-100:]) == focal.prefix_ids
    assert tuple(continuation) == focal.continuation_ids
    assert constituents[-1] == focal.identity()


def test_v2_cell_serial_breaks_mode_length_aliasing():
    for offset in range(4):
        selected = [
            mixture_cell_v2(offset + 4 * serial)[2] % 32 for serial in range(32)
        ]
        assert selected == list(range(32))


def test_v2_separates_capped_focal_answer_from_closed_reasoning_free_history():
    record = {
        "uuid": "long",
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<think>secret work</think>" + "answer" * 600,
            },
        ],
    }
    row = candidates_from_record_v2(FakeTokenizer(), record)[0]
    assert row.reasoning == "on"
    assert len(row.continuation_ids) == 2048
    assert row.uncapped_continuation_tokens > 2048
    filler = bytes(row.complete_filler_ids)
    assert b"secret work" not in filler
    assert filler.endswith(b"</assistant>")


def test_v2_packing_uses_closed_fillers_not_capped_continuations():
    tokenizer = FakeTokenizer()
    rows = []
    for index in range(24):
        record = {
            "uuid": f"row-{index}",
            "messages": [
                {"role": "user", "content": "q" * 500},
                {"role": "assistant", "content": "a" * 300},
            ],
        }
        rows.extend(candidates_from_record_v2(tokenizer, record))
    pool = CandidatePool(rows, seed=42, algorithm="knlp-kv-lingo-stream-v2")
    prefix, _, constituents = build_packed_v2(rows[0], pool, serial=3)
    assert 12_000 <= len(prefix) <= 16_000
    assert constituents[-1] == rows[0].identity()
    filler = bytes(prefix[: -len(rows[0].prefix_ids)])
    assert filler.count(b"</assistant>") == len(constituents) - 1

    for serial in range(64):
        packed, _, _ = build_packed_v2(rows[serial % len(rows)], pool, serial=serial)
        assert 12_000 <= len(packed) <= 16_000


def test_v2_with_pinned_qwen_tokenizer_closes_and_strips_history():
    configured = os.environ.get("KV_LINGO_TOKENIZER")
    if not configured:
        pytest.skip("set KV_LINGO_TOKENIZER to a locally staged tokenizer")
    tokenizer_path = Path(configured)
    if not tokenizer_path.is_dir():
        pytest.skip("pinned KV-Lingo tokenizer is not locally staged")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    record = {
        "uuid": "actual-tokenizer",
        "messages": [
            {
                "role": "user",
                "content": "A sufficiently detailed tokenizer test question.",
            },
            {"role": "assistant", "content": "<think>private trace</think>answer"},
        ],
    }
    row = candidates_from_record_v2(tokenizer, record)[0]
    decoded = tokenizer.decode(row.complete_filler_ids)
    assert "private trace" not in decoded
    assert "answer<|im_end|>" in decoded


def test_v2_uses_structured_reasoning_only_for_the_current_answer():
    configured = os.environ.get("KV_LINGO_TOKENIZER")
    if not configured:
        pytest.skip("set KV_LINGO_TOKENIZER to a locally staged tokenizer")
    tokenizer_path = Path(configured)
    if not tokenizer_path.is_dir():
        pytest.skip("pinned KV-Lingo tokenizer is not locally staged")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    record = {
        "uuid": "structured-reasoning",
        "messages": [
            {"role": "user", "content": "A detailed structured reasoning question."},
            {
                "role": "assistant",
                "reasoning_content": "private structured trace",
                "content": "public answer",
            },
        ],
    }
    row = candidates_from_record_v2(tokenizer, record)[0]
    continuation = tokenizer.decode(row.continuation_ids)
    history = tokenizer.decode(row.complete_filler_ids)
    assert row.reasoning == "on"
    assert continuation.startswith("<think>\nprivate structured trace\n</think>")
    assert "private structured trace" not in history
    assert "public answer<|im_end|>" in history


def test_uint32_store_round_trips_offsets_and_has_a_canonical_hash(tmp_path: Path):
    path = tmp_path / "tokens.u32"
    writer = TokenStoreWriter(path)
    first = writer.append([1, 2, 3])
    second = writer.append([100_000, 4])
    writer.close()
    store = TokenStore(path)
    assert first == (0, 3)
    assert second == (3, 2)
    assert store.read(*first) == [1, 2, 3]
    assert store.read(*second) == [100_000, 4]
    assert token_sha256([1, 2, 3]) == token_sha256(store.read(*first))


def test_coqa_selection_round_robins_domains_before_repeating_one():
    rows = {
        domain: [{"id": f"{domain}-{index}"} for index in range(3)]
        for domain in coqa.DOMAINS
    }
    selected = round_robin_select(rows, 7, "test")
    counts = {
        domain: sum(row["id"].startswith(domain + "-") for row in selected)
        for domain in coqa.DOMAINS
    }
    assert sorted(counts.values()) == [1, 1, 1, 2, 2]


def test_frozen_coqa_story_carries_both_next_turn_stop_variants():
    story = {
        "id": "story",
        "source": "cnn",
        "story": "passage",
        "questions": [{"input_text": f"q{i}"} for i in range(10)],
        "answers": [{"input_text": f"a{i}"} for i in range(10)],
    }
    frozen = frozen_story(FakeTokenizer(), story)
    assert frozen["turn_1_token_ids"]
    assert set(frozen["next_turn_suffix_token_ids"]["2"]) == {
        "after_eos",
        "after_other",
    }


def test_onehop_rows_freeze_turns_1_5_10_with_supplied_reference_history():
    story = {
        "id": "story",
        "source": "cnn",
        "story": "passage",
        "questions": [{"input_text": f"q{i}"} for i in range(10)],
        "answers": [{"input_text": f"a{i}"} for i in range(10)],
        "additional_answers": {},
    }
    rows = freeze_rows(FakeTokenizer(), [story])
    assert [row["turn"] for row in rows] == [1, 5, 10]
    assert [len(row["primary_reference_history"]) for row in rows] == [0, 4, 9]
    assert all(row["prefix_tokens"] == len(row["token_ids"]) - 1 for row in rows)
    assert all(
        row["token_ids_sha256"] == token_sha256(row["token_ids"]) for row in rows
    )


def test_retained_summary_requires_each_start_against_both_native_references():
    translated = []
    native = []
    for start in ("4B", "8B"):
        for domain in coqa.DOMAINS:
            for turn in range(1, 11):
                key = {
                    "conversation_id": f"{start}-{domain}",
                    "starting_model": start,
                    "domain": domain,
                    "turn": turn,
                    "receiver": (
                        start if turn % 2 else ("8B" if start == "4B" else "4B")
                    ),
                }
                translated.append(
                    {
                        **key,
                        "translated_f1": 0.79,
                        "same_transcript_native_f1": 0.80,
                        "translated_health": [],
                        "same_transcript_native_health": [],
                    }
                )
                native.append(
                    {
                        **key,
                        "native_trajectory_f1": 0.81,
                        "native_trajectory_health": [],
                    }
                )
    result = summarize(translated, native, bootstrap=False)
    assert result["passed"] is True
    summary = result["starting_models"]["4B"]["same_transcript_native"]
    assert summary["pooled_turn6_10_deficit"] == summary["turn6_10_deficit"]
    assert "equal_domain_turn6_10_deficit" in summary
    assert set(summary["diagnostic_slices"]) == {"by_receiver", "by_turn", "by_domain"}
    json.dumps(result)
    translated[0]["translated_health"] = ["broken"]
    translated[1]["translated_health"] = ["broken"]
    assert summarize(translated, native, bootstrap=False)["passed"] is False


def onehop_summary(*, transfer_f1, native_f1=0.8, unhealthy=0, questions=100):
    return {
        "equal_domain": {
            "native_f1": native_f1,
            "transfer_f1": transfer_f1,
            "native_minus_transfer_f1": native_f1 - transfer_f1,
        },
        "pooled_official_turn_weighted": {
            "questions": questions,
            "native_unhealthy": 0,
            "transfer_unhealthy": unhealthy,
            "health_excess": unhealthy / questions,
        },
    }


def test_c2_gate_requires_10_percent_kl_gain_and_no_onehop_regression():
    stage1 = onehop_summary(transfer_f1=0.50, unhealthy=1)
    trained = onehop_summary(transfer_f1=0.47, unhealthy=3)
    passed = learning_gate(0.4, 0.36, stage1, trained)
    assert passed["kl_improved_at_least_10_percent"] is True
    assert passed["step1000_vs_stage1"]["passed"] is True
    assert passed["passed"] is True

    assert learning_gate(0.4, 0.361, stage1, trained)["passed"] is False
    assert (
        learning_gate(
            0.4, 0.35, stage1, onehop_summary(transfer_f1=0.469, unhealthy=3)
        )["passed"]
        is False
    )


def test_c2_gate_accepts_direct_mean_screen_without_extra_domain_screen():
    stage1 = onehop_summary(transfer_f1=0.90, unhealthy=0)
    trained = onehop_summary(transfer_f1=0.78, native_f1=0.80, unhealthy=2)
    result = learning_gate(0.4, 0.35, stage1, trained)
    assert result["step1000_vs_stage1"]["passed"] is False
    assert result["direct_step1000"]["passed"] is True
    assert result["passed"] is True
