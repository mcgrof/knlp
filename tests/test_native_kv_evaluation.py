# SPDX-License-Identifier: MIT
"""Task extraction and decision rules, including sparse-discordance limits."""

import pytest
from tools.kv.native.gsm8k import extract, number, render
from tools.kv.native.statistics import paired
from tools.kv.native.storage import reconcile


def test_strict_answer_is_distinct_from_flexible_last_number():
    text = "Compute 7 times 6. The answer is 42."
    assert extract(text) == number("42")
    assert extract("I calculated 42") is None
    assert extract("I calculated 42", flexible=True) == number("42")
    assert extract("The answer is -1,234.") == number("-1234")
    assert number("NaN") is None


def test_chat_prompt_disables_thinking_without_changing_the_question():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False)
            return list(text.encode())

    stock = render(Tokenizer(), "How many?", False)
    adapted = render(Tokenizer(), "How many?", True)
    assert stock["prompt"] == "Q: How many?\nA: Let's think step by step."
    assert adapted["prompt"].startswith(stock["prompt"])
    assert "The answer is <number>." in adapted["prompt"]
    assert stock["prompt_hash"] != adapted["prompt_hash"]


def test_zero_discordance_does_not_imply_zero_population_risk():
    base = {str(i): True for i in range(1319)}
    result = paired(base, base)
    assert result["paired_question_bootstrap_95_two_sided"] == [0, 0]
    upper = result["loss_probability_95_one_sided_bounds"][1]
    assert upper == pytest.approx(1 - 0.05 ** (1 / 1319), abs=1e-13)
    assert 0 < upper < 0.02
    small = {str(i): True for i in range(100)}
    assert not paired(small, small)["supported_margin"]


def test_gains_do_not_cancel_losses_in_the_preservation_gate():
    base = {str(i): i < 150 for i in range(200)}
    arm = {str(i): 10 <= i < 160 for i in range(200)}
    result = paired(arm, base)
    assert result["correct_to_wrong"] == result["wrong_to_correct"] == 10
    assert result["net_accuracy_change"] == 0
    assert not result["supported_margin"]
    with pytest.raises(ValueError):
        paired({"x": True}, {"y": True})


def test_shared_storage_is_counted_once_and_overlapping_planes_fail():
    base = dict(device="cuda:0", storage_pointer=100, storage_bytes=48, contiguous=True)
    key = dict(base, tensor_pointer=100, logical_bytes=32)
    value = dict(base, tensor_pointer=132, logical_bytes=16)
    assert reconcile([key, value]) == dict(
        unique_allocations=1,
        backing_bytes=48,
        occupied_payload_bytes=48,
        unoccupied_backing_bytes=0,
    )
    with pytest.raises(ValueError):
        reconcile([key, dict(value, tensor_pointer=120)])


def test_gsm_dispatch_attempts_preserve_pilot_and_partial_receipts(tmp_path):
    from tools.kv.native.gsm8k import reserve_dispatch

    pilot = reserve_dispatch(tmp_path, "pilot")
    receipt = pilot / "installed.json"
    receipt.write_text('{"original":true}\n')
    full = reserve_dispatch(tmp_path, "full")
    (full / "partial.json").write_text("partial\n")
    resumed = reserve_dispatch(tmp_path, "full")
    assert len({pilot, full, resumed}) == 3
    assert receipt.read_text() == '{"original":true}\n'
    assert (full / "partial.json").read_text() == "partial\n"
    assert resumed.is_dir()
    with pytest.raises(ValueError):
        reserve_dispatch(tmp_path, "../outside")


def test_read_coverage_rejects_missing_decode_or_different_backing():
    from tools.kv.native.storage import verify_attention_reads

    tensor = dict(
        device="cuda:0", storage_pointer=100, storage_bytes=48, dtype="torch.float16"
    )
    layers = [{"name": "layer.0", "cache": [tensor]}]
    reads = [
        dict(wrapper=wrapper, layer="layer.0", cache=[tensor.copy()])
        for wrapper in (
            "BatchPrefillWithPagedKVCacheWrapper",
            "BatchDecodeWithPagedKVCacheWrapper",
        )
    ]
    types = {"torch.float16"}
    assert verify_attention_reads(layers, reads, types) == {
        "layers": 1,
        "layer_phase_reads": 2,
    }
    with pytest.raises(ValueError, match="incomplete"):
        verify_attention_reads(layers, reads[:1], types)
    with pytest.raises(ValueError, match="duplicate"):
        verify_attention_reads(layers, reads + reads[:1], types)
    reads[1]["cache"][0]["storage_pointer"] = 200
    with pytest.raises(ValueError, match="storage differs"):
        verify_attention_reads(layers, reads, types)
